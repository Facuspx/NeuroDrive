"""
NeuroDrive_Wearable - Simulador de Pulsera (hace de ESP32)
===========================================================

Sustituye al firmware del ESP32-S3 para probar TODO el lado Pi sin la pulsera.
Escucha ordenes UDP, "vibra" (registra), responde las verificaciones segun
una politica configurable y transmite telemetria periodica.

Reproduce las conductas del firmware que la Pi puede observar:

  - Descarte de repeticiones: recuerda los ultimos 16 id_paquete (no todos),
    igual que el anillo de red.c.
  - Ventana de respuesta: la toma del campo duracion_ms de la orden tipo 8; si
    viene en 0 usa el valor por defecto del protocolo.
  - Respuesta repetida: cada respuesta sale 3 veces con el mismo id_secuencia.
  - Tipo 10: cancela la verificacion en curso (no se responde despues).
  - Tipo 99: marca que la proxima desconexion es esperada. Cualquier orden
    posterior anula esa marca (el nucleo volvio a operar).

Uso como script (contra un Core/receptor real):
    python -m NeuroDrive_Wearable.simulador_pulsera --pi 127.0.0.1 --ack correcto

Politica de respuesta (que pad "toca" el conductor simulado):
    "correcto"   -> toca el pad K (coincide con los K pulsos)
    "incorrecto" -> toca un pad distinto de K
    "ninguno"    -> no responde (la FSM asciende por vencimiento del plazo)
    callable(K)  -> devuelve el pad tocado (1-4) o None para no responder

Tiempos:
    Por defecto la respuesta sale de inmediato (los tests no esperan). Con
    tiempo_real=True (--tiempo-real) el simulador demora lo que demoraria la
    pulsera: K pulsos de 200+300 ms y despues el tiempo de reaccion. Si ese
    tiempo supera la ventana no responde: la ventana cerro.
"""

from __future__ import annotations

import logging
import random
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from common.contratos import TipoComandoActuador
from NeuroDrive_Wearable import protocolo


@dataclass
class _ComandoRecibido:
    timestamp: float
    tipo: int
    intensidad: int
    duracion_ms: int
    id_secuencia: Optional[int]
    id_paquete: int


@dataclass
class _Desafio:
    """Verificacion en curso."""
    id_secuencia: int
    k: int
    ventana_ms: int
    ts_fin_pulsos: float       # instante en que termina la presentacion
    generacion: int            # para cancelarla con el tipo 10


class SimuladorPulsera:
    def __init__(
        self,
        ip_pi: str = "127.0.0.1",
        puerto_pi: int = 5005,           # a donde manda telemetria/respuestas
        puerto_local: int = 5006,        # donde escucha ordenes
        intervalo_bpm_s: float = 2.0,
        bpm_fn: Optional[Callable[[], Optional[int]]] = None,
        bateria_fn: Optional[Callable[[], Optional[int]]] = None,
        politica_ack: Union[str, Callable[[int], Optional[int]]] = "correcto",
        tiempo_respuesta_fn: Optional[Callable[[], int]] = None,
        k_fn: Optional[Callable[[], int]] = None,
        enviar_telemetria: bool = True,
        modo_interactivo: bool = False,
        tiempo_real: bool = False,
        reenvios_ack: int = protocolo.ACK_REENVIOS,
        espaciado_ack_ms: int = protocolo.ACK_ESPACIADO_MS,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._destino = (ip_pi, puerto_pi)
        self._puerto_local = puerto_local
        self._intervalo_bpm = intervalo_bpm_s
        self._bpm_fn = bpm_fn or (lambda: 75 + random.randint(-3, 3))
        self._bateria_fn = bateria_fn or (lambda: 90)
        self._politica_ack = politica_ack
        # Tiempo de reaccion del conductor simulado, en ms desde el ultimo pulso
        self._tiempo_respuesta_fn = tiempo_respuesta_fn or (
            lambda: random.randint(800, 3000)
        )
        # Cantidad de pulsos de cada verificacion. La pulsera la sortea; los
        # tests la fijan para ser repetibles.
        self._k_fn = k_fn or (lambda: random.randint(1, protocolo.DESAFIO_PULSOS_MAX))
        self._enviar_telemetria = enviar_telemetria
        self._modo_interactivo = modo_interactivo
        self._tiempo_real = tiempo_real
        self._reenvios_ack = max(1, int(reenvios_ack))
        self._espaciado_ack_s = max(0, espaciado_ack_ms) / 1000.0
        self.log = logger or logging.getLogger("NeuroDrive.SimuladorPulsera")

        self._sock: Optional[socket.socket] = None
        self._sock_envio: Optional[socket.socket] = None
        self._hilo_rx: Optional[threading.Thread] = None
        self._hilo_tx: Optional[threading.Thread] = None
        self._activo = False
        self._id_paquete_tx = 0
        self._lock = threading.Lock()

        # Dedup de ordenes: anillo de tamano fijo, como en el firmware
        self._ids_recientes: deque = deque(maxlen=protocolo.DEDUP_IDS_RECORDADOS)

        # Verificacion en curso (solo se guarda en modo interactivo)
        self._pendiente: Optional[_Desafio] = None
        self._generacion_desafio = 0

        # ---- Inspeccion para tests ----
        self.comandos_recibidos: List[_ComandoRecibido] = []   # ya deduplicados
        self.duplicados_descartados: int = 0
        self.vibraciones: List[int] = []       # K de cada verificacion
        self.ventanas_ms: List[int] = []       # ventana aplicada a cada una
        self.apagados: int = 0
        self.acks_enviados: int = 0            # datagramas de respuesta (con copias)
        self.respuestas_fuera_de_ventana: int = 0
        self.apagado_limpio_recibido: bool = False

    # ------------------------------------------------------------------
    def iniciar(self) -> None:
        if self._activo:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self._puerto_local))
        self._sock.settimeout(0.5)
        self._sock_envio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._activo = True
        self._hilo_rx = threading.Thread(target=self._bucle_rx, name="SimRX", daemon=True)
        self._hilo_rx.start()
        if self._enviar_telemetria:
            self._hilo_tx = threading.Thread(target=self._bucle_tx, name="SimTX", daemon=True)
            self._hilo_tx.start()
        self.log.info("SimuladorPulsera escuchando ordenes en :%d", self._puerto_local)

    def detener(self, timeout: float = 2.0) -> None:
        if not self._activo:
            return
        self._activo = False
        with self._lock:
            self._generacion_desafio += 1      # cancela respuestas diferidas
            self._pendiente = None
        for h in (self._hilo_rx, self._hilo_tx):
            if h is not None:
                h.join(timeout=timeout)
        self._hilo_rx = self._hilo_tx = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._sock_envio is not None:
            self._sock_envio.close()
            self._sock_envio = None

    # ------------------------------------------------------------------
    # Recepcion de ordenes
    # ------------------------------------------------------------------
    def _bucle_rx(self) -> None:
        while self._activo:
            try:
                datos, _addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._manejar_comando(protocolo.parsear_comando(datos))
            except protocolo.ErrorProtocolo as e:
                self.log.warning("Orden invalida: %s", e)
            except Exception as e:           # el hilo no debe terminar por una orden
                self.log.error("Error manejando una orden: %s", e, exc_info=True)

    def _manejar_comando(self, cmd: dict) -> None:
        id_paq = cmd["id_paquete"]
        tipo = cmd["tipo"]
        try:
            intensidad = int(cmd.get("intensidad") or 0)
            duracion_ms = int(cmd.get("duracion_ms") or 0)
        except (TypeError, ValueError) as e:
            raise protocolo.ErrorProtocolo(f"campos numericos invalidos: {e}") from e
        id_secuencia = cmd.get("id_secuencia")

        # Dedup: la pulsera ejecuta una sola vez las copias con el mismo id
        with self._lock:
            if id_paq in self._ids_recientes:
                self.duplicados_descartados += 1
                return
            self._ids_recientes.append(id_paq)
            self.comandos_recibidos.append(_ComandoRecibido(
                timestamp=time.time(), tipo=tipo, intensidad=intensidad,
                duracion_ms=duracion_ms, id_secuencia=id_secuencia,
                id_paquete=id_paq,
            ))
            # Tipo 99: la desconexion que viene es esperada. Cualquier otra
            # orden indica que el nucleo esta operando y anula la marca.
            self.apagado_limpio_recibido = (tipo == protocolo.TIPO_APAGADO_LIMPIO)

        if tipo == protocolo.TIPO_APAGADO_LIMPIO:
            self.log.info("[pulsera] APAGADO_LIMPIO: la proxima perdida es esperada")
        elif tipo == int(TipoComandoActuador.APAGAR_TODO):
            with self._lock:
                self.apagados += 1
                self._generacion_desafio += 1      # cancela la verificacion en curso
                self._pendiente = None
            self.log.info("[pulsera] STOP motor (verificacion cancelada)")
        elif tipo == int(TipoComandoActuador.SECUENCIA_ACK):
            self._ejecutar_desafio(id_secuencia, duracion_ms)
        elif tipo in (
            int(TipoComandoActuador.VIBRAR_LEVE),
            int(TipoComandoActuador.VIBRAR_MEDIO),
            int(TipoComandoActuador.VIBRAR_FUERTE),
        ):
            self.log.info("[pulsera] vibra (tipo=%d, int=%d, dur=%d ms)",
                          tipo, intensidad, duracion_ms)

    # ------------------------------------------------------------------
    # Verificacion de atencion
    # ------------------------------------------------------------------
    def _ejecutar_desafio(self, id_secuencia: Optional[int], duracion_ms: int) -> None:
        if id_secuencia is None:
            return
        # La pulsera sortea K (1-4) y emite K pulsos
        k = int(self._k_fn())
        ventana_ms = duracion_ms if duracion_ms > 0 else protocolo.VENTANA_DESAFIO_DEFAULT_MS
        presentacion_s = (
            k * (protocolo.DESAFIO_PULSO_ON_MS + protocolo.DESAFIO_PULSO_OFF_MS) / 1000.0
            if self._tiempo_real else 0.0
        )
        with self._lock:
            self.vibraciones.append(k)
            self.ventanas_ms.append(ventana_ms)
            self._generacion_desafio += 1
            desafio = _Desafio(
                id_secuencia=int(id_secuencia), k=k, ventana_ms=ventana_ms,
                ts_fin_pulsos=time.time() + presentacion_s,
                generacion=self._generacion_desafio,
            )
            self._pendiente = desafio if self._modo_interactivo else None
        self.log.info("[pulsera] DESAFIO id=%s: %d pulso(s), ventana %d ms",
                      id_secuencia, k, ventana_ms)

        if self._modo_interactivo:
            # No responde solo: espera que el usuario escriba el pad
            print(f"\n>>> DESAFIO: la pulsera vibro {k} vez/veces. "
                  f"Tenes {ventana_ms / 1000:.0f} s. "
                  f"Toca el pad (1-4) o 'n' para NO responder: ", flush=True)
            return

        # Modo automatico: decidir que pad toca el "conductor" y cuando
        pad = self._decidir_pad(k)
        if pad is None:
            self.log.info("[pulsera] conductor no responde")
            return
        tiempo_ms = max(1, int(self._tiempo_respuesta_fn()))
        if tiempo_ms > ventana_ms:
            # El toque llegaria con la ventana cerrada: la pulsera no envia
            with self._lock:
                self.respuestas_fuera_de_ventana += 1
            self.log.info("[pulsera] toque a los %d ms con ventana de %d ms: "
                          "cerro antes, no se envia", tiempo_ms, ventana_ms)
            return

        if self._tiempo_real:
            temporizador = threading.Timer(
                presentacion_s + tiempo_ms / 1000.0,
                self._responder_si_vigente, args=(desafio, pad, tiempo_ms),
            )
            temporizador.daemon = True
            temporizador.start()
        else:
            self._responder_si_vigente(desafio, pad, tiempo_ms)

    def _responder_si_vigente(self, d: _Desafio, pad: int, tiempo_ms: int) -> None:
        """Envia la respuesta salvo que un tipo 10 o un desafio nuevo la haya cancelado."""
        with self._lock:
            if not self._activo or d.generacion != self._generacion_desafio:
                return
        correcta = (pad == d.k)
        self._enviar_ack(protocolo.serializar_ack(
            id_secuencia=d.id_secuencia,
            secuencia_correcta=correcta,
            tiempo_respuesta_ms=tiempo_ms,
        ))
        self.log.info("[pulsera] toca pad %d (K=%d) -> %s, %d ms",
                      pad, d.k, "correcto" if correcta else "incorrecto", tiempo_ms)

    def responder(self, entrada: str) -> None:
        """Modo interactivo: interpreta lo que el usuario escribio como el pad
        tocado. '1'-'4' toca ese pad; 'n'/'' = no responde."""
        with self._lock:
            pend = self._pendiente
            self._pendiente = None
        if pend is None:
            return
        entrada = entrada.strip().lower()
        if entrada in ("n", ""):
            print(f"    (sin respuesta al desafio {pend.id_secuencia}; "
                  f"la FSM asciende al vencer el plazo)")
            return
        if entrada not in ("1", "2", "3", "4"):
            print(f"    entrada invalida: {entrada!r} (usa 1-4 o n)")
            with self._lock:
                self._pendiente = pend      # se devuelve para reintentar
            return
        tiempo_ms = max(1, int((time.time() - pend.ts_fin_pulsos) * 1000))
        if tiempo_ms > pend.ventana_ms:
            with self._lock:
                self.respuestas_fuera_de_ventana += 1
            print(f"    tardaste {tiempo_ms} ms y la ventana era de "
                  f"{pend.ventana_ms} ms: la pulsera ya no envia la respuesta")
            return
        pad = int(entrada)
        self._responder_si_vigente(pend, pad, tiempo_ms)
        print(f"    tocaste pad {pad} (K={pend.k}) -> "
              f"{'CORRECTO' if pad == pend.k else 'INCORRECTO'}")

    def _decidir_pad(self, k: int) -> Optional[int]:
        pol = self._politica_ack
        if callable(pol):
            return pol(k)
        if pol == "correcto":
            return k
        if pol == "incorrecto":
            return (k % 4) + 1  # cualquier pad distinto de k
        if pol == "ninguno":
            return None
        return k

    # ------------------------------------------------------------------
    # Envio
    # ------------------------------------------------------------------
    def _enviar_ack(self, datos: bytes) -> None:
        """Respuesta repetida, sin frenar al hilo que la origino."""
        def _rafaga() -> None:
            for i in range(self._reenvios_ack):
                if not self._activo:
                    return
                if self._enviar(datos):
                    with self._lock:
                        self.acks_enviados += 1
                if i < self._reenvios_ack - 1:
                    time.sleep(self._espaciado_ack_s)
        threading.Thread(target=_rafaga, name="SimACK", daemon=True).start()

    def _bucle_tx(self) -> None:
        while self._activo:
            self._id_paquete_tx += 1
            self._enviar(protocolo.serializar_telemetria(
                bpm=self._bpm_fn(), bateria=self._bateria_fn(),
                id_paquete=self._id_paquete_tx,
            ))
            # dormir en pasos chicos para poder cortar rapido
            t = 0.0
            while self._activo and t < self._intervalo_bpm:
                time.sleep(0.05)
                t += 0.05

    def _enviar(self, datos: bytes) -> bool:
        sock = self._sock_envio
        if sock is None:
            return False
        try:
            sock.sendto(datos, self._destino)
            return True
        except OSError as e:
            self.log.error("Fallo enviando a la Pi: %s", e)
            return False


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Simulador de la pulsera NeuroDrive")
    ap.add_argument("--pi", default="127.0.0.1", help="IP de la Raspberry")
    ap.add_argument("--puerto-pi", type=int, default=5005)
    ap.add_argument("--puerto-local", type=int, default=5006)
    ap.add_argument("--ack", default="correcto",
                    choices=["correcto", "incorrecto", "ninguno"])
    ap.add_argument("--interactivo", action="store_true",
                    help="Responder las verificaciones a mano por teclado")
    ap.add_argument("--tiempo-real", action="store_true",
                    help="Demorar la respuesta como la pulsera real (pulsos + reaccion)")
    args = ap.parse_args()
    sim = SimuladorPulsera(
        ip_pi=args.pi, puerto_pi=args.puerto_pi,
        puerto_local=args.puerto_local, politica_ack=args.ack,
        modo_interactivo=args.interactivo,
        tiempo_real=args.tiempo_real or args.interactivo,
    )
    sim.iniciar()
    print(f"Simulador de pulsera activo. Manda a {args.pi}:{args.puerto_pi}, "
          f"escucha en :{args.puerto_local}.")
    try:
        if args.interactivo:
            print("Modo INTERACTIVO: cuando aparezca un desafio, escribi el pad "
                  "(1-4) o 'n' para no responder. Ctrl+C para salir.")
            while True:
                sim.responder(input())
        else:
            print(f"Respuesta automatica: {args.ack}. Ctrl+C para salir.")
            while True:
                time.sleep(1)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        sim.detener()
