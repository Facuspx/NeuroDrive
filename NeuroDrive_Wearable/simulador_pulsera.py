"""
NeuroDrive_Wearable - Simulador de Pulsera (hace de ESP32)
===========================================================

Simula el firmware del ESP32-S3 para testear TODO el lado Pi sin flashear
nada. Escucha comandos UDP, "vibra" (imprime), responde desafios ACK segun
una politica configurable, y manda telemetria BPM periodica.

Uso como script (contra un Core/receptor real):
    python -m NeuroDrive_Wearable.simulador_pulsera --pi 192.168.4.1

Uso en tests: se instancia, se inspeccionan comandos_recibidos, y se
configura la politica de ACK y la fuente de BPM.

Politica de ACK (que pad "toca" el conductor simulado):
    "correcto"  -> toca el pad K (coincide con las K vibraciones) -> correcto
    "incorrecto"-> toca un pad != K -> incorrecto
    "ninguno"   -> no responde (simula conductor dormido -> la FSM hace timeout)
    callable(K) -> devuelve el numero de pad tocado (1-4)
"""

from __future__ import annotations

import logging
import random
import socket
import threading
import time
from dataclasses import dataclass, field
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


class SimuladorPulsera:
    def __init__(
        self,
        ip_pi: str = "127.0.0.1",
        puerto_pi: int = 5005,           # a donde manda telemetria/ack (escucha de la Pi)
        puerto_local: int = 5006,        # donde escucha comandos (envio de la Pi)
        intervalo_bpm_s: float = 2.0,
        bpm_fn: Optional[Callable[[], Optional[int]]] = None,
        bateria_fn: Optional[Callable[[], Optional[int]]] = None,
        politica_ack: Union[str, Callable[[int], int]] = "correcto",
        enviar_telemetria: bool = True,
        modo_interactivo: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._destino = (ip_pi, puerto_pi)
        self._puerto_local = puerto_local
        self._intervalo_bpm = intervalo_bpm_s
        self._bpm_fn = bpm_fn or (lambda: 75 + random.randint(-3, 3))
        self._bateria_fn = bateria_fn or (lambda: 90)
        self._politica_ack = politica_ack
        self._enviar_telemetria = enviar_telemetria
        self._modo_interactivo = modo_interactivo
        self._pendiente = None   # (id_secuencia, K, ts_inicio) en modo interactivo
        self.log = logger or logging.getLogger("NeuroDrive.SimuladorPulsera")

        self._sock: Optional[socket.socket] = None
        self._sock_envio: Optional[socket.socket] = None
        self._hilo_rx: Optional[threading.Thread] = None
        self._hilo_tx: Optional[threading.Thread] = None
        self._activo = False
        self._id_paquete_tx = 0
        self._vistos_id_paquete: set = set()   # dedup de comandos (como el ESP real)

        # Inspeccion para tests
        self.comandos_recibidos: List[_ComandoRecibido] = []
        self.vibraciones: List[int] = []       # cantidad de pulsos por desafio
        self.apagados: int = 0
        self._lock = threading.Lock()

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
        self.log.info("SimuladorPulsera escuchando comandos en :%d", self._puerto_local)

    def detener(self, timeout: float = 2.0) -> None:
        if not self._activo:
            return
        self._activo = False
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
    def _bucle_rx(self) -> None:
        while self._activo:
            try:
                datos, _addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                cmd = protocolo.parsear_comando(datos)
            except protocolo.ErrorProtocolo as e:
                self.log.warning("Comando invalido: %s", e)
                continue
            self._manejar_comando(cmd)

    def _manejar_comando(self, cmd: dict) -> None:
        id_paq = cmd.get("id_paquete", 0)
        # Dedup: el ESP ignora reenvios con el mismo id_paquete
        with self._lock:
            if id_paq in self._vistos_id_paquete:
                return
            self._vistos_id_paquete.add(id_paq)
            self.comandos_recibidos.append(_ComandoRecibido(
                timestamp=time.time(),
                tipo=int(cmd["tipo"]),
                intensidad=int(cmd.get("intensidad", 0)),
                duracion_ms=int(cmd.get("duracion_ms", 0)),
                id_secuencia=cmd.get("id_secuencia"),
                id_paquete=id_paq,
            ))

        tipo = int(cmd["tipo"])
        if tipo == int(TipoComandoActuador.APAGAR_TODO):
            with self._lock:
                self.apagados += 1
            self.log.info("[pulsera] STOP motor")
        elif tipo == int(TipoComandoActuador.SECUENCIA_ACK):
            self._ejecutar_desafio(cmd.get("id_secuencia"))
        elif tipo in (
            int(TipoComandoActuador.VIBRAR_LEVE),
            int(TipoComandoActuador.VIBRAR_MEDIO),
            int(TipoComandoActuador.VIBRAR_FUERTE),
        ):
            self.log.info("[pulsera] vibra (tipo=%d, int=%d)", tipo, cmd.get("intensidad", 0))

    def _ejecutar_desafio(self, id_secuencia: Optional[int]) -> None:
        if id_secuencia is None:
            return
        # El ESP genera K localmente (1-4) y vibra K veces
        k = random.randint(1, 4)
        with self._lock:
            self.vibraciones.append(k)
        self.log.info("[pulsera] DESAFIO id=%s: vibra %d veces", id_secuencia, k)

        if self._modo_interactivo:
            # No responde solo: espera que el usuario escriba el pad por teclado
            with self._lock:
                self._pendiente = (int(id_secuencia), k, time.time())
            print(f"\n>>> DESAFIO: la pulsera vibro {k} vez/veces. "
                  f"Toca el pad correcto (1-4) o 'n' para NO responder: ", flush=True)
            return

        # Decidir que pad toca el "conductor"
        pad = self._decidir_pad(k)
        if pad is None:
            self.log.info("[pulsera] conductor no responde (timeout)")
            return  # no manda ACK: la FSM hara timeout

        correcta = (pad == k)
        tiempo_ms = random.randint(800, 3000)
        datos = protocolo.serializar_ack(
            id_secuencia=int(id_secuencia),
            secuencia_correcta=correcta,
            tiempo_respuesta_ms=tiempo_ms,
        )
        self._enviar(datos)
        self.log.info("[pulsera] toca pad %d (K=%d) -> %s",
                      pad, k, "correcto" if correcta else "incorrecto")

    def responder(self, entrada: str) -> None:
        """Modo interactivo: interpreta lo que el usuario escribio como el pad
        tocado. '1'-'4' toca ese pad; 'n'/'' = no responde (timeout de la FSM)."""
        with self._lock:
            pend = self._pendiente
            self._pendiente = None
        if pend is None:
            return
        id_seq, k, ts_inicio = pend
        entrada = entrada.strip().lower()
        if entrada in ("n", ""):
            print(f"    (no respondiste al desafio {id_seq}; la FSM va a escalar por timeout)")
            return
        if entrada not in ("1", "2", "3", "4"):
            print(f"    entrada invalida: {entrada!r} (usa 1-4 o n)")
            with self._lock:
                self._pendiente = pend  # devolver el pendiente para reintentar
            return
        pad = int(entrada)
        correcta = (pad == k)
        tiempo_ms = int((time.time() - ts_inicio) * 1000)
        datos = protocolo.serializar_ack(
            id_secuencia=id_seq, secuencia_correcta=correcta,
            tiempo_respuesta_ms=max(1, tiempo_ms),
        )
        self._enviar(datos)
        print(f"    tocaste pad {pad} (K={k}) -> "
              f"{'CORRECTO' if correcta else 'INCORRECTO'}")

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
    def _bucle_tx(self) -> None:
        while self._activo:
            bpm = self._bpm_fn()
            bat = self._bateria_fn()
            self._id_paquete_tx += 1
            datos = protocolo.serializar_telemetria(
                bpm=bpm, bateria=bat, id_paquete=self._id_paquete_tx,
            )
            self._enviar(datos)
            # dormir en pasos chicos para poder cortar rapido
            t = 0.0
            while self._activo and t < self._intervalo_bpm:
                time.sleep(0.05)
                t += 0.05

    def _enviar(self, datos: bytes) -> None:
        if self._sock_envio is not None:
            try:
                self._sock_envio.sendto(datos, self._destino)
            except OSError as e:
                self.log.error("Fallo enviando a la Pi: %s", e)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default="127.0.0.1", help="IP de la Raspberry")
    ap.add_argument("--puerto-pi", type=int, default=5005)
    ap.add_argument("--puerto-local", type=int, default=5006)
    ap.add_argument("--ack", default="correcto",
                    choices=["correcto", "incorrecto", "ninguno"])
    ap.add_argument("--interactivo", action="store_true",
                    help="Responder los desafios ACK a mano por teclado")
    args = ap.parse_args()
    sim = SimuladorPulsera(
        ip_pi=args.pi, puerto_pi=args.puerto_pi,
        puerto_local=args.puerto_local, politica_ack=args.ack,
        modo_interactivo=args.interactivo,
    )
    sim.iniciar()
    print(f"Simulador de pulsera activo. Manda a {args.pi}:{args.puerto_pi}, "
          f"escucha en :{args.puerto_local}.")
    if args.interactivo:
        print("Modo INTERACTIVO: cuando aparezca un desafio, escribi el pad "
              "(1-4) o 'n' para no responder. Ctrl+C para salir.")
        try:
            while True:
                linea = input()
                sim.responder(linea)
        except (KeyboardInterrupt, EOFError):
            sim.detener()
    else:
        print(f"ACK automatico: {args.ack}. Ctrl+C para salir.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sim.detener()
