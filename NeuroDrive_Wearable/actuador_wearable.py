"""
NeuroDrive_Wearable - Actuador Wearable (lado Pi, envio de ordenes)
====================================================================

Implementa ActuadorBase. El Despachador le entrega ordenes de vibracion y de
verificacion de atencion; este actuador las envia por UDP al ESP32.

Diseno:

  - NO bloqueante. ejecutar(), apagar() y el aviso de apagado solo ENCOLAN un
    trabajo. Un hilo emisor interno hace el sendto y las repeticiones. Asi el
    hilo trabajador del Despachador nunca se frena esperando la red.

  - REPETICION de las ordenes criticas. UDP puede perder datagramas, de modo
    que las ordenes cuya perdida tiene consecuencias viajan `reenvios` veces
    con el MISMO id_paquete, separadas unos milisegundos. La pulsera descarta
    las repetidas. Son criticas:
        SECUENCIA_ACK   (8)   si se pierde, el conductor nunca ve el desafio
        VIBRAR_FUERTE   (3)
        APAGAR_TODO     (10)  si se pierde, el motor sigue vibrando
        APAGADO_LIMPIO  (99)  si se pierde, la pulsera emite un aviso falso

  - GENERACION. apagar() incrementa un contador; cada trabajo guarda el valor
    vigente al crearse y el emisor lo compara antes de CADA envio. Las
    repeticiones pendientes de una orden anterior a la detencion se abandonan,
    para que no lleguen despues de ella y dejen el motor vibrando. El contador
    NO se incrementa con cada orden nueva: la FSM emite VIBRAR y SECUENCIA_ACK
    en la misma tanda, y la segunda cancelaria a la primera.

  - IDENTIFICADOR DE ENVIO con base aleatoria por sesion (ver protocolo.py).

  - SALUDO DE INICIO. Al iniciar se envia un APAGAR_TODO. Deja a la pulsera en
    un estado conocido (motor detenido, sin desafio pendiente de una sesion
    anterior) y le indica que el nucleo volvio a operar, lo que anula un aviso
    de apagado ordenado previo.

  - COSTURA DE RED: por defecto abre un socket UDP real. En tests se inyecta
    un transporte falso (con .sendto/.close) para inspeccionar lo enviado sin
    tocar la red.
"""

from __future__ import annotations

import logging
import queue
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Set, Tuple

from common.contratos import ComandoActuador, TipoComandoActuador
from NeuroDrive_Core.despachador import ActuadorBase
from NeuroDrive_Wearable import protocolo


# Tipos de la FSM que se envian repetidos. APAGAR_TODO y APAGADO_LIMPIO no
# figuran aca porque no llegan por ejecutar(): tienen su propio camino y son
# siempre criticos.
_TIPOS_CRITICOS_DEFAULT = frozenset({
    TipoComandoActuador.SECUENCIA_ACK,
    TipoComandoActuador.VIBRAR_FUERTE,
})

_SENTINELA_FIN = object()

# La base del identificador se sortea por debajo de 2^30 para que, aun tras
# millones de ordenes, el valor entre en el entero de 32 bits con signo con
# que lo lee el firmware (cJSON valueint).
_BASE_ID_PAQUETE_MAX = 1 << 30


@dataclass
class _TrabajoEnvio:
    datos: bytes
    es_critico: bool
    generacion: int


class _TransporteUDP:
    """Transporte real: un socket UDP hacia (ip, puerto)."""

    def __init__(self, ip: str, puerto: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._destino = (ip, puerto)

    def sendto(self, datos: bytes) -> None:
        self._sock.sendto(datos, self._destino)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class ActuadorWearable(ActuadorBase):
    """Ver docstring del modulo."""

    nombre = "wearable"

    def __init__(
        self,
        ip_wearable: str = "192.168.4.20",
        puerto_envio: int = 5006,
        reenvios_criticos: int = 3,
        espaciado_reenvios_ms: int = 50,
        tipos_criticos: Optional[Set[TipoComandoActuador]] = None,
        transporte: Optional[object] = None,
        capacidad_cola: int = 32,
        id_paquete_inicial: Optional[int] = None,
        saludo_inicio: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        id_paquete_inicial : None (produccion) sortea la base por sesion.
                             Los tests pasan un valor fijo para ser repetibles.
        saludo_inicio      : enviar APAGAR_TODO al iniciar (ver docstring).
        """
        if reenvios_criticos < 1:
            raise ValueError("reenvios_criticos debe ser >= 1")
        if espaciado_reenvios_ms < 0:
            raise ValueError("espaciado_reenvios_ms no puede ser negativo")
        if capacidad_cola < 4:
            # detencion + apagado limpio + sentinela deben entrar siempre
            raise ValueError("capacidad_cola debe ser >= 4")
        self._ip = ip_wearable
        self._puerto = puerto_envio
        self._reenvios = reenvios_criticos
        self._espaciado_s = espaciado_reenvios_ms / 1000.0
        self._tipos_criticos = (
            tipos_criticos if tipos_criticos is not None else set(_TIPOS_CRITICOS_DEFAULT)
        )
        self._transporte_inyectado = transporte
        self._saludo_inicio = saludo_inicio
        self.log = logger or logging.getLogger("NeuroDrive.ActuadorWearable")

        self._transporte = None
        self._cola: "queue.Queue" = queue.Queue(maxsize=capacidad_cola)
        self._hilo: Optional[threading.Thread] = None
        self._abierto = False
        self._lock = threading.Lock()
        self._generacion = 0
        self._id_paquete = (
            id_paquete_inicial
            if id_paquete_inicial is not None
            else secrets.randbelow(_BASE_ID_PAQUETE_MAX)
        )

        # Estadisticas
        self.comandos_enviados = 0
        self.paquetes_enviados = 0        # cada datagrama, repeticiones incluidas
        self.envios_descartados_cola = 0
        self.errores_envio = 0

    # ------------------------------------------------------------------
    # Interfaz ActuadorBase
    # ------------------------------------------------------------------

    def tipos_soportados(self) -> Set[TipoComandoActuador]:
        # Unica definicion: la del protocolo.
        return set(protocolo.TIPOS_PARA_WEARABLE)

    def iniciar(self) -> None:
        if self._abierto:
            return
        if self._transporte_inyectado is not None:
            self._transporte = self._transporte_inyectado
        else:
            self._transporte = _TransporteUDP(self._ip, self._puerto)
        self._abierto = True
        self._hilo = threading.Thread(
            target=self._bucle_emisor, name="WearableEmisor", daemon=True
        )
        self._hilo.start()
        self.log.info("ActuadorWearable enviando a %s:%d", self._ip, self._puerto)
        if self._saludo_inicio:
            self._encolar_apagar()

    def ejecutar(self, comando: ComandoActuador) -> None:
        if not self._abierto:
            raise RuntimeError("ActuadorWearable.ejecutar() sin iniciar()")
        id_paq, gen = self._nuevo_id_paquete()
        datos = protocolo.serializar_comando(comando, id_paq)
        es_critico = comando.tipo in self._tipos_criticos
        if self._encolar(_TrabajoEnvio(datos, es_critico, gen)):
            self.comandos_enviados += 1

    def apagar(self) -> None:
        """
        Detiene toda actuacion en la pulsera. Es la orden donde la perdida
        tiene peor consecuencia (en nivel critico el motor quedaria vibrando
        hasta la orden siguiente), asi que viaja repetida.

        Orden de las operaciones:
          1. Incrementar la generacion: el emisor abandona las repeticiones
             pendientes de cualquier orden anterior.
          2. Vaciar la cola: lo que no salio ya no debe salir.
          3. Encolar el tipo 10 con la generacion NUEVA.
        """
        with self._lock:
            self._generacion += 1
        self._vaciar_cola()
        if self._abierto:
            self._encolar_apagar()

    def detener(self, timeout: float = 2.0) -> None:
        """
        Apagado ORDENADO. Secuencia: tipo 10 (x3) -> tipo 99 (x3) -> fin del
        hilo -> cierre del socket. Los tres trabajos pasan por la cola, de modo
        que salen en ese orden y el socket se cierra recien cuando el emisor
        termino de transmitirlos.

        Si el proceso se interrumpe de manera imprevista esta funcion no se
        ejecuta y el tipo 99 no se envia: es justamente lo que le permite a la
        pulsera distinguir los dos casos.
        """
        if not self._abierto:
            return
        self.apagar()
        id_paq, gen = self._nuevo_id_paquete()
        self._encolar(
            _TrabajoEnvio(protocolo.serializar_apagado_limpio(id_paq), True, gen),
            garantizado=True,
        )
        self._abierto = False          # desde aca ejecutar()/apagar() ya no encolan
        self._encolar_sentinela()
        if self._hilo is not None:
            self._hilo.join(timeout=timeout)
            if self._hilo.is_alive():
                self.log.warning(
                    "El hilo emisor no termino en %.1f s; se cierra el socket igual",
                    timeout,
                )
            self._hilo = None
        if self._transporte is not None:
            self._transporte.close()
            self._transporte = None
        self.log.info("ActuadorWearable detenido")

    # ------------------------------------------------------------------
    # Hilo emisor
    # ------------------------------------------------------------------

    def _bucle_emisor(self) -> None:
        while True:
            item = self._cola.get()
            try:
                if item is _SENTINELA_FIN:
                    return
                self._enviar_trabajo(item)
            finally:
                self._cola.task_done()

    def _enviar_trabajo(self, trabajo: _TrabajoEnvio) -> None:
        veces = self._reenvios if trabajo.es_critico else 1
        for i in range(veces):
            # Si una detencion cambio la generacion, abandonar lo pendiente.
            # La comparacion va DENTRO del ciclo: la orden que invalida a esta
            # puede llegar entre el primer envio y el segundo.
            with self._lock:
                if trabajo.generacion != self._generacion:
                    return
            transporte = self._transporte
            if transporte is None:
                return
            try:
                transporte.sendto(trabajo.datos)
                self.paquetes_enviados += 1
            except OSError as e:
                # Un error aca indica que la interfaz no esta disponible: las
                # repeticiones restantes fallarian igual.
                self.errores_envio += 1
                self.log.error("Fallo enviando al wearable: %s", e)
                return
            if i < veces - 1:
                time.sleep(self._espaciado_s)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _nuevo_id_paquete(self) -> Tuple[int, int]:
        """Devuelve (id_paquete nuevo, generacion vigente), tomados juntos."""
        with self._lock:
            self._id_paquete += 1
            return self._id_paquete, self._generacion

    def _encolar_apagar(self) -> None:
        id_paq, gen = self._nuevo_id_paquete()
        self._encolar(
            _TrabajoEnvio(protocolo.serializar_apagar(id_paq), True, gen),
            garantizado=True,
        )

    def _encolar(self, trabajo: _TrabajoEnvio, garantizado: bool = False) -> bool:
        """
        Encola un trabajo. Con la cola llena:
          - trabajo comun: se descarta y se contabiliza (mismo criterio que el
            resto del sistema: no bloquear al productor);
          - trabajo garantizado (detencion / apagado limpio): se vacia la cola
            para hacerle lugar. Una detencion vuelve obsoleto lo anterior.
        """
        try:
            self._cola.put_nowait(trabajo)
            return True
        except queue.Full:
            if not garantizado:
                self.envios_descartados_cola += 1
                self.log.warning("Cola del emisor wearable llena; se descarta un envio")
                return False
        self._vaciar_cola()
        try:
            self._cola.put_nowait(trabajo)
            return True
        except queue.Full:
            self.envios_descartados_cola += 1
            self.log.error("No se pudo encolar una orden garantizada")
            return False

    def _encolar_sentinela(self) -> None:
        """Pide al hilo emisor que termine, DESPUES de lo ya encolado."""
        try:
            self._cola.put(_SENTINELA_FIN, timeout=1.0)
        except queue.Full:
            # El emisor no esta consumiendo. Se le hace lugar a la fuerza.
            self._vaciar_cola()
            try:
                self._cola.put_nowait(_SENTINELA_FIN)
            except queue.Full:
                pass

    def _vaciar_cola(self) -> None:
        while True:
            try:
                item = self._cola.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINELA_FIN:
                # No se pierde el pedido de terminar: se devuelve a la cola.
                try:
                    self._cola.put_nowait(_SENTINELA_FIN)
                except queue.Full:
                    pass
                self._cola.task_done()
                break
            self._cola.task_done()

    def esperar_vaciado(self, timeout: float = 2.0) -> None:
        """Bloquea hasta que el emisor proceso todo lo encolado (para tests)."""
        inicio = time.monotonic()
        while self._cola.unfinished_tasks > 0:
            if (time.monotonic() - inicio) > timeout:
                return
            time.sleep(0.005)
