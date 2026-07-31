"""
NeuroDrive_Wearable - Actuador Wearable (lado Pi, envio de comandos)
=====================================================================

Implementa ActuadorBase. El Despachador le entrega comandos de vibracion y
de desafio ACK; este actuador los envia por UDP al ESP32.

Diseno (coherente con el resto de actuadores):

  - NO bloqueante: ejecutar() encola un "trabajo de envio" y retorna. Un hilo
    emisor interno hace el sendto y los reenvios temporizados. Asi el hilo
    trabajador del Despachador nunca se frena esperando la red.

  - REENVIOS para comandos criticos: UDP puede perder paquetes. Para los
    tipos criticos (SECUENCIA_ACK y VIBRAR_FUERTE) el mensaje se envia
    `reenvios` veces con el MISMO id_paquete, espaciados unos ms. El ESP32
    deduplica por id_paquete (ignora repetidos). Perder un SECUENCIA_ACK no
    es catastrofico (la FSM escala por timeout, que es un fallo seguro), pero
    los reenvios evitan escaladas por un simple paquete perdido.

  - APAGAR: apagar() cancela reenvios pendientes y manda un comando explicito
    de apagado al ESP para frenar el motor. Lo llama el Despachador en el
    broadcast de APAGAR_TODO y en el shutdown.

  - COSTURA DE RED: por defecto abre un socket UDP real. En tests se inyecta
    un transporte falso (con .sendto/.close) para inspeccionar lo enviado sin
    tocar la red.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Set

from common.contratos import ComandoActuador, TipoComandoActuador
from NeuroDrive_Core.despachador import ActuadorBase
from NeuroDrive_Wearable import protocolo


# Tipos criticos: se reenvian para robustez ante perdida de UDP.
_TIPOS_CRITICOS_DEFAULT = frozenset({
    TipoComandoActuador.SECUENCIA_ACK,
    TipoComandoActuador.VIBRAR_FUERTE,
})

_SENTINELA_FIN = object()


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
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if reenvios_criticos < 1:
            raise ValueError("reenvios_criticos debe ser >= 1")
        self._ip = ip_wearable
        self._puerto = puerto_envio
        self._reenvios = reenvios_criticos
        self._espaciado_s = espaciado_reenvios_ms / 1000.0
        self._tipos_criticos = (
            tipos_criticos if tipos_criticos is not None else set(_TIPOS_CRITICOS_DEFAULT)
        )
        self._transporte_inyectado = transporte
        self.log = logger or logging.getLogger("NeuroDrive.ActuadorWearable")

        self._transporte = None
        self._cola: "queue.Queue" = queue.Queue(maxsize=capacidad_cola)
        self._hilo: Optional[threading.Thread] = None
        self._abierto = False
        self._lock = threading.Lock()
        self._generacion = 0
        self._id_paquete = 0

        # Estadisticas
        self.comandos_enviados = 0
        self.paquetes_enviados = 0        # cuenta reenvios
        self.envios_descartados_cola = 0
        self.errores_envio = 0

    # ------------------------------------------------------------------
    # Interfaz ActuadorBase
    # ------------------------------------------------------------------

    def tipos_soportados(self) -> Set[TipoComandoActuador]:
        return {
            TipoComandoActuador.VIBRAR_LEVE,
            TipoComandoActuador.VIBRAR_MEDIO,
            TipoComandoActuador.VIBRAR_FUERTE,
            TipoComandoActuador.SECUENCIA_ACK,
        }

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

    def ejecutar(self, comando: ComandoActuador) -> None:
        if not self._abierto:
            raise RuntimeError("ActuadorWearable.ejecutar() sin iniciar()")
        with self._lock:
            self._id_paquete += 1
            id_paq = self._id_paquete
            gen = self._generacion
        datos = protocolo.serializar_comando(comando, id_paq)
        es_critico = comando.tipo in self._tipos_criticos
        self._encolar(_TrabajoEnvio(datos, es_critico, gen))
        self.comandos_enviados += 1

    def apagar(self) -> None:
        # Cancelar reenvios pendientes e informar al ESP para frenar el motor.
        with self._lock:
            self._generacion += 1
            self._id_paquete += 1
            id_paq = self._id_paquete
        self._vaciar_cola()
        if self._abierto and self._transporte is not None:
            try:
                self._transporte.sendto(protocolo.serializar_apagar(id_paq))
                self.paquetes_enviados += 1
            except OSError as e:
                self.errores_envio += 1
                self.log.error("Fallo enviando APAGAR al wearable: %s", e)

    def detener(self, timeout: float = 2.0) -> None:
        if not self._abierto:
            return
        self.apagar()
        self._abierto = False
        try:
            self._cola.put_nowait(_SENTINELA_FIN)
        except queue.Full:
            self._vaciar_cola()
            try:
                self._cola.put_nowait(_SENTINELA_FIN)
            except queue.Full:
                pass
        if self._hilo is not None:
            self._hilo.join(timeout=timeout)
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
            # Si un apagar/comando nuevo cambio la generacion, abortar reenvios.
            with self._lock:
                if trabajo.generacion != self._generacion:
                    return
            if self._transporte is None:
                return
            try:
                self._transporte.sendto(trabajo.datos)
                self.paquetes_enviados += 1
            except OSError as e:
                self.errores_envio += 1
                self.log.error("Fallo enviando al wearable: %s", e)
                return
            if i < veces - 1:
                time.sleep(self._espaciado_s)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _encolar(self, trabajo: _TrabajoEnvio) -> None:
        try:
            self._cola.put_nowait(trabajo)
        except queue.Full:
            self.envios_descartados_cola += 1
            self.log.warning("Cola del emisor wearable llena; se descarta un envio")

    def _vaciar_cola(self) -> None:
        while True:
            try:
                item = self._cola.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINELA_FIN:
                try:
                    self._cola.put_nowait(_SENTINELA_FIN)
                except queue.Full:
                    pass
                self._cola.task_done()
                break
            self._cola.task_done()

    def esperar_vaciado(self, timeout: float = 2.0) -> None:
        inicio = time.monotonic()
        while self._cola.unfinished_tasks > 0:
            if (time.monotonic() - inicio) > timeout:
                return
            time.sleep(0.005)
