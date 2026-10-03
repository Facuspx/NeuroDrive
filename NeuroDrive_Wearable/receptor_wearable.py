"""
NeuroDrive_Wearable - Receptor Wearable (lado Pi, recepcion)
=============================================================

Escucha por UDP los mensajes de la pulsera (telemetria y respuestas a la
verificacion), los convierte en Envelope(EventoWearable | EventoAckWearable)
y los publica en la cola POSIX MQ del wearable. El Gestor ya consume esa cola
sin cambios, y su monitor de salud detecta "pulsera caida" si dejan de llegar.

Diseno:
  - Un hilo receptor: recvfrom -> parsear -> construir evento -> publicar.
  - numero_secuencia propio incremental (el Gestor deduplica por
    (id_dispositivo, numero_secuencia)).
  - Publicacion desacoplada por una funcion `publicar_fn(envelope_json)->bool`.
    Por defecto usa AdaptadorMQ (cola POSIX real). En tests se inyecta una
    funcion que captura los envelopes.
  - RESPUESTAS REPETIDAS. La pulsera envia cada respuesta varias veces con el
    mismo id_secuencia, para que la perdida de un datagrama no haga vencer el
    plazo de un conductor que respondio bien. El receptor publica la primera
    copia y descarta las demas, de modo que el nucleo reciba una sola.
  - NINGUN datagrama puede terminar el hilo. Lo que no cumple el protocolo se
    cuenta y se descarta; cualquier otra excepcion tambien. Este socket queda
    expuesto a todo lo que circule por la red local.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from common.contratos import (
    Envelope,
    EventoAckWearable,
    EventoWearable,
    OrigenEvento,
    TipoMensaje,
    generar_id_mensaje,
    generar_id_sesion,
)
from NeuroDrive_Wearable import protocolo


@dataclass
class EstadisticasReceptor:
    paquetes_recibidos: int = 0
    telemetrias: int = 0
    acks: int = 0                  # respuestas publicadas (una por verificacion)
    acks_duplicados: int = 0       # copias repetidas descartadas
    invalidos: int = 0
    publicados: int = 0
    descartes_publicacion: int = 0


class ReceptorWearable:
    # Una respuesta con el mismo id_secuencia que la ultima publicada, llegada
    # dentro de esta ventana, es una copia. Las copias salen a 50 ms entre si;
    # la ventana es amplia frente a eso y corta frente al intervalo entre dos
    # verificaciones distintas, que ademas llevan identificadores distintos.
    VENTANA_ACK_DUPLICADO_S = 2.0

    def __init__(
        self,
        puerto_escucha: int = 5005,
        nombre_cola: str = "/neurodrive_wearable",
        capacidad_cola: int = 10,
        tamano_max_mensaje: int = 1024,
        id_dispositivo: str = "wearable-01",
        id_sesion: Optional[str] = None,
        publicar_fn: Optional[Callable[[str], bool]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._puerto = puerto_escucha
        self._nombre_cola = nombre_cola
        self._capacidad_cola = capacidad_cola
        self._tamano_max = tamano_max_mensaje
        self._id_dispositivo = id_dispositivo
        self._id_sesion = id_sesion or generar_id_sesion("wea")
        self._publicar_fn_inyectada = publicar_fn
        self.log = logger or logging.getLogger("NeuroDrive.ReceptorWearable")

        self._sock: Optional[socket.socket] = None
        self._adaptador = None
        self._publicar_fn: Optional[Callable[[str], bool]] = None
        self._hilo: Optional[threading.Thread] = None
        self._activo = False
        self._numero_secuencia = 0
        # Ultima respuesta publicada, para reconocer sus copias
        self._ultimo_ack_id: Optional[int] = None
        self._ultimo_ack_ts: float = 0.0
        self.stats = EstadisticasReceptor()

    # ------------------------------------------------------------------
    def iniciar(self) -> None:
        if self._activo:
            return

        # Resolver la funcion de publicacion
        if self._publicar_fn_inyectada is not None:
            self._publicar_fn = self._publicar_fn_inyectada
        else:
            from NeuroDrive_Core.adaptador_mq import AdaptadorMQ
            self._adaptador = AdaptadorMQ.abrir(
                nombre=self._nombre_cola,
                modo="escritura",
                capacidad=self._capacidad_cola,
                tamano_max_mensaje=self._tamano_max,
            )
            self._publicar_fn = self._adaptador.enviar

        # Socket UDP de escucha. Si el bind falla (puerto ocupado por otra
        # instancia) no debe quedar la cola abierta a medias.
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("0.0.0.0", self._puerto))
            # Lectura acotada: el hilo no tiene otra tarea, pero una lectura
            # sin limite le impediria advertir el pedido de apagado.
            self._sock.settimeout(0.5)
        except OSError:
            self._cerrar_recursos()
            raise

        self._activo = True
        self._hilo = threading.Thread(
            target=self._bucle_receptor, name="WearableReceptor", daemon=True
        )
        self._hilo.start()
        self.log.info("ReceptorWearable escuchando en UDP :%d", self._puerto)

    def detener(self, timeout: float = 2.0) -> None:
        if not self._activo:
            return
        self._activo = False
        if self._hilo is not None:
            self._hilo.join(timeout=timeout)
            self._hilo = None
        self._cerrar_recursos()
        self.log.info("ReceptorWearable detenido")

    def _cerrar_recursos(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._adaptador is not None:
            try:
                self._adaptador.cerrar()
            except Exception:
                pass
            self._adaptador = None

    # ------------------------------------------------------------------
    def _bucle_receptor(self) -> None:
        while self._activo:
            try:
                datos, _addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break                      # socket cerrado: fin del hilo
            self.stats.paquetes_recibidos += 1
            try:
                self._procesar_datagrama(datos)
            except Exception as e:
                # Red de seguridad: un datagrama no puede terminar el hilo.
                # Si el hilo terminara, el nucleo veria a la pulsera en silencio
                # y pasaria a operacion degradada con la pulsera funcionando.
                self.stats.invalidos += 1
                self.log.error("Error inesperado procesando un datagrama: %s", e,
                               exc_info=True)

    def _procesar_datagrama(self, datos: bytes) -> None:
        # 1. Parsear y construir el evento del contrato
        ts = time.time()
        try:
            crudo = protocolo.parsear_mensaje_pulsera(datos)
            evento = protocolo.construir_evento(crudo, ts)
        except protocolo.ErrorProtocolo as e:
            self.stats.invalidos += 1
            self.log.warning("Datagrama invalido descartado: %s", e)
            return

        # 2. Clasificar. Las copias repetidas de una respuesta terminan aca.
        es_ack = isinstance(evento, EventoAckWearable)
        if es_ack:
            if self._es_ack_duplicado(evento, ts):
                self.stats.acks_duplicados += 1
                return
            tipo_msg = TipoMensaje.EVENTO_ACK_WEARABLE
        elif isinstance(evento, EventoWearable):
            tipo_msg = TipoMensaje.EVENTO_WEARABLE
            self.stats.telemetrias += 1
        else:                               # no deberia ocurrir
            self.stats.invalidos += 1
            return

        # 3. Envolver en Envelope
        self._numero_secuencia += 1
        envelope = Envelope(
            tipo=tipo_msg,
            origen=OrigenEvento.WEARABLE,
            id_dispositivo=self._id_dispositivo,
            id_sesion=self._id_sesion,
            id_mensaje=generar_id_mensaje("wea", self._numero_secuencia),
            numero_secuencia=self._numero_secuencia,
            timestamp_origen=ts,
            evento=evento,
        )

        # 4. Publicar
        try:
            ok = self._publicar_fn(envelope.to_json())
        except Exception as e:
            self.stats.descartes_publicacion += 1
            self.log.error("Fallo publicando en la cola: %s", e)
            return
        if not ok:
            self.stats.descartes_publicacion += 1
            self.log.warning("Cola del wearable llena; evento descartado")
            return
        self.stats.publicados += 1
        if es_ack:
            # Se registra recien ahora: si la publicacion hubiera fallado, la
            # copia siguiente debe tener su oportunidad.
            self.stats.acks += 1
            self._ultimo_ack_id = evento.id_secuencia
            self._ultimo_ack_ts = ts

    def _es_ack_duplicado(self, evento: EventoAckWearable, ts: float) -> bool:
        """True si esta respuesta es una copia de la ultima PUBLICADA."""
        return (
            self._ultimo_ack_id is not None
            and evento.id_secuencia == self._ultimo_ack_id
            and (ts - self._ultimo_ack_ts) <= self.VENTANA_ACK_DUPLICADO_S
        )
