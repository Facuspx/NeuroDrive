"""
NeuroDrive Core - Gestor de Eventos
====================================

Componente que actua como puente entre los procesos externos del sistema
(NeuroDrive_Vision, NeuroDrive_Wearable) y el resto del Core.

Responsabilidades:
  1. Levantar hilos lectores, uno por cada POSIX MQ externa.
  2. Parsear cada Envelope JSON recibido y validar coherencia.
  3. Deduplicar mensajes repetidos por (id_dispositivo, numero_secuencia).
  4. Encolar los eventos validos en una queue.Queue interna que el
     Pre-FSM consume.
  5. Monitorear heartbeats: si un sensor deja de enviar mensajes por
     mas tiempo del configurado, emite EventoFalloSensor sinteticamente.
     Cuando el sensor vuelve, emite EventoRecuperacionSensor.
  6. Manejar SIGINT/SIGTERM para cierre limpio.

Arquitectura interna (4 hilos):
    ┌────────────────────────────────────────────────┐
    │              MAIN (FSM consumidor)             │
    └────────────────────────────────────────────────┘
                          ▲ obtener_evento()
                          │
              queue.Queue interna (cap=256)
                          ▲
       ┌──────────────────┼──────────────────┐
       │                  │                  │
    HILO Vision      HILO Wearable      HILO Monitor
    (lee MQ)         (lee MQ)           (heartbeat)
       │                  │                  │
       │                  │                  └─> emite EventoFalloSensor/
       │                  │                      EventoRecuperacionSensor
       ▼                  ▼
   POSIX MQ           POSIX MQ
   /neurodrive_       /neurodrive_
    vision             wearable

Uso tipico:

    config = cargar_config()
    gestor = GestorEventos(config)
    gestor.iniciar()
    try:
        while gestor.activo:
            envelope = gestor.obtener_evento(timeout=1.0)
            if envelope is not None:
                # pasarlo al Pre-FSM
                pre_fsm.procesar(envelope)
    finally:
        gestor.detener()
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from common.contratos import (
    Envelope,
    EventoFalloSensor,
    EventoRecuperacionSensor,
    OrigenEvento,
    TipoMensaje,
    generar_id_mensaje,
    timestamp_actual,
)
from NeuroDrive_Core.adaptador_mq import (
    AdaptadorMQ,
    ErrorAdaptadorMQ,
)
from NeuroDrive_Core.config_loader import Config


_log = logging.getLogger("NeuroDrive.GestorEventos")


# =============================================================================
#                      ESTADO DE SALUD POR SENSOR
# =============================================================================


@dataclass
class SaludSensor:
    """Estado de salud rastreado por el monitor de heartbeat."""
    nombre: str
    origen: OrigenEvento
    ultimo_mensaje_ts: float = 0.0
    caido: bool = False
    timestamp_inicio_caida: float = 0.0
    # Set de numeros de secuencia recientes para deduplicacion
    secuencias_recientes: deque = field(default_factory=lambda: deque(maxlen=200))


# =============================================================================
#                          ESTADISTICAS DEL GESTOR
# =============================================================================


@dataclass
class EstadisticasGestor:
    """Contadores para monitoreo y debug."""
    mensajes_recibidos_vision: int = 0
    mensajes_recibidos_wearable: int = 0
    mensajes_invalidos: int = 0
    duplicados_descartados: int = 0
    cola_interna_llena_descartes: int = 0
    fallos_sensor_emitidos: int = 0
    recuperaciones_emitidas: int = 0
    errores_lectura_mq: int = 0


# =============================================================================
#                              CLASE PRINCIPAL
# =============================================================================


class GestorEventos:
    """
    Gestor centralizado de eventos del sistema.

    Despues de iniciar(), el codigo principal solo necesita llamar a
    obtener_evento(timeout) en bucle. Todo el manejo de hilos, colas
    POSIX, validacion, deduplicacion y heartbeats es transparente.
    """

    # Capacidad de la cola interna (Gestor -> Pre-FSM)
    CAPACIDAD_COLA_INTERNA = 256

    # Periodo de chequeo del monitor de heartbeats (segundos)
    PERIODO_MONITOR_HEARTBEAT_SEG = 1.0

    # Tolerancia en segundos para timestamps del futuro (clock skew)
    TOLERANCIA_TIMESTAMP_FUTURO_SEG = 5.0

    # Salto en numero_secuencia que se considera "reinicio" del productor
    UMBRAL_SALTO_SECUENCIA = 1000

    def __init__(self, config: Config) -> None:
        self.config = config

        # Cola interna (thread-safe, bloqueante con timeout)
        self.cola_interna: queue.Queue = queue.Queue(
            maxsize=self.CAPACIDAD_COLA_INTERNA
        )

        # Flag de control: cuando se setea, todos los hilos terminan
        self._parar = threading.Event()
        self._iniciado = False

        # Adaptadores MQ (se abren en iniciar())
        self._mq_vision: Optional[AdaptadorMQ] = None
        self._mq_wearable: Optional[AdaptadorMQ] = None

        # Hilos
        self._hilo_vision: Optional[threading.Thread] = None
        self._hilo_wearable: Optional[threading.Thread] = None
        self._hilo_monitor: Optional[threading.Thread] = None

        # Estado de salud por sensor
        self._salud: Dict[OrigenEvento, SaludSensor] = {
            OrigenEvento.VISION: SaludSensor(
                nombre=self.config.identificadores.id_camara,
                origen=OrigenEvento.VISION,
            ),
            OrigenEvento.WEARABLE: SaludSensor(
                nombre=self.config.identificadores.id_wearable,
                origen=OrigenEvento.WEARABLE,
            ),
        }

        # Lock para proteger _salud (accedido por hilos lectores y monitor)
        self._lock_salud = threading.Lock()

        # Contador interno para generar id_mensaje en eventos sinteticos
        self._contador_eventos_internos = 0

        # Estadisticas
        self.stats = EstadisticasGestor()

        # Handlers de senales previos (para restaurarlos al detener)
        self._handler_sigint_previo = None
        self._handler_sigterm_previo = None

    # ==================================================================
    # PROPIEDADES PUBLICAS
    # ==================================================================

    @property
    def activo(self) -> bool:
        """True si el gestor esta corriendo y no se solicito parada."""
        return self._iniciado and not self._parar.is_set()

    # ==================================================================
    # CICLO DE VIDA
    # ==================================================================

    def iniciar(self) -> None:
        """
        Inicia el gestor: abre las MQs, levanta los 3 hilos, registra
        handlers de senales para cierre limpio.

        Idempotente: llamar varias veces no causa error.
        """
        if self._iniciado:
            _log.warning("iniciar() llamado pero gestor ya iniciado")
            return

        _log.info("Iniciando GestorEventos...")

        try:
            self._abrir_colas_mq()
        except Exception as e:
            _log.error("Error al abrir colas MQ: %s", e)
            self._cerrar_colas_mq()
            raise

        # Registrar handlers de senales
        self._registrar_handlers_senales()

        # Inicializar timestamps de salud al momento de arranque
        ahora = timestamp_actual()
        with self._lock_salud:
            for salud in self._salud.values():
                salud.ultimo_mensaje_ts = ahora

        # Lanzar los 3 hilos
        self._hilo_vision = threading.Thread(
            target=self._loop_lector,
            args=(self._mq_vision, OrigenEvento.VISION, "HiloVision"),
            name="HiloLectorVision",
            daemon=True,
        )
        self._hilo_wearable = threading.Thread(
            target=self._loop_lector,
            args=(self._mq_wearable, OrigenEvento.WEARABLE, "HiloWearable"),
            name="HiloLectorWearable",
            daemon=True,
        )
        self._hilo_monitor = threading.Thread(
            target=self._loop_monitor_salud,
            name="HiloMonitorSalud",
            daemon=True,
        )

        self._iniciado = True
        self._parar.clear()

        self._hilo_vision.start()
        self._hilo_wearable.start()
        self._hilo_monitor.start()

        _log.info("GestorEventos iniciado correctamente")

    def detener(self, timeout_join: float = 5.0) -> None:
        """
        Detiene el gestor: senala a los hilos que terminen, los espera,
        cierra las colas, restaura handlers.

        Idempotente.
        """
        if not self._iniciado:
            return

        _log.info("Deteniendo GestorEventos...")
        self._parar.set()

        # Esperar hilos
        for hilo in (self._hilo_vision, self._hilo_wearable, self._hilo_monitor):
            if hilo is not None and hilo.is_alive():
                hilo.join(timeout=timeout_join)
                if hilo.is_alive():
                    _log.warning(
                        "Hilo %s no termino en %.1f s",
                        hilo.name, timeout_join,
                    )

        # Cerrar colas MQ
        self._cerrar_colas_mq()

        # Restaurar handlers
        self._restaurar_handlers_senales()

        self._iniciado = False
        _log.info(
            "GestorEventos detenido. Stats finales: %s",
            self.stats,
        )

    # ==================================================================
    # API DE CONSUMO
    # ==================================================================

    def obtener_evento(self, timeout: float = 1.0) -> Optional[Envelope]:
        """
        Obtiene el proximo Envelope de la cola interna.

        Bloquea hasta `timeout` segundos. Si no hay evento, devuelve None.

        El Envelope devuelto ya tiene el campo `evento` poblado (modo
        interno) para que el consumidor no tenga que desempacarlo de nuevo.
        """
        try:
            return self.cola_interna.get(timeout=timeout)
        except queue.Empty:
            return None

    # ==================================================================
    # HILOS LECTORES
    # ==================================================================

    def _loop_lector(
        self,
        cola_mq: AdaptadorMQ,
        origen: OrigenEvento,
        nombre_hilo: str,
    ) -> None:
        """
        Bucle principal de un hilo lector.

        Lee de la POSIX MQ con timeout de 1s, valida cada mensaje,
        deduplica, y lo encola en la cola interna.
        """
        _log.info("[%s] Hilo lector iniciado para %s", nombre_hilo, origen.name)

        while not self._parar.is_set():
            try:
                payload_json = cola_mq.recibir(timeout_seg=1.0)
            except ErrorAdaptadorMQ as e:
                _log.error("[%s] Error leyendo MQ: %s", nombre_hilo, e)
                self.stats.errores_lectura_mq += 1
                # Esperar un poco antes de reintentar para no inundar logs
                time.sleep(0.5)
                continue
            except Exception as e:
                _log.exception("[%s] Error inesperado: %s", nombre_hilo, e)
                self.stats.errores_lectura_mq += 1
                time.sleep(0.5)
                continue

            if payload_json is None:
                # Timeout normal, seguimos
                continue

            # Tenemos un mensaje. Procesarlo.
            self._procesar_mensaje_recibido(payload_json, origen, nombre_hilo)

        _log.info("[%s] Hilo lector terminado", nombre_hilo)

    def _procesar_mensaje_recibido(
        self,
        payload_json: str,
        origen_esperado: OrigenEvento,
        nombre_hilo: str,
    ) -> None:
        """
        Procesa un mensaje crudo: parsea, valida, deduplica, encola.

        Cualquier error es no-fatal: descarta el mensaje y loggea.
        """
        ts_recepcion = timestamp_actual()

        # 1. Parsear JSON -> Envelope
        try:
            envelope = Envelope.from_json(payload_json)
        except (ValueError, KeyError, TypeError) as e:
            _log.warning(
                "[%s] Envelope malformado (descartado): %s. Payload: %s",
                nombre_hilo, e, payload_json[:200],
            )
            self.stats.mensajes_invalidos += 1
            return

        # 2. Validar coherencia origen <-> tipo
        if not self._validar_coherencia(envelope, origen_esperado, nombre_hilo):
            self.stats.mensajes_invalidos += 1
            return

        # 3. Validar timestamp no del futuro
        if envelope.timestamp_origen > ts_recepcion + self.TOLERANCIA_TIMESTAMP_FUTURO_SEG:
            _log.warning(
                "[%s] Mensaje %s con timestamp del futuro (diff=%.2fs), descartado",
                nombre_hilo,
                envelope.id_mensaje,
                envelope.timestamp_origen - ts_recepcion,
            )
            self.stats.mensajes_invalidos += 1
            return

        # 4. Validar / detectar duplicado por numero_secuencia
        if not self._verificar_no_duplicado(envelope, origen_esperado):
            _log.debug(
                "[%s] Duplicado descartado: id_msg=%s seq=%d",
                nombre_hilo,
                envelope.id_mensaje,
                envelope.numero_secuencia,
            )
            self.stats.duplicados_descartados += 1
            return

        # 5. Desempacar el payload a objeto Python
        try:
            evento = envelope.desempacar()
        except (ValueError, KeyError, TypeError) as e:
            _log.warning(
                "[%s] No se pudo desempacar Envelope %s: %s",
                nombre_hilo, envelope.id_mensaje, e,
            )
            self.stats.mensajes_invalidos += 1
            return

        # 6. Reconstruir Envelope en modo interno con el objeto poblado
        # (asi el Pre-FSM no tiene que volver a desempacar)
        envelope_interno = Envelope(
            tipo=envelope.tipo,
            origen=envelope.origen,
            id_dispositivo=envelope.id_dispositivo,
            id_sesion=envelope.id_sesion,
            id_mensaje=envelope.id_mensaje,
            numero_secuencia=envelope.numero_secuencia,
            timestamp_origen=envelope.timestamp_origen,
            timestamp_recepcion=ts_recepcion,
            evento=evento,
            version=envelope.version,
        )

        # 7. Actualizar estado de salud (heartbeat implicito)
        self._registrar_actividad_sensor(origen_esperado, ts_recepcion)

        # 8. Encolar en cola interna (politica B: descartar si llena)
        if not self._encolar_envelope(envelope_interno):
            self.stats.cola_interna_llena_descartes += 1

        # 9. Actualizar contador segun origen
        if origen_esperado == OrigenEvento.VISION:
            self.stats.mensajes_recibidos_vision += 1
        elif origen_esperado == OrigenEvento.WEARABLE:
            self.stats.mensajes_recibidos_wearable += 1

    # ==================================================================
    # VALIDACIONES
    # ==================================================================

    def _validar_coherencia(
        self,
        envelope: Envelope,
        origen_esperado: OrigenEvento,
        nombre_hilo: str,
    ) -> bool:
        """
        Verifica que el origen y el tipo del Envelope sean coherentes
        con la cola por la que llego.
        """
        # El origen del Envelope debe coincidir con la cola
        if envelope.origen != origen_esperado:
            _log.warning(
                "[%s] Envelope con origen=%s pero llego por cola de %s, descartado",
                nombre_hilo,
                envelope.origen.name,
                origen_esperado.name,
            )
            return False

        # Tipos validos por origen
        tipos_validos_vision = {TipoMensaje.EVENTO_VISION, TipoMensaje.HEARTBEAT}
        tipos_validos_wearable = {
            TipoMensaje.EVENTO_WEARABLE,
            TipoMensaje.EVENTO_ACK_WEARABLE,
            TipoMensaje.HEARTBEAT,
        }

        if origen_esperado == OrigenEvento.VISION and envelope.tipo not in tipos_validos_vision:
            _log.warning(
                "[%s] Tipo %s no esperado desde VISION, descartado",
                nombre_hilo, envelope.tipo.name,
            )
            return False

        if origen_esperado == OrigenEvento.WEARABLE and envelope.tipo not in tipos_validos_wearable:
            _log.warning(
                "[%s] Tipo %s no esperado desde WEARABLE, descartado",
                nombre_hilo, envelope.tipo.name,
            )
            return False

        return True

    def _verificar_no_duplicado(
        self,
        envelope: Envelope,
        origen: OrigenEvento,
    ) -> bool:
        """
        Verifica que (id_dispositivo, numero_secuencia) no se haya visto antes.

        Devuelve True si el mensaje es nuevo, False si es duplicado.

        Tambien detecta saltos grandes en numero_secuencia (reinicio del
        productor) y limpia el deque en ese caso, ya que la nueva secuencia
        empieza de cero.
        """
        with self._lock_salud:
            salud = self._salud[origen]
            seq = envelope.numero_secuencia

            # Detectar reinicio del productor (salto enorme hacia atras)
            if salud.secuencias_recientes:
                ultima_vista = salud.secuencias_recientes[-1]
                if seq < ultima_vista - self.UMBRAL_SALTO_SECUENCIA:
                    _log.info(
                        "Sensor %s parece haberse reiniciado (seq saltó de %d a %d)",
                        origen.name, ultima_vista, seq,
                    )
                    salud.secuencias_recientes.clear()

            # Verificar duplicado
            if seq in salud.secuencias_recientes:
                return False

            # Es nuevo, registrarlo
            salud.secuencias_recientes.append(seq)
            return True

    # ==================================================================
    # MONITOR DE SALUD (heartbeat)
    # ==================================================================

    def _loop_monitor_salud(self) -> None:
        """
        Bucle del monitor de heartbeats.

        Cada N segundos, verifica si algun sensor lleva mas de
        timeout_heartbeat_seg sin enviar mensaje. Si es asi y no estaba
        marcado como caido, emite EventoFalloSensor.

        Si un sensor caido vuelve a enviar (verificado por el hilo lector
        que actualiza ultimo_mensaje_ts), emite EventoRecuperacionSensor.
        """
        _log.info("Hilo monitor de salud iniciado")
        timeout = self.config.wearable.timeout_heartbeat_seg

        while not self._parar.is_set():
            # Dormir con check de parada para terminar rapido
            if self._parar.wait(timeout=self.PERIODO_MONITOR_HEARTBEAT_SEG):
                break

            ahora = timestamp_actual()
            cambios = []  # (sensor, accion) para emitir eventos fuera del lock

            with self._lock_salud:
                for origen, salud in self._salud.items():
                    silencio = ahora - salud.ultimo_mensaje_ts

                    # Sensor que estaba activo y dejo de responder
                    if not salud.caido and silencio > timeout:
                        salud.caido = True
                        salud.timestamp_inicio_caida = ahora
                        cambios.append(("fallo", origen, salud.nombre, silencio))

                    # Sensor que estaba caido y volvio
                    elif salud.caido and silencio <= timeout:
                        tiempo_caido = ahora - salud.timestamp_inicio_caida
                        salud.caido = False
                        salud.timestamp_inicio_caida = 0.0
                        cambios.append(("recuperacion", origen, salud.nombre, tiempo_caido))

            # Emitir eventos fuera del lock
            for accion, origen, nombre, valor in cambios:
                if accion == "fallo":
                    self._emitir_fallo_sensor(origen, nombre, valor)
                elif accion == "recuperacion":
                    self._emitir_recuperacion_sensor(origen, nombre, valor)

        _log.info("Hilo monitor de salud terminado")

    def _emitir_fallo_sensor(
        self,
        origen: OrigenEvento,
        nombre_sensor: str,
        segundos_silencio: float,
    ) -> None:
        """Construye y encola un EventoFalloSensor sintetico."""
        evento = EventoFalloSensor(
            timestamp=timestamp_actual(),
            sensor_afectado=origen,
            motivo=f"heartbeat_timeout:{segundos_silencio:.1f}s",
            severidad=2,
        )
        self._encolar_evento_interno(
            tipo=TipoMensaje.FALLO_SENSOR,
            evento=evento,
        )
        self.stats.fallos_sensor_emitidos += 1
        _log.warning(
            "Sensor %s (%s) caido tras %.1fs de silencio, EventoFalloSensor emitido",
            nombre_sensor, origen.name, segundos_silencio,
        )

    def _emitir_recuperacion_sensor(
        self,
        origen: OrigenEvento,
        nombre_sensor: str,
        tiempo_caido_seg: float,
    ) -> None:
        """Construye y encola un EventoRecuperacionSensor sintetico."""
        evento = EventoRecuperacionSensor(
            timestamp=timestamp_actual(),
            sensor_recuperado=origen,
            tiempo_caido_seg=tiempo_caido_seg,
        )
        self._encolar_evento_interno(
            tipo=TipoMensaje.RECUPERACION_SENSOR,
            evento=evento,
        )
        self.stats.recuperaciones_emitidas += 1
        _log.info(
            "Sensor %s (%s) recuperado tras %.1fs caido",
            nombre_sensor, origen.name, tiempo_caido_seg,
        )

    def _registrar_actividad_sensor(
        self,
        origen: OrigenEvento,
        timestamp: float,
    ) -> None:
        """Actualiza el timestamp del ultimo mensaje recibido por el sensor."""
        with self._lock_salud:
            self._salud[origen].ultimo_mensaje_ts = timestamp

    # ==================================================================
    # ENCOLADO INTERNO
    # ==================================================================

    def _encolar_envelope(self, envelope: Envelope) -> bool:
        """
        Encola un Envelope en la cola interna. Politica B: si esta llena,
        descarta y devuelve False.
        """
        try:
            self.cola_interna.put(envelope, block=False)
            return True
        except queue.Full:
            _log.warning(
                "Cola interna llena, descartando mensaje %s (total: %d descartes)",
                envelope.id_mensaje,
                self.stats.cola_interna_llena_descartes + 1,
            )
            return False

    def _encolar_evento_interno(self, tipo: TipoMensaje, evento) -> None:
        """
        Construye y encola un Envelope sintetico generado por el Gestor mismo.

        Usado para FALLO_SENSOR y RECUPERACION_SENSOR detectados por el
        monitor de heartbeat.
        """
        self._contador_eventos_internos += 1
        prefijo = self.config.identificadores.prefijo_mensaje_interno

        envelope = Envelope(
            tipo=tipo,
            origen=OrigenEvento.INTERNO,
            id_dispositivo=self.config.identificadores.id_core,
            id_sesion="ses-runtime",  # el ID de sesion real se setea aparte
            id_mensaje=generar_id_mensaje(prefijo, self._contador_eventos_internos),
            numero_secuencia=self._contador_eventos_internos,
            timestamp_origen=evento.timestamp,
            timestamp_recepcion=evento.timestamp,
            evento=evento,
        )
        self._encolar_envelope(envelope)

    # ==================================================================
    # APERTURA Y CIERRE DE COLAS MQ
    # ==================================================================

    def _abrir_colas_mq(self) -> None:
        """Abre las dos colas POSIX MQ en modo lectura."""
        self._mq_vision = AdaptadorMQ.abrir(
            nombre=self.config.ipc.cola_vision,
            modo="lectura",
            capacidad=self.config.ipc.capacidad_cola,
            tamano_max_mensaje=self.config.ipc.tamano_max_mensaje_bytes,
        )
        self._mq_wearable = AdaptadorMQ.abrir(
            nombre=self.config.ipc.cola_wearable,
            modo="lectura",
            capacidad=self.config.ipc.capacidad_cola,
            tamano_max_mensaje=self.config.ipc.tamano_max_mensaje_bytes,
        )

    def _cerrar_colas_mq(self) -> None:
        """Cierra las colas MQ. Idempotente."""
        if self._mq_vision is not None:
            try:
                self._mq_vision.cerrar()
            except Exception as e:
                _log.warning("Error cerrando MQ vision: %s", e)
            self._mq_vision = None

        if self._mq_wearable is not None:
            try:
                self._mq_wearable.cerrar()
            except Exception as e:
                _log.warning("Error cerrando MQ wearable: %s", e)
            self._mq_wearable = None

    # ==================================================================
    # MANEJO DE SENALES
    # ==================================================================

    def _registrar_handlers_senales(self) -> None:
        """
        Registra handlers para SIGINT y SIGTERM.

        Solo lo hace si estamos en el hilo principal (signal solo se puede
        registrar desde el main thread). Si se llama desde un hilo, se loggea
        y se ignora (el llamador puede manejar señales por su cuenta).
        """
        if threading.current_thread() is not threading.main_thread():
            _log.info(
                "iniciar() llamado fuera del hilo principal, "
                "no se registran handlers de senales"
            )
            return

        def handler(signum, frame):
            _log.info("Senal %d recibida, deteniendo gestor", signum)
            self._parar.set()

        try:
            self._handler_sigint_previo = signal.signal(signal.SIGINT, handler)
            self._handler_sigterm_previo = signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError) as e:
            _log.warning("No se pudieron registrar handlers de senales: %s", e)

    def _restaurar_handlers_senales(self) -> None:
        """Restaura los handlers de senales previos."""
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            if self._handler_sigint_previo is not None:
                signal.signal(signal.SIGINT, self._handler_sigint_previo)
            if self._handler_sigterm_previo is not None:
                signal.signal(signal.SIGTERM, self._handler_sigterm_previo)
        except (ValueError, OSError):
            pass

    # ==================================================================
    # CONTEXT MANAGER
    # ==================================================================

    def __enter__(self) -> GestorEventos:
        self.iniciar()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detener()


__all__ = [
    "GestorEventos",
    "EstadisticasGestor",
    "SaludSensor",
]
