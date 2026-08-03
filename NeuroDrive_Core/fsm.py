"""
NeuroDrive Core - FSM (Maquina de Estados Finita)
==================================================

Logica pura de decision del nivel de somnolencia.

ENTRADA: EventoProcesado (del Pre-FSM) o EventoAckWearable (del wearable)
        o EventoFalloSensor / EventoRecuperacionSensor (del Gestor)
SALIDA:  SalidaFSM con estado actual + lista de ComandoActuador

Esta clase NO:
  - Lee colas, queues ni archivos
  - Toca GPIO, audio ni red
  - Tiene timers reales (usa el timestamp del evento como reloj)

Esto la hace 100% determinista y testeable sin hardware.

Reglas de transicion (ver diagrama de planificacion):
  S0 NORMAL          -> S1 PRE_ALERTA   por señales leves sostenidas
  S1 PRE_ALERTA      -> S2 ALERTA_LEVE  por evento confirmado (bostezo, microsueño)
  S1 PRE_ALERTA      -> S0 NORMAL       tras 60s sin eventos negativos (configurable)
  S2 ALERTA_LEVE     -> S3 ALERTA_MEDIA por timeout ACK (30s) o microsueño largo
  S2 ALERTA_LEVE     -> S1 PRE_ALERTA   por ACK correcto
  S3 ALERTA_MEDIA    -> S4 CRITICO      por timeout ACK (20s) + BPM critico
  S3 ALERTA_MEDIA    -> S1 PRE_ALERTA   por ACK correcto + BPM normal
  S4 CRITICO         -> S1 PRE_ALERTA   por reaccion activa (ACK en 15s)
  Cualquier estado   -> S5 DEGRADADO    por EventoFalloSensor severidad >= 2
  S5 DEGRADADO       -> S0 NORMAL       por EventoRecuperacionSensor

Reglas adicionales:
  - Pausa de timer por rostro perdido: si el evento dice
    vision_disponible=False, NO se computa tiempo en la histeresis
    de bajada de estado.
  - Contadores siguen corriendo: aunque estemos en S2 esperando ACK,
    un segundo bostezo escala directo a S3 sin esperar timeout.
  - Persistencia: si la sesion previa fue S5, arranca en S0; cualquier
    otro estado guardado arranca en S1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from common.contratos import (
    ComandoActuador,
    EstadoFSM,
    EventoAckWearable,
    EventoFalloSensor,
    EventoProcesado,
    EventoRecuperacionSensor,
    NivelRiesgoBPM,
    OrigenEvento,
    SalidaFSM,
    TipoComandoActuador,
)


# Tipo unión de los eventos que la FSM puede consumir
EventoEntradaFSM = Union[
    EventoProcesado,
    EventoAckWearable,
    EventoFalloSensor,
    EventoRecuperacionSensor,
]


# =============================================================================
#                    CONFIGURACION DE LA FSM (desde config.yaml)
# =============================================================================


@dataclass
class ConfigFSM:
    """
    Parametros configurables de la FSM. Se inicializa con valores por defecto
    y se sobrescribe con los valores de config.yaml en el constructor.

    Todos los tiempos en segundos.
    """
    # Histeresis de bajada
    tiempo_para_bajar_estado_seg: float = 60.0

    # Timeouts de ACK por nivel de alerta
    timeout_ack_leve_seg: float = 30.0
    timeout_ack_medio_seg: float = 20.0
    timeout_ack_critico_seg: float = 15.0

    # Umbrales de escalada por acumulacion
    max_microsuenos_ventana_corta: int = 1
    max_bostezos_ventana_corta: int = 3
    max_cabeceos_ventana_corta: int = 1

    # Mejoras de manejo de somnolencia
    calentamiento_senales_seg: float = 60.0        # ignorar parpadeos/min al arrancar
    persistencia_senales_leves_seg: float = 20.0   # señal continua sostenida para PRE_ALERTA
    perclos_confirmado: float = 0.35               # PERCLOS "parpados pesados"
    perclos_confirmado_sostenido_seg: float = 30.0
    max_eventos_severos_ventana: int = 3           # episodios para fatiga recurrente
    ventana_episodios_seg: float = 900.0           # 15 min
    umbral_respuesta_lenta_ms: int = 5000          # ACK correcto pero lento

    @classmethod
    def desde_dict(cls, config: Dict[str, Any]) -> ConfigFSM:
        """Construye una ConfigFSM desde el dict cargado de config.yaml.

        Espera la estructura:
            {
                "fsm": {...},
                "wearable": {...}
            }
        """
        fsm_cfg = config.get("fsm", {})
        wea_cfg = config.get("wearable", {})

        return cls(
            tiempo_para_bajar_estado_seg=float(
                fsm_cfg.get("tiempo_para_bajar_estado_seg", 60.0)
            ),
            timeout_ack_leve_seg=float(
                wea_cfg.get("timeout_ack_leve_seg", 30.0)
            ),
            timeout_ack_medio_seg=float(
                wea_cfg.get("timeout_ack_medio_seg", 20.0)
            ),
            timeout_ack_critico_seg=float(
                wea_cfg.get("timeout_ack_critico_seg", 15.0)
            ),
            max_microsuenos_ventana_corta=int(
                fsm_cfg.get("max_microsuenos_ventana_corta", 1)
            ),
            max_bostezos_ventana_corta=int(
                fsm_cfg.get("max_bostezos_ventana_corta", 3)
            ),
            max_cabeceos_ventana_corta=int(
                fsm_cfg.get("max_cabeceos_ventana_corta", 1)
            ),
        )


# =============================================================================
#                     ESTADO INTERNO DE LA FSM (mutable)
# =============================================================================


@dataclass
class EstadoInternoFSM:
    """
    Estado mutable que la FSM mantiene entre eventos.

    Se separa del estado declarativo (EstadoFSM enum) para tener
    visibilidad completa sobre los contadores y timers de cada estado.
    """
    estado_actual: EstadoFSM = EstadoFSM.NORMAL

    # Timestamp del ultimo evento procesado (sirve de "reloj")
    ultimo_timestamp: float = 0.0

    # Timestamp en el que entramos al estado actual
    timestamp_entrada_estado: float = 0.0

    # Acumulador de tiempo "limpio" en estado actual (descuenta pausas)
    tiempo_acumulado_sin_eventos: float = 0.0
    ultimo_timestamp_evaluacion: float = 0.0

    # Numero de secuencia de ACK actualmente solicitada (None = ninguna)
    id_secuencia_ack_pendiente: Optional[int] = None
    timestamp_ack_solicitado: float = 0.0

    # Sensores con fallo activo (set de OrigenEvento como int)
    sensores_caidos: set = field(default_factory=set)

    # Estado previo al entrar en S5 (para volver a algo razonable)
    estado_previo_a_degradado: EstadoFSM = EstadoFSM.NORMAL

    # Contador interno de secuencias de ACK
    contador_secuencia_ack: int = 0

    ts_primer_evento: float = 0.0
    ts_inicio_senales: float = 0.0
    ts_inicio_perclos_alto: float = 0.0
    episodios_severos: List[float] = field(default_factory=list)


# =============================================================================
#                              CLASE PRINCIPAL
# =============================================================================


class FSM:
    """
    Maquina de estados finita de NeuroDrive.

    Uso basico:
        fsm = FSM(ConfigFSM())
        salida = fsm.procesar_evento(evento)
        print(salida.estado_actual, salida.comandos)

    La FSM es determinista: el mismo orden de eventos produce siempre
    la misma secuencia de salidas. No tiene timers internos: el tiempo
    se mide a partir del campo .timestamp de cada evento.
    """

    def __init__(
        self,
        config: ConfigFSM,
        estado_inicial: EstadoFSM = EstadoFSM.NORMAL,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.estado = EstadoInternoFSM(estado_actual=estado_inicial)
        self.log = logger or logging.getLogger("NeuroDrive.FSM")
        self._reemitir = False  # re-emitir comandos sin cambio de estado (re-desafio)
        self._comandos_extra: List[ComandoActuador] = []
        self._perclos_confirmado_flag = False

    # ------------------------------------------------------------------
    # API PUBLICA
    # ------------------------------------------------------------------

    def procesar_evento(self, evento: EventoEntradaFSM) -> SalidaFSM:
        """
        Procesa un evento y devuelve la salida de la FSM.

        Esta es la unica API publica para hacer avanzar la FSM.
        El llamador debe registrar la salida (logs, despachador, etc.)
        antes de pasar el siguiente evento.
        """
        if evento.timestamp < self.estado.ultimo_timestamp:
            # Evento desordenado (clock skew o IPC desordenado). Lo
            # procesamos igual pero loggeamos. No queremos romper la FSM.
            self.log.warning(
                "Evento con timestamp %.3f anterior al ultimo procesado %.3f",
                evento.timestamp,
                self.estado.ultimo_timestamp,
            )

        if self.estado.ts_primer_evento == 0.0:
            self.estado.ts_primer_evento = evento.timestamp

        estado_anterior = self.estado.estado_actual

        # Despachar segun tipo de evento
        motivo = ""
        if isinstance(evento, EventoFalloSensor):
            motivo = self._procesar_fallo_sensor(evento)
        elif isinstance(evento, EventoRecuperacionSensor):
            motivo = self._procesar_recuperacion_sensor(evento)
        elif isinstance(evento, EventoAckWearable):
            motivo = self._procesar_ack(evento)
        elif isinstance(evento, EventoProcesado):
            motivo = self._procesar_evento_normal(evento)
        else:
            self.log.error("Tipo de evento desconocido: %s", type(evento).__name__)
            motivo = "evento_desconocido"

        # Actualizar timestamp
        self.estado.ultimo_timestamp = evento.timestamp

        # Construir salida
        transicion = self.estado.estado_actual != estado_anterior
        if transicion:
            self.log.info(
                "Transicion %s -> %s (motivo: %s)",
                estado_anterior.name,
                self.estado.estado_actual.name,
                motivo,
            )
            self.estado.timestamp_entrada_estado = evento.timestamp
            self.estado.tiempo_acumulado_sin_eventos = 0.0
            self.estado.ultimo_timestamp_evaluacion = evento.timestamp

        # emitir puede ser True sin transicion, para re-desafiar en CRITICO
        # tras un ACK incorrecto y no dejar la alarma pegada sin salida.
        emitir = transicion or self._reemitir
        comandos = self._generar_comandos(evento, emitir)
        if emitir and self._comandos_extra:
            comandos = list(comandos) + self._comandos_extra
        self._comandos_extra = []
        self._reemitir = False

        return SalidaFSM(
            timestamp=evento.timestamp,
            estado_actual=self.estado.estado_actual,
            estado_anterior=estado_anterior,
            nivel_alerta=self._nivel_alerta(self.estado.estado_actual),
            comandos=tuple(comandos),
            transicion_ocurrio=transicion,
            motivo_transicion=motivo if transicion else "",
            modo_degradado=(self.estado.estado_actual == EstadoFSM.MODO_DEGRADADO),
            motivo_degradacion=self._descripcion_degradacion(),
        )

    def get_estado_actual(self) -> EstadoFSM:
        """Acceso de solo lectura al estado actual."""
        return self.estado.estado_actual

    def get_estado_interno(self) -> EstadoInternoFSM:
        """Acceso al estado interno completo (para persistencia o debug)."""
        return self.estado

    # ------------------------------------------------------------------
    # MANEJADORES POR TIPO DE EVENTO
    # ------------------------------------------------------------------

    def _procesar_fallo_sensor(self, ev: EventoFalloSensor) -> str:
        """Severidad >= 2 fuerza el paso a MODO_DEGRADADO."""
        self.estado.sensores_caidos.add(int(ev.sensor_afectado))
        if ev.severidad >= 2 and self.estado.estado_actual != EstadoFSM.MODO_DEGRADADO:
            self.estado.estado_previo_a_degradado = self.estado.estado_actual
            self.estado.estado_actual = EstadoFSM.MODO_DEGRADADO
            return f"fallo_sensor:{ev.sensor_afectado.name}:{ev.motivo}"
        return ""

    def _procesar_recuperacion_sensor(self, ev: EventoRecuperacionSensor) -> str:
        """Si todos los sensores se recuperaron, salimos de MODO_DEGRADADO."""
        self.estado.sensores_caidos.discard(int(ev.sensor_recuperado))
        if (
            self.estado.estado_actual == EstadoFSM.MODO_DEGRADADO
            and not self.estado.sensores_caidos
        ):
            self.estado.estado_actual = EstadoFSM.NORMAL
            return f"recuperacion_sensor:{ev.sensor_recuperado.name}"
        return ""

    def _procesar_ack(self, ev: EventoAckWearable) -> str:
        """ACK correcto -> bajar a S1. ACK incorrecto -> escalar."""
        # Verificar que sea el ACK que esperabamos
        if self.estado.id_secuencia_ack_pendiente is None:
            return ""  # No habia ACK pendiente, lo ignoramos

        if ev.id_secuencia != self.estado.id_secuencia_ack_pendiente:
            self.log.warning(
                "ACK con id_secuencia=%d, esperado %d (ignorado)",
                ev.id_secuencia,
                self.estado.id_secuencia_ack_pendiente,
            )
            return ""

        # ACK valido recibido
        self.estado.id_secuencia_ack_pendiente = None

        if not ev.secuencia_correcta:
            # ACK incorrecto: el conductor esta confundido.
            if self.estado.estado_actual == EstadoFSM.CRITICO:
                # No hay nivel mas alto: seguir en CRITICO y RE-DESAFIAR,
                # para no dejar la alarma pegada sin forma de confirmar.
                self._solicitar_ack(ev.timestamp)
                self._reemitir = True
                return "ack_incorrecto_recritico"
            # Escalar un nivel y pedir un desafio nuevo en el estado destino
            motivo = self._escalar_un_nivel("ack_incorrecto")
            self._solicitar_ack(ev.timestamp)
            return motivo

        # ACK correcto
        if self.estado.estado_actual in (
            EstadoFSM.ALERTA_LEVE,
            EstadoFSM.ALERTA_MEDIA,
            EstadoFSM.CRITICO,
        ):
            if ev.tiempo_respuesta_ms > self.config.umbral_respuesta_lenta_ms:
                # Correcto pero LENTO: signo de deterioro. Baja solo un nivel.
                self._bajar_un_nivel_por_ack()
                motivo = f"ack_correcto_lento:{ev.id_secuencia}"
            else:
                self.estado.estado_actual = EstadoFSM.PRE_ALERTA
                motivo = f"ack_correcto:{ev.id_secuencia}"
            if self._fatiga_recurrente(ev.timestamp):
                # Responde pero se sigue durmiendo: responsividad != aptitud
                self._comandos_extra = [
                    ComandoActuador(
                        tipo=TipoComandoActuador.REPRODUCIR_VOZ,
                        mensaje_voz="Fatiga recurrente detectada, detente a descansar",
                    ),
                    ComandoActuador(tipo=TipoComandoActuador.NOTIFICAR_SUPERVISOR),
                ]
                motivo += "_fatiga_recurrente"
            return motivo
        return ""

    def _procesar_evento_normal(self, ev: EventoProcesado) -> str:
        """Procesa un EventoProcesado del Pre-FSM segun el estado actual."""
        # Modo degradado: salimos solo via recuperacion explicita
        if self.estado.estado_actual == EstadoFSM.MODO_DEGRADADO:
            return ""

        # Histeresis de bajada: actualizar acumulador de tiempo sin eventos
        self._actualizar_acumulador_tiempo(ev)
        # PERCLOS sostenido (parpados pesados sin eventos discretos)
        self._actualizar_perclos_confirmado(ev)

        # Despachar por estado
        if self.estado.estado_actual == EstadoFSM.NORMAL:
            return self._desde_normal(ev)
        elif self.estado.estado_actual == EstadoFSM.PRE_ALERTA:
            return self._desde_pre_alerta(ev)
        elif self.estado.estado_actual == EstadoFSM.ALERTA_LEVE:
            return self._desde_alerta_leve(ev)
        elif self.estado.estado_actual == EstadoFSM.ALERTA_MEDIA:
            return self._desde_alerta_media(ev)
        elif self.estado.estado_actual == EstadoFSM.CRITICO:
            return self._desde_critico(ev)
        return ""

    # ------------------------------------------------------------------
    # TRANSICIONES POR ESTADO
    # ------------------------------------------------------------------

    def _desde_normal(self, ev: EventoProcesado) -> str:
        """S0 -> S2 por evento severo corroborado; S0 -> S1 por cabeceo sin
        corroborar o por señales leves sostenidas."""
        if ev.ventana_no_confiable:
            return ""
        corroborado = ev.perclos is not None and ev.perclos >= 0.3
        # Via rapida: microsueno siempre (ojos cerrados es inequivoco);
        # cabeceo solo con corroboracion ocular (PERCLOS elevado), para no
        # confundir miradas al tablero con cabeceos de sueño.
        if ev.microsueno or (ev.cabeceo and corroborado):
            recurrente = self._fatiga_recurrente(ev.timestamp)
            self._registrar_episodio(ev.timestamp)
            if recurrente:
                self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
                self._solicitar_ack(ev.timestamp)
                return "severo_fatiga_recurrente"
            self.estado.estado_actual = EstadoFSM.ALERTA_LEVE
            self._solicitar_ack(ev.timestamp)
            return "evento_severo_desde_normal"
        # Cabeceo sin corroboracion: señal discreta -> PRE_ALERTA inmediato
        if ev.cabeceo:
            self.estado.estado_actual = EstadoFSM.PRE_ALERTA
            self.estado.ts_inicio_senales = 0.0
            return "cabeceo_sin_corroboracion"
        # Señales continuas: requieren persistencia
        if self._senales_continuas(ev):
            if self.estado.ts_inicio_senales == 0.0:
                self.estado.ts_inicio_senales = ev.timestamp
            elif (
                ev.timestamp - self.estado.ts_inicio_senales
                >= self.config.persistencia_senales_leves_seg
            ):
                self.estado.estado_actual = EstadoFSM.PRE_ALERTA
                self.estado.ts_inicio_senales = 0.0
                return "senales_leves_sostenidas"
        else:
            self.estado.ts_inicio_senales = 0.0
        return ""

    def _desde_pre_alerta(self, ev: EventoProcesado) -> str:
        """S1 -> S0 por tiempo sin eventos; S1 -> S2/S3 por evento confirmado."""
        # Evento confirmado escala; con fatiga recurrente, piso mas alto
        if self._hay_evento_confirmado(ev):
            recurrente = self._fatiga_recurrente(ev.timestamp)
            self._registrar_episodio(ev.timestamp)
            if recurrente:
                self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
                self._solicitar_ack(ev.timestamp)
                return "confirmado_fatiga_recurrente"
            self.estado.estado_actual = EstadoFSM.ALERTA_LEVE
            self._solicitar_ack(ev.timestamp)
            return "evento_confirmado"

        # Tiempo sin eventos negativos -> volver a NORMAL
        if (
            self.estado.tiempo_acumulado_sin_eventos
            >= self.config.tiempo_para_bajar_estado_seg
        ):
            self.estado.estado_actual = EstadoFSM.NORMAL
            return "tiempo_sin_eventos_completo"
        return ""

    def _desde_alerta_leve(self, ev: EventoProcesado) -> str:
        """S2 -> S3 por timeout ACK o por escalada por acumulacion."""
        # Evento severo estando en alerta escala directo
        if ev.microsueno or self._hay_evento_confirmado(ev):
            self._registrar_episodio(ev.timestamp)
            self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
            self._solicitar_ack(ev.timestamp)
            return "evento_severo_en_alerta"

        # Timeout del ACK
        if self._timeout_ack(ev.timestamp, self.config.timeout_ack_leve_seg):
            self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
            self._solicitar_ack(ev.timestamp)
            return "timeout_ack_leve"
        return ""

    def _desde_alerta_media(self, ev: EventoProcesado) -> str:
        """S3 -> S4 por timeout ACK + BPM critico, o cabeceo confirmado."""
        # Cabeceo + ojos cerrados + BPM critico = somnolencia confirmada
        if (
            ev.cabeceo
            and ev.nivel_riesgo_bpm == NivelRiesgoBPM.CRITICO
        ):
            self.estado.estado_actual = EstadoFSM.CRITICO
            self._solicitar_ack(ev.timestamp)
            return "cabeceo_confirmado_bpm_critico"

        # Timeout del ACK: el conductor no responde el desafio. La falta de
        # respuesta ya es señal de somnolencia, asi que escala a CRITICO sin
        # importar el BPM (en la muñeca suele ser normal o desconocido).
        if self._timeout_ack(ev.timestamp, self.config.timeout_ack_medio_seg):
            self.estado.estado_actual = EstadoFSM.CRITICO
            self._solicitar_ack(ev.timestamp)
            return "timeout_ack_medio_sin_respuesta"
        return ""

    def _desde_critico(self, ev: EventoProcesado) -> str:
        """S4 -> S1 solo por ACK correcto (manejado en _procesar_ack)."""
        # En CRITICO solo el ACK baja el estado. Si el desafio expiro, lo
        # RE-EMITIMOS (nuevo id + reenvio a la pulsera) para mantener el id
        # sincronizado. Cambiar el id sin reenviar dejaba la alarma pegada:
        # la respuesta del conductor nunca matcheaba el id nuevo.
        if self._timeout_ack(ev.timestamp, self.config.timeout_ack_critico_seg):
            self._solicitar_ack(ev.timestamp)
            self._reemitir = True
        return ""

    # ------------------------------------------------------------------
    # PREDICADOS DE EVENTOS
    # ------------------------------------------------------------------

    def _senales_continuas(self, ev: EventoProcesado) -> bool:
        """Señales leves CONTINUAS (metricas de ventana). Requieren
        persistencia antes de declarar PRE_ALERTA. Los parpadeos/min se
        ignoran durante el calentamiento (ventana incompleta al arrancar)."""
        if ev.ventana_no_confiable:
            return False
        en_calentamiento = (
            ev.timestamp - self.estado.ts_primer_evento
            < self.config.calentamiento_senales_seg
        )
        if (
            not en_calentamiento
            and ev.parpadeos_por_minuto is not None
            and ev.parpadeos_por_minuto < 10
        ):
            return True
        if ev.nivel_riesgo_bpm == NivelRiesgoBPM.ALERTA:
            return True
        if ev.perclos is not None and ev.perclos > 0.3:
            return True
        return False

    def _actualizar_perclos_confirmado(self, ev: EventoProcesado) -> None:
        """Trackea PERCLOS sostenido >= umbral confirmado (somnolencia de
        parpados pesados que nunca cruza un evento discreto)."""
        if (
            ev.ventana_no_confiable
            or ev.perclos is None
            or ev.perclos < self.config.perclos_confirmado
        ):
            self.estado.ts_inicio_perclos_alto = 0.0
            self._perclos_confirmado_flag = False
            return
        if self.estado.ts_inicio_perclos_alto == 0.0:
            self.estado.ts_inicio_perclos_alto = ev.timestamp
        self._perclos_confirmado_flag = (
            ev.timestamp - self.estado.ts_inicio_perclos_alto
            >= self.config.perclos_confirmado_sostenido_seg
        )

    def _registrar_episodio(self, ts: float) -> None:
        self.estado.episodios_severos.append(ts)
        self._purgar_episodios(ts)

    def _purgar_episodios(self, ts: float) -> None:
        limite = ts - self.config.ventana_episodios_seg
        self.estado.episodios_severos = [
            t for t in self.estado.episodios_severos if t >= limite
        ]

    def _fatiga_recurrente(self, ts: float) -> bool:
        """True si YA hay N episodios severos en la ventana larga."""
        self._purgar_episodios(ts)
        return (
            len(self.estado.episodios_severos)
            >= self.config.max_eventos_severos_ventana
        )

    def _bajar_un_nivel_por_ack(self) -> None:
        e = self.estado.estado_actual
        if e == EstadoFSM.CRITICO:
            self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
        elif e == EstadoFSM.ALERTA_MEDIA:
            self.estado.estado_actual = EstadoFSM.ALERTA_LEVE
        elif e == EstadoFSM.ALERTA_LEVE:
            self.estado.estado_actual = EstadoFSM.PRE_ALERTA

    def _hay_evento_confirmado(self, ev: EventoProcesado) -> bool:
        """Para subir de S1 a S2: evento discreto severo."""
        if ev.ventana_no_confiable:
            return False
        if ev.microsueno:
            return True
        # Bostezo UNICO ya no confirma (fisiologia normal); solo la
        # acumulacion en ventana larga (abajo).
        if ev.cabeceo:
            return True
        # PERCLOS sostenido: somnolencia de parpados pesados
        if self._perclos_confirmado_flag:
            return True
        
        # Acumulacion de bostezos
        if (
            ev.bostezos_ventana_larga
            >= self.config.max_bostezos_ventana_corta
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # GESTION DE TIMERS Y ACK
    # ------------------------------------------------------------------

    def _actualizar_acumulador_tiempo(self, ev: EventoProcesado) -> None:
        """
        Suma tiempo desde el ultimo evento al acumulador 'sin eventos',
        PERO solo si la vision esta disponible (regla de pausa por
        rostro perdido) y no hay eventos negativos en este evento.

        Si llega un evento negativo, reinicia el acumulador.
        """
        if self.estado.ultimo_timestamp_evaluacion == 0.0:
            self.estado.ultimo_timestamp_evaluacion = ev.timestamp
            return

        delta = ev.timestamp - self.estado.ultimo_timestamp_evaluacion
        if delta < 0:
            delta = 0.0

        # Reset si hay evento negativo
        if (
            ev.microsueno or ev.bostezo or ev.cabeceo
            or ev.nivel_riesgo_bpm in (NivelRiesgoBPM.ALERTA, NivelRiesgoBPM.CRITICO)
        ):
            self.estado.tiempo_acumulado_sin_eventos = 0.0
            self.estado.ultimo_timestamp_evaluacion = ev.timestamp
            return

        # Pausa por sensores caidos o ventana no confiable
        if (
            not ev.vision_disponible
            or ev.ventana_no_confiable
        ):
            # No acumulamos pero actualizamos el timestamp de referencia
            self.estado.ultimo_timestamp_evaluacion = ev.timestamp
            return

        # Tiempo limpio: acumulamos
        self.estado.tiempo_acumulado_sin_eventos += delta
        self.estado.ultimo_timestamp_evaluacion = ev.timestamp

    def _solicitar_ack(self, timestamp: float) -> None:
        """Genera un nuevo id_secuencia y arranca el timer del ACK."""
        self.estado.contador_secuencia_ack += 1
        self.estado.id_secuencia_ack_pendiente = self.estado.contador_secuencia_ack
        self.estado.timestamp_ack_solicitado = timestamp

    def _timeout_ack(self, timestamp_actual: float, limite_seg: float) -> bool:
        """True si el ACK pendiente excedio su timeout."""
        if self.estado.id_secuencia_ack_pendiente is None:
            return False
        return (timestamp_actual - self.estado.timestamp_ack_solicitado) >= limite_seg

    def _escalar_un_nivel(self, motivo: str) -> str:
        """Sube un nivel desde donde este (ACK incorrecto)."""
        if self.estado.estado_actual == EstadoFSM.ALERTA_LEVE:
            self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
            return motivo
        if self.estado.estado_actual == EstadoFSM.ALERTA_MEDIA:
            self.estado.estado_actual = EstadoFSM.CRITICO
            return motivo
        return ""

    # ------------------------------------------------------------------
    # GENERACION DE COMANDOS
    # ------------------------------------------------------------------

    def _asegurar_ack_pendiente(self, timestamp: float) -> Optional[int]:
        """Garantiza un id de ACK pendiente al emitir un desafio. Si no hay
        ninguno, solicita uno nuevo. Devuelve el id vigente."""
        if self.estado.id_secuencia_ack_pendiente is None:
            self._solicitar_ack(timestamp)
        return self.estado.id_secuencia_ack_pendiente

    def _generar_comandos(
        self,
        evento: EventoEntradaFSM,
        emitir: bool,
    ) -> List[ComandoActuador]:
        """
        Genera los comandos a despachar segun el estado actual.

        Reglas:
          - Al entrar a S2, S3, S4: emite los comandos de alerta + voz.
          - Al entrar a S0, S1, S5: emite APAGAR_TODO para silenciar.
          - emitir=False: no genera comandos (evita spam de actuadores).
          - emitir puede ser True sin transicion (re-desafio en CRITICO).
        """
        if not emitir:
            return []

        estado = self.estado.estado_actual

        if estado == EstadoFSM.NORMAL:
            return [ComandoActuador(tipo=TipoComandoActuador.APAGAR_TODO)]

        if estado == EstadoFSM.PRE_ALERTA:
            # PRE_ALERTA no genera alerta perceptible (solo se observa)
            return [ComandoActuador(tipo=TipoComandoActuador.APAGAR_TODO)]

        if estado == EstadoFSM.ALERTA_LEVE:
            return [
                ComandoActuador(
                    tipo=TipoComandoActuador.VIBRAR_LEVE,
                    intensidad=30,
                    duracion_ms=1500,
                ),
                # Yo edite esta parte se la puede sacar.
                ComandoActuador(
                                    tipo=TipoComandoActuador.BUZZER_CORTO,
                                    intensidad=50,
                                    duracion_ms=200,
                                ),
                #============================================
                ComandoActuador(
                    tipo=TipoComandoActuador.SECUENCIA_ACK,
                    intensidad=50,
                    id_secuencia=self._asegurar_ack_pendiente(evento.timestamp),
                ),
                ComandoActuador(
                    tipo=TipoComandoActuador.REPRODUCIR_VOZ,
                    mensaje_voz="Atencion, signos de fatiga detectados",
                ),
            ]

        if estado == EstadoFSM.ALERTA_MEDIA:
            return [
                ComandoActuador(
                    tipo=TipoComandoActuador.VIBRAR_FUERTE,
                    intensidad=80,
                    duracion_ms=2500,
                ),
                ComandoActuador(
                    tipo=TipoComandoActuador.BUZZER_LARGO,
                    intensidad=70,
                    duracion_ms=2000,
                ),
                ComandoActuador(
                    tipo=TipoComandoActuador.SECUENCIA_ACK,
                    intensidad=80,
                    id_secuencia=self._asegurar_ack_pendiente(evento.timestamp),
                ),
                ComandoActuador(
                    tipo=TipoComandoActuador.REPRODUCIR_VOZ,
                    mensaje_voz="Alerta, por favor confirme en la pulsera",
                ),
            ]

        if estado == EstadoFSM.CRITICO:
            return [
                ComandoActuador(
                    tipo=TipoComandoActuador.VIBRAR_FUERTE,
                    intensidad=100,
                    duracion_ms=0,  # continuo
                ),
                ComandoActuador(
                    tipo=TipoComandoActuador.BUZZER_CONTINUO,
                    intensidad=100,
                ),
                ComandoActuador(
                    tipo=TipoComandoActuador.SECUENCIA_ACK,
                    intensidad=100,
                    id_secuencia=self._asegurar_ack_pendiente(evento.timestamp),
                ),
                ComandoActuador(
                    tipo=TipoComandoActuador.REPRODUCIR_VOZ,
                    mensaje_voz="Detente ahora, peligro de somnolencia",
                ),
                ComandoActuador(
                    tipo=TipoComandoActuador.NOTIFICAR_SUPERVISOR,
                ),
            ]

        if estado == EstadoFSM.MODO_DEGRADADO:
            return [
                ComandoActuador(
                    tipo=TipoComandoActuador.APAGAR_TODO,
                ),
            ]

        return []

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    @staticmethod
    def _nivel_alerta(estado: EstadoFSM) -> int:
        """Mapea el estado al nivel de alerta numerico 0-4."""
        if estado == EstadoFSM.NORMAL:
            return 0
        if estado == EstadoFSM.PRE_ALERTA:
            return 1
        if estado == EstadoFSM.ALERTA_LEVE:
            return 2
        if estado == EstadoFSM.ALERTA_MEDIA:
            return 3
        if estado == EstadoFSM.CRITICO:
            return 4
        if estado == EstadoFSM.MODO_DEGRADADO:
            return 0  # Sin alerta perceptible, solo log
        return 0

    def _descripcion_degradacion(self) -> str:
        """Cadena descriptiva del modo degradado para el log."""
        if not self.estado.sensores_caidos:
            return ""
        nombres = [OrigenEvento(s).name for s in sorted(self.estado.sensores_caidos)]
        return f"sensores_caidos:{','.join(nombres)}"


__all__ = ["FSM", "ConfigFSM", "EstadoInternoFSM", "EventoEntradaFSM"]
