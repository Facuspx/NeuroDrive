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

        comandos = self._generar_comandos(evento, transicion)

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
            # ACK incorrecto: escalar un nivel
            return self._escalar_un_nivel("ack_incorrecto")

        # ACK correcto: bajar a PRE_ALERTA (nunca directo a NORMAL)
        if self.estado.estado_actual in (
            EstadoFSM.ALERTA_LEVE,
            EstadoFSM.ALERTA_MEDIA,
            EstadoFSM.CRITICO,
        ):
            self.estado.estado_actual = EstadoFSM.PRE_ALERTA
            return f"ack_correcto:{ev.id_secuencia}"
        return ""

    def _procesar_evento_normal(self, ev: EventoProcesado) -> str:
        """Procesa un EventoProcesado del Pre-FSM segun el estado actual."""
        # Modo degradado: salimos solo via recuperacion explicita
        if self.estado.estado_actual == EstadoFSM.MODO_DEGRADADO:
            return ""

        # Histeresis de bajada: actualizar acumulador de tiempo sin eventos
        self._actualizar_acumulador_tiempo(ev)

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
        """S0 -> S1 si hay señales leves sostenidas."""
        if self._hay_señales_leves(ev):
            self.estado.estado_actual = EstadoFSM.PRE_ALERTA
            return "señales_leves_detectadas"
        return ""

    def _desde_pre_alerta(self, ev: EventoProcesado) -> str:
        """S1 -> S0 por tiempo sin eventos; S1 -> S2 por evento confirmado."""
        # Evento confirmado escala a ALERTA_LEVE
        if self._hay_evento_confirmado(ev):
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
        # Microsueño o segundo bostezo escala directo (regla 3)
        if ev.microsueno or self._hay_evento_confirmado(ev):
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

        # Timeout del ACK con BPM no normal
        if self._timeout_ack(ev.timestamp, self.config.timeout_ack_medio_seg):
            if ev.nivel_riesgo_bpm in (NivelRiesgoBPM.ALERTA, NivelRiesgoBPM.CRITICO):
                self.estado.estado_actual = EstadoFSM.CRITICO
                self._solicitar_ack(ev.timestamp)
                return "timeout_ack_medio_bpm_bajo"
            # Sin BPM bajo, igual escala pero mas lento (refrescamos solicitud)
            self._solicitar_ack(ev.timestamp)
            return ""
        return ""

    def _desde_critico(self, ev: EventoProcesado) -> str:
        """S4 -> S1 solo por ACK correcto (manejado en _procesar_ack)."""
        # En CRITICO solo el ACK puede bajar el estado.
        # Refrescamos la solicitud de ACK si pasaron mas del timeout
        if self._timeout_ack(ev.timestamp, self.config.timeout_ack_critico_seg):
            self._solicitar_ack(ev.timestamp)
        return ""

    # ------------------------------------------------------------------
    # PREDICADOS DE EVENTOS
    # ------------------------------------------------------------------

    def _hay_señales_leves(self, ev: EventoProcesado) -> bool:
        """Para subir de S0 a S1: señales sutiles de fatiga."""
        if ev.ventana_no_confiable:
            return False
        # Parpadeos bajos
        if (
            ev.parpadeos_por_minuto is not None
            and ev.parpadeos_por_minuto < 10
        ):
            return True
        # BPM levemente bajo
        if ev.nivel_riesgo_bpm == NivelRiesgoBPM.ALERTA:
            return True
        # PERCLOS alto (mas de 30% del tiempo con ojos cerrados)
        if ev.perclos is not None and ev.perclos > 0.3:
            return True
        return False

    def _hay_evento_confirmado(self, ev: EventoProcesado) -> bool:
        """Para subir de S1 a S2: evento discreto severo."""
        if ev.ventana_no_confiable:
            return False
        if ev.microsueno:
            return True
        if ev.bostezo:
            return True
        if ev.cabeceo:
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

    def _generar_comandos(
        self,
        evento: EventoEntradaFSM,
        transicion: bool,
    ) -> List[ComandoActuador]:
        """
        Genera los comandos a despachar segun el estado actual.

        Reglas:
          - En transicion a S2, S3, S4: emite los comandos de alerta + voz.
          - En transicion a S0, S1, S5: emite APAGAR_TODO para silenciar.
          - Sin transicion: no genera comandos (evita spam de actuadores).
        """
        if not transicion:
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
