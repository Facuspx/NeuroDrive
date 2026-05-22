"""
NeuroDrive — Contratos de datos compartidos
============================================
Define todos los tipos de datos que viajan entre los módulos del sistema:

    Vision  ─────►  POSIX MQ  ─────►  Gestor  ─────►  Pre-FSM  ─────►  FSM  ─────►  Despachador
    Wearable ───►  POSIX MQ  ─────►  Gestor                                              │
                                                                                          ▼
                                                                            Actuadores (buzzer, voz, ESP32)

Convenciones:
    - Todos los dataclasses son `frozen=True` (inmutables, thread-safe).
    - Todos llevan un timestamp (segundos desde epoch).
    - Métodos `to_dict()` y `from_dict()` para serialización JSON.
    - Validación en __post_init__ para detectar valores fuera de rango.

Este módulo NO debe importar de ningún otro módulo del proyecto.
Es el "vocabulario común" del sistema.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union


# =============================================================================
#                                 ENUMS
# =============================================================================


class EstadoFSM(IntEnum):
    """
    Estados de la máquina finita de detección de somnolencia.
    Se usa IntEnum para permitir comparaciones jerárquicas:
        if estado >= EstadoFSM.ALERTA_MEDIA: ...
    Y para serialización directa a JSON (los IntEnum se serializan como int).
    """
    NORMAL          = 0   # Conductor alerto, sin alertas activas
    PRE_ALERTA      = 1   # Señales leves observadas, sin acción aún
    ALERTA_LEVE     = 2   # Vibración suave en pulsera, espera ACK
    ALERTA_MEDIA    = 3   # Vibración intensa + buzzer + secuencia táctil
    CRITICO         = 4   # Vibración máxima + alarma + notificación supervisor
    MODO_DEGRADADO  = 5   # Sensor caído, alerta pasiva, log del error


class NivelRiesgoBPM(IntEnum):
    """Clasificación del BPM según los umbrales del config."""
    DESCONOCIDO = 0   # Sin dato disponible (pulsera caída)
    NORMAL      = 1
    ALERTA      = 2   # BPM bajo, sospecha de fatiga
    CRITICO     = 3   # BPM muy bajo, alta probabilidad de somnolencia


class TipoComandoActuador(IntEnum):
    """Comandos que el Despachador envía a los actuadores."""
    NINGUNO            = 0
    VIBRAR_LEVE        = 1
    VIBRAR_MEDIO       = 2
    VIBRAR_FUERTE      = 3
    BUZZER_CORTO       = 4
    BUZZER_LARGO       = 5
    BUZZER_CONTINUO    = 6
    REPRODUCIR_VOZ     = 7
    SECUENCIA_ACK      = 8   # Pedir confirmación táctil en la pantalla
    NOTIFICAR_SUPERVISOR = 9
    APAGAR_TODO        = 10


class OrigenEvento(IntEnum):
    """De qué módulo proviene un evento (útil para logging y debug)."""
    VISION    = 0
    WEARABLE  = 1
    INTERNO   = 2   # Generado dentro del Core (ej: timeout de ACK)


class TipoMensaje(IntEnum):
    """
    Tipo de mensaje que viaja dentro de un Envelope.
    Permite al Gestor hacer un switch rápido sobre el tipo sin tener que
    parsear el payload. Cada tipo corresponde a un dataclass específico.
    """
    EVENTO_VISION         = 1   # Frame procesado por NeuroDrive_Vision
    EVENTO_WEARABLE       = 2   # Lectura de BPM del wearable
    EVENTO_ACK_WEARABLE   = 3   # Conductor confirmó secuencia táctil
    HEARTBEAT             = 4   # "Sigo vivo" sin dato nuevo
    FALLO_SENSOR          = 5   # Un sensor cayó (timeout, error, etc.)
    RECUPERACION_SENSOR   = 6   # Un sensor caído volvió a funcionar
    EVENTO_PROCESADO      = 7   # Salida del Pre-FSM (uso interno)


# =============================================================================
#                          EVENTOS PRIMARIOS (entrada)
# =============================================================================


@dataclass(frozen=True)
class EventoVision:
    """
    Evento crudo producido por NeuroDrive_Vision en cada frame procesado.
    Llega al Gestor por la POSIX MQ /neurodrive_vision.
    El Pre-FSM lo agrega a sus ventanas temporales para producir EventoProcesado.
    Campos opcionales (None) indican que ese dato no pudo medirse en este frame
    (ej: rostro perdido temporalmente, landmarks incompletos).
    """
    timestamp: float
    rostro_detectado: bool

    # Métricas oculares (None si no se pudo calcular)
    ear_izquierdo: Optional[float] = None
    ear_derecho: Optional[float] = None

    # Métrica bucal
    mar: Optional[float] = None

    # Ángulos de Euler de la cabeza en grados (None si no estimables)
    pitch_grados: Optional[float] = None
    yaw_grados: Optional[float] = None
    roll_grados: Optional[float] = None

    # Señalizadores de eventos a corto plazo
    frote_ojos_activo: bool = False

    # Confiabilidad de la detección (0.0 a 1.0)
    confianza_deteccion: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp <= 0:
            raise ValueError(f"timestamp inválido: {self.timestamp}")
        if not (0.0 <= self.confianza_deteccion <= 1.0):
            raise ValueError(
                f"confianza_deteccion fuera de rango [0,1]: {self.confianza_deteccion}"
            )
        # EAR típicamente entre 0.0 y 0.5; MAR entre 0.0 y 1.5
        for nombre, val in (
            ("ear_izquierdo", self.ear_izquierdo),
            ("ear_derecho", self.ear_derecho),
            ("mar", self.mar),
        ):
            if val is not None and val < 0:
                raise ValueError(f"{nombre} negativo: {val}")

    @property
    def ear_promedio(self) -> Optional[float]:
        """EAR promediado entre ambos ojos. None si falta alguno."""
        if self.ear_izquierdo is None or self.ear_derecho is None:
            return None
        return (self.ear_izquierdo + self.ear_derecho) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventoVision:
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> EventoVision:
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class EventoWearable:
    """
    Evento crudo producido por el ESP32-S3 vía UDP y reenviado al Gestor por MQ.
    El campo `ack_recibido` indica que el conductor respondió a una solicitud
    de confirmación táctil. `secuencia_replicada` indica si la replicó correctamente.
    """
    timestamp: float

    # BPM medido (None si la lectura del sensor falló este ciclo)
    bpm: Optional[int] = None

    # Estados de confirmación táctil
    ack_recibido: bool = False
    secuencia_replicada: bool = False

    # Telemetría del wearable
    bateria_porcentaje: Optional[int] = None   # 0-100

    # ID del paquete (para deduplicación de reenvíos en comandos críticos)
    id_paquete: Optional[int] = None

    def __post_init__(self) -> None:
        if self.timestamp <= 0:
            raise ValueError(f"timestamp inválido: {self.timestamp}")
        if self.bpm is not None and not (20 <= self.bpm <= 250):
            raise ValueError(f"BPM fuera de rango fisiológico [20,250]: {self.bpm}")
        if self.bateria_porcentaje is not None and not (0 <= self.bateria_porcentaje <= 100):
            raise ValueError(f"batería fuera de rango [0,100]: {self.bateria_porcentaje}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventoWearable:
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> EventoWearable:
        return cls.from_dict(json.loads(payload))


# =============================================================================
#                   EVENTOS DE CONFIRMACIÓN Y SALUD DEL SISTEMA
# =============================================================================


@dataclass(frozen=True)
class EventoAckWearable:
    """
    Respuesta del conductor a una solicitud de confirmación táctil.
    Cuando la FSM solicita "secuencia ACK" en estado ALERTA_MEDIA o CRÍTICO,
    el wearable muestra una secuencia de vibraciones que el conductor debe
    replicar tocando la pantalla en el orden correcto.
    Diferenciar entre:
      - secuencia_correcta=True  → el conductor está alerto → bajar nivel
      - secuencia_correcta=False → el conductor está confundido → subir nivel
      - timeout (no llega evento) → el conductor no respondió → subir a CRÍTICO
    """
    timestamp: float
    id_secuencia: int             # qué secuencia de ACK está respondiendo
    secuencia_correcta: bool      # True = replicó bien, False = mal
    tiempo_respuesta_ms: int      # cuánto tardó en responder (latencia)

    def __post_init__(self) -> None:
        if self.timestamp <= 0:
            raise ValueError(f"timestamp inválido: {self.timestamp}")
        if self.id_secuencia < 0:
            raise ValueError(f"id_secuencia negativo: {self.id_secuencia}")
        if self.tiempo_respuesta_ms < 0:
            raise ValueError(f"tiempo_respuesta_ms negativo: {self.tiempo_respuesta_ms}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventoAckWearable:
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> EventoAckWearable:
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class EventoFalloSensor:
    """
    Reporta que un sensor cayó o dejó de ser confiable.
    Generado por:
      - El Gestor cuando detecta heartbeat_timeout del wearable
      - NeuroDrive_Vision cuando pierde la cámara o el rostro por mucho tiempo
      - Cualquier módulo que detecte un fallo crítico interno
    La FSM consume estos eventos para decidir si va a MODO_DEGRADADO.
    """
    timestamp: float
    sensor_afectado: OrigenEvento   # VISION o WEARABLE
    motivo: str                      # descripción técnica del fallo
    severidad: int                   # 1=warning, 2=error, 3=critical

    def __post_init__(self) -> None:
        if self.timestamp <= 0:
            raise ValueError(f"timestamp inválido: {self.timestamp}")
        if not (1 <= self.severidad <= 3):
            raise ValueError(f"severidad fuera de [1,3]: {self.severidad}")
        if self.sensor_afectado == OrigenEvento.INTERNO:
            raise ValueError("INTERNO no es un sensor válido para fallo")
        if not self.motivo:
            raise ValueError("motivo no puede estar vacío")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sensor_afectado"] = int(self.sensor_afectado)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventoFalloSensor:
        d = dict(data)
        d["sensor_afectado"] = OrigenEvento(d["sensor_afectado"])
        return cls(**d)

    @classmethod
    def from_json(cls, payload: str) -> EventoFalloSensor:
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class EventoRecuperacionSensor:
    """
    Reporta que un sensor previamente caído volvió a funcionar.
    Es el complemento de EventoFalloSensor. Sin esto, la FSM nunca podría
    salir de MODO_DEGRADADO automáticamente.
    Solo se debe emitir si previamente se había emitido un EventoFalloSensor
    para el mismo sensor.
    """
    timestamp: float
    sensor_recuperado: OrigenEvento
    tiempo_caido_seg: float          # cuánto tiempo estuvo caído

    def __post_init__(self) -> None:
        if self.timestamp <= 0:
            raise ValueError(f"timestamp inválido: {self.timestamp}")
        if self.tiempo_caido_seg < 0:
            raise ValueError(f"tiempo_caido_seg negativo: {self.tiempo_caido_seg}")
        if self.sensor_recuperado == OrigenEvento.INTERNO:
            raise ValueError("INTERNO no es un sensor válido para recuperación")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sensor_recuperado"] = int(self.sensor_recuperado)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventoRecuperacionSensor:
        d = dict(data)
        d["sensor_recuperado"] = OrigenEvento(d["sensor_recuperado"])
        return cls(**d)

    @classmethod
    def from_json(cls, payload: str) -> EventoRecuperacionSensor:
        return cls.from_dict(json.loads(payload))


# =============================================================================
#                       EVENTOS PROCESADOS (Pre-FSM → FSM)
# =============================================================================


@dataclass(frozen=True)
class EventoProcesado:
    """
    Resultado del Pre-FSM: eventos discretos detectados sobre ventanas temporales.
    La FSM consume estos eventos para decidir transiciones. NO consume los
    eventos crudos de Vision/Wearable directamente.
    Los flags booleanos son señales discretas ("ocurrió un microsueño en este ciclo").
    Las métricas continuas (parpadeos_por_minuto, bpm_actual) son valores actuales.
    """
    timestamp: float

    # ---- Señales discretas detectadas en este ciclo ----
    microsueno: bool = False
    bostezo: bool = False
    cabeceo: bool = False
    parpadeo: bool = False        # Parpadeo normal (informativo)

    # ---- Métricas continuas calculadas sobre ventanas ----
    parpadeos_por_minuto: Optional[float] = None
    perclos: Optional[float] = None              # % tiempo con ojos cerrados
    bostezos_ventana_larga: int = 0              # cuántos en los últimos 15 min

    # ---- Estado fisiológico ----
    bpm_actual: Optional[int] = None
    nivel_riesgo_bpm: NivelRiesgoBPM = NivelRiesgoBPM.DESCONOCIDO

    # ---- Calidad de la ventana ----
    # True si en este ciclo hay frote de ojos o rostro perdido (excluir del análisis)
    ventana_no_confiable: bool = False
    motivo_no_confiable: str = ""

    # ---- Disponibilidad de sensores ----
    vision_disponible: bool = True
    wearable_disponible: bool = True

    def __post_init__(self) -> None:
        if self.timestamp <= 0:
            raise ValueError(f"timestamp inválido: {self.timestamp}")
        if self.perclos is not None and not (0.0 <= self.perclos <= 1.0):
            raise ValueError(f"PERCLOS fuera de rango [0,1]: {self.perclos}")
        if self.bostezos_ventana_larga < 0:
            raise ValueError(f"bostezos_ventana_larga negativo: {self.bostezos_ventana_larga}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # IntEnum → int para JSON
        d["nivel_riesgo_bpm"] = int(self.nivel_riesgo_bpm)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventoProcesado:
        d = dict(data)
        if "nivel_riesgo_bpm" in d:
            d["nivel_riesgo_bpm"] = NivelRiesgoBPM(d["nivel_riesgo_bpm"])
        return cls(**d)

    @classmethod
    def from_json(cls, payload: str) -> EventoProcesado:
        return cls.from_dict(json.loads(payload))


# =============================================================================
#                        SALIDA DE LA FSM (FSM → Despachador)
# =============================================================================


@dataclass(frozen=True)
class ComandoActuador:
    """
    Un comando individual generado por el Despachador hacia un actuador concreto.
    Múltiples comandos pueden estar activos simultáneamente (ej: vibrar + voz).
    """
    tipo: TipoComandoActuador
    intensidad: int = 0           # 0-100 (para vibración y volumen)
    duracion_ms: int = 0          # 0 = continuo hasta nuevo comando
    mensaje_voz: str = ""         # Solo usado si tipo == REPRODUCIR_VOZ

    def __post_init__(self) -> None:
        if not (0 <= self.intensidad <= 100):
            raise ValueError(f"intensidad fuera de rango [0,100]: {self.intensidad}")
        if self.duracion_ms < 0:
            raise ValueError(f"duración negativa: {self.duracion_ms}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tipo"] = int(self.tipo)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComandoActuador:
        d = dict(data)
        if "tipo" in d:
            d["tipo"] = TipoComandoActuador(d["tipo"])
        return cls(**d)


@dataclass(frozen=True)
class SalidaFSM:
    """
    Salida estructurada de la FSM en cada ciclo.
    Contiene:
      - El estado actual y el nivel de alerta numérico
      - Los comandos concretos a ejecutar
      - Información de trazabilidad (motivo, evento que disparó la transición)
    """
    timestamp: float
    estado_actual: EstadoFSM
    estado_anterior: EstadoFSM
    nivel_alerta: int                     # 0=normal, 4=crítico

    # Acciones a ejecutar
    comandos: tuple = field(default_factory=tuple)   # tupla de ComandoActuador

    # Trazabilidad
    transicion_ocurrio: bool = False
    motivo_transicion: str = ""

    # Robustez
    modo_degradado: bool = False
    motivo_degradacion: str = ""

    def __post_init__(self) -> None:
        if self.timestamp <= 0:
            raise ValueError(f"timestamp inválido: {self.timestamp}")
        if not (0 <= self.nivel_alerta <= 4):
            raise ValueError(f"nivel_alerta fuera de rango [0,4]: {self.nivel_alerta}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "estado_actual": int(self.estado_actual),
            "estado_anterior": int(self.estado_anterior),
            "nivel_alerta": self.nivel_alerta,
            "comandos": [c.to_dict() for c in self.comandos],
            "transicion_ocurrio": self.transicion_ocurrio,
            "motivo_transicion": self.motivo_transicion,
            "modo_degradado": self.modo_degradado,
            "motivo_degradacion": self.motivo_degradacion,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SalidaFSM:
        d = dict(data)
        d["estado_actual"] = EstadoFSM(d["estado_actual"])
        d["estado_anterior"] = EstadoFSM(d["estado_anterior"])
        d["comandos"] = tuple(ComandoActuador.from_dict(c) for c in d.get("comandos", []))
        return cls(**d)


# =============================================================================
#                       ESTADO DE SESIÓN (persistencia en disco)
# =============================================================================


@dataclass(frozen=True)
class EstadoSesion:
    """
    Snapshot del estado del sistema que se persiste en disco entre sesiones.
    Permite que si el conductor apaga la Pi y vuelve a encenderla en menos
    de N minutos (configurable), el sistema retome la sesión anterior.
    """
    timestamp_guardado: float
    estado_fsm: EstadoFSM

    # Contadores de ventana larga
    bostezos_recientes: tuple = field(default_factory=tuple)   # timestamps
    microsuenos_recientes: tuple = field(default_factory=tuple)
    cabeceos_recientes: tuple = field(default_factory=tuple)

    # Para debug/trazabilidad
    motivo_guardado: str = ""

    def __post_init__(self) -> None:
        if self.timestamp_guardado <= 0:
            raise ValueError(f"timestamp_guardado inválido: {self.timestamp_guardado}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_guardado": self.timestamp_guardado,
            "estado_fsm": int(self.estado_fsm),
            "bostezos_recientes": list(self.bostezos_recientes),
            "microsuenos_recientes": list(self.microsuenos_recientes),
            "cabeceos_recientes": list(self.cabeceos_recientes),
            "motivo_guardado": self.motivo_guardado,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EstadoSesion:
        d = dict(data)
        d["estado_fsm"] = EstadoFSM(d["estado_fsm"])
        for clave in ("bostezos_recientes", "microsuenos_recientes", "cabeceos_recientes"):
            if clave in d:
                d[clave] = tuple(d[clave])
        return cls(**d)

    @classmethod
    def from_json(cls, payload: str) -> EstadoSesion:
        return cls.from_dict(json.loads(payload))


# =============================================================================
#              ENVELOPE — SOBRE COMÚN PARA TODO MENSAJE DEL SISTEMA
# =============================================================================


# Tipo unión para el campo `evento` del Envelope (uso interno)
# Es el conjunto de objetos Python que pueden viajar como payload
EventoPayload = Union[
    "EventoVision",
    "EventoWearable",
    "EventoAckWearable",
    "EventoFalloSensor",
    "EventoRecuperacionSensor",
    "EventoProcesado",
]


@dataclass(frozen=True)
class Envelope:
    """
    Sobre genérico que envuelve cualquier mensaje del sistema NeuroDrive.
    Tiene dos modos de uso:
    --- MODO EXTERNO (POSIX MQ o UDP) ---
    El mensaje viaja serializado entre procesos. Se llena `payload_json` con
    la representación JSON del evento, y `evento` queda en None.
        env = Envelope(
            tipo=TipoMensaje.EVENTO_VISION,
            origen=OrigenEvento.VISION,
            id_dispositivo="cam-01",
            id_sesion="ses-20250512-001",
            id_mensaje="vis-00042",
            numero_secuencia=42,
            timestamp_origen=time.time(),
            payload_json=evento_vision.to_json(),
        )
        mq.send(env.to_json())

    --- MODO INTERNO (queue.Queue) ---
    El mensaje circula entre hilos dentro del Core. No hace falta serializar.
    Se llena `evento` con el objeto Python, y `payload_json` queda vacío.

        env = Envelope(
            tipo=TipoMensaje.EVENTO_VISION,
            ...
            evento=evento_vision,
        )
        cola_interna.put(env)

    El Gestor convierte de modo externo a modo interno al recibir un mensaje:
    parsea el JSON, construye el objeto, y lo pone en `evento` para que el
    Pre-FSM lo use directamente sin re-parsear.
    """
    tipo: TipoMensaje
    origen: OrigenEvento
    id_dispositivo: str            # "cam-01", "wearable-01", "core"
    id_sesion: str                 # "ses-YYYYMMDD-HHMMSS"
    id_mensaje: str                # "vis-00042" — único por origen+sesión
    numero_secuencia: int          # contador incremental por origen
    timestamp_origen: float        # cuándo se generó en el origen

    # Modo externo: payload serializado para transporte
    payload_json: str = ""

    # Modo interno: objeto Python ya deserializado
    evento: Optional[object] = None

    # Lo completa el Gestor al recibir (0.0 = no recibido aún)
    timestamp_recepcion: float = 0.0

    # Versión del protocolo del Envelope (para compatibilidad futura)
    version: int = 1

    def __post_init__(self) -> None:
        if self.timestamp_origen <= 0:
            raise ValueError(f"timestamp_origen inválido: {self.timestamp_origen}")
        if self.timestamp_recepcion < 0:
            raise ValueError(
                f"timestamp_recepcion negativo: {self.timestamp_recepcion}"
            )
        if self.numero_secuencia < 0:
            raise ValueError(f"numero_secuencia negativo: {self.numero_secuencia}")
        if not self.id_dispositivo:
            raise ValueError("id_dispositivo no puede estar vacío")
        if not self.id_sesion:
            raise ValueError("id_sesion no puede estar vacío")
        if not self.id_mensaje:
            raise ValueError("id_mensaje no puede estar vacío")
        if self.version < 1:
            raise ValueError(f"version inválida: {self.version}")
        # Al menos uno de los dos modos debe estar lleno
        if not self.payload_json and self.evento is None:
            raise ValueError(
                "Envelope vacío: debe tener payload_json (externo) o evento (interno)"
            )

    @property
    def latencia_ms(self) -> Optional[float]:
        """Latencia de transporte en milisegundos (None si no fue recibido aún)."""
        if self.timestamp_recepcion <= 0:
            return None
        return (self.timestamp_recepcion - self.timestamp_origen) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el sobre. Si `evento` tiene un objeto, lo serializa al
        `payload_json` automáticamente para no perder información."""
        # Si está en modo interno, serializar el evento a JSON antes de exportar
        payload = self.payload_json
        if not payload and self.evento is not None:
            # Todos los eventos del proyecto tienen to_json()
            if hasattr(self.evento, "to_json"):
                payload = self.evento.to_json()  # type: ignore[attr-defined]

        return {
            "tipo": int(self.tipo),
            "origen": int(self.origen),
            "id_dispositivo": self.id_dispositivo,
            "id_sesion": self.id_sesion,
            "id_mensaje": self.id_mensaje,
            "numero_secuencia": self.numero_secuencia,
            "timestamp_origen": self.timestamp_origen,
            "timestamp_recepcion": self.timestamp_recepcion,
            "payload_json": payload,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Envelope:
        d = dict(data)
        d["tipo"] = TipoMensaje(d["tipo"])
        d["origen"] = OrigenEvento(d["origen"])
        # `evento` no viaja en JSON, se reconstruye en el Gestor con desempacar()
        d.setdefault("payload_json", "")
        d.setdefault("evento", None)
        return cls(**d)

    @classmethod
    def from_json(cls, payload: str) -> Envelope:
        return cls.from_dict(json.loads(payload))

    def desempacar(self) -> EventoPayload:
        """
        Convierte el `payload_json` en el objeto Python correspondiente según `tipo`.
        Es la operación clave del Gestor al recibir un mensaje externo:
        toma el JSON crudo y lo convierte en un dataclass tipado.
        Si ya está en modo interno (evento != None), lo devuelve directo.
        Raises:
            ValueError: si no hay payload_json ni evento, o si el tipo no se reconoce.
        """
        if self.evento is not None:
            return self.evento  # type: ignore[return-value]

        if not self.payload_json:
            raise ValueError("Envelope sin payload_json ni evento, no se puede desempacar")

        # Mapeo de tipo → clase concreta
        if self.tipo == TipoMensaje.EVENTO_VISION:
            return EventoVision.from_json(self.payload_json)
        elif self.tipo == TipoMensaje.EVENTO_WEARABLE:
            return EventoWearable.from_json(self.payload_json)
        elif self.tipo == TipoMensaje.EVENTO_ACK_WEARABLE:
            return EventoAckWearable.from_json(self.payload_json)
        elif self.tipo == TipoMensaje.FALLO_SENSOR:
            return EventoFalloSensor.from_json(self.payload_json)
        elif self.tipo == TipoMensaje.RECUPERACION_SENSOR:
            return EventoRecuperacionSensor.from_json(self.payload_json)
        elif self.tipo == TipoMensaje.EVENTO_PROCESADO:
            return EventoProcesado.from_json(self.payload_json)
        elif self.tipo == TipoMensaje.HEARTBEAT:
            # Heartbeat puede no llevar payload; devolver EventoWearable mínimo
            return EventoWearable.from_json(self.payload_json)
        else:
            raise ValueError(f"Tipo de mensaje desconocido: {self.tipo}")


# =============================================================================
#                              UTILIDADES
# =============================================================================


def timestamp_actual() -> float:
    """Devuelve el timestamp actual en segundos (alias estándar del proyecto)."""
    return time.time()


def generar_id_sesion(prefijo: str = "ses") -> str:
    """
    Genera un identificador único de sesión basado en fecha y hora.
    Formato: 'ses-YYYYMMDD-HHMMSS' (ej: 'ses-20250512-143052').
    Se debe llamar una sola vez por arranque del sistema y usar el mismo
    valor durante toda la sesión.
    """
    if not prefijo:
        raise ValueError("prefijo no puede estar vacío")
    return f"{prefijo}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def generar_id_mensaje(prefijo: str, secuencia: int) -> str:
    """
    Genera un identificador de mensaje a partir de un prefijo y un número.
    Formato: 'PREFIJO-NNNNN' con secuencia en 5 dígitos (con padding de ceros).
    Ejemplos:
        generar_id_mensaje("vis", 42)   → "vis-00042"
        generar_id_mensaje("wea", 128)  → "wea-00128"
    Si secuencia supera 99999, se usa el ancho que haga falta:
        generar_id_mensaje("vis", 123456) → "vis-123456"
    """
    if not prefijo:
        raise ValueError("prefijo no puede estar vacío")
    if secuencia < 0:
        raise ValueError(f"secuencia negativa: {secuencia}")
    return f"{prefijo}-{secuencia:05d}"


# =============================================================================
#                         EXPORTS PÚBLICOS DEL MÓDULO
# =============================================================================

__all__ = [
    # Enums
    "EstadoFSM",
    "NivelRiesgoBPM",
    "TipoComandoActuador",
    "OrigenEvento",
    "TipoMensaje",
    # Eventos primarios
    "EventoVision",
    "EventoWearable",
    # Eventos de confirmación y salud
    "EventoAckWearable",
    "EventoFalloSensor",
    "EventoRecuperacionSensor",
    # Eventos procesados
    "EventoProcesado",
    # Salida FSM
    "ComandoActuador",
    "SalidaFSM",
    # Persistencia
    "EstadoSesion",
    # Envelope
    "Envelope",
    "EventoPayload",
    # Utilidades
    "timestamp_actual",
    "generar_id_sesion",
    "generar_id_mensaje",
]
