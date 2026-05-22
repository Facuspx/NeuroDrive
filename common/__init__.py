"""
Paquete `common` — vocabulario compartido del proyecto NeuroDrive.
Exporta todos los contratos de datos para que cualquier módulo del proyecto
pueda importarlos con:
    from common.contratos import EventoVision, EstadoFSM, Envelope, ...
o usando el atajo:
    from common import EventoVision, EstadoFSM, Envelope, ...
"""

from common.contratos import (
    # Enums
    EstadoFSM,
    NivelRiesgoBPM,
    TipoComandoActuador,
    OrigenEvento,
    TipoMensaje,
    # Eventos primarios
    EventoVision,
    EventoWearable,
    # Eventos de confirmación y salud
    EventoAckWearable,
    EventoFalloSensor,
    EventoRecuperacionSensor,
    # Eventos procesados
    EventoProcesado,
    # Salida FSM
    ComandoActuador,
    SalidaFSM,
    # Persistencia
    EstadoSesion,
    # Envelope
    Envelope,
    EventoPayload,
    # Utilidades
    timestamp_actual,
    generar_id_sesion,
    generar_id_mensaje,
)

__all__ = [
    "EstadoFSM",
    "NivelRiesgoBPM",
    "TipoComandoActuador",
    "OrigenEvento",
    "TipoMensaje",
    "EventoVision",
    "EventoWearable",
    "EventoAckWearable",
    "EventoFalloSensor",
    "EventoRecuperacionSensor",
    "EventoProcesado",
    "ComandoActuador",
    "SalidaFSM",
    "EstadoSesion",
    "Envelope",
    "EventoPayload",
    "timestamp_actual",
    "generar_id_sesion",
    "generar_id_mensaje",
]
