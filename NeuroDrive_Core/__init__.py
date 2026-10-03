"""
Paquete NeuroDrive_Core: nucleo de coordinacion y decision.

Modulos, en el orden en que los recorre un dato:
  - config_loader: carga y valida config/config.yaml
  - adaptador_mq: colas de mensajes POSIX entre procesos
  - gestor_eventos: lee las colas, valida y vigila la salud de las fuentes
  - pre_fsm: convierte metricas crudas en hechos (parpadeo, microsueno, ...)
  - fsm: maquina de estados; decide el nivel de alerta (logica pura)
  - despachador: encamina las ordenes de la FSM a los actuadores
  - actuadores/buzzer: aviso sonoro por GPIO
  - orquestador: une las piezas en el bucle principal
  - persistencia_sesion: conserva el estado entre sesiones
El actuador y el receptor de la pulsera estan en NeuroDrive_Wearable.
"""

from NeuroDrive_Core.fsm import (
    FSM,
    ConfigFSM,
    EstadoInternoFSM,
    EventoEntradaFSM,
)
from NeuroDrive_Core.config_loader import (
    Config,
    ConfigError,
    cargar_config,
    limpiar_cache,
)
from NeuroDrive_Core.adaptador_mq import (
    AdaptadorMQ,
    ErrorAdaptadorMQ,
    ErrorPermisos,
    ErrorTamanoMensaje,
    ErrorLimitesSistema,
    ErrorNoDisponible,
    eliminar_cola,
)
from NeuroDrive_Core.gestor_eventos import (
    GestorEventos,
    EstadisticasGestor,
    SaludSensor,
)
from NeuroDrive_Core.pre_fsm import (
    PreFSM,
    DetectorParpadeos,
    DetectorMicrosuenos,
    DetectorBostezos,
    DetectorCabeceos,
    VentanaPERCLOS,
    ClasificadorBPM,
    DetectorRostroPerdido,
)

__all__ = [
    # FSM
    "FSM",
    "ConfigFSM",
    "EstadoInternoFSM",
    "EventoEntradaFSM",
    # Config
    "Config",
    "ConfigError",
    "cargar_config",
    "limpiar_cache",
    # AdaptadorMQ
    "AdaptadorMQ",
    "ErrorAdaptadorMQ",
    "ErrorPermisos",
    "ErrorTamanoMensaje",
    "ErrorLimitesSistema",
    "ErrorNoDisponible",
    "eliminar_cola",
    # GestorEventos
    "GestorEventos",
    "EstadisticasGestor",
    "SaludSensor",
    # PreFSM
    "PreFSM",
    "DetectorParpadeos",
    "DetectorMicrosuenos",
    "DetectorBostezos",
    "DetectorCabeceos",
    "VentanaPERCLOS",
    "ClasificadorBPM",
    "DetectorRostroPerdido",
]
