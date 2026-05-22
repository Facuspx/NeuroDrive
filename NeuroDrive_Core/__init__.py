"""
Paquete NeuroDrive_Core: nucleo de coordinacion y decision.

Modulos:
  - fsm: maquina de estados finita (Etapa 2)
  - config_loader: cargador del config.yaml (Etapa 3.1)
  - adaptador_mq: wrapper sobre POSIX Message Queues (Etapa 3.2)
  - gestor_eventos: puente entre procesos externos y el Core (Etapa 3.3)
  - pre_fsm: evaluador de eventos primarios (Etapa 3.4)
  - despachador: emisor de comandos a actuadores (Etapa 5, pendiente)
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
