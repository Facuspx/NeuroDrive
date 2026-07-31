"""
Subpaquete de actuadores de NeuroDrive Core.

Cada actuador implementa la interfaz ActuadorBase (definida en
NeuroDrive_Core.despachador) y es gobernado por el DespachadorComandos.

Actuadores:
  - buzzer: alarma sonora local por GPIO (buzzer activo, pin BCM 18).
  - (proximos) wearable, voz.
"""

from NeuroDrive_Core.actuadores.buzzer import (
    ActuadorBuzzer,
    BackendGPIO,
    BackendLgpio,
    BackendSimuladoGPIO,
    detectar_gpiochip,
)

__all__ = [
    "ActuadorBuzzer",
    "BackendGPIO",
    "BackendLgpio",
    "BackendSimuladoGPIO",
    "detectar_gpiochip",
]
