"""
NeuroDrive Core - Actuador Buzzer (GPIO)
=========================================

Actuador local de alarma sonora. Maneja un BUZZER ACTIVO conectado a un pin
GPIO de la Raspberry Pi 5 (por defecto BCM 18). Un buzzer activo suena con
solo aplicarle tension (pin en HIGH) y calla en LOW; no necesita PWM ni
generacion de tono.

Arquitectura del modulo (costura de hardware para poder testear sin Pi):

    ActuadorBuzzer  ->  BackendGPIO (abstracto)
                          |-- BackendLgpio         (real, usa lgpio en la Pi)
                          |-- BackendSimuladoGPIO   (tests, registra HIGH/LOW)

El ActuadorBuzzer contiene TODA la logica (timing no bloqueante, reemplazo
de beep, apagado) y NO sabe de lgpio. Eso lo pone el backend. Asi la misma
logica se valida en sandbox con el backend simulado y corre en la Pi con el
backend real cambiando una linea.

Decisiones:
  - NO bloqueante: ejecutar() pone HIGH y programa un threading.Timer para
    volver a LOW. BUZZER_CONTINUO (duracion 0) queda sonando hasta apagar().
  - Reemplazo seguro: contador de generacion evita que el timer de un beep
    viejo apague un beep nuevo.
  - Buzzer activo => sin control de volumen. `intensidad` se ignora
    (limitacion de hardware documentada para el TFI).
  - APAGAR_TODO NO esta en tipos_soportados: el despachador ya hace broadcast
    de apagar() a todos los actuadores.

Sobre el gpiochip en la Pi 5:
  Segun la version de kernel, los pines del header pueden estar en gpiochip0
  (kernels actuales, Trixie) o gpiochip4 (kernels viejos de Pi 5). El backend
  real auto-detecta el chip con etiqueta 'pinctrl-rp1' y cae a 0 si no puede.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

from common.contratos import ComandoActuador, TipoComandoActuador
from NeuroDrive_Core.despachador import ActuadorBase


# Duraciones por defecto (ms) cuando el comando no especifica duracion_ms.
# BUZZER_CONTINUO ignora esto: siempre es continuo.
_DUR_CORTO_MS_DEFAULT = 150
_DUR_LARGO_MS_DEFAULT = 800


# =============================================================================
#                        BACKEND GPIO (abstracto)
# =============================================================================


class BackendGPIO(ABC):
    """Abstraccion minima de un pin de salida digital."""

    @abstractmethod
    def abrir(self, pin: int) -> None:
        """Reserva el pin como salida y lo deja en LOW."""

    @abstractmethod
    def escribir(self, pin: int, nivel: int) -> None:
        """Escribe 0 (LOW) o 1 (HIGH) en el pin."""

    @abstractmethod
    def cerrar(self) -> None:
        """Libera el pin y cierra el chip."""


# =============================================================================
#                        BACKEND LGPIO (real, en la Pi)
# =============================================================================


def detectar_gpiochip(candidatos: Tuple[int, ...] = (0, 4)) -> int:
    """
    Detecta el numero de gpiochip cuyo controlador es 'pinctrl-rp1' (el chip
    del header de 40 pines en la Pi 5). Parsea la salida de `gpiodetect`.
    Si no puede detectar, devuelve el primer candidato (0 por defecto,
    correcto en kernels actuales/Trixie).

    Formato esperado de gpiodetect:
        gpiochip0 [pinctrl-rp1] (54 lines)
    """
    try:
        res = subprocess.run(
            ["gpiodetect"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        for linea in res.stdout.splitlines():
            if "pinctrl-rp1" in linea:
                m = re.match(r"gpiochip(\d+)", linea.strip())
                if m:
                    return int(m.group(1))
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return candidatos[0] if candidatos else 0


class BackendLgpio(BackendGPIO):
    """
    Backend real basado en la libreria `lgpio` (la recomendada para Pi 5).
    Importa lgpio de forma perezosa dentro de abrir(), asi el modulo se puede
    importar en maquinas sin lgpio (sandbox, CI).
    """

    def __init__(
        self,
        gpiochip: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._gpiochip_pedido = gpiochip   # None => auto-detectar en abrir()
        self._gpiochip_usado: Optional[int] = None
        self._handle: Optional[int] = None
        self._lgpio = None
        self.log = logger or logging.getLogger("NeuroDrive.BackendLgpio")

    def abrir(self, pin: int) -> None:
        import lgpio  # import perezoso: solo existe en la Pi
        self._lgpio = lgpio

        chip = (
            self._gpiochip_pedido
            if self._gpiochip_pedido is not None
            else detectar_gpiochip()
        )
        self._gpiochip_usado = chip
        self.log.info("Abriendo gpiochip%d para el pin BCM %d", chip, pin)

        self._handle = lgpio.gpiochip_open(chip)
        lgpio.gpio_claim_output(self._handle, pin, 0)  # salida, inicia en LOW

    def escribir(self, pin: int, nivel: int) -> None:
        if self._handle is None or self._lgpio is None:
            raise RuntimeError("BackendLgpio.escribir() sin abrir()")
        self._lgpio.gpio_write(self._handle, pin, 1 if nivel else 0)

    def cerrar(self) -> None:
        if self._handle is not None and self._lgpio is not None:
            try:
                self._lgpio.gpiochip_close(self._handle)
            finally:
                self._handle = None


# =============================================================================
#                     BACKEND SIMULADO (para tests)
# =============================================================================


class BackendSimuladoGPIO(BackendGPIO):
    """Backend de prueba: registra cada escritura (timestamp, pin, nivel)."""

    def __init__(self) -> None:
        self.escrituras: List[Tuple[float, int, int]] = []
        self.abierto: bool = False
        self.cerrado: bool = False
        self._lock = threading.Lock()

    def abrir(self, pin: int) -> None:
        self.abierto = True

    def escribir(self, pin: int, nivel: int) -> None:
        with self._lock:
            self.escrituras.append((time.monotonic(), pin, 1 if nivel else 0))

    def cerrar(self) -> None:
        self.cerrado = True
        self.abierto = False

    def ultimo_nivel(self) -> Optional[int]:
        with self._lock:
            return self.escrituras[-1][2] if self.escrituras else None

    def niveles(self) -> List[int]:
        with self._lock:
            return [e[2] for e in self.escrituras]


# =============================================================================
#                        ACTUADOR BUZZER
# =============================================================================


class ActuadorBuzzer(ActuadorBase):
    """
    Actuador de buzzer activo por GPIO. Ver docstring del modulo.

    Parametros:
      pin: numero de pin en BCM (default 18, segun config.yaml).
      backend: implementacion de BackendGPIO. Si es None, usa BackendLgpio
               (real). En tests se pasa un BackendSimuladoGPIO.
      dur_corto_ms / dur_largo_ms: duraciones por defecto cuando el comando
               no trae duracion_ms.
    """

    nombre = "buzzer"

    def __init__(
        self,
        pin: int = 18,
        backend: Optional[BackendGPIO] = None,
        dur_corto_ms: int = _DUR_CORTO_MS_DEFAULT,
        dur_largo_ms: int = _DUR_LARGO_MS_DEFAULT,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not (0 <= pin <= 40):
            raise ValueError(f"pin BCM fuera de rango [0,40]: {pin}")
        self._pin = pin
        self._backend = backend if backend is not None else BackendLgpio(logger=logger)
        self._dur_corto_ms = dur_corto_ms
        self._dur_largo_ms = dur_largo_ms
        self.log = logger or logging.getLogger("NeuroDrive.Buzzer")

        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._generacion = 0
        self._abierto = False
        self._sonando = False

    # ------------------------------------------------------------------
    # Interfaz ActuadorBase
    # ------------------------------------------------------------------

    def tipos_soportados(self) -> Set[TipoComandoActuador]:
        return {
            TipoComandoActuador.BUZZER_CORTO,
            TipoComandoActuador.BUZZER_LARGO,
            TipoComandoActuador.BUZZER_CONTINUO,
        }

    def iniciar(self) -> None:
        if self._abierto:
            return
        self._backend.abrir(self._pin)
        self._backend.escribir(self._pin, 0)  # arrancar callado
        self._abierto = True
        self.log.info("Buzzer listo en pin BCM %d", self._pin)

    def ejecutar(self, comando: ComandoActuador) -> None:
        if not self._abierto:
            raise RuntimeError("ActuadorBuzzer.ejecutar() sin iniciar()")

        tipo = comando.tipo
        if tipo == TipoComandoActuador.BUZZER_CONTINUO:
            self._sonar(0)  # 0 = continuo
        elif tipo == TipoComandoActuador.BUZZER_LARGO:
            dur = comando.duracion_ms if comando.duracion_ms > 0 else self._dur_largo_ms
            self._sonar(dur)
        elif tipo == TipoComandoActuador.BUZZER_CORTO:
            dur = comando.duracion_ms if comando.duracion_ms > 0 else self._dur_corto_ms
            self._sonar(dur)
        else:
            self.log.warning("Buzzer recibio tipo no soportado: %s", tipo.name)

    def apagar(self) -> None:
        with self._lock:
            self._cancelar_timer_locked()
            self._generacion += 1  # invalida cualquier timer en vuelo
            if self._abierto:
                try:
                    self._backend.escribir(self._pin, 0)
                except Exception as e:
                    self.log.error("Fallo al apagar el buzzer: %s", e)
            self._sonando = False

    def detener(self) -> None:
        self.apagar()
        with self._lock:
            if self._abierto:
                try:
                    self._backend.cerrar()
                except Exception as e:
                    self.log.error("Fallo al cerrar el backend del buzzer: %s", e)
                self._abierto = False

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    def _sonar(self, duracion_ms: int) -> None:
        """Pone el pin en HIGH; si duracion_ms > 0 programa el apagado."""
        with self._lock:
            self._cancelar_timer_locked()
            self._generacion += 1
            gen = self._generacion
            try:
                self._backend.escribir(self._pin, 1)
            except Exception as e:
                self.log.error("Fallo al encender el buzzer: %s", e)
                return
            self._sonando = True

            if duracion_ms > 0:
                self._timer = threading.Timer(
                    duracion_ms / 1000.0,
                    self._apagar_por_timer,
                    args=(gen,),
                )
                self._timer.daemon = True
                self._timer.start()

    def _apagar_por_timer(self, gen: int) -> None:
        """Callback del timer: apaga solo si sigue siendo la generacion vigente."""
        with self._lock:
            if gen != self._generacion:
                return  # un comando mas nuevo tomo el control; no tocar
            try:
                self._backend.escribir(self._pin, 0)
            except Exception as e:
                self.log.error("Fallo al apagar el buzzer (timer): %s", e)
            self._sonando = False
            self._timer = None

    def _cancelar_timer_locked(self) -> None:
        """Cancela el timer en curso. Debe llamarse con el lock tomado."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    # ------------------------------------------------------------------
    # Inspeccion (para tests / debug)
    # ------------------------------------------------------------------

    @property
    def sonando(self) -> bool:
        return self._sonando
