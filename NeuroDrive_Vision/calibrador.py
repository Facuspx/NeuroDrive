"""
NeuroDrive Vision - Calibrador
==============================

Modo de calibracion previo a la operacion normal. Durante 60 segundos
recolecta metricas del conductor (EAR, MAR, pose de cabeza) con el
conductor en estado normal/alerta, y calcula valores base personalizados.

Esos valores base se le pasan luego a los analizadores via sus metodos
actualizar_umbrales(), para que la deteccion de somnolencia se adapte
a la fisonomia particular de cada conductor.

Decisiones tomadas (ver chat de planificacion):
  - Duracion: 60 segundos (configurable).
  - Filtrado robusto por percentiles (P25-P75): descarta automaticamente
    parpadeos y bostezos sin necesidad de detectarlos explicitamente.
  - Si la calibracion falla (poco rostro, valores anomalos), exito=False
    y el sistema usa los defaults de los analizadores.
  - Persistencia en JSON con timestamp.
  - El calibrador acumula durante procesar(); el calculo se hace en
    finalizar().

Flujo de uso:
    calib = Calibrador(config)
    # Opcion 1: cargar calibracion previa
    resultado = ResultadoCalibracion.cargar("calibracion.json")
    if resultado is None or not resultado.exito:
        # Opcion 2: calibrar de cero
        calib.iniciar()
        while not calib.terminado:
            frame, _ = captura.leer()
            datos_rostro = detector.procesar(frame)
            calib.procesar(datos_rostro)
        resultado = calib.finalizar()
        if resultado.exito:
            resultado.guardar("calibracion.json")
    # Aplicar a los analizadores
    if resultado.exito:
        analizador_ojos.actualizar_umbrales(resultado.ear_base)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from NeuroDrive_Vision.detector_rostro import DatosRostro
from NeuroDrive_Vision.analizador_ojos import AnalizadorOjos, OJO_IZQ_INDICES, OJO_DER_INDICES
from NeuroDrive_Vision.analizador_boca import AnalizadorBoca
from NeuroDrive_Vision.analizador_cabeza import AnalizadorCabeza

try:
    from NeuroDrive_Core.config_loader import Config
except ImportError:
    Config = None  # type: ignore


_log = logging.getLogger("NeuroDrive.Calibrador")


# =============================================================================
# Estructura de resultado
# =============================================================================

@dataclass
class ResultadoCalibracion:
    """
    Resultado de una calibracion.

    Si exito=False, los valores base NO deben usarse; el sistema debe
    caer a los defaults de los analizadores.
    """
    exito: bool

    timestamp: float = 0.0          # time.time() de cuando se calibro

    # Valores calibrados
    ear_base: float = 0.0
    mar_base: float = 0.0
    pitch_neutro: float = 0.0
    yaw_neutro: float = 0.0
    roll_neutro: float = 0.0

    # Diagnostico
    muestras_totales: int = 0       # frames procesados
    muestras_validas: int = 0       # frames con rostro y metricas validas
    tasa_deteccion: float = 0.0     # validas / totales
    duracion_real_seg: float = 0.0
    motivo_fallo: str = ""

    def __repr__(self) -> str:
        if not self.exito:
            return f"ResultadoCalibracion(FALLO: {self.motivo_fallo})"
        return (
            f"ResultadoCalibracion(ear_base={self.ear_base:.3f}, "
            f"mar_base={self.mar_base:.3f}, pitch_neutro={self.pitch_neutro:+.1f}, "
            f"validas={self.muestras_validas}/{self.muestras_totales})"
        )

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def guardar(self, ruta: str) -> None:
        """Guarda el resultado en un archivo JSON."""
        try:
            with open(ruta, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
            _log.info("Calibracion guardada en %s", ruta)
        except OSError as e:
            _log.error("No se pudo guardar la calibracion en %s: %s", ruta, e)
            raise

    @classmethod
    def cargar(cls, ruta: str) -> Optional["ResultadoCalibracion"]:
        """
        Carga un resultado de calibracion desde JSON.

        Returns:
            ResultadoCalibracion si el archivo existe y es valido.
            None si no existe o esta corrupto (el caller debe recalibrar).
        """
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            _log.info("No existe calibracion previa en %s", ruta)
            return None
        except (OSError, json.JSONDecodeError) as e:
            _log.warning("Calibracion en %s corrupta o ilegible: %s", ruta, e)
            return None

        # Validar que sea un dict (no una lista u otra cosa)
        if not isinstance(data, dict):
            _log.warning("Calibracion en %s no es un objeto JSON valido", ruta)
            return None

        # Validar que tenga los campos esperados
        try:
            resultado = cls(**data)
        except TypeError as e:
            _log.warning("Calibracion en %s tiene formato invalido: %s", ruta, e)
            return None

        return resultado

    def antiguedad_dias(self) -> float:
        """Devuelve cuantos dias hace que se hizo esta calibracion."""
        if self.timestamp <= 0:
            return float("inf")
        return (time.time() - self.timestamp) / 86400.0


class ErrorCalibrador(Exception):
    """Error tecnico en el calibrador."""


# =============================================================================
# Clase principal
# =============================================================================

class Calibrador:
    """
    Calibrador de NeuroDrive Vision.

    Acumula metricas durante una ventana de tiempo y calcula valores base
    personalizados con filtrado robusto.

    Lifecycle:
      - __init__: configura parametros.
      - iniciar(): arranca la ventana de calibracion.
      - procesar(datos_rostro): acumula una muestra. Llamar por cada frame.
      - terminado (property): True cuando se cumplio la duracion.
      - finalizar(): calcula y devuelve el ResultadoCalibracion.
    """

    # Rangos razonables para validar los valores calibrados.
    EAR_BASE_MIN = 0.15
    EAR_BASE_MAX = 0.45
    MAR_BASE_MIN = 0.02
    MAR_BASE_MAX = 0.40

    # Tasa minima de deteccion de rostro para considerar la calibracion valida
    TASA_DETECCION_MINIMA = 0.50

    # Cantidad minima de muestras validas
    MUESTRAS_VALIDAS_MINIMAS = 50

    # Percentiles para el filtrado robusto.
    #
    # EAR base = ojos BIEN ABIERTOS = valores ALTOS de la distribucion.
    # Los parpadeos tiran el EAR hacia abajo (nunca hacia arriba), asi que
    # nos quedamos con la mitad alta. El P90 (no P100) descarta glitches de
    # MediaPipe que inflan artificialmente el EAR.
    PERCENTIL_EAR_INF = 50
    PERCENTIL_EAR_SUP = 90

    # MAR base = boca RELAJADA/CERRADA = valores BAJOS de la distribucion.
    # Hablar o bostezar suben el MAR. Es el caso opuesto al EAR: nos
    # quedamos con la parte baja. El P10 descarta glitches hacia abajo.
    PERCENTIL_MAR_INF = 10
    PERCENTIL_MAR_SUP = 50

    def __init__(
        self,
        config: Optional["Config"] = None,
        duracion_seg: float = 60.0,
    ) -> None:
        """
        Parametros
        ----------
        config : Config | None
            Configuracion global (reservada).
        duracion_seg : float
            Duracion de la ventana de calibracion. Default 60 s.
        """
        self.config = config

        if duracion_seg <= 0:
            raise ValueError("duracion_seg debe ser > 0")
        self.duracion_seg = float(duracion_seg)

        # ---------- Estado ----------
        self._activo: bool = False
        self._ts_inicio: Optional[float] = None
        self._ts_ultima_muestra: Optional[float] = None

        # Acumuladores de muestras
        self._muestras_ear: list[float] = []
        self._muestras_mar: list[float] = []
        self._muestras_pitch: list[float] = []
        self._muestras_yaw: list[float] = []
        self._muestras_roll: list[float] = []

        self._frames_totales: int = 0

        # Analizador de cabeza interno (sin filtro EMA para tener valores crudos)
        self._ana_cabeza = AnalizadorCabeza(alpha_ema=1.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def activo(self) -> bool:
        return self._activo

    @property
    def terminado(self) -> bool:
        """True si ya paso la duracion de calibracion."""
        if not self._activo or self._ts_inicio is None:
            return False
        return (time.monotonic() - self._ts_inicio) >= self.duracion_seg

    @property
    def progreso(self) -> float:
        """Fraccion 0.0-1.0 de la calibracion completada."""
        if not self._activo or self._ts_inicio is None:
            return 0.0
        transcurrido = time.monotonic() - self._ts_inicio
        return min(1.0, transcurrido / self.duracion_seg)

    @property
    def tiempo_restante_seg(self) -> float:
        """Segundos que faltan para terminar."""
        if not self._activo or self._ts_inicio is None:
            return self.duracion_seg
        transcurrido = time.monotonic() - self._ts_inicio
        return max(0.0, self.duracion_seg - transcurrido)

    def iniciar(self) -> None:
        """Arranca la ventana de calibracion."""
        self._activo = True
        self._ts_inicio = time.monotonic()
        self._ts_ultima_muestra = None
        self._muestras_ear.clear()
        self._muestras_mar.clear()
        self._muestras_pitch.clear()
        self._muestras_yaw.clear()
        self._muestras_roll.clear()
        self._frames_totales = 0
        self._ana_cabeza.reset()
        _log.info("Calibracion iniciada (duracion=%.0fs)", self.duracion_seg)

    # ------------------------------------------------------------------
    # Acumulacion de muestras
    # ------------------------------------------------------------------

    def procesar(self, datos_rostro: DatosRostro) -> None:
        """
        Acumula una muestra. Llamar una vez por frame durante la calibracion.

        Si el calibrador no esta activo o ya termino, no hace nada.
        """
        if not self._activo:
            return
        if self.terminado:
            return

        self._frames_totales += 1
        self._ts_ultima_muestra = time.monotonic()

        # Si no hay rostro, el frame no aporta muestras (pero cuenta como total)
        if not datos_rostro.rostro_presente or datos_rostro.puntos_pixeles is None:
            return

        # ----- EAR -----
        ear_izq = AnalizadorOjos._calcular_ear(datos_rostro.puntos_pixeles, OJO_IZQ_INDICES)
        ear_der = AnalizadorOjos._calcular_ear(datos_rostro.puntos_pixeles, OJO_DER_INDICES)
        if ear_izq > 0.0 and ear_der > 0.0:
            self._muestras_ear.append((ear_izq + ear_der) / 2.0)
        elif ear_izq > 0.0:
            self._muestras_ear.append(ear_izq)
        elif ear_der > 0.0:
            self._muestras_ear.append(ear_der)

        # ----- MAR -----
        mar = AnalizadorBoca._calcular_mar(datos_rostro.puntos_pixeles)
        if mar > 0.0:
            self._muestras_mar.append(mar)

        # ----- Pose de cabeza -----
        datos_cabeza = self._ana_cabeza.procesar(datos_rostro)
        if datos_cabeza.valido:
            self._muestras_pitch.append(datos_cabeza.pitch_crudo)
            self._muestras_yaw.append(datos_cabeza.yaw_crudo)
            self._muestras_roll.append(datos_cabeza.roll_crudo)

    # ------------------------------------------------------------------
    # Calculo robusto
    # ------------------------------------------------------------------

    @staticmethod
    def _promedio_robusto(
        muestras: list[float],
        percentil_inf: float,
        percentil_sup: float,
    ) -> Optional[float]:
        """
        Calcula el promedio de las muestras que caen dentro del rango de
        percentiles [percentil_inf, percentil_sup].

        El rango es ASIMETRICO segun la metrica:
          - EAR base: usar P50-P90. El EAR base son los ojos bien abiertos
            (valores altos). Los parpadeos tiran el EAR hacia abajo, asi que
            nos quedamos con la mitad alta. El P90 descarta glitches que
            inflan el EAR.
          - MAR base: usar P10-P50. El MAR base es la boca relajada (valores
            bajos). Hablar/bostezar suben el MAR, asi que nos quedamos con la
            parte baja.

        Returns:
            El promedio robusto, o None si no hay suficientes muestras.
        """
        if len(muestras) < 4:
            return None

        arr = np.array(muestras, dtype=np.float64)
        p_inf = np.percentile(arr, percentil_inf)
        p_sup = np.percentile(arr, percentil_sup)

        mascara = (arr >= p_inf) & (arr <= p_sup)
        seleccionadas = arr[mascara]

        if len(seleccionadas) == 0:
            # Caso degenerado: p_inf == p_sup excluyo todo. Fallback a mediana.
            return float(np.median(arr))

        return float(np.mean(seleccionadas))

    @staticmethod
    def _mediana(muestras: list[float]) -> Optional[float]:
        """Mediana simple. Para pose de cabeza (no necesita filtrado IQR)."""
        if not muestras:
            return None
        return float(np.median(np.array(muestras, dtype=np.float64)))

    # ------------------------------------------------------------------
    # Finalizacion
    # ------------------------------------------------------------------

    def finalizar(self) -> ResultadoCalibracion:
        """
        Calcula los valores base a partir de las muestras acumuladas.

        Returns
        -------
        ResultadoCalibracion
            Con exito=True si todo salio bien, exito=False si la
            calibracion no es confiable (con el motivo en motivo_fallo).
        """
        self._activo = False

        duracion_real = 0.0
        if self._ts_inicio is not None and self._ts_ultima_muestra is not None:
            duracion_real = self._ts_ultima_muestra - self._ts_inicio

        muestras_validas = len(self._muestras_ear)
        tasa = (
            muestras_validas / self._frames_totales
            if self._frames_totales > 0 else 0.0
        )

        resultado = ResultadoCalibracion(
            exito=False,
            timestamp=time.time(),
            muestras_totales=self._frames_totales,
            muestras_validas=muestras_validas,
            tasa_deteccion=tasa,
            duracion_real_seg=duracion_real,
        )

        # ----- Validacion 1: suficientes muestras -----
        if muestras_validas < self.MUESTRAS_VALIDAS_MINIMAS:
            resultado.motivo_fallo = (
                f"muy pocas muestras validas: {muestras_validas} "
                f"(minimo {self.MUESTRAS_VALIDAS_MINIMAS})"
            )
            _log.warning("Calibracion fallida: %s", resultado.motivo_fallo)
            return resultado

        # ----- Validacion 2: tasa de deteccion -----
        if tasa < self.TASA_DETECCION_MINIMA:
            resultado.motivo_fallo = (
                f"tasa de deteccion baja: {tasa:.2f} "
                f"(minimo {self.TASA_DETECCION_MINIMA})"
            )
            _log.warning("Calibracion fallida: %s", resultado.motivo_fallo)
            return resultado

        # ----- Calculo de EAR base -----
        # EAR base = ojos abiertos = valores altos -> P50-P90
        ear_base = self._promedio_robusto(
            self._muestras_ear,
            self.PERCENTIL_EAR_INF,
            self.PERCENTIL_EAR_SUP,
        )
        if ear_base is None:
            resultado.motivo_fallo = "no se pudo calcular EAR base"
            return resultado

        # ----- Validacion 3: EAR base en rango razonable -----
        if not (self.EAR_BASE_MIN <= ear_base <= self.EAR_BASE_MAX):
            resultado.motivo_fallo = (
                f"EAR base fuera de rango: {ear_base:.3f} "
                f"(esperado {self.EAR_BASE_MIN}-{self.EAR_BASE_MAX}). "
                "Posible problema de iluminacion o el conductor tenia "
                "los ojos entrecerrados."
            )
            _log.warning("Calibracion fallida: %s", resultado.motivo_fallo)
            resultado.ear_base = ear_base  # lo guardamos para diagnostico
            return resultado

        resultado.ear_base = ear_base

        # ----- Calculo de MAR base -----
        # MAR base = boca relajada = valores bajos -> P10-P50
        mar_base = self._promedio_robusto(
            self._muestras_mar,
            self.PERCENTIL_MAR_INF,
            self.PERCENTIL_MAR_SUP,
        )
        if mar_base is None:
            _log.warning("No se pudo calcular MAR base, usando 0.10 por defecto")
            mar_base = 0.10
        elif not (self.MAR_BASE_MIN <= mar_base <= self.MAR_BASE_MAX):
            _log.warning(
                "MAR base fuera de rango (%.3f), usando 0.10 por defecto",
                mar_base,
            )
            mar_base = 0.10
        resultado.mar_base = mar_base

        # ----- Calculo de pose neutra de cabeza -----
        pitch = self._mediana(self._muestras_pitch)
        yaw = self._mediana(self._muestras_yaw)
        roll = self._mediana(self._muestras_roll)
        resultado.pitch_neutro = pitch if pitch is not None else 0.0
        resultado.yaw_neutro = yaw if yaw is not None else 0.0
        resultado.roll_neutro = roll if roll is not None else 0.0

        # ----- Todo OK -----
        resultado.exito = True
        _log.info(
            "Calibracion exitosa: ear_base=%.3f, mar_base=%.3f, "
            "pitch_neutro=%.1f, validas=%d/%d",
            resultado.ear_base, resultado.mar_base, resultado.pitch_neutro,
            muestras_validas, self._frames_totales,
        )
        return resultado

    # ------------------------------------------------------------------
    # Aplicacion del resultado a los analizadores
    # ------------------------------------------------------------------

    @staticmethod
    def aplicar(
        resultado: ResultadoCalibracion,
        analizador_ojos: Optional[AnalizadorOjos] = None,
        analizador_boca: Optional[AnalizadorBoca] = None,
    ) -> bool:
        """
        Aplica un ResultadoCalibracion a los analizadores correspondientes.

        Solo aplica si resultado.exito es True. Si es False, no toca nada
        (los analizadores quedan con sus defaults).

        Returns:
            True si se aplico, False si el resultado no era valido.
        """
        if not resultado.exito:
            _log.info("Calibracion no exitosa, los analizadores usan defaults")
            return False

        if analizador_ojos is not None:
            analizador_ojos.actualizar_umbrales(ear_base=resultado.ear_base)
            _log.info("Calibracion aplicada a AnalizadorOjos (ear_base=%.3f)",
                      resultado.ear_base)

        # El AnalizadorBoca usa umbrales absolutos por defecto; la calibracion
        # del MAR es opcional (ver decision en chat 4.5). Se deja el parametro
        # para uso futuro.
        _ = analizador_boca

        return True
