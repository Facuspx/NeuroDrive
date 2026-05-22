"""
NeuroDrive Vision - Analizador de boca
======================================

Calcula el Mouth Aspect Ratio (MAR) y detecta bostezos.

Decisiones tomadas (ver chat de planificacion):
  - Formula MAR clasica con 3 pares verticales / 1 base horizontal.
  - 8 puntos de MediaPipe FaceMesh (mismos del codigo de referencia).
  - Histeresis con doble umbral (Schmitt trigger) para evitar oscilaciones.
  - Umbrales absolutos por default (MAR no depende tanto del individuo
    como el EAR). Pueden calibrarse via actualizar_umbrales().
  - Bostezo = MAR alto sostenido por > 2 segundos.
  - Si dura > 10 seg, se descarta (no es bostezo, es otra cosa).
  - Ventana de bostezos/min = 5 min (son mucho menos frecuentes que parpadeos).
  - Timeout de perdida de rostro = 2 seg (mismo que ojos).

NOTA: El evento de bostezo se emite SOLO en el frame donde la boca se cierra
(al terminar el bostezo), no mientras esta abierta. Esto permite obtener
la duracion total del evento.

API:
    analizador = AnalizadorBoca(config)
    datos = analizador.procesar(datos_rostro)
    if datos.valido and datos.evento_bostezo:
        print(f"Bostezo de {datos.duracion_bostezo_ms:.0f}ms")
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from NeuroDrive_Vision.detector_rostro import DatosRostro

try:
    from NeuroDrive_Core.config_loader import Config
except ImportError:
    Config = None  # type: ignore


_log = logging.getLogger("NeuroDrive.AnalizadorBoca")


# =============================================================================
# Indices de landmarks de MediaPipe FaceMesh para la boca
# =============================================================================
#
# Layout de los 8 puntos:
#
#                81    13   311           (labio superior)
#                 ^     ^    ^
#         78 <---|-----|----|--> 308     (comisuras = base horizontal)
#                 v     v    v
#               178    14   402           (labio inferior)
#
# MAR = (|81-178| + |13-14| + |311-402|) / (2 * |78-308|)

BOCA_INDICES = {
    "comisura_izq":   78,
    "comisura_der":   308,
    "sup_izq":        81,
    "sup_centro":     13,
    "sup_der":        311,
    "inf_izq":        178,
    "inf_centro":     14,
    "inf_der":        402,
}


# =============================================================================
# Estructura de salida
# =============================================================================

@dataclass
class DatosBoca:
    """
    Resultado del analisis de boca para un frame.

    El campo evento_bostezo es True SOLO en el frame donde se completa
    un bostezo (al cerrarse la boca). En todos los demas frames es False.
    """
    valido: bool

    mar: float = 0.0                          # MAR instantaneo
    boca_abierta: bool = False                # estado actual con histeresis

    duracion_apertura_actual_ms: float = 0.0  # cuanto lleva abierta (0 si cerrada)

    # Eventos discretos
    evento_bostezo: bool = False              # True solo al cerrar la boca tras bostezo
    duracion_bostezo_ms: float = 0.0          # solo valido si evento_bostezo=True

    # Metricas temporales
    bostezos_por_minuto: float = 0.0

    # Diagnostico
    tiempo_procesamiento_ms: float = 0.0
    motivo_invalido: str = ""

    def __repr__(self) -> str:
        if not self.valido:
            return f"DatosBoca(invalido: {self.motivo_invalido})"
        estado = "ABIERTA" if self.boca_abierta else "cerrada"
        evt = f", BOSTEZO({self.duracion_bostezo_ms:.0f}ms)" if self.evento_bostezo else ""
        return f"DatosBoca(MAR={self.mar:.3f}, {estado}, bpm={self.bostezos_por_minuto:.1f}{evt})"


class ErrorAnalizadorBoca(Exception):
    """Error tecnico en el analizador."""


# =============================================================================
# Clase principal
# =============================================================================

class AnalizadorBoca:
    """
    Analizador de boca: MAR + detector de bostezos.

    Stateful: mantiene historial temporal.
    No es thread-safe.
    """

    # Defaults
    UMBRAL_APERTURA_DEFAULT = 0.50   # MAR sobre el cual consideramos boca abierta
    UMBRAL_CIERRE_DEFAULT = 0.40     # MAR debajo del cual consideramos boca cerrada (histeresis)

    # Limites de duracion del bostezo (ms)
    DURACION_MIN_BOSTEZO_MS = 2000    # apertura menor a esto NO es bostezo
    DURACION_MAX_BOSTEZO_MS = 10000   # apertura mayor a esto se descarta (anomalo)

    # Timeout de perdida de rostro (segundos)
    TIMEOUT_PERDIDA_ROSTRO_S = 2.0

    def __init__(
        self,
        config: Optional["Config"] = None,
        umbral_apertura: Optional[float] = None,
        umbral_cierre: Optional[float] = None,
        ventana_bostezos_seg: float = 300.0,  # 5 minutos
    ) -> None:
        """
        Parametros
        ----------
        config : Config | None
            Configuracion global (no usada aun, reservada).
        umbral_apertura : float | None
            MAR sobre el cual consideramos la boca abierta. Default 0.50.
        umbral_cierre : float | None
            MAR debajo del cual consideramos boca cerrada (histeresis).
            Default 0.40. Debe ser < umbral_apertura.
        ventana_bostezos_seg : float
            Tamaño de la ventana para bostezos/min. Default 300 (5 min).
        """
        self.config = config

        ua = float(umbral_apertura) if umbral_apertura is not None else self.UMBRAL_APERTURA_DEFAULT
        uc = float(umbral_cierre) if umbral_cierre is not None else self.UMBRAL_CIERRE_DEFAULT

        if not (0.0 < uc < ua < 2.0):
            raise ValueError(
                f"Umbrales invalidos: cierre={uc}, apertura={ua}. "
                "Debe cumplirse 0 < cierre < apertura < 2.0"
            )
        self.umbral_apertura = ua
        self.umbral_cierre = uc

        if ventana_bostezos_seg <= 0:
            raise ValueError("ventana_bostezos_seg debe ser > 0")
        self.ventana_bostezos_seg = float(ventana_bostezos_seg)

        _log.info(
            "Umbrales MAR: cierre=%.3f, apertura=%.3f. Ventana bostezos=%.0fs",
            self.umbral_cierre, self.umbral_apertura, self.ventana_bostezos_seg,
        )

        # ---------- Estado interno ----------

        # Historia de bostezos completados: timestamps de cierre del bostezo
        self._historial_bostezos: deque[float] = deque()

        # Estado del Schmitt trigger
        self._boca_abierta_estado: bool = False

        # Cuando empezo la apertura actual (None si no esta abierta)
        self._ts_inicio_apertura: Optional[float] = None

        # Timestamp del ultimo frame con rostro presente
        self._ts_ultimo_rostro: Optional[float] = None

    # ------------------------------------------------------------------
    # Configuracion
    # ------------------------------------------------------------------

    def actualizar_umbrales(
        self,
        umbral_apertura: float,
        umbral_cierre: float,
    ) -> None:
        """Actualiza los umbrales (usado por el calibrador en 4.7 si aplica)."""
        if not (0.0 < umbral_cierre < umbral_apertura < 2.0):
            raise ValueError(f"umbrales invalidos: cierre={umbral_cierre}, apertura={umbral_apertura}")
        self.umbral_apertura = float(umbral_apertura)
        self.umbral_cierre = float(umbral_cierre)
        _log.info(
            "Umbrales MAR actualizados: cierre=%.3f, apertura=%.3f",
            self.umbral_cierre, self.umbral_apertura,
        )

    def reset(self) -> None:
        """Limpia el estado temporal."""
        self._historial_bostezos.clear()
        self._boca_abierta_estado = False
        self._ts_inicio_apertura = None
        self._ts_ultimo_rostro = None
        _log.info("AnalizadorBoca: estado reseteado")

    # ------------------------------------------------------------------
    # Calculo del MAR
    # ------------------------------------------------------------------

    @staticmethod
    def _calcular_mar(puntos_pixeles: np.ndarray) -> float:
        """
        Calcula el MAR a partir de los 8 puntos de la boca.

        Returns:
            MAR como float >= 0. Si la base horizontal es ~0
            (rostro de perfil), devuelve 0.0.
        """
        try:
            # Comisuras (base horizontal)
            c_izq = puntos_pixeles[BOCA_INDICES["comisura_izq"]].astype(np.float64)
            c_der = puntos_pixeles[BOCA_INDICES["comisura_der"]].astype(np.float64)

            # Pares verticales
            sup_izq = puntos_pixeles[BOCA_INDICES["sup_izq"]].astype(np.float64)
            inf_izq = puntos_pixeles[BOCA_INDICES["inf_izq"]].astype(np.float64)
            sup_centro = puntos_pixeles[BOCA_INDICES["sup_centro"]].astype(np.float64)
            inf_centro = puntos_pixeles[BOCA_INDICES["inf_centro"]].astype(np.float64)
            sup_der = puntos_pixeles[BOCA_INDICES["sup_der"]].astype(np.float64)
            inf_der = puntos_pixeles[BOCA_INDICES["inf_der"]].astype(np.float64)
        except (IndexError, ValueError):
            return 0.0

        d_horiz = np.linalg.norm(c_izq - c_der)
        if d_horiz < 1e-6:
            return 0.0

        d_v1 = np.linalg.norm(sup_izq - inf_izq)
        d_v2 = np.linalg.norm(sup_centro - inf_centro)
        d_v3 = np.linalg.norm(sup_der - inf_der)

        return float((d_v1 + d_v2 + d_v3) / (2.0 * d_horiz))

    # ------------------------------------------------------------------
    # Histeresis
    # ------------------------------------------------------------------

    def _aplicar_histeresis(self, mar: float) -> bool:
        """Devuelve True si la boca esta abierta (con histeresis)."""
        if self._boca_abierta_estado:
            if mar <= self.umbral_cierre:
                self._boca_abierta_estado = False
        else:
            if mar >= self.umbral_apertura:
                self._boca_abierta_estado = True
        return self._boca_abierta_estado

    # ------------------------------------------------------------------
    # Bostezos/min sobre ventana temporal
    # ------------------------------------------------------------------

    def _agregar_bostezo(self, ts: float) -> None:
        """Registra un bostezo y purga la ventana."""
        self._historial_bostezos.append(ts)
        limite = ts - self.ventana_bostezos_seg
        while self._historial_bostezos and self._historial_bostezos[0] < limite:
            self._historial_bostezos.popleft()

    def _calcular_bostezos_por_minuto(self, ts: float) -> float:
        """Calcula la tasa de bostezos en bostezos/min."""
        if not self._historial_bostezos:
            return 0.0
        limite = ts - self.ventana_bostezos_seg
        cantidad = sum(1 for t in self._historial_bostezos if t >= limite)
        return (cantidad / self.ventana_bostezos_seg) * 60.0

    # ------------------------------------------------------------------
    # Procesamiento principal
    # ------------------------------------------------------------------

    def procesar(self, datos_rostro: DatosRostro) -> DatosBoca:
        """
        Procesa un frame y devuelve metricas + posible evento de bostezo.
        """
        t0 = time.monotonic()
        ts_frame = datos_rostro.timestamp if datos_rostro.timestamp > 0 else t0

        # 1) Si no hay rostro, manejar timeout y devolver invalido
        if not datos_rostro.rostro_presente or datos_rostro.puntos_pixeles is None:
            self._manejar_perdida_rostro(ts_frame)
            return DatosBoca(
                valido=False,
                motivo_invalido="rostro no presente",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
                bostezos_por_minuto=self._calcular_bostezos_por_minuto(ts_frame),
            )

        self._ts_ultimo_rostro = ts_frame

        # 2) Calcular MAR
        mar = self._calcular_mar(datos_rostro.puntos_pixeles)
        if mar == 0.0:
            return DatosBoca(
                valido=False,
                motivo_invalido="MAR indeterminado (rostro de perfil?)",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
                bostezos_por_minuto=self._calcular_bostezos_por_minuto(ts_frame),
            )

        # 3) Aplicar histeresis
        estado_anterior = self._boca_abierta_estado
        boca_abierta = self._aplicar_histeresis(mar)

        # 4) Maquina de estados de bostezos
        evento_bostezo = False
        duracion_evento_ms = 0.0

        if boca_abierta and not estado_anterior:
            # Transicion cerrada -> abierta: arranca apertura
            self._ts_inicio_apertura = ts_frame
        elif not boca_abierta and estado_anterior:
            # Transicion abierta -> cerrada: vemos si fue bostezo
            if self._ts_inicio_apertura is not None:
                duracion = (ts_frame - self._ts_inicio_apertura) * 1000.0
                # Solo cuenta como bostezo si esta en el rango valido
                if self.DURACION_MIN_BOSTEZO_MS <= duracion <= self.DURACION_MAX_BOSTEZO_MS:
                    evento_bostezo = True
                    duracion_evento_ms = duracion
                    self._agregar_bostezo(ts_frame)
                elif duracion > self.DURACION_MAX_BOSTEZO_MS:
                    _log.warning(
                        "Apertura de boca de %.0fms descartada (> max %dms)",
                        duracion, self.DURACION_MAX_BOSTEZO_MS,
                    )
                # Si dura < 2000ms (hablar, sonrisa abierta), no es bostezo.
                # No es error, simplemente no es evento.
                self._ts_inicio_apertura = None

        # 5) Duracion de apertura actual
        duracion_apertura_actual_ms = 0.0
        if boca_abierta and self._ts_inicio_apertura is not None:
            duracion_apertura_actual_ms = (ts_frame - self._ts_inicio_apertura) * 1000.0

        # 6) bpm
        bpm = self._calcular_bostezos_por_minuto(ts_frame)

        return DatosBoca(
            valido=True,
            mar=mar,
            boca_abierta=boca_abierta,
            duracion_apertura_actual_ms=duracion_apertura_actual_ms,
            evento_bostezo=evento_bostezo,
            duracion_bostezo_ms=duracion_evento_ms,
            bostezos_por_minuto=bpm,
            tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
        )

    def _manejar_perdida_rostro(self, ts_actual: float) -> None:
        """
        Si la perdida dura mas que TIMEOUT_PERDIDA_ROSTRO_S, resetea
        el estado de boca abierta. Mantiene el historial de bostezos.
        """
        if self._ts_ultimo_rostro is None:
            return

        tiempo_sin_rostro = ts_actual - self._ts_ultimo_rostro
        if tiempo_sin_rostro > self.TIMEOUT_PERDIDA_ROSTRO_S:
            _log.warning(
                "Rostro perdido por %.1fs, reseteando estado de apertura",
                tiempo_sin_rostro,
            )
            self._boca_abierta_estado = False
            self._ts_inicio_apertura = None
            self._ts_ultimo_rostro = None

    # ------------------------------------------------------------------
    # Visualizacion para debug
    # ------------------------------------------------------------------

    @staticmethod
    def dibujar_boca(
        frame_bgr: np.ndarray,
        datos_rostro: DatosRostro,
        datos_boca: DatosBoca,
    ) -> np.ndarray:
        """
        Dibuja los 8 puntos de la boca y los conecta. Color cambia segun estado:
        - Verde: cerrada
        - Amarillo: abierta pero no es bostezo aun (< 2s)
        - Rojo: bostezo en curso (>= 2s)
        """
        import cv2
        if not datos_rostro.rostro_presente or datos_rostro.puntos_pixeles is None:
            return frame_bgr.copy()

        out = frame_bgr.copy()

        # Determinar color
        if not datos_boca.valido or not datos_boca.boca_abierta:
            color = (0, 255, 0)  # verde: cerrada
        elif datos_boca.duracion_apertura_actual_ms >= AnalizadorBoca.DURACION_MIN_BOSTEZO_MS:
            color = (0, 0, 255)  # rojo: bostezo en curso
        else:
            color = (0, 255, 255)  # amarillo: abierta pero corta

        # Contorno externo (comisuras + 3 superiores + 3 inferiores)
        # Orden: c_izq -> sup_izq -> sup_centro -> sup_der -> c_der -> inf_der -> inf_centro -> inf_izq -> volver
        orden = [
            "comisura_izq", "sup_izq", "sup_centro", "sup_der",
            "comisura_der", "inf_der", "inf_centro", "inf_izq",
        ]
        pts = np.array(
            [datos_rostro.puntos_pixeles[BOCA_INDICES[n]] for n in orden],
            dtype=np.int32,
        )
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=1)
        for p in pts:
            cv2.circle(out, (int(p[0]), int(p[1])), 2, color, -1)

        return out
