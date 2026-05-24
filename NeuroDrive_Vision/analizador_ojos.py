"""
NeuroDrive Vision - Analizador de ojos
======================================

Calcula el Eye Aspect Ratio (EAR), el PERCLOS y detecta eventos de
parpadeos, parpadeos lentos y microsueños.

Decisiones tomadas (ver chat de planificacion para detalle):
  - Formula EAR de Soukupova & Cech (2016).
  - 6 puntos por ojo (indices del codigo de referencia del proyecto).
  - Histeresis con doble umbral (Schmitt trigger) para evitar oscilaciones.
  - Umbral configurable: usa default sin calibracion, recibe valores
    calibrados desde el calibrador.py en la etapa 4.7.
  - Ventana PERCLOS de 60 seg, medida en TIEMPO REAL (no en frames).
  - Maquina de estados interna para detectar parpadeo / lento / microsueño.
  - Si se pierde el rostro > 2 seg, se resetea el estado interno.

Eventos emitidos (en el campo evento_parpadeo del DatosOjos):
  - ""               : no hay evento este frame
  - "normal"         : parpadeo de 100-400 ms (sano)
  - "lento"          : parpadeo de 400-1500 ms (señal de fatiga)
  - "microsueño"     : cierre > 1500 ms (señal grave)

NOTA: El evento aparece SOLO en el frame donde el ojo vuelve a abrirse.
El Pre-FSM del Core consume estos eventos y aplica reglas de mayor nivel
(ej. "3 microsueños en 5 min = nivel critico").

API:
    analizador = AnalizadorOjos(config)
    datos = analizador.procesar(datos_rostro)
    if datos.valido and datos.evento_parpadeo:
        print(f"Evento: {datos.evento_parpadeo}")
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


_log = logging.getLogger("NeuroDrive.AnalizadorOjos")


# =============================================================================
# Indices de landmarks de MediaPipe FaceMesh para los ojos
# =============================================================================
#
# Los 6 puntos de cada ojo siguen el orden P1..P6 de Soukupova & Cech:
#   P1 = esquina exterior (lado oreja)
#   P2 = parpado superior, lado exterior
#   P3 = parpado superior, lado interior (lado nariz)
#   P4 = esquina interior (lado nariz)
#   P5 = parpado inferior, lado interior
#   P6 = parpado inferior, lado exterior
#
# El EAR es:
#         |P2 - P6| + |P3 - P5|
#  EAR =  ---------------------
#               2 * |P1 - P4|

# Indices del ojo izquierdo del CONDUCTOR (su izquierda).
# Aparece en el lado derecho de la imagen vista por la camara.
OJO_IZQ_INDICES = (33, 160, 158, 133, 153, 144)

# Indices del ojo derecho del CONDUCTOR (su derecha).
# Aparece en el lado izquierdo de la imagen.
OJO_DER_INDICES = (263, 387, 385, 362, 380, 373)


# =============================================================================
# Estructura de salida
# =============================================================================

@dataclass
class DatosOjos:
    """
    Resultado del analisis de ojos para un frame.

    Si valido=False, los valores numericos son 0.0 y no deben usarse
    (ej. cuando no hay rostro detectado).

    El campo evento_parpadeo aparece SOLO en el frame donde se completa
    un parpadeo (al reabrir el ojo). En todos los demas frames es "".
    """
    valido: bool

    # EAR instantaneo
    ear_izq: float = 0.0
    ear_der: float = 0.0
    ear_promedio: float = 0.0

    # Estado actual con histeresis aplicada
    ojos_cerrados: bool = False

    # Metricas temporales
    perclos: float = 0.0                      # fraccion 0.0-1.0
    parpadeos_por_minuto: float = 0.0         # tasa
    duracion_cierre_actual_ms: float = 0.0    # solo si ojos_cerrados=True

    # Eventos discretos (solo en el frame de cierre del parpadeo)
    evento_parpadeo: str = ""                 # "", "normal", "lento", "microsueño"
    duracion_parpadeo_ms: float = 0.0         # solo si evento_parpadeo != ""

    # Diagnostico
    tiempo_procesamiento_ms: float = 0.0
    motivo_invalido: str = ""

    def __repr__(self) -> str:
        if not self.valido:
            return f"DatosOjos(invalido: {self.motivo_invalido})"
        cerrado = "CERRADO" if self.ojos_cerrados else "abierto"
        evt = f", evt={self.evento_parpadeo}({self.duracion_parpadeo_ms:.0f}ms)" if self.evento_parpadeo else ""
        return (
            f"DatosOjos(EAR={self.ear_promedio:.3f}, {cerrado}, "
            f"PERCLOS={self.perclos:.2f}, bpm={self.parpadeos_por_minuto:.1f}{evt})"
        )


class ErrorAnalizadorOjos(Exception):
    """Error tecnico en el analizador."""


# =============================================================================
# Clase principal
# =============================================================================

class AnalizadorOjos:
    """
    Analizador de ojos: EAR, PERCLOS, parpadeos, microsueños.

    Stateful: mantiene historial temporal de cierre/apertura.
    No es thread-safe.

    Lifecycle:
      - __init__: configura umbrales y ventanas.
      - procesar(datos_rostro): analiza un frame.
      - reset(): limpia el estado (al cambiar de conductor o iniciar sesion).
      - actualizar_umbrales(...): para cuando el calibrador termine.
    """

    # Defaults razonables si no hay calibracion (paper original)
    EAR_BASE_DEFAULT = 0.30           # EAR tipico con ojos abiertos
    FACTOR_CIERRE_DEFAULT = 0.75      # umbral_cierre = base * 0.70
    FACTOR_APERTURA_DEFAULT = 0.85    # umbral_apertura = base * 0.80

    # Limites de duracion de eventos (ms)
    DURACION_MIN_PARPADEO_MS = 60       # menos que esto es ruido, no parpadeo
    DURACION_MAX_NORMAL_MS = 400        # parpadeo normal hasta 400 ms
    DURACION_MAX_LENTO_MS = 1500        # parpadeo lento hasta 1500 ms
    # > 1500 ms = microsueño

    # Si se pierde el rostro mas de este tiempo, reseteamos el estado interno
    TIMEOUT_PERDIDA_ROSTRO_S = 2.0

    def __init__(
        self,
        config: Optional["Config"] = None,
        ear_base: Optional[float] = None,
        factor_cierre: Optional[float] = None,
        factor_apertura: Optional[float] = None,
        ventana_perclos_seg: float = 60.0,
        ventana_parpadeos_seg: float = 60.0,
    ) -> None:
        """
        Parametros
        ----------
        config : Config | None
            Configuracion global (no usada aun, reservada).
        ear_base : float | None
            EAR de referencia con ojos abiertos. Si None, usa 0.30.
            Lo pisa el calibrador con el valor real del conductor.
        factor_cierre : float | None
            Multiplicador del EAR base para el umbral de cierre.
            Default 0.70 -> umbral_cierre = ear_base * 0.70.
        factor_apertura : float | None
            Multiplicador del EAR base para el umbral de apertura
            (histeresis Schmitt). Default 0.80.
        ventana_perclos_seg : float
            Tamaño de la ventana temporal para PERCLOS, en segundos.
        ventana_parpadeos_seg : float
            Tamaño de la ventana para calcular parpadeos/minuto.
        """
        self.config = config

        # Umbrales
        self.ear_base = float(ear_base) if ear_base is not None else self.EAR_BASE_DEFAULT
        fc = float(factor_cierre) if factor_cierre is not None else self.FACTOR_CIERRE_DEFAULT
        fa = float(factor_apertura) if factor_apertura is not None else self.FACTOR_APERTURA_DEFAULT

        if not (0.1 < self.ear_base < 1.0):
            raise ValueError(f"ear_base fuera de rango razonable: {self.ear_base}")
        if not (0.0 < fc < fa <= 1.0):
            raise ValueError(
                f"factores invalidos: cierre={fc}, apertura={fa}. "
                "Debe cumplirse 0 < cierre < apertura <= 1"
            )

        self.factor_cierre = fc
        self.factor_apertura = fa
        self._recalcular_umbrales_absolutos()

        if ventana_perclos_seg <= 0:
            raise ValueError("ventana_perclos_seg debe ser > 0")
        if ventana_parpadeos_seg <= 0:
            raise ValueError("ventana_parpadeos_seg debe ser > 0")
        self.ventana_perclos_seg = float(ventana_perclos_seg)
        self.ventana_parpadeos_seg = float(ventana_parpadeos_seg)

        # =========================================================
        # Estado interno
        # =========================================================

        # Historia de muestras para PERCLOS: (timestamp, cerrado_bool)
        # deque sin maxlen porque el limite es temporal, no de cantidad.
        self._historial_cierres: deque[tuple[float, bool]] = deque()

        # Historia de parpadeos completados: timestamps de cierre
        self._historial_parpadeos: deque[float] = deque()

        # Estado del Schmitt trigger
        self._ojos_cerrados_estado: bool = False

        # Cuando empezo el cierre actual (si lo hay)
        self._ts_inicio_cierre: Optional[float] = None

        # Timestamp del ultimo frame con rostro presente
        self._ts_ultimo_rostro: Optional[float] = None

    # ------------------------------------------------------------------
    # Configuracion de umbrales
    # ------------------------------------------------------------------

    def _recalcular_umbrales_absolutos(self) -> None:
        """Recalcula umbral_cierre y umbral_apertura desde ear_base y factores."""
        self.umbral_cierre = self.ear_base * self.factor_cierre
        self.umbral_apertura = self.ear_base * self.factor_apertura
        _log.info(
            "Umbrales EAR: base=%.3f, cierre=%.3f, apertura=%.3f",
            self.ear_base, self.umbral_cierre, self.umbral_apertura,
        )

    def actualizar_umbrales(
        self,
        ear_base: float,
        factor_cierre: Optional[float] = None,
        factor_apertura: Optional[float] = None,
    ) -> None:
        """
        Actualiza los umbrales con datos calibrados. Llamado tipicamente
        por el calibrador (etapa 4.7) al finalizar su medicion.

        No resetea el estado de PERCLOS / parpadeos.
        """
        if not (0.1 < ear_base < 1.0):
            raise ValueError(f"ear_base fuera de rango: {ear_base}")
        self.ear_base = float(ear_base)
        if factor_cierre is not None:
            self.factor_cierre = float(factor_cierre)
        if factor_apertura is not None:
            self.factor_apertura = float(factor_apertura)
        if not (0.0 < self.factor_cierre < self.factor_apertura <= 1.0):
            raise ValueError("factores invalidos tras actualizacion")
        self._recalcular_umbrales_absolutos()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Limpia el estado temporal (historiales, cierre actual, etc.)."""
        self._historial_cierres.clear()
        self._historial_parpadeos.clear()
        self._ojos_cerrados_estado = False
        self._ts_inicio_cierre = None
        self._ts_ultimo_rostro = None
        _log.info("AnalizadorOjos: estado reseteado")

    # ------------------------------------------------------------------
    # Calculo del EAR
    # ------------------------------------------------------------------

    @staticmethod
    def _calcular_ear(puntos_pixeles: np.ndarray, indices: tuple) -> float:
        """
        Calcula el EAR de un ojo dado sus 6 indices.

        Returns:
            EAR como float. Si la base (P1-P4) es ~0 (rostro casi de perfil
            o landmarks corruptos), devuelve 0.0 (interpretado como "ojo
            indeterminado", el caller decide que hacer).
        """
        # Extraer los 6 puntos (P1..P6)
        try:
            p1 = puntos_pixeles[indices[0]].astype(np.float64)
            p2 = puntos_pixeles[indices[1]].astype(np.float64)
            p3 = puntos_pixeles[indices[2]].astype(np.float64)
            p4 = puntos_pixeles[indices[3]].astype(np.float64)
            p5 = puntos_pixeles[indices[4]].astype(np.float64)
            p6 = puntos_pixeles[indices[5]].astype(np.float64)
        except (IndexError, ValueError):
            return 0.0

        # Distancias verticales
        d_vert_1 = np.linalg.norm(p2 - p6)
        d_vert_2 = np.linalg.norm(p3 - p5)

        # Distancia horizontal (base)
        d_horiz = np.linalg.norm(p1 - p4)

        if d_horiz < 1e-6:
            # Base degenerada: rostro casi de perfil o landmarks malos.
            # 0.0 es senal para el caller de "no usar este valor".
            return 0.0

        return float((d_vert_1 + d_vert_2) / (2.0 * d_horiz))

    # ------------------------------------------------------------------
    # Histeresis (Schmitt trigger)
    # ------------------------------------------------------------------

    def _aplicar_histeresis(self, ear: float) -> bool:
        """
        Decide si los ojos estan cerrados aplicando histeresis sobre el
        estado anterior. Modifica self._ojos_cerrados_estado.

        Returns:
            True si los ojos estan cerrados, False si abiertos.
        """
        if self._ojos_cerrados_estado:
            # Estado actual: cerrado. Solo cambia si EAR sube por encima
            # del umbral de apertura (mas alto).
            if ear >= self.umbral_apertura:
                self._ojos_cerrados_estado = False
        else:
            # Estado actual: abierto. Solo cambia si EAR baja por debajo
            # del umbral de cierre (mas bajo).
            if ear <= self.umbral_cierre:
                self._ojos_cerrados_estado = True

        return self._ojos_cerrados_estado

    # ------------------------------------------------------------------
    # PERCLOS y parpadeos/min sobre ventanas temporales
    # ------------------------------------------------------------------

    def _agregar_muestra_historial(self, ts: float, cerrado: bool) -> None:
        """Agrega una muestra al historial y purga las viejas."""
        self._historial_cierres.append((ts, cerrado))

        # Purgar muestras fuera de la ventana PERCLOS
        limite = ts - self.ventana_perclos_seg
        while self._historial_cierres and self._historial_cierres[0][0] < limite:
            self._historial_cierres.popleft()

    def _calcular_perclos(self) -> float:
        """Devuelve la fraccion de muestras cerradas en la ventana."""
        if not self._historial_cierres:
            return 0.0
        total = len(self._historial_cierres)
        cerradas = sum(1 for _, c in self._historial_cierres if c)
        return cerradas / total

    def _agregar_parpadeo(self, ts: float) -> None:
        """Registra un parpadeo completado y purga la ventana."""
        self._historial_parpadeos.append(ts)
        limite = ts - self.ventana_parpadeos_seg
        while self._historial_parpadeos and self._historial_parpadeos[0] < limite:
            self._historial_parpadeos.popleft()

    def _calcular_parpadeos_por_minuto(self, ts: float) -> float:
        """Calcula la tasa de parpadeos en parpadeos/min."""
        if not self._historial_parpadeos:
            return 0.0
        # Cuantos parpadeos hay en la ventana actual
        limite = ts - self.ventana_parpadeos_seg
        cantidad = sum(1 for t in self._historial_parpadeos if t >= limite)
        # Escalamos a parpadeos/minuto
        return (cantidad / self.ventana_parpadeos_seg) * 60.0

    # ------------------------------------------------------------------
    # Clasificacion de eventos
    # ------------------------------------------------------------------

    @classmethod
    def _clasificar_parpadeo(cls, duracion_ms: float) -> str:
        """
        Devuelve el tipo de evento segun la duracion del cierre:
          - "" si la duracion es muy corta (ruido)
          - "normal" si 60-400 ms
          - "lento" si 400-1500 ms
          - "microsueño" si > 1500 ms
        """
        if duracion_ms < cls.DURACION_MIN_PARPADEO_MS:
            return ""  # ruido, ignorar
        if duracion_ms <= cls.DURACION_MAX_NORMAL_MS:
            return "normal"
        if duracion_ms <= cls.DURACION_MAX_LENTO_MS:
            return "lento"
        return "microsueño"

    # ------------------------------------------------------------------
    # Procesamiento principal
    # ------------------------------------------------------------------

    def procesar(self, datos_rostro: DatosRostro) -> DatosOjos:
        """
        Procesa un frame y devuelve metricas + posible evento.

        Parametros
        ----------
        datos_rostro : DatosRostro
            Salida del DetectorRostro (Etapa 4.2).

        Returns
        -------
        DatosOjos
            Si no hay rostro o landmarks: valido=False.
            Si hay rostro: valido=True con todas las metricas.
        """
        t0 = time.monotonic()
        # Usamos el timestamp del frame (capturado en el momento de la lectura)
        # si esta disponible, sino el monotonic actual. El timestamp del frame
        # es mas preciso porque no incluye el delay del procesamiento.
        ts_frame = datos_rostro.timestamp if datos_rostro.timestamp > 0 else t0

        # 1) Si no hay rostro, manejar el timeout y devolver invalido
        if not datos_rostro.rostro_presente or datos_rostro.puntos_pixeles is None:
            self._manejar_perdida_rostro(ts_frame)
            return DatosOjos(
                valido=False,
                motivo_invalido="rostro no presente",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
                # Aun asi devolvemos PERCLOS y BPM acumulados (info historica util)
                perclos=self._calcular_perclos(),
                parpadeos_por_minuto=self._calcular_parpadeos_por_minuto(ts_frame),
            )

        # Registrar que vimos un rostro en este frame
        self._ts_ultimo_rostro = ts_frame

        # 2) Calcular EAR de ambos ojos
        ear_izq = self._calcular_ear(datos_rostro.puntos_pixeles, OJO_IZQ_INDICES)
        ear_der = self._calcular_ear(datos_rostro.puntos_pixeles, OJO_DER_INDICES)

        # Si uno de los dos es 0.0 (rostro de perfil, landmarks corruptos),
        # usamos solo el otro. Si ambos son 0.0, el frame es invalido.
        if ear_izq == 0.0 and ear_der == 0.0:
            return DatosOjos(
                valido=False,
                motivo_invalido="EAR indeterminado en ambos ojos (rostro de perfil?)",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
                perclos=self._calcular_perclos(),
                parpadeos_por_minuto=self._calcular_parpadeos_por_minuto(ts_frame),
            )

        # Promedio: si uno es 0, usamos solo el otro
        if ear_izq == 0.0:
            ear_promedio = ear_der
        elif ear_der == 0.0:
            ear_promedio = ear_izq
        else:
            ear_promedio = (ear_izq + ear_der) / 2.0

        # 3) Aplicar histeresis para determinar estado abierto/cerrado
        estado_anterior = self._ojos_cerrados_estado
        cerrados = self._aplicar_histeresis(ear_promedio)

        # 4) Maquina de estados de parpadeos
        evento = ""
        duracion_evento_ms = 0.0

        if cerrados and not estado_anterior:
            # Transicion abierto -> cerrado: arranca un cierre
            self._ts_inicio_cierre = ts_frame
        elif not cerrados and estado_anterior:
            # Transicion cerrado -> abierto: terminamos un cierre, clasificamos
            if self._ts_inicio_cierre is not None:
                duracion = (ts_frame - self._ts_inicio_cierre) * 1000.0
                evento = self._clasificar_parpadeo(duracion)
                if evento:
                    duracion_evento_ms = duracion
                    # Solo contamos como "parpadeo" para BPM los normales
                    # (los lentos y microsueños son senales de fatiga, no de
                    # parpadeo regular). El Pre-FSM puede combinar como quiera.
                    if evento == "normal":
                        self._agregar_parpadeo(ts_frame)
                self._ts_inicio_cierre = None

        # 5) Duracion del cierre actual (si esta cerrado en este frame)
        duracion_cierre_actual_ms = 0.0
        if cerrados and self._ts_inicio_cierre is not None:
            duracion_cierre_actual_ms = (ts_frame - self._ts_inicio_cierre) * 1000.0

        # 6) Actualizar PERCLOS
        self._agregar_muestra_historial(ts_frame, cerrados)
        perclos = self._calcular_perclos()
        bpm = self._calcular_parpadeos_por_minuto(ts_frame)

        return DatosOjos(
            valido=True,
            ear_izq=ear_izq,
            ear_der=ear_der,
            ear_promedio=ear_promedio,
            ojos_cerrados=cerrados,
            perclos=perclos,
            parpadeos_por_minuto=bpm,
            duracion_cierre_actual_ms=duracion_cierre_actual_ms,
            evento_parpadeo=evento,
            duracion_parpadeo_ms=duracion_evento_ms,
            tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
        )

    def _manejar_perdida_rostro(self, ts_actual: float) -> None:
        """
        Si la perdida de rostro dura mas de TIMEOUT_PERDIDA_ROSTRO_S,
        resetea el estado interno. Esto evita falsos microsueños cuando
        el conductor mira hacia otro lado o se mueve mucho.
        """
        if self._ts_ultimo_rostro is None:
            return  # no habiamos visto rostro nunca; nada que hacer

        tiempo_sin_rostro = ts_actual - self._ts_ultimo_rostro
        if tiempo_sin_rostro > self.TIMEOUT_PERDIDA_ROSTRO_S:
            _log.warning(
                "Rostro perdido por %.1fs, reseteando estado de cierre",
                tiempo_sin_rostro,
            )
            # Reseteamos solo el estado del Schmitt, no los historiales:
            # PERCLOS y BPM siguen siendo validos (son ventanas, no se borran
            # por una perdida).
            self._ojos_cerrados_estado = False
            self._ts_inicio_cierre = None
            self._ts_ultimo_rostro = None

    # ------------------------------------------------------------------
    # Visualizacion para debug (no usar en runtime de produccion)
    # ------------------------------------------------------------------

    @staticmethod
    def dibujar_ojos(
        frame_bgr: np.ndarray,
        datos_rostro: DatosRostro,
        datos_ojos: DatosOjos,
    ) -> np.ndarray:
        """
        Dibuja los contornos de los 6 puntos de cada ojo y colorea segun
        estado (verde=abierto, rojo=cerrado).
        """
        import cv2
        if not datos_rostro.rostro_presente or datos_rostro.puntos_pixeles is None:
            return frame_bgr.copy()

        out = frame_bgr.copy()
        color = (0, 0, 255) if datos_ojos.valido and datos_ojos.ojos_cerrados else (0, 255, 0)

        for indices in (OJO_IZQ_INDICES, OJO_DER_INDICES):
            pts = datos_rostro.puntos_pixeles[list(indices)].astype(np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=1)
            for (x, y) in pts:
                cv2.circle(out, (int(x), int(y)), 2, color, -1)

        return out
