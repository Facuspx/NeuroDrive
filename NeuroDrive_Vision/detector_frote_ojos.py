"""
NeuroDrive Vision - Detector de frote de ojos
=============================================

Detecta cuando el conductor se frota los ojos. El gesto se considera
"frote" cuando al menos una de las 5 puntas de los dedos de la mano
entra en la region delimitada alrededor de uno de los ojos, sostenido
por al menos 500 ms.

Decisiones tomadas (ver chat de planificacion para detalle):
  - MediaPipe Hands con max_num_hands=2 (cualquier mano puede frotarse).
  - model_complexity=0 (rapido, alcanza para detectar dedos cerca del ojo).
  - Region del ojo = rectangulo proporcional al ancho del ojo (4 x ancho_ojo),
    asi escala automaticamente con la distancia del conductor a la camara.
  - Umbral de duracion sostenida: 500 ms (descarta toques accidentales).
  - El evento se emite al INICIAR el frote (cuando se cumplen los 500 ms),
    no al final. La duracion se reporta continuamente.
  - Tiempo de procesamiento esperado: 25-35 ms cuando hay mano detectada,
    ~5-8 ms cuando no hay mano (palm detector solo es rapido).

API:
    detector = DetectorFroteOjos(config)
    detector.iniciar()
    datos = detector.procesar(frame_bgr, datos_rostro)
    if datos.frote_en_curso:
        print(f"Frote en curso: {datos.duracion_frote_actual_ms:.0f}ms")
    if datos.evento_frote_iniciado:
        print("INICIO DE FROTE DE OJOS")
    detector.detener()
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Silenciar logs ruidosos de TF Lite / Protobuf
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import mediapipe as mp

from NeuroDrive_Vision.detector_rostro import DatosRostro
from NeuroDrive_Vision.analizador_ojos import OJO_IZQ_INDICES, OJO_DER_INDICES

try:
    from NeuroDrive_Core.config_loader import Config
except ImportError:
    Config = None  # type: ignore


_log = logging.getLogger("NeuroDrive.DetectorFroteOjos")


# =============================================================================
# Constantes
# =============================================================================

# Indices de las puntas de los 5 dedos en MediaPipe Hands
TIPS_DEDOS = (4, 8, 12, 16, 20)  # pulgar, indice, medio, anular, meñique


# =============================================================================
# Estructura de salida
# =============================================================================

@dataclass
class DatosFroteOjos:
    """
    Resultado del analisis de frote de ojos para un frame.

    El campo evento_frote_iniciado es True SOLO en el frame donde se
    cumplen los 500 ms de contacto sostenido. En todos los demas frames
    es False.
    """
    valido: bool

    # Cantidad de manos detectadas en el frame (0, 1 o 2)
    manos_detectadas: int = 0

    # Estado actual con histeresis temporal
    frote_en_curso: bool = False

    # Cuanto lleva el frote actual (0 si no hay frote)
    duracion_frote_actual_ms: float = 0.0

    # True SOLO en el frame que arranca el frote (al cumplir umbral)
    evento_frote_iniciado: bool = False

    # Tasa
    frotes_por_minuto: float = 0.0

    # Diagnostico
    tiempo_procesamiento_ms: float = 0.0
    motivo_invalido: str = ""

    # Para debug visual: rectangulos de las regiones de ojo y puntas detectadas
    region_ojo_izq: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    region_ojo_der: Optional[Tuple[int, int, int, int]] = None
    puntas_detectadas: list = field(default_factory=list)  # list[(x, y)]
    puntas_en_zona: list = field(default_factory=list)     # list[(x, y)]

    # Landmarks completos de cada mano detectada: 21 puntos por mano.
    # Estructura: list de manos, cada mano es list de 21 (x, y) en pixeles.
    # Util para dibujar el esqueleto completo de la mano (ventana de malla).
    landmarks_manos: list = field(default_factory=list)  # list[list[(x, y)]]

    def __repr__(self) -> str:
        if not self.valido:
            return f"DatosFroteOjos(invalido: {self.motivo_invalido})"
        estado = "FROTE" if self.frote_en_curso else "ok"
        evt = " [INICIO]" if self.evento_frote_iniciado else ""
        return (
            f"DatosFroteOjos({estado}, manos={self.manos_detectadas}, "
            f"dur={self.duracion_frote_actual_ms:.0f}ms, fpm={self.frotes_por_minuto:.1f}{evt})"
        )


class ErrorDetectorFroteOjos(Exception):
    """Error tecnico en el detector."""


# =============================================================================
# Clase principal
# =============================================================================

class DetectorFroteOjos:
    """
    Detector de frote de ojos usando MediaPipe Hands.

    Stateful: mantiene historial temporal de contactos.
    No es thread-safe.

    Lifecycle:
      - __init__: configura parametros, no carga modelo.
      - iniciar(): carga el modelo de Hands (~1-2 segundos en Pi 5).
      - procesar(frame_bgr, datos_rostro): analiza un frame.
      - detener(): libera el modelo.
      - reset(): limpia el estado temporal.

    Soporta context manager.
    """

    # Duracion minima de contacto sostenido para contar como frote
    DURACION_MIN_FROTE_MS = 500.0

    # Si el rostro se pierde por mas que esto, reseteamos el estado
    TIMEOUT_PERDIDA_ROSTRO_S = 2.0

    # Multiplicadores para definir la region del ojo a partir de su ancho
    # region = 4 * ancho_ojo de ancho, 2 * ancho_ojo de alto
    FACTOR_ANCHO_REGION = 4.0
    FACTOR_ALTO_REGION = 2.0

    # Umbral para warning de tiempo
    UMBRAL_WARNING_MS = 80.0

    def __init__(
        self,
        config: Optional["Config"] = None,
        max_manos: int = 2,
        model_complexity: int = 0,
        min_deteccion: float = 0.5,
        min_tracking: float = 0.5,
        duracion_min_frote_ms: Optional[float] = None,
        ventana_frotes_seg: float = 300.0,  # 5 minutos
        procesar_cada_n_frames: int = 2,
    ) -> None:
        """
        Parametros
        ----------
        config : Config | None
            Configuracion global.
        max_manos : int
            Cuantas manos detectar (1 o 2). Default 2.
        model_complexity : int
            0 = rapido, 1 = preciso. En Pi 5, default 0.
        min_deteccion : float
            Confianza minima para considerar mano detectada.
        min_tracking : float
            Confianza minima para tracking entre frames.
        duracion_min_frote_ms : float | None
            Tiempo de contacto sostenido para contar como frote.
            Default 500 ms.
        ventana_frotes_seg : float
            Ventana para frotes/min. Default 300 s.
        procesar_cada_n_frames : int
            Throttling de MediaPipe Hands. Default 2: Hands corre 1 de cada
            2 frames (~7.5 Hz a 15 FPS). La maquina de estados de frote
            corre SIEMPRE a frame completo. Poner 1 para evaluar Hands en
            cada frame (mas preciso pero ~2x mas costoso).
        """
        self.config = config

        if max_manos not in (1, 2):
            raise ValueError(f"max_manos debe ser 1 o 2, recibido {max_manos}")
        if model_complexity not in (0, 1):
            raise ValueError(f"model_complexity debe ser 0 o 1, recibido {model_complexity}")
        if procesar_cada_n_frames < 1:
            raise ValueError("procesar_cada_n_frames debe ser >= 1")

        self.max_manos = max_manos
        self.model_complexity = model_complexity
        self.min_deteccion = float(min_deteccion)
        self.min_tracking = float(min_tracking)

        if duracion_min_frote_ms is not None:
            if duracion_min_frote_ms <= 0:
                raise ValueError("duracion_min_frote_ms debe ser > 0")
            self.duracion_min_frote_ms = float(duracion_min_frote_ms)
        else:
            self.duracion_min_frote_ms = self.DURACION_MIN_FROTE_MS

        if ventana_frotes_seg <= 0:
            raise ValueError("ventana_frotes_seg debe ser > 0")
        self.ventana_frotes_seg = float(ventana_frotes_seg)
        self.procesar_cada_n_frames = int(procesar_cada_n_frames)

        # Modelo MediaPipe Hands (lazy)
        self._hands: Optional[mp.solutions.hands.Hands] = None
        self._activo: bool = False

        # =========================================================
        # Estado interno
        # =========================================================
        # Contador de frames para throttling
        self._contador_frames: int = 0

        # Resultado del ultimo procesamiento real (para frames que se saltan)
        self._ultimo_resultado: Optional[DatosFroteOjos] = None

        # Ultimas puntas y cantidad de manos detectadas por Hands.
        # Se reutilizan en los frames saltados por throttling para que la
        # maquina de estados de frote siga corriendo a frame completo.
        self._ultimas_puntas: list[Tuple[int, int]] = []
        self._ultimas_manos: int = 0
        # Landmarks completos (21 por mano) de la ultima deteccion
        self._ultimos_landmarks_manos: list[list[Tuple[int, int]]] = []

        # Cuando arranco el contacto actual (si lo hay)
        self._ts_inicio_contacto: Optional[float] = None

        # Si ya emitimos el evento_frote_iniciado para este contacto.
        # Se resetea cuando termina el contacto.
        self._frote_ya_emitido: bool = False

        # Historial de frotes (timestamps de inicio)
        self._historial_frotes: deque[float] = deque()

        # Ultimo frame con rostro presente
        self._ts_ultimo_rostro: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def activo(self) -> bool:
        return self._activo and self._hands is not None

    def iniciar(self) -> None:
        if self._activo:
            _log.warning("DetectorFroteOjos ya estaba iniciado")
            return
        try:
            t0 = time.monotonic()
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_manos,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_deteccion,
                min_tracking_confidence=self.min_tracking,
            )
            t_carga = (time.monotonic() - t0) * 1000.0
            self._activo = True
            _log.info(
                "DetectorFroteOjos iniciado (carga=%.0fms, max_manos=%d, complexity=%d)",
                t_carga, self.max_manos, self.model_complexity,
            )
        except Exception as e:
            self._hands = None
            self._activo = False
            raise ErrorDetectorFroteOjos(f"No se pudo cargar MediaPipe Hands: {e}") from e

    def detener(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception as e:
                _log.warning("Error al cerrar Hands: %s", e)
            self._hands = None
        self._activo = False
        _log.info("DetectorFroteOjos detenido")

    def __enter__(self) -> "DetectorFroteOjos":
        self.iniciar()
        return self

    def __exit__(self, *_) -> None:
        self.detener()

    def __del__(self) -> None:
        try:
            self.detener()
        except Exception:
            pass

    def reset(self) -> None:
        self._contador_frames = 0
        self._ultimo_resultado = None
        self._ultimas_puntas = []
        self._ultimas_manos = 0
        self._ultimos_landmarks_manos = []
        self._ts_inicio_contacto = None
        self._frote_ya_emitido = False
        self._historial_frotes.clear()
        self._ts_ultimo_rostro = None
        _log.info("DetectorFroteOjos: estado reseteado")

    # ------------------------------------------------------------------
    # Calculo de la region del ojo (bounding box)
    # ------------------------------------------------------------------

    @classmethod
    def _calcular_region_ojo(
        cls,
        puntos_pixeles: np.ndarray,
        indices_ojo: tuple,
        ancho_frame: int,
        alto_frame: int,
    ) -> Tuple[int, int, int, int]:
        """
        Calcula la region rectangular (x, y, w, h) alrededor del ojo,
        proporcional a su ancho.

        indices_ojo: tupla de 6 indices (P1..P6) tal como OJO_IZQ_INDICES.
        """
        try:
            p1 = puntos_pixeles[indices_ojo[0]]  # esquina exterior
            p4 = puntos_pixeles[indices_ojo[3]]  # esquina interior
        except IndexError:
            return (0, 0, 0, 0)

        cx = (int(p1[0]) + int(p4[0])) / 2.0
        cy = (int(p1[1]) + int(p4[1])) / 2.0

        ancho_ojo = float(np.linalg.norm(p1.astype(np.float64) - p4.astype(np.float64)))
        if ancho_ojo < 5.0:
            # Ojo muy chico, posible perfil. Devolvemos box pequeño centrado.
            ancho_ojo = 20.0

        w = ancho_ojo * cls.FACTOR_ANCHO_REGION
        h = ancho_ojo * cls.FACTOR_ALTO_REGION
        x = cx - w / 2.0
        y = cy - h / 2.0

        # Clamp al frame
        x = max(0, int(x))
        y = max(0, int(y))
        w = min(ancho_frame - x, int(w))
        h = min(alto_frame - y, int(h))

        return (x, y, w, h)

    @staticmethod
    def _punto_en_rect(px: int, py: int, rect: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = rect
        return (x <= px < x + w) and (y <= py < y + h)

    # ------------------------------------------------------------------
    # Tasa
    # ------------------------------------------------------------------

    def _agregar_frote(self, ts: float) -> None:
        self._historial_frotes.append(ts)
        limite = ts - self.ventana_frotes_seg
        while self._historial_frotes and self._historial_frotes[0] < limite:
            self._historial_frotes.popleft()

    def _calcular_frotes_por_minuto(self, ts: float) -> float:
        if not self._historial_frotes:
            return 0.0
        limite = ts - self.ventana_frotes_seg
        cantidad = sum(1 for t in self._historial_frotes if t >= limite)
        return (cantidad / self.ventana_frotes_seg) * 60.0

    # ------------------------------------------------------------------
    # Procesamiento principal
    # ------------------------------------------------------------------

    def procesar(
        self,
        frame_bgr: np.ndarray,
        datos_rostro: DatosRostro,
    ) -> DatosFroteOjos:
        """
        Procesa un frame.

        Parametros
        ----------
        frame_bgr : np.ndarray
            Frame BGR (mismo formato que DetectorRostro espera).
        datos_rostro : DatosRostro
            Salida del DetectorRostro. Necesario para conocer la posicion
            de los ojos.

        Returns
        -------
        DatosFroteOjos
        """
        if not self._activo or self._hands is None:
            raise ErrorDetectorFroteOjos(
                "DetectorFroteOjos no esta activo. Llama a iniciar() primero."
            )

        t0 = time.monotonic()
        ts_frame = datos_rostro.timestamp if datos_rostro.timestamp > 0 else t0

        # ============================================================
        # 1) Validacion del frame
        # ============================================================
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            raise ErrorDetectorFroteOjos("frame_bgr invalido")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ErrorDetectorFroteOjos(
                f"frame_bgr debe ser shape (H,W,3) uint8, recibido {frame_bgr.shape} {frame_bgr.dtype}"
            )

        alto, ancho = frame_bgr.shape[:2]

        # ============================================================
        # 2) Deteccion de manos con throttling.
        #
        # MediaPipe Hands es lo unico caro (~25-45 ms). Lo evaluamos solo
        # 1 de cada procesar_cada_n_frames frames. En los frames saltados
        # REUTILIZAMOS las ultimas puntas detectadas.
        #
        # IMPORTANTE: la maquina de estados de frote (pasos 4-6) corre
        # SIEMPRE, en todos los frames. Asi la deteccion de frote sigue
        # siendo precisa a frame completo aunque Hands corra a media tasa.
        # Esto evita perder eventos en los frames saltados.
        # ============================================================
        self._contador_frames += 1
        toca_evaluar_hands = (self._contador_frames % self.procesar_cada_n_frames) == 0

        if toca_evaluar_hands:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            try:
                resultado_hands = self._hands.process(frame_rgb)
            except Exception as e:
                _log.error("Error en Hands.process(): %s", e)
                return DatosFroteOjos(
                    valido=False,
                    motivo_invalido=f"Hands.process fallo: {e}",
                    tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
                )

            manos_detectadas = (
                len(resultado_hands.multi_hand_landmarks)
                if resultado_hands.multi_hand_landmarks else 0
            )

            # Lista de coordenadas de las 5 puntas de cada mano detectada,
            # y tambien los 21 landmarks completos de cada mano (para dibujar
            # el esqueleto en la ventana de malla).
            puntas: list[Tuple[int, int]] = []
            landmarks_manos: list[list[Tuple[int, int]]] = []
            if resultado_hands.multi_hand_landmarks:
                for hand_lms in resultado_hands.multi_hand_landmarks:
                    # Los 21 landmarks completos de esta mano
                    mano_completa: list[Tuple[int, int]] = []
                    for lm in hand_lms.landmark:
                        px = int(np.clip(lm.x, 0.0, 1.0) * ancho)
                        py = int(np.clip(lm.y, 0.0, 1.0) * alto)
                        mano_completa.append((px, py))
                    landmarks_manos.append(mano_completa)
                    # Las 5 puntas (subconjunto, para la deteccion de frote)
                    for tip_idx in TIPS_DEDOS:
                        if tip_idx < len(mano_completa):
                            puntas.append(mano_completa[tip_idx])

            # Guardar para reutilizar en frames saltados
            self._ultimas_puntas = puntas
            self._ultimas_manos = manos_detectadas
            self._ultimos_landmarks_manos = landmarks_manos
        else:
            # Frame saltado: reutilizamos la ultima deteccion de Hands.
            # Las puntas pueden estar levemente desactualizadas (1 frame),
            # pero para frotes de varios segundos es irrelevante.
            puntas = self._ultimas_puntas
            manos_detectadas = self._ultimas_manos
            landmarks_manos = self._ultimos_landmarks_manos

        # ============================================================
        # 3) Si no hay rostro, no podemos definir la region de ojos.
        # ============================================================
        if not datos_rostro.rostro_presente or datos_rostro.puntos_pixeles is None:
            self._manejar_perdida_rostro(ts_frame)
            res = DatosFroteOjos(
                valido=True,  # el procesamiento si fue valido, solo no detectamos frote
                manos_detectadas=manos_detectadas,
                frote_en_curso=False,
                puntas_detectadas=puntas,
                landmarks_manos=landmarks_manos,
                frotes_por_minuto=self._calcular_frotes_por_minuto(ts_frame),
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
                motivo_invalido="rostro_ausente",
            )
            self._ultimo_resultado = res
            return res

        self._ts_ultimo_rostro = ts_frame

        # ============================================================
        # 4) Calcular regiones de los ojos
        # ============================================================
        region_izq = self._calcular_region_ojo(
            datos_rostro.puntos_pixeles, OJO_IZQ_INDICES, ancho, alto,
        )
        region_der = self._calcular_region_ojo(
            datos_rostro.puntos_pixeles, OJO_DER_INDICES, ancho, alto,
        )

        # ============================================================
        # 5) Determinar si hay contacto (alguna punta en alguna region)
        # ============================================================
        puntas_en_zona: list[Tuple[int, int]] = []
        for (px, py) in puntas:
            if (
                self._punto_en_rect(px, py, region_izq)
                or self._punto_en_rect(px, py, region_der)
            ):
                puntas_en_zona.append((px, py))

        hay_contacto = len(puntas_en_zona) > 0

        # ============================================================
        # 6) Maquina de estados temporales del frote
        # ============================================================
        evento_iniciado = False
        duracion_actual_ms = 0.0
        frote_en_curso = False

        if hay_contacto:
            if self._ts_inicio_contacto is None:
                # Arranca contacto
                self._ts_inicio_contacto = ts_frame
                self._frote_ya_emitido = False

            duracion_actual_ms = (ts_frame - self._ts_inicio_contacto) * 1000.0

            if duracion_actual_ms >= self.duracion_min_frote_ms:
                frote_en_curso = True
                # Emitimos evento UNA SOLA VEZ por contacto
                if not self._frote_ya_emitido:
                    evento_iniciado = True
                    self._frote_ya_emitido = True
                    self._agregar_frote(ts_frame)
                    _log.info("Frote de ojos iniciado (duracion contacto=%.0fms)", duracion_actual_ms)
        else:
            # No hay contacto: si veniamos con contacto, lo terminamos
            if self._ts_inicio_contacto is not None:
                duracion_total = (ts_frame - self._ts_inicio_contacto) * 1000.0
                if duracion_total >= self.duracion_min_frote_ms:
                    _log.info("Frote de ojos terminado (duracion total=%.0fms)", duracion_total)
                self._ts_inicio_contacto = None
                self._frote_ya_emitido = False

        # ============================================================
        # 7) Construir resultado
        # ============================================================
        bpm = self._calcular_frotes_por_minuto(ts_frame)
        t_proc = (time.monotonic() - t0) * 1000.0

        if t_proc > self.UMBRAL_WARNING_MS:
            _log.warning("Procesamiento lento: %.1f ms", t_proc)

        res = DatosFroteOjos(
            valido=True,
            manos_detectadas=manos_detectadas,
            frote_en_curso=frote_en_curso,
            duracion_frote_actual_ms=duracion_actual_ms,
            evento_frote_iniciado=evento_iniciado,
            frotes_por_minuto=bpm,
            tiempo_procesamiento_ms=t_proc,
            region_ojo_izq=region_izq,
            region_ojo_der=region_der,
            puntas_detectadas=puntas,
            puntas_en_zona=puntas_en_zona,
            landmarks_manos=landmarks_manos,
        )
        self._ultimo_resultado = res
        return res

    def _manejar_perdida_rostro(self, ts_actual: float) -> None:
        """
        Si la perdida de rostro supera el timeout, resetea el estado.
        """
        if self._ts_ultimo_rostro is None:
            return
        tiempo_sin = ts_actual - self._ts_ultimo_rostro
        if tiempo_sin > self.TIMEOUT_PERDIDA_ROSTRO_S:
            _log.warning(
                "Rostro perdido por %.1fs, reseteando estado de frote",
                tiempo_sin,
            )
            self._ts_inicio_contacto = None
            self._frote_ya_emitido = False
            self._ts_ultimo_rostro = None

    # ------------------------------------------------------------------
    # Visualizacion debug
    # ------------------------------------------------------------------

    @staticmethod
    def dibujar(frame_bgr: np.ndarray, datos: DatosFroteOjos) -> np.ndarray:
        """
        Dibuja:
        - Rectangulos amarillos: regiones de los ojos.
        - Puntas verdes: dedos detectados.
        - Puntas rojas: dedos dentro de la region del ojo.
        - Rojo intermitente: frote en curso.
        """
        out = frame_bgr.copy()

        if not datos.valido:
            return out

        # Regiones
        for region in (datos.region_ojo_izq, datos.region_ojo_der):
            if region is not None and region[2] > 0 and region[3] > 0:
                x, y, w, h = region
                color = (0, 0, 255) if datos.frote_en_curso else (0, 255, 255)
                #cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)

        # Puntas de dedos
        for (px, py) in datos.puntas_detectadas:
            cv2.circle(out, (int(px), int(py)), 5, (0, 255, 0), -1)
        # Puntas en zona (encima)
        for (px, py) in datos.puntas_en_zona:
            cv2.circle(out, (int(px), int(py)), 6, (0, 0, 255), -1)

        return out
