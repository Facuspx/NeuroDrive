"""
NeuroDrive Vision - Detector de rostro
======================================

Wrapper de MediaPipe FaceMesh adaptado al pipeline de NeuroDrive.

Recibe un frame BGR de la captura (Etapa 4.1) y devuelve un objeto
DatosRostro con los 468 landmarks tanto en coordenadas de pixel
como normalizadas (necesarias para solvePnP en la Etapa 4.3).

Decisiones tomadas (ver chat de planificacion para detalle):
  - refine_landmarks=False: 468 puntos base son suficientes para
    EAR, MAR y cabeceo. El iris (10 puntos extra) solo sirve para
    gaze tracking, que no esta en el roadmap del proyecto.
  - max_num_faces=1: solo nos interesa el conductor.
  - static_image_mode=False: tracking entre frames (mas rapido).
  - min_detection_confidence=0.5, min_tracking_confidence=0.5:
    defaults conservadores para iluminacion variable de cabina.

API:
    detector = DetectorRostro(config)
    detector.iniciar()
    datos = detector.procesar(frame_bgr)
    if datos.rostro_presente:
        puntos = datos.puntos_pixeles   # list[(x, y)], 468 puntos
        puntos_n = datos.puntos_normalizados  # list[(x, y, z)] en [0,1]
    detector.detener()

Tambien soporta context manager:
    with DetectorRostro(config) as det:
        datos = det.procesar(frame)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe usa Protobuf por debajo y a veces ensucia stdout con
# warnings de TF Lite. Esto los silencia.
import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import mediapipe as mp

# Importacion del Config. Como aun no tengo el config_loader integrado
# en el sandbox, lo dejo como import opcional. En la Pi viene del Core.
try:
    from NeuroDrive_Core.config_loader import Config
except ImportError:
    Config = None  # type: ignore


_log = logging.getLogger("NeuroDrive.DetectorRostro")


# =============================================================================
# Estructuras de datos
# =============================================================================

@dataclass
class DatosRostro:
    """
    Resultado del procesamiento de un frame por el detector.

    Si rostro_presente es False, todos los campos opcionales son None.
    Si rostro_presente es True, puntos_pixeles y puntos_normalizados
    estan garantizados (no son None).
    """
    rostro_presente: bool
    puntos_pixeles: Optional[np.ndarray] = None       # shape (468, 2), int32
    puntos_normalizados: Optional[np.ndarray] = None  # shape (468, 3), float32, valores en [0,1] x [0,1] x [-?, ?]
    resolucion: Tuple[int, int] = (0, 0)              # (ancho, alto) del frame procesado
    timestamp: float = 0.0                            # time.monotonic() al iniciar el procesamiento
    tiempo_procesamiento_ms: float = 0.0              # cuanto tardo procesar este frame

    def __repr__(self) -> str:
        if self.rostro_presente:
            return (
                f"DatosRostro(presente=True, res={self.resolucion}, "
                f"t={self.tiempo_procesamiento_ms:.1f}ms)"
            )
        return f"DatosRostro(presente=False, t={self.tiempo_procesamiento_ms:.1f}ms)"


# =============================================================================
# Excepciones
# =============================================================================

class ErrorDetectorRostro(Exception):
    """Error en el detector de rostro."""


# =============================================================================
# Clase principal
# =============================================================================

class DetectorRostro:
    """
    Detector de rostro basado en MediaPipe FaceMesh.

    No es thread-safe. Una instancia debe ser usada desde un solo hilo
    (igual que CapturaVideo).

    Lifecycle:
      - __init__: solo guarda parametros, no carga el modelo.
      - iniciar(): carga el modelo de MediaPipe (lento, ~1-2 segundos en Pi 5).
      - procesar(frame): procesa un frame y devuelve DatosRostro.
      - detener(): libera el modelo.

    Tambien soporta context manager.
    """

    # Numero de landmarks que devuelve FaceMesh con refine_landmarks=False
    NUM_LANDMARKS = 468

    # Umbral de warning si un frame tarda mas que esto (en ms).
    # A 15 FPS el presupuesto total por frame es ~66 ms; si el detector
    # solo gasta mas de 100ms es sintoma de problema.
    UMBRAL_WARNING_MS = 100.0

    def __init__(
        self,
        config: Optional["Config"] = None,
        min_deteccion: float = 0.5,
        min_tracking: float = 0.5,
        max_rostros: int = 1,
        refine_landmarks: bool = False,
    ) -> None:
        """
        Parametros
        ----------
        config : Config | None
            Configuracion global (no usada aun, reservada).
        min_deteccion : float
            Confianza minima para considerar que hay rostro.
        min_tracking : float
            Confianza minima para mantener el tracking entre frames.
        max_rostros : int
            Cuantos rostros detectar. Default 1 (solo el conductor).
        refine_landmarks : bool
            False = 468 landmarks. True = 478 (incluye iris). Default False.
        """
        self.config = config
        self.min_deteccion = float(min_deteccion)
        self.min_tracking = float(min_tracking)
        self.max_rostros = int(max_rostros)
        self.refine_landmarks = bool(refine_landmarks)

        # Validacion de parametros
        if not (0.0 < self.min_deteccion <= 1.0):
            raise ValueError(f"min_deteccion debe estar en (0, 1], recibido {self.min_deteccion}")
        if not (0.0 < self.min_tracking <= 1.0):
            raise ValueError(f"min_tracking debe estar en (0, 1], recibido {self.min_tracking}")
        if self.max_rostros < 1:
            raise ValueError(f"max_rostros debe ser >= 1, recibido {self.max_rostros}")

        # El modelo se carga en iniciar()
        self._face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
        self._activo: bool = False

        # Buffers pre-asignados para evitar allocaciones por frame.
        # Se llenan en iniciar() cuando ya sabemos NUM_LANDMARKS.
        self._buf_pixeles: np.ndarray = np.zeros((self.NUM_LANDMARKS, 2), dtype=np.int32)
        self._buf_norm: np.ndarray = np.zeros((self.NUM_LANDMARKS, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def activo(self) -> bool:
        """True si el detector ya cargo el modelo y esta listo."""
        return self._activo and self._face_mesh is not None

    def iniciar(self) -> None:
        """
        Carga el modelo de MediaPipe FaceMesh.
        Es lento (~1-2 segundos en Pi 5), llamarlo una sola vez.
        """
        if self._activo:
            _log.warning("DetectorRostro ya estaba iniciado, ignorando llamada")
            return

        try:
            t0 = time.monotonic()
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_rostros,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_deteccion,
                min_tracking_confidence=self.min_tracking,
            )
            t_carga = (time.monotonic() - t0) * 1000.0
            self._activo = True
            _log.info(
                "DetectorRostro iniciado (carga=%.0fms, max_rostros=%d, refine=%s)",
                t_carga, self.max_rostros, self.refine_landmarks,
            )
        except Exception as e:
            self._face_mesh = None
            self._activo = False
            raise ErrorDetectorRostro(f"No se pudo cargar MediaPipe FaceMesh: {e}") from e

    def detener(self) -> None:
        """
        Libera el modelo de MediaPipe.
        Idempotente: llamarlo dos veces no rompe.
        """
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception as e:
                _log.warning("Error al cerrar FaceMesh: %s", e)
            self._face_mesh = None
        self._activo = False
        _log.info("DetectorRostro detenido")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DetectorRostro":
        self.iniciar()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detener()

    def __del__(self) -> None:
        # Ultima linea de defensa
        try:
            self.detener()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Procesamiento
    # ------------------------------------------------------------------

    def procesar(self, frame_bgr: np.ndarray) -> DatosRostro:
        """
        Procesa un frame BGR y devuelve los landmarks del rostro.

        Parametros
        ----------
        frame_bgr : np.ndarray
            Frame en formato BGR (lo que devuelve CapturaVideo.leer()).
            Shape (alto, ancho, 3), dtype uint8.

        Returns
        -------
        DatosRostro
            Si no hay rostro detectado: rostro_presente=False,
            puntos_*=None. Nunca lanza excepcion por falta de rostro,
            solo por errores tecnicos (modelo no cargado, frame invalido).
        """
        if not self._activo or self._face_mesh is None:
            raise ErrorDetectorRostro(
                "DetectorRostro no esta activo. Llama a iniciar() primero."
            )

        # Validacion del frame
        if frame_bgr is None:
            raise ErrorDetectorRostro("frame_bgr es None")
        if not isinstance(frame_bgr, np.ndarray):
            raise ErrorDetectorRostro(
                f"frame_bgr debe ser np.ndarray, recibido {type(frame_bgr).__name__}"
            )
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ErrorDetectorRostro(
                f"frame_bgr debe tener shape (H, W, 3), recibido {frame_bgr.shape}"
            )
        if frame_bgr.dtype != np.uint8:
            raise ErrorDetectorRostro(
                f"frame_bgr debe tener dtype uint8, recibido {frame_bgr.dtype}"
            )

        t0 = time.monotonic()
        alto, ancho = frame_bgr.shape[:2]

        # MediaPipe pide RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Optimizacion documentada por Google: marcar el array como no escribible
        # permite que MediaPipe lo pase por referencia en vez de copiarlo.
        frame_rgb.flags.writeable = False

        try:
            resultado = self._face_mesh.process(frame_rgb)
        except Exception as e:
            t_proc = (time.monotonic() - t0) * 1000.0
            _log.error("Error en FaceMesh.process(): %s", e)
            return DatosRostro(
                rostro_presente=False,
                resolucion=(ancho, alto),
                timestamp=t0,
                tiempo_procesamiento_ms=t_proc,
            )

        # ¿Hubo rostro?
        if not resultado.multi_face_landmarks:
            t_proc = (time.monotonic() - t0) * 1000.0
            return DatosRostro(
                rostro_presente=False,
                resolucion=(ancho, alto),
                timestamp=t0,
                tiempo_procesamiento_ms=t_proc,
            )

        # Si hay mas de un rostro (no deberia, pero por las dudas), tomamos el primero.
        # MediaPipe ya los devuelve ordenados por confianza implicita.
        landmarks = resultado.multi_face_landmarks[0].landmark

        # Sanity check: el numero de landmarks debe coincidir
        if len(landmarks) < self.NUM_LANDMARKS:
            _log.warning(
                "FaceMesh devolvio %d landmarks (esperaba %d), descartando",
                len(landmarks), self.NUM_LANDMARKS,
            )
            t_proc = (time.monotonic() - t0) * 1000.0
            return DatosRostro(
                rostro_presente=False,
                resolucion=(ancho, alto),
                timestamp=t0,
                tiempo_procesamiento_ms=t_proc,
            )

        # Volcar a los buffers pre-asignados de forma vectorizada.
        # IMPORTANTE: MediaPipe FaceMesh con frames no cuadrados (640x480 es 4:3)
        # emite un warning "NORM_RECT without IMAGE_DIMENSIONS is only supported
        # for the square ROI" y devuelve normalizadas X que pueden salirse del
        # rango [0,1] hasta un ~6%. Es bug documentado del calculador interno
        # de proyeccion en MediaPipe legacy. Lo solucionamos con clamp:
        # un punto reportado en x=1.06 realmente esta en el borde derecho de
        # la cara y debe quedar en x=1.0 (o en el pixel ancho-1).
        # Sin este clamp, EAR/MAR y solvePnP recibirian indices fuera de imagen.
        x_n = np.array([lm.x for lm in landmarks[:self.NUM_LANDMARKS]], dtype=np.float32)
        y_n = np.array([lm.y for lm in landmarks[:self.NUM_LANDMARKS]], dtype=np.float32)
        z_n = np.array([lm.z for lm in landmarks[:self.NUM_LANDMARKS]], dtype=np.float32)

        # Clamp normalizadas a [0, 1]
        np.clip(x_n, 0.0, 1.0, out=x_n)
        np.clip(y_n, 0.0, 1.0, out=y_n)
        # z no se clampea: es profundidad relativa, puede ser negativo

        self._buf_norm[:, 0] = x_n
        self._buf_norm[:, 1] = y_n
        self._buf_norm[:, 2] = z_n

        # Pixeles: x_pix en [0, ancho-1], y_pix en [0, alto-1]
        self._buf_pixeles[:, 0] = np.clip(
            (x_n * ancho).astype(np.int32), 0, ancho - 1
        )
        self._buf_pixeles[:, 1] = np.clip(
            (y_n * alto).astype(np.int32), 0, alto - 1
        )

        t_proc = (time.monotonic() - t0) * 1000.0
        if t_proc > self.UMBRAL_WARNING_MS:
            _log.warning(
                "Frame procesado en %.1fms (> %.0fms, presupuesto a 15 FPS es ~66ms)",
                t_proc, self.UMBRAL_WARNING_MS,
            )

        # Devolvemos COPIAS de los buffers para que el caller pueda
        # quedarselos sin riesgo de que los pisemos en el proximo frame.
        # Si el caller no las modifica, podriamos devolver el buffer
        # directo, pero la copia es barata (~5 KB) y evita bugs sutiles.
        return DatosRostro(
            rostro_presente=True,
            puntos_pixeles=self._buf_pixeles.copy(),
            puntos_normalizados=self._buf_norm.copy(),
            resolucion=(ancho, alto),
            timestamp=t0,
            tiempo_procesamiento_ms=t_proc,
        )

    # ------------------------------------------------------------------
    # Utilidad: dibujar landmarks (solo para debug/test_vision)
    # ------------------------------------------------------------------

    @staticmethod
    def dibujar_landmarks(
        frame_bgr: np.ndarray,
        datos: DatosRostro,
        color: Tuple[int, int, int] = (0, 255, 0),
        radio: int = 1,
    ) -> np.ndarray:
        """
        Devuelve una copia del frame con los landmarks dibujados.
        Util para test_vision y para debug visual. NO usar en runtime
        de produccion (cuesta CPU).
        """
        if not datos.rostro_presente or datos.puntos_pixeles is None:
            return frame_bgr.copy()

        out = frame_bgr.copy()
        for (x, y) in datos.puntos_pixeles:
            cv2.circle(out, (int(x), int(y)), radio, color, -1)
        return out
