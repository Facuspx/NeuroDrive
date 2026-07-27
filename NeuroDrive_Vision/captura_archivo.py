"""
NeuroDrive Vision - Captura de video desde archivo
====================================================

Lee frames de un archivo de video (grabado previamente con la Pi Camera)
en lugar de capturar en vivo desde la camara CSI. Util para:

  - Validacion reproducible: mismo video, mismos resultados. Podes
    ajustar umbrales y volver a correr sabiendo que la unica variable
    que cambio es tu ajuste.
  - Casos etiquetados: grabas un video con eventos anotados (bostezos,
    cabeceos, microsuenos en tiempos conocidos) y comparas la deteccion
    contra la verdad conocida.
  - Testear condiciones especiales (por ejemplo, oscuridad con IR) sin
    tener que armar el escenario cada vez.
  - Demo y defensa TFI: reproducis el mismo caso ante el tribunal.

DISENO:
    - Misma API que CapturaVideo (iniciar, leer, detener, activa, fps_real,
      context manager). El resto del pipeline no se entera de que la fuente
      cambio; el integrador solo elige que clase instanciar.
    - Backend: cv2.VideoCapture. Soporta MJPEG, MP4, AVI y cualquier
      formato que OpenCV entienda.
    - Dos modos de timestamp:
        MODO_VIDEO: timestamp = frame_index / fps_video. Rapido: procesa
                    el video lo mas rapido posible. Las ventanas del
                    Pre-FSM funcionan porque usan "tiempo del video".
                    Bueno para iteracion rapida.
        MODO_TIEMPO_REAL: timestamp = time.time() y hace sleep para
                    mantener el ritmo del video. Bueno para demo, para
                    verificar que el sistema no se atrasa en tiempo real,
                    y para integracion con el Core.

Como grabar un video compatible con la Pi Camera Module 2 NoIR:
    rpicam-vid -t 60000 --width 1640 --height 1232 --framerate 15 \
      --codec mjpeg --mode 1640:1232 --hflip \
      --awb custom --awbgains 1.8,1.8 -o test_normal.mjpeg

USO tipico:
    cap = CapturaArchivo(config, ruta="test_normal.mjpeg")
    cap.iniciar()
    while cap.activa:
        frame, ts = cap.leer()
        if frame is None:
            break  # fin del video
        # procesar frame BGR (480, 640, 3)
    cap.detener()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

# La misma excepcion que usa CapturaVideo, para que el integrador maneje
# ambos casos con un unico except.
from NeuroDrive_Vision.captura_video import ErrorCaptura

if TYPE_CHECKING:
    from NeuroDrive_Core.config_loader import Config


_log = logging.getLogger("NeuroDrive.CapturaArchivo")


class CapturaArchivo:
    """
    Fuente de video desde archivo. API identica a CapturaVideo.

    Thread-safe: NO. Debe usarse desde un solo hilo.
    """

    # Modos de timestamp
    MODO_VIDEO = "video"              # rapido: ts = frame_index / fps_video
    MODO_TIEMPO_REAL = "tiempo_real"  # ritmo real: ts = time.time() con sleep

    def __init__(
        self,
        config: "Config",
        ruta: str,
        modo_timestamp: str = MODO_VIDEO,
        loop: bool = False,
        fps_override: Optional[float] = None,
    ) -> None:
        """
        Parametros
        ----------
        config : Config
            Configuracion global. Se leen ancho_objetivo, alto_objetivo,
            fps_deseado (usados para escalar y para el sleep en modo
            tiempo real).
        ruta : str
            Path al archivo de video. Se valida en iniciar().
        modo_timestamp : str
            MODO_VIDEO (default): rapido, sin sleep, ts sintetico basado
            en el FPS del video.
            MODO_TIEMPO_REAL: mantiene el ritmo real del video con sleep,
            usa time.time() como timestamp.
        loop : bool
            Si True, cuando el video termina vuelve a arrancar. Default
            False (una sola pasada).
        fps_override : float | None
            Si se pasa, se USA como FPS del video ignorando lo que reporte
            OpenCV. Necesario para formatos sin metadatos de FPS, como el
            raw MJPEG que produce 'rpicam-vid -o archivo.mjpeg' (OpenCV
            devuelve 25 FPS por default en ese caso, que no es la realidad).
            RECOMENDADO: al grabar con rpicam-vid a un .mjpeg, pasar aca
            el mismo --framerate que usaste al grabar. Alternativa: grabar
            a un contenedor con metadatos (.avi con MJPG, .mp4).
        """
        if modo_timestamp not in (self.MODO_VIDEO, self.MODO_TIEMPO_REAL):
            raise ValueError(
                f"modo_timestamp invalido: {modo_timestamp}. "
                f"Valores validos: {self.MODO_VIDEO!r}, {self.MODO_TIEMPO_REAL!r}"
            )
        if fps_override is not None and fps_override <= 0:
            raise ValueError(f"fps_override debe ser > 0: {fps_override}")

        self.ruta = str(ruta)
        self.modo_timestamp = modo_timestamp
        self.loop = bool(loop)
        self.fps_override = fps_override

        # Resolucion objetivo (para MediaPipe y procesamiento)
        self.ancho_objetivo = config.vision.resolucion_ancho
        self.alto_objetivo = config.vision.resolucion_alto
        # FPS deseado del pipeline; en modo tiempo real, el video se
        # reproduce al FPS NATIVO del archivo (no al deseado).
        self.fps = config.vision.fps_deseado

        # FPS nativo del archivo (se lee en iniciar())
        self.fps_video: float = 0.0
        # Duracion total del video en segundos (informativa)
        self.duracion_video_seg: float = 0.0
        # Cantidad total de frames del video (informativa)
        self.total_frames_video: int = 0

        # Estado runtime
        self._cap: Optional[cv2.VideoCapture] = None
        self._activa = False
        self._fin_de_archivo = False

        # Contadores de frames del ARCHIVO (posicion en el video).
        # Distinto de frames_capturados: al hacer loop, este resetea.
        self._frame_idx_archivo = 0

        # Para MODO_TIEMPO_REAL: cuando arranco cada frame
        self._ts_ultimo_frame_wall = 0.0

        # Metricas (API igual a CapturaVideo)
        self.frames_capturados = 0
        self.frames_fallidos = 0
        self._ts_inicio = 0.0
        self._loops_completados = 0

    # ==================================================================
    # PROPIEDADES
    # ==================================================================

    @property
    def activa(self) -> bool:
        """True si la captura esta corriendo."""
        return (
            self._activa
            and self._cap is not None
            and not self._fin_de_archivo
        )

    @property
    def fps_real(self) -> float:
        """
        FPS real de PROCESAMIENTO desde el inicio.

        En modo VIDEO refleja la velocidad de lectura+decodificacion.
        En modo TIEMPO REAL refleja el FPS del video (con el sleep aplicado).
        """
        if self._ts_inicio == 0 or self.frames_capturados == 0:
            return 0.0
        elapsed = time.time() - self._ts_inicio
        if elapsed <= 0:
            return 0.0
        return self.frames_capturados / elapsed

    @property
    def fin_de_archivo(self) -> bool:
        """True si se llego al final del video y no hay loop."""
        return self._fin_de_archivo

    # ==================================================================
    # CICLO DE VIDA
    # ==================================================================

    def iniciar(self) -> None:
        """
        Abre el archivo de video y valida que se pueda leer.

        Raises:
            ErrorCaptura: si el archivo no existe, no se puede abrir,
            no tiene FPS valido, o no se puede leer el primer frame.
        """
        if self._activa:
            _log.warning("iniciar() llamado pero la captura ya esta activa")
            return

        # Verificar que el archivo existe (mensaje claro antes de OpenCV)
        p = Path(self.ruta)
        if not p.exists():
            raise ErrorCaptura(
                f"El archivo de video no existe: {self.ruta}. "
                f"Grabar con: rpicam-vid -t 60000 --width 1640 --height 1232 "
                f"--framerate 15 --codec mjpeg -o {self.ruta}"
            )
        if not p.is_file():
            raise ErrorCaptura(f"La ruta no es un archivo: {self.ruta}")

        # Abrir con OpenCV
        self._cap = cv2.VideoCapture(self.ruta)
        if not self._cap.isOpened():
            self._cap = None
            raise ErrorCaptura(
                f"OpenCV no pudo abrir el video: {self.ruta}. "
                f"Verificar que el formato sea soportado (MJPEG, MP4, AVI)."
            )

        # Leer metadatos del video
        fps_leido = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.total_frames_video = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ancho = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resolucion del FPS a usar. Prioridad:
        #   1. fps_override si el usuario lo paso explicitamente
        #   2. FPS leido de metadatos si es valido y NO es el default sospechoso
        #   3. FPS deseado del config
        # NOTA: OpenCV devuelve 25.0 FPS por default cuando no hay metadatos
        # (tipico en raw MJPEG). No podemos distinguir "el video es realmente
        # de 25 FPS" de "OpenCV no sabe y devuelve 25". Por eso, si el usuario
        # sospecha esto, debe pasar fps_override.
        if self.fps_override is not None:
            self.fps_video = float(self.fps_override)
            _log.info(
                "Usando fps_override=%.1f (FPS leido de metadatos: %.1f)",
                self.fps_video, fps_leido,
            )
        elif not (fps_leido > 0.0) or fps_leido != fps_leido:  # NaN check
            _log.warning(
                "El archivo no tiene FPS valido en sus metadatos. "
                "Usando FPS deseado del config (%d) como referencia. "
                "Si el video es raw MJPEG, considere pasar fps_override.",
                self.fps
            )
            self.fps_video = float(self.fps)
        else:
            self.fps_video = fps_leido
            # Aviso si es el default sospechoso de OpenCV
            if abs(fps_leido - 25.0) < 0.01 and fps_leido != float(self.fps):
                _log.warning(
                    "El video reporta 25.0 FPS: este es el default de OpenCV "
                    "cuando no encuentra metadatos (comun en raw MJPEG). Si "
                    "el video no es realmente de 25 FPS, pasar fps_override "
                    "con el FPS con que se grabo (ej: 15)."
                )

        # Duracion (aproximada si total_frames es dudoso)
        if self.total_frames_video > 0 and self.fps_video > 0:
            self.duracion_video_seg = self.total_frames_video / self.fps_video
        else:
            self.duracion_video_seg = 0.0

        # Leer un frame de prueba para confirmar que se puede decodificar
        ok, _ = self._cap.read()
        if not ok:
            self._cap.release()
            self._cap = None
            raise ErrorCaptura(
                f"No se pudo leer el primer frame de {self.ruta}. "
                f"El archivo puede estar corrupto o vacio."
            )
        # Volver al inicio (el primer frame se descarto en la prueba)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Todo OK
        self._activa = True
        self._fin_de_archivo = False
        self._frame_idx_archivo = 0
        self.frames_capturados = 0
        self.frames_fallidos = 0
        self._loops_completados = 0
        self._ts_inicio = time.time()
        self._ts_ultimo_frame_wall = self._ts_inicio

        dur = (f"{self.duracion_video_seg:.1f}s" if self.duracion_video_seg > 0
               else "duracion desconocida")
        _log.info(
            "CapturaArchivo iniciada: %s (%dx%d @ %.1f FPS, %s) -> %dx%d @ modo=%s%s",
            self.ruta, ancho, alto, self.fps_video, dur,
            self.ancho_objetivo, self.alto_objetivo,
            self.modo_timestamp,
            ", loop=True" if self.loop else "",
        )

    def detener(self) -> None:
        """Cierra el archivo. Idempotente."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                _log.warning("Error al cerrar el archivo: %s", e)
            self._cap = None

        self._activa = False
        _log.info(
            "CapturaArchivo detenida. Total: %d frames leidos, %d fallidos, "
            "FPS real de procesamiento: %.1f, loops completados: %d",
            self.frames_capturados, self.frames_fallidos,
            self.fps_real, self._loops_completados,
        )

    # ==================================================================
    # LECTURA DE FRAMES
    # ==================================================================

    def leer(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Lee el siguiente frame del video.

        Returns:
            (frame_bgr, timestamp):
                - frame_bgr: numpy array (alto_objetivo, ancho_objetivo, 3)
                  en formato BGR, escalado si hace falta. None si se acabo
                  el video (fin de archivo) o hubo error.
                - timestamp: segun modo_timestamp (video o tiempo real).
        """
        # Si ya sabemos que se acabo, devolvemos None
        if not self._activa or self._cap is None:
            return None, time.time()

        ok, frame_raw = self._cap.read()

        if not ok:
            # Fin de archivo. Decidir si hacer loop o cerrar.
            if self.loop:
                self._loops_completados += 1
                _log.info("Fin del video, reiniciando (loop #%d).",
                          self._loops_completados)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_idx_archivo = 0
                # Reintentar la lectura una vez
                ok, frame_raw = self._cap.read()
                if not ok:
                    _log.warning("Loop pedido pero el video no se pudo reiniciar.")
                    self._fin_de_archivo = True
                    return None, time.time()
            else:
                self._fin_de_archivo = True
                return None, self._calcular_timestamp()

        # Frame leido OK
        self._frame_idx_archivo += 1

        # Escalar a resolucion objetivo si hace falta (misma logica que
        # CapturaVideo: solo escalamos si las dimensiones no coinciden).
        if (frame_raw.shape[1] != self.ancho_objetivo
                or frame_raw.shape[0] != self.alto_objetivo):
            frame = cv2.resize(
                frame_raw,
                (self.ancho_objetivo, self.alto_objetivo),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame = frame_raw

        # Calcular timestamp segun modo (y aplicar sleep si tiempo real)
        ts = self._calcular_timestamp()

        self.frames_capturados += 1
        return frame, ts

    def _calcular_timestamp(self) -> float:
        """
        Calcula el timestamp del frame recien leido segun modo_timestamp.

        En MODO_VIDEO: retorna un ts sintetico basado en la posicion del
        frame en el video y el FPS nativo. Las ventanas del Pre-FSM (60s
        PERCLOS, 15min bostezos, etc.) se calculan en "tiempo del video",
        no "tiempo real de reloj". Perfecto para tests reproducibles.

        En MODO_TIEMPO_REAL: usa time.time() y hace sleep si vamos mas
        rapido que el FPS nativo. Asi 60 segundos de video se procesan
        en 60 segundos de reloj. Perfecto para demo e integracion con
        el Core en escenarios realistas.
        """
        if self.modo_timestamp == self.MODO_VIDEO:
            # Timestamp sintetico basado en la posicion del frame.
            # Sumamos self._ts_inicio para que sea comparable a time.time()
            # (los analizadores comparan diferencias, asi que el offset no
            # importa; pero un ts que arranque en, por ejemplo, 0.066s
            # rompe la validacion timestamp > 0 del EventoVision. Al
            # sumar self._ts_inicio queda en el orden de time.time()).
            fps = self.fps_video if self.fps_video > 0 else float(self.fps)
            return self._ts_inicio + (self._frame_idx_archivo / fps)

        # MODO_TIEMPO_REAL: mantener el ritmo del video con sleep
        fps = self.fps_video if self.fps_video > 0 else float(self.fps)
        periodo = 1.0 / fps  # segundos entre frames
        ahora = time.time()
        elapsed = ahora - self._ts_ultimo_frame_wall
        if elapsed < periodo:
            # Vamos mas rapido que el video: dormir el resto
            time.sleep(periodo - elapsed)
            ahora = time.time()
        self._ts_ultimo_frame_wall = ahora
        return ahora

    # ==================================================================
    # CONTEXT MANAGER
    # ==================================================================

    def __enter__(self) -> "CapturaArchivo":
        self.iniciar()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detener()

    def __repr__(self) -> str:
        estado = "activa" if self.activa else (
            "fin de archivo" if self._fin_de_archivo else "detenida"
        )
        return (
            f"CapturaArchivo({self.ruta!r}, modo={self.modo_timestamp}, "
            f"{self.ancho_objetivo}x{self.alto_objetivo}, estado={estado}, "
            f"frames={self.frames_capturados})"
        )


__all__ = ["CapturaArchivo"]
