"""
NeuroDrive Vision - Captura de video
=====================================

Captura frames de la Pi Camera Module 2 (CSI) en Raspberry Pi 5.

En Pi 5 con Debian Trixie, la camara CSI no funciona con
cv2.VideoCapture() porque el controlador rp1-cfe no expone V4L2
estandar. La solucion es usar rpicam-vid como pipe.

Estrategia de captura:
    - Captura a 1640x1232 (sensor completo, sin crop) para maximo
      campo de vision. A 640x480 el sensor hace crop central y el
      zoom es excesivo para distancias cortas (~50cm en cabina).
    - Codec MJPEG: confiable a cualquier resolucion. YUV420 tiene
      problemas de padding a resoluciones no-estandar.
    - Escalado por software a 640x480 para MediaPipe.
    - ~14 FPS reales en Pi 5 (suficiente para deteccion facial).

Pipeline:
    rpicam-vid --codec mjpeg -o - | [stdout] -> cv2.imdecode -> resize

API:
    cap = CapturaVideo(config)
    cap.iniciar()
    while cap.activa:
        frame, timestamp = cap.leer()
        if frame is not None:
            # procesar frame BGR (480, 640, 3)
            ...
    cap.detener()

Tambien soporta context manager:
    with CapturaVideo(config) as cap:
        frame, ts = cap.leer()
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from NeuroDrive_Core.config_loader import Config


_log = logging.getLogger("NeuroDrive.CapturaVideo")


class ErrorCaptura(Exception):
    """Error en la captura de video."""


class CapturaVideo:
    """
    Captura de video desde la Pi Camera CSI via rpicam-vid con MJPEG.

    Captura a resolucion nativa del sensor completo (1640x1232) y escala
    a la resolucion objetivo (640x480) para procesamiento.

    Thread-safe: NO. Debe ser usada desde un solo hilo.
    """

    # Resolucion de captura nativa (sensor completo, sin crop).
    # Segun 'rpicam-vid --list-cameras', el modo 1640x1232 del IMX219 usa
    # crop (0,0)/3280x2464 = SENSOR COMPLETO. Es el unico modo (junto al
    # 3280x2464, demasiado pesado) que no recorta el campo de vision.
    # El modo 640x480 recorta a una ventana central de 1280x960 -> zoom
    # excesivo, no sirve para distancias cortas en cabina.
    CAPTURA_ANCHO = 1640
    CAPTURA_ALTO = 1232

    # Flip horizontal: la camara apunta al conductor, sin flip la imagen
    # sale espejada (mover la cabeza a la derecha -> se ve a la izquierda).
    HFLIP = True
    VFLIP = False

    # Balance de blancos para el sensor NoIR (sin filtro infrarrojo).
    # El NoIR deja pasar el infrarrojo, lo que tine la imagen de rosa/magenta.
    # El AWB automatico de rpicam-vid no compensa esto bien. Fijamos las
    # ganancias de color manualmente (rojo, azul) para neutralizar el tinte.
    # Valores tipicos para NoIR bajo luz interior; si el tinte persiste se
    # ajustan: subir la primera reduce el rojo, subir la segunda reduce el azul.
    AWB_GAINS = "1.8,1.8"

    # Tamano de lectura del pipe (64KB por chunk)
    CHUNK_SIZE = 65536

    # Marcadores JPEG
    _JPEG_START = b"\xff\xd8"
    _JPEG_END = b"\xff\xd9"

    def __init__(self, config: Config) -> None:
        # Resolucion objetivo (para MediaPipe y procesamiento)
        self.ancho_objetivo = config.vision.resolucion_ancho
        self.alto_objetivo = config.vision.resolucion_alto
        self.fps = config.vision.fps_deseado

        # Estado runtime
        self._proceso: Optional[subprocess.Popen] = None
        self._activa = False
        self._buffer = b""

        # Metricas
        self.frames_capturados = 0
        self.frames_fallidos = 0
        self._ts_inicio = 0.0
        self._ts_ultimo_frame = 0.0

    # ==================================================================
    # PROPIEDADES
    # ==================================================================

    @property
    def activa(self) -> bool:
        """True si la captura esta corriendo."""
        return (
            self._activa
            and self._proceso is not None
            and self._proceso.poll() is None
        )

    @property
    def fps_real(self) -> float:
        """FPS real medido desde el inicio."""
        if self._ts_inicio == 0 or self.frames_capturados == 0:
            return 0.0
        elapsed = time.time() - self._ts_inicio
        if elapsed <= 0:
            return 0.0
        return self.frames_capturados / elapsed

    # ==================================================================
    # CICLO DE VIDA
    # ==================================================================

    def _construir_comando(self, con_flags_avanzados: bool) -> list:
        """
        Construye el comando rpicam-vid.

        Parametros
        ----------
        con_flags_avanzados : bool
            Si True, incluye --mode y los flags de balance de blancos.
            Si False, usa solo los flags basicos (compatibilidad con
            versiones viejas de rpicam-vid que no soportan --mode o
            --awb custom).
        """
        cmd = [
            "rpicam-vid",
            "-t", "0",                                   # duracion infinita
            "--width", str(self.CAPTURA_ANCHO),          # sensor completo
            "--height", str(self.CAPTURA_ALTO),
            "--framerate", str(self.fps),
            "--codec", "mjpeg",                          # MJPEG confiable
            "--nopreview",                               # sin ventana de preview
        ]

        if con_flags_avanzados:
            # --mode fuerza el modo de sensor exacto: garantiza que SIEMPRE
            # use el modo de sensor completo 1640x1232 y nunca caiga al
            # modo 640x480 (que recorta el campo). Formato: ancho:alto.
            cmd[2:2] = ["--mode", f"{self.CAPTURA_ANCHO}:{self.CAPTURA_ALTO}"]
            # Balance de blancos manual para neutralizar el tinte rosa del
            # sensor NoIR (sin filtro infrarrojo).
            cmd += ["--awb", "custom", "--awbgains", self.AWB_GAINS]

        # Flip de imagen (espejado). hflip corrige el efecto espejo.
        if self.HFLIP:
            cmd.append("--hflip")
        if self.VFLIP:
            cmd.append("--vflip")

        cmd += ["-o", "-"]  # salida a stdout
        return cmd

    def _intentar_arrancar(self, cmd: list) -> bool:
        """
        Lanza rpicam-vid con el comando dado y verifica que produzca frames.

        Returns:
            True si arranco bien y produce frames JPEG validos.
            False si fallo (el proceso queda detenido).
        """
        try:
            self._proceso = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.CHUNK_SIZE * 8,
            )
        except FileNotFoundError:
            raise ErrorCaptura(
                "rpicam-vid no encontrado. Instalar con: "
                "sudo apt install libcamera-apps"
            )
        except OSError as e:
            raise ErrorCaptura(f"Error lanzando rpicam-vid: {e}")

        # Esperar un poco y verificar que el proceso sigue corriendo
        time.sleep(0.5)
        if self._proceso.poll() is not None:
            # El proceso murio: probablemente un flag no soportado
            self._proceso = None
            return False

        # Leer un frame de prueba
        self._buffer = b""
        frame_prueba = self._leer_siguiente_jpeg()
        if frame_prueba is None:
            try:
                self._proceso.terminate()
                self._proceso.wait(timeout=2.0)
            except Exception:
                pass
            self._proceso = None
            return False

        return True

    def iniciar(self) -> None:
        """
        Lanza el subproceso rpicam-vid en modo MJPEG y prepara la captura.

        Intenta primero con los flags avanzados (--mode, balance de blancos
        para el NoIR). Si esa version de rpicam-vid no los soporta y el
        proceso falla, reintenta con flags basicos.

        Raises:
            ErrorCaptura: si rpicam-vid no esta disponible o falla incluso
            con flags basicos.
        """
        if self._activa:
            _log.warning("iniciar() llamado pero captura ya activa")
            return

        # Intento 1: con flags avanzados (--mode + balance de blancos NoIR)
        cmd_avanzado = self._construir_comando(con_flags_avanzados=True)
        _log.info("Lanzando captura (flags avanzados): %s", " ".join(cmd_avanzado))

        if not self._intentar_arrancar(cmd_avanzado):
            # Intento 2: flags basicos (compatibilidad con rpicam-vid viejo)
            _log.warning(
                "rpicam-vid fallo con flags avanzados (--mode/--awb). "
                "Reintentando con flags basicos. El tinte rosa del NoIR "
                "podria no corregirse en esta version de rpicam-vid."
            )
            cmd_basico = self._construir_comando(con_flags_avanzados=False)
            _log.info("Lanzando captura (flags basicos): %s", " ".join(cmd_basico))

            if not self._intentar_arrancar(cmd_basico):
                self.detener()
                raise ErrorCaptura(
                    "rpicam-vid no pudo arrancar ni con flags basicos. "
                    "Verificar que la camara CSI este conectada y detectada "
                    "(probar: rpicam-vid --list-cameras)."
                )

        self._activa = True
        self._ts_inicio = time.time()
        self.frames_capturados = 0
        self.frames_fallidos = 0

        _log.info(
            "Captura iniciada: %dx%d (nativo) -> %dx%d (objetivo) @ %d FPS",
            self.CAPTURA_ANCHO, self.CAPTURA_ALTO,
            self.ancho_objetivo, self.alto_objetivo,
            self.fps,
        )

    def detener(self) -> None:
        """
        Detiene la captura y termina el subproceso rpicam-vid.

        Idempotente: llamar multiples veces es seguro.
        """
        if self._proceso is not None:
            try:
                self._proceso.terminate()
                self._proceso.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                _log.warning("rpicam-vid no termino, matando con kill")
                self._proceso.kill()
                self._proceso.wait(timeout=2.0)
            except Exception as e:
                _log.warning("Error deteniendo rpicam-vid: %s", e)
            finally:
                self._proceso = None

        self._activa = False
        self._buffer = b""
        _log.info(
            "Captura detenida. Total: %d frames capturados, %d fallidos, FPS real: %.1f",
            self.frames_capturados, self.frames_fallidos, self.fps_real,
        )

    # ==================================================================
    # LECTURA DE FRAMES
    # ==================================================================

    def leer(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Lee el siguiente frame de la camara.

        Returns:
            (frame_bgr, timestamp):
                - frame_bgr: numpy array (alto_objetivo, ancho_objetivo, 3)
                  en formato BGR, escalado desde la resolucion nativa.
                  None si la lectura fallo.
                - timestamp: time.time() del momento de la lectura.
        """
        ts = time.time()

        if not self.activa:
            return None, ts

        frame_full = self._leer_siguiente_jpeg()
        if frame_full is None:
            self.frames_fallidos += 1
            return None, ts

        # Escalar a resolucion objetivo
        if (
            frame_full.shape[1] != self.ancho_objetivo
            or frame_full.shape[0] != self.alto_objetivo
        ):
            frame_bgr = cv2.resize(
                frame_full,
                (self.ancho_objetivo, self.alto_objetivo),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame_bgr = frame_full

        self.frames_capturados += 1
        self._ts_ultimo_frame = ts
        return frame_bgr, ts

    def _leer_siguiente_jpeg(self) -> Optional[np.ndarray]:
        """
        Lee chunks del pipe hasta encontrar un frame JPEG completo.

        Busca el par de marcadores FFD8 (inicio) y FFD9 (fin) en el
        stream de bytes. Si hay multiples frames en el buffer, toma el
        ultimo (descarta frames viejos para mantener baja latencia).

        Returns:
            Frame BGR decodificado o None si fallo.
        """
        # Leer chunks hasta tener al menos un frame completo
        intentos = 0
        max_intentos = 50  # evitar loop infinito

        while intentos < max_intentos:
            start = self._buffer.find(self._JPEG_START)
            if start != -1:
                end = self._buffer.find(self._JPEG_END, start + 2)
                if end != -1:
                    # Frame completo encontrado
                    jpg_data = self._buffer[start : end + 2]
                    self._buffer = self._buffer[end + 2 :]

                    # Si hay mas frames en el buffer, saltar al ultimo
                    # para mantener baja latencia (descartamos frames viejos)
                    while True:
                        next_start = self._buffer.find(self._JPEG_START)
                        if next_start == -1:
                            break
                        next_end = self._buffer.find(self._JPEG_END, next_start + 2)
                        if next_end == -1:
                            break
                        # Hay otro frame completo mas nuevo: usamos ese
                        jpg_data = self._buffer[next_start : next_end + 2]
                        self._buffer = self._buffer[next_end + 2 :]

                    # Decodificar JPEG
                    frame = cv2.imdecode(
                        np.frombuffer(jpg_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    return frame  # puede ser None si el JPEG esta corrupto

            # Necesitamos mas datos
            try:
                chunk = self._proceso.stdout.read(self.CHUNK_SIZE)
            except Exception:
                return None

            if not chunk:
                # El pipe se cerro
                self._activa = False
                return None

            self._buffer += chunk
            intentos += 1

            # Limitar tamano del buffer (proteccion contra memory leak)
            if len(self._buffer) > self.CHUNK_SIZE * 20:
                # Descartar todo excepto lo ultimo
                self._buffer = self._buffer[-self.CHUNK_SIZE * 5 :]

        _log.warning("No se encontro frame JPEG completo en %d intentos", max_intentos)
        return None

    # ==================================================================
    # CONTEXT MANAGER
    # ==================================================================

    def __enter__(self) -> CapturaVideo:
        self.iniciar()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detener()

    def __repr__(self) -> str:
        estado = "activa" if self.activa else "detenida"
        return (
            f"CapturaVideo({self.CAPTURA_ANCHO}x{self.CAPTURA_ALTO}"
            f"->{self.ancho_objetivo}x{self.alto_objetivo}"
            f"@{self.fps}fps, estado={estado}, "
            f"frames={self.frames_capturados})"
        )


__all__ = ["CapturaVideo", "ErrorCaptura"]
