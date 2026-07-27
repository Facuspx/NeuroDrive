"""
test_captura_archivo.py - Tests de CapturaArchivo.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_captura_archivo

Los tests generan un video sintetico temporal con OpenCV, asi no
requieren un archivo grabado a mano. Corren en la Pi Y en cualquier PC
con OpenCV.

Valida:
  1. Apertura y cierre del archivo.
  2. Lectura de frames con formato correcto.
  3. Fin de archivo detectado limpiamente.
  4. Modo timestamp video: rapido, ts sinteticos.
  5. Modo timestamp tiempo real: respeta el ritmo.
  6. Escalado a resolucion objetivo.
  7. Loop reinicia el video.
  8. Manejo de archivo inexistente.
  9. Context manager.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np

from NeuroDrive_Core.config_loader import cargar_config, limpiar_cache
from NeuroDrive_Vision.captura_archivo import CapturaArchivo
from NeuroDrive_Vision.captura_video import ErrorCaptura


# =============================================================================
# Framework
# =============================================================================

_resultados: list[tuple[str, bool, str]] = []


def _test(nombre: str):
    def wrapper(func):
        try:
            func()
            _resultados.append((nombre, True, ""))
            print(f"  [OK]  {nombre}")
        except AssertionError as e:
            _resultados.append((nombre, False, str(e)))
            print(f"  [FAIL] {nombre}: {e}")
        except Exception as e:
            _resultados.append((nombre, False, f"{type(e).__name__}: {e}"))
            print(f"  [ERROR] {nombre}: {e}")
            traceback.print_exc()
        return func
    return wrapper


# =============================================================================
# Helper: generar un video sintetico para los tests
# =============================================================================

def _generar_video_sintetico(
    ruta: str,
    n_frames: int = 30,
    fps: float = 15.0,
    ancho: int = 640,
    alto: int = 480,
) -> str:
    """
    Genera un video MJPEG en 'ruta' con n_frames.

    Cada frame lleva un patron distinto para poder identificarlo si hace
    falta (rectangulo que se mueve horizontalmente, y el numero de frame
    codificado en el pixel (0,0) del canal azul).
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(ruta, fourcc, fps, (ancho, alto))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de prueba en {ruta}")

    try:
        for i in range(n_frames):
            frame = np.full((alto, ancho, 3), 60, dtype=np.uint8)
            # Rectangulo que se mueve horizontalmente
            x = int((i / max(1, n_frames - 1)) * (ancho - 100))
            cv2.rectangle(frame, (x, 100), (x + 80, 300), (0, 200, 0), -1)
            # Codigo del frame en el pixel (0,0), canal azul (indice 0 en BGR)
            frame[0, 0, 0] = i % 256
            writer.write(frame)
    finally:
        writer.release()

    return ruta


# =============================================================================
# Setup
# =============================================================================

print("\n--- Tests de CapturaArchivo ---")

_tmpdir = tempfile.mkdtemp(prefix="neurodrive_test_")
# AVI en lugar de raw MJPEG: el contenedor AVI guarda FPS en metadatos,
# el raw MJPEG no. En raw MJPEG OpenCV devuelve 25 FPS por default y
# eso rompe los tests que asumen FPS conocido. En produccion, si graban
# con rpicam-vid a .mjpeg, deben pasar fps_override; hay un test dedicado.
_ruta_video = os.path.join(_tmpdir, "test.avi")
_generar_video_sintetico(_ruta_video, n_frames=30, fps=15.0)


# =============================================================================
# TESTS
# =============================================================================

@_test("Archivo inexistente lanza ErrorCaptura con mensaje util")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta="/no/existe/nada.mjpeg")
    try:
        cap.iniciar()
        raise AssertionError("deberia haber lanzado ErrorCaptura")
    except ErrorCaptura as e:
        msg = str(e)
        assert "no existe" in msg.lower(), f"mensaje poco claro: {msg}"


@_test("Iniciar abre el video y lee metadatos")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video)
    cap.iniciar()
    try:
        assert cap.activa
        assert abs(cap.fps_video - 15.0) < 0.1, f"fps={cap.fps_video}"
        assert cap.total_frames_video == 30 or cap.total_frames_video >= 28, (
            f"total_frames={cap.total_frames_video} (esperaba ~30)"
        )
        assert cap.duracion_video_seg > 0
    finally:
        cap.detener()
    assert not cap.activa


@_test("Leer devuelve frame con formato correcto")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video)
    cap.iniciar()
    try:
        frame, ts = cap.leer()
        assert frame is not None
        assert frame.shape == (480, 640, 3), f"forma: {frame.shape}"
        assert frame.dtype == np.uint8
        assert ts > 0
    finally:
        cap.detener()


@_test("Multiples frames se leen en secuencia")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video)
    cap.iniciar()
    try:
        frames_ok = 0
        for _ in range(20):
            frame, _ = cap.leer()
            if frame is not None:
                frames_ok += 1
        assert frames_ok == 20, f"solo {frames_ok}/20 frames leidos"
        assert cap.frames_capturados == 20
    finally:
        cap.detener()


@_test("Fin de archivo devuelve None y marca fin_de_archivo")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video)
    cap.iniciar()
    try:
        # Leer todos los frames + uno extra
        contador = 0
        while cap.activa and contador < 100:
            frame, _ = cap.leer()
            if frame is None:
                break
            contador += 1

        # Al menos habre leido los 30 frames (puede que 29 por dropped)
        assert contador >= 28, f"solo {contador} frames antes de EOF"
        assert cap.fin_de_archivo, "fin_de_archivo deberia ser True"
        # activa debe ser False (fin_de_archivo pone activa=False)
        assert not cap.activa, "activa deberia ser False tras EOF"

        # Otra lectura tambien devuelve None
        frame, _ = cap.leer()
        assert frame is None
    finally:
        cap.detener()


@_test("Modo VIDEO: timestamps siguen el FPS del video")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video,
                          modo_timestamp=CapturaArchivo.MODO_VIDEO)
    cap.iniciar()
    try:
        _, ts1 = cap.leer()
        _, ts2 = cap.leer()
        _, ts3 = cap.leer()
        # 3 frames a 15 FPS: dt entre consecutivos = 1/15 = 0.0667s
        dt = ts2 - ts1
        assert abs(dt - 1/15) < 0.01, f"dt={dt:.4f}s (esperaba ~0.0667s)"
        dt2 = ts3 - ts2
        assert abs(dt2 - 1/15) < 0.01
    finally:
        cap.detener()


@_test("Modo VIDEO: procesamiento rapido (sin sleep)")
def _():
    """Leer 20 frames en modo VIDEO debe tardar mucho menos que 20/15s reales."""
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video,
                          modo_timestamp=CapturaArchivo.MODO_VIDEO)
    cap.iniciar()
    try:
        t0 = time.time()
        for _ in range(20):
            frame, _ = cap.leer()
            if frame is None:
                break
        elapsed = time.time() - t0
        # 20 frames a 15 FPS son 1.33s reales. En modo video debe ser mucho menos.
        assert elapsed < 0.5, (
            f"modo VIDEO tardo {elapsed:.2f}s en 20 frames (deberia ser <0.5s)"
        )
    finally:
        cap.detener()


@_test("Modo TIEMPO_REAL: respeta el ritmo del video (con tolerancia)")
def _():
    """Leer N frames en modo TIEMPO_REAL debe tardar aprox N/fps segundos."""
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video,
                          modo_timestamp=CapturaArchivo.MODO_TIEMPO_REAL)
    cap.iniciar()
    try:
        N = 10
        t0 = time.time()
        for _ in range(N):
            frame, _ = cap.leer()
            if frame is None:
                break
        elapsed = time.time() - t0
        # 10 frames a 15 FPS = 0.666s ideal. Toleramos +-30% por overhead.
        esperado = N / 15.0
        assert 0.7 * esperado <= elapsed <= 1.5 * esperado, (
            f"tiempo real {elapsed:.2f}s fuera de rango "
            f"({0.7*esperado:.2f}-{1.5*esperado:.2f}s)"
        )
    finally:
        cap.detener()


@_test("Escalado a resolucion objetivo funciona si el archivo tiene otra")
def _():
    """Un video 320x240 debe escalarse a la resolucion objetivo (640x480)."""
    limpiar_cache()
    config = cargar_config()
    ruta_chico = os.path.join(_tmpdir, "chico.avi")
    _generar_video_sintetico(ruta_chico, n_frames=10, ancho=320, alto=240)

    cap = CapturaArchivo(config, ruta=ruta_chico)
    cap.iniciar()
    try:
        frame, _ = cap.leer()
        assert frame is not None
        # El pipeline pide 640x480; el video es 320x240; debe escalarse.
        assert frame.shape == (480, 640, 3), f"no se escalo: {frame.shape}"
    finally:
        cap.detener()
    os.remove(ruta_chico)


@_test("Loop reinicia el video al llegar al final")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video, loop=True,
                          modo_timestamp=CapturaArchivo.MODO_VIDEO)
    cap.iniciar()
    try:
        # Leer 50 frames: si el video tiene ~30, deberia hacer al menos un loop
        contador = 0
        for _ in range(50):
            frame, _ = cap.leer()
            if frame is None:
                break
            contador += 1
        assert contador == 50, f"con loop deberia leer 50, leyo {contador}"
        assert cap._loops_completados >= 1
    finally:
        cap.detener()


@_test("Context manager funciona")
def _():
    limpiar_cache()
    config = cargar_config()
    with CapturaArchivo(config, ruta=_ruta_video) as cap:
        assert cap.activa
        frame, _ = cap.leer()
        assert frame is not None
    assert not cap.activa


@_test("Detener es idempotente")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video)
    cap.iniciar()
    cap.detener()
    cap.detener()  # no debe romper
    assert not cap.activa


@_test("Iniciar es idempotente (segundo iniciar es warning)")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaArchivo(config, ruta=_ruta_video)
    cap.iniciar()
    cap.iniciar()  # warning, no error
    assert cap.activa
    cap.detener()


@_test("modo_timestamp invalido lanza ValueError")
def _():
    limpiar_cache()
    config = cargar_config()
    try:
        CapturaArchivo(config, ruta=_ruta_video, modo_timestamp="cualquiera")
        raise AssertionError("deberia haber fallado")
    except ValueError:
        pass


@_test("fps_override sobreescribe el FPS reportado por metadatos")
def _():
    """
    Escenario tipico: el video reporta 15 FPS por metadatos pero el
    usuario forza otro con fps_override. Los timestamps del modo VIDEO
    deben calcularse con el fps_override.
    """
    limpiar_cache()
    config = cargar_config()
    # Forzamos fps_override=30, dobla el FPS real del video (15).
    # Timestamps deberian espaciarse a 1/30, no a 1/15.
    cap = CapturaArchivo(config, ruta=_ruta_video,
                          modo_timestamp=CapturaArchivo.MODO_VIDEO,
                          fps_override=30.0)
    cap.iniciar()
    try:
        assert cap.fps_video == 30.0, f"fps_video={cap.fps_video}"
        _, ts1 = cap.leer()
        _, ts2 = cap.leer()
        dt = ts2 - ts1
        assert abs(dt - 1/30) < 0.005, (
            f"con fps_override=30, dt esperado ~0.033s, dio {dt:.4f}s"
        )
    finally:
        cap.detener()


@_test("fps_override invalido lanza ValueError")
def _():
    limpiar_cache()
    config = cargar_config()
    try:
        CapturaArchivo(config, ruta=_ruta_video, fps_override=-5.0)
        raise AssertionError("deberia haber fallado con fps negativo")
    except ValueError:
        pass
    try:
        CapturaArchivo(config, ruta=_ruta_video, fps_override=0.0)
        raise AssertionError("deberia haber fallado con fps 0")
    except ValueError:
        pass


# =============================================================================
# Cleanup y resumen
# =============================================================================

try:
    os.remove(_ruta_video)
    os.rmdir(_tmpdir)
except OSError:
    pass

print("\n--- Resumen ---")
exitos = sum(1 for _, ok, _ in _resultados if ok)
total = len(_resultados)
print(f"Tests pasados: {exitos}/{total}")
if exitos < total:
    print("\nFALLAS:")
    for nombre, ok, msg in _resultados:
        if not ok:
            print(f"  [FAIL] {nombre}: {msg}")
    sys.exit(1)
print("\nTodos los tests pasaron.")
