"""
test_captura_video.py - Tests funcionales de CapturaVideo.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_captura_video

REQUIERE: Pi Camera CSI conectada y rpicam-vid instalado.

Valida:
  1. Iniciar y detener la captura
  2. Lectura de frames con formato correcto
  3. FPS real razonable
  4. Idempotencia (doble inicio, doble cierre)
  5. Context manager
  6. Visualizacion en vivo (5 segundos)
"""

from __future__ import annotations

import sys
import time
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import subprocess

# Verificar que rpicam-vid esta disponible
try:
    resultado = subprocess.run(
        ["rpicam-vid", "--version"],
        capture_output=True,
        timeout=5,
    )
    RPICAM_DISPONIBLE = resultado.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired):
    RPICAM_DISPONIBLE = False

if not RPICAM_DISPONIBLE:
    print("rpicam-vid no disponible. Estos tests requieren hardware real.")
    print("Ejecutar en la Raspberry Pi con la camara CSI conectada.")
    sys.exit(0)

import cv2
import numpy as np

from NeuroDrive_Core.config_loader import cargar_config, limpiar_cache
from NeuroDrive_Vision.captura_video import CapturaVideo, ErrorCaptura


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
# TESTS
# =============================================================================

print("\n--- Tests de CapturaVideo ---")


@_test("Iniciar y detener la captura funciona")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaVideo(config)
    cap.iniciar()
    assert cap.activa, "captura deberia estar activa"
    cap.detener()
    assert not cap.activa, "captura deberia estar detenida"


@_test("Leer frame devuelve formato correcto (alto, ancho, 3) BGR")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaVideo(config)
    cap.iniciar()
    try:
        frame, ts = cap.leer()
        assert frame is not None, "frame no deberia ser None"
        assert frame.shape == (480, 640, 3), f"forma incorrecta: {frame.shape}"
        assert frame.dtype == np.uint8, f"dtype incorrecto: {frame.dtype}"
        assert ts > 0, "timestamp deberia ser positivo"
    finally:
        cap.detener()


@_test("Multiples frames se leen en secuencia")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaVideo(config)
    cap.iniciar()
    try:
        frames_ok = 0
        for _ in range(15):
            frame, ts = cap.leer()
            if frame is not None:
                frames_ok += 1

        assert frames_ok >= 12, f"solo {frames_ok}/15 frames ok (esperaba >= 12)"
        assert cap.frames_capturados >= 12
    finally:
        cap.detener()


@_test("FPS real es razonable (> 5 y < 60)")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaVideo(config)
    cap.iniciar()
    try:
        for _ in range(30):
            cap.leer()

        fps = cap.fps_real
        assert 5.0 < fps < 60.0, f"FPS irrazonable: {fps:.1f}"
        print(f"    (FPS real medido: {fps:.1f})")
    finally:
        cap.detener()


@_test("Detener es idempotente (doble cierre no rompe)")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaVideo(config)
    cap.iniciar()
    cap.detener()
    cap.detener()  # no deberia romper
    assert not cap.activa


@_test("Iniciar es idempotente (doble inicio no rompe)")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaVideo(config)
    cap.iniciar()
    cap.iniciar()  # warning, no error
    assert cap.activa
    cap.detener()


@_test("Context manager funciona correctamente")
def _():
    limpiar_cache()
    config = cargar_config()
    with CapturaVideo(config) as cap:
        assert cap.activa
        frame, ts = cap.leer()
        assert frame is not None
    assert not cap.activa


@_test("Leer despues de detener devuelve None")
def _():
    limpiar_cache()
    config = cargar_config()
    cap = CapturaVideo(config)
    cap.iniciar()
    cap.detener()
    frame, ts = cap.leer()
    assert frame is None


@_test("Visualizacion en vivo (5 segundos)")
def _():
    """Muestra video en vivo con overlay de FPS y forma del frame.
    Verifica que la camara produce frames reales."""
    limpiar_cache()
    config = cargar_config()
    cap = CapturaVideo(config)
    cap.iniciar()

    frames_totales = 0
    ts_inicio = time.time()
    duracion = 5.0  # segundos

    print(f"    Mostrando video en vivo por {duracion}s...")
    print(f"    (Si no ves ventana, cierra con Ctrl+C)")

    try:
        while time.time() - ts_inicio < duracion:
            frame, ts = cap.leer()
            if frame is None:
                continue

            frames_totales += 1

            # Overlay informativo
            fps_txt = f"FPS: {cap.fps_real:.1f}"
            shape_txt = f"Shape: {frame.shape}"
            frames_txt = f"Frames: {frames_totales}"

            cv2.putText(frame, fps_txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, shape_txt, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, frames_txt, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("NeuroDrive - Test Captura", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        cap.detener()

    assert frames_totales >= 30, (
        f"solo {frames_totales} frames en {duracion}s (esperaba >= 30)"
    )
    print(f"    Total: {frames_totales} frames en {duracion}s")


# =============================================================================
# RESUMEN
# =============================================================================

def _resumen() -> int:
    total = len(_resultados)
    pasaron = sum(1 for _, ok, _ in _resultados if ok)
    fallaron = total - pasaron

    print("\n" + "=" * 60)
    print(f"  Resumen: {pasaron}/{total} tests pasaron")
    if fallaron:
        print(f"  Fallaron {fallaron}:")
        for nombre, ok, err in _resultados:
            if not ok:
                print(f"    - {nombre}: {err}")
    print("=" * 60)
    return 0 if fallaron == 0 else 1


if __name__ == "__main__":
    print("=" * 60)
    print("  NeuroDrive - Tests de Captura de Video")
    print("=" * 60)
    sys.exit(_resumen())
