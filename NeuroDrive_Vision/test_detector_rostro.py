"""
test_detector_rostro.py - Tests funcionales de DetectorRostro.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_detector_rostro

REQUIERE: Pi Camera CSI conectada y MediaPipe instalado.

Valida:
  1. Iniciar y detener el detector funciona
  2. Procesar un frame sin rostro NO tira excepcion (devuelve presente=False)
  3. Procesar frames REALES (agrupado): detecta rostro + 468 landmarks +
     pixeles en el frame + normalizados en [0,1]
  4. Tiempo de procesamiento es razonable (< 150ms en Pi 5)
  5. Validacion de input: None, sin iniciar, shape incorrecta
  6. Idempotencia (doble inicio, doble cierre)
  7. Context manager funciona
  8. Visualizacion en vivo (5 segundos) con landmarks dibujados
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

# Verificar que rpicam-vid esta disponible (necesario para CapturaVideo)
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
from NeuroDrive_Vision.captura_video import CapturaVideo
from NeuroDrive_Vision.detector_rostro import (
    DetectorRostro,
    DatosRostro,
    ErrorDetectorRostro,
)


# =============================================================================
# Framework de tests (mismo patron que test_captura_video.py)
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


def _capturar_n_frames(n: int = 10, descartar: int = 10) -> list[np.ndarray]:
    """
    Helper: abre la camara UNA sola vez, descarta los primeros frames para
    que el AE/AWB se estabilice, y devuelve n frames consecutivos.

    Usar para tests que necesitan capturar rostro real. Es mucho mas confiable
    que abrir y cerrar la camara en cada llamada (cada apertura resetea el AE
    y los primeros frames vienen oscuros).
    """
    config = cargar_config()
    cap = CapturaVideo(config)
    cap.iniciar()
    frames: list[np.ndarray] = []
    try:
        # Warmup: descartamos para estabilizar AE/AWB
        for _ in range(descartar):
            cap.leer()
        # Captura util
        intentos = 0
        while len(frames) < n and intentos < n * 3:
            frame, _ = cap.leer()
            intentos += 1
            if frame is not None:
                frames.append(frame)
        if len(frames) == 0:
            raise RuntimeError("No se pudo capturar ningun frame")
        return frames
    finally:
        cap.detener()


# =============================================================================
# TESTS
# =============================================================================

print("\n--- Tests de DetectorRostro ---")


@_test("Iniciar y detener el detector funciona")
def _():
    det = DetectorRostro()
    assert not det.activo, "no deberia estar activo antes de iniciar"
    det.iniciar()
    assert det.activo, "deberia estar activo despues de iniciar"
    det.detener()
    assert not det.activo, "no deberia estar activo despues de detener"


@_test("Procesar frame negro (sin rostro) NO lanza excepcion")
def _():
    det = DetectorRostro()
    det.iniciar()
    try:
        # Frame totalmente negro: no hay rostro
        frame_negro = np.zeros((480, 640, 3), dtype=np.uint8)
        datos = det.procesar(frame_negro)
        assert isinstance(datos, DatosRostro), f"tipo incorrecto: {type(datos)}"
        assert datos.rostro_presente is False, "no deberia detectar rostro en frame negro"
        assert datos.puntos_pixeles is None
        assert datos.puntos_normalizados is None
        assert datos.resolucion == (640, 480)
    finally:
        det.detener()


@_test("Procesar frames REALES: detecta rostro, 468 landmarks, pixeles en frame, normalizados en [0,1]")
def _():
    """
    Test agrupado: abre la camara UNA sola vez, captura 15 frames calentados,
    los procesa todos y valida que al menos uno tenga rostro con landmarks
    bien formados.

    Combina los tres tests que antes abrian camara cada uno (lo cual hacia
    que el AE/AWB no se estabilizara y el primer frame viniera oscuro).
    """
    limpiar_cache()
    frames = _capturar_n_frames(n=15, descartar=15)
    det = DetectorRostro()
    det.iniciar()
    try:
        # Procesamos TODOS los frames y nos quedamos con el primero
        # que tenga rostro detectado.
        datos_ok = None
        for f in frames:
            d = det.procesar(f)
            if d.rostro_presente:
                datos_ok = d
                break

        assert datos_ok is not None, (
            f"no se detecto rostro en ninguno de los {len(frames)} frames. "
            "Asegurate de estar mirando a la camara durante el test."
        )

        # 1) Shapes correctas
        assert datos_ok.puntos_pixeles is not None
        assert datos_ok.puntos_normalizados is not None
        assert datos_ok.puntos_pixeles.shape == (468, 2), (
            f"shape pixeles incorrecto: {datos_ok.puntos_pixeles.shape}"
        )
        assert datos_ok.puntos_normalizados.shape == (468, 3), (
            f"shape normalizados incorrecto: {datos_ok.puntos_normalizados.shape}"
        )

        # 2) Pixeles ESTRICTAMENTE dentro del frame (porque ahora hacemos clamp)
        ancho, alto = datos_ok.resolucion
        xs = datos_ok.puntos_pixeles[:, 0]
        ys = datos_ok.puntos_pixeles[:, 1]
        assert (xs >= 0).all() and (xs <= ancho - 1).all(), (
            f"x fuera de rango: min={xs.min()}, max={xs.max()}, ancho={ancho}"
        )
        assert (ys >= 0).all() and (ys <= alto - 1).all(), (
            f"y fuera de rango: min={ys.min()}, max={ys.max()}, alto={alto}"
        )

        # 3) Normalizados ESTRICTAMENTE en [0, 1] (clamp interno los garantiza)
        xs_n = datos_ok.puntos_normalizados[:, 0]
        ys_n = datos_ok.puntos_normalizados[:, 1]
        assert (xs_n >= 0.0).all() and (xs_n <= 1.0).all(), (
            f"x_norm fuera de [0,1]: min={xs_n.min():.3f}, max={xs_n.max():.3f}"
        )
        assert (ys_n >= 0.0).all() and (ys_n <= 1.0).all(), (
            f"y_norm fuera de [0,1]: min={ys_n.min():.3f}, max={ys_n.max():.3f}"
        )
        # z puede ser positivo o negativo, no validamos rango
    finally:
        det.detener()


@_test("Tiempo de procesamiento razonable (< 150ms)")
def _():
    limpiar_cache()
    frames = _capturar_n_frames(n=10, descartar=10)
    det = DetectorRostro()
    det.iniciar()
    try:
        # Procesamos los 10 frames calentados
        tiempos = []
        for f in frames:
            datos = det.procesar(f)
            tiempos.append(datos.tiempo_procesamiento_ms)
        # Descartamos los 2 primeros (warm-up de MediaPipe)
        promedio = sum(tiempos[2:]) / len(tiempos[2:])
        print(f"    (tiempo promedio: {promedio:.1f}ms)")
        assert promedio < 150.0, f"tiempo promedio muy alto: {promedio:.1f}ms"
    finally:
        det.detener()


@_test("Procesar None lanza ErrorDetectorRostro")
def _():
    det = DetectorRostro()
    det.iniciar()
    try:
        try:
            det.procesar(None)  # type: ignore
            raise AssertionError("deberia haber lanzado ErrorDetectorRostro")
        except ErrorDetectorRostro:
            pass
    finally:
        det.detener()


@_test("Procesar frame sin iniciar lanza ErrorDetectorRostro")
def _():
    det = DetectorRostro()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        det.procesar(frame)
        raise AssertionError("deberia haber lanzado ErrorDetectorRostro")
    except ErrorDetectorRostro:
        pass


@_test("Procesar frame con shape incorrecta lanza ErrorDetectorRostro")
def _():
    det = DetectorRostro()
    det.iniciar()
    try:
        # Grayscale en vez de BGR
        frame_malo = np.zeros((480, 640), dtype=np.uint8)
        try:
            det.procesar(frame_malo)
            raise AssertionError("deberia haber lanzado ErrorDetectorRostro")
        except ErrorDetectorRostro:
            pass
    finally:
        det.detener()


@_test("Iniciar es idempotente (doble inicio no rompe)")
def _():
    det = DetectorRostro()
    det.iniciar()
    det.iniciar()  # warning, no error
    assert det.activo
    det.detener()


@_test("Detener es idempotente (doble cierre no rompe)")
def _():
    det = DetectorRostro()
    det.iniciar()
    det.detener()
    det.detener()  # no deberia romper
    assert not det.activo


@_test("Context manager funciona correctamente")
def _():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with DetectorRostro() as det:
        assert det.activo
        datos = det.procesar(frame)
        assert isinstance(datos, DatosRostro)
    assert not det.activo


@_test("Visualizacion en vivo con landmarks (5 segundos)")
def _():
    """Muestra video con los 468 landmarks dibujados encima.
    Verifica que el sistema completo (captura + deteccion) funciona."""
    limpiar_cache()
    config = cargar_config()

    cap = CapturaVideo(config)
    det = DetectorRostro()
    cap.iniciar()
    det.iniciar()

    frames_totales = 0
    frames_con_rostro = 0
    tiempos = []
    ts_inicio = time.time()
    duracion = 5.0

    print(f"    Mostrando video con landmarks por {duracion}s...")
    print(f"    (Mira a la camara. Cierra con 'q'.)")

    try:
        while time.time() - ts_inicio < duracion:
            frame, _ = cap.leer()
            if frame is None:
                continue

            frames_totales += 1
            datos = det.procesar(frame)
            tiempos.append(datos.tiempo_procesamiento_ms)

            if datos.rostro_presente:
                frames_con_rostro += 1
                frame_viz = DetectorRostro.dibujar_landmarks(frame, datos, color=(0, 255, 0), radio=1)
            else:
                frame_viz = frame.copy()

            # Overlay
            estado = "ROSTRO OK" if datos.rostro_presente else "SIN ROSTRO"
            color = (0, 255, 0) if datos.rostro_presente else (0, 0, 255)
            cv2.putText(frame_viz, estado, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame_viz, f"t: {datos.tiempo_procesamiento_ms:.1f}ms",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_viz, f"FPS cap: {cap.fps_real:.1f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("NeuroDrive - Test Detector Rostro", frame_viz)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        det.detener()
        cap.detener()

    promedio_ms = sum(tiempos) / len(tiempos) if tiempos else 0.0
    tasa = (frames_con_rostro / frames_totales * 100) if frames_totales > 0 else 0.0
    print(f"    Frames totales: {frames_totales}")
    print(f"    Frames con rostro: {frames_con_rostro} ({tasa:.1f}%)")
    print(f"    Tiempo medio de deteccion: {promedio_ms:.1f}ms")
    assert frames_totales > 30, f"muy pocos frames procesados: {frames_totales}"
    assert tasa > 50.0, (
        f"tasa de deteccion baja: {tasa:.1f}% "
        "(asegurate de estar mirando a la camara)"
    )


# =============================================================================
# Resumen
# =============================================================================

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
sys.exit(0)
