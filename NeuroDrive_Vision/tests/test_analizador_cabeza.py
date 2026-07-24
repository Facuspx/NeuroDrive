"""
test_analizador_cabeza.py - Tests funcionales de AnalizadorCabeza.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_analizador_cabeza

Valida:
  Tests sin hardware (12):
   1. Construccion con parametros default
   2. Construccion con alpha_ema invalido falla
   3. Procesar DatosRostro con rostro_presente=False -> invalido
   4. Procesar DatosRostro con puntos=None -> invalido
   5. Procesar con landmarks sinteticos frontales -> angulos ~0
   6. Procesar con landmarks rotados +30deg yaw -> yaw_deg detecta giro
   7. Procesar con landmarks rotados +20deg pitch (cabeceo) -> pitch detecta
   8. Reset limpia el estado del filtro EMA
   9. Resolucion invalida -> invalido
  10. La matriz de camara se construye con focal=ancho por default
  11. Filtro EMA reduce jitter en valores oscilantes
  12. Detecta metodo PnP disponible

  Tests con hardware (3):
  13. Pipeline completo: captura -> detector -> analizador (rostro neutro)
  14. Tiempo de procesamiento < 5 ms
  15. Visualizacion en vivo con ejes 3D (5 segundos)
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

# Detectar si tenemos hardware
try:
    resultado = subprocess.run(
        ["rpicam-vid", "--version"], capture_output=True, timeout=5,
    )
    RPICAM_DISPONIBLE = resultado.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired):
    RPICAM_DISPONIBLE = False

import cv2
import numpy as np

from NeuroDrive_Vision.detector_rostro import DatosRostro, DetectorRostro
from NeuroDrive_Vision.analizador_cabeza import (
    AnalizadorCabeza,
    DatosCabeza,
    INDICES_PNP,
    MODELO_3D_MM,
)

if RPICAM_DISPONIBLE:
    from NeuroDrive_Core.config_loader import cargar_config, limpiar_cache
    from NeuroDrive_Vision.captura_video import CapturaVideo


# =============================================================================
# Framework de tests
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
# Helper: generar landmarks sinteticos desde el modelo 3D
# =============================================================================

def _generar_landmarks_sinteticos(
    ancho: int = 640,
    alto: int = 480,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
    distancia_mm: float = 500.0,
) -> DatosRostro:
    """
    Genera un DatosRostro sintetico con un rostro virtual rotado
    los angulos especificados. Util para validar el analizador sin hardware.

    Toma el modelo 3D del rostro, lo rota, lo traslada (alejandolo de la
    camara) y lo proyecta usando la misma matriz aproximada que usaria
    el analizador. Luego pone los pixeles resultantes en los indices
    correspondientes de un array de 468 landmarks (los demas en 0).
    """
    # Matriz de rotacion desde angulos de Euler XYZ
    # OJO: usamos la misma convencion que cv2.RQDecomp3x3 para que sea
    # consistente con el analizador.
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)
    roll_rad = np.deg2rad(roll_deg)

    # Rx (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)],
    ])
    # Ry (yaw)
    Ry = np.array([
        [ np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [ 0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)],
    ])
    # Rz (roll)
    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad),  np.cos(roll_rad), 0],
        [0, 0, 1],
    ])

    # Composicion XYZ
    R = Rz @ Ry @ Rx

    # Rotar el modelo 3D
    modelo_rotado = (R @ MODELO_3D_MM.T).T

    # Trasladar al frente de la camara (z positivo, alejando)
    modelo_trasladado = modelo_rotado + np.array([0, 0, distancia_mm])

    # Proyeccion con la misma matriz que el analizador (focal=ancho, centro=medio)
    focal = float(ancho)
    cx = ancho / 2.0
    cy = alto / 2.0

    pixeles_2d = np.zeros((6, 2), dtype=np.float32)
    for i in range(6):
        x, y, z = modelo_trasladado[i]
        # NOTA: OpenCV usa Y hacia abajo en imagen, pero nuestro modelo
        # tiene Y hacia arriba. Por eso aplicamos -y en la proyeccion.
        pixeles_2d[i, 0] = focal * x / z + cx
        pixeles_2d[i, 1] = focal * (-y) / z + cy

    # Construir el array de 468 landmarks (los demas no nos importan)
    puntos_pixeles = np.zeros((468, 2), dtype=np.int32)
    puntos_norm = np.zeros((468, 3), dtype=np.float32)
    for i, idx_lm in enumerate(INDICES_PNP):
        puntos_pixeles[idx_lm, 0] = int(pixeles_2d[i, 0])
        puntos_pixeles[idx_lm, 1] = int(pixeles_2d[i, 1])
        puntos_norm[idx_lm, 0] = pixeles_2d[i, 0] / ancho
        puntos_norm[idx_lm, 1] = pixeles_2d[i, 1] / alto

    return DatosRostro(
        rostro_presente=True,
        puntos_pixeles=puntos_pixeles,
        puntos_normalizados=puntos_norm,
        resolucion=(ancho, alto),
        timestamp=time.monotonic(),
        tiempo_procesamiento_ms=1.0,
    )


# =============================================================================
# TESTS SIN HARDWARE
# =============================================================================

print("\n--- Tests de AnalizadorCabeza (sin hardware) ---")


@_test("Construccion con parametros default")
def _():
    a = AnalizadorCabeza()
    assert a.alpha_ema == 0.5
    assert a._matriz_camara is None  # se construye lazy


@_test("Construccion con alpha_ema invalido falla")
def _():
    try:
        AnalizadorCabeza(alpha_ema=1.5)
        raise AssertionError("deberia haber lanzado ValueError")
    except ValueError:
        pass
    try:
        AnalizadorCabeza(alpha_ema=0.0)
        raise AssertionError("deberia haber lanzado ValueError con alpha=0")
    except ValueError:
        pass


@_test("Procesar DatosRostro sin rostro -> invalido")
def _():
    a = AnalizadorCabeza()
    datos_vacios = DatosRostro(
        rostro_presente=False,
        resolucion=(640, 480),
    )
    res = a.procesar(datos_vacios)
    assert isinstance(res, DatosCabeza)
    assert res.valido is False
    assert "no presente" in res.motivo_invalido.lower()


@_test("Procesar DatosRostro con puntos=None -> invalido")
def _():
    a = AnalizadorCabeza()
    datos_malos = DatosRostro(
        rostro_presente=True,
        puntos_pixeles=None,
        puntos_normalizados=None,
        resolucion=(640, 480),
    )
    res = a.procesar(datos_malos)
    assert res.valido is False
    assert "puntos_pixeles" in res.motivo_invalido


@_test("Rostro frontal (sin rotacion) -> angulos cercanos a 0")
def _():
    a = AnalizadorCabeza(alpha_ema=1.0)  # sin filtrado para test deterministico
    datos = _generar_landmarks_sinteticos(pitch_deg=0, yaw_deg=0, roll_deg=0)
    res = a.procesar(datos)
    assert res.valido, f"deberia ser valido, motivo: {res.motivo_invalido}"
    # Tolerancia: el modelo 3D tiene Y hacia arriba pero la proyeccion
    # invierte Y, y la conversion a Euler puede introducir un offset
    # de unos pocos grados. Lo importante: los 3 angulos son chicos.
    assert abs(res.pitch_crudo) < 10.0, f"pitch={res.pitch_crudo}"
    assert abs(res.yaw_crudo) < 10.0, f"yaw={res.yaw_crudo}"
    assert abs(res.roll_crudo) < 10.0, f"roll={res.roll_crudo}"


@_test("Rotacion yaw +30deg -> yaw_deg detecta giro")
def _():
    a = AnalizadorCabeza(alpha_ema=1.0)
    datos = _generar_landmarks_sinteticos(yaw_deg=30.0)
    res = a.procesar(datos)
    assert res.valido, f"motivo: {res.motivo_invalido}"
    # Tolerancia de 10 grados (el signo puede invertirse segun convencion
    # interna de OpenCV, asi que validamos solo la magnitud)
    assert abs(abs(res.yaw_crudo) - 30.0) < 10.0, (
        f"yaw esperado ~30, obtenido {res.yaw_crudo:.1f}"
    )
    # Los otros angulos deben ser chicos
    assert abs(res.pitch_crudo) < 15.0, f"pitch deberia ser chico, es {res.pitch_crudo}"
    assert abs(res.roll_crudo) < 15.0, f"roll deberia ser chico, es {res.roll_crudo}"


@_test("Rotacion pitch +20deg (cabeceo) -> pitch detecta inclinacion")
def _():
    a = AnalizadorCabeza(alpha_ema=1.0)
    datos = _generar_landmarks_sinteticos(pitch_deg=20.0)
    res = a.procesar(datos)
    assert res.valido, f"motivo: {res.motivo_invalido}"
    assert abs(abs(res.pitch_crudo) - 20.0) < 10.0, (
        f"pitch esperado ~20, obtenido {res.pitch_crudo:.1f}"
    )
    assert abs(res.yaw_crudo) < 15.0
    assert abs(res.roll_crudo) < 15.0


@_test("Reset limpia el estado del filtro EMA")
def _():
    a = AnalizadorCabeza()
    datos = _generar_landmarks_sinteticos(yaw_deg=30.0)
    # Procesamos varios frames para llenar el filtro
    for _ in range(5):
        a.procesar(datos)
    assert a._pitch_filtrado is not None
    a.reset()
    assert a._pitch_filtrado is None
    assert a._yaw_filtrado is None
    assert a._roll_filtrado is None


@_test("Resolucion invalida -> invalido")
def _():
    a = AnalizadorCabeza()
    datos = DatosRostro(
        rostro_presente=True,
        puntos_pixeles=np.zeros((468, 2), dtype=np.int32),
        puntos_normalizados=np.zeros((468, 3), dtype=np.float32),
        resolucion=(0, 0),
    )
    res = a.procesar(datos)
    assert res.valido is False
    assert "resolucion" in res.motivo_invalido.lower()


@_test("Matriz de camara se construye con focal=ancho por default")
def _():
    a = AnalizadorCabeza()
    datos = _generar_landmarks_sinteticos(ancho=800, alto=600)
    a.procesar(datos)
    assert a._matriz_camara is not None
    # focal en (0,0) y (1,1) debe ser ancho=800
    assert a._matriz_camara[0, 0] == 800.0, f"fx={a._matriz_camara[0, 0]}"
    assert a._matriz_camara[1, 1] == 800.0, f"fy={a._matriz_camara[1, 1]}"
    # centro en (0,2) y (1,2) debe ser ancho/2 y alto/2
    assert a._matriz_camara[0, 2] == 400.0
    assert a._matriz_camara[1, 2] == 300.0


@_test("Filtro EMA reduce jitter")
def _():
    a = AnalizadorCabeza(alpha_ema=0.3)  # filtrado moderado
    # Alternamos entre dos valores muy distintos
    datos_pos = _generar_landmarks_sinteticos(yaw_deg=30.0)
    datos_neg = _generar_landmarks_sinteticos(yaw_deg=-30.0)

    yaws_filtrados = []
    for i in range(10):
        d = datos_pos if i % 2 == 0 else datos_neg
        res = a.procesar(d)
        yaws_filtrados.append(res.yaw_deg)

    # El primer valor del filtro es el valor crudo (init).
    # Despues, los valores deberian oscilar pero con AMPLITUD MENOR a 30.
    # En los ultimos 4 frames la amplitud filtrada deberia ser claramente
    # menor a la diferencia cruda de ~60.
    max_filtrado = max(abs(y) for y in yaws_filtrados[-4:])
    print(f"    (amplitud maxima yaw filtrado: {max_filtrado:.1f})")
    assert max_filtrado < 28.0, (
        f"EMA no esta reduciendo jitter: amplitud filtrada={max_filtrado:.1f}, "
        f"esperaba < 28 (crudo es 30)"
    )


@_test("Detecta metodo PnP disponible")
def _():
    a = AnalizadorCabeza()
    datos = _generar_landmarks_sinteticos()
    a.procesar(datos)
    assert a._metodo_pnp is not None
    # Debe ser SQPNP o ITERATIVE
    posibles = []
    try:
        posibles.append(int(cv2.SOLVEPNP_SQPNP))
    except AttributeError:
        pass
    posibles.append(int(cv2.SOLVEPNP_ITERATIVE))
    assert a._metodo_pnp in posibles, f"metodo_pnp={a._metodo_pnp}, esperaba uno de {posibles}"


# =============================================================================
# TESTS CON HARDWARE
# =============================================================================

if not RPICAM_DISPONIBLE:
    print("\n--- Tests con hardware: SALTEADOS (rpicam-vid no disponible) ---")
else:
    print("\n--- Tests de AnalizadorCabeza (con hardware) ---")

    def _capturar_frames_con_rostro(n: int = 10) -> list[np.ndarray]:
        config = cargar_config()
        cap = CapturaVideo(config)
        cap.iniciar()
        frames = []
        try:
            for _ in range(15):  # warmup
                cap.leer()
            intentos = 0
            while len(frames) < n and intentos < n * 3:
                frame, _ = cap.leer()
                intentos += 1
                if frame is not None:
                    frames.append(frame)
            return frames
        finally:
            cap.detener()


    @_test("Pipeline completo captura->detector->analizador")
    def _():
        limpiar_cache()
        frames = _capturar_frames_con_rostro(n=10)
        det = DetectorRostro()
        ana = AnalizadorCabeza()
        det.iniciar()
        try:
            # Buscamos un frame con rostro
            datos_cabeza = None
            for f in frames:
                datos_r = det.procesar(f)
                if datos_r.rostro_presente:
                    datos_cabeza = ana.procesar(datos_r)
                    if datos_cabeza.valido:
                        break

            assert datos_cabeza is not None and datos_cabeza.valido, (
                "no se obtuvo pose valida en ninguno de los 10 frames"
            )
            # Ningun angulo deberia ser NaN
            assert np.isfinite(datos_cabeza.pitch_deg)
            assert np.isfinite(datos_cabeza.yaw_deg)
            assert np.isfinite(datos_cabeza.roll_deg)
            # La distancia estimada deberia ser razonable (entre 100mm y 2000mm)
            dist = float(np.linalg.norm(datos_cabeza.vector_traslacion))
            print(f"    pose: pitch={datos_cabeza.pitch_deg:+.1f}, "
                  f"yaw={datos_cabeza.yaw_deg:+.1f}, "
                  f"roll={datos_cabeza.roll_deg:+.1f}, "
                  f"distancia~{dist:.0f}mm")
            assert 100 < dist < 2000, f"distancia improbable: {dist}mm"
        finally:
            det.detener()


    @_test("Tiempo de procesamiento del analizador < 5ms")
    def _():
        limpiar_cache()
        frames = _capturar_frames_con_rostro(n=10)
        det = DetectorRostro()
        ana = AnalizadorCabeza()
        det.iniciar()
        try:
            tiempos = []
            for f in frames:
                datos_r = det.procesar(f)
                if datos_r.rostro_presente:
                    datos_c = ana.procesar(datos_r)
                    if datos_c.valido:
                        tiempos.append(datos_c.tiempo_procesamiento_ms)
            assert len(tiempos) >= 5, f"muy pocos frames validos: {len(tiempos)}"
            promedio = sum(tiempos) / len(tiempos)
            print(f"    (tiempo promedio analizador: {promedio:.2f}ms)")
            assert promedio < 5.0, f"tiempo promedio muy alto: {promedio:.2f}ms"
        finally:
            det.detener()


    @_test("Visualizacion en vivo con ejes 3D (5 segundos)")
    def _():
        """Muestra ejes 3D saliendo de la nariz para confirmar visualmente
        que pitch/yaw/roll son correctos."""
        limpiar_cache()
        config = cargar_config()

        cap = CapturaVideo(config)
        det = DetectorRostro()
        ana = AnalizadorCabeza()
        cap.iniciar()
        det.iniciar()

        ts_inicio = time.time()
        duracion = 5.0
        frames_validos = 0
        frames_totales = 0

        print(f"    Mira a la camara y mueve la cabeza por {duracion}s...")
        print(f"    Ejes: Rojo=X, Verde=Y, Azul=Z (sale del rostro)")

        try:
            while time.time() - ts_inicio < duracion:
                frame, _ = cap.leer()
                if frame is None:
                    continue
                frames_totales += 1

                datos_r = det.procesar(frame)
                datos_c = ana.procesar(datos_r)

                if datos_c.valido:
                    frames_validos += 1
                    frame_viz = AnalizadorCabeza.dibujar_ejes(
                        frame, datos_r, datos_c, ana, longitud_mm=70.0,
                    )
                    txt_pitch = f"pitch: {datos_c.pitch_deg:+6.1f}"
                    txt_yaw = f"yaw:   {datos_c.yaw_deg:+6.1f}"
                    txt_roll = f"roll:  {datos_c.roll_deg:+6.1f}"
                    color = (0, 255, 0)
                else:
                    frame_viz = frame.copy()
                    txt_pitch = "pitch: ---"
                    txt_yaw = "yaw:   ---"
                    txt_roll = "roll:  ---"
                    color = (0, 0, 255)

                cv2.putText(frame_viz, txt_pitch, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame_viz, txt_yaw, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame_viz, txt_roll, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow("NeuroDrive - Test Analizador Cabeza", frame_viz)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            det.detener()
            cap.detener()

        tasa = (frames_validos / frames_totales * 100) if frames_totales else 0.0
        print(f"    Frames totales: {frames_totales}, validos: {frames_validos} ({tasa:.1f}%)")
        assert frames_totales > 30
        assert tasa > 50.0, f"tasa de pose valida baja: {tasa:.1f}%"


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
