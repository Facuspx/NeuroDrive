"""
test_calibrador.py - Tests funcionales de Calibrador.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_calibrador

Tests sin hardware (18):
   1. Construccion con parametros default
   2. Construccion con duracion invalida falla
   3. Estado inicial: no activo, no terminado
   4. iniciar() activa el calibrador
   5. procesar() sin iniciar no hace nada
   6. procesar() acumula muestras de EAR/MAR
   7. _promedio_robusto EAR (P50-P90) descarta parpadeos
   8. _promedio_robusto EAR ignora parpadeos aunque sean muchos
   9. _promedio_robusto MAR (P10-P50) descarta habla/bostezos
  10. _promedio_robusto con pocas muestras devuelve None
  11. finalizar() con pocas muestras -> exito=False
  12. finalizar() con muestras buenas -> exito=True
  13. finalizar() con EAR base anomalo -> exito=False
  14. progreso y tiempo_restante son coherentes
  15. terminado=True cuando se cumple la duracion
  16. guardar() y cargar() preservan los datos
  17. cargar() de archivo inexistente devuelve None
  18. aplicar() actualiza el AnalizadorOjos solo si exito=True

Tests con hardware (2):
  19. Calibracion real corta (5 seg) - mira a la camara
  20. Pipeline: calibrar -> aplicar -> el analizador usa el nuevo umbral
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

import subprocess

try:
    resultado = subprocess.run(
        ["rpicam-vid", "--version"], capture_output=True, timeout=5,
    )
    RPICAM_DISPONIBLE = resultado.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired):
    RPICAM_DISPONIBLE = False

import numpy as np

from NeuroDrive_Vision.detector_rostro import DatosRostro
from NeuroDrive_Vision.analizador_ojos import AnalizadorOjos, OJO_IZQ_INDICES, OJO_DER_INDICES
from NeuroDrive_Vision.calibrador import (
    Calibrador,
    ResultadoCalibracion,
    ErrorCalibrador,
)

if RPICAM_DISPONIBLE:
    from NeuroDrive_Core.config_loader import cargar_config, limpiar_cache
    from NeuroDrive_Vision.captura_video import CapturaVideo
    from NeuroDrive_Vision.detector_rostro import DetectorRostro


_resultados = []


def _test(nombre):
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


def _crear_rostro_con_ear(ear_objetivo, ts):
    """Genera un DatosRostro cuyos ojos producen el EAR pedido (escala grande)."""
    pp = np.zeros((468, 2), dtype=np.int32)
    base = 400.0
    media_altura = int((ear_objetivo * base) / 2.0)
    for indices in (OJO_IZQ_INDICES, OJO_DER_INDICES):
        pp[indices[0]] = (0, 0)
        pp[indices[1]] = (120, -media_altura)
        pp[indices[2]] = (280, -media_altura)
        pp[indices[3]] = (int(base), 0)
        pp[indices[4]] = (280, media_altura)
        pp[indices[5]] = (120, media_altura)
    return DatosRostro(
        rostro_presente=True,
        puntos_pixeles=pp,
        puntos_normalizados=np.zeros((468, 3), dtype=np.float32),
        resolucion=(640, 480),
        timestamp=ts,
    )


print("\n--- Tests de Calibrador (sin hardware) ---")


@_test("Construccion con parametros default")
def _():
    c = Calibrador()
    assert c.duracion_seg == 60.0
    assert not c.activo


@_test("Construccion con duracion invalida falla")
def _():
    for dur in (0, -5):
        try:
            Calibrador(duracion_seg=dur)
            raise AssertionError(f"deberia fallar con duracion={dur}")
        except ValueError:
            pass


@_test("Estado inicial: no activo, no terminado")
def _():
    c = Calibrador()
    assert not c.activo
    assert not c.terminado
    assert c.progreso == 0.0


@_test("iniciar() activa el calibrador")
def _():
    c = Calibrador(duracion_seg=10.0)
    c.iniciar()
    assert c.activo
    assert not c.terminado
    assert 0.0 <= c.progreso < 0.1


@_test("procesar() sin iniciar no hace nada")
def _():
    c = Calibrador()
    c.procesar(_crear_rostro_con_ear(0.30, 0.0))
    assert c._frames_totales == 0
    assert len(c._muestras_ear) == 0


@_test("procesar() acumula muestras de EAR")
def _():
    c = Calibrador(duracion_seg=100.0)
    c.iniciar()
    for i in range(10):
        c.procesar(_crear_rostro_con_ear(0.30, time.monotonic()))
    assert c._frames_totales == 10
    assert len(c._muestras_ear) == 10
    for m in c._muestras_ear:
        assert abs(m - 0.30) < 0.05, f"muestra fuera de rango: {m}"


@_test("_promedio_robusto EAR (P50-P90) descarta parpadeos")
def _():
    # 80 muestras de ojos abiertos (0.30) + 20 parpadeos (0.08).
    # Con P50-P90 nos quedamos con la mitad alta -> solo los 0.30.
    muestras = [0.30] * 80 + [0.08] * 20
    promedio = Calibrador._promedio_robusto(
        muestras,
        Calibrador.PERCENTIL_EAR_INF,
        Calibrador.PERCENTIL_EAR_SUP,
    )
    assert promedio is not None
    assert abs(promedio - 0.30) < 0.02, (
        f"promedio EAR={promedio}, esperaba ~0.30 (deberia ignorar parpadeos)"
    )


@_test("_promedio_robusto EAR ignora parpadeos AUNQUE sean muchos")
def _():
    # Caso dificil: 55 abiertos + 45 parpadeos (casi mitad y mitad).
    # Con el viejo P25-P75 esto contaminaba el resultado porque parte
    # de los parpadeos caia dentro del rango central.
    # Con P50-P90 el limite inferior es la mediana: si los abiertos son
    # mayoria (>50%), la mediana cae en la zona "abierto" y filtra todo
    # parpadeo. Probamos con 55/45 (los abiertos siguen siendo mayoria).
    muestras = [0.28] * 55 + [0.09] * 45
    promedio = Calibrador._promedio_robusto(
        muestras,
        Calibrador.PERCENTIL_EAR_INF,
        Calibrador.PERCENTIL_EAR_SUP,
    )
    assert promedio is not None
    assert abs(promedio - 0.28) < 0.03, (
        f"promedio EAR={promedio}, esperaba ~0.28. El filtrado asimetrico "
        "deberia descartar los parpadeos aunque sean el 45% de las muestras"
    )


@_test("_promedio_robusto MAR (P10-P50) descarta habla/bostezos")
def _():
    # MAR base es la boca cerrada (valores bajos). 70 muestras de boca
    # cerrada (0.10) + 30 de boca abierta hablando/bostezando (0.45).
    # Con P10-P50 nos quedamos con la parte baja -> solo los 0.10.
    muestras = [0.10] * 70 + [0.45] * 30
    promedio = Calibrador._promedio_robusto(
        muestras,
        Calibrador.PERCENTIL_MAR_INF,
        Calibrador.PERCENTIL_MAR_SUP,
    )
    assert promedio is not None
    assert abs(promedio - 0.10) < 0.02, (
        f"promedio MAR={promedio}, esperaba ~0.10 (deberia ignorar boca abierta)"
    )


@_test("_promedio_robusto con pocas muestras devuelve None")
def _():
    assert Calibrador._promedio_robusto([], 50, 90) is None
    assert Calibrador._promedio_robusto([0.3, 0.3], 50, 90) is None


@_test("finalizar() con pocas muestras -> exito=False")
def _():
    c = Calibrador(duracion_seg=100.0)
    c.iniciar()
    for _ in range(10):
        c.procesar(_crear_rostro_con_ear(0.30, time.monotonic()))
    res = c.finalizar()
    assert res.exito is False
    assert "pocas muestras" in res.motivo_fallo.lower()


@_test("finalizar() con muestras buenas -> exito=True")
def _():
    c = Calibrador(duracion_seg=100.0)
    c.iniciar()
    for i in range(200):
        ear = 0.30 if i % 10 != 0 else 0.10
        c.procesar(_crear_rostro_con_ear(ear, time.monotonic()))
    res = c.finalizar()
    assert res.exito is True, f"deberia ser exitosa, motivo: {res.motivo_fallo}"
    assert abs(res.ear_base - 0.30) < 0.03, f"ear_base={res.ear_base}"
    assert res.muestras_validas == 200


@_test("finalizar() con EAR base anomalo -> exito=False")
def _():
    c = Calibrador(duracion_seg=100.0)
    c.iniciar()
    for _ in range(100):
        c.procesar(_crear_rostro_con_ear(0.10, time.monotonic()))
    res = c.finalizar()
    assert res.exito is False
    assert "fuera de rango" in res.motivo_fallo.lower()


@_test("progreso y tiempo_restante son coherentes")
def _():
    c = Calibrador(duracion_seg=10.0)
    c.iniciar()
    c._ts_inicio = time.monotonic() - 5.0
    assert 0.45 <= c.progreso <= 0.55, f"progreso={c.progreso}"
    assert 4.5 <= c.tiempo_restante_seg <= 5.5, f"restante={c.tiempo_restante_seg}"


@_test("terminado=True cuando se cumple la duracion")
def _():
    c = Calibrador(duracion_seg=10.0)
    c.iniciar()
    assert not c.terminado
    c._ts_inicio = time.monotonic() - 11.0
    assert c.terminado


@_test("guardar() y cargar() preservan los datos")
def _():
    res = ResultadoCalibracion(
        exito=True,
        timestamp=time.time(),
        ear_base=0.285,
        mar_base=0.12,
        pitch_neutro=-8.5,
        yaw_neutro=2.1,
        roll_neutro=-1.0,
        muestras_totales=900,
        muestras_validas=850,
        tasa_deteccion=0.944,
        duracion_real_seg=60.2,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        ruta = f.name
    try:
        res.guardar(ruta)
        cargado = ResultadoCalibracion.cargar(ruta)
        assert cargado is not None
        assert cargado.exito is True
        assert abs(cargado.ear_base - 0.285) < 1e-6
        assert abs(cargado.mar_base - 0.12) < 1e-6
        assert abs(cargado.pitch_neutro - (-8.5)) < 1e-6
        assert cargado.muestras_totales == 900
    finally:
        os.unlink(ruta)


@_test("cargar() de archivo inexistente devuelve None")
def _():
    assert ResultadoCalibracion.cargar("/tmp/no_existe_neurodrive_xyz.json") is None


@_test("aplicar() actualiza el AnalizadorOjos solo si exito=True")
def _():
    ana = AnalizadorOjos()
    umbral_original = ana.umbral_cierre

    res_ok = ResultadoCalibracion(exito=True, ear_base=0.20)
    aplicado = Calibrador.aplicar(res_ok, analizador_ojos=ana)
    assert aplicado is True
    assert ana.ear_base == 0.20
    assert ana.umbral_cierre != umbral_original

    ana2 = AnalizadorOjos()
    umbral2 = ana2.umbral_cierre
    res_fallo = ResultadoCalibracion(exito=False, motivo_fallo="test")
    aplicado2 = Calibrador.aplicar(res_fallo, analizador_ojos=ana2)
    assert aplicado2 is False
    assert ana2.umbral_cierre == umbral2


if not RPICAM_DISPONIBLE:
    print("\n--- Tests con hardware: SALTEADOS ---")
else:
    print("\n--- Tests de Calibrador (con hardware) ---")

    @_test("Calibracion real corta (5 seg)")
    def _():
        limpiar_cache()
        config = cargar_config()
        cap = CapturaVideo(config)
        det = DetectorRostro()
        calib = Calibrador(config, duracion_seg=5.0)

        cap.iniciar()
        det.iniciar()
        calib.iniciar()

        print(f"    Mira a la camara con ojos abiertos por 5 segundos...")
        try:
            while not calib.terminado:
                frame, _ = cap.leer()
                if frame is None:
                    continue
                datos_r = det.procesar(frame)
                calib.procesar(datos_r)
        finally:
            det.detener()
            cap.detener()

        res = calib.finalizar()
        print(f"    Resultado: {res}")
        print(f"    Muestras: {res.muestras_validas}/{res.muestras_totales} "
              f"(tasa {res.tasa_deteccion:.2f})")
        assert res.muestras_totales > 30, "muy pocos frames procesados"
        if res.exito:
            print(f"    ear_base calibrado: {res.ear_base:.3f}")
            assert 0.15 <= res.ear_base <= 0.45
        else:
            print(f"    (calibracion no exitosa: {res.motivo_fallo})")


    @_test("Pipeline: calibrar -> aplicar -> analizador usa nuevo umbral")
    def _():
        limpiar_cache()
        config = cargar_config()
        cap = CapturaVideo(config)
        det = DetectorRostro()
        calib = Calibrador(config, duracion_seg=5.0)
        ana_ojos = AnalizadorOjos()

        umbral_antes = ana_ojos.umbral_cierre

        cap.iniciar()
        det.iniciar()
        calib.iniciar()

        print(f"    Calibrando 5 seg para luego aplicar al analizador...")
        try:
            while not calib.terminado:
                frame, _ = cap.leer()
                if frame is None:
                    continue
                datos_r = det.procesar(frame)
                calib.procesar(datos_r)
        finally:
            det.detener()
            cap.detener()

        res = calib.finalizar()
        aplicado = Calibrador.aplicar(res, analizador_ojos=ana_ojos)

        if res.exito:
            assert aplicado is True
            print(f"    Umbral de cierre: {umbral_antes:.3f} -> {ana_ojos.umbral_cierre:.3f}")
        else:
            assert aplicado is False
            assert ana_ojos.umbral_cierre == umbral_antes
            print(f"    (calibracion no exitosa, analizador mantiene default)")


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
