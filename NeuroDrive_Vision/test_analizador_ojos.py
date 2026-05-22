"""
test_analizador_ojos.py - Tests funcionales de AnalizadorOjos.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_analizador_ojos

Tests sin hardware (15):
   1. Construccion con parametros default
   2. Construccion con factores invalidos falla
   3. Construccion con ear_base invalido falla
   4. EAR matematico: ojo abierto sintetico = 0.30
   5. EAR matematico: ojo cerrado sintetico = 0.05
   6. EAR matematico: base degenerada (perfil) -> 0.0
   7. Procesar sin rostro -> invalido
   8. Histeresis: zona gris mantiene estado abierto
   9. Histeresis: zona gris mantiene estado cerrado
  10. Parpadeo normal (200ms) -> evento "normal"
  11. Parpadeo lento (800ms) -> evento "lento"
  12. Microsueño (2s) -> evento "microsueño"
  13. Ruido (30ms) -> sin evento
  14. PERCLOS calculado correctamente
  15. Parpadeos/min calculado correctamente
  16. Ventana PERCLOS expira muestras viejas
  17. Timeout de perdida de rostro resetea estado
  18. Actualizar umbrales no resetea historial

Tests con hardware (3):
  19. Pipeline completo captura -> detector -> analizador_ojos
  20. Tiempo de procesamiento < 2 ms
  21. Visualizacion en vivo con contornos de ojos (5 seg)
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
from NeuroDrive_Vision.analizador_ojos import (
    AnalizadorOjos,
    DatosOjos,
    OJO_IZQ_INDICES,
    OJO_DER_INDICES,
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
# Helpers: generar DatosRostro sinteticos
# =============================================================================
#
# Importante: usamos ESCALA GRANDE (base horizontal = 400 px en vez de 40)
# para que la cuantizacion a int32 no rompa los EAR cercanos a los umbrales.
# Con cara real a distancia normal, el ojo mide 30-40 px, pero la histeresis
# del analizador absorbe el ruido de cuantizacion gracias al gap de umbrales.
# En tests sinteticos donde queremos precision sub-percentual, usamos escala x10.

def _crear_rostro_con_ear(ear_objetivo: float, ts: float) -> DatosRostro:
    """
    Genera un DatosRostro sintetico cuyos landmarks de ojo producen
    exactamente el EAR deseado. Usa escala grande (400 px) para evitar
    cuantizacion.
    """
    pp = np.zeros((468, 2), dtype=np.int32)
    base = 400.0
    media_altura = int((ear_objetivo * base) / 2.0)

    # Ambos ojos iguales para que el promedio == ear_objetivo
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
        tiempo_procesamiento_ms=1.0,
    )


def _crear_rostro_perfil(ts: float) -> DatosRostro:
    """
    Genera un DatosRostro con landmarks colapsados (rostro de perfil
    simulado: todos los puntos del ojo en la misma X).
    """
    pp = np.zeros((468, 2), dtype=np.int32)
    for indices in (OJO_IZQ_INDICES, OJO_DER_INDICES):
        for idx in indices:
            pp[idx] = (100, 100)  # todos en el mismo punto
    return DatosRostro(
        rostro_presente=True,
        puntos_pixeles=pp,
        puntos_normalizados=np.zeros((468, 3), dtype=np.float32),
        resolucion=(640, 480),
        timestamp=ts,
    )


# =============================================================================
# TESTS SIN HARDWARE
# =============================================================================

print("\n--- Tests de AnalizadorOjos (sin hardware) ---")


@_test("Construccion con parametros default")
def _():
    a = AnalizadorOjos()
    assert a.ear_base == AnalizadorOjos.EAR_BASE_DEFAULT
    assert 0 < a.umbral_cierre < a.umbral_apertura
    assert a.ventana_perclos_seg == 60.0


@_test("Construccion con factores invalidos falla")
def _():
    for kwargs in (
        {"factor_cierre": 0.9, "factor_apertura": 0.7},  # cierre > apertura
        {"factor_cierre": 1.2},                           # > 1
        {"factor_cierre": 0.0},                           # = 0
    ):
        try:
            AnalizadorOjos(**kwargs)
            raise AssertionError(f"deberia haber fallado con {kwargs}")
        except ValueError:
            pass


@_test("Construccion con ear_base invalido falla")
def _():
    for ear in (0.0, 1.5, -0.1):
        try:
            AnalizadorOjos(ear_base=ear)
            raise AssertionError(f"deberia haber fallado con ear_base={ear}")
        except ValueError:
            pass


@_test("EAR matematico: ojo abierto sintetico ~= 0.30")
def _():
    pp = np.zeros((468, 2), dtype=np.int32)
    # base 400, altura total 120 -> EAR = (60+60)/(2*400) = 0.15? No.
    # EAR = (|P2-P6| + |P3-P5|) / (2*|P1-P4|)
    # vertical = 2 * media_altura = 60 -> EAR = (60+60)/(2*400) = 0.15
    # Para EAR=0.30: media_altura = (0.30 * 400)/2 = 60
    pp[OJO_IZQ_INDICES[0]] = (0, 0)
    pp[OJO_IZQ_INDICES[1]] = (120, -60)
    pp[OJO_IZQ_INDICES[2]] = (280, -60)
    pp[OJO_IZQ_INDICES[3]] = (400, 0)
    pp[OJO_IZQ_INDICES[4]] = (280, 60)
    pp[OJO_IZQ_INDICES[5]] = (120, 60)
    ear = AnalizadorOjos._calcular_ear(pp, OJO_IZQ_INDICES)
    assert abs(ear - 0.30) < 0.001, f"EAR={ear}, esperaba 0.30"


@_test("EAR matematico: ojo cerrado sintetico ~= 0.05")
def _():
    pp = np.zeros((468, 2), dtype=np.int32)
    # Para EAR=0.05: media_altura = (0.05 * 400)/2 = 10
    pp[OJO_IZQ_INDICES[0]] = (0, 0)
    pp[OJO_IZQ_INDICES[1]] = (120, -10)
    pp[OJO_IZQ_INDICES[2]] = (280, -10)
    pp[OJO_IZQ_INDICES[3]] = (400, 0)
    pp[OJO_IZQ_INDICES[4]] = (280, 10)
    pp[OJO_IZQ_INDICES[5]] = (120, 10)
    ear = AnalizadorOjos._calcular_ear(pp, OJO_IZQ_INDICES)
    assert abs(ear - 0.05) < 0.001, f"EAR={ear}, esperaba 0.05"


@_test("EAR matematico: base degenerada (perfil) -> 0.0")
def _():
    pp = np.zeros((468, 2), dtype=np.int32)
    # P1 == P4: base degenerada
    for idx in OJO_IZQ_INDICES:
        pp[idx] = (100, 100)
    ear = AnalizadorOjos._calcular_ear(pp, OJO_IZQ_INDICES)
    assert ear == 0.0


@_test("Procesar sin rostro -> invalido")
def _():
    a = AnalizadorOjos()
    sin_rostro = DatosRostro(rostro_presente=False, resolucion=(640, 480))
    r = a.procesar(sin_rostro)
    assert isinstance(r, DatosOjos)
    assert r.valido is False
    assert "no presente" in r.motivo_invalido.lower()


@_test("Histeresis: zona gris mantiene estado abierto")
def _():
    a = AnalizadorOjos(ear_base=0.30)
    # umbral_cierre = 0.21, umbral_apertura = 0.24
    ts = 100.0
    # Empieza abierto con EAR=0.30
    a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 0.05
    # EAR=0.225 (entre los umbrales): debe mantener abierto
    r = a.procesar(_crear_rostro_con_ear(0.225, ts))
    assert not r.ojos_cerrados, f"esperaba abierto, pero EAR={r.ear_promedio} y ojos_cerrados={r.ojos_cerrados}"


@_test("Histeresis: zona gris mantiene estado cerrado")
def _():
    a = AnalizadorOjos(ear_base=0.30)
    ts = 100.0
    a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 0.05
    # Cerramos primero
    r = a.procesar(_crear_rostro_con_ear(0.10, ts)); ts += 0.05
    assert r.ojos_cerrados
    # Zona gris: debe mantenerse cerrado
    r = a.procesar(_crear_rostro_con_ear(0.225, ts))
    assert r.ojos_cerrados, f"esperaba cerrado, pero EAR={r.ear_promedio} y ojos_cerrados={r.ojos_cerrados}"


@_test("Parpadeo normal (200ms) -> evento 'normal'")
def _():
    a = AnalizadorOjos(ear_base=0.30)
    ts = 100.0
    a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 0.5
    a.procesar(_crear_rostro_con_ear(0.10, ts)); ts += 0.20
    r = a.procesar(_crear_rostro_con_ear(0.30, ts))
    assert r.evento_parpadeo == "normal", f"evento={r.evento_parpadeo!r}"
    assert 180 < r.duracion_parpadeo_ms < 220


@_test("Parpadeo lento (800ms) -> evento 'lento'")
def _():
    a = AnalizadorOjos(ear_base=0.30)
    ts = 100.0
    a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 0.5
    a.procesar(_crear_rostro_con_ear(0.10, ts)); ts += 0.80
    r = a.procesar(_crear_rostro_con_ear(0.30, ts))
    assert r.evento_parpadeo == "lento", f"evento={r.evento_parpadeo!r}"


@_test("Microsueño (2s) -> evento 'microsueño'")
def _():
    a = AnalizadorOjos(ear_base=0.30)
    ts = 100.0
    a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 0.5
    a.procesar(_crear_rostro_con_ear(0.10, ts)); ts += 2.0
    r = a.procesar(_crear_rostro_con_ear(0.30, ts))
    assert r.evento_parpadeo == "microsueño", f"evento={r.evento_parpadeo!r}"


@_test("Ruido (30ms) -> sin evento")
def _():
    a = AnalizadorOjos(ear_base=0.30)
    ts = 100.0
    a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 0.5
    a.procesar(_crear_rostro_con_ear(0.10, ts)); ts += 0.03
    r = a.procesar(_crear_rostro_con_ear(0.30, ts))
    assert r.evento_parpadeo == "", f"evento={r.evento_parpadeo!r}, no deberia haber evento"


@_test("PERCLOS: 5 cerrados de 10 -> ~0.50")
def _():
    a = AnalizadorOjos(ear_base=0.30, ventana_perclos_seg=2.0)
    ts = 100.0
    for i in range(10):
        ear = 0.10 if i < 5 else 0.30
        r = a.procesar(_crear_rostro_con_ear(ear, ts))
        ts += 0.2
    assert 0.45 <= r.perclos <= 0.55, f"PERCLOS={r.perclos}"


@_test("Parpadeos/min: 10 parpadeos en ~57s -> ~10 bpm")
def _():
    a = AnalizadorOjos(ear_base=0.30, ventana_parpadeos_seg=60.0)
    ts = 100.0
    for i in range(10):
        a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 5.0
        a.procesar(_crear_rostro_con_ear(0.10, ts)); ts += 0.2
        r = a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 0.5
    assert 9.0 <= r.parpadeos_por_minuto <= 11.0, f"bpm={r.parpadeos_por_minuto}"


@_test("Ventana PERCLOS expira muestras viejas")
def _():
    a = AnalizadorOjos(ear_base=0.30, ventana_perclos_seg=2.0)
    ts = 100.0
    # Cerrados al inicio
    for _ in range(5):
        a.procesar(_crear_rostro_con_ear(0.10, ts)); ts += 0.1
    # Esperamos > ventana
    ts += 5.0
    # Un solo frame nuevo abierto
    r = a.procesar(_crear_rostro_con_ear(0.30, ts))
    # Todas las muestras viejas (cerradas) ya expiraron, solo queda la nueva (abierta)
    assert r.perclos == 0.0, f"PERCLOS={r.perclos}, deberia ser 0 tras expirar"


@_test("Timeout de perdida de rostro resetea estado")
def _():
    a = AnalizadorOjos(ear_base=0.30)
    ts = 100.0
    # Cerramos los ojos
    a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 0.05
    a.procesar(_crear_rostro_con_ear(0.10, ts)); ts += 0.05
    assert a._ojos_cerrados_estado is True
    # Perdida de rostro > 2 seg (TIMEOUT_PERDIDA_ROSTRO_S)
    sin_rostro = DatosRostro(rostro_presente=False, resolucion=(640, 480), timestamp=ts)
    a.procesar(sin_rostro); ts += 3.0
    a.procesar(DatosRostro(rostro_presente=False, resolucion=(640, 480), timestamp=ts))
    # Estado deberia haberse reseteado
    assert a._ojos_cerrados_estado is False, "deberia haberse reseteado"


@_test("Actualizar umbrales no resetea historial")
def _():
    a = AnalizadorOjos(ear_base=0.30)
    ts = 100.0
    # Llenamos historial con varios frames
    for _ in range(5):
        a.procesar(_crear_rostro_con_ear(0.30, ts)); ts += 0.1
    cant_muestras_antes = len(a._historial_cierres)
    assert cant_muestras_antes >= 5
    # Actualizamos umbrales
    a.actualizar_umbrales(ear_base=0.25)
    assert a.ear_base == 0.25
    assert len(a._historial_cierres) == cant_muestras_antes, "historial no debe resetearse"


# =============================================================================
# TESTS CON HARDWARE
# =============================================================================

if not RPICAM_DISPONIBLE:
    print("\n--- Tests con hardware: SALTEADOS (rpicam-vid no disponible) ---")
else:
    print("\n--- Tests de AnalizadorOjos (con hardware) ---")

    def _capturar_frames_con_rostro(n: int = 10) -> list[np.ndarray]:
        config = cargar_config()
        cap = CapturaVideo(config)
        cap.iniciar()
        frames = []
        try:
            for _ in range(15):
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


    @_test("Pipeline completo captura -> detector -> analizador_ojos")
    def _():
        limpiar_cache()
        frames = _capturar_frames_con_rostro(n=10)
        det = DetectorRostro()
        ana = AnalizadorOjos()
        det.iniciar()
        try:
            datos_ojos = None
            for f in frames:
                datos_r = det.procesar(f)
                if datos_r.rostro_presente:
                    datos_ojos = ana.procesar(datos_r)
                    if datos_ojos.valido:
                        break
            assert datos_ojos is not None and datos_ojos.valido, (
                "no se obtuvo medicion valida en 10 frames"
            )
            print(f"    EAR izq={datos_ojos.ear_izq:.3f}, der={datos_ojos.ear_der:.3f}, "
                  f"prom={datos_ojos.ear_promedio:.3f}, cerrados={datos_ojos.ojos_cerrados}")
            # EAR humano realista: entre 0.10 (cerrado) y 0.45 (muy abierto)
            assert 0.05 < datos_ojos.ear_promedio < 0.50, (
                f"EAR fuera de rango realista: {datos_ojos.ear_promedio}"
            )
        finally:
            det.detener()


    @_test("Tiempo de procesamiento del analizador < 2ms")
    def _():
        limpiar_cache()
        frames = _capturar_frames_con_rostro(n=10)
        det = DetectorRostro()
        ana = AnalizadorOjos()
        det.iniciar()
        try:
            tiempos = []
            for f in frames:
                datos_r = det.procesar(f)
                if datos_r.rostro_presente:
                    datos_o = ana.procesar(datos_r)
                    if datos_o.valido:
                        tiempos.append(datos_o.tiempo_procesamiento_ms)
            assert len(tiempos) >= 5
            promedio = sum(tiempos) / len(tiempos)
            print(f"    (tiempo promedio: {promedio:.3f}ms)")
            assert promedio < 2.0
        finally:
            det.detener()


    @_test("Visualizacion en vivo con contornos de ojos (5 seg)")
    def _():
        limpiar_cache()
        config = cargar_config()
        cap = CapturaVideo(config)
        det = DetectorRostro()
        ana = AnalizadorOjos()
        cap.iniciar()
        det.iniciar()

        ts_inicio = time.time()
        duracion = 5.0
        eventos_detectados = {"normal": 0, "lento": 0, "microsueño": 0}
        frames_totales = 0
        frames_validos = 0

        print(f"    Mira a la camara y parpadea normal por {duracion}s...")
        print(f"    Verde = abierto, Rojo = cerrado")

        try:
            while time.time() - ts_inicio < duracion:
                frame, _ = cap.leer()
                if frame is None:
                    continue
                frames_totales += 1

                datos_r = det.procesar(frame)
                datos_o = ana.procesar(datos_r)

                if datos_o.valido:
                    frames_validos += 1
                    frame_viz = AnalizadorOjos.dibujar_ojos(frame, datos_r, datos_o)
                    if datos_o.evento_parpadeo:
                        eventos_detectados[datos_o.evento_parpadeo] = eventos_detectados.get(datos_o.evento_parpadeo, 0) + 1
                    cv2.putText(frame_viz,
                                f"EAR: {datos_o.ear_promedio:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    cv2.putText(frame_viz,
                                f"PERCLOS: {datos_o.perclos:.2f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 1)
                    cv2.putText(frame_viz,
                                f"bpm: {datos_o.parpadeos_por_minuto:.1f}",
                                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 1)
                    if datos_o.ojos_cerrados:
                        cv2.putText(frame_viz, "OJOS CERRADOS",
                                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255), 2)
                else:
                    frame_viz = frame.copy()
                    cv2.putText(frame_viz, "SIN ROSTRO",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                cv2.imshow("NeuroDrive - Test Analizador Ojos", frame_viz)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            det.detener()
            cap.detener()

        tasa = (frames_validos / frames_totales * 100) if frames_totales else 0.0
        print(f"    Frames totales: {frames_totales}, validos: {frames_validos} ({tasa:.1f}%)")
        print(f"    Eventos detectados: {eventos_detectados}")
        assert frames_totales > 30
        assert tasa > 70.0, f"tasa de validos baja: {tasa:.1f}%"


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
