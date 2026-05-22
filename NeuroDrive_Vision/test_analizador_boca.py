"""
test_analizador_boca.py - Tests funcionales de AnalizadorBoca.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_analizador_boca

Tests sin hardware (15):
   1. Construccion con parametros default
   2. Construccion con umbrales invalidos falla
   3. MAR matematico: boca cerrada sintetica
   4. MAR matematico: boca muy abierta sintetica
   5. MAR matematico: base degenerada (perfil) -> 0.0
   6. Procesar sin rostro -> invalido
   7. Histeresis: zona gris mantiene estado cerrado
   8. Histeresis: zona gris mantiene estado abierto
   9. Bostezo valido (3s) -> evento_bostezo=True
  10. Apertura corta (1s) NO es bostezo
  11. Apertura larga (12s) se descarta como anomalia
  12. Bostezos/min cuenta correctamente
  13. Ventana de bostezos expira muestras viejas
  14. Timeout de perdida de rostro resetea estado
  15. Actualizar umbrales no resetea historial

Tests con hardware (3):
  16. Pipeline completo captura -> detector -> analizador_boca
  17. Tiempo de procesamiento < 2 ms
  18. Visualizacion en vivo con contornos de boca (5 seg)
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
from NeuroDrive_Vision.analizador_boca import (
    AnalizadorBoca,
    DatosBoca,
    BOCA_INDICES,
)

if RPICAM_DISPONIBLE:
    from NeuroDrive_Core.config_loader import cargar_config, limpiar_cache
    from NeuroDrive_Vision.captura_video import CapturaVideo


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
# Helpers de generacion sintetica
# =============================================================================

def _crear_rostro_con_mar(mar_objetivo: float, ts: float) -> DatosRostro:
    """
    Genera un DatosRostro sintetico con landmarks de boca que producen
    el MAR pedido.

    MAR = (d_v1 + d_v2 + d_v3) / (2 * d_horiz)
    Si d_horiz = 200 px y queremos MAR=X:
        d_v_total = X * 2 * 200 = 400 * X
        d_v_cada = 400 * X / 3 (si los 3 son iguales)
    Usamos escala grande (200) para evitar cuantizacion int32.
    """
    pp = np.zeros((468, 2), dtype=np.int32)

    base = 200.0
    d_v_promedio_total = mar_objetivo * 2.0 * base
    # Repartimos en 3 pares verticales iguales
    d_v_cada = d_v_promedio_total / 3.0
    media_v = d_v_cada / 2.0  # mitad arriba, mitad abajo del centro

    # Comisuras
    pp[BOCA_INDICES["comisura_izq"]] = (0, 0)
    pp[BOCA_INDICES["comisura_der"]] = (int(base), 0)

    # Pares verticales en x=50, 100, 150 (dentro de la base de 200)
    for x, key_sup, key_inf in (
        (50,  "sup_izq",    "inf_izq"),
        (100, "sup_centro", "inf_centro"),
        (150, "sup_der",    "inf_der"),
    ):
        pp[BOCA_INDICES[key_sup]] = (x, -int(media_v))
        pp[BOCA_INDICES[key_inf]] = (x,  int(media_v))

    return DatosRostro(
        rostro_presente=True,
        puntos_pixeles=pp,
        puntos_normalizados=np.zeros((468, 3), dtype=np.float32),
        resolucion=(640, 480),
        timestamp=ts,
        tiempo_procesamiento_ms=1.0,
    )


# =============================================================================
# TESTS SIN HARDWARE
# =============================================================================

print("\n--- Tests de AnalizadorBoca (sin hardware) ---")


@_test("Construccion con parametros default")
def _():
    a = AnalizadorBoca()
    assert a.umbral_apertura == 0.50
    assert a.umbral_cierre == 0.40
    assert a.ventana_bostezos_seg == 300.0


@_test("Construccion con umbrales invalidos falla")
def _():
    for kwargs in (
        {"umbral_apertura": 0.30, "umbral_cierre": 0.40},  # cierre >= apertura
        {"umbral_apertura": 0.0},                           # = 0
        {"umbral_apertura": 2.5},                           # > 2
    ):
        try:
            AnalizadorBoca(**kwargs)
            raise AssertionError(f"deberia haber fallado con {kwargs}")
        except ValueError:
            pass


@_test("MAR matematico: boca cerrada sintetica ~= 0.10")
def _():
    # base=200, mar=0.10 -> d_v_total = 40, cada = 13.3, media = 6.6
    # MAR real = (3 * 2 * round(6.6)) / (2*200) = (3*14)/400 = 0.105
    datos = _crear_rostro_con_mar(0.10, 0.0)
    mar = AnalizadorBoca._calcular_mar(datos.puntos_pixeles)
    assert abs(mar - 0.10) < 0.02, f"MAR={mar}, esperaba ~0.10"


@_test("MAR matematico: boca muy abierta sintetica ~= 0.60")
def _():
    datos = _crear_rostro_con_mar(0.60, 0.0)
    mar = AnalizadorBoca._calcular_mar(datos.puntos_pixeles)
    assert abs(mar - 0.60) < 0.02, f"MAR={mar}, esperaba ~0.60"


@_test("MAR matematico: base degenerada -> 0.0")
def _():
    pp = np.zeros((468, 2), dtype=np.int32)
    for nombre, idx in BOCA_INDICES.items():
        pp[idx] = (100, 100)
    mar = AnalizadorBoca._calcular_mar(pp)
    assert mar == 0.0


@_test("Procesar sin rostro -> invalido")
def _():
    a = AnalizadorBoca()
    sin = DatosRostro(rostro_presente=False, resolucion=(640, 480))
    r = a.procesar(sin)
    assert r.valido is False
    assert "no presente" in r.motivo_invalido.lower()


@_test("Histeresis: zona gris mantiene estado cerrado")
def _():
    a = AnalizadorBoca()  # cierre=0.40, apertura=0.50
    ts = 100.0
    r = a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 0.05
    assert not r.boca_abierta
    # MAR=0.45 (entre umbrales): debe seguir cerrada
    r = a.procesar(_crear_rostro_con_mar(0.45, ts))
    assert not r.boca_abierta, f"esperaba cerrada, MAR={r.mar} abierta={r.boca_abierta}"


@_test("Histeresis: zona gris mantiene estado abierto")
def _():
    a = AnalizadorBoca()
    ts = 100.0
    a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 0.05
    # Abre con MAR=0.60
    r = a.procesar(_crear_rostro_con_mar(0.60, ts)); ts += 0.05
    assert r.boca_abierta
    # Zona gris: debe seguir abierta
    r = a.procesar(_crear_rostro_con_mar(0.45, ts))
    assert r.boca_abierta, f"esperaba abierta, MAR={r.mar} abierta={r.boca_abierta}"


@_test("Bostezo valido (3s) -> evento_bostezo=True")
def _():
    a = AnalizadorBoca()
    ts = 100.0
    a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 0.5
    # Boca abierta por 3 segundos
    a.procesar(_crear_rostro_con_mar(0.70, ts)); ts += 3.0
    # Cierra
    r = a.procesar(_crear_rostro_con_mar(0.10, ts))
    assert r.evento_bostezo, f"esperaba bostezo, evento={r.evento_bostezo}, duracion={r.duracion_bostezo_ms}"
    assert 2900 < r.duracion_bostezo_ms < 3100


@_test("Apertura corta (1s) NO es bostezo")
def _():
    a = AnalizadorBoca()
    ts = 100.0
    a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 0.5
    a.procesar(_crear_rostro_con_mar(0.70, ts)); ts += 1.0
    r = a.procesar(_crear_rostro_con_mar(0.10, ts))
    assert not r.evento_bostezo, f"no deberia ser bostezo: {r.duracion_bostezo_ms}ms"


@_test("Apertura larga (12s) se descarta como anomalia")
def _():
    a = AnalizadorBoca()
    ts = 100.0
    a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 0.5
    a.procesar(_crear_rostro_con_mar(0.70, ts)); ts += 12.0
    r = a.procesar(_crear_rostro_con_mar(0.10, ts))
    assert not r.evento_bostezo, f"12s no deberia contar como bostezo"


@_test("Bostezos/min cuenta correctamente")
def _():
    # 3 bostezos en 60 seg con ventana de 5 min:
    # bpm = (3 / 300) * 60 = 0.6
    a = AnalizadorBoca(ventana_bostezos_seg=300.0)
    ts = 100.0
    for i in range(3):
        a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 5.0
        a.procesar(_crear_rostro_con_mar(0.70, ts)); ts += 3.0
        r = a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 12.0
    # Esperado: 3 bostezos en 60 seg, ventana 300s -> bpm = 0.6
    assert 0.55 <= r.bostezos_por_minuto <= 0.65, f"bpm={r.bostezos_por_minuto}"


@_test("Ventana de bostezos expira muestras viejas")
def _():
    a = AnalizadorBoca(ventana_bostezos_seg=10.0)  # ventana corta para test
    ts = 100.0
    # 1 bostezo
    a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 0.5
    a.procesar(_crear_rostro_con_mar(0.70, ts)); ts += 3.0
    r = a.procesar(_crear_rostro_con_mar(0.10, ts))
    assert len(a._historial_bostezos) == 1
    # Avanzo > 10 seg
    ts += 15.0
    r = a.procesar(_crear_rostro_con_mar(0.10, ts))
    assert r.bostezos_por_minuto == 0.0, "ventana no expiro"


@_test("Timeout de perdida de rostro resetea estado")
def _():
    a = AnalizadorBoca()
    ts = 100.0
    # Abre boca
    a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 0.05
    a.procesar(_crear_rostro_con_mar(0.70, ts)); ts += 0.5
    assert a._boca_abierta_estado is True
    # Perdida de rostro > 2s
    sin = DatosRostro(rostro_presente=False, resolucion=(640, 480), timestamp=ts)
    a.procesar(sin); ts += 3.0
    a.procesar(DatosRostro(rostro_presente=False, resolucion=(640, 480), timestamp=ts))
    assert a._boca_abierta_estado is False, "deberia haberse reseteado"


@_test("Actualizar umbrales no resetea historial")
def _():
    a = AnalizadorBoca()
    ts = 100.0
    # Genero 1 bostezo
    a.procesar(_crear_rostro_con_mar(0.10, ts)); ts += 0.5
    a.procesar(_crear_rostro_con_mar(0.70, ts)); ts += 3.0
    a.procesar(_crear_rostro_con_mar(0.10, ts))
    cant_antes = len(a._historial_bostezos)
    assert cant_antes == 1
    a.actualizar_umbrales(umbral_apertura=0.45, umbral_cierre=0.35)
    assert len(a._historial_bostezos) == cant_antes, "historial no debe resetearse"


# =============================================================================
# TESTS CON HARDWARE
# =============================================================================

if not RPICAM_DISPONIBLE:
    print("\n--- Tests con hardware: SALTEADOS ---")
else:
    print("\n--- Tests de AnalizadorBoca (con hardware) ---")

    def _capturar_frames(n: int = 10) -> list[np.ndarray]:
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


    @_test("Pipeline completo captura -> detector -> analizador_boca")
    def _():
        limpiar_cache()
        frames = _capturar_frames(n=10)
        det = DetectorRostro()
        ana = AnalizadorBoca()
        det.iniciar()
        try:
            datos_boca = None
            for f in frames:
                datos_r = det.procesar(f)
                if datos_r.rostro_presente:
                    datos_boca = ana.procesar(datos_r)
                    if datos_boca.valido:
                        break
            assert datos_boca is not None and datos_boca.valido
            print(f"    MAR={datos_boca.mar:.3f}, boca_abierta={datos_boca.boca_abierta}")
            # MAR humano realista: 0.02 (cerrada) a 1.0 (muy abierta)
            assert 0.0 < datos_boca.mar < 1.5, f"MAR fuera de rango: {datos_boca.mar}"
        finally:
            det.detener()


    @_test("Tiempo de procesamiento del analizador < 2ms")
    def _():
        limpiar_cache()
        frames = _capturar_frames(n=10)
        det = DetectorRostro()
        ana = AnalizadorBoca()
        det.iniciar()
        try:
            tiempos = []
            for f in frames:
                datos_r = det.procesar(f)
                if datos_r.rostro_presente:
                    datos_b = ana.procesar(datos_r)
                    if datos_b.valido:
                        tiempos.append(datos_b.tiempo_procesamiento_ms)
            assert len(tiempos) >= 5
            promedio = sum(tiempos) / len(tiempos)
            print(f"    (tiempo promedio: {promedio:.3f}ms)")
            assert promedio < 2.0
        finally:
            det.detener()


    @_test("Visualizacion en vivo con contornos de boca (5 seg)")
    def _():
        limpiar_cache()
        config = cargar_config()
        cap = CapturaVideo(config)
        det = DetectorRostro()
        ana = AnalizadorBoca()
        cap.iniciar()
        det.iniciar()

        ts_inicio = time.time()
        duracion = 5.0
        bostezos = 0
        frames_totales = 0
        frames_validos = 0

        print(f"    Mira a la camara por {duracion}s.")
        print(f"    Verde=cerrada, Amarillo=abierta corta, Rojo=bostezo en curso")
        print(f"    Probar abrir bien grande la boca > 2 seg para detectar bostezo")

        try:
            while time.time() - ts_inicio < duracion:
                frame, _ = cap.leer()
                if frame is None:
                    continue
                frames_totales += 1

                datos_r = det.procesar(frame)
                datos_b = ana.procesar(datos_r)

                if datos_b.valido:
                    frames_validos += 1
                    frame_viz = AnalizadorBoca.dibujar_boca(frame, datos_r, datos_b)
                    if datos_b.evento_bostezo:
                        bostezos += 1
                    cv2.putText(frame_viz, f"MAR: {datos_b.mar:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    if datos_b.boca_abierta:
                        cv2.putText(frame_viz,
                                    f"abierta: {datos_b.duracion_apertura_actual_ms:.0f}ms",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 1)
                else:
                    frame_viz = frame.copy()
                    cv2.putText(frame_viz, "SIN ROSTRO", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("NeuroDrive - Test Analizador Boca", frame_viz)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            det.detener()
            cap.detener()

        tasa = (frames_validos / frames_totales * 100) if frames_totales else 0.0
        print(f"    Frames totales: {frames_totales}, validos: {frames_validos} ({tasa:.1f}%)")
        print(f"    Bostezos detectados: {bostezos}")
        assert frames_totales > 30
        assert tasa > 70.0, f"tasa baja: {tasa:.1f}%"


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
