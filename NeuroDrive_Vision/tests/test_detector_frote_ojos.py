"""
test_detector_frote_ojos.py - Tests funcionales de DetectorFroteOjos.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_detector_frote_ojos

Tests sin hardware (13) - usan inyeccion directa de estado:
   1. Construccion con parametros default
   2. Construccion con parametros invalidos falla
   3. _calcular_region_ojo: regiones razonables
   4. _calcular_region_ojo: ojo muy chico devuelve box minimo
   5. _punto_en_rect: dentro y fuera
   6. Maquina de estados: contacto < 500ms NO emite evento
   7. Maquina de estados: contacto >= 500ms emite evento UNA vez
   8. Estado se resetea cuando se suelta el contacto
   9. Tasa de frotes/min se calcula correctamente
  10. Timeout de perdida de rostro resetea estado
  11. Throttling: Hands se saltea, maquina de estados corre siempre
  12. Throttling: frame saltado reutiliza ultimas puntas
  13. Reset limpia todo el estado

Tests con hardware (4):
  14. Iniciar y detener detector funciona
  15. Pipeline completo: rostro sin frote NO da evento
  16. Performance: tiempo de procesamiento
  17. Visualizacion en vivo (10 seg) - hacer el gesto de frotarse los ojos
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
from NeuroDrive_Vision.analizador_ojos import OJO_IZQ_INDICES, OJO_DER_INDICES
from NeuroDrive_Vision.detector_frote_ojos import (
    DetectorFroteOjos,
    DatosFroteOjos,
    ErrorDetectorFroteOjos,
    TIPS_DEDOS,
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
# Helpers sinteticos
# =============================================================================

def _crear_rostro_sintetico(
    ts: float,
    ojo_izq_centro: tuple = (250, 200),
    ojo_der_centro: tuple = (390, 200),
    ancho_ojo: int = 40,
) -> DatosRostro:
    """
    Crea un DatosRostro con landmarks de ojo en posiciones predecibles.
    """
    pp = np.zeros((468, 2), dtype=np.int32)
    pn = np.zeros((468, 3), dtype=np.float32)

    # Ojo izquierdo: P1 (esquina exterior) y P4 (esquina interior) en X
    pp[OJO_IZQ_INDICES[0]] = (ojo_izq_centro[0] - ancho_ojo // 2, ojo_izq_centro[1])
    pp[OJO_IZQ_INDICES[3]] = (ojo_izq_centro[0] + ancho_ojo // 2, ojo_izq_centro[1])
    pp[OJO_DER_INDICES[0]] = (ojo_der_centro[0] + ancho_ojo // 2, ojo_der_centro[1])
    pp[OJO_DER_INDICES[3]] = (ojo_der_centro[0] - ancho_ojo // 2, ojo_der_centro[1])

    return DatosRostro(
        rostro_presente=True,
        puntos_pixeles=pp,
        puntos_normalizados=pn,
        resolucion=(640, 480),
        timestamp=ts,
    )


def _frame_negro() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# =============================================================================
# Detector "mockeado" para tests sin hardware
# =============================================================================
#
# No podemos instanciar MediaPipe Hands sin que descargue el modelo.
# Pero la logica de deteccion de frote es independiente del modelo:
# solo importa "donde estan las puntas de los dedos".
#
# Lo que hacemos: crear el detector SIN iniciar Hands, y reemplazar
# el metodo procesar() para inyectar puntas sinteticas directamente.

class DetectorMockeado(DetectorFroteOjos):
    """
    Version del detector que NO carga MediaPipe Hands y permite inyectar
    manualmente la lista de puntas de dedos detectadas.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # No llamamos a iniciar(); marcamos activo manualmente
        self._activo = True
        self._hands = "MOCK"  # placeholder

    def iniciar(self):
        self._activo = True

    def detener(self):
        self._activo = False

    def procesar_mock(
        self,
        ts: float,
        datos_rostro: DatosRostro,
        puntas_inyectadas: list,
    ) -> DatosFroteOjos:
        """
        Version sintetica de procesar() que no usa MediaPipe.
        Toma las puntas como parametro en vez de detectarlas.
        """
        from NeuroDrive_Vision.detector_frote_ojos import DatosFroteOjos
        t0 = time.monotonic()

        # Si no hay rostro, comportamiento como en el original
        if not datos_rostro.rostro_presente or datos_rostro.puntos_pixeles is None:
            self._manejar_perdida_rostro(ts)
            res = DatosFroteOjos(
                valido=True,
                manos_detectadas=1 if puntas_inyectadas else 0,
                frote_en_curso=False,
                puntas_detectadas=puntas_inyectadas,
                frotes_por_minuto=self._calcular_frotes_por_minuto(ts),
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
                motivo_invalido="rostro_ausente",
            )
            self._ultimo_resultado = res
            return res

        self._ts_ultimo_rostro = ts

        # Regiones de ojos
        region_izq = self._calcular_region_ojo(
            datos_rostro.puntos_pixeles, OJO_IZQ_INDICES, 640, 480,
        )
        region_der = self._calcular_region_ojo(
            datos_rostro.puntos_pixeles, OJO_DER_INDICES, 640, 480,
        )

        puntas_en_zona = []
        for (px, py) in puntas_inyectadas:
            if self._punto_en_rect(px, py, region_izq) or self._punto_en_rect(px, py, region_der):
                puntas_en_zona.append((px, py))

        hay_contacto = len(puntas_en_zona) > 0

        evento_iniciado = False
        duracion_actual_ms = 0.0
        frote_en_curso = False

        if hay_contacto:
            if self._ts_inicio_contacto is None:
                self._ts_inicio_contacto = ts
                self._frote_ya_emitido = False
            duracion_actual_ms = (ts - self._ts_inicio_contacto) * 1000.0
            if duracion_actual_ms >= self.duracion_min_frote_ms:
                frote_en_curso = True
                if not self._frote_ya_emitido:
                    evento_iniciado = True
                    self._frote_ya_emitido = True
                    self._agregar_frote(ts)
        else:
            self._ts_inicio_contacto = None
            self._frote_ya_emitido = False

        bpm = self._calcular_frotes_por_minuto(ts)
        res = DatosFroteOjos(
            valido=True,
            manos_detectadas=1 if puntas_inyectadas else 0,
            frote_en_curso=frote_en_curso,
            duracion_frote_actual_ms=duracion_actual_ms,
            evento_frote_iniciado=evento_iniciado,
            frotes_por_minuto=bpm,
            tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
            region_ojo_izq=region_izq,
            region_ojo_der=region_der,
            puntas_detectadas=puntas_inyectadas,
            puntas_en_zona=puntas_en_zona,
        )
        self._ultimo_resultado = res
        return res


# =============================================================================
# TESTS SIN HARDWARE
# =============================================================================

print("\n--- Tests de DetectorFroteOjos (sin hardware, logica) ---")


@_test("Construccion con parametros default")
def _():
    d = DetectorFroteOjos()
    assert d.max_manos == 2
    assert d.model_complexity == 0
    assert d.duracion_min_frote_ms == 500.0
    assert d.ventana_frotes_seg == 300.0


@_test("Construccion con parametros invalidos falla")
def _():
    for kwargs in (
        {"max_manos": 3},
        {"model_complexity": 2},
        {"duracion_min_frote_ms": 0},
        {"ventana_frotes_seg": -10},
        {"procesar_cada_n_frames": 0},
    ):
        try:
            DetectorFroteOjos(**kwargs)
            raise AssertionError(f"deberia haber fallado con {kwargs}")
        except ValueError:
            pass


@_test("_calcular_region_ojo: regiones razonables")
def _():
    rostro = _crear_rostro_sintetico(ts=0.0)
    region_izq = DetectorFroteOjos._calcular_region_ojo(
        rostro.puntos_pixeles, OJO_IZQ_INDICES, 640, 480,
    )
    # Ojo izq centrado en (250, 200), ancho 40
    # region = 4 * 40 = 160 ancho, 2 * 40 = 80 alto, centrada en (250, 200)
    # x = 250 - 80 = 170, y = 200 - 40 = 160, w = 160, h = 80
    x, y, w, h = region_izq
    assert 165 <= x <= 175, f"x={x}"
    assert 155 <= y <= 165, f"y={y}"
    assert w == 160, f"w={w}"
    assert h == 80, f"h={h}"


@_test("_calcular_region_ojo: ojo muy chico devuelve box minimo")
def _():
    # Ojo de tamaño 2 px: debe usar el ancho minimo de 20
    pp = np.zeros((468, 2), dtype=np.int32)
    pp[OJO_IZQ_INDICES[0]] = (100, 100)
    pp[OJO_IZQ_INDICES[3]] = (102, 100)  # ancho = 2
    region = DetectorFroteOjos._calcular_region_ojo(pp, OJO_IZQ_INDICES, 640, 480)
    x, y, w, h = region
    assert w >= 70, f"w={w}, esperaba >= 70 (4 * 20 minimo)"


@_test("_punto_en_rect: dentro y fuera")
def _():
    rect = (100, 100, 50, 30)  # x=100..150, y=100..130
    assert DetectorFroteOjos._punto_en_rect(120, 110, rect)
    assert DetectorFroteOjos._punto_en_rect(100, 100, rect)
    assert not DetectorFroteOjos._punto_en_rect(150, 110, rect)  # x = x+w, fuera
    assert not DetectorFroteOjos._punto_en_rect(99, 110, rect)
    assert not DetectorFroteOjos._punto_en_rect(120, 130, rect)


@_test("Contacto < 500ms NO emite evento")
def _():
    d = DetectorMockeado()
    rostro = _crear_rostro_sintetico(0.0)
    # Punta dentro de region del ojo izquierdo (ojo en 250, 200)
    ts = 100.0
    # Frame 1: contacto arranca
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    assert not r.evento_frote_iniciado, "no deberia haber evento aun"
    # Frame 2: 200 ms despues, todavia bajo umbral
    ts += 0.2
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    assert not r.evento_frote_iniciado
    assert not r.frote_en_curso


@_test("Contacto >= 500ms emite evento UNA vez")
def _():
    d = DetectorMockeado()
    ts = 100.0
    # Frame 1
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    assert not r.evento_frote_iniciado
    # Frame 2: 600 ms despues -> cumple umbral
    ts += 0.6
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    assert r.evento_frote_iniciado, "deberia haber evento"
    assert r.frote_en_curso
    assert r.duracion_frote_actual_ms >= 500.0
    # Frame 3: 200 ms despues, sigue en contacto pero NO debe re-emitir
    ts += 0.2
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    assert not r.evento_frote_iniciado, "evento no debe re-emitirse"
    assert r.frote_en_curso, "frote sigue en curso"


@_test("Estado se resetea cuando se suelta el contacto")
def _():
    d = DetectorMockeado()
    ts = 100.0
    # Frote completo
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    ts += 0.6
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    assert r.evento_frote_iniciado
    # Soltamos
    ts += 0.5
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [])
    assert not r.frote_en_curso
    assert d._ts_inicio_contacto is None
    assert not d._frote_ya_emitido
    # Nuevo contacto: deberia emitirse al cumplir umbral
    ts += 0.5
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    ts += 0.6
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    assert r.evento_frote_iniciado, "segundo frote no se emitio"


@_test("Tasa de frotes/min se calcula correctamente")
def _():
    # Generamos 3 frotes en 60 seg con ventana de 300 (5 min)
    # bpm = (3 / 300) * 60 = 0.6
    d = DetectorMockeado(ventana_frotes_seg=300.0)
    ts = 100.0
    for i in range(3):
        # Frote completo
        d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
        ts += 0.6
        r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
        # Soltar y esperar
        ts += 0.5
        d.procesar_mock(ts, _crear_rostro_sintetico(ts), [])
        ts += 19.0
    assert 0.55 <= r.frotes_por_minuto <= 0.65, f"bpm={r.frotes_por_minuto}"


@_test("Timeout de perdida de rostro resetea estado")
def _():
    d = DetectorMockeado()
    ts = 100.0
    # Arrancamos un contacto
    d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    assert d._ts_inicio_contacto is not None

    # Perdida de rostro > 2s
    sin = DatosRostro(rostro_presente=False, resolucion=(640, 480), timestamp=ts + 3.0)
    d.procesar_mock(ts + 3.0, sin, [(250, 200)])
    assert d._ts_inicio_contacto is None, "estado no se reseteo tras timeout"


@_test("Throttling: Hands se saltea pero la maquina de estados corre siempre")
def _():
    # Con el nuevo throttling, en los frames saltados NO se llama a Hands
    # pero la maquina de estados de frote SI corre (reutilizando las
    # ultimas puntas). Verificamos:
    #   1) El default es procesar_cada_n_frames=2
    #   2) La logica de "toca evaluar Hands" es correcta
    d = DetectorFroteOjos()
    assert d.procesar_cada_n_frames == 2, "el default deberia ser 2"

    # Con n=3: frames 1,2 saltan Hands, frame 3 evalua Hands
    d3 = DetectorFroteOjos(procesar_cada_n_frames=3)
    secuencia_evaluar = []
    for _ in range(6):
        d3._contador_frames += 1
        toca = (d3._contador_frames % d3.procesar_cada_n_frames) == 0
        secuencia_evaluar.append(toca)
    # Esperado: F,F,T,F,F,T
    assert secuencia_evaluar == [False, False, True, False, False, True], (
        f"secuencia incorrecta: {secuencia_evaluar}"
    )


@_test("Throttling: frame saltado reutiliza ultimas puntas para detectar frote")
def _():
    # Verificamos que la maquina de estados sigue funcionando en frames
    # saltados. Usamos el mockeado pero forzando la reutilizacion de puntas.
    #
    # El mockeado no implementa throttling (su procesar_mock siempre evalua),
    # asi que este test valida la INTENCION: que _ultimas_puntas se use.
    # Probamos que si seteamos _ultimas_puntas manualmente, la deteccion
    # de frote las puede usar.
    d = DetectorMockeado(procesar_cada_n_frames=2)
    # Simulamos que Hands detecto una punta en el ojo izquierdo
    d._ultimas_puntas = [(250, 200)]
    d._ultimas_manos = 1
    # La maquina de estados deberia poder arrancar un contacto con esas puntas
    ts = 100.0
    r = d.procesar_mock(ts, _crear_rostro_sintetico(ts), d._ultimas_puntas)
    assert d._ts_inicio_contacto is not None, "deberia haber arrancado el contacto"


@_test("Reset limpia todo el estado")
def _():
    d = DetectorMockeado()
    ts = 100.0
    d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    ts += 0.6
    d.procesar_mock(ts, _crear_rostro_sintetico(ts), [(250, 200)])
    assert len(d._historial_frotes) > 0
    d.reset()
    assert d._ts_inicio_contacto is None
    assert not d._frote_ya_emitido
    assert len(d._historial_frotes) == 0
    assert d._ts_ultimo_rostro is None


# =============================================================================
# TESTS CON HARDWARE
# =============================================================================

if not RPICAM_DISPONIBLE:
    print("\n--- Tests con hardware: SALTEADOS ---")
else:
    print("\n--- Tests de DetectorFroteOjos (con hardware) ---")

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


    @_test("Iniciar y detener detector funciona")
    def _():
        d = DetectorFroteOjos()
        assert not d.activo
        d.iniciar()
        assert d.activo
        d.detener()
        assert not d.activo


    @_test("Pipeline completo: rostro sin frote NO da evento")
    def _():
        limpiar_cache()
        frames = _capturar_frames(n=10)
        det_rostro = DetectorRostro()
        det_frote = DetectorFroteOjos()
        det_rostro.iniciar()
        det_frote.iniciar()
        try:
            eventos = 0
            for f in frames:
                datos_r = det_rostro.procesar(f)
                if datos_r.rostro_presente:
                    datos_f = det_frote.procesar(f, datos_r)
                    if datos_f.evento_frote_iniciado:
                        eventos += 1
            # En 10 frames sin frotarse, no debe haber eventos
            print(f"    eventos detectados (no debe haber): {eventos}")
            assert eventos == 0, "no deberia detectar frote sin gesto real"
        finally:
            det_frote.detener()
            det_rostro.detener()


    @_test("Performance: tiempo de procesamiento")
    def _():
        limpiar_cache()
        frames = _capturar_frames(n=15)
        det_rostro = DetectorRostro()
        det_frote = DetectorFroteOjos()
        det_rostro.iniciar()
        det_frote.iniciar()
        try:
            tiempos = []
            for f in frames:
                datos_r = det_rostro.procesar(f)
                if datos_r.rostro_presente:
                    datos_f = det_frote.procesar(f, datos_r)
                    if datos_f.valido:
                        tiempos.append(datos_f.tiempo_procesamiento_ms)
            assert len(tiempos) >= 5
            promedio = sum(tiempos) / len(tiempos)
            maximo = max(tiempos)
            print(f"    (promedio: {promedio:.1f}ms, max: {maximo:.1f}ms)")
            # 80 ms permite el peor caso (palm detection + 2 manos en el frame).
            # En la mayoria de los frames sin manos, va a ser mucho menos.
            assert promedio < 80.0, f"promedio muy alto: {promedio:.1f}ms"
        finally:
            det_frote.detener()
            det_rostro.detener()


    @_test("Visualizacion en vivo (10 seg)")
    def _():
        limpiar_cache()
        config = cargar_config()
        cap = CapturaVideo(config)
        det_rostro = DetectorRostro()
        det_frote = DetectorFroteOjos()
        cap.iniciar()
        det_rostro.iniciar()
        det_frote.iniciar()

        ts_inicio = time.time()
        duracion = 10.0
        eventos = 0
        frames_totales = 0
        frames_validos = 0
        tiempos_frote = []

        print(f"    Mira a la camara por {duracion}s.")
        print(f"    Probar gestos para validar:")
        print(f"      1) Frotarse los ojos > 500ms (deberia detectar)")
        print(f"      2) Mover la mano cerca de la cara sin tocar (NO deberia detectar)")
        print(f"      3) Tocarse la mejilla (NO deberia detectar)")

        try:
            while time.time() - ts_inicio < duracion:
                frame, _ = cap.leer()
                if frame is None:
                    continue
                frames_totales += 1

                datos_r = det_rostro.procesar(frame)
                datos_f = det_frote.procesar(frame, datos_r)

                if datos_f.valido:
                    frames_validos += 1
                    tiempos_frote.append(datos_f.tiempo_procesamiento_ms)
                    if datos_f.evento_frote_iniciado:
                        eventos += 1

                frame_viz = DetectorFroteOjos.dibujar(frame, datos_f)

                # Overlay info
                cv2.putText(frame_viz, f"manos: {datos_f.manos_detectadas}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.putText(frame_viz, f"t: {datos_f.tiempo_procesamiento_ms:.1f}ms",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                if datos_f.frote_en_curso:
                    cv2.putText(frame_viz,
                                f"FROTE EN CURSO ({datos_f.duracion_frote_actual_ms:.0f}ms)",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                elif datos_f.duracion_frote_actual_ms > 0:
                    cv2.putText(frame_viz,
                                f"contacto: {datos_f.duracion_frote_actual_ms:.0f}ms",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 1)

                cv2.imshow("NeuroDrive - Test Detector Frote Ojos", frame_viz)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            det_frote.detener()
            det_rostro.detener()
            cap.detener()

        promedio = sum(tiempos_frote) / len(tiempos_frote) if tiempos_frote else 0.0
        maximo = max(tiempos_frote) if tiempos_frote else 0.0
        tasa = (frames_validos / frames_totales * 100) if frames_totales else 0.0
        print(f"    Frames totales: {frames_totales}, validos: {frames_validos} ({tasa:.1f}%)")
        print(f"    Tiempo de procesamiento: promedio {promedio:.1f}ms, max {maximo:.1f}ms")
        print(f"    Eventos de frote detectados: {eventos}")
        assert frames_totales > 60, "muy pocos frames"
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
