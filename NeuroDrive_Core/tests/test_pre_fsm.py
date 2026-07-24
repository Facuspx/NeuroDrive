"""
test_pre_fsm.py - Tests funcionales del Pre-FSM y sus subdetectores.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Core.test_pre_fsm

Cada subdetector se testea individualmente con timestamps controlados.
Luego tests de integracion del PreFSM completo con envelopes simulados.

NO usa hilos ni MQ. Todo deterministico.
"""

from __future__ import annotations

import sys
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from common.contratos import (
    Envelope,
    EventoAckWearable,
    EventoFalloSensor,
    EventoProcesado,
    EventoRecuperacionSensor,
    EventoVision,
    EventoWearable,
    NivelRiesgoBPM,
    OrigenEvento,
    TipoMensaje,
    generar_id_mensaje,
)
from NeuroDrive_Core.config_loader import (
    Config,
    ConfigBocaSeccion,
    ConfigCabezaSeccion,
    ConfigOjosSeccion,
    ConfigVisionSeccion,
    ConfigWearableSeccion,
)
from NeuroDrive_Core.pre_fsm import (
    ClasificadorBPM,
    DetectorBostezos,
    DetectorCabeceos,
    DetectorMicrosuenos,
    DetectorParpadeos,
    DetectorRostroPerdido,
    PreFSM,
    VentanaPERCLOS,
)


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
# Helpers para construir configs
# =============================================================================

def _config_test() -> Config:
    """Config minimo para tests con valores predecibles."""
    cfg = Config()
    cfg.ojos = ConfigOjosSeccion(
        umbral_ear_cerrar=0.18,
        umbral_ear_abrir=0.22,
        dur_min_parpadeo_seg=0.10,
        dur_max_parpadeo_seg=0.40,
        dur_min_microsueno_seg=1.5,
        refractario_parpadeo_seg=0.25,
    )
    cfg.boca = ConfigBocaSeccion(
        umbral_mar_bostezo=0.6,
        dur_min_bostezo_seg=1.0,
        ventana_bostezos_seg=900.0,
        max_bostezos_ventana_larga=3,
    )
    cfg.cabeza = ConfigCabezaSeccion(
        umbral_pitch_grados=20.0,
        dur_min_cabeceo_seg=0.8,
        umbral_yaw_max_grados=35.0,
    )
    cfg.wearable = ConfigWearableSeccion(
        bpm_normal_min=60,
        bpm_normal_max=100,
        bpm_umbral_alerta=70,
        bpm_umbral_critico=60,
    )
    cfg.vision = ConfigVisionSeccion(max_frames_sin_rostro=5)
    return cfg


def _envelope_vision(
    ts: float,
    rostro: bool = True,
    ear_izq: float = 0.25,
    ear_der: float = 0.25,
    mar: float = 0.3,
    pitch: float = 0.0,
    yaw: float = 0.0,
    frote: bool = False,
    secuencia: int = 1,
) -> Envelope:
    ev = EventoVision(
        timestamp=ts,
        rostro_detectado=rostro,
        ear_izquierdo=ear_izq if rostro else None,
        ear_derecho=ear_der if rostro else None,
        mar=mar if rostro else None,
        pitch_grados=pitch if rostro else None,
        yaw_grados=yaw if rostro else None,
        roll_grados=0.0 if rostro else None,
        frote_ojos_activo=frote,
    )
    return Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo="cam-01",
        id_sesion="ses-test",
        id_mensaje=generar_id_mensaje("vis", secuencia),
        numero_secuencia=secuencia,
        timestamp_origen=ts,
        evento=ev,
    )


def _envelope_wearable(ts: float, bpm: int, secuencia: int = 1) -> Envelope:
    ev = EventoWearable(timestamp=ts, bpm=bpm)
    return Envelope(
        tipo=TipoMensaje.EVENTO_WEARABLE,
        origen=OrigenEvento.WEARABLE,
        id_dispositivo="wearable-01",
        id_sesion="ses-test",
        id_mensaje=generar_id_mensaje("wea", secuencia),
        numero_secuencia=secuencia,
        timestamp_origen=ts,
        evento=ev,
    )


# =============================================================================
# TESTS: DetectorParpadeos
# =============================================================================

print("\n--- Tests de DetectorParpadeos ---")


@_test("Parpadeo normal: cruzar umbral hacia abajo y luego arriba")
def _():
    cfg = _config_test()
    det = DetectorParpadeos(cfg.ojos)
    base = 1000.0

    # Ojos abiertos
    assert not det.procesar(0.25, base + 0.0)
    # Empezando a cerrar (no llega al umbral)
    assert not det.procesar(0.20, base + 0.05)
    # Cerrados
    assert not det.procesar(0.15, base + 0.10)
    # Abriendo (cruza arriba) -> parpadeo confirmado
    assert det.procesar(0.25, base + 0.25)


@_test("Parpadeo demasiado largo NO se cuenta (es microsueño)")
def _():
    cfg = _config_test()
    det = DetectorParpadeos(cfg.ojos)
    base = 1000.0

    det.procesar(0.25, base)
    det.procesar(0.15, base + 0.10)
    # 0.5 segundos cerrado supera dur_max_parpadeo (0.40s)
    assert not det.procesar(0.25, base + 0.60)


@_test("Refractario evita doble conteo")
def _():
    cfg = _config_test()
    det = DetectorParpadeos(cfg.ojos)
    base = 1000.0

    # Primer parpadeo
    det.procesar(0.25, base)
    det.procesar(0.15, base + 0.10)
    assert det.procesar(0.25, base + 0.25)

    # Segundo parpadeo inmediato (dentro del refractario 0.25s)
    det.procesar(0.15, base + 0.30)
    assert not det.procesar(0.25, base + 0.40), "no deberia contar dentro del refractario"


@_test("Frecuencia por minuto devuelve None en los primeros 30 segundos")
def _():
    cfg = _config_test()
    det = DetectorParpadeos(cfg.ojos)
    base = 1000.0
    det.procesar(0.25, base)
    assert det.frecuencia_por_minuto(base + 10) is None
    assert det.frecuencia_por_minuto(base + 29) is None
    assert det.frecuencia_por_minuto(base + 31) is not None


@_test("Frecuencia se extrapola correctamente")
def _():
    cfg = _config_test()
    det = DetectorParpadeos(cfg.ojos)
    base = 1000.0

    # Hacemos 10 parpadeos en los primeros 30 segundos
    # Esperamos ~20 por minuto extrapolado
    for i in range(10):
        t = base + i * 3.0  # cada 3 segundos
        det.procesar(0.25, t)
        det.procesar(0.15, t + 0.10)
        det.procesar(0.25, t + 0.25)

    # Avanzamos a t=32s
    det.procesar(0.25, base + 32.0)
    freq = det.frecuencia_por_minuto(base + 32.0)
    assert freq is not None
    # 10 parpadeos en 32s -> ~18.75 por minuto
    assert 17 < freq < 22, f"frecuencia inesperada: {freq}"


# =============================================================================
# TESTS: DetectorMicrosuenos
# =============================================================================

print("\n--- Tests de DetectorMicrosuenos ---")


@_test("Microsueño confirmado cuando duracion > umbral")
def _():
    cfg = _config_test()
    det = DetectorMicrosuenos(cfg.ojos)
    base = 1000.0

    det.procesar(0.25, base)
    det.procesar(0.15, base + 0.1)  # cierra
    # Sigue cerrado por 2 segundos (dur_min_microsueno = 1.5)
    det.procesar(0.15, base + 1.0)
    det.procesar(0.15, base + 1.8)
    # Abre -> emite microsueño
    assert det.procesar(0.25, base + 2.0)


@_test("Cierre corto NO genera microsueño")
def _():
    cfg = _config_test()
    det = DetectorMicrosuenos(cfg.ojos)
    base = 1000.0

    det.procesar(0.25, base)
    det.procesar(0.15, base + 0.1)
    # Solo 0.5s cerrado -> no es microsueño
    assert not det.procesar(0.25, base + 0.6)


@_test("Microsueño se emite UNA SOLA VEZ al terminar")
def _():
    cfg = _config_test()
    det = DetectorMicrosuenos(cfg.ojos)
    base = 1000.0

    det.procesar(0.25, base)
    det.procesar(0.15, base + 0.1)
    det.procesar(0.15, base + 1.8)
    # Abre y emite
    assert det.procesar(0.25, base + 2.0)
    # Siguientes frames no deben emitir nada
    assert not det.procesar(0.25, base + 2.1)
    assert not det.procesar(0.25, base + 2.2)


# =============================================================================
# TESTS: DetectorBostezos
# =============================================================================

print("\n--- Tests de DetectorBostezos ---")


@_test("Bostezo confirmado y registrado en ventana larga")
def _():
    cfg = _config_test()
    det = DetectorBostezos(cfg.boca)
    base = 1000.0

    det.procesar(0.3, base)
    det.procesar(0.7, base + 0.1)  # boca abierta
    det.procesar(0.7, base + 1.2)  # sostenido (>1s)
    # Boca cierra -> bostezo confirmado
    assert det.procesar(0.3, base + 1.5)
    assert det.contar_ventana_larga(base + 1.5) == 1


@_test("Apertura corta NO es bostezo")
def _():
    cfg = _config_test()
    det = DetectorBostezos(cfg.boca)
    base = 1000.0

    det.procesar(0.3, base)
    det.procesar(0.7, base + 0.1)
    # Apertura de 0.5s, menor a dur_min_bostezo
    assert not det.procesar(0.3, base + 0.6)


@_test("Ventana larga purga bostezos viejos")
def _():
    cfg = _config_test()
    det = DetectorBostezos(cfg.boca)
    base = 1000.0

    # Bostezo en t=0
    det.procesar(0.3, base)
    det.procesar(0.7, base + 0.1)
    det.procesar(0.7, base + 1.2)
    det.procesar(0.3, base + 1.5)

    # Pasamos 16 minutos (mayor a ventana de 15 min)
    futuro = base + 16 * 60
    assert det.contar_ventana_larga(futuro) == 0


# =============================================================================
# TESTS: DetectorCabeceos
# =============================================================================

print("\n--- Tests de DetectorCabeceos ---")


@_test("Cabeceo confirmado con pitch sostenido")
def _():
    cfg = _config_test()
    det = DetectorCabeceos(cfg.cabeza)
    base = 1000.0

    det.procesar(5.0, 0.0, base)   # pitch bajo, no cabeceo
    det.procesar(25.0, 0.0, base + 0.1)  # cruza umbral 20
    det.procesar(25.0, 0.0, base + 1.0)  # sostenido (>0.8s)
    # Baja el pitch -> cabeceo confirmado
    assert det.procesar(5.0, 0.0, base + 1.5)


@_test("Cabeceo NO se cuenta si yaw esta muy girado")
def _():
    cfg = _config_test()
    det = DetectorCabeceos(cfg.cabeza)
    base = 1000.0

    # Cabeza muy girada (yaw 40 > umbral 35) -> invalida
    det.procesar(25.0, 40.0, base)
    det.procesar(25.0, 40.0, base + 1.0)
    assert not det.procesar(5.0, 40.0, base + 1.5)


@_test("Inclinacion corta NO es cabeceo")
def _():
    cfg = _config_test()
    det = DetectorCabeceos(cfg.cabeza)
    base = 1000.0

    det.procesar(5.0, 0.0, base)
    det.procesar(25.0, 0.0, base + 0.1)
    # Solo 0.5s inclinado, menor a 0.8s
    assert not det.procesar(5.0, 0.0, base + 0.6)


# =============================================================================
# TESTS: VentanaPERCLOS
# =============================================================================

print("\n--- Tests de VentanaPERCLOS ---")


@_test("PERCLOS devuelve None antes de 10 segundos")
def _():
    cfg = _config_test()
    v = VentanaPERCLOS(cfg.ojos, ventana_seg=60.0)
    base = 1000.0
    v.procesar(0.25, base)
    v.procesar(0.25, base + 5.0)
    assert v.calcular(base + 5.0) is None


@_test("PERCLOS = 0 con ojos siempre abiertos")
def _():
    cfg = _config_test()
    v = VentanaPERCLOS(cfg.ojos, ventana_seg=60.0)
    base = 1000.0
    for i in range(0, 30):
        v.procesar(0.25, base + i)
    perclos = v.calcular(base + 30)
    assert perclos is not None
    assert perclos < 0.05, f"PERCLOS deberia ser ~0, obtuve {perclos}"


@_test("PERCLOS = 1 con ojos siempre cerrados")
def _():
    cfg = _config_test()
    v = VentanaPERCLOS(cfg.ojos, ventana_seg=60.0)
    base = 1000.0
    for i in range(0, 30):
        v.procesar(0.10, base + i)
    perclos = v.calcular(base + 30)
    assert perclos is not None
    assert perclos > 0.95, f"PERCLOS deberia ser ~1, obtuve {perclos}"


@_test("PERCLOS intermedio ~0.5 con mitad cerrado/mitad abierto")
def _():
    cfg = _config_test()
    v = VentanaPERCLOS(cfg.ojos, ventana_seg=60.0)
    base = 1000.0
    # Primeros 15s abierto
    for i in range(0, 15):
        v.procesar(0.25, base + i)
    # Siguientes 15s cerrado
    for i in range(15, 30):
        v.procesar(0.10, base + i)
    perclos = v.calcular(base + 30)
    assert perclos is not None
    assert 0.4 < perclos < 0.6, f"PERCLOS deberia ser ~0.5, obtuve {perclos}"


# =============================================================================
# TESTS: ClasificadorBPM
# =============================================================================

print("\n--- Tests de ClasificadorBPM ---")


@_test("BPM None devuelve DESCONOCIDO")
def _():
    cfg = _config_test()
    c = ClasificadorBPM(cfg.wearable)
    assert c.clasificar() == NivelRiesgoBPM.DESCONOCIDO


@_test("BPM normal (75) devuelve NORMAL")
def _():
    cfg = _config_test()
    c = ClasificadorBPM(cfg.wearable)
    c.actualizar(75)
    assert c.clasificar() == NivelRiesgoBPM.NORMAL


@_test("BPM bajo (65) devuelve ALERTA")
def _():
    cfg = _config_test()
    c = ClasificadorBPM(cfg.wearable)
    c.actualizar(65)
    assert c.clasificar() == NivelRiesgoBPM.ALERTA


@_test("BPM muy bajo (55) devuelve CRITICO")
def _():
    cfg = _config_test()
    c = ClasificadorBPM(cfg.wearable)
    c.actualizar(55)
    assert c.clasificar() == NivelRiesgoBPM.CRITICO


@_test("BPM alto (120) devuelve NORMAL (taquicardia != fatiga)")
def _():
    cfg = _config_test()
    c = ClasificadorBPM(cfg.wearable)
    c.actualizar(120)
    assert c.clasificar() == NivelRiesgoBPM.NORMAL


# =============================================================================
# TESTS: DetectorRostroPerdido
# =============================================================================

print("\n--- Tests de DetectorRostroPerdido ---")


@_test("Rostro presente: confiable y disponible")
def _():
    cfg = _config_test()
    d = DetectorRostroPerdido(cfg.vision)
    no_conf, disp = d.procesar(True)
    assert not no_conf
    assert disp


@_test("Rostro perdido 1-N frames: marca no confiable pero disponible")
def _():
    cfg = _config_test()  # max_frames_sin_rostro = 5
    d = DetectorRostroPerdido(cfg.vision)
    for i in range(5):
        no_conf, disp = d.procesar(False)
        assert no_conf
        assert disp, f"frame {i}: deberia seguir disponible"


@_test("Rostro perdido > N frames: marca vision_disponible=False")
def _():
    cfg = _config_test()
    d = DetectorRostroPerdido(cfg.vision)
    for _ in range(5):
        d.procesar(False)
    # frame 6: ya supera el umbral
    no_conf, disp = d.procesar(False)
    assert no_conf
    assert not disp


@_test("Reaparecer rostro resetea el contador")
def _():
    cfg = _config_test()
    d = DetectorRostroPerdido(cfg.vision)
    for _ in range(10):
        d.procesar(False)
    # Vuelve
    d.procesar(True)
    # Y se pierde de nuevo, deberia volver a estar disponible (contador 1)
    no_conf, disp = d.procesar(False)
    assert disp


# =============================================================================
# TESTS: PreFSM (clase principal)
# =============================================================================

print("\n--- Tests de integracion del PreFSM ---")


@_test("PreFSM produce EventoProcesado a partir de EventoVision")
def _():
    cfg = _config_test()
    pre = PreFSM(cfg)
    env = _envelope_vision(ts=1000.0)
    ep = pre.procesar(env)
    assert ep is not None
    assert isinstance(ep, EventoProcesado)
    assert ep.timestamp == 1000.0


@_test("PreFSM produce EventoProcesado a partir de EventoWearable y clasifica BPM")
def _():
    cfg = _config_test()
    pre = PreFSM(cfg)
    env = _envelope_wearable(ts=1000.0, bpm=55)
    ep = pre.procesar(env)
    assert ep is not None
    assert ep.bpm_actual == 55
    assert ep.nivel_riesgo_bpm == NivelRiesgoBPM.CRITICO


@_test("PreFSM devuelve None para ACK del wearable")
def _():
    cfg = _config_test()
    pre = PreFSM(cfg)
    ack = EventoAckWearable(
        timestamp=1000.0,
        id_secuencia=1,
        secuencia_correcta=True,
        tiempo_respuesta_ms=1500,
    )
    env = Envelope(
        tipo=TipoMensaje.EVENTO_ACK_WEARABLE,
        origen=OrigenEvento.WEARABLE,
        id_dispositivo="wearable-01",
        id_sesion="ses-test",
        id_mensaje="wea-00099",
        numero_secuencia=99,
        timestamp_origen=1000.0,
        evento=ack,
    )
    assert pre.procesar(env) is None
    # Pero get_evento_ack devuelve el ACK
    assert pre.get_evento_ack(env) is ack


@_test("PreFSM marca vision_disponible=False tras EventoFalloSensor")
def _():
    cfg = _config_test()
    pre = PreFSM(cfg)
    fallo = EventoFalloSensor(
        timestamp=1000.0,
        sensor_afectado=OrigenEvento.VISION,
        motivo="test",
        severidad=2,
    )
    env = Envelope(
        tipo=TipoMensaje.FALLO_SENSOR,
        origen=OrigenEvento.INTERNO,
        id_dispositivo="core",
        id_sesion="ses-test",
        id_mensaje="int-00001",
        numero_secuencia=1,
        timestamp_origen=1000.0,
        evento=fallo,
    )
    pre.procesar(env)
    assert not pre.vision_disponible


@_test("PreFSM recupera disponibilidad con EventoRecuperacionSensor")
def _():
    cfg = _config_test()
    pre = PreFSM(cfg)
    # Primero falla
    pre._procesar_fallo_sensor(EventoFalloSensor(
        timestamp=1000.0,
        sensor_afectado=OrigenEvento.WEARABLE,
        motivo="test",
        severidad=2,
    ))
    assert not pre.wearable_disponible

    # Luego recupera
    pre._procesar_recuperacion_sensor(EventoRecuperacionSensor(
        timestamp=1050.0,
        sensor_recuperado=OrigenEvento.WEARABLE,
        tiempo_caido_seg=50.0,
    ))
    assert pre.wearable_disponible


@_test("Detecta bostezo completo desde frames de Vision")
def _():
    cfg = _config_test()
    pre = PreFSM(cfg)
    base = 1000.0

    # Frame normal
    ep = pre.procesar(_envelope_vision(ts=base, mar=0.3, secuencia=1))
    assert not ep.bostezo

    # Frames con MAR alto sostenido (>1s)
    pre.procesar(_envelope_vision(ts=base + 0.1, mar=0.7, secuencia=2))
    pre.procesar(_envelope_vision(ts=base + 0.8, mar=0.7, secuencia=3))
    pre.procesar(_envelope_vision(ts=base + 1.2, mar=0.7, secuencia=4))

    # Frame con MAR bajo: cierra el bostezo
    ep_final = pre.procesar(_envelope_vision(ts=base + 1.5, mar=0.3, secuencia=5))
    assert ep_final.bostezo, "deberia haberse emitido bostezo=True"

    # Frame siguiente NO debe emitir bostezo de nuevo
    ep_post = pre.procesar(_envelope_vision(ts=base + 1.6, mar=0.3, secuencia=6))
    assert not ep_post.bostezo
    assert ep_post.bostezos_ventana_larga == 1


@_test("Frote de ojos invalida deteccion ocular pero permite bostezo")
def _():
    cfg = _config_test()
    pre = PreFSM(cfg)
    base = 1000.0

    # Frame con frote de ojos activo Y mar alto
    pre.procesar(_envelope_vision(ts=base, mar=0.3, frote=False, secuencia=1))
    pre.procesar(_envelope_vision(ts=base + 0.1, ear_izq=0.10, ear_der=0.10, mar=0.7, frote=True, secuencia=2))
    pre.procesar(_envelope_vision(ts=base + 1.0, ear_izq=0.10, ear_der=0.10, mar=0.7, frote=True, secuencia=3))
    ep = pre.procesar(_envelope_vision(ts=base + 1.5, ear_izq=0.10, ear_der=0.10, mar=0.3, frote=True, secuencia=4))

    # Tiene que estar marcado como no confiable
    assert ep.ventana_no_confiable
    assert ep.motivo_no_confiable == "frote_ojos"
    # Microsueño NO se cuenta (EAR bajo pero hay frote)
    assert not ep.microsueno


@_test("Rostro perdido marca ventana_no_confiable y luego vision_disponible=False")
def _():
    cfg = _config_test()  # max_frames_sin_rostro = 5
    pre = PreFSM(cfg)
    base = 1000.0

    # 5 frames sin rostro -> no confiable pero disponible
    for i in range(5):
        ep = pre.procesar(_envelope_vision(ts=base + i, rostro=False, secuencia=i + 1))
        assert ep.ventana_no_confiable
        assert ep.vision_disponible, f"frame {i}: deberia seguir disponible"

    # frame 6 (>5) -> ya no disponible
    ep_critico = pre.procesar(_envelope_vision(ts=base + 6, rostro=False, secuencia=7))
    assert ep_critico.ventana_no_confiable
    assert not ep_critico.vision_disponible


@_test("Wearable disponible afecta el EventoProcesado tras fallo")
def _():
    cfg = _config_test()
    pre = PreFSM(cfg)

    # Primero un envelope normal
    pre.procesar(_envelope_wearable(ts=1000.0, bpm=75, secuencia=1))
    assert pre.wearable_disponible

    # Fallo del wearable
    fallo_env = Envelope(
        tipo=TipoMensaje.FALLO_SENSOR,
        origen=OrigenEvento.INTERNO,
        id_dispositivo="core",
        id_sesion="ses-test",
        id_mensaje="int-00001",
        numero_secuencia=1,
        timestamp_origen=1100.0,
        evento=EventoFalloSensor(
            timestamp=1100.0,
            sensor_afectado=OrigenEvento.WEARABLE,
            motivo="heartbeat_timeout",
            severidad=2,
        ),
    )
    pre.procesar(fallo_env)
    assert not pre.wearable_disponible
    assert pre.clasificador_bpm.bpm_actual is None

    # Siguiente vision: marca wearable_disponible=False
    ep = pre.procesar(_envelope_vision(ts=1110.0, secuencia=2))
    assert not ep.wearable_disponible
    assert ep.bpm_actual is None
    assert ep.nivel_riesgo_bpm == NivelRiesgoBPM.DESCONOCIDO


# =============================================================================
# TEST INTEGRACION: escenario completo
# =============================================================================

print("\n--- Test de escenario integrado ---")


@_test("Escenario completo: conductor con sintomas progresivos")
def _():
    """Simulamos varios minutos con eventos crecientes y verificamos
    que el PreFSM produce el flujo correcto de EventoProcesado."""
    cfg = _config_test()
    pre = PreFSM(cfg)
    base = 1000.0

    # Minuto 1: todo normal, BPM 75
    pre.procesar(_envelope_wearable(ts=base, bpm=75, secuencia=1))
    for i in range(15):
        pre.procesar(_envelope_vision(ts=base + i, secuencia=10 + i))

    # Minuto 2: aparece un bostezo sostenido completo
    # Necesita dur_min_bostezo_seg=1.0 con MAR alto sostenido
    base2 = base + 60
    seq = 100
    # Apertura
    pre.procesar(_envelope_vision(ts=base2 + 1, mar=0.7, secuencia=seq)); seq += 1
    # Sostenido
    pre.procesar(_envelope_vision(ts=base2 + 1.5, mar=0.7, secuencia=seq)); seq += 1
    pre.procesar(_envelope_vision(ts=base2 + 2.2, mar=0.7, secuencia=seq)); seq += 1
    # Cierre -> confirma bostezo
    ep_bostezo = pre.procesar(_envelope_vision(ts=base2 + 2.5, mar=0.3, secuencia=seq)); seq += 1
    assert ep_bostezo.bostezo, "bostezo deberia haberse confirmado"

    # Minuto 3: microsueño (>= 1.5s con EAR bajo)
    base3 = base + 120
    pre.procesar(_envelope_vision(ts=base3, ear_izq=0.25, ear_der=0.25, secuencia=300))
    pre.procesar(_envelope_vision(ts=base3 + 0.1, ear_izq=0.10, ear_der=0.10, secuencia=301))
    pre.procesar(_envelope_vision(ts=base3 + 1.0, ear_izq=0.10, ear_der=0.10, secuencia=302))
    pre.procesar(_envelope_vision(ts=base3 + 1.8, ear_izq=0.10, ear_der=0.10, secuencia=303))
    ep_microsueño = pre.procesar(
        _envelope_vision(ts=base3 + 2.0, ear_izq=0.25, ear_der=0.25, secuencia=304)
    )
    assert ep_microsueño.microsueno, "microsueño deberia haberse detectado"

    # BPM baja: el clasificador detecta CRITICO
    pre.procesar(_envelope_wearable(ts=base3 + 3, bpm=55, secuencia=500))
    ep_post = pre.procesar(_envelope_vision(ts=base3 + 4, secuencia=305))
    assert ep_post.bpm_actual == 55
    assert ep_post.nivel_riesgo_bpm == NivelRiesgoBPM.CRITICO

    # Verificamos contadores
    stats = pre.estadisticas()
    assert stats["envelopes_procesados"] > 20
    assert stats["eventos_procesados_emitidos"] > 20
    assert stats["bostezos_ultimos_15min"] >= 1, (
        f"esperaba >=1 bostezo, hay {stats['bostezos_ultimos_15min']}"
    )


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
    print("  NeuroDrive - Tests del Pre-FSM")
    print("=" * 60)
    sys.exit(_resumen())
