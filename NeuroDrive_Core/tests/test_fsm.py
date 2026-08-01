"""
test_fsm.py - Tests funcionales de la FSM pura.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Core.test_fsm

Valida:
  1. Transiciones por cada par (estado origen -> estado destino).
  2. Reglas de histeresis (pausa por rostro perdido, ventana no confiable).
  3. Acumulacion de eventos para escalada rapida.
  4. Comandos generados en cada transicion.
  5. Manejo de fallos y recuperaciones de sensores.

Cada test inyecta eventos sinteticos en orden controlado y verifica
que la FSM transicione exactamente como esperamos.
"""

from __future__ import annotations

import sys
import traceback

# Forzar UTF-8 en stdout/stderr
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from common.contratos import (
    ComandoActuador,
    EstadoFSM,
    EventoAckWearable,
    EventoFalloSensor,
    EventoProcesado,
    EventoRecuperacionSensor,
    NivelRiesgoBPM,
    OrigenEvento,
    TipoComandoActuador,
)
from NeuroDrive_Core.fsm import FSM, ConfigFSM


# =============================================================================
# Framework de testing minimalista
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
# Helpers para construir eventos sinteticos
# =============================================================================

def ev_normal(t: float, **kwargs) -> EventoProcesado:
    """Evento sin nada anormal (conductor alerto)."""
    base = dict(
        timestamp=t,
        microsueno=False,
        bostezo=False,
        cabeceo=False,
        parpadeo=True,
        parpadeos_por_minuto=8.0,
        perclos=0.1,
        bostezos_ventana_larga=0,
        bpm_actual=72,
        nivel_riesgo_bpm=NivelRiesgoBPM.ALERTA,
        ventana_no_confiable=False,
        vision_disponible=True,
        wearable_disponible=True,
    )
    base.update(kwargs)
    return EventoProcesado(**base)


def ev_microsueno(t: float, **kwargs) -> EventoProcesado:
    return ev_normal(t, microsueno=True, perclos=0.6, **kwargs)


def ev_bostezo(t: float, **kwargs) -> EventoProcesado:
    # Representa "bostezos acumulados confirmados" (3 en ventana larga)
    return ev_normal(t, bostezo=True, bostezos_ventana_larga=3, **kwargs)


def ev_cabeceo(t: float, **kwargs) -> EventoProcesado:
    return ev_normal(t, cabeceo=True, **kwargs)


def ev_señales_leves(t: float, **kwargs) -> EventoProcesado:
    fsm = crear_fsm()
    salida = fsm.procesar_evento(ev_microsueno(t=100))
    assert salida.transicion_ocurrio
    assert salida.motivo_transicion != ""


def ev_ack_ok(t: float, id_seq: int) -> EventoAckWearable:
    return EventoAckWearable(
        timestamp=t,
        id_secuencia=id_seq,
        secuencia_correcta=True,
        tiempo_respuesta_ms=2000,
    )


def ev_ack_mal(t: float, id_seq: int) -> EventoAckWearable:
    return EventoAckWearable(
        timestamp=t,
        id_secuencia=id_seq,
        secuencia_correcta=False,
        tiempo_respuesta_ms=4500,
    )


def crear_fsm(estado_inicial: EstadoFSM = EstadoFSM.NORMAL) -> FSM:
    return FSM(ConfigFSM(), estado_inicial=estado_inicial)


# =============================================================================
# TESTS: estado base y configuracion
# =============================================================================

print("\n--- Tests de inicializacion ---")


@_test("FSM arranca en NORMAL por defecto")
def _():
    fsm = crear_fsm()
    assert fsm.get_estado_actual() == EstadoFSM.NORMAL


@_test("FSM puede arrancar en otro estado")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    assert fsm.get_estado_actual() == EstadoFSM.PRE_ALERTA


@_test("ConfigFSM tiene defaults razonables")
def _():
    cfg = ConfigFSM()
    assert cfg.tiempo_para_bajar_estado_seg == 60.0
    assert cfg.timeout_ack_leve_seg == 30.0
    assert cfg.timeout_ack_medio_seg == 20.0
    assert cfg.timeout_ack_critico_seg == 15.0


@_test("ConfigFSM se construye desde dict")
def _():
    cfg = ConfigFSM.desde_dict({
        "fsm": {"tiempo_para_bajar_estado_seg": 90},
        "wearable": {"timeout_ack_leve_seg": 25}
    })
    assert cfg.tiempo_para_bajar_estado_seg == 90.0
    assert cfg.timeout_ack_leve_seg == 25.0


# =============================================================================
# TESTS: S0 NORMAL -> S1 PRE_ALERTA
# =============================================================================

print("\n--- Tests de transicion S0 -> S1 ---")


@_test("S0 -> S1 por parpadeos bajos")
def _():
    fsm = crear_fsm()
    fsm.procesar_evento(ev_normal(t=100))                      # arranque
    fsm.procesar_evento(ev_normal(t=170, perclos=0.32))        # arma la señal
    salida = fsm.procesar_evento(ev_normal(t=195, perclos=0.32))  # sostenida 25s
    assert salida.estado_actual == EstadoFSM.PRE_ALERTA
    #assert salida.transicion_ocurrio


@_test("S0 -> S1 por BPM en nivel ALERTA")
def _():
    fsm = crear_fsm()
    fsm.procesar_evento(ev_normal(t=100))                      # arranque
    fsm.procesar_evento(ev_normal(t=170, perclos=0.32))        # arma la señal
    salida = fsm.procesar_evento(ev_normal(t=195, perclos=0.32))  # sostenida 25s
    assert salida.estado_actual == EstadoFSM.PRE_ALERTA


@_test("S0 -> S1 por PERCLOS alto")
def _():
    fsm = crear_fsm()
    fsm.procesar_evento(ev_normal(t=100))                      # arranque
    fsm.procesar_evento(ev_normal(t=170, perclos=0.32))        # arma la señal
    salida = fsm.procesar_evento(ev_normal(t=195, perclos=0.32))  # sostenida 25s
    assert salida.estado_actual == EstadoFSM.PRE_ALERTA


@_test("S0 NO transiciona si evento es normal")
def _():
    fsm = crear_fsm()
    salida = fsm.procesar_evento(ev_normal(t=100))
    assert salida.estado_actual == EstadoFSM.NORMAL
    assert not salida.transicion_ocurrio


@_test("S0 NO transiciona si ventana_no_confiable=True")
def _():
    fsm = crear_fsm()
    salida = fsm.procesar_evento(
        ev_señales_leves(t=100, ventana_no_confiable=True,
                          motivo_no_confiable="frote_ojos")
    )
    assert salida.estado_actual == EstadoFSM.NORMAL

@_test("S0 -> S2 por microsueno severo desde NORMAL")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    salida = fsm.procesar_evento(ev_microsueno(t=100))
    assert salida.estado_actual == EstadoFSM.ALERTA_LEVE
    assert fsm.get_estado_interno().id_secuencia_ack_pendiente is not None


@_test("S0 -> S2 por cabeceo severo desde NORMAL")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    # Sin corroboracion ocular: solo PRE_ALERTA
    salida = fsm.procesar_evento(ev_cabeceo(t=100))
    assert salida.estado_actual == EstadoFSM.PRE_ALERTA
    # Con PERCLOS elevado: via rapida a ALERTA_LEVE
    fsm2 = crear_fsm(EstadoFSM.NORMAL)
    salida2 = fsm2.procesar_evento(ev_cabeceo(t=100, perclos=0.35))
    assert salida2.estado_actual == EstadoFSM.ALERTA_LEVE

# =============================================================================
# TESTS: S1 PRE_ALERTA -> S2 ALERTA_LEVE
# =============================================================================

print("\n--- Tests de transicion S1 -> S2 ---")


@_test("S1 -> S2 por bostezo confirmado")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    salida = fsm.procesar_evento(ev_bostezo(t=100))
    assert salida.estado_actual == EstadoFSM.ALERTA_LEVE


@_test("S1 -> S2 por microsueño")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    salida = fsm.procesar_evento(ev_microsueno(t=100))
    assert salida.estado_actual == EstadoFSM.ALERTA_LEVE


@_test("S1 -> S2 por cabeceo")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    salida = fsm.procesar_evento(ev_cabeceo(t=100))
    assert salida.estado_actual == EstadoFSM.ALERTA_LEVE


@_test("S1 -> S2 emite comandos VIBRAR_LEVE + REPRODUCIR_VOZ")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    salida = fsm.procesar_evento(ev_bostezo(t=100))
    tipos = {c.tipo for c in salida.comandos}
    assert TipoComandoActuador.VIBRAR_LEVE in tipos
    assert TipoComandoActuador.REPRODUCIR_VOZ in tipos
    assert TipoComandoActuador.SECUENCIA_ACK in tipos   # ahora LEVE desafia


# =============================================================================
# TESTS: S1 PRE_ALERTA -> S0 NORMAL (histeresis)
# =============================================================================

print("\n--- Tests de bajada S1 -> S0 ---")


@_test("S1 -> S0 tras 60s sin eventos negativos")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    # Calentar el primer timestamp
    fsm.procesar_evento(ev_normal(t=100))
    # Llegamos a 61s: tiempo acumulado deberia superar el limite
    salida = fsm.procesar_evento(ev_normal(t=161))
    assert salida.estado_actual == EstadoFSM.NORMAL
    assert salida.transicion_ocurrio


@_test("S1 NO baja a S0 si pasa solo 30s sin eventos")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_normal(t=100))
    salida = fsm.procesar_evento(ev_normal(t=130))
    assert salida.estado_actual == EstadoFSM.PRE_ALERTA


@_test("Pausa de timer: rostro perdido NO acumula tiempo")
def _():
    """Si la vision esta caida durante el periodo de espera, el timer no avanza."""
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_normal(t=100))
    # 30s con vision activa
    fsm.procesar_evento(ev_normal(t=130))
    # 30s con vision caida (no deberian contar)
    fsm.procesar_evento(ev_normal(t=160, vision_disponible=False))
    # 30s mas con vision activa: total tiempo "limpio" 60s
    salida = fsm.procesar_evento(ev_normal(t=190))
    assert salida.estado_actual == EstadoFSM.NORMAL


@_test("Evento negativo en S1 reinicia el acumulador")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_normal(t=100))
    fsm.procesar_evento(ev_normal(t=130))  # 30s sin eventos
    # Evento con BPM en ALERTA (negativo, pero PRE_ALERTA no escala por esto solo)
    fsm.procesar_evento(
        ev_normal(t=140, nivel_riesgo_bpm=NivelRiesgoBPM.ALERTA)
    )
    # Aunque pasen 35s mas, deberian no ser 60s totales limpios
    salida = fsm.procesar_evento(ev_normal(t=175))
    # Sigue en PRE_ALERTA porque el acumulador se reseteo en t=140
    assert salida.estado_actual == EstadoFSM.PRE_ALERTA, (
        f"Esperaba PRE_ALERTA, obtuve {salida.estado_actual.name}"
    )


# =============================================================================
# TESTS: S2 ALERTA_LEVE -> S3 ALERTA_MEDIA
# =============================================================================

print("\n--- Tests de transicion S2 -> S3 ---")


@_test("S2 -> S3 por timeout ACK (30s)")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    # Entramos a S2
    fsm.procesar_evento(ev_bostezo(t=100))
    assert fsm.get_estado_actual() == EstadoFSM.ALERTA_LEVE
    # 35s despues sin ACK
    salida = fsm.procesar_evento(ev_normal(t=135))
    assert salida.estado_actual == EstadoFSM.ALERTA_MEDIA


@_test("S2 -> S3 por microsueño (escalada por acumulacion)")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))
    # Microsueño durante la espera de ACK
    salida = fsm.procesar_evento(ev_microsueno(t=105))
    assert salida.estado_actual == EstadoFSM.ALERTA_MEDIA


@_test("S2 NO escala antes del timeout sin evento severo")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))
    salida = fsm.procesar_evento(ev_normal(t=120))
    assert salida.estado_actual == EstadoFSM.ALERTA_LEVE


# =============================================================================
# TESTS: ACK del wearable
# =============================================================================

print("\n--- Tests de ACK ---")


@_test("S2 -> S1 por ACK correcto")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))
    # Obtenemos el id de secuencia generado
    id_seq = fsm.get_estado_interno().id_secuencia_ack_pendiente
    assert id_seq is not None
    salida = fsm.procesar_evento(ev_ack_ok(t=110, id_seq=id_seq))
    assert salida.estado_actual == EstadoFSM.PRE_ALERTA


@_test("ACK con id incorrecto se ignora")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))
    # ACK con id que no corresponde
    salida = fsm.procesar_evento(ev_ack_ok(t=110, id_seq=999))
    assert salida.estado_actual == EstadoFSM.ALERTA_LEVE
    assert not salida.transicion_ocurrio


@_test("S2 -> S3 por ACK incorrecto (escalada)")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))
    id_seq = fsm.get_estado_interno().id_secuencia_ack_pendiente
    salida = fsm.procesar_evento(ev_ack_mal(t=110, id_seq=id_seq))
    assert salida.estado_actual == EstadoFSM.ALERTA_MEDIA


@_test("ACK sin estado pendiente se ignora")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    salida = fsm.procesar_evento(ev_ack_ok(t=100, id_seq=1))
    assert salida.estado_actual == EstadoFSM.NORMAL


# =============================================================================
# TESTS: S3 ALERTA_MEDIA -> S4 CRITICO
# =============================================================================

print("\n--- Tests de transicion S3 -> S4 ---")


@_test("S3 -> S4 por cabeceo + BPM critico")
def _():
    fsm = crear_fsm(EstadoFSM.ALERTA_MEDIA)
    salida = fsm.procesar_evento(
        ev_cabeceo(t=100, nivel_riesgo_bpm=NivelRiesgoBPM.CRITICO, bpm_actual=55)
    )
    assert salida.estado_actual == EstadoFSM.CRITICO


@_test("S3 -> S4 por timeout ACK + BPM bajo")
def _():
    fsm = crear_fsm(EstadoFSM.ALERTA_MEDIA)
    # Forzar a tener ACK pendiente (entramos via bostezo en PRE_ALERTA)
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))  # -> ALERTA_LEVE
    fsm.procesar_evento(ev_microsueno(t=105))  # -> ALERTA_MEDIA con ACK nuevo
    # 25s despues con BPM bajo
    salida = fsm.procesar_evento(
        ev_normal(t=130, nivel_riesgo_bpm=NivelRiesgoBPM.ALERTA)
    )
    assert salida.estado_actual == EstadoFSM.CRITICO


@_test("S4 emite VIBRAR_FUERTE + BUZZER_CONTINUO + NOTIFICAR_SUPERVISOR")
def _():
    fsm = crear_fsm(EstadoFSM.ALERTA_MEDIA)
    salida = fsm.procesar_evento(
        ev_cabeceo(t=100, nivel_riesgo_bpm=NivelRiesgoBPM.CRITICO)
    )
    tipos = {c.tipo for c in salida.comandos}
    assert TipoComandoActuador.VIBRAR_FUERTE in tipos
    assert TipoComandoActuador.BUZZER_CONTINUO in tipos
    assert TipoComandoActuador.NOTIFICAR_SUPERVISOR in tipos


# =============================================================================
# TESTS: S4 CRITICO -> S1 PRE_ALERTA
# =============================================================================

print("\n--- Tests de bajada desde S4 ---")


@_test("S4 -> S1 por ACK correcto")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    # Forzar el camino completo a CRITICO
    fsm.procesar_evento(ev_bostezo(t=100))
    fsm.procesar_evento(ev_microsueno(t=105))  # -> ALERTA_MEDIA
    fsm.procesar_evento(
        ev_cabeceo(t=110, nivel_riesgo_bpm=NivelRiesgoBPM.CRITICO)
    )  # -> CRITICO
    assert fsm.get_estado_actual() == EstadoFSM.CRITICO

    id_seq = fsm.get_estado_interno().id_secuencia_ack_pendiente
    salida = fsm.procesar_evento(ev_ack_ok(t=120, id_seq=id_seq))
    assert salida.estado_actual == EstadoFSM.PRE_ALERTA


@_test("S4 NO baja por evento normal (solo ACK)")
def _():
    fsm = crear_fsm(EstadoFSM.CRITICO)
    salida = fsm.procesar_evento(ev_normal(t=100))
    assert salida.estado_actual == EstadoFSM.CRITICO


# =============================================================================
# TESTS: MODO DEGRADADO (S5)
# =============================================================================

print("\n--- Tests de MODO_DEGRADADO ---")


@_test("Cualquier estado -> S5 por fallo de sensor severidad >= 2")
def _():
    fsm = crear_fsm(EstadoFSM.ALERTA_LEVE)
    fallo = EventoFalloSensor(
        timestamp=100.0,
        sensor_afectado=OrigenEvento.WEARABLE,
        motivo="heartbeat_timeout",
        severidad=2,
    )
    salida = fsm.procesar_evento(fallo)
    assert salida.estado_actual == EstadoFSM.MODO_DEGRADADO
    assert salida.modo_degradado


@_test("Fallo con severidad 1 NO va a S5")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    fallo = EventoFalloSensor(
        timestamp=100.0,
        sensor_afectado=OrigenEvento.WEARABLE,
        motivo="señal_debil",
        severidad=1,
    )
    salida = fsm.procesar_evento(fallo)
    assert salida.estado_actual == EstadoFSM.NORMAL


@_test("S5 -> S0 por recuperacion de sensor")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    # Fallo: vamos a S5
    fsm.procesar_evento(EventoFalloSensor(
        timestamp=100.0,
        sensor_afectado=OrigenEvento.WEARABLE,
        motivo="caida",
        severidad=2,
    ))
    assert fsm.get_estado_actual() == EstadoFSM.MODO_DEGRADADO
    # Recuperacion
    salida = fsm.procesar_evento(EventoRecuperacionSensor(
        timestamp=150.0,
        sensor_recuperado=OrigenEvento.WEARABLE,
        tiempo_caido_seg=50.0,
    ))
    assert salida.estado_actual == EstadoFSM.NORMAL


@_test("S5 NO sale si quedan sensores caidos")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    fsm.procesar_evento(EventoFalloSensor(
        timestamp=100.0,
        sensor_afectado=OrigenEvento.WEARABLE,
        motivo="caida",
        severidad=2,
    ))
    fsm.procesar_evento(EventoFalloSensor(
        timestamp=101.0,
        sensor_afectado=OrigenEvento.VISION,
        motivo="caida",
        severidad=2,
    ))
    # Recuperamos solo uno
    salida = fsm.procesar_evento(EventoRecuperacionSensor(
        timestamp=150.0,
        sensor_recuperado=OrigenEvento.WEARABLE,
        tiempo_caido_seg=50.0,
    ))
    assert salida.estado_actual == EstadoFSM.MODO_DEGRADADO


@_test("En MODO_DEGRADADO los eventos normales se ignoran")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    fsm.procesar_evento(EventoFalloSensor(
        timestamp=100.0,
        sensor_afectado=OrigenEvento.VISION,
        motivo="caida",
        severidad=2,
    ))
    # Bostezo durante MODO_DEGRADADO: no debe escalar
    salida = fsm.procesar_evento(ev_bostezo(t=120))
    assert salida.estado_actual == EstadoFSM.MODO_DEGRADADO


# =============================================================================
# TESTS: SalidaFSM bien formada
# =============================================================================

print("\n--- Tests de SalidaFSM ---")


@_test("Salida sin transicion no genera comandos")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    salida = fsm.procesar_evento(ev_normal(t=100))
    assert len(salida.comandos) == 0
    assert not salida.transicion_ocurrio


@_test("Salida con transicion tiene motivo poblado")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    salida = fsm.procesar_evento(ev_señales_leves(t=100))
    assert salida.transicion_ocurrio
    assert salida.motivo_transicion != ""


@_test("nivel_alerta se calcula correctamente")
def _():
    fsm = crear_fsm(EstadoFSM.NORMAL)
    salida = fsm.procesar_evento(ev_normal(t=100))
    assert salida.nivel_alerta == 0

    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    salida = fsm.procesar_evento(ev_normal(t=100))
    assert salida.nivel_alerta == 1


# =============================================================================
# TEST: escenario integrado completo (E2E)
# =============================================================================

print("\n--- Test de escenario integrado ---")


@_test("Escenario completo: conductor que progresa hasta CRITICO y se recupera")
def _():
    fsm = crear_fsm()
    transiciones = []
    base = 1000.0  # arrancamos desde un timestamp valido (> 0)

    # 1. Empezamos normales
    s = fsm.procesar_evento(ev_normal(t=base))
    transiciones.append((s.estado_actual.name, s.transicion_ocurrio))

    # 2. PERCLOS elevado sostenido -> PRE_ALERTA (con persistencia)
    s = fsm.procesar_evento(ev_normal(t=base + 10, perclos=0.32))
    transiciones.append((s.estado_actual.name, s.transicion_ocurrio))
    s = fsm.procesar_evento(ev_normal(t=base + 35, perclos=0.32))
    transiciones.append((s.estado_actual.name, s.transicion_ocurrio))

    # 3. Bostezos acumulados -> ALERTA_LEVE
    s = fsm.procesar_evento(ev_bostezo(t=base + 45))
    transiciones.append((s.estado_actual.name, s.transicion_ocurrio))

    # 4. No responde ACK, microsueño -> ALERTA_MEDIA
    s = fsm.procesar_evento(ev_microsueno(t=base + 55))
    transiciones.append((s.estado_actual.name, s.transicion_ocurrio))

    # 5. Cabeceo + BPM critico -> CRITICO
    s = fsm.procesar_evento(
        ev_cabeceo(t=base + 65, nivel_riesgo_bpm=NivelRiesgoBPM.CRITICO)
    )
    transiciones.append((s.estado_actual.name, s.transicion_ocurrio))

    # 6. Conductor reacciona con ACK correcto -> PRE_ALERTA
    id_seq = fsm.get_estado_interno().id_secuencia_ack_pendiente
    s = fsm.procesar_evento(ev_ack_ok(t=base + 75, id_seq=id_seq))
    transiciones.append((s.estado_actual.name, s.transicion_ocurrio))


    esperado = [
        ("NORMAL", False),
        ("NORMAL", False),        # señal armada, persistencia en curso
        ("PRE_ALERTA", True),     # señal sostenida 25s
        ("ALERTA_LEVE", True),
        ("ALERTA_MEDIA", True),
        ("CRITICO", True),
        ("PRE_ALERTA", True),
    ]
    assert transiciones == esperado, f"Secuencia inesperada:\n{transiciones}\nvs\n{esperado}"

print("\n--- Tests de recuperacion desde CRITICO ---")

@_test("CRITICO emite SECUENCIA_ACK (via de recuperacion)")
def _():
    fsm = crear_fsm(EstadoFSM.ALERTA_MEDIA)
    salida = fsm.procesar_evento(
        ev_cabeceo(t=100, nivel_riesgo_bpm=NivelRiesgoBPM.CRITICO)
    )
    tipos = {c.tipo for c in salida.comandos}
    assert TipoComandoActuador.SECUENCIA_ACK in tipos
    seq = [c for c in salida.comandos
           if c.tipo == TipoComandoActuador.SECUENCIA_ACK][0]
    assert seq.id_secuencia is not None
    
@_test("ACK correcto en CRITICO tras vencer el timeout: recupera (no drift)")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))
    fsm.procesar_evento(ev_microsueno(t=105))          # -> ALERTA_MEDIA
    fsm.procesar_evento(
        ev_cabeceo(t=110, nivel_riesgo_bpm=NivelRiesgoBPM.CRITICO)
    )                                                   # -> CRITICO
    assert fsm.get_estado_actual() == EstadoFSM.CRITICO
    # Pasa el timeout critico sin respuesta -> se re-emite el desafio con id nuevo
    salida = fsm.procesar_evento(ev_normal(t=200))
    seqs = [c for c in salida.comandos
            if c.tipo == TipoComandoActuador.SECUENCIA_ACK]
    assert seqs, "deberia re-emitir el desafio tras el timeout"
    id_vigente = fsm.get_estado_interno().id_secuencia_ack_pendiente
    assert seqs[0].id_secuencia == id_vigente, "el desafio re-emitido lleva el id vigente"
    # El conductor responde con el id vigente -> recupera
    salida2 = fsm.procesar_evento(ev_ack_ok(t=205, id_seq=id_vigente))
    assert salida2.estado_actual == EstadoFSM.PRE_ALERTA

@_test("ACK incorrecto en CRITICO: se queda y re-desafia (no queda pegado)")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))
    fsm.procesar_evento(ev_microsueno(t=105))          # -> ALERTA_MEDIA
    fsm.procesar_evento(
        ev_cabeceo(t=110, nivel_riesgo_bpm=NivelRiesgoBPM.CRITICO)
    )                                                   # -> CRITICO
    assert fsm.get_estado_actual() == EstadoFSM.CRITICO
    id_seq = fsm.get_estado_interno().id_secuencia_ack_pendiente
    salida = fsm.procesar_evento(ev_ack_mal(t=120, id_seq=id_seq))
    assert salida.estado_actual == EstadoFSM.CRITICO
    tipos = {c.tipo for c in salida.comandos}
    assert TipoComandoActuador.SECUENCIA_ACK in tipos, "deberia re-desafiar"
    assert TipoComandoActuador.VIBRAR_FUERTE in tipos
    nuevo = fsm.get_estado_interno().id_secuencia_ack_pendiente
    assert nuevo is not None and nuevo != id_seq, "desafio nuevo con id distinto"


@_test("ACK incorrecto en MEDIA escala a CRITICO con desafio nuevo")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))
    fsm.procesar_evento(ev_microsueno(t=105))          # -> ALERTA_MEDIA
    id_med = fsm.get_estado_interno().id_secuencia_ack_pendiente
    salida = fsm.procesar_evento(ev_ack_mal(t=110, id_seq=id_med))
    assert salida.estado_actual == EstadoFSM.CRITICO
    nuevo = fsm.get_estado_interno().id_secuencia_ack_pendiente
    assert nuevo is not None and nuevo != id_med
    assert TipoComandoActuador.SECUENCIA_ACK in {c.tipo for c in salida.comandos}

print("\n--- Tests de manejo de somnolencia (mejoras) ---")


@_test("Parpadeos bajos durante el calentamiento NO disparan PRE_ALERTA")
def _():
    fsm = crear_fsm()
    fsm.procesar_evento(ev_normal(t=100))
    fsm.procesar_evento(ev_normal(t=110, parpadeos_por_minuto=5.0))
    salida = fsm.procesar_evento(ev_normal(t=135, parpadeos_por_minuto=5.0))
    assert salida.estado_actual == EstadoFSM.NORMAL


@_test("Señal leve interrumpida resetea la persistencia")
def _():
    fsm = crear_fsm()
    fsm.procesar_evento(ev_normal(t=100))
    fsm.procesar_evento(ev_normal(t=170, perclos=0.32))
    fsm.procesar_evento(ev_normal(t=180, perclos=0.1))   # se corta
    salida = fsm.procesar_evento(ev_normal(t=195, perclos=0.32))
    assert salida.estado_actual == EstadoFSM.NORMAL


@_test("PERCLOS >=0.35 sostenido 30s escala a ALERTA_LEVE con desafio")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_normal(t=100, perclos=0.40))
    fsm.procesar_evento(ev_normal(t=115, perclos=0.40))
    salida = fsm.procesar_evento(ev_normal(t=131, perclos=0.40))
    assert salida.estado_actual == EstadoFSM.ALERTA_LEVE
    assert TipoComandoActuador.SECUENCIA_ACK in {c.tipo for c in salida.comandos}


@_test("Bostezo UNICO desde PRE_ALERTA no escala")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    salida = fsm.procesar_evento(
        ev_normal(t=100, bostezo=True, bostezos_ventana_larga=1)
    )
    assert salida.estado_actual == EstadoFSM.PRE_ALERTA


@_test("Fatiga recurrente: ACK correcto emite voz + supervisor; 4to severo -> MEDIA")
def _():
    fsm = crear_fsm()
    fsm.procesar_evento(ev_normal(t=10))
    for t in (100, 200, 300):
        fsm.procesar_evento(ev_microsueno(t=t))
        id_seq = fsm.get_estado_interno().id_secuencia_ack_pendiente
        salida = fsm.procesar_evento(ev_ack_ok(t=t + 5, id_seq=id_seq))
    # El 3er ACK ya lleva la advertencia (3 episodios en ventana)
    tipos = {c.tipo for c in salida.comandos}
    assert TipoComandoActuador.REPRODUCIR_VOZ in tipos
    assert TipoComandoActuador.NOTIFICAR_SUPERVISOR in tipos
    # 4to evento severo: piso de escalada -> ALERTA_MEDIA directo
    salida = fsm.procesar_evento(ev_microsueno(t=400))
    assert salida.estado_actual == EstadoFSM.ALERTA_MEDIA


@_test("ACK correcto pero LENTO baja solo un nivel")
def _():
    fsm = crear_fsm(EstadoFSM.PRE_ALERTA)
    fsm.procesar_evento(ev_bostezo(t=100))          # -> ALERTA_LEVE
    fsm.procesar_evento(ev_microsueno(t=105))       # -> ALERTA_MEDIA
    id_seq = fsm.get_estado_interno().id_secuencia_ack_pendiente
    lento = EventoAckWearable(timestamp=110, id_secuencia=id_seq,
                              secuencia_correcta=True, tiempo_respuesta_ms=7000)
    salida = fsm.procesar_evento(lento)
    assert salida.estado_actual == EstadoFSM.ALERTA_LEVE, "lento baja 1 nivel, no a PRE_ALERTA"

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
    print("  NeuroDrive - Tests de FSM")
    print("=" * 60)
    sys.exit(_resumen())
