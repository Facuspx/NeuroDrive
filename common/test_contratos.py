"""
test_contratos.py - Pruebas funcionales del modulo de contratos.

Ejecutar con:
    cd /ruta/al/proyecto/NeuroDrive
    python -m common.test_contratos

Valida:
  1. Que todos los enums se crean correctamente y son comparables.
  2. Que cada dataclass se puede crear, serializar a JSON y deserializar
     sin perder informacion.
  3. Que la validacion en __post_init__ rechaza valores invalidos.
  4. Que los IntEnum se serializan correctamente como enteros.

Este test NO depende de hardware, NO depende de la camara, NO depende
del wearable. Se puede correr en cualquier maquina con Python 3.10+.
"""

from __future__ import annotations

import sys
import time
import json
import traceback

# Forzar UTF-8 en stdout/stderr para evitar errores en sistemas con locale
# distinto de UTF-8 (Raspberry Pi con LANG=POSIX, contenedores minimalistas, etc.)
# Esto se hace antes de cualquier print() y es idempotente.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, Exception):
    # Si por algun motivo no se puede reconfigurar, seguimos.
    # En ese caso el codigo evita caracteres no-ASCII en sus prints.
    pass

# Permitir ejecutar tanto con `python -m common.test_contratos`
# como con `python test_contratos.py` desde dentro de la carpeta common/
try:
    from common.contratos import (
        EstadoFSM,
        NivelRiesgoBPM,
        TipoComandoActuador,
        OrigenEvento,
        TipoMensaje,
        EventoVision,
        EventoWearable,
        EventoAckWearable,
        EventoFalloSensor,
        EventoRecuperacionSensor,
        EventoProcesado,
        ComandoActuador,
        SalidaFSM,
        EstadoSesion,
        Envelope,
        timestamp_actual,
        generar_id_sesion,
        generar_id_mensaje,
    )
except ImportError:
    # Fallback si se ejecuta desde otra ubicación
    from contratos import (  # type: ignore
        EstadoFSM,
        NivelRiesgoBPM,
        TipoComandoActuador,
        OrigenEvento,
        TipoMensaje,
        EventoVision,
        EventoWearable,
        EventoAckWearable,
        EventoFalloSensor,
        EventoRecuperacionSensor,
        EventoProcesado,
        ComandoActuador,
        SalidaFSM,
        EstadoSesion,
        Envelope,
        timestamp_actual,
        generar_id_sesion,
        generar_id_mensaje,
    )


# =============================================================================
# Pequeño framework de testeo sin dependencias externas
# =============================================================================

_resultados: list[tuple[str, bool, str]] = []


def _test(nombre: str):
    """Decorador para registrar tests."""
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


def _debe_fallar(func, mensaje_test: str) -> None:
    """Verifica que `func()` lanza una excepción esperada."""
    from dataclasses import FrozenInstanceError
    try:
        func()
    except (ValueError, TypeError, FrozenInstanceError):
        return
    raise AssertionError(f"{mensaje_test} (no lanzo excepcion)")


# =============================================================================
#                          TESTS DE ENUMS
# =============================================================================

print("\n--- Tests de enums ---")


@_test("EstadoFSM tiene 6 estados con valores 0-5")
def _():
    valores = [int(e) for e in EstadoFSM]
    assert valores == [0, 1, 2, 3, 4, 5], f"Valores inesperados: {valores}"


@_test("EstadoFSM permite comparación jerárquica")
def _():
    assert EstadoFSM.NORMAL < EstadoFSM.ALERTA_LEVE
    assert EstadoFSM.CRITICO > EstadoFSM.ALERTA_MEDIA
    assert EstadoFSM.ALERTA_MEDIA >= EstadoFSM.ALERTA_LEVE


@_test("NivelRiesgoBPM ordenado correctamente")
def _():
    assert NivelRiesgoBPM.NORMAL < NivelRiesgoBPM.ALERTA < NivelRiesgoBPM.CRITICO


@_test("TipoComandoActuador tiene los 11 comandos")
def _():
    valores = [int(c) for c in TipoComandoActuador]
    assert len(valores) == 11, f"Esperaba 11 comandos, hay {len(valores)}"


@_test("OrigenEvento tiene 3 orígenes")
def _():
    assert len(list(OrigenEvento)) == 3


# =============================================================================
#                       TESTS DE EVENTOVISION
# =============================================================================

print("\n--- Tests de EventoVision ---")


@_test("EventoVision se crea con campos mínimos")
def _():
    ev = EventoVision(timestamp=timestamp_actual(), rostro_detectado=True)
    assert ev.rostro_detectado is True
    assert ev.ear_izquierdo is None


@_test("EventoVision calcula EAR promedio correctamente")
def _():
    ev = EventoVision(
        timestamp=timestamp_actual(),
        rostro_detectado=True,
        ear_izquierdo=0.20,
        ear_derecho=0.30,
    )
    assert abs(ev.ear_promedio - 0.25) < 1e-9


@_test("EventoVision retorna None en EAR promedio si falta un ojo")
def _():
    ev = EventoVision(
        timestamp=timestamp_actual(),
        rostro_detectado=True,
        ear_izquierdo=0.20,
    )
    assert ev.ear_promedio is None


@_test("EventoVision rechaza timestamp negativo")
def _():
    _debe_fallar(
        lambda: EventoVision(timestamp=-1.0, rostro_detectado=True),
        "timestamp negativo debería fallar",
    )


@_test("EventoVision rechaza confianza fuera de [0,1]")
def _():
    _debe_fallar(
        lambda: EventoVision(
            timestamp=timestamp_actual(),
            rostro_detectado=True,
            confianza_deteccion=1.5,
        ),
        "confianza > 1 debería fallar",
    )


@_test("EventoVision serializa y deserializa de JSON")
def _():
    ev1 = EventoVision(
        timestamp=12345.67,
        rostro_detectado=True,
        ear_izquierdo=0.21,
        ear_derecho=0.23,
        mar=0.4,
        pitch_grados=-3.5,
        yaw_grados=2.1,
        roll_grados=0.5,
        frote_ojos_activo=False,
        confianza_deteccion=0.95,
    )
    payload = ev1.to_json()
    ev2 = EventoVision.from_json(payload)
    assert ev1 == ev2, f"Round-trip falló:\n  {ev1}\n  {ev2}"


@_test("EventoVision es inmutable (frozen)")
def _():
    ev = EventoVision(timestamp=timestamp_actual(), rostro_detectado=True)
    _debe_fallar(
        lambda: setattr(ev, "rostro_detectado", False),
        "asignación a campo de frozen debería fallar",
    )


# =============================================================================
#                       TESTS DE EVENTOWEARABLE
# =============================================================================

print("\n--- Tests de EventoWearable ---")


@_test("EventoWearable se crea con BPM válido")
def _():
    ev = EventoWearable(timestamp=timestamp_actual(), bpm=72)
    assert ev.bpm == 72


@_test("EventoWearable rechaza BPM fuera de rango fisiológico")
def _():
    _debe_fallar(
        lambda: EventoWearable(timestamp=timestamp_actual(), bpm=10),
        "BPM=10 debería fallar",
    )
    _debe_fallar(
        lambda: EventoWearable(timestamp=timestamp_actual(), bpm=300),
        "BPM=300 debería fallar",
    )


@_test("EventoWearable rechaza batería fuera de [0,100]")
def _():
    _debe_fallar(
        lambda: EventoWearable(timestamp=timestamp_actual(), bateria_porcentaje=150),
        "batería=150 debería fallar",
    )


@_test("EventoWearable serializa y deserializa")
def _():
    ev1 = EventoWearable(
        timestamp=999.5,
        bpm=68,
        ack_recibido=True,
        secuencia_replicada=True,
        bateria_porcentaje=85,
        id_paquete=42,
    )
    ev2 = EventoWearable.from_json(ev1.to_json())
    assert ev1 == ev2


# =============================================================================
#                       TESTS DE EVENTOPROCESADO
# =============================================================================

print("\n--- Tests de EventoProcesado ---")


@_test("EventoProcesado se crea con valores por defecto")
def _():
    ev = EventoProcesado(timestamp=timestamp_actual())
    assert ev.microsueno is False
    assert ev.nivel_riesgo_bpm == NivelRiesgoBPM.DESCONOCIDO


@_test("EventoProcesado rechaza PERCLOS fuera de [0,1]")
def _():
    _debe_fallar(
        lambda: EventoProcesado(timestamp=timestamp_actual(), perclos=1.5),
        "PERCLOS=1.5 debería fallar",
    )


@_test("EventoProcesado serializa el enum NivelRiesgoBPM como int")
def _():
    ev = EventoProcesado(
        timestamp=100.0,
        bpm_actual=55,
        nivel_riesgo_bpm=NivelRiesgoBPM.CRITICO,
    )
    payload = ev.to_json()
    parsed = json.loads(payload)
    assert parsed["nivel_riesgo_bpm"] == 3
    ev2 = EventoProcesado.from_json(payload)
    assert ev2.nivel_riesgo_bpm == NivelRiesgoBPM.CRITICO


# =============================================================================
#                       TESTS DE COMANDOACTUADOR
# =============================================================================

print("\n--- Tests de ComandoActuador ---")


@_test("ComandoActuador se crea correctamente")
def _():
    c = ComandoActuador(
        tipo=TipoComandoActuador.VIBRAR_MEDIO,
        intensidad=60,
        duracion_ms=500,
    )
    assert c.intensidad == 60


@_test("ComandoActuador rechaza intensidad fuera de [0,100]")
def _():
    _debe_fallar(
        lambda: ComandoActuador(
            tipo=TipoComandoActuador.VIBRAR_LEVE,
            intensidad=150,
        ),
        "intensidad=150 debería fallar",
    )


@_test("ComandoActuador serializa con tipo como int")
def _():
    c = ComandoActuador(
        tipo=TipoComandoActuador.REPRODUCIR_VOZ,
        mensaje_voz="Atención conductor",
    )
    parsed = json.loads(c.to_json())
    assert parsed["tipo"] == 7
    assert parsed["mensaje_voz"] == "Atención conductor"


# =============================================================================
#                          TESTS DE SALIDAFSM
# =============================================================================

print("\n--- Tests de SalidaFSM ---")


@_test("SalidaFSM se crea con comandos")
def _():
    salida = SalidaFSM(
        timestamp=timestamp_actual(),
        estado_actual=EstadoFSM.ALERTA_LEVE,
        estado_anterior=EstadoFSM.PRE_ALERTA,
        nivel_alerta=2,
        comandos=(
            ComandoActuador(tipo=TipoComandoActuador.VIBRAR_LEVE, intensidad=30),
            ComandoActuador(
                tipo=TipoComandoActuador.REPRODUCIR_VOZ,
                mensaje_voz="Atención, signos de fatiga",
            ),
        ),
        transicion_ocurrio=True,
        motivo_transicion="primer bostezo confirmado",
    )
    assert len(salida.comandos) == 2
    assert salida.transicion_ocurrio


@_test("SalidaFSM rechaza nivel_alerta fuera de [0,4]")
def _():
    _debe_fallar(
        lambda: SalidaFSM(
            timestamp=timestamp_actual(),
            estado_actual=EstadoFSM.NORMAL,
            estado_anterior=EstadoFSM.NORMAL,
            nivel_alerta=10,
        ),
        "nivel_alerta=10 debería fallar",
    )


@_test("SalidaFSM serializa y deserializa correctamente con comandos anidados")
def _():
    s1 = SalidaFSM(
        timestamp=2000.0,
        estado_actual=EstadoFSM.CRITICO,
        estado_anterior=EstadoFSM.ALERTA_MEDIA,
        nivel_alerta=4,
        comandos=(
            ComandoActuador(tipo=TipoComandoActuador.VIBRAR_FUERTE, intensidad=100),
            ComandoActuador(tipo=TipoComandoActuador.BUZZER_CONTINUO, intensidad=80),
        ),
        transicion_ocurrio=True,
        motivo_transicion="timeout ACK + BPM crítico",
    )
    payload = s1.to_json()
    s2 = SalidaFSM.from_dict(json.loads(payload))
    assert s1 == s2, f"Round-trip falló:\n  {s1}\n  {s2}"


# =============================================================================
#                         TESTS DE ESTADOSESION
# =============================================================================

print("\n--- Tests de EstadoSesion ---")


@_test("EstadoSesion se crea con contadores")
def _():
    ahora = timestamp_actual()
    s = EstadoSesion(
        timestamp_guardado=ahora,
        estado_fsm=EstadoFSM.ALERTA_LEVE,
        bostezos_recientes=(ahora - 100, ahora - 50),
        microsuenos_recientes=(),
        cabeceos_recientes=(ahora - 30,),
        motivo_guardado="apagado manual",
    )
    assert len(s.bostezos_recientes) == 2
    assert s.estado_fsm == EstadoFSM.ALERTA_LEVE


@_test("EstadoSesion serializa y deserializa")
def _():
    s1 = EstadoSesion(
        timestamp_guardado=time.time(),
        estado_fsm=EstadoFSM.PRE_ALERTA,
        bostezos_recientes=(1.0, 2.0, 3.0),
        microsuenos_recientes=(10.5,),
        cabeceos_recientes=(),
        motivo_guardado="test",
    )
    s2 = EstadoSesion.from_json(s1.to_json())
    assert s1 == s2


# =============================================================================
#                       TESTS DE ENUM TIPOMENSAJE
# =============================================================================

print("\n--- Tests de TipoMensaje ---")


@_test("TipoMensaje tiene los 7 tipos esperados")
def _():
    valores = [int(t) for t in TipoMensaje]
    assert len(valores) == 7, f"Esperaba 7 tipos, hay {len(valores)}"


@_test("TipoMensaje.EVENTO_VISION es 1")
def _():
    assert int(TipoMensaje.EVENTO_VISION) == 1


# =============================================================================
#                       TESTS DE EVENTOACKWEARABLE
# =============================================================================

print("\n--- Tests de EventoAckWearable ---")


@_test("EventoAckWearable se crea correctamente")
def _():
    ev = EventoAckWearable(
        timestamp=timestamp_actual(),
        id_secuencia=7,
        secuencia_correcta=True,
        tiempo_respuesta_ms=2300,
    )
    assert ev.secuencia_correcta is True
    assert ev.id_secuencia == 7


@_test("EventoAckWearable rechaza id_secuencia negativo")
def _():
    _debe_fallar(
        lambda: EventoAckWearable(
            timestamp=timestamp_actual(),
            id_secuencia=-1,
            secuencia_correcta=True,
            tiempo_respuesta_ms=1000,
        ),
        "id_secuencia=-1 debería fallar",
    )


@_test("EventoAckWearable rechaza tiempo_respuesta negativo")
def _():
    _debe_fallar(
        lambda: EventoAckWearable(
            timestamp=timestamp_actual(),
            id_secuencia=1,
            secuencia_correcta=True,
            tiempo_respuesta_ms=-50,
        ),
        "tiempo_respuesta negativo debería fallar",
    )


@_test("EventoAckWearable serializa y deserializa")
def _():
    ev1 = EventoAckWearable(
        timestamp=1234.5,
        id_secuencia=3,
        secuencia_correcta=False,
        tiempo_respuesta_ms=4500,
    )
    ev2 = EventoAckWearable.from_json(ev1.to_json())
    assert ev1 == ev2


# =============================================================================
#                       TESTS DE EVENTOFALLOSENSOR
# =============================================================================

print("\n--- Tests de EventoFalloSensor ---")


@_test("EventoFalloSensor se crea correctamente")
def _():
    ev = EventoFalloSensor(
        timestamp=timestamp_actual(),
        sensor_afectado=OrigenEvento.WEARABLE,
        motivo="heartbeat_timeout",
        severidad=2,
    )
    assert ev.sensor_afectado == OrigenEvento.WEARABLE


@_test("EventoFalloSensor rechaza severidad fuera de [1,3]")
def _():
    _debe_fallar(
        lambda: EventoFalloSensor(
            timestamp=timestamp_actual(),
            sensor_afectado=OrigenEvento.VISION,
            motivo="cámara perdida",
            severidad=5,
        ),
        "severidad=5 debería fallar",
    )


@_test("EventoFalloSensor rechaza INTERNO como sensor")
def _():
    _debe_fallar(
        lambda: EventoFalloSensor(
            timestamp=timestamp_actual(),
            sensor_afectado=OrigenEvento.INTERNO,
            motivo="algo",
            severidad=1,
        ),
        "INTERNO no es un sensor válido",
    )


@_test("EventoFalloSensor rechaza motivo vacío")
def _():
    _debe_fallar(
        lambda: EventoFalloSensor(
            timestamp=timestamp_actual(),
            sensor_afectado=OrigenEvento.VISION,
            motivo="",
            severidad=2,
        ),
        "motivo vacío debería fallar",
    )


@_test("EventoFalloSensor serializa el enum como int")
def _():
    ev = EventoFalloSensor(
        timestamp=100.0,
        sensor_afectado=OrigenEvento.WEARABLE,
        motivo="timeout",
        severidad=3,
    )
    parsed = json.loads(ev.to_json())
    assert parsed["sensor_afectado"] == 1   # WEARABLE = 1
    ev2 = EventoFalloSensor.from_json(ev.to_json())
    assert ev2.sensor_afectado == OrigenEvento.WEARABLE


# =============================================================================
#                  TESTS DE EVENTORECUPERACIONSENSOR
# =============================================================================

print("\n--- Tests de EventoRecuperacionSensor ---")


@_test("EventoRecuperacionSensor se crea correctamente")
def _():
    ev = EventoRecuperacionSensor(
        timestamp=timestamp_actual(),
        sensor_recuperado=OrigenEvento.WEARABLE,
        tiempo_caido_seg=23.5,
    )
    assert ev.tiempo_caido_seg == 23.5


@_test("EventoRecuperacionSensor rechaza tiempo_caido negativo")
def _():
    _debe_fallar(
        lambda: EventoRecuperacionSensor(
            timestamp=timestamp_actual(),
            sensor_recuperado=OrigenEvento.VISION,
            tiempo_caido_seg=-1.0,
        ),
        "tiempo_caido negativo debería fallar",
    )


@_test("EventoRecuperacionSensor serializa y deserializa")
def _():
    ev1 = EventoRecuperacionSensor(
        timestamp=500.0,
        sensor_recuperado=OrigenEvento.VISION,
        tiempo_caido_seg=10.0,
    )
    ev2 = EventoRecuperacionSensor.from_json(ev1.to_json())
    assert ev1 == ev2


# =============================================================================
#                          TESTS DE ENVELOPE
# =============================================================================

print("\n--- Tests de Envelope ---")


@_test("Envelope se crea en modo externo con payload_json")
def _():
    ev = EventoVision(timestamp=100.0, rostro_detectado=True)
    env = Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo="cam-01",
        id_sesion="ses-20250512-001",
        id_mensaje="vis-00001",
        numero_secuencia=1,
        timestamp_origen=100.0,
        payload_json=ev.to_json(),
    )
    assert env.payload_json
    assert env.evento is None


@_test("Envelope se crea en modo interno con evento")
def _():
    ev = EventoVision(timestamp=100.0, rostro_detectado=True)
    env = Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo="cam-01",
        id_sesion="ses-20250512-001",
        id_mensaje="vis-00002",
        numero_secuencia=2,
        timestamp_origen=100.0,
        evento=ev,
    )
    assert env.evento is ev
    assert env.payload_json == ""


@_test("Envelope rechaza si no hay payload_json NI evento")
def _():
    _debe_fallar(
        lambda: Envelope(
            tipo=TipoMensaje.EVENTO_VISION,
            origen=OrigenEvento.VISION,
            id_dispositivo="cam-01",
            id_sesion="ses-001",
            id_mensaje="vis-00001",
            numero_secuencia=1,
            timestamp_origen=100.0,
            # ni payload_json ni evento
        ),
        "envelope vacío debería fallar",
    )


@_test("Envelope rechaza id_dispositivo vacío")
def _():
    _debe_fallar(
        lambda: Envelope(
            tipo=TipoMensaje.EVENTO_VISION,
            origen=OrigenEvento.VISION,
            id_dispositivo="",
            id_sesion="ses-001",
            id_mensaje="vis-00001",
            numero_secuencia=1,
            timestamp_origen=100.0,
            payload_json="{}",
        ),
        "id_dispositivo vacío debería fallar",
    )


@_test("Envelope rechaza numero_secuencia negativo")
def _():
    _debe_fallar(
        lambda: Envelope(
            tipo=TipoMensaje.EVENTO_VISION,
            origen=OrigenEvento.VISION,
            id_dispositivo="cam-01",
            id_sesion="ses-001",
            id_mensaje="vis-00001",
            numero_secuencia=-1,
            timestamp_origen=100.0,
            payload_json="{}",
        ),
        "numero_secuencia negativo debería fallar",
    )


@_test("Envelope calcula latencia correctamente")
def _():
    env = Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo="cam-01",
        id_sesion="ses-001",
        id_mensaje="vis-00001",
        numero_secuencia=1,
        timestamp_origen=100.0,
        timestamp_recepcion=100.150,   # 150 ms después
        payload_json="{}",
    )
    assert abs(env.latencia_ms - 150.0) < 0.001


@_test("Envelope latencia es None si no recibido aún")
def _():
    env = Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo="cam-01",
        id_sesion="ses-001",
        id_mensaje="vis-00001",
        numero_secuencia=1,
        timestamp_origen=100.0,
        payload_json="{}",
    )
    assert env.latencia_ms is None


@_test("Envelope serializa y deserializa")
def _():
    ev = EventoVision(timestamp=100.0, rostro_detectado=True, ear_izquierdo=0.2)
    env1 = Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo="cam-01",
        id_sesion="ses-001",
        id_mensaje="vis-00042",
        numero_secuencia=42,
        timestamp_origen=100.0,
        timestamp_recepcion=100.05,
        payload_json=ev.to_json(),
    )
    env2 = Envelope.from_json(env1.to_json())
    assert env1.id_mensaje == env2.id_mensaje
    assert env1.numero_secuencia == env2.numero_secuencia
    assert env1.payload_json == env2.payload_json
    assert env1.tipo == env2.tipo


@_test("Envelope.desempacar() reconstruye EventoVision")
def _():
    ev = EventoVision(
        timestamp=100.0,
        rostro_detectado=True,
        ear_izquierdo=0.21,
        ear_derecho=0.23,
    )
    env = Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo="cam-01",
        id_sesion="ses-001",
        id_mensaje="vis-00001",
        numero_secuencia=1,
        timestamp_origen=100.0,
        payload_json=ev.to_json(),
    )
    reconstruido = env.desempacar()
    assert isinstance(reconstruido, EventoVision)
    assert reconstruido == ev


@_test("Envelope.desempacar() reconstruye EventoFalloSensor")
def _():
    ev = EventoFalloSensor(
        timestamp=200.0,
        sensor_afectado=OrigenEvento.WEARABLE,
        motivo="heartbeat_timeout",
        severidad=2,
    )
    env = Envelope(
        tipo=TipoMensaje.FALLO_SENSOR,
        origen=OrigenEvento.INTERNO,
        id_dispositivo="core",
        id_sesion="ses-001",
        id_mensaje="int-00001",
        numero_secuencia=1,
        timestamp_origen=200.0,
        payload_json=ev.to_json(),
    )
    reconstruido = env.desempacar()
    assert isinstance(reconstruido, EventoFalloSensor)
    assert reconstruido.motivo == "heartbeat_timeout"


@_test("Envelope en modo interno devuelve el evento directo en desempacar()")
def _():
    ev = EventoVision(timestamp=100.0, rostro_detectado=True)
    env = Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo="cam-01",
        id_sesion="ses-001",
        id_mensaje="vis-00001",
        numero_secuencia=1,
        timestamp_origen=100.0,
        evento=ev,
    )
    reconstruido = env.desempacar()
    assert reconstruido is ev   # mismo objeto, no copia


@_test("Envelope serializa el evento interno al exportar")
def _():
    """Si el Envelope tiene evento (modo interno) pero se exporta a JSON,
    debe serializar el evento al payload_json automáticamente."""
    ev = EventoVision(timestamp=100.0, rostro_detectado=True, mar=0.5)
    env = Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo="cam-01",
        id_sesion="ses-001",
        id_mensaje="vis-00001",
        numero_secuencia=1,
        timestamp_origen=100.0,
        evento=ev,   # solo modo interno
    )
    payload = env.to_json()
    parsed = json.loads(payload)
    assert parsed["payload_json"]   # debe haber sido autocompletado
    # Y se puede reconstruir
    env2 = Envelope.from_json(payload)
    ev2 = env2.desempacar()
    assert ev == ev2


# =============================================================================
#                       TESTS DE UTILIDADES (generadores de IDs)
# =============================================================================

print("\n--- Tests de utilidades ---")


@_test("generar_id_sesion devuelve formato esperado")
def _():
    id_ses = generar_id_sesion()
    # Formato: ses-YYYYMMDD-HHMMSS
    assert id_ses.startswith("ses-")
    partes = id_ses.split("-")
    assert len(partes) == 3, f"Formato inesperado: {id_ses}"
    assert len(partes[1]) == 8   # fecha
    assert len(partes[2]) == 6   # hora


@_test("generar_id_sesion respeta prefijo personalizado")
def _():
    id_ses = generar_id_sesion(prefijo="conduccion")
    assert id_ses.startswith("conduccion-")


@_test("generar_id_sesion rechaza prefijo vacío")
def _():
    _debe_fallar(
        lambda: generar_id_sesion(prefijo=""),
        "prefijo vacío debería fallar",
    )


@_test("generar_id_mensaje produce formato con padding")
def _():
    assert generar_id_mensaje("vis", 42) == "vis-00042"
    assert generar_id_mensaje("wea", 1) == "wea-00001"
    assert generar_id_mensaje("int", 0) == "int-00000"


@_test("generar_id_mensaje permite secuencias grandes")
def _():
    resultado = generar_id_mensaje("vis", 123456)
    assert resultado == "vis-123456"


@_test("generar_id_mensaje rechaza secuencia negativa")
def _():
    _debe_fallar(
        lambda: generar_id_mensaje("vis", -1),
        "secuencia negativa debería fallar",
    )


@_test("generar_id_mensaje rechaza prefijo vacío")
def _():
    _debe_fallar(
        lambda: generar_id_mensaje("", 1),
        "prefijo vacío debería fallar",
    )


# =============================================================================
#                              RESUMEN FINAL
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
    print("  NeuroDrive - Tests de contratos compartidos")
    print("=" * 60)
    sys.exit(_resumen())
