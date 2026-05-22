"""
test_gestor_eventos.py - Tests funcionales del Gestor de Eventos.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Core.test_gestor_eventos

IMPORTANTE:
- Crean colas POSIX MQ reales. Cada test limpia las suyas.
- Usan timeouts cortos (heartbeats acelerados) para no demorar.
- Si un test crashea, podrian quedar colas en /dev/mqueue/.
  Limpiar manualmente con: sudo rm /dev/mqueue/test_gestor_*

Valida:
  1. Ciclo de vida (iniciar/detener)
  2. Recepcion basica de Envelopes (vision y wearable)
  3. Validacion de coherencia (rechaza mensajes con origen/tipo malos)
  4. Validacion de timestamps del futuro
  5. Deduplicacion por numero_secuencia
  6. Reset de secuencias al detectar reinicio del productor
  7. Politica de cola interna llena (descarta)
  8. Heartbeat: deteccion de fallo de sensor
  9. Heartbeat: deteccion de recuperacion
 10. Estadisticas y contadores
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from dataclasses import replace

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from common.contratos import (
    Envelope,
    EventoVision,
    EventoWearable,
    EventoFalloSensor,
    EventoRecuperacionSensor,
    OrigenEvento,
    TipoMensaje,
    generar_id_mensaje,
    timestamp_actual,
)
from NeuroDrive_Core.adaptador_mq import AdaptadorMQ, eliminar_cola
from NeuroDrive_Core.gestor_eventos import GestorEventos
from NeuroDrive_Core.config_loader import (
    Config,
    ConfigIPCSeccion,
    ConfigWearableSeccion,
    ConfigIdentificadoresSeccion,
)


# =============================================================================
# Framework de testing
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
# Helpers
# =============================================================================

# Cada test usa un sufijo unico para no chocar
_contador_test = 0


def _config_para_test(
    heartbeat_seg: float = 1.0,
    intervalo_bpm_seg: float = 0.3,
) -> tuple[Config, str, str]:
    """
    Construye un Config minimo para testing con colas unicas y heartbeats rapidos.

    Devuelve (config, nombre_cola_vision, nombre_cola_wearable).
    """
    global _contador_test
    _contador_test += 1
    sufijo = f"{_contador_test}_{int(time.time() * 1000) % 100000}"

    cola_vision = f"/test_gestor_v_{sufijo}"
    cola_wearable = f"/test_gestor_w_{sufijo}"

    # Limpiar restos previos
    eliminar_cola(cola_vision)
    eliminar_cola(cola_wearable)

    config = Config()
    config.ipc = ConfigIPCSeccion(
        modo="posix_mq",
        cola_vision=cola_vision,
        cola_wearable=cola_wearable,
        capacidad_cola=8,
        tamano_max_mensaje_bytes=1024,
    )
    config.wearable = ConfigWearableSeccion(
        timeout_heartbeat_seg=heartbeat_seg,
        intervalo_envio_bpm_seg=intervalo_bpm_seg,
    )
    config.identificadores = ConfigIdentificadoresSeccion()

    return config, cola_vision, cola_wearable


def _crear_envelope_vision(
    config: Config,
    secuencia: int,
    timestamp: Optional[float] = None,
) -> Envelope:
    ts = timestamp if timestamp is not None else timestamp_actual()
    ev = EventoVision(
        timestamp=ts,
        rostro_detectado=True,
        ear_izquierdo=0.22,
        ear_derecho=0.23,
        mar=0.3,
    )
    return Envelope(
        tipo=TipoMensaje.EVENTO_VISION,
        origen=OrigenEvento.VISION,
        id_dispositivo=config.identificadores.id_camara,
        id_sesion="ses-test",
        id_mensaje=generar_id_mensaje("vis", secuencia),
        numero_secuencia=secuencia,
        timestamp_origen=ts,
        payload_json=ev.to_json(),
    )


def _crear_envelope_wearable(
    config: Config,
    secuencia: int,
    timestamp: Optional[float] = None,
    bpm: int = 72,
) -> Envelope:
    ts = timestamp if timestamp is not None else timestamp_actual()
    ev = EventoWearable(timestamp=ts, bpm=bpm)
    return Envelope(
        tipo=TipoMensaje.EVENTO_WEARABLE,
        origen=OrigenEvento.WEARABLE,
        id_dispositivo=config.identificadores.id_wearable,
        id_sesion="ses-test",
        id_mensaje=generar_id_mensaje("wea", secuencia),
        numero_secuencia=secuencia,
        timestamp_origen=ts,
        payload_json=ev.to_json(),
    )


from typing import Optional


def _limpiar(config: Config) -> None:
    eliminar_cola(config.ipc.cola_vision)
    eliminar_cola(config.ipc.cola_wearable)


def _gestor_con_productores(config: Config) -> tuple[GestorEventos, AdaptadorMQ, AdaptadorMQ]:
    """
    Crea gestor y abre los lados productores de las MQ.
    Retorna (gestor, productor_vision, productor_wearable).
    Asegurarse de llamar _detener_todo() al final.
    """
    gestor = GestorEventos(config)
    gestor.iniciar()

    productor_vision = AdaptadorMQ.abrir(
        nombre=config.ipc.cola_vision,
        modo="escritura",
        capacidad=config.ipc.capacidad_cola,
        tamano_max_mensaje=config.ipc.tamano_max_mensaje_bytes,
    )
    productor_wearable = AdaptadorMQ.abrir(
        nombre=config.ipc.cola_wearable,
        modo="escritura",
        capacidad=config.ipc.capacidad_cola,
        tamano_max_mensaje=config.ipc.tamano_max_mensaje_bytes,
    )
    return gestor, productor_vision, productor_wearable


def _detener_todo(
    gestor: GestorEventos,
    productores: list[AdaptadorMQ],
    config: Config,
) -> None:
    """Cierre limpio: detener gestor, cerrar productores, eliminar colas."""
    try:
        gestor.detener(timeout_join=2.0)
    except Exception as e:
        print(f"Error deteniendo gestor: {e}")

    for p in productores:
        try:
            p.cerrar()
        except Exception:
            pass

    _limpiar(config)


# =============================================================================
# TESTS DE CICLO DE VIDA
# =============================================================================

print("\n--- Tests de ciclo de vida ---")


@_test("iniciar() y detener() funcionan limpiamente")
def _():
    config, _, _ = _config_para_test()
    try:
        gestor = GestorEventos(config)
        assert not gestor.activo
        gestor.iniciar()
        assert gestor.activo
        gestor.detener(timeout_join=2.0)
        assert not gestor.activo
    finally:
        _limpiar(config)


@_test("iniciar() doble es idempotente (warning, no error)")
def _():
    config, _, _ = _config_para_test()
    try:
        gestor = GestorEventos(config)
        gestor.iniciar()
        gestor.iniciar()  # no deberia romper
        assert gestor.activo
        gestor.detener(timeout_join=2.0)
    finally:
        _limpiar(config)


@_test("detener() sin iniciar es idempotente")
def _():
    config, _, _ = _config_para_test()
    try:
        gestor = GestorEventos(config)
        gestor.detener()  # no deberia romper
        assert not gestor.activo
    finally:
        _limpiar(config)


@_test("Context manager funciona correctamente")
def _():
    config, _, _ = _config_para_test()
    try:
        with GestorEventos(config) as gestor:
            assert gestor.activo
        assert not gestor.activo
    finally:
        _limpiar(config)


# =============================================================================
# TESTS DE RECEPCION BASICA
# =============================================================================

print("\n--- Tests de recepcion basica ---")


@_test("Recibe Envelope de Vision correctamente")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        env_orig = _crear_envelope_vision(config, secuencia=1)
        prod_v.enviar(env_orig.to_json())

        # Esperar a que el hilo procese
        envelope = gestor.obtener_evento(timeout=2.0)
        assert envelope is not None, "no llego envelope"
        assert envelope.tipo == TipoMensaje.EVENTO_VISION
        assert envelope.origen == OrigenEvento.VISION
        assert envelope.evento is not None, "evento deberia estar poblado"
        assert isinstance(envelope.evento, EventoVision)
        assert envelope.timestamp_recepcion > 0
    finally:
        _detener_todo(gestor, [prod_v, prod_w], config)


@_test("Recibe Envelope de Wearable correctamente")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        env_orig = _crear_envelope_wearable(config, secuencia=1, bpm=68)
        prod_w.enviar(env_orig.to_json())

        envelope = gestor.obtener_evento(timeout=2.0)
        assert envelope is not None
        assert envelope.tipo == TipoMensaje.EVENTO_WEARABLE
        assert envelope.origen == OrigenEvento.WEARABLE
        assert isinstance(envelope.evento, EventoWearable)
        assert envelope.evento.bpm == 68
    finally:
        _detener_todo(gestor, [prod_v, prod_w], config)


@_test("obtener_evento() con timeout devuelve None si nada llego")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        envelope = gestor.obtener_evento(timeout=0.3)
        assert envelope is None
    finally:
        _detener_todo(gestor, [prod_v, prod_w], config)


@_test("Multiples mensajes en orden FIFO")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        for i in range(1, 6):
            prod_v.enviar(_crear_envelope_vision(config, secuencia=i).to_json())

        recibidos = []
        for _ in range(5):
            env = gestor.obtener_evento(timeout=2.0)
            assert env is not None
            recibidos.append(env.numero_secuencia)

        assert recibidos == [1, 2, 3, 4, 5], f"FIFO roto: {recibidos}"
    finally:
        _detener_todo(gestor, [prod_v, prod_w], config)


# =============================================================================
# TESTS DE VALIDACION
# =============================================================================

print("\n--- Tests de validacion ---")


@_test("Rechaza Envelope con origen incongruente con la cola")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        # Mandamos un envelope de WEARABLE por la cola de VISION
        ev = EventoVision(timestamp=timestamp_actual(), rostro_detectado=True)
        env_malo = Envelope(
            tipo=TipoMensaje.EVENTO_WEARABLE,  # Tipo mal
            origen=OrigenEvento.WEARABLE,      # Origen mal
            id_dispositivo="cam-01",
            id_sesion="ses-test",
            id_mensaje="vis-00099",
            numero_secuencia=99,
            timestamp_origen=ev.timestamp,
            payload_json=ev.to_json(),
        )
        prod_v.enviar(env_malo.to_json())

        # Tambien mandar uno valido despues, asi confirmamos que sigue funcionando
        prod_v.enviar(_crear_envelope_vision(config, secuencia=100).to_json())

        # Solo debe llegar el valido
        env = gestor.obtener_evento(timeout=2.0)
        assert env is not None
        assert env.numero_secuencia == 100

        # No deberian llegar mas
        otro = gestor.obtener_evento(timeout=0.5)
        assert otro is None, "deberia haberse descartado el mensaje invalido"

        assert gestor.stats.mensajes_invalidos >= 1
    finally:
        _detener_todo(gestor, [prod_v, prod_w], config)


@_test("Rechaza Envelope con timestamp del futuro")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        # Timestamp 60 segundos en el futuro (tolerancia es 5s)
        ts_futuro = timestamp_actual() + 60
        env_futuro = _crear_envelope_vision(config, secuencia=1, timestamp=ts_futuro)
        prod_v.enviar(env_futuro.to_json())

        # Tambien uno valido
        env_ok = _crear_envelope_vision(config, secuencia=2)
        prod_v.enviar(env_ok.to_json())

        env = gestor.obtener_evento(timeout=2.0)
        assert env is not None
        assert env.numero_secuencia == 2

        assert gestor.stats.mensajes_invalidos >= 1
    finally:
        _detener_todo(gestor, [prod_v, prod_w], config)


@_test("Rechaza JSON malformado")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        # Mandar basura
        prod_v.enviar("{esto no es JSON valido")
        # Y algo valido
        prod_v.enviar(_crear_envelope_vision(config, secuencia=1).to_json())

        env = gestor.obtener_evento(timeout=2.0)
        assert env is not None
        assert env.numero_secuencia == 1

        assert gestor.stats.mensajes_invalidos >= 1
    finally:
        _detener_todo(gestor, [prod_v, prod_w], config)


# =============================================================================
# TESTS DE DEDUPLICACION
# =============================================================================

print("\n--- Tests de deduplicacion ---")


@_test("Descarta duplicados por numero_secuencia")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        # Mandar el mismo numero_secuencia 3 veces
        for _ in range(3):
            prod_v.enviar(_crear_envelope_vision(config, secuencia=42).to_json())

        # Solo deberia llegar 1
        env = gestor.obtener_evento(timeout=2.0)
        assert env is not None
        assert env.numero_secuencia == 42

        otro = gestor.obtener_evento(timeout=0.5)
        assert otro is None

        assert gestor.stats.duplicados_descartados >= 2
    finally:
        _detener_todo(gestor, [prod_v, prod_w], config)


@_test("Reinicio del productor (salto grande) limpia historial de secuencias")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        # Primero mandamos secuencias altas
        for s in [9000, 9001, 9002]:
            prod_v.enviar(_crear_envelope_vision(config, secuencia=s).to_json())
            time.sleep(0.05)

        # Drenamos
        for _ in range(3):
            gestor.obtener_evento(timeout=1.0)

        # Ahora simulamos reinicio: secuencias bajas (salto > 1000)
        for s in [1, 2, 3]:
            prod_v.enviar(_crear_envelope_vision(config, secuencia=s).to_json())
            time.sleep(0.05)

        recibidos = []
        for _ in range(3):
            env = gestor.obtener_evento(timeout=1.0)
            if env is not None:
                recibidos.append(env.numero_secuencia)

        # Las nuevas secuencias deben pasar (no las trate como duplicado de "9000-ish")
        assert recibidos == [1, 2, 3], f"reinicio no funciono: {recibidos}"
    finally:
        _detener_todo(gestor, [prod_v, prod_w], config)


# =============================================================================
# TESTS DE HEARTBEAT
# =============================================================================

print("\n--- Tests de heartbeat ---")


@_test("Detecta fallo de sensor wearable por timeout")
def _():
    # heartbeat 1s, intervalo BPM 0.3s
    config, _, _ = _config_para_test(heartbeat_seg=1.0, intervalo_bpm_seg=0.3)
    gestor, prod_v, prod_w = _gestor_con_productores(config)

    eventos_capturados = []
    parar_drenado = threading.Event()

    # Hilo drenador para que la cola interna no se sature
    def drenar_continuo():
        while not parar_drenado.is_set():
            env = gestor.obtener_evento(timeout=0.1)
            if env is not None:
                eventos_capturados.append(env)

    hilo_drenado = threading.Thread(target=drenar_continuo, daemon=True)
    hilo_drenado.start()

    try:
        # Mandar 1 mensaje wearable inicial para que el monitor lo "vea vivo"
        prod_w.enviar(_crear_envelope_wearable(config, secuencia=1).to_json())
        time.sleep(0.3)

        # Mantener vision viva, ritmo moderado (3/seg)
        n = 100
        deadline_envio = time.time() + 2.5
        while time.time() < deadline_envio:
            try:
                prod_v.enviar(
                    _crear_envelope_vision(config, secuencia=n).to_json()
                )
            except Exception:
                pass
            n += 1
            time.sleep(0.3)

        # Buscar EventoFalloSensor sobre WEARABLE
        encontrado = False
        for env in eventos_capturados:
            if env.tipo == TipoMensaje.FALLO_SENSOR:
                evento = env.evento
                if (
                    isinstance(evento, EventoFalloSensor)
                    and evento.sensor_afectado == OrigenEvento.WEARABLE
                ):
                    encontrado = True
                    break

        assert encontrado, (
            f"no se emitio EventoFalloSensor para WEARABLE. "
            f"Capturados: {[e.tipo.name for e in eventos_capturados]}"
        )
        assert gestor.stats.fallos_sensor_emitidos >= 1
    finally:
        parar_drenado.set()
        hilo_drenado.join(timeout=1.0)
        _detener_todo(gestor, [prod_v, prod_w], config)


@_test("Detecta recuperacion despues de fallo")
def _():
    config, _, _ = _config_para_test(heartbeat_seg=1.0, intervalo_bpm_seg=0.3)
    gestor, prod_v, prod_w = _gestor_con_productores(config)

    # Variables compartidas con el hilo de vision
    eventos_capturados = []
    parar_drenado = threading.Event()
    t_limite_vision = [0.0]   # mutable para que el hilo lo vea

    # Hilo drenador: corre TODO el test drenando la cola interna
    # asi nunca se llena y los EventoFalloSensor/RecuperacionSensor
    # del monitor llegan al consumidor.
    def drenar_continuo():
        while not parar_drenado.is_set():
            env = gestor.obtener_evento(timeout=0.1)
            if env is not None:
                eventos_capturados.append(env)

    # Hilo que mantiene vision vivo
    def mantener_vision():
        n = 100
        # Ratio mas lento para no saturar (3/seg en lugar de 10/seg)
        while not gestor._parar.is_set() and time.time() < t_limite_vision[0]:
            try:
                prod_v.enviar(_crear_envelope_vision(config, secuencia=n).to_json())
            except Exception:
                pass
            n += 1
            time.sleep(0.3)

    hilo_drenado = threading.Thread(target=drenar_continuo, daemon=True)
    hilo_drenado.start()

    try:
        # Mensaje inicial al wearable para marcarlo "vivo"
        prod_w.enviar(_crear_envelope_wearable(config, secuencia=1).to_json())
        time.sleep(0.3)  # dar tiempo a que se procese

        t_limite_vision[0] = time.time() + 6.0
        hilo_vision = threading.Thread(target=mantener_vision, daemon=True)
        hilo_vision.start()

        # Esperar a que el wearable caiga (heartbeat 1s + margen del monitor)
        time.sleep(2.5)

        # Verificamos que el monitor declaro caida
        fallo_detectado = any(
            env.tipo == TipoMensaje.FALLO_SENSOR
            and isinstance(env.evento, EventoFalloSensor)
            and env.evento.sensor_afectado == OrigenEvento.WEARABLE
            for env in eventos_capturados
        )
        assert fallo_detectado, "el fallo previo no se detecto (precondicion)"

        # Reactivar wearable
        prod_w.enviar(_crear_envelope_wearable(config, secuencia=2).to_json())

        # Esperar a que el monitor detecte la recuperacion (max 2.5s)
        deadline = time.time() + 2.5
        encontrado_recup = False
        while time.time() < deadline and not encontrado_recup:
            for env in eventos_capturados:
                if env.tipo == TipoMensaje.RECUPERACION_SENSOR:
                    evento = env.evento
                    if (
                        isinstance(evento, EventoRecuperacionSensor)
                        and evento.sensor_recuperado == OrigenEvento.WEARABLE
                    ):
                        encontrado_recup = True
                        assert evento.tiempo_caido_seg > 0
                        break
            if not encontrado_recup:
                time.sleep(0.1)

        assert encontrado_recup, (
            f"no se emitio EventoRecuperacionSensor. "
            f"Eventos capturados: {[e.tipo.name for e in eventos_capturados]}"
        )

        t_limite_vision[0] = 0  # detener hilo de vision
        hilo_vision.join(timeout=1.0)
    finally:
        parar_drenado.set()
        hilo_drenado.join(timeout=1.0)
        _detener_todo(gestor, [prod_v, prod_w], config)


# =============================================================================
# TESTS DE INTEGRIDAD AL CIERRE
# =============================================================================

print("\n--- Tests de integridad al cierre ---")


@_test("Cierre con mensajes en vuelo no rompe")
def _():
    config, _, _ = _config_para_test()
    gestor, prod_v, prod_w = _gestor_con_productores(config)
    try:
        # Mandamos varios mensajes y cerramos sin drenar
        for i in range(1, 6):
            prod_v.enviar(_crear_envelope_vision(config, secuencia=i).to_json())

        time.sleep(0.3)
        # detener fuerza cierre limpio
        gestor.detener(timeout_join=2.0)
        assert not gestor.activo
    finally:
        for p in [prod_v, prod_w]:
            try:
                p.cerrar()
            except Exception:
                pass
        _limpiar(config)


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
    print("  NeuroDrive - Tests del Gestor de Eventos")
    print("=" * 60)
    sys.exit(_resumen())
