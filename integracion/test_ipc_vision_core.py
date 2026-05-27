"""
test_ipc_vision_core.py - Smoke test de IPC Vision <-> Core (Etapa II)
======================================================================

Verifica que un EventoVision viaja de PUNTA A PUNTA entre el subsistema
de Vision y el de Core a traves de la cola POSIX MQ real.

Que valida:
  - El PublicadorMQ de Vision construye, serializa y envia Envelopes.
  - La cola POSIX MQ los transporta.
  - El AdaptadorMQ del Core (lado lectura) los recibe.
  - Envelope.from_json() los reconstruye sin error.
  - desempacar() devuelve un EventoVision identico al original.
  - El orden FIFO se mantiene (numeros de secuencia correlativos).

Que NO valida (es de etapas posteriores):
  - El Gestor de Eventos completo (hilos lectores, deduplicacion).
  - El Pre-FSM ni la FSM.
  - La camara real.

Arquitectura del test:
  Un solo proceso, dos hilos.
    - Hilo CONSUMIDOR: usa AdaptadorMQ del Core en modo lectura. Es el
      "duenio" de la cola: la crea al arrancar.
    - Hilo PRODUCTOR: usa PublicadorMQ de Vision. Arranca DESPUES del
      consumidor (orden correcto del sistema real: Core primero).
  La cola POSIX funciona igual entre hilos que entre procesos, asi que
  esto valida el canal de IPC correctamente.

Ejecutar:
    cd ~/NeuroDrive
    python -m integracion.test_ipc_vision_core
"""

from __future__ import annotations

import sys
import threading
import time
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

# --- Imports del Core ---
from NeuroDrive_Core.config_loader import cargar_config, limpiar_cache
from NeuroDrive_Core.adaptador_mq import AdaptadorMQ, eliminar_cola
from common.contratos import Envelope, EventoVision, TipoMensaje

# --- Imports de Vision ---
from NeuroDrive_Vision.publicador_mq import PublicadorMQ
from NeuroDrive_Vision.detector_rostro import DatosRostro
from NeuroDrive_Vision.analizador_ojos import DatosOjos
from NeuroDrive_Vision.analizador_boca import DatosBoca
from NeuroDrive_Vision.analizador_cabeza import DatosCabeza
from NeuroDrive_Vision.detector_frote_ojos import DatosFroteOjos


# =============================================================================
# Framework minimo
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
# Helpers: datos sinteticos de los analizadores de vision
# =============================================================================

def _datos_rostro(ts: float, n: int) -> DatosRostro:
    """Crea un DatosRostro con un timestamp y datos predecibles."""
    return DatosRostro(
        rostro_presente=True,
        puntos_pixeles=np.zeros((468, 2), dtype=np.int32),
        puntos_normalizados=np.zeros((468, 3), dtype=np.float32),
        resolucion=(640, 480),
        timestamp=ts,
    )


def _datos_ojos(ear_izq: float, ear_der: float) -> DatosOjos:
    return DatosOjos(
        valido=True, ear_izq=ear_izq, ear_der=ear_der,
        ear_promedio=(ear_izq + ear_der) / 2.0,
    )


def _datos_boca(mar: float) -> DatosBoca:
    return DatosBoca(valido=True, mar=mar)


def _datos_cabeza(pitch: float, yaw: float, roll: float) -> DatosCabeza:
    return DatosCabeza(valido=True, pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)


def _datos_frote(frote: bool) -> DatosFroteOjos:
    return DatosFroteOjos(valido=True, frote_en_curso=frote)


# =============================================================================
# Verificacion previa: posix_ipc y contratos disponibles
# =============================================================================

print("=" * 60)
print("  NeuroDrive - Smoke Test IPC Vision <-> Core (Etapa II)")
print("=" * 60)

try:
    import posix_ipc  # noqa: F401
    POSIX_OK = True
except ImportError:
    POSIX_OK = False

if not POSIX_OK:
    print("\nposix_ipc no esta instalado. Este smoke test requiere POSIX MQ.")
    print("Instalar con: pip install posix_ipc")
    sys.exit(1)


# =============================================================================
# Cola de test (separada de la cola real de produccion)
# =============================================================================

COLA_SMOKE = "/neurodrive_smoke_test"


def _limpiar():
    """Elimina la cola de smoke test si quedo de una corrida anterior."""
    try:
        eliminar_cola(COLA_SMOKE)
    except Exception:
        pass


# =============================================================================
# TEST 1 - Round-trip de un solo EventoVision
# =============================================================================

print("\n--- Test 1: round-trip de un EventoVision ---")


@_test("Un EventoVision viaja Vision -> cola -> Core intacto")
def _():
    _limpiar()
    limpiar_cache()
    config = cargar_config()

    # --- Lado CONSUMIDOR (Core): crea la cola, modo lectura ---
    consumidor = AdaptadorMQ.abrir(
        nombre=COLA_SMOKE,
        modo="lectura",
        capacidad=config.ipc.capacidad_cola,
        tamano_max_mensaje=config.ipc.tamano_max_mensaje_bytes,
    )

    # --- Lado PRODUCTOR (Vision): se conecta a la cola existente ---
    # Decisiones de integracion: no drena, no elimina (el Core es duenio).
    productor = PublicadorMQ(
        config=config,
        nombre_cola=COLA_SMOKE,
        drenar_al_iniciar=False,
        eliminar_al_detener=False,
    )

    try:
        productor.iniciar()
        assert not productor.modo_simulado, "el productor deberia estar en modo real"

        # Vision publica un evento con valores conocidos
        ts = time.time()
        ok = productor.publicar(
            _datos_rostro(ts, 1),
            _datos_ojos(0.27, 0.25),
            _datos_boca(0.12),
            _datos_cabeza(pitch=-8.0, yaw=3.0, roll=1.5),
            _datos_frote(False),
        )
        assert ok, "publicar() deberia devolver True"

        # Core recibe
        payload = consumidor.recibir(timeout_seg=2.0)
        assert payload is not None, "el Core no recibio el mensaje"

        # Core reconstruye el Envelope y desempaca el EventoVision
        envelope = Envelope.from_json(payload)
        assert envelope.tipo == TipoMensaje.EVENTO_VISION
        evento = envelope.desempacar()
        assert isinstance(evento, EventoVision)

        # Verificar que los datos llegaron intactos
        assert evento.rostro_detectado is True
        assert abs(evento.ear_izquierdo - 0.27) < 1e-6, f"ear_izq={evento.ear_izquierdo}"
        assert abs(evento.ear_derecho - 0.25) < 1e-6, f"ear_der={evento.ear_derecho}"
        assert abs(evento.mar - 0.12) < 1e-6, f"mar={evento.mar}"
        assert abs(evento.pitch_grados - (-8.0)) < 1e-6, f"pitch={evento.pitch_grados}"
        assert abs(evento.yaw_grados - 3.0) < 1e-6
        assert evento.frote_ojos_activo is False
        print(f"    EventoVision recibido OK: EAR=({evento.ear_izquierdo:.2f},"
              f"{evento.ear_derecho:.2f}) pitch={evento.pitch_grados:.1f}")
    finally:
        productor.detener()
        consumidor.cerrar()
        _limpiar()


# =============================================================================
# TEST 2 - Multiples eventos en orden FIFO
# =============================================================================

print("\n--- Test 2: rafaga de eventos en orden ---")


@_test("20 eventos viajan en orden FIFO con secuencia correlativa")
def _():
    _limpiar()
    limpiar_cache()
    config = cargar_config()

    consumidor = AdaptadorMQ.abrir(
        nombre=COLA_SMOKE, modo="lectura",
        capacidad=config.ipc.capacidad_cola,
        tamano_max_mensaje=config.ipc.tamano_max_mensaje_bytes,
    )
    productor = PublicadorMQ(
        config=config, nombre_cola=COLA_SMOKE,
        drenar_al_iniciar=False, eliminar_al_detener=False,
    )

    N = 20
    try:
        productor.iniciar()

        # Productor y consumidor alternados para no llenar la cola
        # (capacidad 10, mandamos 20: hay que ir leyendo).
        secuencias_recibidas = []
        ear_recibidos = []

        for i in range(N):
            ts = time.time()
            # Cada evento lleva un EAR distinto para identificarlo
            ear = 0.20 + i * 0.005
            ok = productor.publicar(
                _datos_rostro(ts, i),
                _datos_ojos(ear, ear),
                _datos_boca(0.10),
                _datos_cabeza(0.0, 0.0, 0.0),
                _datos_frote(False),
            )
            assert ok, f"publicar #{i} fallo"

            # Leer inmediatamente para no saturar
            payload = consumidor.recibir(timeout_seg=2.0)
            assert payload is not None, f"no se recibio el evento #{i}"
            envelope = Envelope.from_json(payload)
            evento = envelope.desempacar()
            secuencias_recibidas.append(envelope.numero_secuencia)
            ear_recibidos.append(evento.ear_izquierdo)

        # Las secuencias deben ser 1, 2, 3, ..., N
        assert secuencias_recibidas == list(range(1, N + 1)), (
            f"secuencias fuera de orden: {secuencias_recibidas}"
        )

        # Los EAR deben llegar en el orden enviado
        for i, ear in enumerate(ear_recibidos):
            esperado = 0.20 + i * 0.005
            assert abs(ear - esperado) < 1e-6, (
                f"evento #{i}: EAR esperado {esperado}, recibido {ear}"
            )

        print(f"    {N} eventos recibidos en orden, secuencias 1..{N} correctas")
    finally:
        productor.detener()
        consumidor.cerrar()
        _limpiar()


# =============================================================================
# TEST 3 - Evento con campos None (rostro ausente)
# =============================================================================

print("\n--- Test 3: evento con rostro ausente (campos None) ---")


@_test("EventoVision con campos None viaja y se reconstruye")
def _():
    _limpiar()
    limpiar_cache()
    config = cargar_config()

    consumidor = AdaptadorMQ.abrir(
        nombre=COLA_SMOKE, modo="lectura",
        capacidad=config.ipc.capacidad_cola,
        tamano_max_mensaje=config.ipc.tamano_max_mensaje_bytes,
    )
    productor = PublicadorMQ(
        config=config, nombre_cola=COLA_SMOKE,
        drenar_al_iniciar=False, eliminar_al_detener=False,
    )

    try:
        productor.iniciar()

        # Rostro ausente: el publicador manda campos en None
        rostro_ausente = DatosRostro(
            rostro_presente=False, resolucion=(640, 480), timestamp=time.time(),
        )
        ok = productor.publicar(rostro_ausente, None, None, None, None)
        assert ok

        payload = consumidor.recibir(timeout_seg=2.0)
        assert payload is not None
        evento = Envelope.from_json(payload).desempacar()

        assert evento.rostro_detectado is False
        assert evento.ear_izquierdo is None, "EAR deberia ser None sin rostro"
        assert evento.ear_derecho is None
        assert evento.mar is None
        assert evento.pitch_grados is None
        assert evento.confianza_deteccion == 0.0
        print("    EventoVision con None viaja y se reconstruye OK")
    finally:
        productor.detener()
        consumidor.cerrar()
        _limpiar()


# =============================================================================
# TEST 4 - Orden de arranque: Core (consumidor) primero
# =============================================================================

print("\n--- Test 4: arranque concurrente con dos hilos ---")


@_test("Productor y consumidor en hilos separados se comunican")
def _():
    _limpiar()
    limpiar_cache()
    config = cargar_config()

    N = 15
    recibidos = []
    errores_hilo = []
    consumidor_listo = threading.Event()

    def hilo_consumidor():
        """El Core: crea la cola, espera mensajes."""
        try:
            cola = AdaptadorMQ.abrir(
                nombre=COLA_SMOKE, modo="lectura",
                capacidad=config.ipc.capacidad_cola,
                tamano_max_mensaje=config.ipc.tamano_max_mensaje_bytes,
            )
            consumidor_listo.set()  # avisa que la cola ya existe
            try:
                for _ in range(N):
                    payload = cola.recibir(timeout_seg=3.0)
                    if payload is None:
                        errores_hilo.append("timeout esperando mensaje")
                        break
                    evento = Envelope.from_json(payload).desempacar()
                    recibidos.append(evento)
            finally:
                cola.cerrar()
        except Exception as e:
            errores_hilo.append(f"consumidor: {e}")

    def hilo_productor():
        """Vision: espera que la cola exista, despues publica."""
        try:
            # Esperar a que el consumidor haya creado la cola
            if not consumidor_listo.wait(timeout=5.0):
                errores_hilo.append("el consumidor no creo la cola a tiempo")
                return
            productor = PublicadorMQ(
                config=config, nombre_cola=COLA_SMOKE,
                drenar_al_iniciar=False, eliminar_al_detener=False,
            )
            productor.iniciar()
            try:
                for i in range(N):
                    ok = productor.publicar(
                        _datos_rostro(time.time(), i),
                        _datos_ojos(0.25, 0.25),
                        _datos_boca(0.10),
                        _datos_cabeza(0.0, 0.0, 0.0),
                        _datos_frote(False),
                    )
                    if not ok:
                        # cola llena: esperar un poco y reintentar una vez
                        time.sleep(0.05)
                        productor.publicar(
                            _datos_rostro(time.time(), i),
                            _datos_ojos(0.25, 0.25), _datos_boca(0.10),
                            _datos_cabeza(0.0, 0.0, 0.0), _datos_frote(False),
                        )
                    time.sleep(0.02)  # ritmo realista (~50 Hz)
            finally:
                productor.detener()
        except Exception as e:
            errores_hilo.append(f"productor: {e}")

    # Arrancar: consumidor primero (es el duenio de la cola)
    t_cons = threading.Thread(target=hilo_consumidor, name="consumidor")
    t_prod = threading.Thread(target=hilo_productor, name="productor")
    t_cons.start()
    t_prod.start()
    t_cons.join(timeout=15.0)
    t_prod.join(timeout=15.0)

    _limpiar()

    assert not errores_hilo, f"errores en hilos: {errores_hilo}"
    assert len(recibidos) == N, f"se esperaban {N} eventos, llegaron {len(recibidos)}"
    # Todos deben ser EventoVision validos
    for ev in recibidos:
        assert isinstance(ev, EventoVision)
        assert ev.rostro_detectado is True
    print(f"    {len(recibidos)}/{N} eventos transmitidos entre hilos OK")


# =============================================================================
# Resumen
# =============================================================================

print("\n" + "=" * 60)
exitos = sum(1 for _, ok, _ in _resultados if ok)
total = len(_resultados)
print(f"  Resumen: {exitos}/{total} tests pasaron")

if exitos < total:
    print("\n  FALLAS:")
    for nombre, ok, msg in _resultados:
        if not ok:
            print(f"    [FAIL] {nombre}: {msg}")
    print("=" * 60)
    sys.exit(1)

print("  SMOKE TEST OK - el canal IPC Vision <-> Core funciona")
print("=" * 60)
sys.exit(0)
