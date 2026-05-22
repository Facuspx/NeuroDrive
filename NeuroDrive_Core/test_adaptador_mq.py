"""
test_adaptador_mq.py - Tests funcionales del adaptador POSIX MQ.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Core.test_adaptador_mq

IMPORTANTE: estos tests CREAN colas reales en el kernel. Cada test
limpia su propia cola al terminar. Si un test crashea, podrian
quedar colas huerfanas en /dev/mqueue/. Para limpiar manualmente:
    sudo ls /dev/mqueue/
    sudo rm /dev/mqueue/test_*

Valida:
  1. Apertura, escritura y lectura basica
  2. Validacion de parametros (nombre, modo, capacidad, tamanos)
  3. Disciplina de modo (escritura no puede leer, lectura no puede escribir)
  4. Politica de cola llena (descarta y devuelve False)
  5. Mensaje sobre tamano maximo (rechaza con error)
  6. Timeouts en recibir
  7. Estadisticas y contadores
  8. Limpieza correcta (close vs unlink)
  9. Context manager (with statement)
 10. Round-trip con Envelope real del proyecto
"""

from __future__ import annotations

import sys
import time
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from NeuroDrive_Core.adaptador_mq import (
    AdaptadorMQ,
    ErrorAdaptadorMQ,
    ErrorTamanoMensaje,
    eliminar_cola,
)
from common.contratos import (
    Envelope,
    EventoVision,
    TipoMensaje,
    OrigenEvento,
    generar_id_sesion,
    generar_id_mensaje,
)


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


def _debe_fallar(func, mensaje_test: str, excepciones=(ErrorAdaptadorMQ, ValueError)) -> None:
    try:
        func()
    except excepciones:
        return
    raise AssertionError(f"{mensaje_test} (no lanzo excepcion)")


# Cada test usa un nombre de cola unico para no interferir con otros
_COLA_BASE = "/test_neurodrive_"
_contador_cola = 0


def _nuevo_nombre_cola() -> str:
    global _contador_cola
    _contador_cola += 1
    return f"{_COLA_BASE}{_contador_cola}_{int(time.time() * 1000) % 100000}"


# =============================================================================
# TESTS DE APERTURA Y VALIDACIONES BASICAS
# =============================================================================

print("\n--- Tests de apertura y validaciones ---")


@_test("Apertura basica funciona y se cierra limpio")
def _():
    nombre = _nuevo_nombre_cola()
    cola = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256)
    assert cola is not None
    cola.cerrar()
    cola.eliminar()


@_test("Rechaza nombre sin /")
def _():
    _debe_fallar(
        lambda: AdaptadorMQ.abrir("sin_barra", modo="escritura", capacidad=4, tamano_max_mensaje=256),
        "nombre sin / deberia fallar",
    )


@_test("Rechaza modo invalido")
def _():
    _debe_fallar(
        lambda: AdaptadorMQ.abrir("/test_modo_mal", modo="lectoescritura", capacidad=4, tamano_max_mensaje=256),
        "modo invalido deberia fallar",
    )


@_test("Rechaza capacidad <= 0")
def _():
    _debe_fallar(
        lambda: AdaptadorMQ.abrir("/test_cap_mal", modo="escritura", capacidad=0, tamano_max_mensaje=256),
        "capacidad 0 deberia fallar",
    )


@_test("Rechaza tamano_max_mensaje muy chico")
def _():
    _debe_fallar(
        lambda: AdaptadorMQ.abrir("/test_tam_mal", modo="escritura", capacidad=4, tamano_max_mensaje=10),
        "tamano_max < 128 deberia fallar",
    )


@_test("Rechaza nombre demasiado largo")
def _():
    nombre = "/" + "x" * 300
    _debe_fallar(
        lambda: AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256),
        "nombre demasiado largo deberia fallar",
    )


# =============================================================================
# TESTS DE ENVIO Y RECEPCION
# =============================================================================

print("\n--- Tests de envio y recepcion ---")


@_test("Envio y recepcion basicos (round-trip)")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=512)
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=4, tamano_max_mensaje=512)

        ok = cola_w.enviar("mensaje de prueba")
        assert ok, "envio basico deberia haber funcionado"

        msg = cola_r.recibir(timeout_seg=1.0)
        assert msg == "mensaje de prueba", f"esperaba 'mensaje de prueba', obtuve: {msg!r}"

        cola_w.cerrar()
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


@_test("Mensajes UTF-8 con acentos viajan bien")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=512)
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=4, tamano_max_mensaje=512)

        original = "Atención: somnolencia detectada en conductor"
        cola_w.enviar(original)
        msg = cola_r.recibir(timeout_seg=1.0)
        assert msg == original, f"UTF-8 corrupto: {msg!r} != {original!r}"

        cola_w.cerrar()
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


@_test("Multiples mensajes en orden FIFO")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=8, tamano_max_mensaje=256)
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=8, tamano_max_mensaje=256)

        for i in range(5):
            cola_w.enviar(f"mensaje-{i}")

        recibidos = []
        for _ in range(5):
            recibidos.append(cola_r.recibir(timeout_seg=1.0))

        assert recibidos == [f"mensaje-{i}" for i in range(5)], (
            f"orden FIFO roto: {recibidos}"
        )

        cola_w.cerrar()
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


@_test("Timeout en recibir devuelve None")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=4, tamano_max_mensaje=256)
        inicio = time.time()
        msg = cola_r.recibir(timeout_seg=0.3)
        duracion = time.time() - inicio
        assert msg is None, f"timeout deberia devolver None, devolvio {msg!r}"
        assert 0.2 < duracion < 0.6, f"timeout duro {duracion}s, esperaba ~0.3s"
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


@_test("Timeout 0 es no-bloqueante")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=4, tamano_max_mensaje=256)
        inicio = time.time()
        msg = cola_r.recibir(timeout_seg=0.0)
        duracion = time.time() - inicio
        assert msg is None
        assert duracion < 0.1, f"timeout=0 duro {duracion}s, deberia ser instantaneo"
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


# =============================================================================
# TESTS DE DISCIPLINA DE MODO
# =============================================================================

print("\n--- Tests de disciplina de modo ---")


@_test("Modo escritura rechaza recibir()")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256)
        _debe_fallar(
            lambda: cola_w.recibir(timeout_seg=0.1),
            "recibir() en modo escritura deberia fallar",
        )
        cola_w.cerrar()
    finally:
        eliminar_cola(nombre)


@_test("Modo lectura rechaza enviar()")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=4, tamano_max_mensaje=256)
        _debe_fallar(
            lambda: cola_r.enviar("hola"),
            "enviar() en modo lectura deberia fallar",
        )
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


# =============================================================================
# TESTS DE POLITICA DE COLA LLENA (descartar)
# =============================================================================

print("\n--- Tests de cola llena ---")


@_test("Cola llena devuelve False y NO lanza excepcion")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=2, tamano_max_mensaje=256)

        # Llenamos la cola hasta el limite
        assert cola_w.enviar("msg-1") is True
        assert cola_w.enviar("msg-2") is True

        # Tercera deberia devolver False (cola llena)
        resultado = cola_w.enviar("msg-3-overflow")
        assert resultado is False, "envio en cola llena deberia devolver False"

        # Verificar contador de descartes
        assert cola_w.mensajes_descartados_cola_llena == 1

        cola_w.cerrar()
    finally:
        eliminar_cola(nombre)


@_test("Despues de leer, se puede volver a enviar (cola libera espacio)")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=2, tamano_max_mensaje=256)
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=2, tamano_max_mensaje=256)

        cola_w.enviar("a")
        cola_w.enviar("b")
        assert cola_w.enviar("c") is False  # llena

        # Leer libera espacio
        cola_r.recibir(timeout_seg=0.5)
        assert cola_w.enviar("c") is True

        cola_w.cerrar()
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


# =============================================================================
# TESTS DE TAMANO DE MENSAJE
# =============================================================================

print("\n--- Tests de tamano de mensaje ---")


@_test("Mensaje sobre tamano maximo lanza ErrorTamanoMensaje")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256)
        mensaje_gigante = "x" * 1000

        _debe_fallar(
            lambda: cola_w.enviar(mensaje_gigante),
            "mensaje sobre tamano deberia lanzar ErrorTamanoMensaje",
            excepciones=(ErrorTamanoMensaje,),
        )

        cola_w.cerrar()
    finally:
        eliminar_cola(nombre)


@_test("Mensaje exactamente del tamano maximo se envia OK")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=128)
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=4, tamano_max_mensaje=128)

        # Mensaje de exactamente 128 bytes
        msg = "x" * 128
        assert cola_w.enviar(msg) is True
        recibido = cola_r.recibir(timeout_seg=0.5)
        assert recibido == msg

        cola_w.cerrar()
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


# =============================================================================
# TESTS DE ESTADISTICAS
# =============================================================================

print("\n--- Tests de estadisticas ---")


@_test("Contadores de mensajes enviados/recibidos funcionan")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256)
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=4, tamano_max_mensaje=256)

        for i in range(3):
            cola_w.enviar(f"m-{i}")
        for _ in range(3):
            cola_r.recibir(timeout_seg=0.5)

        assert cola_w.mensajes_enviados == 3
        assert cola_r.mensajes_recibidos == 3
        assert cola_w.errores == 0
        assert cola_r.errores == 0

        cola_w.cerrar()
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


@_test("mensajes_en_cola() devuelve la cantidad correcta")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola_w = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256)
        cola_r = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=4, tamano_max_mensaje=256)

        assert cola_w.mensajes_en_cola() == 0
        cola_w.enviar("a")
        cola_w.enviar("b")
        assert cola_w.mensajes_en_cola() == 2
        cola_r.recibir(timeout_seg=0.5)
        assert cola_w.mensajes_en_cola() == 1

        cola_w.cerrar()
        cola_r.cerrar()
    finally:
        eliminar_cola(nombre)


# =============================================================================
# TESTS DE LIMPIEZA Y CONTEXT MANAGER
# =============================================================================

print("\n--- Tests de limpieza ---")


@_test("Doble cerrar() es seguro (idempotente)")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256)
        cola.cerrar()
        cola.cerrar()  # no deberia fallar
    finally:
        eliminar_cola(nombre)


@_test("Operacion sobre cola cerrada lanza error")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        cola = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256)
        cola.cerrar()
        _debe_fallar(
            lambda: cola.enviar("post-cierre"),
            "enviar() en cola cerrada deberia fallar",
        )
    finally:
        eliminar_cola(nombre)


@_test("Context manager cierra automaticamente")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        with AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256) as cola:
            cola.enviar("dentro del with")
            assert cola._abierta
        # Despues del with deberia estar cerrada
        assert cola._cerrada
    finally:
        eliminar_cola(nombre)


@_test("eliminar() borra la cola del kernel")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    cola = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=4, tamano_max_mensaje=256)
    cola.cerrar()
    cola.eliminar()
    # Si la borramos, eliminar_cola() devuelve False (no existe)
    asunto_existia = eliminar_cola(nombre)
    assert asunto_existia is False, "eliminar() deberia haber borrado la cola"


# =============================================================================
# TEST DE INTEGRACION: ROUND-TRIP CON ENVELOPE REAL
# =============================================================================

print("\n--- Test de integracion con Envelope ---")


@_test("Envelope completo viaja Vision -> Gestor por MQ")
def _():
    nombre = _nuevo_nombre_cola()
    eliminar_cola(nombre)
    try:
        # Setup: dos extremos de la cola
        cola_vision = AdaptadorMQ.abrir(nombre, modo="escritura", capacidad=8, tamano_max_mensaje=1024)
        cola_gestor = AdaptadorMQ.abrir(nombre, modo="lectura", capacidad=8, tamano_max_mensaje=1024)

        # Vision construye un evento y lo envuelve
        id_ses = generar_id_sesion()
        evento_orig = EventoVision(
            timestamp=1234.5,
            rostro_detectado=True,
            ear_izquierdo=0.21,
            ear_derecho=0.23,
            mar=0.4,
            pitch_grados=-2.5,
            yaw_grados=1.0,
            roll_grados=0.5,
            confianza_deteccion=0.93,
        )
        envelope_orig = Envelope(
            tipo=TipoMensaje.EVENTO_VISION,
            origen=OrigenEvento.VISION,
            id_dispositivo="cam-01",
            id_sesion=id_ses,
            id_mensaje=generar_id_mensaje("vis", 1),
            numero_secuencia=1,
            timestamp_origen=evento_orig.timestamp,
            payload_json=evento_orig.to_json(),
        )

        # Vision envia
        ok = cola_vision.enviar(envelope_orig.to_json())
        assert ok

        # Gestor recibe
        payload = cola_gestor.recibir(timeout_seg=1.0)
        assert payload is not None

        # Gestor reconstruye
        envelope_recv = Envelope.from_json(payload)
        evento_recv = envelope_recv.desempacar()

        # Verificar que llego todo intacto
        assert isinstance(evento_recv, EventoVision)
        assert evento_recv == evento_orig
        assert envelope_recv.id_mensaje == envelope_orig.id_mensaje
        assert envelope_recv.id_sesion == id_ses

        cola_vision.cerrar()
        cola_gestor.cerrar()
    finally:
        eliminar_cola(nombre)


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
    print("  NeuroDrive - Tests del Adaptador POSIX MQ")
    print("=" * 60)
    sys.exit(_resumen())
