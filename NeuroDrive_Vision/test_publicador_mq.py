"""
test_publicador_mq.py - Tests funcionales de PublicadorMQ.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_publicador_mq

Tests sin hardware (16):
   1. Construccion con parametros default
   2. Construccion con capacidad invalida falla
   3. Modo simulado: iniciar/detener funciona
   4. Publicar sin iniciar lanza error
   5. Publicar con datos_rostro None lanza error
   6. Modo simulado: publicar cuenta como enviado
   7. _calcular_confianza: sin rostro -> 0.0
   8. _calcular_confianza: 3 analizadores validos -> 1.0
   9. _calcular_confianza: 2 de 3 validos -> 0.66
  10. _construir_evento_dict: campos correctos con rostro completo
  11. _construir_evento_dict: campos en None sin rostro
  12. _construir_evento_dict: EAR=0 (ojo indeterminado) -> None
  13. numero_secuencia incrementa por mensaje
  14. id_sesion se genera al iniciar
  15. obtener_estadisticas devuelve dict coherente
  16. Context manager funciona

Tests con hardware (3) - requieren posix_ipc + contratos del Core:
  17. Modo real: abrir y cerrar cola POSIX MQ
  18. Modo real: publicar un Envelope real y verificar tamanio < 1024
  19. Modo real: cola llena -> descarta sin congelar
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

import numpy as np

from NeuroDrive_Vision.detector_rostro import DatosRostro
from NeuroDrive_Vision.analizador_ojos import DatosOjos
from NeuroDrive_Vision.analizador_boca import DatosBoca
from NeuroDrive_Vision.analizador_cabeza import DatosCabeza
from NeuroDrive_Vision.detector_frote_ojos import DatosFroteOjos
from NeuroDrive_Vision.publicador_mq import (
    PublicadorMQ,
    ErrorPublicadorMQ,
    POSIX_IPC_DISPONIBLE,
    CONTRATOS_DISPONIBLES,
)

# Los tests "con hardware" del publicador en realidad necesitan posix_ipc
# y los contratos del Core, no la camara. Detectamos eso.
MODO_REAL_DISPONIBLE = POSIX_IPC_DISPONIBLE and CONTRATOS_DISPONIBLES


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


# =============================================================================
# Helpers: construir datos sinteticos de los analizadores
# =============================================================================

def _datos_rostro(presente=True, ts=None):
    if ts is None:
        ts = time.time()
    if presente:
        return DatosRostro(
            rostro_presente=True,
            puntos_pixeles=np.zeros((468, 2), dtype=np.int32),
            puntos_normalizados=np.zeros((468, 3), dtype=np.float32),
            resolucion=(640, 480),
            timestamp=ts,
        )
    return DatosRostro(rostro_presente=False, resolucion=(640, 480), timestamp=ts)


def _datos_ojos(valido=True, ear_izq=0.27, ear_der=0.26):
    return DatosOjos(
        valido=valido,
        ear_izq=ear_izq,
        ear_der=ear_der,
        ear_promedio=(ear_izq + ear_der) / 2.0,
    )


def _datos_boca(valido=True, mar=0.10):
    return DatosBoca(valido=valido, mar=mar)


def _datos_cabeza(valido=True, pitch=-7.0, yaw=2.0, roll=1.0):
    return DatosCabeza(
        valido=valido,
        pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll,
    )


def _datos_frote(valido=True, frote=False):
    return DatosFroteOjos(valido=valido, frote_en_curso=frote)


# =============================================================================
# TESTS SIN HARDWARE (modo simulado)
# =============================================================================

print("\n--- Tests de PublicadorMQ (modo simulado) ---")


@_test("Construccion con parametros default")
def _():
    p = PublicadorMQ()
    assert p.nombre_cola == "/neurodrive_vision"
    assert p.id_dispositivo == "cam-01"
    assert p.capacidad_cola == 10
    assert not p.activo


@_test("Construccion con capacidad invalida falla")
def _():
    try:
        PublicadorMQ(capacidad_cola=0)
        raise AssertionError("deberia fallar con capacidad 0")
    except ValueError:
        pass


@_test("Modo simulado: iniciar/detener funciona")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    assert p.modo_simulado is True
    assert not p.activo
    p.iniciar()
    assert p.activo
    assert p.id_sesion is not None
    p.detener()
    assert not p.activo


@_test("Publicar sin iniciar lanza error")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    try:
        p.publicar(_datos_rostro())
        raise AssertionError("deberia lanzar ErrorPublicadorMQ")
    except ErrorPublicadorMQ:
        pass


@_test("Publicar con datos_rostro None lanza error")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    p.iniciar()
    try:
        p.publicar(None)
        raise AssertionError("deberia lanzar ErrorPublicadorMQ")
    except ErrorPublicadorMQ:
        pass
    finally:
        p.detener()


@_test("Modo simulado: publicar cuenta como enviado")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    p.iniciar()
    try:
        for _ in range(5):
            ok = p.publicar(
                _datos_rostro(), _datos_ojos(), _datos_boca(),
                _datos_cabeza(), _datos_frote(),
            )
            assert ok is True
        assert p.mensajes_enviados == 5
        assert p.mensajes_descartados == 0
    finally:
        p.detener()


@_test("_calcular_confianza: sin rostro -> 0.0")
def _():
    conf = PublicadorMQ._calcular_confianza(
        _datos_rostro(presente=False), None, None, None,
    )
    assert conf == 0.0


@_test("_calcular_confianza: 3 analizadores validos -> 1.0")
def _():
    conf = PublicadorMQ._calcular_confianza(
        _datos_rostro(), _datos_ojos(), _datos_boca(), _datos_cabeza(),
    )
    assert conf == 1.0, f"conf={conf}"


@_test("_calcular_confianza: 2 de 3 validos -> 0.66")
def _():
    conf = PublicadorMQ._calcular_confianza(
        _datos_rostro(),
        _datos_ojos(valido=True),
        _datos_boca(valido=False),
        _datos_cabeza(valido=True),
    )
    assert abs(conf - 2.0 / 3.0) < 1e-6, f"conf={conf}"


@_test("_construir_evento_dict: campos correctos con rostro completo")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    d = p._construir_evento_dict(
        _datos_rostro(ts=12345.0),
        _datos_ojos(ear_izq=0.27, ear_der=0.26),
        _datos_boca(mar=0.11),
        _datos_cabeza(pitch=-7.0, yaw=2.0, roll=1.0),
        _datos_frote(frote=True),
    )
    assert d["rostro_detectado"] is True
    assert d["timestamp"] == 12345.0
    assert abs(d["ear_izquierdo"] - 0.27) < 1e-6
    assert abs(d["ear_derecho"] - 0.26) < 1e-6
    assert abs(d["mar"] - 0.11) < 1e-6
    assert abs(d["pitch_grados"] - (-7.0)) < 1e-6
    assert d["frote_ojos_activo"] is True
    assert d["confianza_deteccion"] == 1.0


@_test("_construir_evento_dict: campos en None sin rostro")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    d = p._construir_evento_dict(
        _datos_rostro(presente=False),
        None, None, None, None,
    )
    assert d["rostro_detectado"] is False
    assert d["ear_izquierdo"] is None
    assert d["ear_derecho"] is None
    assert d["mar"] is None
    assert d["pitch_grados"] is None
    assert d["confianza_deteccion"] == 0.0


@_test("_construir_evento_dict: EAR=0 (ojo indeterminado) -> None")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    # Ojo izquierdo indeterminado (EAR 0.0), derecho OK
    d = p._construir_evento_dict(
        _datos_rostro(),
        _datos_ojos(ear_izq=0.0, ear_der=0.26),
        _datos_boca(),
        _datos_cabeza(),
        _datos_frote(),
    )
    assert d["ear_izquierdo"] is None, "EAR 0.0 deberia enviarse como None"
    assert abs(d["ear_derecho"] - 0.26) < 1e-6


@_test("numero_secuencia incrementa por mensaje")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    p.iniciar()
    try:
        p.publicar(_datos_rostro())
        p.publicar(_datos_rostro())
        p.publicar(_datos_rostro())
        stats = p.obtener_estadisticas()
        assert stats["numero_secuencia"] == 3, f"secuencia={stats['numero_secuencia']}"
    finally:
        p.detener()


@_test("id_sesion se genera al iniciar")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    assert p.id_sesion is None
    p.iniciar()
    assert p.id_sesion is not None
    assert len(p.id_sesion) > 0
    p.detener()


@_test("obtener_estadisticas devuelve dict coherente")
def _():
    p = PublicadorMQ(forzar_simulado=True)
    p.iniciar()
    try:
        for _ in range(4):
            p.publicar(_datos_rostro())
        stats = p.obtener_estadisticas()
        assert stats["modo_simulado"] is True
        assert stats["activo"] is True
        assert stats["mensajes_enviados"] == 4
        assert stats["mensajes_descartados"] == 0
        assert stats["tasa_envio"] == 1.0
    finally:
        p.detener()


@_test("Context manager funciona")
def _():
    with PublicadorMQ(forzar_simulado=True) as p:
        assert p.activo
        p.publicar(_datos_rostro())
        assert p.mensajes_enviados == 1
    assert not p.activo


# =============================================================================
# TESTS CON MODO REAL (requieren posix_ipc + contratos del Core)
# =============================================================================

if not MODO_REAL_DISPONIBLE:
    motivo = []
    if not POSIX_IPC_DISPONIBLE:
        motivo.append("posix_ipc no instalado")
    if not CONTRATOS_DISPONIBLES:
        motivo.append("contratos del Core no disponibles")
    print(f"\n--- Tests modo real: SALTEADOS ({', '.join(motivo)}) ---")
    print("    Para correrlos: pip install posix_ipc, y tener el paquete common/")
else:
    print("\n--- Tests de PublicadorMQ (modo real, POSIX MQ) ---")

    import posix_ipc

    # Usamos una cola de test distinta para no interferir con la real
    COLA_TEST = "/neurodrive_vision_test"

    def _limpiar_cola_test():
        try:
            posix_ipc.unlink_message_queue(COLA_TEST)
        except posix_ipc.ExistentialError:
            pass


    @_test("Modo real: abrir y cerrar cola POSIX MQ")
    def _():
        _limpiar_cola_test()
        p = PublicadorMQ(nombre_cola=COLA_TEST)
        assert p.modo_simulado is False
        p.iniciar()
        try:
            assert p.activo
        finally:
            p.detener()
            _limpiar_cola_test()


    @_test("Modo real: publicar Envelope real, tamanio < 1024 bytes")
    def _():
        _limpiar_cola_test()
        p = PublicadorMQ(nombre_cola=COLA_TEST)
        p.iniciar()
        try:
            ok = p.publicar(
                _datos_rostro(), _datos_ojos(), _datos_boca(),
                _datos_cabeza(), _datos_frote(),
            )
            assert ok is True
            assert p.mensajes_enviados == 1

            # Leemos el mensaje de la cola para verificar
            cola = posix_ipc.MessageQueue(COLA_TEST)
            mensaje, _prio = cola.receive(timeout=2)
            cola.close()
            tam = len(mensaje)
            print(f"    Tamanio del Envelope: {tam} bytes")
            assert tam < 1024, f"mensaje muy grande: {tam} bytes"
            # Verificar que es JSON valido
            import json
            data = json.loads(mensaje.decode("utf-8"))
            assert "tipo" in data or "payload_json" in data
        finally:
            p.detener()
            _limpiar_cola_test()


    @_test("Modo real: cola llena -> descarta sin congelar")
    def _():
        _limpiar_cola_test()
        # Cola chica de capacidad 3
        p = PublicadorMQ(nombre_cola=COLA_TEST, capacidad_cola=3)
        p.iniciar()
        try:
            # Publicamos 10 mensajes sin que nadie consuma.
            # Los primeros 3 entran, el resto se descarta.
            t0 = time.monotonic()
            for _ in range(10):
                p.publicar(_datos_rostro(), _datos_ojos())
            elapsed = time.monotonic() - t0

            print(f"    enviados={p.mensajes_enviados}, "
                  f"descartados={p.mensajes_descartados}, "
                  f"tiempo={elapsed*1000:.1f}ms")
            assert p.mensajes_enviados == 3, (
                f"deberian entrar 3, entraron {p.mensajes_enviados}"
            )
            assert p.mensajes_descartados == 7, (
                f"deberian descartarse 7, se descartaron {p.mensajes_descartados}"
            )
            # CRITICO: no debe haberse colgado. 10 publicaciones en < 500ms.
            assert elapsed < 0.5, (
                f"el publicador se congelo: {elapsed:.2f}s para 10 mensajes"
            )
        finally:
            p.detener()
            _limpiar_cola_test()


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
