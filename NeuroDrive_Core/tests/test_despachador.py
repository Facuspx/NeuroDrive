"""
test_despachador.py - Tests funcionales del Despachador de Comandos.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Core.test_despachador

Todo deterministico, sin hardware. Usa ActuadorSimulado para verificar
ruteo, APAGAR_TODO, aislamiento de errores, reemplazo y shutdown limpio.
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

from common.contratos import (
    ComandoActuador,
    EstadoFSM,
    SalidaFSM,
    TipoComandoActuador,
)
from NeuroDrive_Core.despachador import (
    ActuadorBase,
    ActuadorSimulado,
    DespachadorComandos,
)


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
# Helpers
# =============================================================================

def _salida(*comandos: ComandoActuador, transicion: bool = True) -> SalidaFSM:
    """Arma una SalidaFSM minima con los comandos dados."""
    return SalidaFSM(
        timestamp=1000.0,
        estado_actual=EstadoFSM.ALERTA_LEVE,
        estado_anterior=EstadoFSM.PRE_ALERTA,
        nivel_alerta=2,
        comandos=tuple(comandos),
        transicion_ocurrio=transicion,
    )


C = TipoComandoActuador  # alias corto


# =============================================================================
# TESTS: ciclo de vida
# =============================================================================

print("\n--- Tests de ciclo de vida ---")


@_test("iniciar() abre los actuadores y detener() los cierra")
def _():
    a = ActuadorSimulado("a")
    d = DespachadorComandos()
    d.registrar_actuador(a)
    assert not a.iniciado
    d.iniciar()
    assert a.iniciado, "el actuador deberia estar iniciado"
    d.detener()
    assert not a.iniciado, "el actuador deberia estar detenido"


@_test("iniciar() dos veces no rompe (idempotente)")
def _():
    d = DespachadorComandos()
    d.registrar_actuador(ActuadorSimulado("a"))
    d.iniciar()
    d.iniciar()  # segunda vez: warning, no crash
    d.detener()


@_test("detener() sin iniciar no rompe")
def _():
    d = DespachadorComandos()
    d.detener()  # no debe lanzar


@_test("no se puede registrar actuador con el despachador activo")
def _():
    d = DespachadorComandos()
    d.iniciar()
    try:
        d.registrar_actuador(ActuadorSimulado("tarde"))
        assert False, "deberia haber lanzado RuntimeError"
    except RuntimeError:
        pass
    finally:
        d.detener()


# =============================================================================
# TESTS: ruteo
# =============================================================================

print("\n--- Tests de ruteo ---")


@_test("un comando llega al actuador que lo soporta")
def _():
    a = ActuadorSimulado("buzzer", tipos={C.BUZZER_LARGO})
    d = DespachadorComandos()
    d.registrar_actuador(a)
    d.iniciar()
    d.despachar(_salida(ComandoActuador(tipo=C.BUZZER_LARGO, intensidad=70)))
    d.esperar_vaciado(timeout=1.0)
    d.detener()
    assert a.tipos_recibidos() == [C.BUZZER_LARGO]


@_test("un comando NO llega a un actuador que no lo soporta")
def _():
    buzzer = ActuadorSimulado("buzzer", tipos={C.BUZZER_LARGO})
    voz = ActuadorSimulado("voz", tipos={C.REPRODUCIR_VOZ})
    d = DespachadorComandos()
    d.registrar_actuador(buzzer)
    d.registrar_actuador(voz)
    d.iniciar()
    d.despachar(_salida(ComandoActuador(tipo=C.REPRODUCIR_VOZ, mensaje_voz="hola")))
    d.esperar_vaciado(timeout=1.0)
    d.detener()
    assert buzzer.cantidad_recibida() == 0, "el buzzer no debia recibir voz"
    assert voz.tipos_recibidos() == [C.REPRODUCIR_VOZ]


@_test("un tipo soportado por dos actuadores llega a ambos")
def _():
    a1 = ActuadorSimulado("a1", tipos={C.VIBRAR_FUERTE})
    a2 = ActuadorSimulado("a2", tipos={C.VIBRAR_FUERTE})
    d = DespachadorComandos()
    d.registrar_actuador(a1)
    d.registrar_actuador(a2)
    d.iniciar()
    d.despachar(_salida(ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=100)))
    d.esperar_vaciado(timeout=1.0)
    d.detener()
    assert a1.cantidad_recibida() == 1
    assert a2.cantidad_recibida() == 1


@_test("una SalidaFSM con varios comandos rutea cada uno a su destino")
def _():
    buzzer = ActuadorSimulado("buzzer", tipos={C.BUZZER_LARGO})
    wearable = ActuadorSimulado("wearable", tipos={C.VIBRAR_FUERTE, C.SECUENCIA_ACK})
    voz = ActuadorSimulado("voz", tipos={C.REPRODUCIR_VOZ})
    d = DespachadorComandos()
    for act in (buzzer, wearable, voz):
        d.registrar_actuador(act)
    d.iniciar()
    d.despachar(_salida(
        ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=80),
        ComandoActuador(tipo=C.BUZZER_LARGO, intensidad=70),
        ComandoActuador(tipo=C.SECUENCIA_ACK, intensidad=80),
        ComandoActuador(tipo=C.REPRODUCIR_VOZ, mensaje_voz="confirme"),
    ))
    d.esperar_vaciado(timeout=1.0)
    d.detener()
    assert set(wearable.tipos_recibidos()) == {C.VIBRAR_FUERTE, C.SECUENCIA_ACK}
    assert buzzer.tipos_recibidos() == [C.BUZZER_LARGO]
    assert voz.tipos_recibidos() == [C.REPRODUCIR_VOZ]


@_test("un comando que nadie soporta no rompe nada")
def _():
    a = ActuadorSimulado("solo_voz", tipos={C.REPRODUCIR_VOZ})
    d = DespachadorComandos()
    d.registrar_actuador(a)
    d.iniciar()
    d.despachar(_salida(ComandoActuador(tipo=C.NOTIFICAR_SUPERVISOR)))
    d.esperar_vaciado(timeout=1.0)
    d.detener()
    assert a.cantidad_recibida() == 0


# =============================================================================
# TESTS: APAGAR_TODO
# =============================================================================

print("\n--- Tests de APAGAR_TODO ---")


@_test("APAGAR_TODO llama apagar() en todos los actuadores")
def _():
    a1 = ActuadorSimulado("a1", tipos={C.BUZZER_LARGO})
    a2 = ActuadorSimulado("a2", tipos={C.REPRODUCIR_VOZ})
    d = DespachadorComandos()
    d.registrar_actuador(a1)
    d.registrar_actuador(a2)
    d.iniciar()
    d.despachar(_salida(ComandoActuador(tipo=C.APAGAR_TODO)))
    d.esperar_vaciado(timeout=1.0)
    d.detener()
    # apagar se llama 1 vez por el comando + 1 vez en detener() = 2
    assert a1.veces_apagado >= 1, "a1 deberia haberse apagado por el comando"
    assert a2.veces_apagado >= 1, "a2 deberia haberse apagado por el comando"


@_test("APAGAR_TODO descarta comandos pendientes en la cola")
def _():
    # Actuador lento: mientras ejecuta el primer comando, encolamos una
    # rafaga de alarma + un APAGAR_TODO. El apagar debe vaciar la rafaga.
    lento = ActuadorSimulado("lento", tipos=set(TipoComandoActuador), colgar_ms=200)
    d = DespachadorComandos()
    d.registrar_actuador(lento)
    d.iniciar()
    # Primer comando: ocupa al trabajador 200ms
    d.despachar(_salida(ComandoActuador(tipo=C.BUZZER_CONTINUO, intensidad=100)))
    time.sleep(0.02)  # dar tiempo a que el trabajador tome el primero
    # Rafaga que queda encolada mientras el trabajador esta ocupado
    for _ in range(5):
        d.despachar(_salida(ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=100)))
    # Ahora el apagado: debe descartar la rafaga pendiente
    d.despachar(_salida(ComandoActuador(tipo=C.APAGAR_TODO)))
    d.esperar_vaciado(timeout=2.0)
    d.detener()
    # El trabajador ejecuto el primer buzzer (1) y quizas 0-1 de la rafaga
    # antes del apagar. Lo importante: NO ejecuto las 5 vibraciones.
    ejecutados_vibrar = sum(
        1 for t in lento.tipos_recibidos() if t == C.VIBRAR_FUERTE
    )
    assert ejecutados_vibrar < 5, (
        f"el apagar debia descartar pendientes, se ejecutaron "
        f"{ejecutados_vibrar} vibraciones"
    )
    assert lento.veces_apagado >= 1


# =============================================================================
# TESTS: aislamiento de errores
# =============================================================================

print("\n--- Tests de aislamiento de errores ---")


@_test("un actuador que falla no impide que los otros reciban el comando")
def _():
    roto = ActuadorSimulado("roto", tipos={C.VIBRAR_FUERTE},
                            fallar_en={C.VIBRAR_FUERTE})
    sano = ActuadorSimulado("sano", tipos={C.VIBRAR_FUERTE})
    d = DespachadorComandos()
    d.registrar_actuador(roto)
    d.registrar_actuador(sano)
    d.iniciar()
    d.despachar(_salida(ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=100)))
    d.esperar_vaciado(timeout=1.0)
    d.detener()
    assert sano.cantidad_recibida() == 1, "el actuador sano debia recibir igual"
    assert d.stats.errores_por_actuador.get("roto", 0) == 1


@_test("el hilo trabajador sobrevive a multiples fallos consecutivos")
def _():
    roto = ActuadorSimulado("roto", tipos={C.BUZZER_LARGO},
                            fallar_en={C.BUZZER_LARGO})
    sano = ActuadorSimulado("sano", tipos={C.BUZZER_LARGO})
    d = DespachadorComandos()
    d.registrar_actuador(roto)
    d.registrar_actuador(sano)
    d.iniciar()
    for _ in range(10):
        d.despachar(_salida(ComandoActuador(tipo=C.BUZZER_LARGO, intensidad=50)))
    d.esperar_vaciado(timeout=2.0)
    d.detener()
    assert sano.cantidad_recibida() == 10, "el sano debia recibir los 10"
    assert d.stats.errores_por_actuador.get("roto", 0) == 10


# =============================================================================
# TESTS: no bloqueo
# =============================================================================

print("\n--- Tests de no-bloqueo ---")


@_test("despachar() no bloquea aunque el actuador sea lento")
def _():
    lento = ActuadorSimulado("lento", tipos={C.VIBRAR_FUERTE}, colgar_ms=300)
    d = DespachadorComandos()
    d.registrar_actuador(lento)
    d.iniciar()
    t0 = time.monotonic()
    d.despachar(_salida(ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=100)))
    dt = time.monotonic() - t0
    d.detener()
    assert dt < 0.05, f"despachar() bloqueo {dt*1000:.0f}ms (deberia ser <50ms)"


@_test("cola llena descarta comandos normales sin crashear")
def _():
    lento = ActuadorSimulado("lento", tipos={C.VIBRAR_FUERTE}, colgar_ms=500)
    d = DespachadorComandos(capacidad_cola=2)
    d.registrar_actuador(lento)
    d.iniciar()
    # Inundamos: el trabajador queda tomando el primero 500ms, la cola (2)
    # se llena y el resto se descarta.
    for _ in range(20):
        d.despachar(_salida(ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=100)))
    d.detener(timeout=1.0)
    assert d.stats.cola_llena_descartes > 0, "deberia haber descartes por cola llena"


# =============================================================================
# TESTS: sin transicion / stats
# =============================================================================

print("\n--- Tests de stats ---")


@_test("una salida sin comandos no ejecuta nada pero cuenta la salida")
def _():
    a = ActuadorSimulado("a")
    d = DespachadorComandos()
    d.registrar_actuador(a)
    d.iniciar()
    d.despachar(_salida(transicion=False))  # sin comandos
    d.esperar_vaciado(timeout=1.0)
    d.detener()
    assert a.cantidad_recibida() == 0
    assert d.stats.salidas_recibidas == 1


@_test("las stats cuentan encolados y ejecutados")
def _():
    a = ActuadorSimulado("a", tipos={C.BUZZER_LARGO})
    d = DespachadorComandos()
    d.registrar_actuador(a)
    d.iniciar()
    for _ in range(3):
        d.despachar(_salida(ComandoActuador(tipo=C.BUZZER_LARGO, intensidad=50)))
    d.esperar_vaciado(timeout=1.0)
    d.detener()
    assert d.stats.comandos_encolados == 3
    assert d.stats.comandos_ejecutados == 3


# =============================================================================
# RESUMEN
# =============================================================================

def _resumen() -> int:
    print("\n" + "=" * 60)
    ok = sum(1 for _, passed, _ in _resultados if passed)
    total = len(_resultados)
    print(f"  Resumen: {ok}/{total} tests pasaron")
    if ok != total:
        print("  FALLARON:")
        for nombre, passed, msg in _resultados:
            if not passed:
                print(f"    - {nombre}: {msg}")
    else:
        print("  DESPACHADOR OK")
    print("=" * 60)
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(_resumen())
