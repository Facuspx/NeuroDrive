"""
test_buzzer.py - Tests del Actuador Buzzer.

Dos modos:

  1. LOGICA (default, sin hardware):
        python -m NeuroDrive_Core.actuadores.test_buzzer
     Usa BackendSimuladoGPIO. Valida timing, reemplazo, apagado, e
     integracion con el despachador. Deterministico.

  2. HARDWARE REAL (hace sonar el buzzer en la Pi):
        python -m NeuroDrive_Core.actuadores.test_buzzer --real
        python -m NeuroDrive_Core.actuadores.test_buzzer --real --pin 18
     Usa BackendLgpio. Suena una secuencia de beeps para verificar la
     conexion fisica y que el gpiochip detectado es el correcto.
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
from NeuroDrive_Core.despachador import DespachadorComandos
from NeuroDrive_Core.actuadores.buzzer import (
    ActuadorBuzzer,
    BackendSimuladoGPIO,
    detectar_gpiochip,
)

C = TipoComandoActuador

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


def _cmd(tipo, dur_ms=0, inten=80):
    return ComandoActuador(tipo=tipo, intensidad=inten, duracion_ms=dur_ms)


def _buzzer_simulado(dur_corto_ms=40, dur_largo_ms=80):
    backend = BackendSimuladoGPIO()
    buz = ActuadorBuzzer(
        pin=18, backend=backend,
        dur_corto_ms=dur_corto_ms, dur_largo_ms=dur_largo_ms,
    )
    return buz, backend


# =============================================================================
# TESTS DE LOGICA
# =============================================================================

def correr_tests_logica() -> int:
    print("\n--- Ciclo de vida ---")

    @_test("iniciar() abre el backend y deja el pin en LOW")
    def _():
        buz, backend = _buzzer_simulado()
        buz.iniciar()
        assert backend.abierto
        assert backend.ultimo_nivel() == 0, "deberia arrancar en LOW"
        buz.detener()

    @_test("detener() cierra el backend y baja el pin")
    def _():
        buz, backend = _buzzer_simulado()
        buz.iniciar()
        buz.detener()
        assert backend.cerrado
        assert backend.ultimo_nivel() == 0

    @_test("tipos_soportados son los tres de buzzer, sin APAGAR_TODO")
    def _():
        buz, _ = _buzzer_simulado()
        assert buz.tipos_soportados() == {
            C.BUZZER_CORTO, C.BUZZER_LARGO, C.BUZZER_CONTINUO
        }

    @_test("ejecutar() sin iniciar() lanza error")
    def _():
        buz, _ = _buzzer_simulado()
        try:
            buz.ejecutar(_cmd(C.BUZZER_CORTO))
            assert False, "deberia haber lanzado RuntimeError"
        except RuntimeError:
            pass

    print("\n--- Beep temporizado ---")

    @_test("BUZZER_CORTO enciende y se apaga solo tras la duracion")
    def _():
        buz, backend = _buzzer_simulado(dur_corto_ms=40)
        buz.iniciar()
        buz.ejecutar(_cmd(C.BUZZER_CORTO))  # usa default 40ms
        assert buz.sonando, "deberia estar sonando justo despues"
        assert backend.ultimo_nivel() == 1, "pin deberia estar HIGH"
        time.sleep(0.12)
        assert not buz.sonando, "deberia haberse apagado solo"
        assert backend.ultimo_nivel() == 0, "pin deberia estar LOW"
        buz.detener()

    @_test("BUZZER_LARGO respeta duracion_ms explicita del comando")
    def _():
        buz, backend = _buzzer_simulado()
        buz.iniciar()
        buz.ejecutar(_cmd(C.BUZZER_LARGO, dur_ms=60))
        assert buz.sonando
        time.sleep(0.03)
        assert buz.sonando, "a los 30ms deberia seguir sonando (dur=60ms)"
        time.sleep(0.09)
        assert not buz.sonando, "a los 120ms ya deberia estar apagado"
        buz.detener()

    @_test("BUZZER_CONTINUO queda sonando hasta apagar()")
    def _():
        buz, backend = _buzzer_simulado()
        buz.iniciar()
        buz.ejecutar(_cmd(C.BUZZER_CONTINUO))
        assert buz.sonando
        time.sleep(0.15)
        assert buz.sonando, "continuo no deberia apagarse solo"
        assert backend.ultimo_nivel() == 1
        buz.apagar()
        assert not buz.sonando
        assert backend.ultimo_nivel() == 0
        buz.detener()

    print("\n--- Reemplazo y carrera de timers ---")

    @_test("un CONTINUO tras un CORTO no se apaga por el timer del CORTO")
    def _():
        # Regresion del bug de generacion: el timer del CORTO (40ms) NO debe
        # apagar el CONTINUO que arranco a los 10ms.
        buz, backend = _buzzer_simulado(dur_corto_ms=40)
        buz.iniciar()
        buz.ejecutar(_cmd(C.BUZZER_CORTO))   # timer a 40ms
        time.sleep(0.01)
        buz.ejecutar(_cmd(C.BUZZER_CONTINUO))  # toma el control
        time.sleep(0.08)  # pasa el vencimiento del timer viejo (40ms)
        assert buz.sonando, "el CONTINUO no debia ser apagado por el timer viejo"
        assert backend.ultimo_nivel() == 1
        buz.apagar()
        buz.detener()

    @_test("un beep nuevo reemplaza al anterior (siempre HIGH, sin hueco)")
    def _():
        buz, backend = _buzzer_simulado()
        buz.iniciar()
        buz.ejecutar(_cmd(C.BUZZER_LARGO, dur_ms=200))
        time.sleep(0.02)
        buz.ejecutar(_cmd(C.BUZZER_LARGO, dur_ms=200))
        assert buz.sonando
        assert backend.ultimo_nivel() == 1
        buz.apagar()
        buz.detener()

    print("\n--- Integracion con el despachador ---")

    @_test("el despachador rutea un BUZZER_LARGO al buzzer")
    def _():
        buz, backend = _buzzer_simulado(dur_largo_ms=40)
        d = DespachadorComandos()
        d.registrar_actuador(buz)
        d.iniciar()
        salida = SalidaFSM(
            timestamp=1000.0,
            estado_actual=EstadoFSM.ALERTA_MEDIA,
            estado_anterior=EstadoFSM.ALERTA_LEVE,
            nivel_alerta=3,
            comandos=(_cmd(C.BUZZER_LARGO, dur_ms=40),),
            transicion_ocurrio=True,
        )
        d.despachar(salida)
        d.esperar_vaciado(timeout=1.0)
        time.sleep(0.02)
        assert 1 in backend.niveles(), "el buzzer deberia haber sonado"
        d.detener()
        assert backend.ultimo_nivel() == 0, "tras detener, pin en LOW"

    @_test("APAGAR_TODO del despachador apaga el buzzer continuo")
    def _():
        buz, backend = _buzzer_simulado()
        d = DespachadorComandos()
        d.registrar_actuador(buz)
        d.iniciar()
        # Primero CONTINUO (CRITICO)
        d.despachar(SalidaFSM(
            timestamp=1000.0, estado_actual=EstadoFSM.CRITICO,
            estado_anterior=EstadoFSM.ALERTA_MEDIA, nivel_alerta=4,
            comandos=(_cmd(C.BUZZER_CONTINUO),), transicion_ocurrio=True,
        ))
        d.esperar_vaciado(timeout=1.0)
        time.sleep(0.02)
        assert buz.sonando, "deberia estar sonando en CRITICO"
        # Ahora baja a PRE_ALERTA (ACK correcto) -> APAGAR_TODO
        d.despachar(SalidaFSM(
            timestamp=1001.0, estado_actual=EstadoFSM.PRE_ALERTA,
            estado_anterior=EstadoFSM.CRITICO, nivel_alerta=1,
            comandos=(_cmd(C.APAGAR_TODO, inten=0),), transicion_ocurrio=True,
        ))
        d.esperar_vaciado(timeout=1.0)
        time.sleep(0.02)
        assert not buz.sonando, "APAGAR_TODO deberia haber callado el buzzer"
        assert backend.ultimo_nivel() == 0
        d.detener()

    return _resumen()


def _resumen() -> int:
    print("\n" + "=" * 60)
    ok = sum(1 for _, p, _ in _resultados if p)
    total = len(_resultados)
    print(f"  Resumen: {ok}/{total} tests pasaron")
    if ok != total:
        print("  FALLARON:")
        for nombre, p, msg in _resultados:
            if not p:
                print(f"    - {nombre}: {msg}")
    else:
        print("  BUZZER (logica) OK")
    print("=" * 60)
    return 0 if ok == total else 1


# =============================================================================
# MODO HARDWARE REAL
# =============================================================================

def correr_smoke_real(pin: int) -> int:
    print(f"\n=== SMOKE TEST REAL DEL BUZZER (pin BCM {pin}) ===")
    chip = detectar_gpiochip()
    print(f"gpiochip detectado: {chip}")
    print("Vas a escuchar: 3 beeps cortos, 1 largo, y 1 continuo de 2s.")
    print("Si no suena nada, proba forzar el otro chip o revisa el cableado.\n")

    buz = ActuadorBuzzer(pin=pin)  # backend real lgpio
    try:
        buz.iniciar()
    except Exception as e:
        print(f"[ERROR] No se pudo abrir el GPIO: {e}")
        print("Sugerencia: instala lgpio (pip install lgpio) y gpiod "
              "(sudo apt install gpiod), y verifica permisos.")
        return 1

    try:
        print(">> 3 cortos")
        for _ in range(3):
            buz.ejecutar(_cmd(C.BUZZER_CORTO, dur_ms=120))
            time.sleep(0.35)
        time.sleep(0.5)

        print(">> 1 largo (800ms)")
        buz.ejecutar(_cmd(C.BUZZER_LARGO, dur_ms=800))
        time.sleep(1.2)

        print(">> 1 continuo (2s)")
        buz.ejecutar(_cmd(C.BUZZER_CONTINUO))
        time.sleep(2.0)
        buz.apagar()
        print(">> apagado")
    finally:
        buz.detener()

    print("\nSi escuchaste todos los beeps, el buzzer y el gpiochip estan OK.")
    return 0


if __name__ == "__main__":
    if "--real" in sys.argv:
        pin = 18
        if "--pin" in sys.argv:
            try:
                pin = int(sys.argv[sys.argv.index("--pin") + 1])
            except (ValueError, IndexError):
                print("Uso: --pin <numero_bcm>")
                sys.exit(2)
        sys.exit(correr_smoke_real(pin))
    else:
        sys.exit(correr_tests_logica())
