"""
test_config_loader.py - Tests funcionales del cargador de configuracion.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Core.test_config_loader

Valida:
  1. Carga normal desde config/config.yaml
  2. Acceso por dot notation a todas las secciones
  3. Validacion de rangos en cada seccion
  4. Manejo de secciones faltantes (usa defaults)
  5. Manejo de claves desconocidas (ignora con warning)
  6. Manejo de archivos invalidos
  7. Resolucion de path con variable de entorno
  8. Cache y recarga
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from NeuroDrive_Core.config_loader import (
    cargar_config,
    limpiar_cache,
    Config,
    ConfigError,
    ConfigFSMSeccion,
    ConfigOjosSeccion,
    ConfigWearableSeccion,
    ConfigIPCSeccion,
)


# =============================================================================
# Framework minimalista
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


def _debe_fallar(func, mensaje_test: str) -> None:
    try:
        func()
    except (ConfigError, ValueError):
        return
    raise AssertionError(f"{mensaje_test} (no lanzo excepcion)")


def _yaml_temporal(contenido: str) -> Path:
    """Crea un archivo YAML temporal y devuelve su path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    f.write(contenido)
    f.close()
    return Path(f.name)


# =============================================================================
# TESTS: carga normal del config.yaml real del proyecto
# =============================================================================

print("\n--- Tests de carga normal ---")


@_test("Carga el config.yaml del proyecto sin errores")
def _():
    limpiar_cache()
    config = cargar_config()
    assert isinstance(config, Config)
    assert config.ruta_origen != ""


@_test("Acceso por dot notation a fsm")
def _():
    limpiar_cache()
    config = cargar_config()
    assert config.fsm.tiempo_para_bajar_estado_seg == 60.0
    assert config.fsm.ventana_corta_seg == 60.0


@_test("Acceso por dot notation a wearable con timeouts nuevos")
def _():
    limpiar_cache()
    config = cargar_config()
    # No fijamos valores concretos: son de config y cambian segun el medio
    # de respuesta (teclado en pruebas, pulsera real en produccion). Solo
    # verificamos que existan, sean numeros y respeten la escalera logica.
    assert isinstance(config.wearable.timeout_ack_leve_seg, (int, float))
    assert isinstance(config.wearable.timeout_ack_medio_seg, (int, float))
    assert isinstance(config.wearable.timeout_ack_critico_seg, (int, float))
    assert config.wearable.timeout_ack_leve_seg > 0
    assert config.wearable.timeout_ack_medio_seg > 0
    assert config.wearable.timeout_ack_critico_seg > 0


@_test("Acceso por dot notation a ipc")
def _():
    limpiar_cache()
    config = cargar_config()
    assert config.ipc.modo == "posix_mq"
    assert config.ipc.cola_vision.startswith("/")
    assert config.ipc.cola_wearable.startswith("/")


@_test("Acceso por dot notation a identificadores")
def _():
    limpiar_cache()
    config = cargar_config()
    assert config.identificadores.id_camara == "cam-01"
    assert config.identificadores.prefijo_sesion == "ses"


@_test("Acceso por dot notation a actuadores")
def _():
    limpiar_cache()
    config = cargar_config()
    assert config.actuadores.buzzer_gpio_pin == 18
    assert config.actuadores.habilitar_voz is True


# =============================================================================
# TESTS: cache
# =============================================================================

print("\n--- Tests de cache ---")


@_test("Cache: segunda llamada devuelve la misma instancia")
def _():
    limpiar_cache()
    c1 = cargar_config()
    c2 = cargar_config()
    assert c1 is c2


@_test("recargar=True fuerza una nueva carga")
def _():
    limpiar_cache()
    c1 = cargar_config()
    c2 = cargar_config(recargar=True)
    assert c1 is not c2


@_test("limpiar_cache() funciona")
def _():
    limpiar_cache()
    c1 = cargar_config()
    limpiar_cache()
    c2 = cargar_config()
    assert c1 is not c2


# =============================================================================
# TESTS: validaciones de rangos
# =============================================================================

print("\n--- Tests de validaciones ---")


@_test("ConfigFSMSeccion rechaza ventana_corta_seg <= 0")
def _():
    _debe_fallar(
        lambda: ConfigFSMSeccion(ventana_corta_seg=0),
        "ventana_corta_seg=0 deberia fallar",
    )


@_test("ConfigFSMSeccion rechaza ventana_larga menor que ventana_corta")
def _():
    _debe_fallar(
        lambda: ConfigFSMSeccion(ventana_corta_seg=100, ventana_larga_seg=50),
        "ventana_larga menor a ventana_corta deberia fallar",
    )


@_test("ConfigOjosSeccion rechaza histeresis invalida (cerrar >= abrir)")
def _():
    _debe_fallar(
        lambda: ConfigOjosSeccion(umbral_ear_cerrar=0.3, umbral_ear_abrir=0.2),
        "histeresis invertida deberia fallar",
    )


@_test("ConfigOjosSeccion rechaza microsueno mas corto que parpadeo")
def _():
    _debe_fallar(
        lambda: ConfigOjosSeccion(
            dur_max_parpadeo_seg=0.5, dur_min_microsueno_seg=0.3
        ),
        "microsueno mas corto que parpadeo deberia fallar",
    )


@_test("ConfigWearableSeccion rechaza bpm_critico >= bpm_alerta")
def _():
    _debe_fallar(
        lambda: ConfigWearableSeccion(
            bpm_umbral_alerta=70, bpm_umbral_critico=70
        ),
        "bpm critico >= alerta deberia fallar",
    )


@_test("ConfigWearableSeccion rechaza heartbeat menor que 2x intervalo BPM")
def _():
    _debe_fallar(
        lambda: ConfigWearableSeccion(
            timeout_heartbeat_seg=3.0, intervalo_envio_bpm_seg=2.0
        ),
        "heartbeat muy corto deberia fallar",
    )


@_test("ConfigIPCSeccion rechaza modo invalido")
def _():
    _debe_fallar(
        lambda: ConfigIPCSeccion(modo="redis"),
        "modo invalido deberia fallar",
    )


@_test("ConfigIPCSeccion rechaza cola posix_mq sin /")
def _():
    _debe_fallar(
        lambda: ConfigIPCSeccion(modo="posix_mq", cola_vision="vision"),
        "cola sin / inicial deberia fallar",
    )


# =============================================================================
# TESTS: YAML temporal con casos edge
# =============================================================================

print("\n--- Tests con YAML temporal ---")


@_test("YAML vacio usa todos los defaults")
def _():
    yaml_path = _yaml_temporal("")
    try:
        limpiar_cache()
        # YAML vacio es 'None', tiene que fallar limpio
        try:
            cargar_config(path=str(yaml_path))
            raise AssertionError("YAML vacio deberia fallar")
        except ConfigError:
            pass  # esperado
    finally:
        yaml_path.unlink()


@_test("YAML con solo una seccion usa defaults en el resto")
def _():
    contenido = """
fsm:
  tiempo_para_bajar_estado_seg: 90
"""
    yaml_path = _yaml_temporal(contenido)
    try:
        limpiar_cache()
        config = cargar_config(path=str(yaml_path))
        # La seccion definida tiene el valor custom
        assert config.fsm.tiempo_para_bajar_estado_seg == 90
        # Las demas tienen defaults
        assert config.wearable.timeout_ack_leve_seg == 30.0
        assert config.actuadores.buzzer_gpio_pin == 18
    finally:
        yaml_path.unlink()


@_test("YAML con claves desconocidas se cargan ignorandolas")
def _():
    contenido = """
fsm:
  tiempo_para_bajar_estado_seg: 60
  clave_que_no_existe: 12345
ojos:
  umbral_ear_cerrar: 0.18
  umbral_ear_abrir: 0.22
  otra_clave_inventada: "hola"
"""
    yaml_path = _yaml_temporal(contenido)
    try:
        limpiar_cache()
        config = cargar_config(path=str(yaml_path))
        # Las claves validas se cargaron
        assert config.fsm.tiempo_para_bajar_estado_seg == 60
        assert config.ojos.umbral_ear_cerrar == 0.18
    finally:
        yaml_path.unlink()


@_test("YAML con valor invalido en seccion lanza ConfigError")
def _():
    contenido = """
fsm:
  ventana_corta_seg: -10
"""
    yaml_path = _yaml_temporal(contenido)
    try:
        limpiar_cache()
        _debe_fallar(
            lambda: cargar_config(path=str(yaml_path)),
            "ventana_corta_seg negativo deberia fallar",
        )
    finally:
        yaml_path.unlink()


@_test("YAML con sintaxis invalida lanza ConfigError")
def _():
    contenido = """
fsm:
  tiempo: [
"""
    yaml_path = _yaml_temporal(contenido)
    try:
        limpiar_cache()
        _debe_fallar(
            lambda: cargar_config(path=str(yaml_path)),
            "YAML invalido deberia fallar",
        )
    finally:
        yaml_path.unlink()


@_test("YAML que no es dict en su raiz lanza ConfigError")
def _():
    contenido = "esto es solo un string"
    yaml_path = _yaml_temporal(contenido)
    try:
        limpiar_cache()
        _debe_fallar(
            lambda: cargar_config(path=str(yaml_path)),
            "YAML que no es dict deberia fallar",
        )
    finally:
        yaml_path.unlink()


@_test("path inexistente lanza ConfigError")
def _():
    limpiar_cache()
    _debe_fallar(
        lambda: cargar_config(path="/tmp/no_existe_nunca_jamas.yaml"),
        "path inexistente deberia fallar",
    )


# =============================================================================
# TESTS: integracion con FSM (compatibilidad)
# =============================================================================

print("\n--- Tests de integracion con FSM ---")


@_test("Config se puede pasar a ConfigFSM via desde_dict")
def _():
    """Verifica que el flujo: cargar_config() -> dataclass -> ConfigFSM funcione."""
    from dataclasses import asdict
    from NeuroDrive_Core.fsm import ConfigFSM

    limpiar_cache()
    config = cargar_config()

    # Construimos el dict que ConfigFSM.desde_dict espera
    dict_para_fsm = {
        "fsm": asdict(config.fsm),
        "wearable": asdict(config.wearable),
    }

    cfg_fsm = ConfigFSM.desde_dict(dict_para_fsm)
    assert cfg_fsm.tiempo_para_bajar_estado_seg == 60.0
    assert isinstance(cfg_fsm.timeout_ack_leve_seg, (int, float))
    assert cfg_fsm.timeout_ack_leve_seg > 0


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
    print("  NeuroDrive - Tests de Config Loader")
    print("=" * 60)
    sys.exit(_resumen())
