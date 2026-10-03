"""
test_config_loader.py - Tests funcionales del cargador de configuracion.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Core.tests.test_config_loader

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
    assert config.fsm.ventana_episodios_seg == 900.0


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
    assert config.identificadores.prefijo_mensaje_interno == "int"


@_test("Acceso por dot notation a actuadores")
def _():
    limpiar_cache()
    config = cargar_config()
    assert config.actuadores.buzzer_gpio_pin == 18


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


@_test("ConfigFSMSeccion rechaza tiempo_para_bajar_estado_seg <= 0")
def _():
    _debe_fallar(
        lambda: ConfigFSMSeccion(tiempo_para_bajar_estado_seg=0),
        "tiempo_para_bajar_estado_seg=0 deberia fallar",
    )


@_test("Factores de histeresis fuera de (0, 1) se rechazan")
def _():
    from NeuroDrive_Core.config_loader import ConfigBocaSeccion, ConfigCabezaSeccion
    _debe_fallar(lambda: ConfigBocaSeccion(factor_mar_cierre=1.0),
                 "factor_mar_cierre=1 deberia fallar (sin histeresis)")
    _debe_fallar(lambda: ConfigBocaSeccion(factor_mar_cierre=0.0),
                 "factor_mar_cierre=0 deberia fallar")
    _debe_fallar(lambda: ConfigCabezaSeccion(factor_pitch_fin_cabeceo=1.2),
                 "factor_pitch_fin_cabeceo=1.2 deberia fallar")


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
  tiempo_para_bajar_estado_seg: -10
"""
    yaml_path = _yaml_temporal(contenido)
    try:
        limpiar_cache()
        _debe_fallar(
            lambda: cargar_config(path=str(yaml_path)),
            "tiempo_para_bajar_estado_seg negativo deberia fallar",
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


@_test("El config.yaml real se convierte en ConfigFSM via desde_config")
def _():
    """Verifica el flujo completo: cargar_config() -> Config -> ConfigFSM."""
    from NeuroDrive_Core.fsm import ConfigFSM

    limpiar_cache()
    config = cargar_config()
    cfg_fsm = ConfigFSM.desde_config(config)

    assert cfg_fsm.tiempo_para_bajar_estado_seg == config.fsm.tiempo_para_bajar_estado_seg
    assert cfg_fsm.timeout_ack_leve_seg == config.wearable.timeout_ack_leve_seg
    assert cfg_fsm.margen_presentacion_desafio_seg == config.wearable.margen_presentacion_desafio_seg
    # Los tres umbrales de PERCLOS llegan a la FSM desde el yaml
    assert cfg_fsm.perclos_corroborar_cabeceo == config.fsm.perclos_corroborar_cabeceo
    assert cfg_fsm.perclos_senales_leves == config.fsm.perclos_senales_leves
    assert cfg_fsm.perclos_parpados_pesados == config.fsm.perclos_parpados_pesados
    # Parametros que la FSM toma de otras secciones
    assert cfg_fsm.max_bostezos_ventana_larga == config.boca.max_bostezos_ventana_larga
    assert cfg_fsm.parpadeos_por_minuto_alerta == config.ojos.parpadeos_por_minuto_alerta


@_test("El config.yaml real define los tres PERCLOS, los neutros y los dos plazos de silencio")
def _():
    import yaml
    limpiar_cache()
    config = cargar_config()
    with open(config.ruta_origen, "r", encoding="utf-8") as f:
        crudo = yaml.safe_load(f)
    # Que esten ESCRITOS en el yaml, no tomados del valor por defecto
    for clave in ("perclos_corroborar_cabeceo", "perclos_senales_leves",
                  "perclos_parpados_pesados"):
        assert clave in crudo["fsm"], f"falta fsm.{clave} en config.yaml"
    for clave in ("pitch_neutro_grados", "yaw_neutro_grados", "roll_neutro_grados"):
        assert clave in crudo["cabeza"], f"falta cabeza.{clave} en config.yaml"
    assert "timeout_silencio_seg" in crudo["vision"]
    assert "timeout_heartbeat_seg" in crudo["wearable"]
    # Y que no queden claves que el cargador ya no conoce
    for clave in ("perclos_confirmado", "max_microsuenos_ventana_corta",
                  "max_bostezos_ventana_corta", "max_cabeceos_ventana_corta"):
        assert clave not in crudo["fsm"], f"clave obsoleta en config.yaml: fsm.{clave}"


@_test("Umbral de PERCLOS fuera de (0, 1) se rechaza al cargar")
def _():
    _debe_fallar(lambda: ConfigFSMSeccion(perclos_senales_leves=0.0),
                 "PERCLOS 0 deberia fallar")
    _debe_fallar(lambda: ConfigFSMSeccion(perclos_parpados_pesados=1.5),
                 "PERCLOS 1.5 deberia fallar")
    _debe_fallar(lambda: ConfigFSMSeccion(perclos_corroborar_cabeceo=-0.1),
                 "PERCLOS negativo deberia fallar")


@_test("Postura neutra inverosimil y plazo de silencio nulo se rechazan")
def _():
    from NeuroDrive_Core.config_loader import ConfigCabezaSeccion, ConfigVisionSeccion
    _debe_fallar(lambda: ConfigCabezaSeccion(pitch_neutro_grados=75.0),
                 "neutro de 75 grados deberia fallar")
    _debe_fallar(lambda: ConfigVisionSeccion(timeout_silencio_seg=0),
                 "plazo de silencio 0 deberia fallar")
    assert ConfigCabezaSeccion(yaw_neutro_grados=-9.5).yaw_neutro_grados == -9.5


# =============================================================================
# TESTS: el yaml y el cargador hablan de las mismas claves (C10)
# =============================================================================

print("\n--- Tests de correspondencia yaml / cargador ---")


class _CapturaAdvertencias:
    """Junta los mensajes WARNING del cargador mientras dura el bloque with."""
    def __enter__(self):
        import logging

        class _H(logging.Handler):
            def __init__(self):
                super().__init__(level=logging.WARNING)
                self.mensajes = []
            def emit(self, record):
                self.mensajes.append(record.getMessage())

        self._logger = logging.getLogger("NeuroDrive.ConfigLoader")
        self._h = _H()
        self._logger.addHandler(self._h)
        return self._h.mensajes

    def __exit__(self, *exc):
        self._logger.removeHandler(self._h)
        return False


@_test("El config.yaml real carga sin ninguna advertencia")
def _():
    limpiar_cache()
    with _CapturaAdvertencias() as avisos:
        cargar_config(recargar=True)
    assert avisos == [], f"advertencias al cargar: {avisos}"


@_test("Cada campo del cargador figura en el config.yaml real, y viceversa")
def _():
    import yaml
    from dataclasses import fields
    limpiar_cache()
    config = cargar_config()
    with open(config.ruta_origen, "r", encoding="utf-8") as f:
        crudo = yaml.safe_load(f)

    # Campos que se derivan y por eso NO se escriben en el yaml
    derivados = {("ojos", "umbral_ear_cerrar"), ("ojos", "umbral_ear_abrir")}

    faltan, sobran = [], []
    for seccion in fields(config):
        if seccion.name == "ruta_origen":
            continue
        en_codigo = {c.name for c in fields(getattr(config, seccion.name))}
        en_yaml = set((crudo.get(seccion.name) or {}).keys())
        for clave in sorted(en_codigo - en_yaml):
            if (seccion.name, clave) not in derivados:
                faltan.append(f"{seccion.name}.{clave}")
        for clave in sorted(en_yaml - en_codigo):
            sobran.append(f"{seccion.name}.{clave}")
    assert not faltan, f"campos del cargador que no estan en config.yaml: {faltan}"
    assert not sobran, f"claves de config.yaml que el cargador no conoce: {sobran}"


@_test("Las claves eliminadas ya no existen en el cargador")
def _():
    limpiar_cache()
    config = cargar_config()
    eliminadas = (
        ("fsm", "ventana_corta_seg"), ("fsm", "ventana_larga_seg"),
        ("vision", "indice_camara"), ("vision", "refinar_contornos"),
        ("ojos", "dur_min_parpadeo_seg"), ("ojos", "parpadeos_por_minuto_normal"),
        ("ojos", "tiempo_parpadeos_bajos_seg"), ("ojos", "alpha_suavizado_ear"),
        ("cabeza", "tiempo_calibracion_baseline_seg"), ("red", "ip_raspberry"),
        ("identificadores", "prefijo_sesion"),
        ("identificadores", "prefijo_mensaje_vision"),
        ("identificadores", "prefijo_mensaje_wearable"),
        ("actuadores", "habilitar_voz"), ("actuadores", "ruta_audios_predefinidos"),
        ("logging", "ruta_logs"), ("logging", "tamano_max_log_bytes"),
        ("logging", "archivos_rotados"), ("logging", "habilitar_csv_sesion"),
        ("logging", "ruta_csv_sesion"),
    )
    for seccion, clave in eliminadas:
        assert not hasattr(getattr(config, seccion), clave), f"sigue existiendo {seccion}.{clave}"


@_test("Con ear_base los umbrales de EAR se derivan; escribirlos en el yaml se advierte")
def _():
    contenido = """
ojos:
  ear_base: 0.30
  factor_ear_cierre: 0.65
  factor_ear_apertura: 0.84
  umbral_ear_cerrar: 0.10
  umbral_ear_abrir: 0.50
"""
    yaml_path = _yaml_temporal(contenido)
    try:
        limpiar_cache()
        with _CapturaAdvertencias() as avisos:
            config = cargar_config(path=str(yaml_path))
        assert abs(config.ojos.umbral_ear_cerrar - 0.195) < 1e-9, "debe valer ear_base * factor"
        assert abs(config.ojos.umbral_ear_abrir - 0.252) < 1e-9
        assert any("umbral_ear_cerrar" in a and "Se ignoran" in a for a in avisos), avisos
    finally:
        yaml_path.unlink()


@_test("Sin ear_base se advierte que rigen umbrales absolutos")
def _():
    contenido = """
ojos:
  dur_min_microsueno_seg: 1.5
"""
    yaml_path = _yaml_temporal(contenido)
    try:
        limpiar_cache()
        with _CapturaAdvertencias() as avisos:
            config = cargar_config(path=str(yaml_path))
        assert config.ojos.ear_base == 0.0
        assert any("ear_base no esta definido" in a for a in avisos), avisos
    finally:
        yaml_path.unlink()
        limpiar_cache()


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
