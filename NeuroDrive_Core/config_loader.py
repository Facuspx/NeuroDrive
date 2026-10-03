"""
NeuroDrive Core - Cargador de configuracion
============================================

Carga, valida y expone el archivo config/config.yaml al resto del sistema.

Uso tipico:
    from NeuroDrive_Core.config_loader import cargar_config

    config = cargar_config()
    timeout = config.wearable.timeout_ack_leve_seg
    pin_buzzer = config.actuadores.buzzer_gpio_pin

Ventajas vs leer el yaml crudo:
  - Autocompletado en IDE (sabes que campos existen)
  - Errores de tipeo detectados en tiempo de carga, no en runtime
  - Validacion de rangos en __post_init__ de cada dataclass
  - Defaults explicitos si una clave falta

Resolucion del archivo:
  1. Argumento explicito a cargar_config(path=...)
  2. Variable de entorno NEURODRIVE_CONFIG
  3. config/config.yaml relativo al cwd
  4. <ruta_de_este_modulo>/../config/config.yaml
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_log = logging.getLogger("NeuroDrive.ConfigLoader")


# =============================================================================
#                  EXCEPCIONES Y CACHE
# =============================================================================


class ConfigError(Exception):
    """Error fatal al cargar o validar la configuracion."""


# Cache del singleton. La primera llamada a cargar_config() lo llena.
# Las siguientes lo reutilizan a menos que se pase recargar=True.
_config_cache: Optional[Config] = None


# =============================================================================
#                  DATACLASSES POR SECCION
# =============================================================================


@dataclass
class ConfigFSMSeccion:
    """Seccion [fsm] del config.yaml."""
    tiempo_para_bajar_estado_seg: float = 60.0
    # Senales leves continuas
    calentamiento_senales_seg: float = 60.0
    persistencia_senales_leves_seg: float = 20.0
    # PERCLOS: tres decisiones distintas, tres umbrales (no unificar)
    perclos_corroborar_cabeceo: float = 0.30
    perclos_senales_leves: float = 0.30
    perclos_parpados_pesados: float = 0.35
    perclos_parpados_pesados_sostenido_seg: float = 30.0
    # Fatiga recurrente
    max_eventos_severos_ventana: int = 3
    ventana_episodios_seg: float = 900.0
    umbral_respuesta_lenta_ms: int = 5000

    def __post_init__(self) -> None:
        if self.tiempo_para_bajar_estado_seg <= 0:
            raise ConfigError(
                f"tiempo_para_bajar_estado_seg debe ser > 0: "
                f"{self.tiempo_para_bajar_estado_seg}"
            )
        for nombre in (
            "perclos_corroborar_cabeceo",
            "perclos_senales_leves",
            "perclos_parpados_pesados",
        ):
            val = getattr(self, nombre)
            # PERCLOS es una proporcion: un umbral fuera de (0, 1) no se
            # alcanzaria nunca, o se cumpliria siempre.
            if not (0.0 < val < 1.0):
                raise ConfigError(f"{nombre} debe estar en (0, 1): {val}")
        if self.perclos_parpados_pesados_sostenido_seg <= 0:
            raise ConfigError(
                f"perclos_parpados_pesados_sostenido_seg debe ser > 0: "
                f"{self.perclos_parpados_pesados_sostenido_seg}"
            )


@dataclass
class ConfigVisionSeccion:
    """Seccion [vision] del config.yaml."""
    fps_deseado: int = 15
    resolucion_ancho: int = 640
    resolucion_alto: int = 480
    # MediaPipe FaceMesh (los recibe DetectorRostro)
    confianza_minima_deteccion: float = 0.5
    confianza_minima_seguimiento: float = 0.5
    max_frames_sin_rostro: int = 15
    # Silencio maximo del programa de vision antes de declararlo fuera de
    # servicio. Publica ~15 mensajes por segundo, haya rostro o no.
    timeout_silencio_seg: float = 5.0

    def __post_init__(self) -> None:
        if not (1 <= self.fps_deseado <= 60):
            raise ConfigError(f"fps_deseado fuera de [1,60]: {self.fps_deseado}")
        if self.resolucion_ancho < 160 or self.resolucion_alto < 120:
            raise ConfigError(
                f"resolucion muy chica: {self.resolucion_ancho}x{self.resolucion_alto}"
            )
        if not (0.0 <= self.confianza_minima_deteccion <= 1.0):
            raise ConfigError(
                f"confianza_minima_deteccion fuera de [0,1]: "
                f"{self.confianza_minima_deteccion}"
            )
        if not (0.0 <= self.confianza_minima_seguimiento <= 1.0):
            raise ConfigError(
                f"confianza_minima_seguimiento fuera de [0,1]: "
                f"{self.confianza_minima_seguimiento}"
            )
        if self.max_frames_sin_rostro < 1:
            raise ConfigError(
                f"max_frames_sin_rostro debe ser >= 1: {self.max_frames_sin_rostro}"
            )
        if self.timeout_silencio_seg <= 0:
            raise ConfigError(
                f"timeout_silencio_seg debe ser > 0: {self.timeout_silencio_seg}"
            )


@dataclass
class ConfigOjosSeccion:
    """Seccion [ojos] del config.yaml."""
    # Relativo unificado: los umbrales de ojo cerrado/abierto se DERIVAN del
    # EAR del conductor con los ojos abiertos (ear_base, de la calibracion):
    #     umbral_ear_cerrar = ear_base * factor_ear_cierre
    #     umbral_ear_abrir  = ear_base * factor_ear_apertura
    # Asi la vision (pantalla) y el Core (FSM) usan un unico criterio.
    ear_base: float = 0.0
    factor_ear_cierre: float = 0.68
    factor_ear_apertura: float = 0.84
    # CAMPOS DERIVADOS: no se escriben en config.yaml. Con ear_base > 0 se
    # calculan en __post_init__ y cualquier valor dado se pisa. Los de aca
    # abajo solo rigen sin calibracion (ear_base = 0), que cargar_config()
    # informa con una advertencia; tambien los usan las pruebas.
    umbral_ear_cerrar: float = 0.18
    umbral_ear_abrir: float = 0.22
    dur_max_parpadeo_seg: float = 0.40
    dur_min_microsueno_seg: float = 1.5
    refractario_parpadeo_seg: float = 0.25
    parpadeos_por_minuto_alerta: int = 10

    def __post_init__(self) -> None:
        # Relativo unificado: si hay ear_base calibrado, derivar los umbrales.
        if self.ear_base > 0.0:
            if not (0.0 < self.factor_ear_cierre < self.factor_ear_apertura <= 1.0):
                raise ConfigError(
                    f"factores EAR invalidos: cierre={self.factor_ear_cierre}, "
                    f"apertura={self.factor_ear_apertura} "
                    f"(debe cumplirse 0 < cierre < apertura <= 1)"
                )
            if not (0.1 < self.ear_base < 1.0):
                raise ConfigError(
                    f"ear_base fuera de rango razonable: {self.ear_base}"
                )
            self.umbral_ear_cerrar = self.ear_base * self.factor_ear_cierre
            self.umbral_ear_abrir = self.ear_base * self.factor_ear_apertura

        if self.umbral_ear_cerrar >= self.umbral_ear_abrir:
            raise ConfigError(
                f"umbral_ear_cerrar ({self.umbral_ear_cerrar}) debe ser menor que "
                f"umbral_ear_abrir ({self.umbral_ear_abrir}) para tener histeresis"
            )
        if self.dur_max_parpadeo_seg <= 0:
            raise ConfigError(
                f"dur_max_parpadeo_seg debe ser > 0: {self.dur_max_parpadeo_seg}"
            )
        if self.dur_min_microsueno_seg <= self.dur_max_parpadeo_seg:
            raise ConfigError(
                f"dur_min_microsueno_seg ({self.dur_min_microsueno_seg}) debe ser "
                f"mayor que dur_max_parpadeo_seg ({self.dur_max_parpadeo_seg})"
            )
        if self.parpadeos_por_minuto_alerta <= 0:
            raise ConfigError(
                f"parpadeos_por_minuto_alerta debe ser > 0: "
                f"{self.parpadeos_por_minuto_alerta}"
            )


@dataclass
class ConfigBocaSeccion:
    """Seccion [boca] del config.yaml."""
    umbral_mar_bostezo: float = 0.6
    # Histeresis: el bostezo termina cuando el MAR baja de
    # umbral_mar_bostezo * factor_mar_cierre
    factor_mar_cierre: float = 0.9
    dur_min_bostezo_seg: float = 1.0
    ventana_bostezos_seg: float = 900.0
    max_bostezos_ventana_larga: int = 3

    def __post_init__(self) -> None:
        if self.umbral_mar_bostezo <= 0:
            raise ConfigError(f"umbral_mar_bostezo debe ser > 0: {self.umbral_mar_bostezo}")
        if not (0.0 < self.factor_mar_cierre < 1.0):
            raise ConfigError(
                f"factor_mar_cierre debe estar en (0, 1): {self.factor_mar_cierre}"
            )
        if self.ventana_bostezos_seg <= 0:
            raise ConfigError(
                f"ventana_bostezos_seg debe ser > 0: {self.ventana_bostezos_seg}"
            )
        if self.dur_min_bostezo_seg <= 0:
            raise ConfigError(
                f"dur_min_bostezo_seg debe ser > 0: {self.dur_min_bostezo_seg}"
            )
        if self.max_bostezos_ventana_larga < 1:
            raise ConfigError(
                f"max_bostezos_ventana_larga debe ser >= 1: "
                f"{self.max_bostezos_ventana_larga}"
            )


@dataclass
class ConfigCabezaSeccion:
    """Seccion [cabeza] del config.yaml."""
    umbral_pitch_grados: float = 20.0
    # Histeresis: el cabeceo termina cuando el pitch baja de
    # umbral_pitch_grados * factor_pitch_fin_cabeceo
    factor_pitch_fin_cabeceo: float = 0.85
    dur_min_cabeceo_seg: float = 0.8
    umbral_yaw_max_grados: float = 35.0
    # Postura neutra del conductor, medida en la calibracion. La vision resta
    # estos valores antes de transmitir, de modo que el nucleo razona sobre
    # desviaciones respecto de la postura habitual. Se copian del resultado de
    # la calibracion, igual que ojos.ear_base. 0.0 = sin normalizar.
    pitch_neutro_grados: float = 0.0
    yaw_neutro_grados: float = 0.0
    roll_neutro_grados: float = 0.0

    def __post_init__(self) -> None:
        for nombre in ("pitch_neutro_grados", "yaw_neutro_grados", "roll_neutro_grados"):
            val = getattr(self, nombre)
            # Un neutro mayor a 60 grados indica una calibracion fallida o un
            # valor mal copiado: con esa postura el rostro ni se detecta.
            if not (-60.0 <= val <= 60.0):
                raise ConfigError(f"{nombre} fuera de [-60, 60]: {val}")
        if self.umbral_pitch_grados <= 0:
            raise ConfigError(
                f"umbral_pitch_grados debe ser > 0: {self.umbral_pitch_grados}"
            )
        if not (0.0 < self.factor_pitch_fin_cabeceo < 1.0):
            raise ConfigError(
                f"factor_pitch_fin_cabeceo debe estar en (0, 1): "
                f"{self.factor_pitch_fin_cabeceo}"
            )
        if self.dur_min_cabeceo_seg <= 0:
            raise ConfigError(
                f"dur_min_cabeceo_seg debe ser > 0: {self.dur_min_cabeceo_seg}"
            )
        if self.umbral_yaw_max_grados <= 0:
            raise ConfigError(
                f"umbral_yaw_max_grados debe ser > 0: {self.umbral_yaw_max_grados}"
            )


@dataclass
class ConfigWearableSeccion:
    """Seccion [wearable] del config.yaml."""
    bpm_normal_min: int = 60
    bpm_normal_max: int = 100
    bpm_umbral_alerta: int = 70
    bpm_umbral_critico: int = 60
    timeout_ack_leve_seg: float = 30.0
    timeout_ack_medio_seg: float = 20.0
    timeout_ack_critico_seg: float = 15.0
    # Lo que el nucleo espera por encima de la ventana de respuesta (ver yaml)
    margen_presentacion_desafio_seg: float = 2.5
    timeout_heartbeat_seg: float = 10.0
    intervalo_envio_bpm_seg: float = 2.0

    def __post_init__(self) -> None:
        if self.bpm_normal_min >= self.bpm_normal_max:
            raise ConfigError(
                f"bpm_normal_min ({self.bpm_normal_min}) debe ser menor que "
                f"bpm_normal_max ({self.bpm_normal_max})"
            )
        if self.bpm_umbral_critico >= self.bpm_umbral_alerta:
            raise ConfigError(
                f"bpm_umbral_critico ({self.bpm_umbral_critico}) debe ser menor que "
                f"bpm_umbral_alerta ({self.bpm_umbral_alerta})"
            )
        for nombre, val in (
            ("timeout_ack_leve_seg", self.timeout_ack_leve_seg),
            ("timeout_ack_medio_seg", self.timeout_ack_medio_seg),
            ("timeout_ack_critico_seg", self.timeout_ack_critico_seg),
            ("timeout_heartbeat_seg", self.timeout_heartbeat_seg),
            ("intervalo_envio_bpm_seg", self.intervalo_envio_bpm_seg),
        ):
            if val <= 0:
                raise ConfigError(f"{nombre} debe ser > 0: {val}")
        if self.margen_presentacion_desafio_seg < 0:
            raise ConfigError(
                f"margen_presentacion_desafio_seg no puede ser negativo: "
                f"{self.margen_presentacion_desafio_seg}"
            )
        # Heartbeat debe ser mayor que intervalo de envio para no dar falsos positivos
        if self.timeout_heartbeat_seg <= self.intervalo_envio_bpm_seg * 2:
            raise ConfigError(
                f"timeout_heartbeat_seg ({self.timeout_heartbeat_seg}) debe ser al "
                f"menos 2x intervalo_envio_bpm_seg ({self.intervalo_envio_bpm_seg})"
            )


@dataclass
class ConfigIPCSeccion:
    """Seccion [ipc] del config.yaml."""
    modo: str = "posix_mq"
    cola_vision: str = "/neurodrive_vision"
    cola_wearable: str = "/neurodrive_wearable"
    capacidad_cola: int = 16
    tamano_max_mensaje_bytes: int = 1024

    def __post_init__(self) -> None:
        if self.modo not in ("posix_mq", "multiprocessing"):
            raise ConfigError(
                f"modo IPC desconocido: {self.modo} "
                f"(esperado 'posix_mq' o 'multiprocessing')"
            )
        # POSIX MQ requiere nombres que empiecen con /
        if self.modo == "posix_mq":
            if not self.cola_vision.startswith("/"):
                raise ConfigError(
                    f"cola_vision debe empezar con /: {self.cola_vision}"
                )
            if not self.cola_wearable.startswith("/"):
                raise ConfigError(
                    f"cola_wearable debe empezar con /: {self.cola_wearable}"
                )
        if self.capacidad_cola < 1:
            raise ConfigError(f"capacidad_cola debe ser >= 1: {self.capacidad_cola}")
        if self.tamano_max_mensaje_bytes < 128:
            raise ConfigError(
                f"tamano_max_mensaje_bytes muy chico: {self.tamano_max_mensaje_bytes}"
            )


@dataclass
class ConfigRedSeccion:
    """Seccion [red] del config.yaml."""
    ip_wearable: str = "192.168.4.20"
    puerto_udp_escucha: int = 5005
    puerto_udp_envio: int = 5006
    reenvios_comandos_criticos: int = 3
    espaciado_reenvios_ms: int = 50

    def __post_init__(self) -> None:
        if not (1 <= self.puerto_udp_escucha <= 65535):
            raise ConfigError(
                f"puerto_udp_escucha fuera de rango: {self.puerto_udp_escucha}"
            )
        if not (1 <= self.puerto_udp_envio <= 65535):
            raise ConfigError(
                f"puerto_udp_envio fuera de rango: {self.puerto_udp_envio}"
            )
        if self.reenvios_comandos_criticos < 1:
            raise ConfigError(
                f"reenvios_comandos_criticos debe ser >= 1: "
                f"{self.reenvios_comandos_criticos}"
            )
        if self.espaciado_reenvios_ms < 0:
            raise ConfigError(
                f"espaciado_reenvios_ms no puede ser negativo: "
                f"{self.espaciado_reenvios_ms}"
            )


@dataclass
class ConfigIdentificadoresSeccion:
    """Seccion [identificadores] del config.yaml."""
    id_camara: str = "cam-01"
    id_wearable: str = "wearable-01"
    id_core: str = "core"
    prefijo_mensaje_interno: str = "int"

    def __post_init__(self) -> None:
        for nombre, val in (
            ("id_camara", self.id_camara),
            ("id_wearable", self.id_wearable),
            ("id_core", self.id_core),
            ("prefijo_mensaje_interno", self.prefijo_mensaje_interno),
        ):
            if not val or not isinstance(val, str):
                raise ConfigError(f"{nombre} debe ser un string no vacio")


@dataclass
class ConfigActuadoresSeccion:
    """Seccion [actuadores] del config.yaml."""
    buzzer_gpio_pin: int = 18

    def __post_init__(self) -> None:
        if not (0 <= self.buzzer_gpio_pin <= 40):
            raise ConfigError(
                f"buzzer_gpio_pin fuera de rango BCM [0,40]: {self.buzzer_gpio_pin}"
            )


@dataclass
class ConfigLoggingSeccion:
    """Seccion [logging] del config.yaml."""
    nivel: str = "INFO"

    def __post_init__(self) -> None:
        niveles_validos = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if self.nivel.upper() not in niveles_validos:
            raise ConfigError(
                f"nivel de log invalido: {self.nivel} "
                f"(validos: {niveles_validos})"
            )
        self.nivel = self.nivel.upper()


@dataclass
class ConfigSesionSeccion:
    """Seccion [sesion] del config.yaml."""
    archivo_estado: str = "NeuroDrive_Core/estado_sesion.json"
    ttl_sesion_seg: float = 900.0

    def __post_init__(self) -> None:
        if self.ttl_sesion_seg <= 0:
            raise ConfigError(f"ttl_sesion_seg debe ser > 0: {self.ttl_sesion_seg}")


# =============================================================================
#                       CONFIG RAIZ (agrupa todas las secciones)
# =============================================================================


@dataclass
class Config:
    """
    Configuracion completa del sistema NeuroDrive.

    Se accede por dot notation:
        config.fsm.tiempo_para_bajar_estado_seg
        config.actuadores.buzzer_gpio_pin
        config.ipc.modo
    """
    fsm: ConfigFSMSeccion = field(default_factory=ConfigFSMSeccion)
    vision: ConfigVisionSeccion = field(default_factory=ConfigVisionSeccion)
    ojos: ConfigOjosSeccion = field(default_factory=ConfigOjosSeccion)
    boca: ConfigBocaSeccion = field(default_factory=ConfigBocaSeccion)
    cabeza: ConfigCabezaSeccion = field(default_factory=ConfigCabezaSeccion)
    wearable: ConfigWearableSeccion = field(default_factory=ConfigWearableSeccion)
    ipc: ConfigIPCSeccion = field(default_factory=ConfigIPCSeccion)
    red: ConfigRedSeccion = field(default_factory=ConfigRedSeccion)
    identificadores: ConfigIdentificadoresSeccion = field(
        default_factory=ConfigIdentificadoresSeccion
    )
    actuadores: ConfigActuadoresSeccion = field(default_factory=ConfigActuadoresSeccion)
    logging: ConfigLoggingSeccion = field(default_factory=ConfigLoggingSeccion)
    sesion: ConfigSesionSeccion = field(default_factory=ConfigSesionSeccion)

    # Ruta del archivo de origen (para debug)
    ruta_origen: str = ""


# =============================================================================
#                      LOGICA DE CARGA Y RESOLUCION DE PATH
# =============================================================================


def _resolver_ruta_config(path_explicito: Optional[str] = None) -> Path:
    """
    Resuelve la ruta del archivo de configuracion.

    Orden de busqueda:
      1. path_explicito (si se paso como argumento)
      2. Variable de entorno NEURODRIVE_CONFIG
      3. config/config.yaml relativo al cwd
      4. <directorio_de_este_modulo>/../config/config.yaml
    """
    # 1. Argumento explicito
    if path_explicito:
        p = Path(path_explicito)
        if not p.is_file():
            raise ConfigError(
                f"Archivo de config explicito no existe: {path_explicito}"
            )
        return p

    # 2. Variable de entorno
    env = os.environ.get("NEURODRIVE_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        _log.warning(
            "NEURODRIVE_CONFIG apunta a un archivo inexistente: %s "
            "(buscando alternativas)",
            env,
        )

    # 3. Relativo al cwd
    p = Path("config/config.yaml")
    if p.is_file():
        return p.resolve()

    # 4. Relativo a este modulo
    aqui = Path(__file__).resolve().parent
    p = aqui.parent / "config" / "config.yaml"
    if p.is_file():
        return p

    raise ConfigError(
        "No se encontro config.yaml. Lugares buscados:\n"
        "  - config/config.yaml (relativo al cwd)\n"
        f"  - {aqui.parent / 'config' / 'config.yaml'}\n"
        "Usa la variable NEURODRIVE_CONFIG o pasa path explicito."
    )


def _construir_seccion(
    cls: type,
    seccion_dict: Dict[str, Any],
    nombre_seccion: str,
) -> Any:
    """
    Construye un dataclass de seccion desde un dict, ignorando claves desconocidas.

    Si el dict tiene una clave que el dataclass no espera, la ignoramos
    silenciosamente (despues de loggear warning). Esto permite que el YAML
    tenga claves "extra" sin romper la carga.

    Si el dataclass espera una clave que el dict no tiene, usa el default.
    """
    # Obtener campos validos del dataclass
    campos_validos = set(cls.__dataclass_fields__.keys())
    campos_dict = set(seccion_dict.keys())

    # Loggear claves desconocidas
    desconocidas = campos_dict - campos_validos
    for clave in desconocidas:
        _log.warning(
            "Clave desconocida en seccion [%s]: '%s' (sera ignorada)",
            nombre_seccion,
            clave,
        )

    # Construir solo con las claves validas
    kwargs = {k: v for k, v in seccion_dict.items() if k in campos_validos}

    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"Error construyendo seccion [{nombre_seccion}]: {e}"
        ) from e


def cargar_config(
    path: Optional[str] = None,
    recargar: bool = False,
) -> Config:
    """
    Carga el config.yaml y devuelve un Config validado.

    Args:
        path: Ruta explicita al archivo (opcional). Si no se da, usa la
              estrategia de resolucion automatica.
        recargar: Si True, ignora el cache y recarga desde disco. Util
                  para tests o para recargar config en runtime.

    Returns:
        Config: configuracion validada y lista para usar.

    Raises:
        ConfigError: si el archivo no existe, no es YAML valido, o tiene
                     valores invalidos.
    """
    global _config_cache

    # Si esta cacheado y no se pidio recargar, devolver el cache
    if _config_cache is not None and not recargar and path is None:
        return _config_cache

    # Resolver ruta
    ruta = _resolver_ruta_config(path)
    _log.info("Cargando config desde: %s", ruta)

    # Leer y parsear YAML
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            datos = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML invalido en {ruta}: {e}") from e
    except OSError as e:
        raise ConfigError(f"No se pudo leer {ruta}: {e}") from e

    if not isinstance(datos, dict):
        raise ConfigError(
            f"El YAML debe ser un dict en su raiz, se obtuvo: {type(datos).__name__}"
        )

    # Construir cada seccion (las que falten usan defaults)
    mapeo_secciones = (
        ("fsm", ConfigFSMSeccion),
        ("vision", ConfigVisionSeccion),
        ("ojos", ConfigOjosSeccion),
        ("boca", ConfigBocaSeccion),
        ("cabeza", ConfigCabezaSeccion),
        ("wearable", ConfigWearableSeccion),
        ("ipc", ConfigIPCSeccion),
        ("red", ConfigRedSeccion),
        ("identificadores", ConfigIdentificadoresSeccion),
        ("actuadores", ConfigActuadoresSeccion),
        ("logging", ConfigLoggingSeccion),
        ("sesion", ConfigSesionSeccion),
    )

    secciones_construidas: Dict[str, Any] = {}
    for nombre, cls in mapeo_secciones:
        seccion = datos.get(nombre)
        if seccion is None:
            _log.warning(
                "Seccion [%s] no encontrada en config.yaml, usando defaults",
                nombre,
            )
            secciones_construidas[nombre] = cls()
        elif not isinstance(seccion, dict):
            raise ConfigError(
                f"Seccion [{nombre}] debe ser un dict, se obtuvo: "
                f"{type(seccion).__name__}"
            )
        else:
            secciones_construidas[nombre] = _construir_seccion(cls, seccion, nombre)

    # Umbrales de ojos: avisar las dos situaciones que cambian el criterio
    # sin que se note.
    ojos_yaml = datos.get("ojos") if isinstance(datos.get("ojos"), dict) else {}
    ojos_cfg = secciones_construidas["ojos"]
    if ojos_cfg.ear_base <= 0.0:
        _log.warning(
            "ojos.ear_base no esta definido: se usan umbrales de EAR absolutos "
            "(cerrar=%.3f, abrir=%.3f), no los del conductor. Calibrar y copiar "
            "ear_base a config.yaml.",
            ojos_cfg.umbral_ear_cerrar, ojos_cfg.umbral_ear_abrir,
        )
    else:
        escritos = [k for k in ("umbral_ear_cerrar", "umbral_ear_abrir") if k in ojos_yaml]
        if escritos:
            _log.warning(
                "Se ignoran %s de config.yaml: con ear_base definido los "
                "umbrales se derivan (cerrar=%.3f, abrir=%.3f). Quitarlos del yaml.",
                ", ".join(f"ojos.{k}" for k in escritos),
                ojos_cfg.umbral_ear_cerrar, ojos_cfg.umbral_ear_abrir,
            )

    # Detectar secciones extra en el YAML que no conocemos
    secciones_esperadas = {nombre for nombre, _ in mapeo_secciones}
    secciones_en_yaml = set(datos.keys())
    extras = secciones_en_yaml - secciones_esperadas
    for extra in extras:
        _log.warning(
            "Seccion desconocida en config.yaml: [%s] (sera ignorada)",
            extra,
        )

    # Construir el Config raiz
    config = Config(
        ruta_origen=str(ruta),
        **secciones_construidas,
    )

    _config_cache = config
    return config


def limpiar_cache() -> None:
    """Limpia el cache del config (util para tests)."""
    global _config_cache
    _config_cache = None


__all__ = [
    "Config",
    "ConfigError",
    "ConfigFSMSeccion",
    "ConfigVisionSeccion",
    "ConfigOjosSeccion",
    "ConfigBocaSeccion",
    "ConfigCabezaSeccion",
    "ConfigWearableSeccion",
    "ConfigIPCSeccion",
    "ConfigRedSeccion",
    "ConfigIdentificadoresSeccion",
    "ConfigActuadoresSeccion",
    "ConfigLoggingSeccion",
    "ConfigSesionSeccion",
    "cargar_config",
    "limpiar_cache",
]
