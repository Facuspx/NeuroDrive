"""
NeuroDrive Vision - Publicador de eventos por POSIX Message Queue
=================================================================

Toma los resultados de los analizadores de vision (rostro, ojos, boca,
cabeza, frote) y los publica al NeuroDrive Core como mensajes EventoVision
envueltos en un Envelope, a traves de una POSIX Message Queue.

Decisiones tomadas (ver chat de planificacion):
  - Envia datos INSTANTANEOS por frame (EAR, MAR, angulos, flag de frote).
    El PERCLOS, conteo de parpadeos/bostezos, etc. los calcula el Pre-FSM
    del Core sobre ventanas temporales. La vision NO los envia.
  - confianza_deteccion se deriva de cuantos analizadores dieron datos
    validos en el frame.
  - Importa los contratos del Core (common.contratos). NO los duplica.
  - posix_ipc es opcional: si no esta instalado, funciona en modo simulado
    (loguea lo que enviaria, no envia). Permite testear sin el Core.
  - mq_send NO bloqueante: si la cola esta llena, se descarta el mensaje
    nuevo y se sigue. Un frame de vision viejo no sirve; mejor perderlo
    que congelar el pipeline de captura.

API:
    pub = PublicadorMQ(config)
    pub.iniciar()
    enviado = pub.publicar(datos_rostro, datos_ojos, datos_boca,
                           datos_cabeza, datos_frote)
    pub.detener()

Tambien soporta context manager.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from NeuroDrive_Vision.detector_rostro import DatosRostro
from NeuroDrive_Vision.analizador_ojos import DatosOjos
from NeuroDrive_Vision.analizador_boca import DatosBoca
from NeuroDrive_Vision.analizador_cabeza import DatosCabeza
from NeuroDrive_Vision.detector_frote_ojos import DatosFroteOjos

# Config solo se usa para anotaciones de tipo. El bloque TYPE_CHECKING
# es leido por el analizador de tipos (Pylance) pero NUNCA se ejecuta en
# runtime, asi que no necesitamos el try/except: si el modulo no existe,
# no pasa nada porque en ejecucion esta linea no corre.
if TYPE_CHECKING:
    from NeuroDrive_Core.config_loader import Config

# ---------------------------------------------------------------------------
# Importacion de los contratos del Core.
#
# El Core define EventoVision, Envelope, TipoMensaje, OrigenEvento en
# common/contratos.py. El publicador los importa: NO los duplica, para
# evitar que vision y core se desincronicen.
#
# Si el paquete common no esta disponible (ej. testeando vision aislado),
# CONTRATOS_DISPONIBLES queda en False y el publicador funciona en modo
# simulado.
# ---------------------------------------------------------------------------
try:
    from common.contratos import (
        EventoVision,
        Envelope,
        TipoMensaje,
        OrigenEvento,
        generar_id_sesion,
        generar_id_mensaje,
    )
    CONTRATOS_DISPONIBLES = True
except ImportError:
    try:
        from common import (  # type: ignore
            EventoVision,
            Envelope,
            TipoMensaje,
            OrigenEvento,
            generar_id_sesion,
            generar_id_mensaje,
        )
        CONTRATOS_DISPONIBLES = True
    except ImportError:
        CONTRATOS_DISPONIBLES = False
        EventoVision = None       # type: ignore
        Envelope = None           # type: ignore
        TipoMensaje = None        # type: ignore
        OrigenEvento = None       # type: ignore
        generar_id_sesion = None  # type: ignore
        generar_id_mensaje = None # type: ignore

# ---------------------------------------------------------------------------
# Importacion de posix_ipc (la libreria de POSIX MQ).
# Opcional: si no esta, el publicador funciona en modo simulado.
# ---------------------------------------------------------------------------
try:
    import posix_ipc
    POSIX_IPC_DISPONIBLE = True
except ImportError:
    posix_ipc = None  # type: ignore
    POSIX_IPC_DISPONIBLE = False


_log = logging.getLogger("NeuroDrive.PublicadorMQ")


# =============================================================================
# Excepciones
# =============================================================================

class ErrorPublicadorMQ(Exception):
    """Error tecnico en el publicador."""


class ErrorLimitesSistema(ErrorPublicadorMQ):
    """Los limites del kernel no permiten la configuracion de cola pedida."""


# =============================================================================
# Chequeo de limites del kernel para POSIX MQ
# =============================================================================

def _leer_limite_kernel(path: str) -> Optional[int]:
    """Lee un limite entero desde /proc/sys/fs/mqueue/. None si no se puede."""
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def chequear_limites_sistema(capacidad: int, tamano_max_mensaje: int) -> None:
    """
    Verifica que los limites del kernel permitan crear una cola con la
    capacidad y tamano pedidos.

    Las POSIX MQ tienen dos limites en /proc/sys/fs/mqueue/:
      - msg_max: cantidad maxima de mensajes por cola
      - msgsize_max: tamano maximo de cada mensaje

    Si la configuracion los excede, posix_ipc falla con un error criptico
    ("Invalid parameter(s)"). Esta funcion detecta el problema antes y
    lanza un error claro con la instruccion para resolverlo.

    En sistemas que no son Linux, el chequeo se omite silenciosamente.

    Raises:
        ErrorLimitesSistema: si los limites del kernel son insuficientes.
    """
    if not Path("/proc/sys/fs/mqueue").is_dir():
        # No es Linux o no hay soporte de mqueue: no podemos chequear
        return

    msg_max = _leer_limite_kernel("/proc/sys/fs/mqueue/msg_max")
    msgsize_max = _leer_limite_kernel("/proc/sys/fs/mqueue/msgsize_max")

    problemas = []

    if msg_max is not None and capacidad > msg_max:
        problemas.append(
            f"  - capacidad de cola pedida ({capacidad}) > limite del kernel "
            f"msg_max ({msg_max}).\n"
            f"    Opcion A (recomendada): bajar 'capacidad_cola' a {msg_max} "
            f"o menos en config.yaml, seccion [ipc].\n"
            f"    Opcion B: subir el limite del kernel (requiere root):\n"
            f"      sudo sysctl -w fs.mqueue.msg_max={max(capacidad, 32)}"
        )

    if msgsize_max is not None and tamano_max_mensaje > msgsize_max:
        problemas.append(
            f"  - tamano de mensaje pedido ({tamano_max_mensaje}) > limite "
            f"del kernel msgsize_max ({msgsize_max}).\n"
            f"      sudo sysctl -w fs.mqueue.msgsize_max={max(tamano_max_mensaje, 8192)}"
        )

    if problemas:
        raise ErrorLimitesSistema(
            "Los limites del kernel no permiten crear la cola POSIX MQ:\n"
            + "\n".join(problemas)
        )


# =============================================================================
# Clase principal
# =============================================================================

class PublicadorMQ:
    """
    Publicador de eventos de vision hacia el Core por POSIX MQ.

    No es thread-safe. Una instancia por hilo.

    Modos de operacion:
      - REAL: posix_ipc y los contratos estan disponibles. Envia de verdad.
      - SIMULADO: falta posix_ipc o los contratos. Loguea lo que enviaria
        pero no envia. Util para testear vision sin el Core corriendo.

    Lifecycle:
      - __init__: configura parametros, no abre la cola.
      - iniciar(): abre/crea la cola POSIX MQ.
      - publicar(...): arma y envia un EventoVision.
      - detener(): cierra la cola.
    """

    # Nombre de la cola POSIX MQ (fallback si no hay config)
    NOMBRE_COLA_DEFAULT = "/neurodrive_vision"

    # Tamanio maximo de mensaje en bytes (fallback si no hay config)
    TAMANIO_MAX_DEFAULT = 1024

    # Capacidad maxima de la cola (fallback si no hay config)
    CAPACIDAD_COLA_DEFAULT = 16

    # ID de este dispositivo de vision (fallback si no hay config)
    ID_DISPOSITIVO_DEFAULT = "cam-01"

    def __init__(
        self,
        config: Optional["Config"] = None,
        nombre_cola: Optional[str] = None,
        id_dispositivo: Optional[str] = None,
        capacidad_cola: Optional[int] = None,
        tamano_max_mensaje: Optional[int] = None,
        forzar_simulado: bool = False,
        drenar_al_iniciar: bool = True,
        eliminar_al_detener: bool = True,
        pitch_neutro: float = 0.0,
        yaw_neutro: float = 0.0,
        roll_neutro: float = 0.0,
    ) -> None:
        """
        Parametros
        ----------
        config : Config | None
            Configuracion global. Si se pasa, el publicador lee de la
            seccion 'ipc' el nombre de cola, la capacidad y el tamano
            maximo de mensaje. Asi vision y Core usan la MISMA fuente de
            verdad (config.yaml) y no se desincronizan.
        nombre_cola : str | None
            Nombre de la POSIX MQ. Si se pasa, tiene prioridad sobre el
            config. Si no, se usa config.ipc.cola_vision, y si no hay
            config, "/neurodrive_vision".
        id_dispositivo : str | None
            Identificador de esta camara. Prioridad: parametro > config >
            "cam-01".
        capacidad_cola : int | None
            Capacidad de la cola. Prioridad: parametro > config.ipc.
            capacidad_cola > 16.
        tamano_max_mensaje : int | None
            Tamano maximo de mensaje en bytes. Prioridad: parametro >
            config.ipc.tamano_max_mensaje_bytes > 1024.
        forzar_simulado : bool
            Si True, opera en modo simulado aunque posix_ipc este disponible.
        drenar_al_iniciar : bool
            Si True (default), al iniciar vacia los mensajes viejos de la
            cola. INTEGRADO CON EL CORE: poner en False. El Core es el
            duenio de la cola; la vision no debe drenarla (borraria eventos
            que el Core todavia no consumio).
        eliminar_al_detener : bool
            Si True (default), al detener elimina la cola del kernel.
            INTEGRADO CON EL CORE: poner en False. El Core gestiona el
            ciclo de vida de la cola.
        pitch_neutro : float
            Angulo de pitch (en grados) de la pose neutra del conductor,
            obtenido de la calibracion. La vision se lo RESTA al pitch
            crudo antes de enviar el EventoVision, para que el umbral de
            cabeceo absoluto del Core (config.cabeza.umbral_pitch_grados)
            se interprete relativo a la pose neutra real.
            Default 0.0 = sin normalizacion (se envia el pitch crudo).
            El integrador (test_vision / main) lo setea tras calibrar.
        yaw_neutro : float
            Igual que pitch_neutro pero para el yaw. IMPORTANTE: el Core
            usa umbral_yaw_max_grados=35 absoluto para invalidar cabeceos
            cuando la cabeza esta muy girada. Sin normalizar, un conductor
            con yaw_neutro=-35 (camara al costado) veria TODOS sus
            cabeceos invalidados. Por eso el yaw se normaliza igual que
            el pitch. Default 0.0 = sin normalizacion.
        roll_neutro : float
            Igual, para el roll. El Core no lo usa hoy en filtros, pero
            lo enviamos normalizado por consistencia. Default 0.0.
        """
        self.config = config

        # ---- Resolucion de parametros: explicito > config > default ----
        # Nombre de cola
        if nombre_cola is not None:
            self.nombre_cola = nombre_cola
        elif config is not None:
            self.nombre_cola = config.ipc.cola_vision
        else:
            self.nombre_cola = self.NOMBRE_COLA_DEFAULT

        # ID de dispositivo
        if id_dispositivo is not None:
            self.id_dispositivo = id_dispositivo
        elif config is not None:
            self.id_dispositivo = config.identificadores.id_camara
        else:
            self.id_dispositivo = self.ID_DISPOSITIVO_DEFAULT

        # Capacidad de cola
        if capacidad_cola is not None:
            cap = int(capacidad_cola)
        elif config is not None:
            cap = int(config.ipc.capacidad_cola)
        else:
            cap = self.CAPACIDAD_COLA_DEFAULT
        if cap < 1:
            raise ValueError(f"capacidad_cola debe ser >= 1, recibido {cap}")
        self.capacidad_cola = cap

        # Tamano maximo de mensaje
        if tamano_max_mensaje is not None:
            tam = int(tamano_max_mensaje)
        elif config is not None:
            tam = int(config.ipc.tamano_max_mensaje_bytes)
        else:
            tam = self.TAMANIO_MAX_DEFAULT
        if tam < 128:
            raise ValueError(f"tamano_max_mensaje muy chico: {tam}")
        self.tamano_max_mensaje = tam

        self.drenar_al_iniciar = bool(drenar_al_iniciar)
        self.eliminar_al_detener = bool(eliminar_al_detener)

        # Normalizacion del pitch (Opcion A de integracion con el Core).
        # El Core usa un umbral de cabeceo ABSOLUTO (config.cabeza.
        # umbral_pitch_grados = 20). Pero la pose neutra del conductor NO
        # es 0 grados: depende de como este montada la camara (la
        # calibracion mide un pitch_neutro tipicamente negativo).
        # Para que el umbral absoluto del Core funcione bien, la vision
        # NORMALIZA el pitch: le resta el pitch_neutro antes de enviarlo.
        # Asi el Core recibe pitch "compensado por montaje", y su umbral
        # de 20 grados pasa a significar 20 grados DESDE el neutro real.
        #
        # IMPORTANTE (para la documentacion del TFI): con esto, el campo
        # pitch_grados del EventoVision NO es el angulo de Euler crudo,
        # sino el angulo normalizado respecto a la pose neutra calibrada.
        # Es una decision de diseño deliberada, no un descuido.
        #
        # pitch_neutro = 0.0 (default) => sin normalizacion (pitch crudo).
        # yaw_neutro y roll_neutro funcionan igual: se restan al valor
        # crudo antes de enviar el EventoVision al Core.
        # El integrador setea los tres tras la calibracion.
        self.pitch_neutro = float(pitch_neutro)
        self.yaw_neutro = float(yaw_neutro)
        self.roll_neutro = float(roll_neutro)

        # Decidir el modo de operacion
        self.modo_simulado = (
            forzar_simulado
            or not POSIX_IPC_DISPONIBLE
            or not CONTRATOS_DISPONIBLES
        )

        # Cola POSIX MQ (None hasta iniciar())
        self._cola = None
        self._activo: bool = False

        # ID de sesion: se genera al iniciar()
        self._id_sesion: Optional[str] = None

        # Contadores
        self._numero_secuencia: int = 0
        self._mensajes_enviados: int = 0
        self._mensajes_descartados: int = 0
        self._errores: int = 0
        self._mensajes_drenados: int = 0  # cuantos viejos vaciamos al iniciar

    # ------------------------------------------------------------------
    # Propiedades de estado
    # ------------------------------------------------------------------

    @property
    def activo(self) -> bool:
        return self._activo

    @property
    def mensajes_enviados(self) -> int:
        return self._mensajes_enviados

    @property
    def mensajes_descartados(self) -> int:
        return self._mensajes_descartados

    @property
    def errores(self) -> int:
        return self._errores

    @property
    def id_sesion(self) -> Optional[str]:
        return self._id_sesion

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        """
        Abre la cola POSIX MQ, recreandola limpia.

        Si esta en modo simulado, solo genera el id de sesion y marca activo.

        IMPORTANTE sobre la cola: las POSIX MQ persisten en el kernel entre
        ejecuciones. Si una corrida anterior dejo la cola llena (porque no
        habia consumidor), la proxima arrancaria con la cola "envenenada".
        Para evitarlo, al iniciar eliminamos cualquier cola preexistente con
        este nombre y la creamos de cero. Asi siempre arrancamos limpios y
        con la capacidad correcta.
        """
        if self._activo:
            _log.warning("PublicadorMQ ya estaba iniciado")
            return

        # Generar id de sesion
        if CONTRATOS_DISPONIBLES and generar_id_sesion is not None:
            self._id_sesion = generar_id_sesion()
        else:
            self._id_sesion = f"ses-{int(time.time())}"

        self._numero_secuencia = 0
        self._mensajes_enviados = 0
        self._mensajes_descartados = 0
        self._errores = 0
        self._mensajes_drenados = 0

        if self.modo_simulado:
            motivo = []
            if not POSIX_IPC_DISPONIBLE:
                motivo.append("posix_ipc no instalado")
            if not CONTRATOS_DISPONIBLES:
                motivo.append("contratos del Core no disponibles")
            if not motivo:
                motivo.append("forzado por parametro")
            _log.warning(
                "PublicadorMQ en MODO SIMULADO (%s). No se enviaran mensajes reales.",
                ", ".join(motivo),
            )
            self._activo = True
            return

        # ----- Modo real -----

        # Paso 1: si pidieron drenar, eliminamos cualquier cola preexistente.
        # Esto garantiza arrancar con una cola vacia y con la capacidad
        # correcta (si la cola vieja tenia otra capacidad, posix_ipc la
        # ignoraria al solo abrirla).
        if self.drenar_al_iniciar:
            try:
                cola_vieja = posix_ipc.MessageQueue(self.nombre_cola)
                # Contar cuantos mensajes viejos habia (solo informativo)
                pendientes = cola_vieja.current_messages
                cola_vieja.close()
                posix_ipc.unlink_message_queue(self.nombre_cola)
                if pendientes > 0:
                    _log.warning(
                        "Cola '%s' de una corrida anterior tenia %d mensajes. "
                        "Eliminada para arrancar limpio.",
                        self.nombre_cola, pendientes,
                    )
                self._mensajes_drenados = pendientes
            except posix_ipc.ExistentialError:
                # No existia cola previa: perfecto, nada que limpiar
                pass
            except Exception as e:
                _log.warning("No se pudo limpiar la cola preexistente: %s", e)

        # Paso 2: verificar que los limites del kernel permitan esta config.
        # Si no, lanza ErrorLimitesSistema con instrucciones claras (en vez
        # del criptico "Invalid parameter(s)" que daria posix_ipc).
        try:
            chequear_limites_sistema(self.capacidad_cola, self.tamano_max_mensaje)
        except ErrorLimitesSistema:
            self._activo = False
            raise

        # Paso 3: crear la cola nueva y limpia
        try:
            self._cola = posix_ipc.MessageQueue(
                self.nombre_cola,
                flags=posix_ipc.O_CREAT,
                max_messages=self.capacidad_cola,
                max_message_size=self.tamano_max_mensaje,
            )
            self._activo = True
            _log.info(
                "PublicadorMQ iniciado (cola=%s, capacidad=%d, sesion=%s)",
                self.nombre_cola, self.capacidad_cola, self._id_sesion,
            )
        except ErrorLimitesSistema:
            raise
        except Exception as e:
            self._cola = None
            self._activo = False
            raise ErrorPublicadorMQ(
                f"No se pudo abrir la cola POSIX MQ '{self.nombre_cola}': {e}"
            ) from e

    def detener(self) -> None:
        """
        Cierra la cola. Idempotente.

        Si eliminar_al_detener es True (default mientras no exista el Core),
        ademas elimina la cola del kernel con unlink. Esto es importante:
        sin el unlink, la cola sobrevive a la ejecucion y, si quedo llena,
        consume memoria del kernel hasta reiniciar la Pi.

        Cuando se integre el Core, eliminar_al_detener debe ser False: el
        Core es el duenio del ciclo de vida de la cola.
        """
        if self._cola is not None:
            try:
                self._cola.close()
            except Exception as e:
                _log.warning("Error al cerrar la cola: %s", e)
            self._cola = None

        # Eliminar la cola del kernel si corresponde
        if (not self.modo_simulado) and self.eliminar_al_detener:
            try:
                posix_ipc.unlink_message_queue(self.nombre_cola)
                _log.info("Cola '%s' eliminada del kernel", self.nombre_cola)
            except posix_ipc.ExistentialError:
                pass  # ya no existia
            except Exception as e:
                _log.warning("No se pudo eliminar la cola: %s", e)

        self._activo = False
        _log.info(
            "PublicadorMQ detenido (enviados=%d, descartados=%d, errores=%d)",
            self._mensajes_enviados, self._mensajes_descartados, self._errores,
        )

    def setear_neutros_cabeza(
        self,
        pitch_neutro: float,
        yaw_neutro: float = 0.0,
        roll_neutro: float = 0.0,
    ) -> None:
        """
        Actualiza los valores neutros usados para normalizar pitch/yaw/roll.

        Pensado para llamarse despues de la calibracion: el publicador se
        crea e inicia temprano (para detectar problemas de cola al
        arrancar), pero los valores neutros recien se conocen cuando
        termina la calibracion. Este metodo cierra esa brecha.

        Parametros
        ----------
        pitch_neutro : float
            Angulo de pitch de la pose neutra del conductor, en grados.
        yaw_neutro : float
            Idem para yaw. Default 0.0.
        roll_neutro : float
            Idem para roll. Default 0.0.
        """
        p_ant, y_ant, r_ant = self.pitch_neutro, self.yaw_neutro, self.roll_neutro
        self.pitch_neutro = float(pitch_neutro)
        self.yaw_neutro = float(yaw_neutro)
        self.roll_neutro = float(roll_neutro)
        _log.info(
            "PublicadorMQ: neutros actualizados "
            "pitch %.1f -> %.1f, yaw %.1f -> %.1f, roll %.1f -> %.1f grados",
            p_ant, self.pitch_neutro,
            y_ant, self.yaw_neutro,
            r_ant, self.roll_neutro,
        )

    # Alias de compatibilidad: mantiene el nombre viejo funcionando.
    # Si en algun lugar del codigo se llama a setear_pitch_neutro(x),
    # seguira funcionando (solo actualiza pitch, deja yaw/roll como estaban).
    def setear_pitch_neutro(self, pitch_neutro: float) -> None:
        """Alias de compatibilidad. Prefiera setear_neutros_cabeza."""
        self.setear_neutros_cabeza(
            pitch_neutro, self.yaw_neutro, self.roll_neutro,
        )

    def __enter__(self) -> "PublicadorMQ":
        self.iniciar()
        return self

    def __exit__(self, *_) -> None:
        self.detener()

    def __del__(self) -> None:
        try:
            self.detener()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Construccion del EventoVision
    # ------------------------------------------------------------------

    @staticmethod
    def _calcular_confianza(
        datos_rostro: DatosRostro,
        datos_ojos: Optional[DatosOjos],
        datos_boca: Optional[DatosBoca],
        datos_cabeza: Optional[DatosCabeza],
    ) -> float:
        """
        Deriva una confianza de deteccion 0.0-1.0 segun cuantos analizadores
        produjeron datos validos en este frame.

        Si no hay rostro, la confianza es 0.0.
        Si hay rostro, se reparte el credito entre los 3 analizadores
        (ojos, boca, cabeza): cada uno valido suma 1/3.
        """
        if not datos_rostro.rostro_presente:
            return 0.0

        validos = 0
        total = 3
        if datos_ojos is not None and datos_ojos.valido:
            validos += 1
        if datos_boca is not None and datos_boca.valido:
            validos += 1
        if datos_cabeza is not None and datos_cabeza.valido:
            validos += 1

        return validos / total

    def _construir_evento_dict(
        self,
        datos_rostro: DatosRostro,
        datos_ojos: Optional[DatosOjos],
        datos_boca: Optional[DatosBoca],
        datos_cabeza: Optional[DatosCabeza],
        datos_frote: Optional[DatosFroteOjos],
    ) -> dict:
        """
        Construye el diccionario de campos del EventoVision a partir de
        los resultados de los analizadores.

        Los campos van en None cuando el dato no se pudo medir.
        """
        rostro = datos_rostro.rostro_presente

        # EAR: solo si hay rostro y el analizador de ojos dio datos validos
        ear_izq = None
        ear_der = None
        if rostro and datos_ojos is not None and datos_ojos.valido:
            # El analizador devuelve 0.0 cuando un ojo es indeterminado;
            # en ese caso enviamos None (no medido) en vez de 0.0.
            ear_izq = datos_ojos.ear_izq if datos_ojos.ear_izq > 0.0 else None
            ear_der = datos_ojos.ear_der if datos_ojos.ear_der > 0.0 else None

        # MAR
        mar = None
        if rostro and datos_boca is not None and datos_boca.valido:
            mar = datos_boca.mar if datos_boca.mar > 0.0 else None

        # Angulos de cabeza
        pitch = None
        yaw = None
        roll = None
        if rostro and datos_cabeza is not None and datos_cabeza.valido:
            # NORMALIZACION DE ANGULOS (Opcion A de integracion):
            # restamos los valores neutros de la calibracion. El Core
            # recibe angulos relativos a la pose neutra del conductor y
            # sus umbrales absolutos se interpretan correctamente.
            #
            # Motivo del yaw: el Core invalida cabeceos con abs(yaw) > 35.
            # Si el conductor tiene la camara al costado y su yaw_neutro
            # calibrado es -35, cualquier micro-movimiento tira el yaw
            # crudo lejos de 0 y TODOS los cabeceos quedan invalidados.
            # Normalizando, el filtro pasa a significar "35 grados desde
            # la pose neutra real", que es el proposito buscado.
            #
            # Roll: el Core no lo usa hoy en filtros, pero lo normalizamos
            # por consistencia (si ma�ana lo usa, ya viene bien).
            #
            # Con neutros = 0.0 (default), esto no cambia nada.
            pitch = datos_cabeza.pitch_deg - self.pitch_neutro
            yaw = datos_cabeza.yaw_deg - self.yaw_neutro
            roll = datos_cabeza.roll_deg - self.roll_neutro

        # Frote de ojos: flag booleano
        frote_activo = False
        if datos_frote is not None and datos_frote.valido:
            frote_activo = datos_frote.frote_en_curso

        confianza = self._calcular_confianza(
            datos_rostro, datos_ojos, datos_boca, datos_cabeza,
        )

        # timestamp: el contrato EventoVision exige timestamp > 0 en su
        # __post_init__ (lanza ValueError si no). Blindamos: usamos el del
        # frame si es valido, y si no, time.time(). Como ultimo resguardo,
        # si por algun borde el valor no fuera positivo, forzamos time.time().
        ts = datos_rostro.timestamp
        if ts is None or ts <= 0:
            ts = time.time()
        # Resguardo final: time.time() siempre es > 0 en un sistema con reloj
        # valido, pero garantizamos el invariante del contrato de todos modos.
        if ts <= 0:
            ts = time.time()

        return {
            "timestamp": ts,
            "rostro_detectado": rostro,
            "ear_izquierdo": ear_izq,
            "ear_derecho": ear_der,
            "mar": mar,
            "pitch_grados": pitch,
            "yaw_grados": yaw,
            "roll_grados": roll,
            "frote_ojos_activo": frote_activo,
            "confianza_deteccion": confianza,
        }

    # ------------------------------------------------------------------
    # Publicacion
    # ------------------------------------------------------------------

    def publicar(
        self,
        datos_rostro: DatosRostro,
        datos_ojos: Optional[DatosOjos] = None,
        datos_boca: Optional[DatosBoca] = None,
        datos_cabeza: Optional[DatosCabeza] = None,
        datos_frote: Optional[DatosFroteOjos] = None,
    ) -> bool:
        """
        Construye un EventoVision con los datos del frame y lo publica
        al Core como un Envelope serializado.

        Parametros
        ----------
        datos_rostro : DatosRostro
            Obligatorio. Resultado del DetectorRostro.
        datos_ojos, datos_boca, datos_cabeza, datos_frote
            Opcionales. Si alguno es None, los campos correspondientes
            del EventoVision van en None.

        Returns
        -------
        bool
            True si el mensaje se envio (o se "envio" en modo simulado).
            False si se descarto (cola llena) o hubo error.
        """
        if not self._activo:
            raise ErrorPublicadorMQ(
                "PublicadorMQ no esta activo. Llama a iniciar() primero."
            )

        if datos_rostro is None:
            raise ErrorPublicadorMQ("datos_rostro no puede ser None")

        # Construir el diccionario del evento
        evento_dict = self._construir_evento_dict(
            datos_rostro, datos_ojos, datos_boca, datos_cabeza, datos_frote,
        )

        # Incrementar secuencia
        self._numero_secuencia += 1

        # ----- Modo simulado -----
        if self.modo_simulado:
            _log.debug(
                "[SIMULADO] EventoVision #%d: rostro=%s, conf=%.2f",
                self._numero_secuencia,
                evento_dict["rostro_detectado"],
                evento_dict["confianza_deteccion"],
            )
            self._mensajes_enviados += 1
            return True

        # ----- Modo real -----
        try:
            # Construir el EventoVision tipado
            evento = EventoVision(**evento_dict)

            # Generar id de mensaje
            if generar_id_mensaje is not None:
                id_msg = generar_id_mensaje("vis", self._numero_secuencia)
            else:
                id_msg = f"vis-{self._numero_secuencia}"

            # Envolver en Envelope
            envelope = Envelope(
                tipo=TipoMensaje.EVENTO_VISION,
                origen=OrigenEvento.VISION,
                id_dispositivo=self.id_dispositivo,
                id_sesion=self._id_sesion,
                id_mensaje=id_msg,
                numero_secuencia=self._numero_secuencia,
                timestamp_origen=evento_dict["timestamp"],
                payload_json=evento.to_json(),
            )

            mensaje = envelope.to_json()
            mensaje_bytes = mensaje.encode("utf-8")

            # Validar tamanio
            if len(mensaje_bytes) > self.tamano_max_mensaje:
                _log.error(
                    "Mensaje #%d demasiado grande (%d bytes > %d). Descartado.",
                    self._numero_secuencia, len(mensaje_bytes), self.tamano_max_mensaje,
                )
                self._errores += 1
                return False

        except Exception as e:
            _log.error("Error al construir el mensaje #%d: %s",
                       self._numero_secuencia, e)
            self._errores += 1
            return False

        # Enviar NO bloqueante. Si la cola esta llena, posix_ipc lanza
        # BusyError; lo capturamos y descartamos el mensaje (no congelamos).
        try:
            self._cola.send(mensaje_bytes, timeout=0)
            self._mensajes_enviados += 1
            return True
        except posix_ipc.BusyError:
            # Cola llena: el Core esta lento. Descartamos este frame.
            self._mensajes_descartados += 1
            if self._mensajes_descartados % 30 == 1:
                _log.warning(
                    "Cola llena, mensaje descartado (total descartados: %d). "
                    "El Core puede estar lento consumiendo.",
                    self._mensajes_descartados,
                )
            return False
        except Exception as e:
            _log.error("Error al enviar mensaje #%d: %s",
                       self._numero_secuencia, e)
            self._errores += 1
            return False

    # ------------------------------------------------------------------
    # Diagnostico
    # ------------------------------------------------------------------

    def obtener_estadisticas(self) -> dict:
        """Devuelve un diccionario con las estadisticas del publicador."""
        total = self._mensajes_enviados + self._mensajes_descartados
        tasa_envio = (
            self._mensajes_enviados / total if total > 0 else 0.0
        )
        return {
            "modo_simulado": self.modo_simulado,
            "activo": self._activo,
            "id_sesion": self._id_sesion,
            "mensajes_enviados": self._mensajes_enviados,
            "mensajes_descartados": self._mensajes_descartados,
            "mensajes_drenados": self._mensajes_drenados,
            "errores": self._errores,
            "numero_secuencia": self._numero_secuencia,
            "tasa_envio": round(tasa_envio, 3),
        }
