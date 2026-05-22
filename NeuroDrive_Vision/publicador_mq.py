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

    # Nombre de la cola POSIX MQ (debe coincidir con el del Gestor del Core)
    NOMBRE_COLA_DEFAULT = "/neurodrive_vision"

    # Tamanio maximo de mensaje (bytes). Debe coincidir con el config del Core.
    TAMANIO_MAX_MENSAJE = 1024

    # Capacidad maxima de la cola (cantidad de mensajes)
    CAPACIDAD_COLA_DEFAULT = 10

    # ID de este dispositivo de vision
    ID_DISPOSITIVO_DEFAULT = "cam-01"

    def __init__(
        self,
        config: Optional["Config"] = None,
        nombre_cola: Optional[str] = None,
        id_dispositivo: Optional[str] = None,
        capacidad_cola: Optional[int] = None,
        forzar_simulado: bool = False,
    ) -> None:
        """
        Parametros
        ----------
        config : Config | None
            Configuracion global (reservada).
        nombre_cola : str | None
            Nombre de la POSIX MQ. Default "/neurodrive_vision".
        id_dispositivo : str | None
            Identificador de esta camara. Default "cam-01".
        capacidad_cola : int | None
            Cantidad maxima de mensajes en la cola. Default 10.
        forzar_simulado : bool
            Si True, opera en modo simulado aunque posix_ipc este disponible.
            Util para tests.
        """
        self.config = config
        self.nombre_cola = nombre_cola or self.NOMBRE_COLA_DEFAULT
        self.id_dispositivo = id_dispositivo or self.ID_DISPOSITIVO_DEFAULT

        if capacidad_cola is not None:
            if capacidad_cola < 1:
                raise ValueError("capacidad_cola debe ser >= 1")
            self.capacidad_cola = int(capacidad_cola)
        else:
            self.capacidad_cola = self.CAPACIDAD_COLA_DEFAULT

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
        Abre (o crea) la cola POSIX MQ.

        Si esta en modo simulado, solo genera el id de sesion y marca activo.
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

        # Modo real: abrir la cola
        try:
            self._cola = posix_ipc.MessageQueue(
                self.nombre_cola,
                flags=posix_ipc.O_CREAT,
                max_messages=self.capacidad_cola,
                max_message_size=self.TAMANIO_MAX_MENSAJE,
            )
            self._activo = True
            _log.info(
                "PublicadorMQ iniciado (cola=%s, capacidad=%d, sesion=%s)",
                self.nombre_cola, self.capacidad_cola, self._id_sesion,
            )
        except Exception as e:
            self._cola = None
            self._activo = False
            raise ErrorPublicadorMQ(
                f"No se pudo abrir la cola POSIX MQ '{self.nombre_cola}': {e}"
            ) from e

    def detener(self) -> None:
        """
        Cierra la cola. Idempotente.

        NOTA: cerramos la cola pero NO la desvinculamos (unlink). El Core
        es el "duenio" de la cola; si la desvinculamos, romperiamos al Core.
        Solo cerramos nuestro descriptor.
        """
        if self._cola is not None:
            try:
                self._cola.close()
            except Exception as e:
                _log.warning("Error al cerrar la cola: %s", e)
            self._cola = None
        self._activo = False
        _log.info(
            "PublicadorMQ detenido (enviados=%d, descartados=%d, errores=%d)",
            self._mensajes_enviados, self._mensajes_descartados, self._errores,
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
            pitch = datos_cabeza.pitch_deg
            yaw = datos_cabeza.yaw_deg
            roll = datos_cabeza.roll_deg

        # Frote de ojos: flag booleano
        frote_activo = False
        if datos_frote is not None and datos_frote.valido:
            frote_activo = datos_frote.frote_en_curso

        confianza = self._calcular_confianza(
            datos_rostro, datos_ojos, datos_boca, datos_cabeza,
        )

        # timestamp: usamos el del frame si esta, sino el actual
        ts = datos_rostro.timestamp if datos_rostro.timestamp > 0 else time.time()

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
            if len(mensaje_bytes) > self.TAMANIO_MAX_MENSAJE:
                _log.error(
                    "Mensaje #%d demasiado grande (%d bytes > %d). Descartado.",
                    self._numero_secuencia, len(mensaje_bytes), self.TAMANIO_MAX_MENSAJE,
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
            "errores": self._errores,
            "numero_secuencia": self._numero_secuencia,
            "tasa_envio": round(tasa_envio, 3),
        }
