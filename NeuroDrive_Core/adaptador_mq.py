"""
NeuroDrive Core - Adaptador de POSIX Message Queues
====================================================

Encapsula la libreria `posix_ipc` detras de una clase simple y robusta.

Uso tipico (lado consumidor - Gestor):

    cola = AdaptadorMQ.abrir(
        nombre="/neurodrive_vision",
        modo="lectura",
        capacidad=16,
        tamano_max_mensaje=1024,
    )
    try:
        while not parar:
            envelope_json = cola.recibir(timeout_seg=1.0)
            if envelope_json is not None:
                envelope = Envelope.from_json(envelope_json)
                ...
    finally:
        cola.cerrar()

Uso tipico (lado productor - Vision/Wearable):

    cola = AdaptadorMQ.abrir(
        nombre="/neurodrive_vision",
        modo="escritura",
        capacidad=16,
        tamano_max_mensaje=1024,
    )
    try:
        ok = cola.enviar(envelope.to_json())
        if not ok:
            log.warning("Cola llena, mensaje descartado")
    finally:
        cola.cerrar()

Politica de cola llena:
    Al enviar, si la cola esta llena, el mensaje se DESCARTA (politica B).
    Esto evita deadlocks y acumulacion de eventos viejos irrelevantes.
    El llamador recibe False y puede loggear el evento.

Limpieza:
    cerrar()   -> libera el descriptor de archivo del proceso
    eliminar() -> borra la cola del kernel (mq_unlink). Solo el que
                  crea la cola deberia eliminarla, tipicamente al
                  shutdown del sistema.
"""

from __future__ import annotations

import errno
import logging
import os
import time
from pathlib import Path
from typing import Optional


_log = logging.getLogger("NeuroDrive.AdaptadorMQ")


# =============================================================================
#                         IMPORT CONDICIONAL DE posix_ipc
# =============================================================================
#
# La libreria posix_ipc solo existe en Linux/macOS. En Windows ni siquiera
# importa. Hacemos el import condicional para que el archivo se pueda
# inspeccionar/importar en cualquier OS, pero falle claro si se intenta usar
# fuera de Linux.

try:
    import posix_ipc  # type: ignore[import-not-found]
    _POSIX_IPC_DISPONIBLE = True
    _ERROR_IMPORT_POSIX_IPC: Optional[str] = None
except ImportError as e:
    posix_ipc = None  # type: ignore[assignment]
    _POSIX_IPC_DISPONIBLE = False
    _ERROR_IMPORT_POSIX_IPC = str(e)


# =============================================================================
#                              EXCEPCIONES
# =============================================================================


class ErrorAdaptadorMQ(Exception):
    """Error generico del adaptador."""


class ErrorPermisos(ErrorAdaptadorMQ):
    """Permisos insuficientes para abrir/usar la cola."""


class ErrorTamanoMensaje(ErrorAdaptadorMQ):
    """El mensaje supera el tamano maximo configurado."""


class ErrorLimitesSistema(ErrorAdaptadorMQ):
    """Los limites del kernel (msg_max, msgsize_max) no permiten esta config."""


class ErrorNoDisponible(ErrorAdaptadorMQ):
    """posix_ipc no esta instalado o no esta disponible en este sistema."""


# =============================================================================
#                          UTILIDADES DEL SISTEMA
# =============================================================================


def _leer_limite_kernel(path: str) -> Optional[int]:
    """
    Lee un limite del kernel desde /proc/sys/fs/mqueue/.

    Devuelve None si no se puede leer (el archivo no existe o no se tiene
    permiso de lectura). En sistemas Linux estandar siempre se puede leer.
    """
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def chequear_limites_sistema(capacidad: int, tamano_max_mensaje: int) -> None:
    """
    Verifica que los limites del kernel permitan la configuracion deseada.

    Si no los permite, lanza ErrorLimitesSistema con instrucciones claras
    para subir los limites en /proc/sys/fs/mqueue/.

    En sistemas que no son Linux, este chequeo se omite silenciosamente.
    """
    if not Path("/proc/sys/fs/mqueue").is_dir():
        # No estamos en Linux, no podemos chequear (ni hace falta)
        return

    msg_max = _leer_limite_kernel("/proc/sys/fs/mqueue/msg_max")
    msgsize_max = _leer_limite_kernel("/proc/sys/fs/mqueue/msgsize_max")

    problemas = []

    if msg_max is not None and capacidad > msg_max:
        problemas.append(
            f"  - capacidad solicitada ({capacidad}) > kernel msg_max ({msg_max})\n"
            f"    Para subirlo (requiere root):\n"
            f"      sudo sysctl -w fs.mqueue.msg_max={max(capacidad, 32)}\n"
            f"    Para hacerlo permanente, agregar a /etc/sysctl.conf:\n"
            f"      fs.mqueue.msg_max = {max(capacidad, 32)}"
        )

    if msgsize_max is not None and tamano_max_mensaje > msgsize_max:
        problemas.append(
            f"  - tamano_max_mensaje ({tamano_max_mensaje}) > kernel "
            f"msgsize_max ({msgsize_max})\n"
            f"    Para subirlo:\n"
            f"      sudo sysctl -w fs.mqueue.msgsize_max={max(tamano_max_mensaje, 8192)}"
        )

    if problemas:
        raise ErrorLimitesSistema(
            "Los limites del kernel no permiten la configuracion solicitada:\n"
            + "\n".join(problemas)
        )


# =============================================================================
#                              CLASE PRINCIPAL
# =============================================================================


class AdaptadorMQ:
    """
    Adaptador sobre una POSIX Message Queue.

    No instanciar directamente: usar AdaptadorMQ.abrir(...).

    Una instancia representa un descriptor abierto sobre una cola del kernel.
    Multiples procesos pueden tener sus propios descriptores sobre la misma
    cola (uno escribe, otro lee).
    """

    # Constantes de modos
    MODO_LECTURA = "lectura"
    MODO_ESCRITURA = "escritura"
    MODOS_VALIDOS = (MODO_LECTURA, MODO_ESCRITURA)

    def __init__(
        self,
        nombre: str,
        modo: str,
        capacidad: int,
        tamano_max_mensaje: int,
        permisos: int = 0o660,
    ) -> None:
        """Constructor privado. Usar AdaptadorMQ.abrir() en su lugar."""
        self.nombre = nombre
        self.modo = modo
        self.capacidad = capacidad
        self.tamano_max_mensaje = tamano_max_mensaje
        self.permisos = permisos

        # Estado runtime
        self._mq: Optional["posix_ipc.MessageQueue"] = None
        self._abierta = False
        self._cerrada = False

        # Estadisticas para debug/monitoreo
        self.mensajes_enviados = 0
        self.mensajes_recibidos = 0
        self.mensajes_descartados_cola_llena = 0
        self.errores = 0

    # ------------------------------------------------------------------
    #                       CONSTRUCTORES PUBLICOS
    # ------------------------------------------------------------------

    @classmethod
    def abrir(
        cls,
        nombre: str,
        modo: str,
        capacidad: int = 16,
        tamano_max_mensaje: int = 1024,
        permisos: int = 0o660,
        crear_si_no_existe: bool = True,
    ) -> AdaptadorMQ:
        """
        Abre (o crea) una POSIX Message Queue y devuelve un adaptador listo.

        Args:
            nombre: Nombre de la cola, debe empezar con '/' (ej: "/neurodrive_vision").
            modo: "lectura" o "escritura". Determina el flag de open.
            capacidad: Cantidad maxima de mensajes en cola. Limitado por el
                       kernel (ver /proc/sys/fs/mqueue/msg_max).
            tamano_max_mensaje: Tamano maximo de cada mensaje en bytes.
            permisos: Permisos UNIX para la cola (default 0o660).
            crear_si_no_existe: Si True, crea la cola si no existe. Si False,
                                falla si la cola no existe.

        Raises:
            ErrorNoDisponible: si posix_ipc no esta instalado.
            ErrorPermisos: si los permisos no permiten abrir la cola.
            ErrorLimitesSistema: si los limites del kernel son insuficientes.
            ErrorAdaptadorMQ: para otros errores.
        """
        if not _POSIX_IPC_DISPONIBLE:
            raise ErrorNoDisponible(
                f"posix_ipc no esta disponible: {_ERROR_IMPORT_POSIX_IPC}\n"
                f"Instalar con: pip install posix_ipc"
            )

        if modo not in cls.MODOS_VALIDOS:
            raise ErrorAdaptadorMQ(
                f"Modo invalido: '{modo}' (validos: {cls.MODOS_VALIDOS})"
            )

        if not nombre.startswith("/"):
            raise ErrorAdaptadorMQ(
                f"Nombre de cola POSIX debe empezar con '/': {nombre}"
            )

        if len(nombre) > 255:
            raise ErrorAdaptadorMQ(
                f"Nombre demasiado largo (max 255): {nombre}"
            )

        if capacidad < 1:
            raise ErrorAdaptadorMQ(f"capacidad debe ser >= 1: {capacidad}")

        if tamano_max_mensaje < 128:
            raise ErrorAdaptadorMQ(
                f"tamano_max_mensaje muy chico (min 128): {tamano_max_mensaje}"
            )

        # Validar limites del kernel ANTES de intentar crear la cola
        chequear_limites_sistema(capacidad, tamano_max_mensaje)

        # Construir flags para posix_ipc
        # NOTA: en kernels Linux modernos, posix_ipc con O_WRONLY puro genera
        # errores espurios de "queue does not exist". La solucion estandar es
        # usar siempre O_RDWR a nivel de kernel y mantener la disciplina de
        # modo logico ("lectura" o "escritura") a nivel del adaptador. El
        # validador _validar_modo_operacion() impone la restriccion logica.
        flag_acceso = os.O_RDWR

        flags = flag_acceso
        if crear_si_no_existe:
            flags |= posix_ipc.O_CREAT  # type: ignore[union-attr]

        # Crear la instancia y abrir la cola
        adapt = cls(
            nombre=nombre,
            modo=modo,
            capacidad=capacidad,
            tamano_max_mensaje=tamano_max_mensaje,
            permisos=permisos,
        )

        try:
            adapt._mq = posix_ipc.MessageQueue(  # type: ignore[union-attr]
                name=nombre,
                flags=flags,
                mode=permisos,
                max_messages=capacidad,
                max_message_size=tamano_max_mensaje,
            )
            adapt._abierta = True
            _log.info(
                "Cola abierta: name=%s modo=%s capacidad=%d tamano_max=%d",
                nombre, modo, capacidad, tamano_max_mensaje,
            )
        except posix_ipc.PermissionsError as e:  # type: ignore[union-attr]
            raise ErrorPermisos(
                f"Permisos insuficientes para abrir {nombre}: {e}\n"
                f"Verificar permisos de la cola (debe ser accesible por este usuario)."
            ) from e
        except posix_ipc.ExistentialError as e:  # type: ignore[union-attr]
            raise ErrorAdaptadorMQ(
                f"La cola {nombre} no existe y crear_si_no_existe=False: {e}"
            ) from e
        except OSError as e:
            # Catch all para errores del kernel no manejados arriba
            if e.errno == errno.EMFILE:
                raise ErrorAdaptadorMQ(
                    f"Demasiados descriptores abiertos (EMFILE) al abrir {nombre}"
                ) from e
            if e.errno == errno.ENOMEM:
                raise ErrorAdaptadorMQ(
                    f"Sin memoria suficiente (ENOMEM) al abrir {nombre}"
                ) from e
            raise ErrorAdaptadorMQ(
                f"Error de sistema al abrir {nombre}: errno={e.errno} {e}"
            ) from e

        return adapt

    # ------------------------------------------------------------------
    #                         API DE LECTURA
    # ------------------------------------------------------------------

    def recibir(self, timeout_seg: Optional[float] = None) -> Optional[str]:
        """
        Recibe un mensaje de la cola.

        Args:
            timeout_seg: Tiempo maximo de espera en segundos.
                         - None: bloquea hasta que llegue un mensaje
                         - 0.0: no bloquea, devuelve None inmediato si vacia
                         - >0: bloquea hasta ese tiempo, luego None

        Returns:
            El mensaje como string (UTF-8) o None si hubo timeout.

        Raises:
            ErrorAdaptadorMQ: si la cola esta cerrada, en modo escritura,
                              o hubo un error de sistema.
        """
        self._validar_modo_operacion(self.MODO_LECTURA)

        try:
            mensaje_bytes, _prioridad = self._mq.receive(timeout=timeout_seg)  # type: ignore[union-attr]
        except posix_ipc.BusyError:  # type: ignore[union-attr]
            # Timeout en recepcion no es un error, solo "no hubo mensaje"
            return None
        except posix_ipc.SignalError:  # type: ignore[union-attr]
            # Interrupcion por senal del SO (ej: SIGINT). Tratamos como timeout.
            _log.debug("recibir() interrumpido por senal")
            return None
        except OSError as e:
            self.errores += 1
            raise ErrorAdaptadorMQ(
                f"Error al recibir de {self.nombre}: errno={e.errno} {e}"
            ) from e

        self.mensajes_recibidos += 1
        try:
            return mensaje_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            self.errores += 1
            raise ErrorAdaptadorMQ(
                f"Mensaje recibido no es UTF-8 valido: {e}"
            ) from e

    # ------------------------------------------------------------------
    #                         API DE ESCRITURA
    # ------------------------------------------------------------------

    def enviar(self, mensaje: str, prioridad: int = 0) -> bool:
        """
        Envia un mensaje a la cola. Politica de cola llena: DESCARTA.

        Args:
            mensaje: String a enviar (sera codificado a UTF-8).
            prioridad: Prioridad POSIX (0-31, default 0). Mayor = se entrega
                       antes que mensajes con prioridad menor.

        Returns:
            True si se envio, False si la cola estaba llena y se descarto.

        Raises:
            ErrorAdaptadorMQ: si la cola esta cerrada, en modo lectura, o
                              hubo error de sistema (NO se levanta por cola
                              llena: eso devuelve False).
            ErrorTamanoMensaje: si el mensaje supera tamano_max_mensaje.
        """
        self._validar_modo_operacion(self.MODO_ESCRITURA)

        # Codificar a bytes
        try:
            mensaje_bytes = mensaje.encode("utf-8")
        except UnicodeEncodeError as e:
            self.errores += 1
            raise ErrorAdaptadorMQ(
                f"Mensaje no se pudo codificar a UTF-8: {e}"
            ) from e

        # Verificar tamano antes de enviar (mensaje claro vs error del kernel)
        if len(mensaje_bytes) > self.tamano_max_mensaje:
            self.errores += 1
            raise ErrorTamanoMensaje(
                f"Mensaje de {len(mensaje_bytes)} bytes supera el maximo "
                f"de {self.tamano_max_mensaje} bytes configurado para {self.nombre}"
            )

        # Intentar enviar con timeout=0 (no bloqueante)
        try:
            self._mq.send(  # type: ignore[union-attr]
                message=mensaje_bytes,
                timeout=0,
                priority=prioridad,
            )
            self.mensajes_enviados += 1
            return True
        except posix_ipc.BusyError:  # type: ignore[union-attr]
            # Cola llena: DESCARTAR (politica B)
            self.mensajes_descartados_cola_llena += 1
            _log.warning(
                "Cola %s llena, mensaje descartado (total descartados: %d)",
                self.nombre,
                self.mensajes_descartados_cola_llena,
            )
            return False
        except posix_ipc.SignalError:  # type: ignore[union-attr]
            # Interrupcion por senal: tratamos como descarte
            _log.debug("enviar() interrumpido por senal")
            return False
        except OSError as e:
            self.errores += 1
            raise ErrorAdaptadorMQ(
                f"Error al enviar a {self.nombre}: errno={e.errno} {e}"
            ) from e

    # ------------------------------------------------------------------
    #                         ADMINISTRACION
    # ------------------------------------------------------------------

    def cerrar(self) -> None:
        """
        Cierra el descriptor del proceso actual.

        La cola sigue existiendo en el kernel y otros procesos pueden seguir
        usandola. Es seguro llamar varias veces (es idempotente).
        """
        if self._cerrada or not self._abierta or self._mq is None:
            return

        try:
            self._mq.close()
            _log.info("Cola %s cerrada (descriptor liberado)", self.nombre)
        except Exception as e:
            _log.warning("Error al cerrar cola %s: %s", self.nombre, e)
        finally:
            self._cerrada = True
            self._abierta = False
            self._mq = None

    def eliminar(self) -> None:
        """
        Borra la cola del kernel (mq_unlink).

        Solo el creador de la cola deberia eliminarla, tipicamente al
        shutdown del sistema. Despues de esto, ningun proceso puede
        usar la cola sin recrearla.

        Es seguro llamar aunque la cola ya este cerrada.
        """
        try:
            # Si esta abierta, primero cerrar el descriptor
            if self._abierta and self._mq is not None:
                try:
                    self._mq.close()
                except Exception:
                    pass
                self._abierta = False
                self._cerrada = True

            # Unlink la cola del kernel
            if _POSIX_IPC_DISPONIBLE:
                try:
                    posix_ipc.unlink_message_queue(self.nombre)  # type: ignore[union-attr]
                    _log.info("Cola %s eliminada del kernel", self.nombre)
                except posix_ipc.ExistentialError:  # type: ignore[union-attr]
                    # Ya estaba borrada, ok
                    pass
        except Exception as e:
            _log.warning("Error al eliminar cola %s: %s", self.nombre, e)
        finally:
            self._mq = None

    def mensajes_en_cola(self) -> int:
        """
        Cantidad actual de mensajes en la cola.

        Util para monitoreo: si esto crece sostenidamente, el consumidor
        no esta dando abasto.

        Returns:
            Numero de mensajes esperando ser leidos.
        """
        if not self._abierta or self._mq is None:
            return 0
        try:
            return self._mq.current_messages
        except Exception:
            return 0

    def estadisticas(self) -> dict:
        """Devuelve un snapshot de las estadisticas del adaptador."""
        return {
            "nombre": self.nombre,
            "modo": self.modo,
            "abierta": self._abierta,
            "mensajes_enviados": self.mensajes_enviados,
            "mensajes_recibidos": self.mensajes_recibidos,
            "mensajes_descartados_cola_llena": self.mensajes_descartados_cola_llena,
            "errores": self.errores,
            "mensajes_en_cola_actual": self.mensajes_en_cola(),
        }

    # ------------------------------------------------------------------
    #                       CONTEXT MANAGER
    # ------------------------------------------------------------------

    def __enter__(self) -> AdaptadorMQ:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cerrar()

    def __repr__(self) -> str:
        estado = "abierta" if self._abierta else "cerrada"
        return (
            f"AdaptadorMQ(nombre={self.nombre}, modo={self.modo}, "
            f"capacidad={self.capacidad}, estado={estado})"
        )

    # ------------------------------------------------------------------
    #                          INTERNAS
    # ------------------------------------------------------------------

    def _validar_modo_operacion(self, modo_requerido: str) -> None:
        """Valida que la cola este abierta y en el modo correcto."""
        if self._cerrada:
            raise ErrorAdaptadorMQ(
                f"Cola {self.nombre} ya esta cerrada, no se puede operar"
            )
        if not self._abierta or self._mq is None:
            raise ErrorAdaptadorMQ(
                f"Cola {self.nombre} no esta abierta"
            )
        if self.modo != modo_requerido:
            raise ErrorAdaptadorMQ(
                f"Cola {self.nombre} esta en modo '{self.modo}', "
                f"se requiere '{modo_requerido}'"
            )


# =============================================================================
#                       UTILIDAD: ELIMINAR COLA SIN ABRIRLA
# =============================================================================


def eliminar_cola(nombre: str) -> bool:
    """
    Elimina una cola del kernel sin necesidad de tenerla abierta.

    Util para limpieza de pruebas o reset del sistema.

    Returns:
        True si se elimino, False si no existia.
    """
    if not _POSIX_IPC_DISPONIBLE:
        raise ErrorNoDisponible(_ERROR_IMPORT_POSIX_IPC or "posix_ipc no disponible")

    try:
        posix_ipc.unlink_message_queue(nombre)  # type: ignore[union-attr]
        _log.info("Cola %s eliminada", nombre)
        return True
    except posix_ipc.ExistentialError:  # type: ignore[union-attr]
        return False
    except Exception as e:
        _log.warning("Error eliminando cola %s: %s", nombre, e)
        return False


__all__ = [
    "AdaptadorMQ",
    "ErrorAdaptadorMQ",
    "ErrorPermisos",
    "ErrorTamanoMensaje",
    "ErrorLimitesSistema",
    "ErrorNoDisponible",
    "chequear_limites_sistema",
    "eliminar_cola",
]
