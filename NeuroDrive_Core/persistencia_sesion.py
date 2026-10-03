"""
NeuroDrive Core - Persistencia del estado entre sesiones
=========================================================

La somnolencia del conductor no desaparece porque un programa se reinicie.
Este modulo conserva en disco lo que el nucleo sabe del conductor y lo
recupera al iniciar, siempre que la sesion anterior sea reciente.

Que se guarda (contrato EstadoSesion):
  - el estado de la FSM en ese momento,
  - las marcas de tiempo de los episodios severos (fatiga recurrente),
  - las marcas de tiempo de los bostezos de la ventana larga.

Cuando se guarda (lo decide el Orquestador):
  - en cada transicion de la FSM y cada vez que emite ordenes,
  - en cada bostezo confirmado,
  - en el apagado ordenado.
  Guardar en los cambios, y no solo al apagar, es lo que cubre el reinicio
  IMPREVISTO: un corte de alimentacion no ejecuta ningun apagado.

Como se retoma (resolver_arranque):
  - sin archivo, archivo ilegible o sesion vencida (ttl_sesion_seg) -> NORMAL;
  - si la sesion anterior estaba en PRE_ALERTA o por encima, o dejo episodios
    severos todavia vigentes -> PRE_ALERTA: un nivel por encima del normal,
    sin alerta perceptible, del que se desciende tras el tiempo habitual sin
    incidentes;
  - en cualquier otro caso (NORMAL o MODO_DEGRADADO sin episodios) -> NORMAL.
  Nunca se retoma directamente en una alerta: la verificacion que estaba
  pendiente ya no existe en la pulsera.

Este modulo es el unico del nucleo que escribe en disco. La FSM sigue siendo
logica pura: recibe el estado inicial y los episodios por su constructor.

Robustez:
  - Escritura atomica (archivo temporal + os.replace): un corte de energia a
    mitad de la escritura deja el archivo anterior intacto, nunca uno a medias.
  - guardar() y cargar() no lanzan excepciones: un problema de disco se
    registra y el sistema continua. Perder la persistencia no puede detener
    la vigilancia.
  - Reloj retrasado: la Raspberry Pi sin red puede iniciar con una hora
    anterior a la del guardado. En ese caso la sesion se considera reciente
    (es el supuesto conservador) y las marcas "del futuro" se recortan a la
    hora actual.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from common.contratos import EstadoFSM, EstadoSesion


_log = logging.getLogger("NeuroDrive.PersistenciaSesion")

# Estados de la sesion anterior que hacen retomar un nivel por encima del normal
_ESTADOS_CON_INDICIOS = frozenset({
    EstadoFSM.PRE_ALERTA,
    EstadoFSM.ALERTA_LEVE,
    EstadoFSM.ALERTA_MEDIA,
    EstadoFSM.CRITICO,
})


# =============================================================================
#                       LECTURA Y ESCRITURA EN DISCO
# =============================================================================


class PersistenciaSesion:
    """
    Guarda y recupera un EstadoSesion en un archivo JSON.

        p = PersistenciaSesion("NeuroDrive_Core/estado_sesion.json", ttl_seg=900)
        previa = p.cargar()            # None si no hay sesion reciente
        ...
        p.guardar(estado_sesion)       # atomico; devuelve False si no pudo
    """

    def __init__(
        self,
        ruta: Union[str, Path],
        ttl_seg: float = 900.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if ttl_seg <= 0:
            raise ValueError(f"ttl_seg debe ser > 0: {ttl_seg}")
        self.ruta = Path(ruta)
        self.ttl_seg = float(ttl_seg)
        self.log = logger or _log
        self.guardados_ok = 0
        self.guardados_fallidos = 0

    def guardar(self, estado: EstadoSesion) -> bool:
        """Escribe el estado de forma atomica. No lanza: devuelve False si fallo."""
        temporal = self.ruta.with_name(self.ruta.name + ".tmp")
        try:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            with open(temporal, "w", encoding="utf-8") as f:
                f.write(estado.to_json())
                f.flush()
                os.fsync(f.fileno())       # que llegue al disco antes del reemplazo
            os.replace(temporal, self.ruta)   # atomico dentro del mismo directorio
            self.guardados_ok += 1
            return True
        except (OSError, TypeError, ValueError) as e:
            self.guardados_fallidos += 1
            self.log.error("No se pudo guardar el estado de sesion en %s: %s",
                           self.ruta, e)
            try:
                temporal.unlink()
            except OSError:
                pass
            return False

    def cargar(self, ahora: Optional[float] = None) -> Optional[EstadoSesion]:
        """
        Devuelve el EstadoSesion guardado si existe, es legible y no vencio.
        En cualquier otro caso devuelve None. No lanza.
        """
        ahora = time.time() if ahora is None else ahora
        try:
            texto = self.ruta.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            self.log.warning("No se pudo leer %s: %s", self.ruta, e)
            return None

        try:
            estado = EstadoSesion.from_json(texto)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError) as e:
            # Archivo danado o escrito por una version anterior del contrato
            self.log.warning("Estado de sesion ilegible en %s (%s): se ignora",
                             self.ruta, e)
            return None

        antiguedad = ahora - estado.timestamp_guardado
        if antiguedad > self.ttl_seg:
            self.log.info("Sesion anterior vencida (%.0f s > %.0f s): se ignora",
                          antiguedad, self.ttl_seg)
            return None
        if antiguedad < 0:
            self.log.warning(
                "El reloj esta %.0f s por detras del ultimo guardado; "
                "la sesion anterior se toma como reciente", -antiguedad,
            )
        return estado

    def borrar(self) -> None:
        """Elimina el archivo (para iniciar una sesion limpia). No lanza."""
        try:
            self.ruta.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            self.log.warning("No se pudo borrar %s: %s", self.ruta, e)


# =============================================================================
#                 DE LA SESION ANTERIOR AL ARRANQUE DE ESTA
# =============================================================================


@dataclass(frozen=True)
class ArranqueSesion:
    """Con que arranca el nucleo: lo que main.py le pasa a la FSM y al Pre-FSM."""
    estado_inicial: EstadoFSM = EstadoFSM.NORMAL
    episodios_severos: Tuple[float, ...] = ()
    bostezos: Tuple[float, ...] = ()
    retomada: bool = False
    descripcion: str = "sin sesion anterior reciente: se inicia en NORMAL"


def _vigentes(marcas, ahora: float, ventana_seg: float) -> Tuple[float, ...]:
    """Marcas de tiempo dentro de la ventana. Las posteriores a `ahora` (reloj
    retrasado) se recortan a `ahora`, que es tratarlas como recientes."""
    limite = ahora - ventana_seg
    return tuple(sorted(min(float(t), ahora) for t in marcas if float(t) >= limite))


def resolver_arranque(
    previa: Optional[EstadoSesion],
    ahora: float,
    ventana_episodios_seg: float,
    ventana_bostezos_seg: float,
) -> ArranqueSesion:
    """
    Decide el estado inicial y que indicadores se conservan. Funcion pura: no
    toca el disco, de modo que la regla se verifica sin archivos.
    """
    if previa is None:
        return ArranqueSesion()

    episodios = _vigentes(previa.episodios_severos, ahora, ventana_episodios_seg)
    bostezos = _vigentes(previa.bostezos_recientes, ahora, ventana_bostezos_seg)

    con_indicios = previa.estado_fsm in _ESTADOS_CON_INDICIOS or len(episodios) > 0
    estado_inicial = EstadoFSM.PRE_ALERTA if con_indicios else EstadoFSM.NORMAL

    antiguedad = max(0.0, ahora - previa.timestamp_guardado)
    descripcion = (
        f"sesion anterior de hace {antiguedad:.0f} s en {previa.estado_fsm.name} "
        f"({len(episodios)} episodio(s) severo(s), {len(bostezos)} bostezo(s) vigentes): "
        f"se inicia en {estado_inicial.name}"
    )
    return ArranqueSesion(
        estado_inicial=estado_inicial,
        episodios_severos=episodios,
        bostezos=bostezos,
        retomada=True,
        descripcion=descripcion,
    )


def capturar(fsm, pre_fsm, motivo: str, ahora: Optional[float] = None) -> EstadoSesion:
    """
    Arma el EstadoSesion con lo que el nucleo sabe en este momento.

    `fsm` y `pre_fsm` solo necesitan exponer get_estado_actual(),
    get_episodios_severos() y get_bostezos_recientes().
    """
    return EstadoSesion(
        timestamp_guardado=time.time() if ahora is None else ahora,
        estado_fsm=fsm.get_estado_actual(),
        bostezos_recientes=tuple(pre_fsm.get_bostezos_recientes()),
        episodios_severos=tuple(fsm.get_episodios_severos()),
        motivo_guardado=motivo,
    )


__all__ = ["PersistenciaSesion", "ArranqueSesion", "resolver_arranque", "capturar"]
