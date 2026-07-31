"""
NeuroDrive Core - Orquestador
==============================

Une todas las piezas del Core en un unico bucle de decision:

    Gestor --(Envelope)--> PreFSM --(EventoProcesado)--> FSM --(SalidaFSM)--> Despachador --> Actuadores
                             |                             ^
                             | (ACK / fallo / recuperacion)|
                             +-----------------------------+

Ruteo de cada envelope (el PreFSM ya sabe procesar todos los tipos):
  - EventoVision / EventoWearable  -> PreFSM produce un EventoProcesado -> FSM
  - EventoAckWearable              -> PreFSM devuelve None; el crudo va directo a la FSM
  - EventoFalloSensor / Recuperacion -> PreFSM actualiza disponibilidad y
                                        devuelve None; el crudo va directo a la FSM
En todos los casos, si la FSM produce una SalidaFSM con comandos, se despacha.

El Orquestador recibe sus dependencias ya construidas (inyeccion), asi se
puede testear con stubs sin hardware. El main.py arma las piezas reales.

Robustez: un error procesando un evento se loguea y NO tira el bucle. Un
sistema de seguridad no puede morir por un evento mal formado.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from common.contratos import (
    EstadoFSM,
    EventoAckWearable,
    EventoFalloSensor,
    EventoRecuperacionSensor,
)


@dataclass
class EstadisticasOrquestador:
    envelopes: int = 0
    eventos_a_fsm: int = 0
    transiciones: int = 0
    comandos_despachados: int = 0
    errores_procesamiento: int = 0


class Orquestador:
    def __init__(
        self,
        gestor,
        pre_fsm,
        fsm,
        despachador,
        receptor_wearable=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.gestor = gestor
        self.pre_fsm = pre_fsm
        self.fsm = fsm
        self.despachador = despachador
        self.receptor = receptor_wearable
        self.log = logger or logging.getLogger("NeuroDrive.Orquestador")

        self.stats = EstadisticasOrquestador()
        self._estado_previo: Optional[EstadoFSM] = None
        self._iniciado = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        if self._iniciado:
            return
        # Orden: Gestor primero (es duenio de las colas POSIX), despues el
        # receptor del wearable (escribe en la cola que el Gestor ya creo),
        # y el despachador (deja los actuadores listos).
        self.gestor.iniciar()
        if self.receptor is not None:
            self.receptor.iniciar()
        self.despachador.iniciar()
        self._iniciado = True
        self.log.info("Orquestador iniciado")

    def detener(self) -> None:
        if not self._iniciado:
            return
        self._iniciado = False
        # Apagar actuadores PRIMERO (que no quede nada vibrando/sonando),
        # despues el receptor y por ultimo el Gestor.
        try:
            self.despachador.detener()
        except Exception as e:
            self.log.error("Error deteniendo el despachador: %s", e)
        if self.receptor is not None:
            try:
                self.receptor.detener()
            except Exception as e:
                self.log.error("Error deteniendo el receptor: %s", e)
        try:
            self.gestor.detener()
        except Exception as e:
            self.log.error("Error deteniendo el gestor: %s", e)
        self.log.info("Orquestador detenido")

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def correr(self, duracion_max: float = 0.0, periodo_resumen: float = 5.0) -> None:
        """
        Bucle principal. Bloquea hasta que el Gestor deje de estar activo
        (Ctrl+C, que el Gestor captura) o hasta duracion_max segundos.
        """
        t_inicio = time.monotonic()
        t_ultimo_resumen = t_inicio
        while self.gestor.activo:
            envelope = self.gestor.obtener_evento(timeout=1.0)
            if envelope is not None:
                self.stats.envelopes += 1
                self._procesar_envelope(envelope)

            ahora = time.monotonic()
            if periodo_resumen > 0 and (ahora - t_ultimo_resumen) >= periodo_resumen:
                self.log.info(self.resumen())
                t_ultimo_resumen = ahora
            if duracion_max > 0 and (ahora - t_inicio) >= duracion_max:
                self.log.info("Duracion maxima (%.0fs) alcanzada", duracion_max)
                break

    def procesar_uno(self, timeout: float = 1.0) -> bool:
        """Procesa un solo evento (util para tests). Devuelve True si hubo evento."""
        envelope = self.gestor.obtener_evento(timeout=timeout)
        if envelope is None:
            return False
        self.stats.envelopes += 1
        self._procesar_envelope(envelope)
        return True

    # ------------------------------------------------------------------
    # Ruteo de un envelope
    # ------------------------------------------------------------------

    def _procesar_envelope(self, envelope) -> None:
        try:
            ev = envelope.evento
            # El PreFSM procesa SIEMPRE: actualiza su estado interno
            # (disponibilidad de sensores, BPM) y, para vision/wearable,
            # produce un EventoProcesado.
            evento_proc = self.pre_fsm.procesar(envelope)

            salida = None
            if evento_proc is not None:
                salida = self.fsm.procesar_evento(evento_proc)
            elif isinstance(ev, (EventoAckWearable, EventoFalloSensor,
                                 EventoRecuperacionSensor)):
                # Eventos que la FSM consume directamente (el PreFSM devolvio None)
                salida = self.fsm.procesar_evento(ev)

            if salida is None:
                return

            self.stats.eventos_a_fsm += 1
            if salida.transicion_ocurrio:
                self.stats.transiciones += 1
                self.log.info(
                    "FSM %s -> %s (%s)",
                    salida.estado_anterior.name,
                    salida.estado_actual.name,
                    salida.motivo_transicion,
                )
            if salida.comandos:
                self.stats.comandos_despachados += len(salida.comandos)
            self.despachador.despachar(salida)

        except Exception as e:
            self.stats.errores_procesamiento += 1
            self.log.error("Error procesando envelope: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    def resumen(self) -> str:
        estado = "?"
        try:
            estado = self.fsm.get_estado_actual().name
        except Exception:
            pass
        return (
            f"[ORQ] estado={estado} envelopes={self.stats.envelopes} "
            f"aFSM={self.stats.eventos_a_fsm} transiciones={self.stats.transiciones} "
            f"comandos={self.stats.comandos_despachados} "
            f"errores={self.stats.errores_procesamiento}"
        )
