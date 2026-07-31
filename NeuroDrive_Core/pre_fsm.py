"""
NeuroDrive Core - Pre-FSM (Evaluador de eventos primarios)
============================================================

Componente que transforma eventos crudos de Vision y Wearable en
EventoProcesado, que es lo que la FSM consume.

Responsabilidades:
  1. State tracking de eventos discretos:
     - Bostezos: detecta inicio/fin via MAR, emite UNA VEZ cuando termina
     - Microsuenos: detecta cierre prolongado de ojos via EAR
     - Cabeceos: detecta inclinacion sostenida via pitch
     - Parpadeos: detecta cruces de EAR (para freq/min)
  2. Calculos en ventanas temporales:
     - PERCLOS: % tiempo con ojos cerrados (60s)
     - Frecuencia de parpadeos por minuto (sliding window)
     - Bostezos en ventana larga (15 min)
  3. Clasificacion del BPM segun umbrales del config
  4. Marcado de ventanas no confiables:
     - frote_ojos_activo
     - rostro perdido (1-5 frames -> no confiable; >5 -> vision_disponible=False)

Arquitectura interna: subdetectores especializados sin estado compartido.

API:
    pre_fsm = PreFSM(config)
    envelope = gestor.obtener_evento()
    evento_proc = pre_fsm.procesar(envelope)  # puede ser None
    if evento_proc is not None:
        fsm.procesar_evento(evento_proc)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional, Tuple

from common.contratos import (
    Envelope,
    EventoAckWearable,
    EventoFalloSensor,
    EventoProcesado,
    EventoRecuperacionSensor,
    EventoVision,
    EventoWearable,
    NivelRiesgoBPM,
    OrigenEvento,
)
from NeuroDrive_Core.config_loader import Config


_log = logging.getLogger("NeuroDrive.PreFSM")


# =============================================================================
#                    SUB-DETECTORES (testeables por separado)
# =============================================================================


class DetectorParpadeos:
    """
    Cuenta parpadeos individuales y calcula su frecuencia por minuto.

    Algoritmo: detecta cruces hacia abajo del EAR (ojos cerrandose),
    impone un periodo refractario para no contar dos veces el mismo
    parpadeo. Mantiene una ventana deslizante de timestamps.
    """

    def __init__(self, config_ojos, ventana_calculo_seg: float = 60.0):
        self.umbral_cerrar = config_ojos.umbral_ear_cerrar
        self.umbral_abrir = config_ojos.umbral_ear_abrir
        self.dur_max_parpadeo = config_ojos.dur_max_parpadeo_seg
        self.refractario = config_ojos.refractario_parpadeo_seg
        self.ventana_calculo_seg = ventana_calculo_seg

        self.ojos_cerrados = False
        self.ts_inicio_cierre = 0.0
        self.ts_ultimo_parpadeo = 0.0
        self.parpadeos: deque = deque()
        self.ts_arranque = 0.0

    def procesar(self, ear: Optional[float], timestamp: float) -> bool:
        """Devuelve True si en este frame se confirmo un parpadeo nuevo."""
        if self.ts_arranque == 0.0:
            self.ts_arranque = timestamp

        if ear is None:
            return False

        parpadeo_detectado = False

        if not self.ojos_cerrados and ear < self.umbral_cerrar:
            self.ojos_cerrados = True
            self.ts_inicio_cierre = timestamp

        elif self.ojos_cerrados and ear > self.umbral_abrir:
            duracion = timestamp - self.ts_inicio_cierre
            self.ojos_cerrados = False

            tiempo_desde_ultimo = timestamp - self.ts_ultimo_parpadeo
            if (
                duracion <= self.dur_max_parpadeo
                and tiempo_desde_ultimo >= self.refractario
            ):
                self.parpadeos.append(timestamp)
                self.ts_ultimo_parpadeo = timestamp
                parpadeo_detectado = True

        self._purgar_viejos(timestamp)
        return parpadeo_detectado

    def frecuencia_por_minuto(self, timestamp: float) -> Optional[float]:
        """
        Devuelve None durante los primeros 30 segundos.
        Despues devuelve parpadeos/min extrapolados.
        """
        if self.ts_arranque == 0.0:
            return None

        tiempo_transcurrido = timestamp - self.ts_arranque
        if tiempo_transcurrido < 30.0:
            return None

        ventana_efectiva = min(tiempo_transcurrido, self.ventana_calculo_seg)
        if ventana_efectiva <= 0:
            return None

        return len(self.parpadeos) * 60.0 / ventana_efectiva

    def _purgar_viejos(self, ahora: float) -> None:
        limite = ahora - self.ventana_calculo_seg
        while self.parpadeos and self.parpadeos[0] < limite:
            self.parpadeos.popleft()


class DetectorMicrosuenos:
    """
    Detecta cierres de ojos prolongados (>= dur_min_microsueno_seg).
    Emite microsueno=True UNA SOLA VEZ al terminar el cierre.
    """

    def __init__(self, config_ojos):
        self.umbral_cerrar = config_ojos.umbral_ear_cerrar
        self.umbral_abrir = config_ojos.umbral_ear_abrir
        self.dur_min_microsueno = config_ojos.dur_min_microsueno_seg

        self.ojos_cerrados = False
        self.ts_inicio_cierre = 0.0
        self.microsueno_en_curso = False

    def procesar(self, ear: Optional[float], timestamp: float) -> bool:
        """Devuelve True si TERMINA un microsueno confirmado en este frame."""
        if ear is None:
            return False

        if not self.ojos_cerrados and ear < self.umbral_cerrar:
            self.ojos_cerrados = True
            self.ts_inicio_cierre = timestamp
            self.microsueno_en_curso = False

        elif self.ojos_cerrados and ear < self.umbral_abrir:
            if not self.microsueno_en_curso:
                duracion = timestamp - self.ts_inicio_cierre
                if duracion >= self.dur_min_microsueno:
                    self.microsueno_en_curso = True
                    return True  # emite al CRUZAR el umbral (ojos aun cerrados)

        elif self.ojos_cerrados and ear > self.umbral_abrir:
            self.ojos_cerrados = False
            self.microsueno_en_curso = False

        return False


class DetectorBostezos:
    """
    Detecta bostezos via MAR sostenido.
    Emite bostezo=True UNA SOLA VEZ al terminar.
    Mantiene ventana larga (15 min) de bostezos confirmados.
    """

    def __init__(self, config_boca):
        self.umbral_bostezo = config_boca.umbral_mar_bostezo
        self.dur_min_bostezo = config_boca.dur_min_bostezo_seg
        self.ventana_larga_seg = config_boca.ventana_bostezos_seg

        self.boca_abierta = False
        self.ts_inicio_apertura = 0.0
        self.bostezo_en_curso = False
        self.bostezos_recientes: deque = deque()

    def procesar(self, mar: Optional[float], timestamp: float) -> bool:
        if mar is None:
            return False

        umbral_bajar = self.umbral_bostezo * 0.9

        if not self.boca_abierta and mar > self.umbral_bostezo:
            self.boca_abierta = True
            self.ts_inicio_apertura = timestamp
            self.bostezo_en_curso = False

        elif self.boca_abierta and mar > umbral_bajar:
            if not self.bostezo_en_curso:
                duracion = timestamp - self.ts_inicio_apertura
                if duracion >= self.dur_min_bostezo:
                    self.bostezo_en_curso = True

        elif self.boca_abierta and mar <= umbral_bajar:
            self.boca_abierta = False
            if self.bostezo_en_curso:
                self.bostezo_en_curso = False
                self.bostezos_recientes.append(timestamp)
                self._purgar_viejos(timestamp)
                return True

        return False

    def contar_ventana_larga(self, timestamp: float) -> int:
        self._purgar_viejos(timestamp)
        return len(self.bostezos_recientes)

    def _purgar_viejos(self, ahora: float) -> None:
        limite = ahora - self.ventana_larga_seg
        while self.bostezos_recientes and self.bostezos_recientes[0] < limite:
            self.bostezos_recientes.popleft()


class DetectorCabeceos:
    """
    Detecta cabeceos via pitch sostenido por encima del umbral.
    Solo cuenta si la cabeza no esta muy girada (yaw bajo).
    """

    def __init__(self, config_cabeza):
        self.umbral_pitch = config_cabeza.umbral_pitch_grados
        self.dur_min_cabeceo = config_cabeza.dur_min_cabeceo_seg
        self.umbral_yaw_max = config_cabeza.umbral_yaw_max_grados

        self.inclinado = False
        self.ts_inicio_inclinacion = 0.0
        self.cabeceo_en_curso = False

    def procesar(
        self,
        pitch: Optional[float],
        yaw: Optional[float],
        timestamp: float,
    ) -> bool:
        if pitch is None:
            return False

        if yaw is not None and abs(yaw) > self.umbral_yaw_max:
            if self.inclinado:
                self.inclinado = False
                self.cabeceo_en_curso = False
            return False

        if not self.inclinado and pitch > self.umbral_pitch:
            self.inclinado = True
            self.ts_inicio_inclinacion = timestamp
            self.cabeceo_en_curso = False

        elif self.inclinado and pitch > self.umbral_pitch * 0.85:
            if not self.cabeceo_en_curso:
                duracion = timestamp - self.ts_inicio_inclinacion
                if duracion >= self.dur_min_cabeceo:
                    self.cabeceo_en_curso = True
                    return True  # emite al CRUZAR el umbral (cabeza aun abajo)

        elif self.inclinado and pitch <= self.umbral_pitch * 0.85:
            self.inclinado = False
            self.cabeceo_en_curso = False

        return False


class VentanaPERCLOS:
    """
    Calcula PERCLOS: porcentaje de tiempo con ojos cerrados en una ventana.

    Algoritmo basado en tiempo (no en frames): para cada frame guardamos
    (timestamp, ojos_cerrados). Integramos el tiempo cerrado sobre la ventana
    asumiendo que el estado entre frames es el del frame anterior.

    Necesita 10s minimos de historia para ser confiable (devuelve None antes).
    """

    def __init__(self, config_ojos, ventana_seg: float = 60.0):
        self.umbral_cerrar = config_ojos.umbral_ear_cerrar
        self.umbral_abrir = config_ojos.umbral_ear_abrir
        self.ventana_seg = ventana_seg

        self.historial: deque = deque()
        self.ts_arranque = 0.0
        self.estado_actual_cerrado = False

    def procesar(self, ear: Optional[float], timestamp: float) -> None:
        if self.ts_arranque == 0.0:
            self.ts_arranque = timestamp

        if ear is None:
            self.historial.append((timestamp, self.estado_actual_cerrado))
        else:
            if self.estado_actual_cerrado:
                if ear > self.umbral_abrir:
                    self.estado_actual_cerrado = False
            else:
                if ear < self.umbral_cerrar:
                    self.estado_actual_cerrado = True
            self.historial.append((timestamp, self.estado_actual_cerrado))

        self._purgar_viejos(timestamp)

    def calcular(self, timestamp: float) -> Optional[float]:
        if self.ts_arranque == 0.0:
            return None
        if timestamp - self.ts_arranque < 10.0:
            return None
        if len(self.historial) < 2:
            return None

        inicio_ventana = timestamp - self.ventana_seg
        tiempo_cerrado = 0.0
        tiempo_total = 0.0

        items = list(self.historial)
        for i in range(len(items) - 1):
            ts1, cerrado1 = items[i]
            ts2, _ = items[i + 1]

            ts1_clip = max(ts1, inicio_ventana)
            if ts2 <= inicio_ventana:
                continue

            duracion = ts2 - ts1_clip
            if duracion <= 0:
                continue

            tiempo_total += duracion
            if cerrado1:
                tiempo_cerrado += duracion

        if tiempo_total <= 0:
            return None
        return tiempo_cerrado / tiempo_total

    def _purgar_viejos(self, ahora: float) -> None:
        limite = ahora - self.ventana_seg
        while len(self.historial) > 2 and self.historial[1][0] < limite:
            self.historial.popleft()


class ClasificadorBPM:
    """
    Clasifica BPM en NivelRiesgoBPM segun los umbrales del config.

    BPM = None      -> DESCONOCIDO
    BPM > normal_max o en rango normal -> NORMAL
    BPM < umbral_alerta -> ALERTA
    BPM < umbral_critico -> CRITICO
    """

    def __init__(self, config_wearable):
        self.normal_min = config_wearable.bpm_normal_min
        self.normal_max = config_wearable.bpm_normal_max
        self.umbral_alerta = config_wearable.bpm_umbral_alerta
        self.umbral_critico = config_wearable.bpm_umbral_critico
        self.bpm_actual: Optional[int] = None

    def actualizar(self, bpm: Optional[int]) -> None:
        if bpm is not None:
            self.bpm_actual = bpm

    def clasificar(self) -> NivelRiesgoBPM:
        if self.bpm_actual is None:
            return NivelRiesgoBPM.DESCONOCIDO
        if self.bpm_actual < self.umbral_critico:
            return NivelRiesgoBPM.CRITICO
        if self.bpm_actual < self.umbral_alerta:
            return NivelRiesgoBPM.ALERTA
        return NivelRiesgoBPM.NORMAL


class DetectorRostroPerdido:
    """
    Cuenta frames consecutivos sin rostro detectado.

    Escalada:
        1 a N frames -> ventana_no_confiable=True
        > N frames   -> vision_disponible=False

    N = config.vision.max_frames_sin_rostro (default 15)
    """

    def __init__(self, config_vision):
        self.max_frames_sin_rostro = config_vision.max_frames_sin_rostro
        self.frames_consecutivos_sin_rostro = 0

    def procesar(self, rostro_detectado: bool) -> Tuple[bool, bool]:
        """Devuelve (ventana_no_confiable, vision_disponible)."""
        if rostro_detectado:
            self.frames_consecutivos_sin_rostro = 0
            return (False, True)

        self.frames_consecutivos_sin_rostro += 1

        if self.frames_consecutivos_sin_rostro > self.max_frames_sin_rostro:
            return (True, False)
        return (True, True)


# =============================================================================
#                              CLASE PRINCIPAL
# =============================================================================


class PreFSM:
    """
    Evaluador de eventos primarios. Convierte Envelopes en EventoProcesado.

    Diseño sincronico: el caller invoca procesar() y obtiene el resultado.
    No tiene hilos internos. Mantiene estado entre invocaciones.
    """

    VENTANA_PERCLOS_SEG = 60.0
    VENTANA_PARPADEOS_SEG = 60.0

    def __init__(self, config: Config) -> None:
        self.config = config

        self.detector_parpadeos = DetectorParpadeos(
            config.ojos, ventana_calculo_seg=self.VENTANA_PARPADEOS_SEG
        )
        self.detector_microsuenos = DetectorMicrosuenos(config.ojos)
        self.detector_bostezos = DetectorBostezos(config.boca)
        self.detector_cabeceos = DetectorCabeceos(config.cabeza)
        self.ventana_perclos = VentanaPERCLOS(
            config.ojos, ventana_seg=self.VENTANA_PERCLOS_SEG
        )
        self.clasificador_bpm = ClasificadorBPM(config.wearable)
        self.detector_rostro_perdido = DetectorRostroPerdido(config.vision)

        self.vision_disponible = True
        self.wearable_disponible = True

        self.envelopes_procesados = 0
        self.eventos_procesados_emitidos = 0
        self.envelopes_ignorados = 0

    # ==================================================================
    # API PUBLICA
    # ==================================================================

    def procesar(self, envelope: Envelope) -> Optional[EventoProcesado]:
        """
        Procesa un Envelope y devuelve un EventoProcesado, o None si el
        envelope no debe generar uno (ACKs, eventos de salud).

        Los ACKs y los EventoFalloSensor/EventoRecuperacionSensor se
        devuelven via los helpers get_evento_ack() / get_evento_salud()
        para que el main loop los pase directamente a la FSM.
        """
        self.envelopes_procesados += 1
        evento = envelope.evento

        if isinstance(evento, EventoVision):
            return self._procesar_vision(evento)
        elif isinstance(evento, EventoWearable):
            return self._procesar_wearable(evento)
        elif isinstance(evento, EventoFalloSensor):
            self._procesar_fallo_sensor(evento)
            self.envelopes_ignorados += 1
            return None
        elif isinstance(evento, EventoRecuperacionSensor):
            self._procesar_recuperacion_sensor(evento)
            self.envelopes_ignorados += 1
            return None
        elif isinstance(evento, EventoAckWearable):
            self.envelopes_ignorados += 1
            return None
        else:
            _log.warning(
                "Pre-FSM no sabe procesar evento de tipo %s",
                type(evento).__name__,
            )
            self.envelopes_ignorados += 1
            return None

    def get_evento_ack(self, envelope: Envelope) -> Optional[EventoAckWearable]:
        """Helper: extrae el ACK si el envelope lo contiene, para pasarlo a la FSM."""
        if isinstance(envelope.evento, EventoAckWearable):
            return envelope.evento
        return None

    def get_evento_salud(self, envelope: Envelope):
        """Helper: extrae fallo/recuperacion de sensor, para pasarlos a la FSM."""
        if isinstance(envelope.evento, (EventoFalloSensor, EventoRecuperacionSensor)):
            return envelope.evento
        return None

    def estadisticas(self) -> dict:
        return {
            "envelopes_procesados": self.envelopes_procesados,
            "eventos_procesados_emitidos": self.eventos_procesados_emitidos,
            "envelopes_ignorados": self.envelopes_ignorados,
            "vision_disponible": self.vision_disponible,
            "wearable_disponible": self.wearable_disponible,
            "bpm_actual": self.clasificador_bpm.bpm_actual,
            "bostezos_ultimos_15min": len(self.detector_bostezos.bostezos_recientes),
        }

    # ==================================================================
    # PROCESAMIENTO INTERNO
    # ==================================================================

    def _procesar_vision(self, ev: EventoVision) -> EventoProcesado:
        ts = ev.timestamp

        # Rostro perdido
        if not ev.rostro_detectado:
            _, vision_disp = self.detector_rostro_perdido.procesar(False)
            return EventoProcesado(
                timestamp=ts,
                microsueno=False,
                bostezo=False,
                cabeceo=False,
                parpadeo=False,
                parpadeos_por_minuto=self.detector_parpadeos.frecuencia_por_minuto(ts),
                perclos=self.ventana_perclos.calcular(ts),
                bostezos_ventana_larga=self.detector_bostezos.contar_ventana_larga(ts),
                bpm_actual=self.clasificador_bpm.bpm_actual,
                nivel_riesgo_bpm=self.clasificador_bpm.clasificar(),
                ventana_no_confiable=True,
                motivo_no_confiable="rostro_no_detectado",
                vision_disponible=vision_disp and self.vision_disponible,
                wearable_disponible=self.wearable_disponible,
            )

        # Rostro detectado: reset del contador
        _, vision_disp = self.detector_rostro_perdido.procesar(True)

        ventana_no_confiable = ev.frote_ojos_activo
        motivo = "frote_ojos" if ev.frote_ojos_activo else ""

        ear_promedio = ev.ear_promedio

        if not ev.frote_ojos_activo:
            parpadeo = self.detector_parpadeos.procesar(ear_promedio, ts)
            microsueno = self.detector_microsuenos.procesar(ear_promedio, ts)
            self.ventana_perclos.procesar(ear_promedio, ts)
        else:
            parpadeo = False
            microsueno = False

        bostezo = self.detector_bostezos.procesar(ev.mar, ts)
        cabeceo = self.detector_cabeceos.procesar(ev.pitch_grados, ev.yaw_grados, ts)

        evento_proc = EventoProcesado(
            timestamp=ts,
            microsueno=microsueno,
            bostezo=bostezo,
            cabeceo=cabeceo,
            parpadeo=parpadeo,
            parpadeos_por_minuto=self.detector_parpadeos.frecuencia_por_minuto(ts),
            perclos=self.ventana_perclos.calcular(ts),
            bostezos_ventana_larga=self.detector_bostezos.contar_ventana_larga(ts),
            bpm_actual=self.clasificador_bpm.bpm_actual,
            nivel_riesgo_bpm=self.clasificador_bpm.clasificar(),
            ventana_no_confiable=ventana_no_confiable,
            motivo_no_confiable=motivo,
            vision_disponible=vision_disp and self.vision_disponible,
            wearable_disponible=self.wearable_disponible,
        )

        self.eventos_procesados_emitidos += 1
        return evento_proc

    def _procesar_wearable(self, ev: EventoWearable) -> EventoProcesado:
        self.clasificador_bpm.actualizar(ev.bpm)
        ts = ev.timestamp

        evento_proc = EventoProcesado(
            timestamp=ts,
            microsueno=False,
            bostezo=False,
            cabeceo=False,
            parpadeo=False,
            parpadeos_por_minuto=self.detector_parpadeos.frecuencia_por_minuto(ts),
            perclos=self.ventana_perclos.calcular(ts),
            bostezos_ventana_larga=self.detector_bostezos.contar_ventana_larga(ts),
            bpm_actual=self.clasificador_bpm.bpm_actual,
            nivel_riesgo_bpm=self.clasificador_bpm.clasificar(),
            ventana_no_confiable=False,
            motivo_no_confiable="",
            vision_disponible=self.vision_disponible,
            wearable_disponible=self.wearable_disponible,
        )

        self.eventos_procesados_emitidos += 1
        return evento_proc

    def _procesar_fallo_sensor(self, ev: EventoFalloSensor) -> None:
        if ev.sensor_afectado == OrigenEvento.VISION:
            self.vision_disponible = False
        elif ev.sensor_afectado == OrigenEvento.WEARABLE:
            self.wearable_disponible = False
            self.clasificador_bpm.bpm_actual = None

    def _procesar_recuperacion_sensor(self, ev: EventoRecuperacionSensor) -> None:
        if ev.sensor_recuperado == OrigenEvento.VISION:
            self.vision_disponible = True
        elif ev.sensor_recuperado == OrigenEvento.WEARABLE:
            self.wearable_disponible = True


__all__ = [
    "PreFSM",
    "DetectorParpadeos",
    "DetectorMicrosuenos",
    "DetectorBostezos",
    "DetectorCabeceos",
    "VentanaPERCLOS",
    "ClasificadorBPM",
    "DetectorRostroPerdido",
]
