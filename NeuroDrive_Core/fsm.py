"""
NeuroDrive Core - FSM (Maquina de Estados Finita)
==================================================

Logica pura de decision del nivel de somnolencia.

ENTRADA: EventoProcesado (del Pre-FSM), EventoAckWearable (respuesta del
         conductor) o EventoFalloSensor / EventoRecuperacionSensor (del Gestor)
SALIDA:  SalidaFSM con el estado resultante y la lista de ComandoActuador

Esta clase NO:
  - Lee colas ni archivos
  - Toca GPIO, audio ni red
  - Tiene temporizadores reales (usa el timestamp del evento como reloj)

Eso la hace determinista y verificable sin hardware.

Transiciones (S0..S5 = NORMAL, PRE_ALERTA, ALERTA_LEVE, ALERTA_MEDIA, CRITICO,
MODO_DEGRADADO):

  S0 -> S1   senales leves sostenidas (persistencia_senales_leves_seg), o
             cabeceo sin corroboracion ocular
  S0 -> S2   microsueno, o cabeceo corroborado por PERCLOS
  S1 -> S2   hecho confirmado: microsueno, cabeceo, parpados pesados
             sostenidos o acumulacion de bostezos
  S1 -> S0   tiempo_para_bajar_estado_seg sin hechos negativos
  S0/S1 -> S3  lo anterior, cuando ya hay fatiga recurrente
  S2 -> S3   hecho severo durante la alerta, respuesta incorrecta, o
             verificacion vencida sin respuesta
  S3 -> S4   cabeceo con BPM critico, respuesta incorrecta, o verificacion
             vencida sin respuesta
  S4 -> S4   verificacion vencida o respuesta incorrecta: se plantea otra
  S2/S3/S4 -> S1   respuesta correcta
  S3 -> S2, S4 -> S3   respuesta correcta pero lenta (baja un solo nivel)
  cualquiera -> S5   EventoFalloSensor con severidad >= 2
  S5 -> S0   se recuperaron todas las fuentes

Reglas adicionales:
  - Verificacion: el plazo es la ventana de respuesta que se envia a la
    pulsera mas el margen de presentacion (ver _timeout_ack).
  - Pausa por rostro perdido: con vision_disponible=False o ventana no
    confiable no se acumula tiempo "limpio" para el descenso.
  - Modo degradado: el sistema sigue vigilando con lo que tiene. Si la vision
    continua informando, un microsueno o un cabeceo corroborado disparan un
    aviso sonoro. No hay verificacion (puede faltar la pulsera para
    responderla) ni escalada.
  - Sesion previa: la FSM puede iniciarse en otro estado y con los episodios
    severos de la sesion anterior (ver persistencia_sesion.py).

Ordenes que la FSM emite y el prototipo no ejecuta:
  REPRODUCIR_VOZ y NOTIFICAR_SUPERVISOR son salidas previstas para una etapa
  futura (asistente de voz y aviso remoto desde la unidad de procesamiento).
  Se emiten para que la logica de decision ya este completa y verificada; el
  Despachador las contabiliza como ordenes sin destinatario.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union

from common.contratos import (
    ComandoActuador,
    EstadoFSM,
    EventoAckWearable,
    EventoFalloSensor,
    EventoProcesado,
    EventoRecuperacionSensor,
    NivelRiesgoBPM,
    OrigenEvento,
    SalidaFSM,
    TipoComandoActuador,
)


# Tipo union de los eventos que la FSM puede consumir
EventoEntradaFSM = Union[
    EventoProcesado,
    EventoAckWearable,
    EventoFalloSensor,
    EventoRecuperacionSensor,
]


# =============================================================================
#                    CONFIGURACION DE LA FSM (desde config.yaml)
# =============================================================================


@dataclass
class ConfigFSM:
    """
    Parametros de la FSM. En produccion se construye con desde_config(); los
    valores por defecto de aca son los que usan las pruebas.

    Todos los tiempos en segundos salvo indicacion.
    """
    # Tiempo sin hechos negativos para descender de PRE_ALERTA a NORMAL
    tiempo_para_bajar_estado_seg: float = 60.0

    # Ventana de respuesta a la verificacion, por nivel de alerta. Es el tiempo
    # que la PULSERA le da al conductor desde el ultimo pulso; viaja en la
    # orden SECUENCIA_ACK (campo duracion_ms).
    timeout_ack_leve_seg: float = 30.0
    timeout_ack_medio_seg: float = 20.0
    timeout_ack_critico_seg: float = 15.0
    # Lo que la FSM espera por encima de esa ventana antes de dar por vencida
    # la verificacion: el tiempo de los pulsos (hasta 2,0 s) mas el transito.
    margen_presentacion_desafio_seg: float = 2.5

    # Bostezos en la ventana larga (15 min) que confirman somnolencia
    max_bostezos_ventana_larga: int = 3

    # Senales leves continuas
    calentamiento_senales_seg: float = 60.0        # ignorar parpadeos/min al iniciar
    persistencia_senales_leves_seg: float = 20.0   # cuanto deben sostenerse
    parpadeos_por_minuto_alerta: float = 10.0      # por debajo: senal leve

    # PERCLOS (proporcion de tiempo con los ojos cerrados). Son TRES decisiones
    # distintas y por eso tres umbrales; no unificarlos.
    perclos_corroborar_cabeceo: float = 0.30   # >= : el cabeceo es de sueno, no una mirada al tablero
    perclos_senales_leves: float = 0.30        # >  : cuenta como senal leve continua
    perclos_parpados_pesados: float = 0.35     # >= sostenido: hecho confirmado
    perclos_parpados_pesados_sostenido_seg: float = 30.0

    # Fatiga recurrente
    max_eventos_severos_ventana: int = 3           # episodios para declararla
    ventana_episodios_seg: float = 900.0           # 15 min

    # Respuesta correcta pero lenta (ms): baja un solo nivel
    umbral_respuesta_lenta_ms: int = 5000

    @classmethod
    def desde_config(cls, config) -> "ConfigFSM":
        """
        Construye la ConfigFSM a partir del Config del sistema
        (NeuroDrive_Core.config_loader.Config). Unico lugar donde se decide
        de que seccion del config.yaml sale cada parametro.
        """
        fsm, wea = config.fsm, config.wearable
        return cls(
            tiempo_para_bajar_estado_seg=float(fsm.tiempo_para_bajar_estado_seg),
            timeout_ack_leve_seg=float(wea.timeout_ack_leve_seg),
            timeout_ack_medio_seg=float(wea.timeout_ack_medio_seg),
            timeout_ack_critico_seg=float(wea.timeout_ack_critico_seg),
            margen_presentacion_desafio_seg=float(wea.margen_presentacion_desafio_seg),
            max_bostezos_ventana_larga=int(config.boca.max_bostezos_ventana_larga),
            calentamiento_senales_seg=float(fsm.calentamiento_senales_seg),
            persistencia_senales_leves_seg=float(fsm.persistencia_senales_leves_seg),
            parpadeos_por_minuto_alerta=float(config.ojos.parpadeos_por_minuto_alerta),
            perclos_corroborar_cabeceo=float(fsm.perclos_corroborar_cabeceo),
            perclos_senales_leves=float(fsm.perclos_senales_leves),
            perclos_parpados_pesados=float(fsm.perclos_parpados_pesados),
            perclos_parpados_pesados_sostenido_seg=float(
                fsm.perclos_parpados_pesados_sostenido_seg
            ),
            max_eventos_severos_ventana=int(fsm.max_eventos_severos_ventana),
            ventana_episodios_seg=float(fsm.ventana_episodios_seg),
            umbral_respuesta_lenta_ms=int(fsm.umbral_respuesta_lenta_ms),
        )


# =============================================================================
#                     ESTADO INTERNO DE LA FSM (mutable)
# =============================================================================


@dataclass
class EstadoInternoFSM:
    """
    Estado mutable que la FSM mantiene entre eventos.

    Se separa del estado declarativo (EstadoFSM enum) para tener visibilidad
    completa sobre los contadores y marcas de tiempo.
    """
    estado_actual: EstadoFSM = EstadoFSM.NORMAL

    # Timestamp del ultimo evento procesado (sirve de "reloj")
    ultimo_timestamp: float = 0.0

    # Timestamp en el que se entro al estado actual
    timestamp_entrada_estado: float = 0.0

    # Acumulador de tiempo "limpio" en el estado actual (descuenta pausas)
    tiempo_acumulado_sin_eventos: float = 0.0
    ultimo_timestamp_evaluacion: float = 0.0

    # Verificacion pendiente (None = ninguna) y cuando se solicito
    id_secuencia_ack_pendiente: Optional[int] = None
    timestamp_ack_solicitado: float = 0.0

    # Fuentes con falla activa (set de OrigenEvento como int)
    sensores_caidos: set = field(default_factory=set)

    # Contador de verificaciones (genera los id_secuencia)
    contador_secuencia_ack: int = 0

    ts_primer_evento: float = 0.0
    ts_inicio_senales: float = 0.0
    ts_inicio_perclos_alto: float = 0.0

    # Marcas de tiempo de los episodios severos recientes (fatiga recurrente)
    episodios_severos: List[float] = field(default_factory=list)


# =============================================================================
#                              CLASE PRINCIPAL
# =============================================================================


class FSM:
    """
    Maquina de estados finita de NeuroDrive.

    Uso basico:
        fsm = FSM(ConfigFSM())
        salida = fsm.procesar_evento(evento)
        print(salida.estado_actual, salida.comandos)

    La FSM es determinista: el mismo orden de eventos produce siempre la misma
    secuencia de salidas. No tiene temporizadores: el tiempo se mide con el
    campo .timestamp de cada evento.
    """

    def __init__(
        self,
        config: ConfigFSM,
        estado_inicial: EstadoFSM = EstadoFSM.NORMAL,
        episodios_previos: Optional[Iterable[float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        estado_inicial    : estado de arranque (lo decide persistencia_sesion
                            cuando se retoma una sesion reciente).
        episodios_previos : marcas de tiempo de los episodios severos de la
                            sesion anterior, para que la fatiga recurrente no
                            se pierda con un reinicio.
        """
        self.config = config
        self.estado = EstadoInternoFSM(estado_actual=estado_inicial)
        if episodios_previos:
            self.estado.episodios_severos = sorted(float(t) for t in episodios_previos)
        self.log = logger or logging.getLogger("NeuroDrive.FSM")
        # Reemitir las ordenes del estado sin que haya transicion (nueva
        # verificacion en CRITICO).
        self._reemitir = False
        # Ordenes adicionales de este evento, haya o no transicion.
        self._comandos_extra: List[ComandoActuador] = []
        self._perclos_confirmado_flag = False

    # ------------------------------------------------------------------
    # API PUBLICA
    # ------------------------------------------------------------------

    def procesar_evento(self, evento: EventoEntradaFSM) -> SalidaFSM:
        """
        Procesa un evento y devuelve la salida de la FSM.

        Es el unico punto de entrada para hacer avanzar la maquina.
        """
        if evento.timestamp < self.estado.ultimo_timestamp:
            # Evento desordenado. Se procesa igual y se deja constancia.
            self.log.warning(
                "Evento con timestamp %.3f anterior al ultimo procesado %.3f",
                evento.timestamp,
                self.estado.ultimo_timestamp,
            )

        if self.estado.ts_primer_evento == 0.0:
            self.estado.ts_primer_evento = evento.timestamp

        estado_anterior = self.estado.estado_actual

        # Despachar segun tipo de evento
        motivo = ""
        if isinstance(evento, EventoFalloSensor):
            motivo = self._procesar_fallo_sensor(evento)
        elif isinstance(evento, EventoRecuperacionSensor):
            motivo = self._procesar_recuperacion_sensor(evento)
        elif isinstance(evento, EventoAckWearable):
            motivo = self._procesar_ack(evento)
        elif isinstance(evento, EventoProcesado):
            motivo = self._procesar_evento_normal(evento)
        else:
            self.log.error("Tipo de evento desconocido: %s", type(evento).__name__)
            motivo = "evento_desconocido"

        self.estado.ultimo_timestamp = evento.timestamp

        transicion = self.estado.estado_actual != estado_anterior
        if transicion:
            self.log.info(
                "Transicion %s -> %s (motivo: %s)",
                estado_anterior.name,
                self.estado.estado_actual.name,
                motivo,
            )
            self.estado.timestamp_entrada_estado = evento.timestamp
            self.estado.tiempo_acumulado_sin_eventos = 0.0
            self.estado.ultimo_timestamp_evaluacion = evento.timestamp

        # Las ordenes del estado se emiten al entrar en el, o cuando se pide
        # reemitirlas sin transicion (nueva verificacion en CRITICO, para no
        # dejar la alarma sin forma de confirmar). Las ordenes extra del
        # evento se agregan siempre.
        comandos: List[ComandoActuador] = []
        if transicion or self._reemitir:
            comandos = self._generar_comandos(evento)
        comandos = comandos + self._comandos_extra
        self._comandos_extra = []
        self._reemitir = False

        return SalidaFSM(
            timestamp=evento.timestamp,
            estado_actual=self.estado.estado_actual,
            estado_anterior=estado_anterior,
            nivel_alerta=self._nivel_alerta(self.estado.estado_actual),
            comandos=tuple(comandos),
            transicion_ocurrio=transicion,
            motivo_transicion=motivo if transicion else "",
            modo_degradado=(self.estado.estado_actual == EstadoFSM.MODO_DEGRADADO),
            motivo_degradacion=self._descripcion_degradacion(),
        )

    def get_estado_actual(self) -> EstadoFSM:
        """Acceso de solo lectura al estado actual."""
        return self.estado.estado_actual

    def get_estado_interno(self) -> EstadoInternoFSM:
        """Acceso al estado interno completo (para depuracion y pruebas)."""
        return self.estado

    def get_episodios_severos(self) -> List[float]:
        """Copia de las marcas de tiempo de los episodios severos (para persistir)."""
        return list(self.estado.episodios_severos)

    # ------------------------------------------------------------------
    # MANEJADORES POR TIPO DE EVENTO
    # ------------------------------------------------------------------

    def _procesar_fallo_sensor(self, ev: EventoFalloSensor) -> str:
        """Severidad >= 2 fuerza el paso a MODO_DEGRADADO."""
        self.estado.sensores_caidos.add(int(ev.sensor_afectado))
        if ev.severidad >= 2 and self.estado.estado_actual != EstadoFSM.MODO_DEGRADADO:
            self.estado.estado_actual = EstadoFSM.MODO_DEGRADADO
            return f"fallo_sensor:{ev.sensor_afectado.name}:{ev.motivo}"
        return ""

    def _procesar_recuperacion_sensor(self, ev: EventoRecuperacionSensor) -> str:
        """Cuando todas las fuentes se recuperaron se sale de MODO_DEGRADADO."""
        self.estado.sensores_caidos.discard(int(ev.sensor_recuperado))
        if (
            self.estado.estado_actual == EstadoFSM.MODO_DEGRADADO
            and not self.estado.sensores_caidos
        ):
            self.estado.estado_actual = EstadoFSM.NORMAL
            return f"recuperacion_sensor:{ev.sensor_recuperado.name}"
        return ""

    def _procesar_ack(self, ev: EventoAckWearable) -> str:
        """Respuesta correcta -> desciende. Respuesta incorrecta -> asciende."""
        if self.estado.id_secuencia_ack_pendiente is None:
            return ""  # No habia verificacion pendiente: se ignora

        if ev.id_secuencia != self.estado.id_secuencia_ack_pendiente:
            self.log.warning(
                "ACK con id_secuencia=%d, esperado %d (ignorado)",
                ev.id_secuencia,
                self.estado.id_secuencia_ack_pendiente,
            )
            return ""

        # Respuesta a la verificacion vigente
        self.estado.id_secuencia_ack_pendiente = None

        if not ev.secuencia_correcta:
            if self.estado.estado_actual == EstadoFSM.CRITICO:
                # No hay nivel mas alto: se queda en CRITICO y plantea otra
                # verificacion, para no dejar la alarma sin forma de confirmar.
                self._solicitar_ack(ev.timestamp)
                self._reemitir = True
                return "ack_incorrecto_recritico"
            # Asciende un nivel y plantea una verificacion nueva en el destino
            motivo = self._escalar_un_nivel("ack_incorrecto")
            self._solicitar_ack(ev.timestamp)
            return motivo

        # Respuesta correcta
        if self.estado.estado_actual in (
            EstadoFSM.ALERTA_LEVE,
            EstadoFSM.ALERTA_MEDIA,
            EstadoFSM.CRITICO,
        ):
            if ev.tiempo_respuesta_ms > self.config.umbral_respuesta_lenta_ms:
                # Correcta pero LENTA: signo de deterioro. Baja un solo nivel.
                self._bajar_un_nivel_por_ack()
                motivo = f"ack_correcto_lento:{ev.id_secuencia}"
            else:
                self.estado.estado_actual = EstadoFSM.PRE_ALERTA
                motivo = f"ack_correcto:{ev.id_secuencia}"
            if self._fatiga_recurrente(ev.timestamp):
                # Responde pero se sigue durmiendo: responder no es estar apto.
                # Ambas ordenes son salidas previstas sin destinatario en el
                # prototipo (ver docstring del modulo).
                self._comandos_extra = [
                    ComandoActuador(
                        tipo=TipoComandoActuador.REPRODUCIR_VOZ,
                        mensaje_voz="Fatiga recurrente detectada, detente a descansar",
                    ),
                    ComandoActuador(tipo=TipoComandoActuador.NOTIFICAR_SUPERVISOR),
                ]
                motivo += "_fatiga_recurrente"
            return motivo
        return ""

    def _procesar_evento_normal(self, ev: EventoProcesado) -> str:
        """Procesa un EventoProcesado del Pre-FSM segun el estado actual."""
        if self.estado.estado_actual == EstadoFSM.MODO_DEGRADADO:
            # De este estado se sale solo por recuperacion explicita, pero la
            # vigilancia continua con la informacion disponible.
            self._vigilar_en_degradado(ev)
            return ""

        # Histeresis de bajada: actualizar acumulador de tiempo sin eventos
        self._actualizar_acumulador_tiempo(ev)
        # PERCLOS sostenido (parpados pesados sin eventos discretos)
        self._actualizar_perclos_confirmado(ev)

        if self.estado.estado_actual == EstadoFSM.NORMAL:
            return self._desde_normal(ev)
        elif self.estado.estado_actual == EstadoFSM.PRE_ALERTA:
            return self._desde_pre_alerta(ev)
        elif self.estado.estado_actual == EstadoFSM.ALERTA_LEVE:
            return self._desde_alerta_leve(ev)
        elif self.estado.estado_actual == EstadoFSM.ALERTA_MEDIA:
            return self._desde_alerta_media(ev)
        elif self.estado.estado_actual == EstadoFSM.CRITICO:
            return self._desde_critico(ev)
        return ""

    # ------------------------------------------------------------------
    # TRANSICIONES POR ESTADO
    # ------------------------------------------------------------------

    def _desde_normal(self, ev: EventoProcesado) -> str:
        """S0 -> S2 por hecho severo corroborado; S0 -> S1 por cabeceo sin
        corroborar o por senales leves sostenidas."""
        if ev.ventana_no_confiable:
            return ""
        # Via rapida: microsueno siempre (ojos cerrados es inequivoco);
        # cabeceo solo con corroboracion ocular, para no confundir una mirada
        # al tablero con un cabeceo de sueno.
        if ev.microsueno or (ev.cabeceo and self._cabeceo_corroborado(ev)):
            recurrente = self._fatiga_recurrente(ev.timestamp)
            self._registrar_episodio(ev.timestamp)
            if recurrente:
                self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
                self._solicitar_ack(ev.timestamp)
                return "severo_fatiga_recurrente"
            self.estado.estado_actual = EstadoFSM.ALERTA_LEVE
            self._solicitar_ack(ev.timestamp)
            return "evento_severo_desde_normal"
        # Cabeceo sin corroboracion: senal discreta -> PRE_ALERTA inmediato
        if ev.cabeceo:
            self.estado.estado_actual = EstadoFSM.PRE_ALERTA
            self.estado.ts_inicio_senales = 0.0
            return "cabeceo_sin_corroboracion"
        # Senales continuas: requieren persistencia
        if self._senales_continuas(ev):
            if self.estado.ts_inicio_senales == 0.0:
                self.estado.ts_inicio_senales = ev.timestamp
            elif (
                ev.timestamp - self.estado.ts_inicio_senales
                >= self.config.persistencia_senales_leves_seg
            ):
                self.estado.estado_actual = EstadoFSM.PRE_ALERTA
                self.estado.ts_inicio_senales = 0.0
                return "senales_leves_sostenidas"
        else:
            self.estado.ts_inicio_senales = 0.0
        return ""

    def _desde_pre_alerta(self, ev: EventoProcesado) -> str:
        """S1 -> S0 por tiempo sin eventos; S1 -> S2/S3 por hecho confirmado."""
        # Hecho confirmado asciende; con fatiga recurrente, piso mas alto
        if self._hay_evento_confirmado(ev):
            recurrente = self._fatiga_recurrente(ev.timestamp)
            self._registrar_episodio(ev.timestamp)
            if recurrente:
                self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
                self._solicitar_ack(ev.timestamp)
                return "confirmado_fatiga_recurrente"
            self.estado.estado_actual = EstadoFSM.ALERTA_LEVE
            self._solicitar_ack(ev.timestamp)
            return "evento_confirmado"

        # Tiempo sin hechos negativos -> volver a NORMAL
        if (
            self.estado.tiempo_acumulado_sin_eventos
            >= self.config.tiempo_para_bajar_estado_seg
        ):
            self.estado.estado_actual = EstadoFSM.NORMAL
            return "tiempo_sin_eventos_completo"
        return ""

    def _desde_alerta_leve(self, ev: EventoProcesado) -> str:
        """S2 -> S3 por hecho severo durante la alerta o por verificacion vencida."""
        if ev.microsueno or self._hay_evento_confirmado(ev):
            self._registrar_episodio(ev.timestamp)
            self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
            self._solicitar_ack(ev.timestamp)
            return "evento_severo_en_alerta"

        if self._timeout_ack(ev.timestamp, self.config.timeout_ack_leve_seg):
            self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
            self._solicitar_ack(ev.timestamp)
            return "timeout_ack_leve"
        return ""

    def _desde_alerta_media(self, ev: EventoProcesado) -> str:
        """S3 -> S4 por cabeceo con BPM critico, o por verificacion vencida."""
        # Cabeceo + BPM critico = somnolencia confirmada por dos fuentes
        if (
            ev.cabeceo
            and ev.nivel_riesgo_bpm == NivelRiesgoBPM.CRITICO
        ):
            self.estado.estado_actual = EstadoFSM.CRITICO
            self._solicitar_ack(ev.timestamp)
            return "cabeceo_confirmado_bpm_critico"

        # Vencio el plazo: el conductor no responde la verificacion. La falta
        # de respuesta ya es senal de somnolencia, asi que asciende a CRITICO
        # sin importar el BPM (en la muneca suele ser normal o desconocido).
        if self._timeout_ack(ev.timestamp, self.config.timeout_ack_medio_seg):
            self.estado.estado_actual = EstadoFSM.CRITICO
            self._solicitar_ack(ev.timestamp)
            return "timeout_ack_medio_sin_respuesta"
        return ""

    def _desde_critico(self, ev: EventoProcesado) -> str:
        """S4 solo desciende por respuesta correcta (ver _procesar_ack)."""
        # Si la verificacion vencio se plantea OTRA: identificador nuevo y
        # reenvio de la orden a la pulsera, para que ambos extremos hablen de
        # la misma verificacion. Cambiar el identificador sin reenviar dejaba
        # la alarma trabada: la respuesta nunca coincidia con el id vigente.
        if self._timeout_ack(ev.timestamp, self.config.timeout_ack_critico_seg):
            self._solicitar_ack(ev.timestamp)
            self._reemitir = True
        return ""

    def _vigilar_en_degradado(self, ev: EventoProcesado) -> None:
        """
        Vigilancia reducida en MODO_DEGRADADO.

        Si la vision sigue informando (la fuente caida es la pulsera), un
        microsueno o un cabeceo corroborado disparan un aviso sonoro. No se
        plantea verificacion, porque puede no haber pulsera para responderla, y
        no hay escalada: sin la respuesta del conductor no hay con que decidir
        el descenso. El estado no cambia.

        Con la vision caida no llegan hechos del rostro y este metodo no tiene
        nada que evaluar: el pulso por si solo no dispara alertas.
        """
        if ev.ventana_no_confiable or not ev.vision_disponible:
            return
        if ev.microsueno or (ev.cabeceo and self._cabeceo_corroborado(ev)):
            self._registrar_episodio(ev.timestamp)
            self._comandos_extra = [
                ComandoActuador(
                    tipo=TipoComandoActuador.BUZZER_LARGO,
                    intensidad=70,
                    duracion_ms=2000,
                ),
            ]
            self.log.warning(
                "Hecho severo en MODO_DEGRADADO (%s): aviso sonoro sin verificacion",
                "microsueno" if ev.microsueno else "cabeceo",
            )

    # ------------------------------------------------------------------
    # PREDICADOS DE EVENTOS
    # ------------------------------------------------------------------

    def _cabeceo_corroborado(self, ev: EventoProcesado) -> bool:
        """El estado de los ojos respalda que el cabeceo es de sueno."""
        return (
            ev.perclos is not None
            and ev.perclos >= self.config.perclos_corroborar_cabeceo
        )

    def _senales_continuas(self, ev: EventoProcesado) -> bool:
        """Senales leves CONTINUAS (metricas de ventana). Requieren
        persistencia antes de declarar PRE_ALERTA. Los parpadeos/min se
        ignoran durante el calentamiento (ventana incompleta al iniciar)."""
        if ev.ventana_no_confiable:
            return False
        en_calentamiento = (
            ev.timestamp - self.estado.ts_primer_evento
            < self.config.calentamiento_senales_seg
        )
        if (
            not en_calentamiento
            and ev.parpadeos_por_minuto is not None
            and ev.parpadeos_por_minuto < self.config.parpadeos_por_minuto_alerta
        ):
            return True
        if ev.nivel_riesgo_bpm == NivelRiesgoBPM.ALERTA:
            return True
        if ev.perclos is not None and ev.perclos > self.config.perclos_senales_leves:
            return True
        return False

    def _actualizar_perclos_confirmado(self, ev: EventoProcesado) -> None:
        """Sigue el PERCLOS sostenido sobre el umbral de parpados pesados
        (somnolencia que nunca llega a producir un hecho discreto)."""
        if (
            ev.ventana_no_confiable
            or ev.perclos is None
            or ev.perclos < self.config.perclos_parpados_pesados
        ):
            self.estado.ts_inicio_perclos_alto = 0.0
            self._perclos_confirmado_flag = False
            return
        if self.estado.ts_inicio_perclos_alto == 0.0:
            self.estado.ts_inicio_perclos_alto = ev.timestamp
        self._perclos_confirmado_flag = (
            ev.timestamp - self.estado.ts_inicio_perclos_alto
            >= self.config.perclos_parpados_pesados_sostenido_seg
        )

    def _registrar_episodio(self, ts: float) -> None:
        self.estado.episodios_severos.append(ts)
        self._purgar_episodios(ts)

    def _purgar_episodios(self, ts: float) -> None:
        limite = ts - self.config.ventana_episodios_seg
        self.estado.episodios_severos = [
            t for t in self.estado.episodios_severos if t >= limite
        ]

    def _fatiga_recurrente(self, ts: float) -> bool:
        """True si YA hay N episodios severos en la ventana larga."""
        self._purgar_episodios(ts)
        return (
            len(self.estado.episodios_severos)
            >= self.config.max_eventos_severos_ventana
        )

    def _bajar_un_nivel_por_ack(self) -> None:
        e = self.estado.estado_actual
        if e == EstadoFSM.CRITICO:
            self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
        elif e == EstadoFSM.ALERTA_MEDIA:
            self.estado.estado_actual = EstadoFSM.ALERTA_LEVE
        elif e == EstadoFSM.ALERTA_LEVE:
            self.estado.estado_actual = EstadoFSM.PRE_ALERTA

    def _hay_evento_confirmado(self, ev: EventoProcesado) -> bool:
        """Para ascender de S1 a S2: hecho discreto severo."""
        if ev.ventana_no_confiable:
            return False
        if ev.microsueno:
            return True
        # Un bostezo aislado no confirma (es fisiologia normal); solo su
        # acumulacion en la ventana larga (abajo).
        if ev.cabeceo:
            return True
        # PERCLOS sostenido: somnolencia de parpados pesados
        if self._perclos_confirmado_flag:
            return True
        # Acumulacion de bostezos
        if ev.bostezos_ventana_larga >= self.config.max_bostezos_ventana_larga:
            return True
        return False

    # ------------------------------------------------------------------
    # TIEMPOS Y VERIFICACION
    # ------------------------------------------------------------------

    def _actualizar_acumulador_tiempo(self, ev: EventoProcesado) -> None:
        """
        Suma al acumulador "sin eventos" el tiempo transcurrido desde el
        evento anterior, pero solo si la vision esta disponible (pausa por
        rostro perdido) y este evento no trae hechos negativos.

        Si llega un hecho negativo, el acumulador vuelve a cero.
        """
        if self.estado.ultimo_timestamp_evaluacion == 0.0:
            self.estado.ultimo_timestamp_evaluacion = ev.timestamp
            return

        delta = ev.timestamp - self.estado.ultimo_timestamp_evaluacion
        if delta < 0:
            delta = 0.0

        # Reinicio si hay hecho negativo
        if (
            ev.microsueno or ev.bostezo or ev.cabeceo
            or ev.nivel_riesgo_bpm in (NivelRiesgoBPM.ALERTA, NivelRiesgoBPM.CRITICO)
        ):
            self.estado.tiempo_acumulado_sin_eventos = 0.0
            self.estado.ultimo_timestamp_evaluacion = ev.timestamp
            return

        # Pausa por fuente caida o ventana no confiable: no se acumula, pero
        # se actualiza la referencia.
        if not ev.vision_disponible or ev.ventana_no_confiable:
            self.estado.ultimo_timestamp_evaluacion = ev.timestamp
            return

        # Tiempo limpio: se acumula
        self.estado.tiempo_acumulado_sin_eventos += delta
        self.estado.ultimo_timestamp_evaluacion = ev.timestamp

    def _solicitar_ack(self, timestamp: float) -> None:
        """Genera un id_secuencia nuevo y arranca el plazo de la verificacion."""
        self.estado.contador_secuencia_ack += 1
        self.estado.id_secuencia_ack_pendiente = self.estado.contador_secuencia_ack
        self.estado.timestamp_ack_solicitado = timestamp

    def _timeout_ack(self, timestamp_actual: float, ventana_seg: float) -> bool:
        """
        True si la verificacion pendiente vencio sin respuesta.

        `ventana_seg` es la ventana de respuesta que se le envio a la pulsera.
        La FSM espera esa ventana MAS el margen de presentacion: la pulsera
        abre la ventana recien al terminar los pulsos, de modo que una
        respuesta que ella todavia acepta nunca llega con el plazo vencido.
        """
        if self.estado.id_secuencia_ack_pendiente is None:
            return False
        transcurrido = timestamp_actual - self.estado.timestamp_ack_solicitado
        return transcurrido >= ventana_seg + self.config.margen_presentacion_desafio_seg

    def _escalar_un_nivel(self, motivo: str) -> str:
        """Asciende un nivel desde donde este (respuesta incorrecta)."""
        if self.estado.estado_actual == EstadoFSM.ALERTA_LEVE:
            self.estado.estado_actual = EstadoFSM.ALERTA_MEDIA
            return motivo
        if self.estado.estado_actual == EstadoFSM.ALERTA_MEDIA:
            self.estado.estado_actual = EstadoFSM.CRITICO
            return motivo
        return ""

    # ------------------------------------------------------------------
    # GENERACION DE ORDENES
    # ------------------------------------------------------------------

    def _asegurar_ack_pendiente(self, timestamp: float) -> Optional[int]:
        """Garantiza un id de verificacion pendiente al emitir un desafio. Si
        no hay ninguno, solicita uno nuevo. Devuelve el id vigente."""
        if self.estado.id_secuencia_ack_pendiente is None:
            self._solicitar_ack(timestamp)
        return self.estado.id_secuencia_ack_pendiente

    def _orden_verificacion(self, timestamp: float, ventana_seg: float) -> ComandoActuador:
        """
        Orden SECUENCIA_ACK. Es la UNICA orden que recibe la pulsera en una
        alerta: sortea K entre 1 y 4, emite K pulsos y espera el toque en el
        pad K durante `ventana_seg`, valor que viaja en duracion_ms.

        No se emiten ordenes VIBRAR_* junto con el desafio. El desafio toma el
        motor en cuanto llega, de modo que una vibracion previa quedaria
        absorbida en el primer pulso; y si se la hiciera durar, alteraria la
        cuenta que el conductor debe hacer.
        """
        return ComandoActuador(
            tipo=TipoComandoActuador.SECUENCIA_ACK,
            duracion_ms=int(ventana_seg * 1000),
            id_secuencia=self._asegurar_ack_pendiente(timestamp),
        )

    def _generar_comandos(self, evento: EventoEntradaFSM) -> List[ComandoActuador]:
        """
        Ordenes propias del estado actual. Se invoca al entrar al estado, o al
        reemitir en CRITICO.

          - S2, S3, S4: aviso sonoro + verificacion + voz (+ supervisor en S4).
            La diferencia entre niveles esta en el aviso sonoro (corto, largo,
            continuo) y en la ventana de respuesta.
          - S0, S1, S5: APAGAR_TODO, que silencia todos los actuadores.
        """
        estado = self.estado.estado_actual
        ts = evento.timestamp

        if estado in (EstadoFSM.NORMAL, EstadoFSM.PRE_ALERTA, EstadoFSM.MODO_DEGRADADO):
            # PRE_ALERTA no produce nada perceptible: solo se observa.
            return [ComandoActuador(tipo=TipoComandoActuador.APAGAR_TODO)]

        if estado == EstadoFSM.ALERTA_LEVE:
            return [
                ComandoActuador(
                    tipo=TipoComandoActuador.BUZZER_CORTO,
                    intensidad=50,
                    duracion_ms=200,
                ),
                self._orden_verificacion(ts, self.config.timeout_ack_leve_seg),
                ComandoActuador(
                    tipo=TipoComandoActuador.REPRODUCIR_VOZ,
                    mensaje_voz="Atencion, signos de fatiga detectados",
                ),
            ]

        if estado == EstadoFSM.ALERTA_MEDIA:
            return [
                ComandoActuador(
                    tipo=TipoComandoActuador.BUZZER_LARGO,
                    intensidad=70,
                    duracion_ms=2000,
                ),
                self._orden_verificacion(ts, self.config.timeout_ack_medio_seg),
                ComandoActuador(
                    tipo=TipoComandoActuador.REPRODUCIR_VOZ,
                    mensaje_voz="Alerta, por favor confirme en la pulsera",
                ),
            ]

        if estado == EstadoFSM.CRITICO:
            return [
                ComandoActuador(
                    tipo=TipoComandoActuador.BUZZER_CONTINUO,
                    intensidad=100,
                ),
                self._orden_verificacion(ts, self.config.timeout_ack_critico_seg),
                ComandoActuador(
                    tipo=TipoComandoActuador.REPRODUCIR_VOZ,
                    mensaje_voz="Detente ahora, peligro de somnolencia",
                ),
                ComandoActuador(
                    tipo=TipoComandoActuador.NOTIFICAR_SUPERVISOR,
                ),
            ]

        return []

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    @staticmethod
    def _nivel_alerta(estado: EstadoFSM) -> int:
        """Nivel de alerta numerico 0-4. MODO_DEGRADADO no es un nivel de
        somnolencia y se informa como 0."""
        if estado == EstadoFSM.MODO_DEGRADADO:
            return 0
        return int(estado)

    def _descripcion_degradacion(self) -> str:
        """Cadena descriptiva del modo degradado para el registro."""
        if not self.estado.sensores_caidos:
            return ""
        nombres = [OrigenEvento(s).name for s in sorted(self.estado.sensores_caidos)]
        return f"sensores_caidos:{','.join(nombres)}"


__all__ = ["FSM", "ConfigFSM", "EstadoInternoFSM", "EventoEntradaFSM"]
