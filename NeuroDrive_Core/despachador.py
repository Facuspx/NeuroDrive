"""
NeuroDrive Core - Despachador de Comandos
==========================================

Puente entre la logica pura de la FSM y los actuadores fisicos (buzzer,
wearable, voz). La FSM emite `SalidaFSM` con una tupla de `ComandoActuador`;
el Despachador se encarga de RUTEAR y EJECUTAR cada comando contra los
actuadores que lo soporten, sin bloquear el bucle principal del Core.

Lugar en la arquitectura (Tramo 2 de la planificacion):

    Gestor -> Pre-FSM -> FSM -> [Despachador] -> Actuadores
                                     |
                            hilo trabajador + queue.Queue

Decisiones de diseno:

  1. HILO TRABAJADOR UNICO. `despachar()` solo encola y retorna en
     microsegundos. Un hilo interno saca de la cola y ejecuta contra los
     actuadores. Asi el bucle de deteccion (Gestor->PreFSM->FSM) nunca
     queda rehen de un actuador lento (un aplay de 2s, un socket UDP con
     timeout). Es el mismo patron productor/consumidor del GestorEventos.

  2. RUTEO DECLARATIVO. El Despachador no tiene ifs por tipo de comando.
     Le pregunta a cada actuador registrado que tipos soporta
     (`tipos_soportados()`) y le manda lo que corresponda. Agregar un
     actuador nuevo (voz, wearable) es registrarlo, sin tocar esta clase.

  3. AISLAMIENTO DE ERRORES. Si un actuador lanza una excepcion, se captura,
     se loguea, se cuenta, y se sigue con los demas. Un buzzer roto jamas
     puede impedir que vibre la pulsera. El hilo trabajador no muere por
     un error de actuador.

  4. PRIORIDAD DE APAGAR_TODO. Cuando llega, se vacia la cola de comandos
     pendientes y se llama `apagar()` en todos los actuadores. Cubre el caso
     de bajar de CRITICO a PRE_ALERTA por un ACK correcto: la alarma
     encolada no debe sonar despues de que el conductor ya confirmo.

Contrato con los actuadores: interfaz `ActuadorBase`. Todo actuador
(buzzer, wearable, voz, simulado) la implementa. `ejecutar()` debe retornar
rapido (no bloqueante): si el comando tiene duracion, el actuador gestiona
su propio timing internamente.
"""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from common.contratos import (
    ComandoActuador,
    SalidaFSM,
    TipoComandoActuador,
)


# =============================================================================
#                        INTERFAZ DE ACTUADOR
# =============================================================================


class ActuadorBase(ABC):
    """
    Interfaz que todo actuador debe implementar para ser gobernado por el
    Despachador. Ejemplos concretos: ActuadorBuzzer (GPIO), ActuadorWearable
    (UDP al ESP32), ActuadorVoz (aplay), ActuadorSimulado (tests).

    Reglas del contrato:
      - `ejecutar(comando)` NO debe bloquear. Si el comando implica una
        duracion (vibrar 2s), el actuador arranca la accion y retorna;
        gestiona su propio timing con hilos/timers internos si hace falta.
      - `apagar()` corta cualquier actividad en curso de inmediato.
      - `iniciar()` / `detener()` abren y cierran recursos (GPIO, socket).
        Deben poder llamarse de forma idempotente.
    """

    #: Nombre legible para logs y estadisticas. Sobrescribir en subclases.
    nombre: str = "actuador"

    @abstractmethod
    def tipos_soportados(self) -> Set[TipoComandoActuador]:
        """Conjunto de tipos de comando que este actuador sabe ejecutar."""
        raise NotImplementedError

    @abstractmethod
    def ejecutar(self, comando: ComandoActuador) -> None:
        """Ejecuta un comando. DEBE retornar rapido (no bloqueante)."""
        raise NotImplementedError

    def apagar(self) -> None:
        """Corta toda actividad en curso. Default: no-op (sobrescribir)."""
        return None

    def iniciar(self) -> None:
        """Abre recursos (GPIO, socket). Default: no-op. Idempotente."""
        return None

    def detener(self) -> None:
        """Cierra recursos. Default: no-op. Idempotente."""
        return None


# =============================================================================
#                        ACTUADOR SIMULADO (para tests)
# =============================================================================


@dataclass
class _RegistroComando:
    """Un comando recibido por el actuador simulado, con su timestamp."""
    timestamp_recepcion: float
    comando: ComandoActuador


class ActuadorSimulado(ActuadorBase):
    """
    Actuador de prueba: no toca hardware, solo registra en memoria cada
    comando que recibe. Es la herramienta de test de TODOS los modulos de
    la Etapa IV (ruteo, APAGAR_TODO, aislamiento de errores, reemplazo).

    Parametros:
      nombre: identificador del actuador.
      tipos: que tipos declara soportar. Por defecto, TODOS.
      fallar_en: conjunto de tipos que, al ejecutarse, lanzan excepcion.
                 Sirve para testear el aislamiento de errores.
      colgar_ms: si > 0, ejecutar() duerme ese tiempo (simula actuador lento;
                 sirve para verificar que despachar() no bloquea igual).
    """

    def __init__(
        self,
        nombre: str = "simulado",
        tipos: Optional[Set[TipoComandoActuador]] = None,
        fallar_en: Optional[Set[TipoComandoActuador]] = None,
        colgar_ms: int = 0,
    ) -> None:
        self.nombre = nombre
        self._tipos = tipos if tipos is not None else set(TipoComandoActuador)
        self._fallar_en = fallar_en or set()
        self._colgar_ms = colgar_ms

        self.comandos_recibidos: List[_RegistroComando] = []
        self.veces_apagado: int = 0
        self.iniciado: bool = False
        self._lock = threading.Lock()

    def tipos_soportados(self) -> Set[TipoComandoActuador]:
        return set(self._tipos)

    def ejecutar(self, comando: ComandoActuador) -> None:
        if self._colgar_ms > 0:
            import time
            time.sleep(self._colgar_ms / 1000.0)
        if comando.tipo in self._fallar_en:
            raise RuntimeError(
                f"[{self.nombre}] fallo simulado ejecutando {comando.tipo.name}"
            )
        import time
        with self._lock:
            self.comandos_recibidos.append(
                _RegistroComando(time.monotonic(), comando)
            )

    def apagar(self) -> None:
        with self._lock:
            self.veces_apagado += 1

    def iniciar(self) -> None:
        self.iniciado = True

    def detener(self) -> None:
        self.iniciado = False

    # -- Helpers de inspeccion para los tests --

    def tipos_recibidos(self) -> List[TipoComandoActuador]:
        with self._lock:
            return [r.comando.tipo for r in self.comandos_recibidos]

    def cantidad_recibida(self) -> int:
        with self._lock:
            return len(self.comandos_recibidos)


# =============================================================================
#                        ESTADISTICAS
# =============================================================================


@dataclass
class EstadisticasDespachador:
    """Contadores observables del Despachador (para el resumen del main.py)."""
    salidas_recibidas: int = 0
    comandos_encolados: int = 0
    comandos_ejecutados: int = 0
    apagados_totales: int = 0
    cola_llena_descartes: int = 0
    errores_por_actuador: Dict[str, int] = field(default_factory=dict)

    def sumar_error(self, nombre_actuador: str) -> None:
        self.errores_por_actuador[nombre_actuador] = (
            self.errores_por_actuador.get(nombre_actuador, 0) + 1
        )


# =============================================================================
#                        DESPACHADOR
# =============================================================================


# Sentinela para pedirle al hilo trabajador que termine.
_SENTINELA_FIN = object()


class DespachadorComandos:
    """
    Rutea y ejecuta los ComandoActuador que emite la FSM.

    Uso tipico:
        desp = DespachadorComandos()
        desp.registrar_actuador(ActuadorBuzzer(pin=18))
        desp.registrar_actuador(ActuadorWearable(...))
        desp.iniciar()
        ...
        salida = fsm.procesar_evento(evento)
        desp.despachar(salida)      # no bloquea
        ...
        desp.detener()

    Es seguro registrar actuadores solo ANTES de iniciar(). Registrar en
    caliente no esta soportado (no lo necesitamos y complica el shutdown).
    """

    def __init__(
        self,
        capacidad_cola: int = 64,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if capacidad_cola < 1:
            raise ValueError(f"capacidad_cola debe ser >= 1: {capacidad_cola}")

        self.log = logger or logging.getLogger("NeuroDrive.Despachador")
        self._actuadores: List[ActuadorBase] = []
        self._cola: "queue.Queue" = queue.Queue(maxsize=capacidad_cola)
        self._hilo: Optional[threading.Thread] = None
        self._activo = False
        self._lock_estado = threading.Lock()
        self.stats = EstadisticasDespachador()

    # ------------------------------------------------------------------
    # REGISTRO Y CICLO DE VIDA
    # ------------------------------------------------------------------

    def registrar_actuador(self, actuador: ActuadorBase) -> None:
        """Registra un actuador. Llamar antes de iniciar()."""
        if self._activo:
            raise RuntimeError(
                "No se pueden registrar actuadores con el despachador activo"
            )
        if not isinstance(actuador, ActuadorBase):
            raise TypeError(
                f"El actuador debe heredar de ActuadorBase: {type(actuador).__name__}"
            )
        self._actuadores.append(actuador)
        self.log.info(
            "Actuador registrado: %s (soporta %d tipos)",
            actuador.nombre,
            len(actuador.tipos_soportados()),
        )

    def iniciar(self) -> None:
        """Abre los actuadores y arranca el hilo trabajador. Idempotente."""
        with self._lock_estado:
            if self._activo:
                self.log.warning("iniciar() llamado pero ya estaba activo")
                return
            self._activo = True

        # Iniciar cada actuador; si uno falla al abrir, lo dejamos registrado
        # igual (por ejemplo el buzzer sin conectar) pero lo avisamos.
        for act in self._actuadores:
            try:
                act.iniciar()
            except Exception as e:
                self.log.error(
                    "Actuador %s fallo al iniciar: %s (se sigue sin el)",
                    act.nombre, e,
                )
                self.stats.sumar_error(act.nombre)

        self._hilo = threading.Thread(
            target=self._bucle_trabajador,
            name="DespachadorTrabajador",
            daemon=True,
        )
        self._hilo.start()
        self.log.info(
            "Despachador iniciado con %d actuadores", len(self._actuadores)
        )

    def detener(self, timeout: float = 2.0) -> None:
        """Para el hilo trabajador y cierra los actuadores. Idempotente."""
        with self._lock_estado:
            if not self._activo:
                return
            self._activo = False

        # Despertar al hilo con el sentinela para que salga del get()
        try:
            self._cola.put_nowait(_SENTINELA_FIN)
        except queue.Full:
            # Cola llena: la vaciamos y reintentamos para no colgar el shutdown
            self._vaciar_cola()
            try:
                self._cola.put_nowait(_SENTINELA_FIN)
            except queue.Full:
                pass

        if self._hilo is not None:
            self._hilo.join(timeout=timeout)
            if self._hilo.is_alive():
                self.log.warning(
                    "El hilo trabajador no termino en %.1fs", timeout
                )
            self._hilo = None

        # Apagar y cerrar cada actuador
        for act in self._actuadores:
            try:
                act.apagar()
            except Exception as e:
                self.log.error("Actuador %s fallo al apagar: %s", act.nombre, e)
            try:
                act.detener()
            except Exception as e:
                self.log.error("Actuador %s fallo al detener: %s", act.nombre, e)

        self.log.info("Despachador detenido")

    # ------------------------------------------------------------------
    # API PUBLICA: DESPACHAR
    # ------------------------------------------------------------------

    def despachar(self, salida: SalidaFSM) -> None:
        """
        Encola los comandos de una SalidaFSM para ejecucion asincrona.

        NO bloquea: retorna apenas encola. Si algun comando es APAGAR_TODO,
        primero vacia la cola de pendientes (para que el apagado tenga
        prioridad) y luego encola el APAGAR_TODO, que nunca se descarta.
        """
        if not self._activo:
            self.log.warning(
                "despachar() llamado con el despachador detenido; se ignora"
            )
            return

        self.stats.salidas_recibidas += 1

        for comando in salida.comandos:
            if comando.tipo == TipoComandoActuador.APAGAR_TODO:
                # Prioridad: descartar lo pendiente y garantizar el encolado.
                descartados = self._vaciar_cola()
                if descartados:
                    self.log.debug(
                        "APAGAR_TODO: %d comandos pendientes descartados",
                        descartados,
                    )
                self._cola.put(comando)  # bloqueante minimo: cola recien vaciada
                self.stats.comandos_encolados += 1
            else:
                try:
                    self._cola.put_nowait(comando)
                    self.stats.comandos_encolados += 1
                except queue.Full:
                    self.stats.cola_llena_descartes += 1
                    self.log.warning(
                        "Cola del despachador llena; se descarta %s",
                        comando.tipo.name,
                    )

    # ------------------------------------------------------------------
    # HILO TRABAJADOR
    # ------------------------------------------------------------------

    def _bucle_trabajador(self) -> None:
        """Saca comandos de la cola y los rutea a los actuadores."""
        while True:
            item = self._cola.get()
            try:
                if item is _SENTINELA_FIN:
                    return
                comando: ComandoActuador = item
                if comando.tipo == TipoComandoActuador.APAGAR_TODO:
                    self._ejecutar_apagar_todo()
                else:
                    self._rutear_comando(comando)
            finally:
                self._cola.task_done()

    def _rutear_comando(self, comando: ComandoActuador) -> None:
        """Envia un comando a todos los actuadores que declaran soportarlo."""
        alguien_lo_tomo = False
        for act in self._actuadores:
            if comando.tipo in act.tipos_soportados():
                alguien_lo_tomo = True
                try:
                    act.ejecutar(comando)
                    self.stats.comandos_ejecutados += 1
                except Exception as e:
                    self.stats.sumar_error(act.nombre)
                    self.log.error(
                        "Actuador %s fallo ejecutando %s: %s",
                        act.nombre, comando.tipo.name, e,
                    )
        if not alguien_lo_tomo:
            self.log.warning(
                "Ningun actuador soporta el comando %s (se ignora)",
                comando.tipo.name,
            )

    def _ejecutar_apagar_todo(self) -> None:
        """Llama apagar() en TODOS los actuadores, soporten lo que soporten."""
        self.stats.apagados_totales += 1
        for act in self._actuadores:
            try:
                act.apagar()
            except Exception as e:
                self.stats.sumar_error(act.nombre)
                self.log.error("Actuador %s fallo al apagar: %s", act.nombre, e)

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    def _vaciar_cola(self) -> int:
        """Descarta todos los comandos pendientes. Devuelve cuantos saco."""
        descartados = 0
        while True:
            try:
                item = self._cola.get_nowait()
            except queue.Empty:
                break
            # Si sacamos el sentinela por accidente, lo devolvemos.
            if item is _SENTINELA_FIN:
                try:
                    self._cola.put_nowait(_SENTINELA_FIN)
                except queue.Full:
                    pass
                self._cola.task_done()
                break
            descartados += 1
            self._cola.task_done()
        return descartados

    def esperar_vaciado(self, timeout: Optional[float] = None) -> None:
        """
        Bloquea hasta que la cola se procese por completo. Util en tests
        para sincronizar antes de inspeccionar los actuadores.
        """
        # queue.Queue.join no acepta timeout; implementamos uno simple.
        import time
        inicio = time.monotonic()
        while self._cola.unfinished_tasks > 0:
            if timeout is not None and (time.monotonic() - inicio) > timeout:
                return
            time.sleep(0.005)
