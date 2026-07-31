"""
test_orquestador.py - Tests de cableado del Orquestador con stubs fieles.

    python -m NeuroDrive_Core.test_orquestador

Valida el RUTEO (la logica nueva): que los envelopes fluyan por
PreFSM -> FSM -> Despachador, que los ACK/fallo vayan directo a la FSM,
que los comandos lleguen a los actuadores, y que un error no tire el bucle.
El Gestor/PreFSM/FSM se stubbean; el Despachador y el ActuadorSimulado son
REALES (ya testeados).
"""
from __future__ import annotations

import sys
import traceback
from typing import List, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from common.contratos import (
    ComandoActuador, Envelope, EstadoFSM, EventoAckWearable, EventoFalloSensor,
    EventoProcesado, EventoVision, OrigenEvento, SalidaFSM, TipoComandoActuador,
    TipoMensaje,
)
from NeuroDrive_Core.despachador import DespachadorComandos, ActuadorSimulado
from NeuroDrive_Core.orquestador import Orquestador

C = TipoComandoActuador
_resultados = []

def _test(nombre):
    def wrap(func):
        try:
            func(); _resultados.append((nombre, True, "")); print(f"  [OK]  {nombre}")
        except AssertionError as e:
            _resultados.append((nombre, False, str(e))); print(f"  [FAIL] {nombre}: {e}")
        except Exception as e:
            _resultados.append((nombre, False, f"{type(e).__name__}: {e}"))
            print(f"  [ERROR] {nombre}: {e}"); traceback.print_exc()
        return func
    return wrap


# ---------------------------- STUBS FIELES ----------------------------------

class GestorStub:
    """Emite una lista predefinida de envelopes y luego se apaga."""
    def __init__(self, envelopes: List[Envelope]):
        self._cola = list(envelopes)
        self._iniciado = False
        self.detenido = False
    def iniciar(self): self._iniciado = True
    def detener(self): self.detenido = True; self._iniciado = False
    @property
    def activo(self): return self._iniciado and len(self._cola) > 0
    def obtener_evento(self, timeout=1.0):
        if self._cola:
            return self._cola.pop(0)
        return None


class PreFSMStub:
    """Convierte EventoVision/EventoProcesado en EventoProcesado; el resto None."""
    def __init__(self):
        self.vistos = []
    def procesar(self, envelope):
        ev = envelope.evento
        self.vistos.append(type(ev).__name__)
        if isinstance(ev, EventoProcesado):
            return ev
        if isinstance(ev, EventoVision):
            return EventoProcesado(timestamp=ev.timestamp)
        return None  # ACK, fallo, recuperacion


class FSMStub:
    """
    FSM de mentira con reglas simples para probar el ruteo:
      - EventoProcesado con microsueno=True -> transiciona a ALERTA_MEDIA + VIBRAR_FUERTE
      - EventoAckWearable correcto           -> transiciona a PRE_ALERTA + APAGAR_TODO
      - EventoFalloSensor                    -> transiciona a MODO_DEGRADADO + VIBRAR_LEVE
      - lo demas: sin transicion, sin comandos
    """
    def __init__(self):
        self.estado = EstadoFSM.NORMAL
        self.eventos_vistos = []
    def get_estado_actual(self): return self.estado
    def procesar_evento(self, evento):
        self.eventos_vistos.append(type(evento).__name__)
        anterior = self.estado
        comandos = ()
        motivo = ""
        if isinstance(evento, EventoProcesado) and evento.microsueno:
            self.estado = EstadoFSM.ALERTA_MEDIA
            comandos = (ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=80),
                        ComandoActuador(tipo=C.SECUENCIA_ACK, intensidad=80, id_secuencia=1))
            motivo = "microsueno"
        elif isinstance(evento, EventoAckWearable) and evento.secuencia_correcta:
            self.estado = EstadoFSM.PRE_ALERTA
            comandos = (ComandoActuador(tipo=C.APAGAR_TODO),)
            motivo = "ack_correcto"
        elif isinstance(evento, EventoFalloSensor):
            self.estado = EstadoFSM.MODO_DEGRADADO
            comandos = (ComandoActuador(tipo=C.VIBRAR_LEVE, intensidad=30),)
            motivo = "fallo_sensor"
        transicion = self.estado != anterior
        return SalidaFSM(
            timestamp=evento.timestamp, estado_actual=self.estado,
            estado_anterior=anterior, nivel_alerta=0, comandos=comandos,
            transicion_ocurrio=transicion, motivo_transicion=motivo,
        )


def _env(evento, tipo, seq=1):
    return Envelope(
        tipo=tipo, origen=OrigenEvento.WEARABLE, id_dispositivo="d",
        id_sesion="s", id_mensaje=f"m-{seq}", numero_secuencia=seq,
        timestamp_origen=evento.timestamp, evento=evento,
    )

def _orq(envelopes):
    gestor = GestorStub(envelopes)
    pre = PreFSMStub()
    fsm = FSMStub()
    act = ActuadorSimulado("todo")
    desp = DespachadorComandos()
    desp.registrar_actuador(act)
    orq = Orquestador(gestor, pre, fsm, desp)
    return orq, gestor, pre, fsm, act, desp


# ------------------------------ TESTS ---------------------------------------

print("\n--- Ciclo de vida y orden ---")

@_test("iniciar() arranca gestor, receptor y despachador")
def _():
    class RecStub:
        def __init__(self): self.on=False
        def iniciar(self): self.on=True
        def detener(self): self.on=False
    orq, gestor, pre, fsm, act, desp = _orq([])
    rec = RecStub(); orq.receptor = rec
    orq.iniciar()
    assert gestor._iniciado and rec.on and act.iniciado
    orq.detener()
    assert gestor.detenido and not rec.on

print("\n--- Ruteo vision -> FSM -> actuador ---")

@_test("EventoVision con microsueno dispara VIBRAR en el actuador")
def _():
    ep = EventoProcesado(timestamp=1000.0, microsueno=True)
    orq, gestor, pre, fsm, act, desp = _orq([_env(ep, TipoMensaje.EVENTO_VISION)])
    orq.iniciar()
    orq.procesar_uno()
    desp.esperar_vaciado(1.0)
    orq.detener()
    tipos = act.tipos_recibidos()
    assert C.VIBRAR_FUERTE in tipos, f"esperaba VIBRAR_FUERTE, hubo {tipos}"
    assert fsm.estado == EstadoFSM.ALERTA_MEDIA
    assert orq.stats.transiciones == 1

@_test("EventoProcesado normal no genera comandos")
def _():
    ep = EventoProcesado(timestamp=1000.0, microsueno=False)
    orq, gestor, pre, fsm, act, desp = _orq([_env(ep, TipoMensaje.EVENTO_PROCESADO)])
    orq.iniciar(); orq.procesar_uno(); desp.esperar_vaciado(1.0); orq.detener()
    assert act.cantidad_recibida() == 0
    assert orq.stats.transiciones == 0

print("\n--- Ruteo ACK directo a la FSM ---")

@_test("EventoAckWearable va directo a la FSM (PreFSM devuelve None)")
def _():
    ack = EventoAckWearable(timestamp=1000.0, id_secuencia=1,
                            secuencia_correcta=True, tiempo_respuesta_ms=900)
    orq, gestor, pre, fsm, act, desp = _orq([_env(ack, TipoMensaje.EVENTO_ACK_WEARABLE)])
    orq.iniciar(); orq.procesar_uno(); desp.esperar_vaciado(1.0); orq.detener()
    # El ACK llego a la FSM
    assert "EventoAckWearable" in fsm.eventos_vistos
    # y produjo APAGAR_TODO -> el actuador se apago
    assert act.veces_apagado >= 1
    assert fsm.estado == EstadoFSM.PRE_ALERTA

print("\n--- Ruteo fallo de sensor directo a la FSM ---")

@_test("EventoFalloSensor va directo a la FSM y transiciona a DEGRADADO")
def _():
    fallo = EventoFalloSensor(timestamp=1000.0, sensor_afectado=OrigenEvento.VISION,
                              motivo="heartbeat", severidad=2)
    orq, gestor, pre, fsm, act, desp = _orq([_env(fallo, TipoMensaje.FALLO_SENSOR)])
    orq.iniciar(); orq.procesar_uno(); desp.esperar_vaciado(1.0); orq.detener()
    assert "EventoFalloSensor" in fsm.eventos_vistos
    assert fsm.estado == EstadoFSM.MODO_DEGRADADO
    assert C.VIBRAR_LEVE in act.tipos_recibidos()

print("\n--- Secuencia completa y robustez ---")

@_test("correr() procesa una secuencia completa hasta que el gestor se apaga")
def _():
    ep_ms = EventoProcesado(timestamp=1000.0, microsueno=True)
    ack = EventoAckWearable(timestamp=1001.0, id_secuencia=1,
                            secuencia_correcta=True, tiempo_respuesta_ms=900)
    envs = [_env(ep_ms, TipoMensaje.EVENTO_VISION, 1),
            _env(ack, TipoMensaje.EVENTO_ACK_WEARABLE, 2)]
    orq, gestor, pre, fsm, act, desp = _orq(envs)
    orq.iniciar()
    orq.correr(periodo_resumen=0)
    desp.esperar_vaciado(1.0)
    orq.detener()
    assert orq.stats.envelopes == 2
    assert orq.stats.transiciones == 2   # microsueno y ack
    assert fsm.estado == EstadoFSM.PRE_ALERTA

@_test("un evento que rompe el PreFSM no tira el bucle")
def _():
    class PreFSMRoto(PreFSMStub):
        def procesar(self, envelope):
            raise RuntimeError("boom")
    ep = EventoProcesado(timestamp=1000.0, microsueno=True)
    gestor = GestorStub([_env(ep, TipoMensaje.EVENTO_VISION)])
    desp = DespachadorComandos(); desp.registrar_actuador(ActuadorSimulado("a"))
    orq = Orquestador(gestor, PreFSMRoto(), FSMStub(), desp)
    orq.iniciar()
    orq.procesar_uno()   # no debe lanzar
    orq.detener()
    assert orq.stats.errores_procesamiento == 1

@_test("detener() apaga el despachador aunque el receptor falle")
def _():
    class RecRoto:
        def iniciar(self): pass
        def detener(self): raise RuntimeError("no cierra")
    orq, gestor, pre, fsm, act, desp = _orq([])
    orq.receptor = RecRoto()
    orq.iniciar()
    orq.detener()  # no debe lanzar
    assert not act.iniciado  # el despachador se detuvo igual


def _resumen():
    print("\n" + "=" * 60)
    ok = sum(1 for _, p, _ in _resultados if p); total = len(_resultados)
    print(f"  Resumen: {ok}/{total} tests pasaron")
    if ok != total:
        for n, p, m in _resultados:
            if not p: print(f"    - {n}: {m}")
    else:
        print("  ORQUESTADOR OK")
    print("=" * 60)
    return 0 if ok == total else 1

if __name__ == "__main__":
    sys.exit(_resumen())
