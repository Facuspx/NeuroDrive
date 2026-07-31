"""
test_wearable.py - Tests del lado Python del wearable.

    python -m NeuroDrive_Wearable.test_wearable

Cubre: protocolo, ActuadorWearable (transporte falso), ReceptorWearable
(publicar_fn inyectada), y END-TO-END sobre localhost con cola POSIX real:
actuador -> UDP -> simulador -> UDP -> receptor -> cola POSIX -> lectura.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from common.contratos import (
    ComandoActuador, EstadoFSM, Envelope, EventoAckWearable, EventoWearable,
    SalidaFSM, TipoComandoActuador, TipoMensaje,
)
from NeuroDrive_Core.despachador import DespachadorComandos
from NeuroDrive_Wearable import protocolo
from NeuroDrive_Wearable.actuador_wearable import ActuadorWearable
from NeuroDrive_Wearable.receptor_wearable import ReceptorWearable
from NeuroDrive_Wearable.simulador_pulsera import SimuladorPulsera

C = TipoComandoActuador
_resultados = []


def _test(nombre):
    def wrap(func):
        try:
            func()
            _resultados.append((nombre, True, ""))
            print(f"  [OK]  {nombre}")
        except AssertionError as e:
            _resultados.append((nombre, False, str(e)))
            print(f"  [FAIL] {nombre}: {e}")
        except Exception as e:
            _resultados.append((nombre, False, f"{type(e).__name__}: {e}"))
            print(f"  [ERROR] {nombre}: {e}")
            traceback.print_exc()
        return func
    return wrap


class _TransporteFalso:
    """Registra los datagramas 'enviados' sin tocar la red."""
    def __init__(self):
        self.enviados = []
        self._lock = threading.Lock()
    def sendto(self, datos):
        with self._lock:
            self.enviados.append(datos)
    def close(self):
        pass
    def paquetes(self):
        with self._lock:
            return list(self.enviados)


def _puerto_libre():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# =============================================================================
print("\n--- Protocolo ---")

@_test("serializar/parsear comando round-trip")
def _():
    c = ComandoActuador(tipo=C.SECUENCIA_ACK, intensidad=80, id_secuencia=5)
    datos = protocolo.serializar_comando(c, id_paquete=7)
    obj = protocolo.parsear_comando(datos)
    assert obj["tipo"] == int(C.SECUENCIA_ACK)
    assert obj["id_secuencia"] == 5
    assert obj["id_paquete"] == 7

@_test("telemetria -> EventoWearable")
def _():
    datos = protocolo.serializar_telemetria(bpm=72, bateria=88, id_paquete=1)
    crudo = protocolo.parsear_mensaje_pulsera(datos)
    ev = protocolo.construir_evento(crudo, timestamp=1000.0)
    assert isinstance(ev, EventoWearable)
    assert ev.bpm == 72 and ev.bateria_porcentaje == 88

@_test("telemetria con bpm None es valida")
def _():
    datos = protocolo.serializar_telemetria(bpm=None, id_paquete=2)
    ev = protocolo.construir_evento(protocolo.parsear_mensaje_pulsera(datos), 1000.0)
    assert ev.bpm is None

@_test("ack -> EventoAckWearable")
def _():
    datos = protocolo.serializar_ack(id_secuencia=3, secuencia_correcta=True, tiempo_respuesta_ms=1500)
    ev = protocolo.construir_evento(protocolo.parsear_mensaje_pulsera(datos), 1000.0)
    assert isinstance(ev, EventoAckWearable)
    assert ev.id_secuencia == 3 and ev.secuencia_correcta is True

@_test("JSON basura lanza ErrorProtocolo")
def _():
    try:
        protocolo.parsear_mensaje_pulsera(b"no soy json {{{")
        assert False, "deberia fallar"
    except protocolo.ErrorProtocolo:
        pass


# =============================================================================
print("\n--- ActuadorWearable (transporte falso) ---")

@_test("tipos_soportados son los de vibracion + SECUENCIA_ACK")
def _():
    a = ActuadorWearable(transporte=_TransporteFalso())
    assert a.tipos_soportados() == {C.VIBRAR_LEVE, C.VIBRAR_MEDIO, C.VIBRAR_FUERTE, C.SECUENCIA_ACK}

@_test("un VIBRAR_LEVE (no critico) se envia una sola vez")
def _():
    t = _TransporteFalso()
    a = ActuadorWearable(transporte=t)
    a.iniciar()
    a.ejecutar(ComandoActuador(tipo=C.VIBRAR_LEVE, intensidad=30))
    a.esperar_vaciado(1.0)
    a.detener()
    import json
    vibr = [x for x in map(json.loads, t.paquetes()) if x["tipo"] == int(C.VIBRAR_LEVE)]
    assert len(vibr) == 1, f"esperaba 1 envio de vibrar, hubo {len(vibr)}"

@_test("SECUENCIA_ACK (critico) se reenvia 3 veces con mismo id_paquete")
def _():
    t = _TransporteFalso()
    a = ActuadorWearable(transporte=t, reenvios_criticos=3, espaciado_reenvios_ms=5)
    a.iniciar()
    a.ejecutar(ComandoActuador(tipo=C.SECUENCIA_ACK, intensidad=80, id_secuencia=9))
    a.esperar_vaciado(1.0)
    time.sleep(0.05)
    a.detener()
    import json
    acks = [json.loads(p) for p in t.paquetes() if b'"tipo":8' in p]
    assert len(acks) == 3, f"esperaba 3 reenvios, hubo {len(acks)}"
    ids = {x["id_paquete"] for x in acks}
    assert len(ids) == 1, f"los reenvios deben compartir id_paquete: {ids}"
    assert all(x["id_secuencia"] == 9 for x in acks)

@_test("ejecutar() no bloquea aunque el envio tenga espaciado")
def _():
    t = _TransporteFalso()
    a = ActuadorWearable(transporte=t, reenvios_criticos=3, espaciado_reenvios_ms=100)
    a.iniciar()
    t0 = time.monotonic()
    a.ejecutar(ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=100))
    dt = time.monotonic() - t0
    a.detener()
    assert dt < 0.05, f"ejecutar bloqueo {dt*1000:.0f}ms"

@_test("apagar() envia un comando APAGAR_TODO al ESP")
def _():
    t = _TransporteFalso()
    a = ActuadorWearable(transporte=t)
    a.iniciar()
    a.apagar()
    a.detener()
    apagados = [p for p in t.paquetes() if b'"tipo":10' in p]
    assert len(apagados) >= 1, "deberia haber al menos un APAGAR_TODO"

@_test("integracion con el despachador: VIBRAR llega al wearable")
def _():
    t = _TransporteFalso()
    a = ActuadorWearable(transporte=t)
    d = DespachadorComandos()
    d.registrar_actuador(a)
    d.iniciar()
    salida = SalidaFSM(
        timestamp=1.0, estado_actual=EstadoFSM.ALERTA_LEVE,
        estado_anterior=EstadoFSM.PRE_ALERTA, nivel_alerta=2,
        comandos=(ComandoActuador(tipo=C.VIBRAR_LEVE, intensidad=30),),
        transicion_ocurrio=True,
    )
    d.despachar(salida)
    d.esperar_vaciado(1.0)
    a.esperar_vaciado(1.0)
    d.detener()
    import json
    vibr = [x for x in map(json.loads, t.paquetes()) if x["tipo"] == int(C.VIBRAR_LEVE)]
    assert len(vibr) >= 1


# =============================================================================
print("\n--- ReceptorWearable (publicar_fn inyectada) ---")

@_test("telemetria recibida -> Envelope EVENTO_WEARABLE publicado")
def _():
    capturados = []
    r = ReceptorWearable(puerto_escucha=_puerto_libre(),
                         publicar_fn=lambda j: (capturados.append(j) or True))
    r.iniciar()
    # mandamos un datagrama a mano
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(protocolo.serializar_telemetria(bpm=70, id_paquete=1),
             ("127.0.0.1", r._puerto))
    time.sleep(0.3)
    r.detener()
    s.close()
    assert len(capturados) == 1, f"esperaba 1 publicado, hubo {len(capturados)}"
    env = Envelope.from_json(capturados[0])
    assert env.tipo == TipoMensaje.EVENTO_WEARABLE
    ev = env.desempacar()
    assert ev.bpm == 70

@_test("ack recibido -> Envelope EVENTO_ACK_WEARABLE publicado")
def _():
    capturados = []
    r = ReceptorWearable(puerto_escucha=_puerto_libre(),
                         publicar_fn=lambda j: (capturados.append(j) or True))
    r.iniciar()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(protocolo.serializar_ack(id_secuencia=4, secuencia_correcta=False, tiempo_respuesta_ms=900),
             ("127.0.0.1", r._puerto))
    time.sleep(0.3)
    r.detener()
    s.close()
    assert len(capturados) == 1
    env = Envelope.from_json(capturados[0])
    assert env.tipo == TipoMensaje.EVENTO_ACK_WEARABLE
    ev = env.desempacar()
    assert ev.id_secuencia == 4 and ev.secuencia_correcta is False

@_test("datagrama basura se cuenta como invalido y no rompe el hilo")
def _():
    r = ReceptorWearable(puerto_escucha=_puerto_libre(),
                         publicar_fn=lambda j: True)
    r.iniciar()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"basura no json", ("127.0.0.1", r._puerto))
    s.sendto(protocolo.serializar_telemetria(bpm=80, id_paquete=1), ("127.0.0.1", r._puerto))
    time.sleep(0.3)
    r.detener()
    s.close()
    assert r.stats.invalidos == 1
    assert r.stats.telemetrias == 1


# =============================================================================
print("\n--- END-TO-END (localhost + cola POSIX real) ---")

def _limpiar_cola(nombre):
    try:
        import posix_ipc
        posix_ipc.unlink_message_queue(nombre)
    except Exception:
        pass

@_test("E2E: telemetria del simulador llega a la cola POSIX como EventoWearable")
def _():
    cola = f"/nd_wea_test_{int(time.time()*1000)%100000}"
    _limpiar_cola(cola)
    p_pi = _puerto_libre()
    p_sim = _puerto_libre()
    receptor = ReceptorWearable(puerto_escucha=p_pi, nombre_cola=cola, capacidad_cola=10)
    receptor.iniciar()
    sim = SimuladorPulsera(ip_pi="127.0.0.1", puerto_pi=p_pi, puerto_local=p_sim,
                           intervalo_bpm_s=0.2, bpm_fn=lambda: 66)
    sim.iniciar()
    time.sleep(0.6)  # deja pasar varias telemetrias
    sim.detener()
    receptor.detener()
    # Leer de la cola real
    from NeuroDrive_Core.adaptador_mq import AdaptadorMQ
    lector = AdaptadorMQ.abrir(cola, modo="lectura", capacidad=10)
    leidos = []
    while True:
        j = lector.recibir(timeout_seg=0.2)
        if j is None:
            break
        leidos.append(Envelope.from_json(j))
    lector.cerrar()
    lector.eliminar()
    assert len(leidos) >= 1, "no llego ninguna telemetria a la cola"
    ev = leidos[0].desempacar()
    assert isinstance(ev, EventoWearable) and ev.bpm == 66

@_test("E2E: desafio ACK completo (actuador -> sim -> receptor -> cola)")
def _():
    cola = f"/nd_ack_test_{int(time.time()*1000)%100000}"
    _limpiar_cola(cola)
    p_pi = _puerto_libre()
    p_sim = _puerto_libre()
    # Receptor (Pi escucha en p_pi)
    receptor = ReceptorWearable(puerto_escucha=p_pi, nombre_cola=cola, capacidad_cola=10)
    receptor.iniciar()
    # Simulador escucha comandos en p_sim, manda a p_pi, responde ACK correcto
    sim = SimuladorPulsera(ip_pi="127.0.0.1", puerto_pi=p_pi, puerto_local=p_sim,
                           enviar_telemetria=False, politica_ack="correcto")
    sim.iniciar()
    # Actuador envia comandos al simulador (p_sim)
    act = ActuadorWearable(ip_wearable="127.0.0.1", puerto_envio=p_sim,
                           reenvios_criticos=2, espaciado_reenvios_ms=10)
    act.iniciar()
    act.ejecutar(ComandoActuador(tipo=C.SECUENCIA_ACK, intensidad=80, id_secuencia=11))
    time.sleep(0.6)
    act.detener()
    sim.detener()
    receptor.detener()
    # Verificaciones
    assert len(sim.vibraciones) == 1, f"el sim debia recibir 1 desafio, recibio {len(sim.vibraciones)}"
    from NeuroDrive_Core.adaptador_mq import AdaptadorMQ
    lector = AdaptadorMQ.abrir(cola, modo="lectura", capacidad=10)
    acks = []
    while True:
        j = lector.recibir(timeout_seg=0.2)
        if j is None:
            break
        env = Envelope.from_json(j)
        if env.tipo == TipoMensaje.EVENTO_ACK_WEARABLE:
            acks.append(env.desempacar())
    lector.cerrar()
    lector.eliminar()
    assert len(acks) == 1, f"esperaba 1 ACK en la cola, hubo {len(acks)}"
    assert acks[0].id_secuencia == 11
    assert acks[0].secuencia_correcta is True   # politica 'correcto' -> pad K

@_test("E2E: los reenvios NO causan multiples desafios (dedup por id_paquete)")
def _():
    p_sim = _puerto_libre()
    sim = SimuladorPulsera(ip_pi="127.0.0.1", puerto_pi=_puerto_libre(),
                           puerto_local=p_sim, enviar_telemetria=False,
                           politica_ack="ninguno")
    sim.iniciar()
    act = ActuadorWearable(ip_wearable="127.0.0.1", puerto_envio=p_sim,
                           reenvios_criticos=3, espaciado_reenvios_ms=10)
    act.iniciar()
    act.ejecutar(ComandoActuador(tipo=C.SECUENCIA_ACK, intensidad=80, id_secuencia=22))
    time.sleep(0.4)
    act.detener()
    sim.detener()
    # 3 datagramas iguales (mismo id_paquete) -> 1 solo desafio
    assert len(sim.vibraciones) == 1, (
        f"dedup fallo: {len(sim.vibraciones)} desafios por 3 reenvios")


# =============================================================================
def _resumen():
    print("\n" + "=" * 60)
    ok = sum(1 for _, p, _ in _resultados if p)
    total = len(_resultados)
    print(f"  Resumen: {ok}/{total} tests pasaron")
    if ok != total:
        for n, p, m in _resultados:
            if not p:
                print(f"    - {n}: {m}")
    else:
        print("  WEARABLE (lado Python) OK")
    print("=" * 60)
    return 0 if ok == total else 1

if __name__ == "__main__":
    sys.exit(_resumen())
