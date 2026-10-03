"""
test_wearable.py - Tests del lado Python del wearable.

    cd ~/NeuroDrive
    python -m NeuroDrive_Wearable.test_wearable

No requiere la pulsera ni el punto de acceso: todo corre sobre localhost.

Cubre:
  - protocolo            formato de los mensajes y rechazo de los invalidos
  - ActuadorWearable     con un transporte falso que registra lo enviado
  - ReceptorWearable     con la funcion de publicacion inyectada
  - SimuladorPulsera     que reproduzca las conductas del firmware
  - EXTREMO A EXTREMO    actuador -> UDP -> simulador -> UDP -> receptor ->
                         cola POSIX real -> lectura
"""
from __future__ import annotations

import json
import logging
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

# Los descartes de datagramas invalidos se loguean como WARNING/ERROR. Aca se
# provocan a proposito, asi que se silencian para no ensuciar la salida.
logging.disable(logging.CRITICAL)

C = TipoComandoActuador
T_APAGAR = int(C.APAGAR_TODO)
T_DESAFIO = int(C.SECUENCIA_ACK)
T_LIMPIO = protocolo.TIPO_APAGADO_LIMPIO
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
        self.cerrado = False
        self.envios_tras_cierre = 0
        self._lock = threading.Lock()

    def sendto(self, datos):
        with self._lock:
            if self.cerrado:
                self.envios_tras_cierre += 1
            self.enviados.append(datos)

    def close(self):
        with self._lock:
            self.cerrado = True

    def paquetes(self):
        with self._lock:
            return [json.loads(p) for p in self.enviados]

    def tipos(self):
        return [p["tipo"] for p in self.paquetes()]


def _actuador(t, **kw):
    """Actuador de prueba: sin saludo de inicio e ids desde 0, salvo que se pida."""
    kw.setdefault("saludo_inicio", False)
    kw.setdefault("id_paquete_inicial", 0)
    kw.setdefault("espaciado_reenvios_ms", 5)
    return ActuadorWearable(transporte=t, **kw)


def _puerto_libre():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _esperar(condicion, timeout=2.0, paso=0.01):
    """Espera activa acotada: devuelve True apenas se cumple la condicion."""
    fin = time.monotonic() + timeout
    while time.monotonic() < fin:
        if condicion():
            return True
        time.sleep(paso)
    return condicion()


def _espera_rechazo(datos, que):
    try:
        protocolo.construir_evento(protocolo.parsear_mensaje_pulsera(datos), 1000.0)
    except protocolo.ErrorProtocolo:
        return
    raise AssertionError(f"{que}: deberia lanzar ErrorProtocolo")


# =============================================================================
print("\n--- Protocolo ---")

@_test("serializar/parsear comando round-trip")
def _():
    c = ComandoActuador(tipo=C.SECUENCIA_ACK, intensidad=80, duracion_ms=7000, id_secuencia=5)
    obj = protocolo.parsear_comando(protocolo.serializar_comando(c, id_paquete=7))
    assert obj["tipo"] == T_DESAFIO
    assert obj["id_secuencia"] == 5
    assert obj["id_paquete"] == 7
    assert obj["duracion_ms"] == 7000, "el tipo 8 transporta la ventana de respuesta"

@_test("apagar y apagado limpio usan los tipos 10 y 99")
def _():
    assert json.loads(protocolo.serializar_apagar(3))["tipo"] == 10
    limpio = json.loads(protocolo.serializar_apagado_limpio(4))
    assert limpio["tipo"] == 99 and limpio["id_paquete"] == 4

@_test("telemetria -> EventoWearable")
def _():
    datos = protocolo.serializar_telemetria(bpm=72, bateria=88, id_paquete=1)
    ev = protocolo.construir_evento(protocolo.parsear_mensaje_pulsera(datos), 1000.0)
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
    _espera_rechazo(b"no soy json {{{", "texto que no es JSON")
    _espera_rechazo(b"\xff\xfe\x00", "bytes que no son UTF-8")

@_test("JSON valido que no es un objeto lanza ErrorProtocolo")
def _():
    # json.loads no falla con ninguno de estos: antes terminaban el hilo receptor
    for datos in (b"[1,2]", b"5", b"null", b'"hola"'):
        _espera_rechazo(datos, datos.decode())

@_test("campos con tipo equivocado lanzan ErrorProtocolo (no TypeError)")
def _():
    _espera_rechazo(b'{"msg":"ack","id_secuencia":null}', "ack con id nulo")
    _espera_rechazo(b'{"msg":"ack","id_secuencia":true}', "ack con id booleano")
    _espera_rechazo(b'{"msg":"ack","id_secuencia":3,"tiempo_respuesta_ms":"x"}', "tiempo texto")
    _espera_rechazo(b'{"msg":"telemetria","bpm":"setenta"}', "bpm texto")
    _espera_rechazo(b'{"msg":"telemetria","bpm":999}', "bpm fuera de rango")
    _espera_rechazo(b'{"msg":"otra_cosa"}', "msg desconocido")

@_test("la presentacion maxima del desafio es 4 x (200 + 300) ms")
def _():
    assert protocolo.DESAFIO_PRESENTACION_MAX_MS == 2000


# =============================================================================
print("\n--- ActuadorWearable (transporte falso) ---")

@_test("tipos_soportados son los que declara el protocolo")
def _():
    a = _actuador(_TransporteFalso())
    assert a.tipos_soportados() == {C.VIBRAR_LEVE, C.VIBRAR_MEDIO, C.VIBRAR_FUERTE, C.SECUENCIA_ACK}
    assert a.tipos_soportados() == set(protocolo.TIPOS_PARA_WEARABLE)

@_test("un VIBRAR_LEVE (no critico) se envia una sola vez")
def _():
    t = _TransporteFalso()
    a = _actuador(t)
    a.iniciar()
    a.ejecutar(ComandoActuador(tipo=C.VIBRAR_LEVE, intensidad=30))
    a.esperar_vaciado(1.0)
    a.detener()
    assert t.tipos().count(int(C.VIBRAR_LEVE)) == 1

@_test("SECUENCIA_ACK (critico) se envia 3 veces con el mismo id_paquete")
def _():
    t = _TransporteFalso()
    a = _actuador(t)
    a.iniciar()
    a.ejecutar(ComandoActuador(tipo=C.SECUENCIA_ACK, intensidad=80, id_secuencia=9))
    a.esperar_vaciado(1.0)
    a.detener()
    acks = [p for p in t.paquetes() if p["tipo"] == T_DESAFIO]
    assert len(acks) == 3, f"esperaba 3 envios, hubo {len(acks)}"
    assert len({p["id_paquete"] for p in acks}) == 1
    assert all(p["id_secuencia"] == 9 for p in acks)

@_test("apagar() envia el tipo 10 TRES veces con el mismo id_paquete (C4)")
def _():
    t = _TransporteFalso()
    a = _actuador(t)
    a.iniciar()
    a.apagar()
    a.esperar_vaciado(1.0)
    apagados = [p for p in t.paquetes() if p["tipo"] == T_APAGAR]
    a.detener()
    assert len(apagados) == 3, f"esperaba 3 copias del tipo 10, hubo {len(apagados)}"
    assert len({p["id_paquete"] for p in apagados}) == 1

@_test("ejecutar() y apagar() no bloquean aunque los envios tengan espaciado")
def _():
    t = _TransporteFalso()
    a = _actuador(t, espaciado_reenvios_ms=100)
    a.iniciar()
    t0 = time.monotonic()
    a.ejecutar(ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=100))
    a.apagar()
    dt = time.monotonic() - t0
    a.detener()
    assert dt < 0.05, f"bloqueo {dt*1000:.0f} ms"

@_test("apagar() abandona las repeticiones pendientes de la orden anterior")
def _():
    t = _TransporteFalso()
    a = _actuador(t, espaciado_reenvios_ms=80)
    a.iniciar()
    a.ejecutar(ComandoActuador(tipo=C.SECUENCIA_ACK, id_secuencia=1))
    assert _esperar(lambda: T_DESAFIO in t.tipos()), "no salio la primera copia"
    a.apagar()                      # llega entre la 1ra y la 2da copia
    a.esperar_vaciado(1.5)
    tipos = t.tipos()
    a.detener()
    assert tipos.count(T_DESAFIO) < 3, f"siguio repitiendo la orden vieja: {tipos}"
    # Despues del primer tipo 10 no puede salir ninguna copia de la orden vieja
    desde_apagar = tipos[tipos.index(T_APAGAR):]
    assert T_DESAFIO not in desde_apagar, f"orden obsoleta tras la detencion: {tipos}"
    assert desde_apagar.count(T_APAGAR) == 3

@_test("saludo de inicio: lo primero que sale es un tipo 10 (x3)")
def _():
    t = _TransporteFalso()
    a = _actuador(t, saludo_inicio=True)
    a.iniciar()
    a.esperar_vaciado(1.0)
    tipos = t.tipos()
    a.detener()
    assert tipos[:3] == [T_APAGAR] * 3, tipos

@_test("detener(): tipo 10 (x3), despues tipo 99 (x3), y nada tras cerrar (C2)")
def _():
    t = _TransporteFalso()
    a = _actuador(t)
    a.iniciar()
    a.ejecutar(ComandoActuador(tipo=C.VIBRAR_FUERTE, intensidad=100))
    a.detener()
    tipos = t.tipos()
    assert tipos[-6:] == [T_APAGAR] * 3 + [T_LIMPIO] * 3, tipos
    limpios = [p for p in t.paquetes() if p["tipo"] == T_LIMPIO]
    assert len({p["id_paquete"] for p in limpios}) == 1, "las copias comparten id_paquete"
    assert t.cerrado and t.envios_tras_cierre == 0, "hubo envios con el socket cerrado"

@_test("apagar() solo NO envia el aviso de apagado ordenado")
def _():
    t = _TransporteFalso()
    a = _actuador(t)
    a.iniciar()
    a.apagar()
    a.esperar_vaciado(1.0)
    tipos = t.tipos()
    a.detener()
    assert T_LIMPIO not in tipos, "un descenso de nivel no es un apagado del sistema"

@_test("detener() es idempotente y despues de detener no se encola nada")
def _():
    t = _TransporteFalso()
    a = _actuador(t)
    a.iniciar()
    a.detener()
    n = len(t.paquetes())
    a.detener()
    a.apagar()
    time.sleep(0.05)
    assert len(t.paquetes()) == n
    try:
        a.ejecutar(ComandoActuador(tipo=C.VIBRAR_LEVE))
        assert False, "ejecutar() tras detener() debe fallar"
    except RuntimeError:
        pass

@_test("id_paquete: base distinta en cada sesion, incremental y de 31 bits")
def _():
    bases = {ActuadorWearable(transporte=_TransporteFalso())._id_paquete for _ in range(8)}
    assert len(bases) > 1, "la base deberia sortearse por sesion"
    assert all(0 <= b < 2 ** 30 for b in bases)
    t = _TransporteFalso()
    a = ActuadorWearable(transporte=t, saludo_inicio=False, espaciado_reenvios_ms=1)
    base = a._id_paquete
    a.iniciar()
    a.ejecutar(ComandoActuador(tipo=C.VIBRAR_LEVE))
    a.ejecutar(ComandoActuador(tipo=C.VIBRAR_MEDIO))
    a.esperar_vaciado(1.0)
    ids = [p["id_paquete"] for p in t.paquetes()]
    a.detener()
    assert ids == [base + 1, base + 2], (base, ids)

@_test("con la cola llena se descarta la orden comun, nunca la detencion")
def _():
    class _TransporteLento(_TransporteFalso):
        def sendto(self, datos):
            time.sleep(0.05)
            super().sendto(datos)
    t = _TransporteLento()
    a = _actuador(t, capacidad_cola=4, espaciado_reenvios_ms=0)
    a.iniciar()
    for _ in range(12):
        a.ejecutar(ComandoActuador(tipo=C.VIBRAR_LEVE))
    assert a.envios_descartados_cola > 0, "deberia haber descartado por cola llena"
    a.apagar()
    a.esperar_vaciado(2.0)
    tipos = t.tipos()
    a.detener()
    assert tipos.count(T_APAGAR) == 3, f"la detencion debe salir siempre: {tipos}"

@_test("integracion con el despachador: VIBRAR llega y APAGAR_TODO sale triple")
def _():
    t = _TransporteFalso()
    a = _actuador(t)
    d = DespachadorComandos()
    d.registrar_actuador(a)
    d.iniciar()

    def salida(*comandos):
        return SalidaFSM(
            timestamp=1.0, estado_actual=EstadoFSM.ALERTA_LEVE,
            estado_anterior=EstadoFSM.PRE_ALERTA, nivel_alerta=2,
            comandos=tuple(comandos), transicion_ocurrio=True,
        )
    d.despachar(salida(ComandoActuador(tipo=C.VIBRAR_LEVE, intensidad=30)))
    d.esperar_vaciado(1.0)
    a.esperar_vaciado(1.0)
    assert int(C.VIBRAR_LEVE) in t.tipos()
    d.despachar(salida(ComandoActuador(tipo=C.APAGAR_TODO)))
    d.esperar_vaciado(1.0)
    a.esperar_vaciado(1.0)
    tipos = t.tipos()
    d.detener()
    assert tipos.count(T_APAGAR) == 3, tipos


# =============================================================================
print("\n--- ReceptorWearable (publicar_fn inyectada) ---")

def _receptor(publicar_fn):
    r = ReceptorWearable(puerto_escucha=_puerto_libre(), publicar_fn=publicar_fn)
    r.iniciar()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return r, s, ("127.0.0.1", r._puerto)

@_test("telemetria recibida -> Envelope EVENTO_WEARABLE publicado")
def _():
    capturados = []
    r, s, dst = _receptor(lambda j: (capturados.append(j) or True))
    s.sendto(protocolo.serializar_telemetria(bpm=70, id_paquete=1), dst)
    _esperar(lambda: len(capturados) >= 1)
    r.detener(); s.close()
    assert len(capturados) == 1, f"esperaba 1 publicado, hubo {len(capturados)}"
    env = Envelope.from_json(capturados[0])
    assert env.tipo == TipoMensaje.EVENTO_WEARABLE
    assert env.desempacar().bpm == 70

@_test("ack recibido -> Envelope EVENTO_ACK_WEARABLE publicado")
def _():
    capturados = []
    r, s, dst = _receptor(lambda j: (capturados.append(j) or True))
    s.sendto(protocolo.serializar_ack(id_secuencia=4, secuencia_correcta=False,
                                      tiempo_respuesta_ms=900), dst)
    _esperar(lambda: len(capturados) >= 1)
    r.detener(); s.close()
    assert len(capturados) == 1
    env = Envelope.from_json(capturados[0])
    assert env.tipo == TipoMensaje.EVENTO_ACK_WEARABLE
    ev = env.desempacar()
    assert ev.id_secuencia == 4 and ev.secuencia_correcta is False

@_test("tres copias de la misma respuesta -> se publica UNA (C3)")
def _():
    capturados = []
    r, s, dst = _receptor(lambda j: (capturados.append(j) or True))
    ack = protocolo.serializar_ack(id_secuencia=7, secuencia_correcta=True, tiempo_respuesta_ms=1200)
    for _ in range(3):
        s.sendto(ack, dst)
        time.sleep(0.02)
    _esperar(lambda: r.stats.paquetes_recibidos >= 3)
    r.detener(); s.close()
    assert len(capturados) == 1, f"esperaba 1 publicado, hubo {len(capturados)}"
    assert r.stats.acks == 1 and r.stats.acks_duplicados == 2, r.stats

@_test("respuestas a verificaciones distintas se publican las dos")
def _():
    capturados = []
    r, s, dst = _receptor(lambda j: (capturados.append(j) or True))
    for id_seq in (7, 8):
        for _ in range(3):
            s.sendto(protocolo.serializar_ack(id_seq, True, 1000), dst)
    _esperar(lambda: r.stats.paquetes_recibidos >= 6)
    r.detener(); s.close()
    ids = [Envelope.from_json(j).desempacar().id_secuencia for j in capturados]
    assert ids == [7, 8], ids
    assert r.stats.acks_duplicados == 4

@_test("si la primera copia no se pudo publicar, la segunda tiene su oportunidad")
def _():
    resultados = iter([False, True, True])
    r, s, dst = _receptor(lambda j: next(resultados))
    ack = protocolo.serializar_ack(id_secuencia=5, secuencia_correcta=True, tiempo_respuesta_ms=800)
    for _ in range(3):
        s.sendto(ack, dst)
        time.sleep(0.02)
    _esperar(lambda: r.stats.paquetes_recibidos >= 3)
    r.detener(); s.close()
    assert r.stats.descartes_publicacion == 1
    assert r.stats.acks == 1 and r.stats.acks_duplicados == 1, r.stats

@_test("datagramas invalidos se cuentan y NO terminan el hilo (R1)")
def _():
    r, s, dst = _receptor(lambda j: True)
    malos = (
        b"basura no json",
        b"[1,2]",                                       # JSON valido, no objeto
        b"5",
        b'{"msg":"ack","id_secuencia":null}',           # antes: TypeError
        b'{"msg":"ack","id_secuencia":"tres"}',         # texto donde va un numero
        b'{"msg":"telemetria","bpm":"x"}',
    )
    for d in malos:
        s.sendto(d, dst)
    s.sendto(protocolo.serializar_telemetria(bpm=80, id_paquete=1), dst)
    _esperar(lambda: r.stats.paquetes_recibidos >= len(malos) + 1)
    vivo = r._hilo is not None and r._hilo.is_alive()
    r.detener(); s.close()
    assert vivo, "el hilo receptor termino por un datagrama invalido"
    assert r.stats.invalidos == len(malos), r.stats
    assert r.stats.telemetrias == 1, "la telemetria valida posterior debe procesarse"

@_test("una excepcion al publicar no termina el hilo")
def _():
    def publicar(j):
        raise RuntimeError("cola rota")
    r, s, dst = _receptor(publicar)
    s.sendto(protocolo.serializar_telemetria(bpm=70, id_paquete=1), dst)
    s.sendto(protocolo.serializar_telemetria(bpm=71, id_paquete=2), dst)
    _esperar(lambda: r.stats.paquetes_recibidos >= 2)
    vivo = r._hilo.is_alive()
    r.detener(); s.close()
    assert vivo and r.stats.descartes_publicacion == 2


# =============================================================================
print("\n--- SimuladorPulsera (conductas del firmware) ---")

def _simulador(**kw):
    kw.setdefault("enviar_telemetria", False)
    p_sim = _puerto_libre()
    sim = SimuladorPulsera(ip_pi="127.0.0.1", puerto_pi=kw.pop("puerto_pi", _puerto_libre()),
                           puerto_local=p_sim, **kw)
    sim.iniciar()
    return sim, p_sim

def _orden(tipo, id_paquete, duracion_ms=0, id_secuencia=None):
    return protocolo._serializar_orden(tipo, id_paquete, duracion_ms=duracion_ms,
                                       id_secuencia=id_secuencia)

@_test("la ventana de respuesta sale de duracion_ms; con 0 usa la de defecto (C5)")
def _():
    sim, p = _simulador(politica_ack="ninguno")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(_orden(T_DESAFIO, 1, duracion_ms=7000, id_secuencia=1), ("127.0.0.1", p))
    s.sendto(_orden(T_DESAFIO, 2, duracion_ms=0, id_secuencia=2), ("127.0.0.1", p))
    _esperar(lambda: len(sim.ventanas_ms) >= 2)
    sim.detener(); s.close()
    assert sim.ventanas_ms == [7000, protocolo.VENTANA_DESAFIO_DEFAULT_MS], sim.ventanas_ms

@_test("un toque posterior al cierre de la ventana no se envia")
def _():
    sim, p = _simulador(politica_ack="correcto", tiempo_respuesta_fn=lambda: 8000)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(_orden(T_DESAFIO, 1, duracion_ms=7000, id_secuencia=1), ("127.0.0.1", p))
    _esperar(lambda: sim.respuestas_fuera_de_ventana >= 1)
    time.sleep(0.1)
    sim.detener(); s.close()
    assert sim.respuestas_fuera_de_ventana == 1 and sim.acks_enviados == 0

@_test("la respuesta sale 3 veces (C3)")
def _():
    sim, p = _simulador(politica_ack="correcto", espaciado_ack_ms=5)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(_orden(T_DESAFIO, 1, duracion_ms=7000, id_secuencia=1), ("127.0.0.1", p))
    _esperar(lambda: sim.acks_enviados >= 3)
    time.sleep(0.05)
    sim.detener(); s.close()
    assert sim.acks_enviados == 3, sim.acks_enviados

@_test("el tipo 10 cancela la verificacion en curso: no hay respuesta despues")
def _():
    sim, p = _simulador(politica_ack="correcto", tiempo_real=True,
                        k_fn=lambda: 1, tiempo_respuesta_fn=lambda: 100)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(_orden(T_DESAFIO, 1, duracion_ms=7000, id_secuencia=1), ("127.0.0.1", p))
    _esperar(lambda: len(sim.vibraciones) >= 1)
    s.sendto(_orden(T_APAGAR, 2), ("127.0.0.1", p))
    time.sleep(0.9)             # 1 pulso (0,5 s) + 0,1 s de reaccion + holgura
    sim.detener(); s.close()
    assert sim.apagados == 1 and sim.acks_enviados == 0, (sim.apagados, sim.acks_enviados)

@_test("tipo 99 marca la desconexion como esperada; otra orden lo anula (R3)")
def _():
    sim, p = _simulador(politica_ack="ninguno")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(_orden(T_LIMPIO, 1), ("127.0.0.1", p))
    assert _esperar(lambda: sim.apagado_limpio_recibido), "no registro el tipo 99"
    s.sendto(_orden(T_APAGAR, 2), ("127.0.0.1", p))        # el nucleo volvio
    assert _esperar(lambda: not sim.apagado_limpio_recibido), "la marca quedo pegada"
    sim.detener(); s.close()

@_test("el descarte de repetidas recuerda 16 identificadores, como el firmware")
def _():
    sim, p = _simulador(politica_ack="ninguno")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = ("127.0.0.1", p)
    s.sendto(_orden(int(C.VIBRAR_LEVE), 100), dst)
    s.sendto(_orden(int(C.VIBRAR_LEVE), 100), dst)          # repetida: se descarta
    _esperar(lambda: sim.duplicados_descartados >= 1)
    assert len(sim.comandos_recibidos) == 1
    for i in range(protocolo.DEDUP_IDS_RECORDADOS):         # 16 ordenes mas
        s.sendto(_orden(int(C.VIBRAR_LEVE), 200 + i), dst)
    _esperar(lambda: len(sim.comandos_recibidos) >= 17)
    s.sendto(_orden(int(C.VIBRAR_LEVE), 100), dst)          # ya salio del anillo
    _esperar(lambda: len(sim.comandos_recibidos) >= 18)
    sim.detener(); s.close()
    assert len(sim.comandos_recibidos) == 18


# =============================================================================
print("\n--- EXTREMO A EXTREMO (localhost + cola POSIX real) ---")

def _limpiar_cola(nombre):
    try:
        import posix_ipc
        posix_ipc.unlink_message_queue(nombre)
    except Exception:
        pass

def _leer_cola(cola):
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
    return leidos

@_test("E2E: telemetria del simulador llega a la cola POSIX como EventoWearable")
def _():
    cola = f"/nd_wea_test_{int(time.time()*1000)%100000}"
    _limpiar_cola(cola)
    p_pi = _puerto_libre()
    receptor = ReceptorWearable(puerto_escucha=p_pi, nombre_cola=cola, capacidad_cola=10)
    receptor.iniciar()
    sim, _p = _simulador(puerto_pi=p_pi, enviar_telemetria=True,
                         intervalo_bpm_s=0.2, bpm_fn=lambda: 66)
    _esperar(lambda: receptor.stats.publicados >= 2)
    sim.detener()
    receptor.detener()
    leidos = _leer_cola(cola)
    assert len(leidos) >= 1, "no llego ninguna telemetria a la cola"
    ev = leidos[0].desempacar()
    assert isinstance(ev, EventoWearable) and ev.bpm == 66

@_test("E2E: verificacion completa; 3 copias de la respuesta -> 1 en la cola")
def _():
    cola = f"/nd_ack_test_{int(time.time()*1000)%100000}"
    _limpiar_cola(cola)
    p_pi = _puerto_libre()
    receptor = ReceptorWearable(puerto_escucha=p_pi, nombre_cola=cola, capacidad_cola=10)
    receptor.iniciar()
    sim, p_sim = _simulador(puerto_pi=p_pi, politica_ack="correcto", espaciado_ack_ms=10)
    act = ActuadorWearable(ip_wearable="127.0.0.1", puerto_envio=p_sim,
                           reenvios_criticos=3, espaciado_reenvios_ms=10)
    act.iniciar()
    act.ejecutar(ComandoActuador(tipo=C.SECUENCIA_ACK, intensidad=80,
                                 duracion_ms=7000, id_secuencia=11))
    _esperar(lambda: receptor.stats.paquetes_recibidos >= 3)
    act.detener()
    sim.detener()
    receptor.detener()
    assert len(sim.vibraciones) == 1, f"3 copias de la orden -> 1 desafio; hubo {len(sim.vibraciones)}"
    assert sim.ventanas_ms == [7000]
    assert sim.acks_enviados == 3
    acks = [e.desempacar() for e in _leer_cola(cola)
            if e.tipo == TipoMensaje.EVENTO_ACK_WEARABLE]
    assert len(acks) == 1, f"esperaba 1 respuesta en la cola, hubo {len(acks)}"
    assert acks[0].id_secuencia == 11 and acks[0].secuencia_correcta is True

@_test("E2E: apagado ordenado; al relanzar el nucleo la marca se anula")
def _():
    sim, p_sim = _simulador(politica_ack="ninguno")
    act = ActuadorWearable(ip_wearable="127.0.0.1", puerto_envio=p_sim,
                           espaciado_reenvios_ms=10)
    act.iniciar()
    _esperar(lambda: sim.apagados >= 1)                    # saludo de inicio
    assert sim.apagado_limpio_recibido is False
    act.detener()                                          # tipo 10 + tipo 99
    assert _esperar(lambda: sim.apagado_limpio_recibido), "no llego el tipo 99"
    # Se relanza el nucleo SIN que la pulsera pierda la red
    act2 = ActuadorWearable(ip_wearable="127.0.0.1", puerto_envio=p_sim,
                            espaciado_reenvios_ms=10)
    act2.iniciar()
    ok = _esperar(lambda: not sim.apagado_limpio_recibido)
    act2.detener()
    sim.detener()
    assert ok, "tras relanzar, una caida imprevista quedaria sin aviso"

@_test("E2E: tras reiniciar el nucleo, sus ordenes NO se descartan como repetidas (R2)")
def _():
    def sesion(sim, p_sim, id_inicial):
        act = ActuadorWearable(ip_wearable="127.0.0.1", puerto_envio=p_sim,
                               espaciado_reenvios_ms=5, saludo_inicio=False,
                               id_paquete_inicial=id_inicial)
        act.iniciar()
        act.ejecutar(ComandoActuador(tipo=C.SECUENCIA_ACK, id_secuencia=1, duracion_ms=7000))
        act.esperar_vaciado(1.0)
        time.sleep(0.1)
        act.detener()

    # Como era antes: el contador arrancaba siempre en 0
    sim, p_sim = _simulador(politica_ack="ninguno")
    sesion(sim, p_sim, 0)
    sesion(sim, p_sim, 0)
    _esperar(lambda: sim.duplicados_descartados >= 1)
    antes = len(sim.vibraciones)
    sim.detener()
    assert antes == 1, "con base fija, el desafio de la 2da sesion se pierde (caso que se corrige)"

    # Ahora: base aleatoria por sesion
    sim, p_sim = _simulador(politica_ack="ninguno")
    sesion(sim, p_sim, None)
    sesion(sim, p_sim, None)
    _esperar(lambda: len(sim.vibraciones) >= 2)
    ahora = len(sim.vibraciones)
    sim.detener()
    assert ahora == 2, f"el desafio de la 2da sesion debe llegar; llegaron {ahora}"


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
