"""
Banco de pruebas del firmware en PC.

red.c y motor_pads.c se compilan SIN modificar en una biblioteca; el Wi-Fi, el
motor, los pads y el reloj se reemplazan por funciones que controla este
script. Del otro lado van los modulos REALES de la Pi (ActuadorWearable,
ReceptorWearable, protocolo), comunicados por UDP sobre localhost.

Se ejecuta con correr.sh, que antes compila la biblioteca:

    ./correr.sh            corre todos los casos (cada uno en su proceso)
    ./correr.sh <caso>     corre uno

El reloj del firmware lo maneja el test: los 60 s que la pulsera espera antes
de pasar a la red de respaldo transcurren en milisegundos.
"""
import ctypes, json, logging, os, socket, subprocess, sys, time, urllib.parse

AQUI = os.path.dirname(os.path.abspath(__file__))
# Raiz del repositorio: donde estan NeuroDrive_Wearable y common. Por defecto,
# dos niveles arriba (el firmware vive en <repo>/NeuroDrive_Firmware).
REPO = os.environ.get("NEURODRIVE_REPO") or os.path.dirname(os.path.dirname(AQUI))
if not os.path.isdir(os.path.join(REPO, "NeuroDrive_Wearable")):
    sys.exit(f"No encuentro NeuroDrive_Wearable en {REPO}. "
             f"Definir NEURODRIVE_REPO con la raiz del repositorio.")
sys.path.insert(0, REPO)
logging.disable(logging.CRITICAL)

CONECTANDO, OPERANDO, CONMUTAR, TESTIGO = 0, 1, 2, 3
VUELTA_MS = 20


class Pulsera:
    """El firmware, con su ciclo principal y su reloj manejados desde aca."""
    def __init__(self, bpm=None, bateria=87):
        self.lib = L = ctypes.CDLL(os.path.join(AQUI, "build", "libpulsera.so"))
        for f in ("banco_ssid", "banco_registro", "banco_http_cuerpo", "banco_http_url"):
            getattr(L, f).restype = ctypes.c_char_p
        L.banco_ahora_ms.restype = ctypes.c_int64
        CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_int)
        L.motor_pads_ejecutar_comando.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_int]
        self._cb = ctypes.cast(L.motor_pads_ejecutar_comando, CB)
        L.red_set_comando_handler(self._cb)
        BPM = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int))
        def _bpm(p):
            if bpm is None: return False
            p[0] = bpm; return True
        self._bpm = BPM(_bpm); L.red_set_bpm_provider(self._bpm)
        BAT = ctypes.CFUNCTYPE(ctypes.c_int)
        self._bat = BAT(lambda: bateria); L.red_set_bateria_provider(self._bat)
        L.red_init(); L.motor_pads_init()

    def vueltas(self, ms):
        """Hace girar el ciclo principal durante `ms` del reloj del firmware."""
        for _ in range(max(1, ms // VUELTA_MS)):
            self.lib.red_update(); self.lib.motor_pads_update()
            self.lib.banco_avanzar_ms(VUELTA_MS)
    def atender(self, seg):
        """Gira el ciclo durante `seg` segundos REALES sin mover el reloj del
        firmware: da tiempo a los hilos de la Pi sin que avance la verificacion."""
        fin = time.time() + seg
        while time.time() < fin:
            self.lib.red_update(); self.lib.motor_pads_update(); time.sleep(0.002)
    def estado(self): return self.lib.red_obtener_estado_conec()
    def ssid(self): return self.lib.banco_ssid().decode()
    def registro(self): return self.lib.banco_registro().decode()
    def conectar(self): self.lib.banco_evento_conectado()
    def desconectar(self): self.lib.banco_evento_desconectado()
    def asociar_a_pi(self):
        self.conectar(); self.vueltas(40)
        assert self.estado() == OPERANDO, self.estado()


def escucha_5005():
    """Ocupa el lugar de la Pi: recibe lo que la pulsera transmite."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 5005)); s.setblocking(False)
    return s

def leer_todo(s):
    out = []
    while True:
        try: out.append(json.loads(s.recvfrom(2048)[0]))
        except BlockingIOError: return out

def orden(tipo, id_paquete, duracion_ms=0, id_secuencia=None, intensidad=0):
    from NeuroDrive_Wearable import protocolo
    return protocolo._serializar_orden(tipo, id_paquete, intensidad, duracion_ms, id_secuencia)

def enviar(datos):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(datos, ("127.0.0.1", 5006)); s.close()

def perder_pi_hasta_respaldo(p):
    """Corta el enlace y deja pasar el tiempo hasta que la pulsera pide la red de respaldo."""
    t0 = p.lib.banco_ahora_ms()
    p.desconectar(); p.vueltas(40)
    assert p.estado() == CONECTANDO
    while p.estado() == CONECTANDO and p.lib.banco_ahora_ms() - t0 < 90000:
        p.vueltas(20)
    demora = p.lib.banco_ahora_ms() - t0
    assert p.estado() == CONMUTAR, p.estado()
    # 60 s de umbral; se evalua al vencer un intento (cada 13 s): 62 s
    assert 60000 <= demora <= 66000, demora
    assert p.ssid() == "RedDeRespaldo" and p.lib.banco_ip_fija() == 0


# ============================== CASOS ==============================

def caso_arranque_y_reintentos():
    """Sin la Pi: reintenta cada 13 s y el ciclo principal nunca se detiene."""
    p = Pulsera()
    assert p.ssid() == "NeuroDrive_AP" and p.lib.banco_ip_fija() == 1
    assert p.lib.banco_asociaciones() == 1
    p.vueltas(9980);  assert p.lib.banco_asociaciones() == 1          # sigue esperando
    p.vueltas(3040);  assert p.lib.banco_asociaciones() == 2, "debio reintentar a los 13 s"
    p.vueltas(13000); assert p.lib.banco_asociaciones() == 3
    assert p.estado() == CONECTANDO
    assert p.lib.banco_pausas() == 0, "el firmware llamo a vTaskDelay"

def caso_telemetria():
    """Asociada a la Pi: telemetria cada 2 s, con null cuando no hay pulso."""
    rx = escucha_5005(); p = Pulsera(bpm=None)
    p.asociar_a_pi()
    enviar(orden(10, 1)); time.sleep(0.02)      # con la primera orden aprende la direccion de la Pi
    p.vueltas(6100)
    t = leer_todo(rx)
    assert 3 <= len(t) <= 4, len(t)
    assert all(m["msg"] == "telemetria" and m["bpm"] is None and m["bateria"] == 87 for m in t), t
    ids = [m["id_paquete"] for m in t]; assert ids == sorted(set(ids))

def caso_telemetria_con_pulso():
    rx = escucha_5005(); p = Pulsera(bpm=72)
    p.asociar_a_pi(); enviar(orden(10, 1)); time.sleep(0.02); p.vueltas(2100)
    assert leer_todo(rx)[-1]["bpm"] == 72

def caso_verificacion_completa_con_la_pi_real():
    """Actuador y receptor REALES de la Pi contra el firmware: orden triple ->
    un solo desafio; ventana de la orden; respuesta triple -> una publicada."""
    from common.contratos import ComandoActuador, TipoComandoActuador as C, Envelope
    from NeuroDrive_Wearable.actuador_wearable import ActuadorWearable
    from NeuroDrive_Wearable.receptor_wearable import ReceptorWearable
    publicados = []
    rec = ReceptorWearable(puerto_escucha=5005, publicar_fn=lambda j: (publicados.append(j) or True))
    rec.iniciar()
    p = Pulsera(); p.asociar_a_pi(); p.lib.banco_fijar_k(3)
    act = ActuadorWearable(ip_wearable="127.0.0.1", puerto_envio=5006, espaciado_reenvios_ms=20)
    act.iniciar()
    act.ejecutar(ComandoActuador(tipo=C.SECUENCIA_ACK, duracion_ms=7000, id_secuencia=41))
    p.atender(0.4)                                 # llegan las 3 copias de la orden
    reg = p.registro()
    assert reg.count("desafio id=41") == 1, "las 3 copias de la orden deben dar UN desafio"
    assert "ventana 7000 ms" in reg, reg[-400:]
    assert reg.count("dup id=") >= 2
    p.lib.banco_reiniciar_pulsos()
    p.vueltas(1600)                                # 3 pulsos de 200+300 ms
    assert "esperando toque (7000 ms)" in p.registro()
    assert p.lib.banco_duty() == 0 and p.lib.banco_led_pads() == 1
    p.vueltas(1000)
    p.lib.banco_tocar_pad(3, 1)
    p.vueltas(300)                                 # antirrebote + 3 copias de la respuesta
    p.lib.banco_tocar_pad(3, 0)
    act.detener(); p.atender(0.3); rec.detener()
    acks = [Envelope.from_json(j).desempacar() for j in publicados if '"msg' not in j and "secuencia_correcta" in j]
    assert rec.stats.acks == 1 and rec.stats.acks_duplicados == 2, rec.stats
    a = [e for e in acks if hasattr(e, "secuencia_correcta")][0]
    assert a.id_secuencia == 41 and a.secuencia_correcta is True
    assert 900 <= a.tiempo_respuesta_ms <= 1300, a.tiempo_respuesta_ms
    assert p.lib.banco_led_pads() == 0
    assert "APAGADO_LIMPIO" in p.registro(), "el detener() del actuador debe llegar como tipo 99"

def _desafio(p, rx, k, ventana, id_seq=5, id_paq=100):
    p.lib.banco_fijar_k(k); p.lib.banco_reiniciar_pulsos()
    enviar(orden(8, id_paq, ventana, id_seq)); time.sleep(0.02); p.vueltas(40)
    leer_todo(rx)

def caso_cantidad_de_pulsos():
    """K pulsos de 200 ms separados por 300 ms, para K de 1 a 4."""
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    for k in (1, 2, 3, 4):
        _desafio(p, rx, k, 7000, id_seq=k, id_paq=200 + k)
        p.vueltas(k * 500 + 60)
        assert p.lib.banco_pulsos_motor() == k, (k, p.lib.banco_pulsos_motor())
        assert f"esperando toque (7000 ms)" in p.registro()
        enviar(orden(10, 300 + k)); time.sleep(0.02); p.vueltas(40)

def caso_ventana_vence_sin_respuesta():
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    _desafio(p, rx, 1, 7000)
    p.vueltas(500)                                  # 1 pulso
    p.vueltas(6900); assert p.lib.banco_led_pads() == 1, "la ventana cerro antes de tiempo"
    p.vueltas(200);  assert p.lib.banco_led_pads() == 0, "la ventana no cerro a los 7 s"
    p.lib.banco_tocar_pad(1, 1); p.vueltas(200)     # toque tardio
    assert [m for m in leer_todo(rx) if m["msg"] == "ack"] == [], "respondio fuera de la ventana"

def caso_ventana_por_defecto_y_fuera_de_rango():
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    _desafio(p, rx, 1, 0, id_paq=1);       assert "ventana 10000 ms" in p.registro()
    _desafio(p, rx, 1, 500, id_paq=2);     assert "fuera de rango" in p.registro()
    _desafio(p, rx, 1, 900000, id_paq=3);  assert p.registro().count("fuera de rango") == 2
    _desafio(p, rx, 1, 10000, id_paq=4)
    p.vueltas(500 + 9900); assert p.lib.banco_led_pads() == 1
    p.vueltas(200);        assert p.lib.banco_led_pads() == 0

def caso_respuesta_incorrecta_y_triple():
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    _desafio(p, rx, 2, 7000, id_seq=9)
    p.vueltas(1100); p.lib.banco_tocar_pad(4, 1); p.vueltas(400)
    acks = [m for m in leer_todo(rx) if m["msg"] == "ack"]
    assert len(acks) == 3, f"esperaba 3 copias, llegaron {len(acks)}"
    assert all(a == acks[0] for a in acks), "las copias deben ser identicas"
    assert acks[0]["id_secuencia"] == 9 and acks[0]["secuencia_correcta"] is False
    assert acks[0]["v"] == 1 and isinstance(acks[0]["tiempo_respuesta_ms"], int)

def caso_copias_espaciadas():
    """Las copias de la respuesta salen separadas ~50 ms, no en la misma vuelta."""
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    _desafio(p, rx, 1, 7000); p.vueltas(560); p.lib.banco_tocar_pad(1, 1)
    llegadas = []
    for _ in range(20):
        p.vueltas(20); n = len([m for m in leer_todo(rx) if m["msg"] == "ack"])
        if n: llegadas.append((p.lib.banco_ahora_ms(), n))
    assert [n for _, n in llegadas] == [1, 1, 1], llegadas
    d1, d2 = llegadas[1][0] - llegadas[0][0], llegadas[2][0] - llegadas[1][0]
    assert 40 <= d1 <= 80 and 40 <= d2 <= 80, (d1, d2)

def caso_apagar_cancela_verificacion():
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    _desafio(p, rx, 4, 7000)
    p.vueltas(700); assert p.lib.banco_led_pads() == 1
    enviar(orden(10, 101)); time.sleep(0.02); p.vueltas(40)
    assert p.lib.banco_duty() == 0 and p.lib.banco_led_pads() == 0
    p.vueltas(3000); p.lib.banco_tocar_pad(4, 1); p.vueltas(300)
    assert [m for m in leer_todo(rx) if m["msg"] == "ack"] == []

def caso_vibrar_no_interrumpe_verificacion():
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    enviar(orden(3, 50, 2500, intensidad=80)); time.sleep(0.02); p.vueltas(40)
    assert p.lib.banco_duty() == 204                 # 80 % de 255
    _desafio(p, rx, 2, 7000, id_paq=51)
    enviar(orden(1, 52, 1500, intensidad=30)); time.sleep(0.02); p.vueltas(40)
    assert "ignorada" in p.registro()
    p.vueltas(1100); assert p.lib.banco_pulsos_motor() <= 2 and p.lib.banco_duty() == 0

def caso_desafio_sin_identificador_se_ignora():
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    enviar(orden(8, 60, 7000, None)); time.sleep(0.02); p.vueltas(40)
    assert "sin id_secuencia" in p.registro() and p.lib.banco_led_pads() == 0

def caso_datagramas_invalidos():
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    for d in (b"basura", b"[1,2]", b"5", b"{}", b'{"tipo":"x"}', b'{"tipo":8}', b"\xff\xfe", b"x" * 400):
        enviar(d)
    time.sleep(0.05); p.vueltas(400)
    assert p.lib.banco_duty() == 0 and p.estado() == OPERANDO
    enviar(orden(3, 70, 0, intensidad=100)); time.sleep(0.02); p.vueltas(40)
    assert p.lib.banco_duty() == 255, "despues de la basura debe seguir atendiendo"

def caso_rafaga_de_ordenes():
    """Mas ordenes que el tope por vuelta: se atienden todas, en vueltas sucesivas."""
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    for i in range(12): enviar(orden(1, 400 + i, 100, intensidad=30))
    time.sleep(0.05)
    p.vueltas(20); assert p.registro().count("[cmd] VIBRAR") == 4
    p.vueltas(60); assert p.registro().count("[cmd] VIBRAR") == 12

def caso_perdida_en_plena_vibracion():
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    enviar(orden(3, 80, 0, intensidad=100)); time.sleep(0.02); p.vueltas(40)
    assert p.lib.banco_duty() == 255
    p.desconectar(); p.vueltas(40)
    assert p.lib.banco_duty() == 0, "el motor quedo encendido tras perder el enlace"
    assert p.estado() == CONECTANDO

def caso_perdida_no_anunciada_avisa():
    """60 s sin la Pi -> red de respaldo -> aviso en tarea propia -> modo testigo."""
    rx = escucha_5005(); p = Pulsera(bateria=64); p.asociar_a_pi()
    perder_pi_hasta_respaldo(p)
    assert p.lib.banco_tareas_creadas() == 0, "no puede avisar antes de tener red"
    p.conectar(); p.vueltas(40)
    assert p.lib.banco_tareas_creadas() == 1 and p.lib.banco_tarea_pila() == 8192
    assert p.lib.banco_http_envios() == 0
    # El aviso tarda: el ciclo principal sigue girando y el estado no avanza
    p.vueltas(5000); assert p.estado() == CONMUTAR
    p.lib.banco_correr_tarea(); p.vueltas(40)
    assert p.lib.banco_http_envios() == 1 and p.estado() == TESTIGO
    assert p.lib.banco_http_url().decode() == "https://api.telegram.org/botTOKEN/sendMessage"
    assert p.lib.banco_http_verifica_nombre() == 1, "debe verificar el nombre del certificado"
    assert p.lib.banco_http_timeout() == 10000
    campos = urllib.parse.parse_qs(p.lib.banco_http_cuerpo().decode(), strict_parsing=True)
    assert campos["chat_id"] == ["12345"]
    texto = campos["text"][0]
    assert texto.startswith("NeuroDrive: perdida de conexion con la unidad.\n"), repr(texto)
    assert texto.endswith("Ultima bateria conocida: 64%"), repr(texto)
    assert p.lib.banco_pausas() == 0

def caso_modo_testigo_revisa_y_vuelve():
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    perder_pi_hasta_respaldo(p); p.conectar(); p.vueltas(40); p.lib.banco_correr_tarea(); p.vueltas(40)
    assert p.estado() == TESTIGO and p.ssid() == "RedDeRespaldo"
    p.vueltas(19900); assert p.ssid() == "RedDeRespaldo"
    p.vueltas(200);   assert p.ssid() == "NeuroDrive_AP" and p.lib.banco_ip_fija() == 1
    p.vueltas(10100); assert p.ssid() == "RedDeRespaldo", "debio volver al respaldo"
    p.conectar(); p.vueltas(40)
    assert p.lib.banco_tareas_creadas() == 1 and p.lib.banco_http_envios() == 1, "el aviso no se repite"
    p.vueltas(20100); assert p.ssid() == "NeuroDrive_AP"
    p.conectar(); p.vueltas(40)
    assert p.estado() == OPERANDO
    enviar(orden(10, 1)); time.sleep(0.02)
    p.vueltas(2100); assert any(m["msg"] == "telemetria" for m in leer_todo(rx))

def caso_no_confunde_el_respaldo_con_la_pi():
    """Al ir a buscar a la Pi, el 'conectado' de la red de respaldo no debe contar."""
    p = Pulsera(); p.asociar_a_pi()
    perder_pi_hasta_respaldo(p); p.conectar(); p.vueltas(40); p.lib.banco_correr_tarea(); p.vueltas(40)
    assert p.estado() == TESTIGO
    p.vueltas(20100)                     # pide la Pi; NO llega ningun evento de desconexion
    assert p.ssid() == "NeuroDrive_AP"
    p.vueltas(200); assert p.estado() == TESTIGO, "tomo como Pi el enlace que venia del respaldo"

def caso_perdida_anunciada_no_avisa():
    from NeuroDrive_Wearable import protocolo
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    for _ in range(3): enviar(protocolo.serializar_apagado_limpio(900))
    time.sleep(0.03); p.vueltas(60)
    assert p.registro().count("APAGADO_LIMPIO") == 1
    perder_pi_hasta_respaldo(p); p.conectar(); p.vueltas(40)
    assert p.lib.banco_tareas_creadas() == 0 and p.estado() == TESTIGO
    assert "ANUNCIADA" in p.registro()
    p.vueltas(60000); assert p.lib.banco_http_envios() == 0

def caso_relanzar_la_pi_anula_el_anuncio():
    """Tipo 99, la Pi se relanza sin que caiga la red (saludo tipo 10) y despues se corta: SI avisa."""
    from NeuroDrive_Wearable.actuador_wearable import ActuadorWearable
    rx = escucha_5005(); p = Pulsera(); p.asociar_a_pi()
    a1 = ActuadorWearable(ip_wearable="127.0.0.1", espaciado_reenvios_ms=10)
    a1.iniciar(); p.atender(0.15); a1.detener(); p.atender(0.15)
    assert "APAGADO_LIMPIO" in p.registro()
    a2 = ActuadorWearable(ip_wearable="127.0.0.1", espaciado_reenvios_ms=10)
    a2.iniciar(); p.atender(0.15)                 # el saludo de inicio (tipo 10)
    assert "se anula el anuncio" in p.registro()
    # Corte imprevisto: nadie llama a a2.detener(), no sale ningun tipo 99
    perder_pi_hasta_respaldo(p); p.conectar(); p.vueltas(40)
    assert p.lib.banco_tareas_creadas() == 1, "tras relanzar la Pi, una caida imprevista debe avisar"

def caso_aviso_fallido_se_reintenta():
    p = Pulsera(); p.asociar_a_pi()
    perder_pi_hasta_respaldo(p); p.conectar(); p.vueltas(40)
    p.lib.banco_http_resultado(0); p.lib.banco_correr_tarea(); p.vueltas(40)
    assert p.estado() == TESTIGO and p.lib.banco_http_envios() == 1
    p.vueltas(20100); p.vueltas(10100)          # revisa la Pi, no esta, vuelve al respaldo
    p.conectar(); p.vueltas(40)
    assert p.lib.banco_tareas_creadas() == 2, "debio reintentar el aviso"
    p.lib.banco_http_resultado(1); p.lib.banco_correr_tarea(); p.vueltas(40)
    p.vueltas(20100); p.vueltas(10100); p.conectar(); p.vueltas(40)
    assert p.lib.banco_tareas_creadas() == 2, "ya enviado: no se repite"

def caso_sin_red_de_respaldo():
    """El respaldo tampoco aparece: no hay tarea, queda en testigo y sigue revisando."""
    p = Pulsera(); p.asociar_a_pi()
    perder_pi_hasta_respaldo(p)
    p.vueltas(14900); assert p.estado() == CONMUTAR
    p.vueltas(200);   assert p.estado() == TESTIGO and p.lib.banco_tareas_creadas() == 0
    p.vueltas(20100); assert p.ssid() == "NeuroDrive_AP"
    p.vueltas(10100); assert p.ssid() == "RedDeRespaldo"
    p.conectar(); p.vueltas(15100)
    assert p.lib.banco_tareas_creadas() == 1, "cuando aparece el respaldo, avisa"

def caso_aviso_colgado_no_traba():
    """Si la tarea del aviso no termina, la maquina sigue; no lanza otra en paralelo."""
    p = Pulsera(); p.asociar_a_pi()
    perder_pi_hasta_respaldo(p); p.conectar(); p.vueltas(40)
    p.vueltas(44900); assert p.estado() == CONMUTAR
    p.vueltas(200);   assert p.estado() == TESTIGO
    p.vueltas(30000); assert p.ssid() == "RedDeRespaldo", "no debe cortar la red con el aviso en curso"
    assert p.lib.banco_tareas_creadas() == 1
    # Cuando la tarea por fin termina, la revision demorada se hace en el acto
    p.lib.banco_correr_tarea(); p.vueltas(100)
    assert p.ssid() == "NeuroDrive_AP"


if __name__ == "__main__":
    casos = {k[5:]: v for k, v in globals().items() if k.startswith("caso_")}
    if len(sys.argv) > 1:
        casos[sys.argv[1]](); sys.exit(0)
    ok = 0
    for nombre in casos:
        r = subprocess.run([sys.executable, __file__, nombre], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            ok += 1; print(f"  [OK]   {nombre}")
        else:
            print(f"  [FAIL] {nombre}\n" + "\n".join("         " + l for l in r.stderr.strip().split("\n")[-6:]))
    print(f"\n  Resumen: {ok}/{len(casos)} casos pasaron")
    sys.exit(0 if ok == len(casos) else 1)
