#!/usr/bin/env python3
"""
Banco de pruebas del firmware de la pulsera, sin el nucleo.

Envia ordenes UDP a la pulsera y muestra las respuestas y la telemetria que
devuelve. Sirve para verificar el firmware de manera aislada: no importa
ningun modulo de NeuroDrive, de modo que un error del lado Pi no puede
enmascarar uno del firmware (ni al reves).

No correr junto con main.py: los dos escuchan en el puerto 5005.

USO (desde la raiz del repo, con la pulsera asociada al punto de acceso):
    python3 herramientas/probar_comandos.py                 # 192.168.4.20
    python3 herramientas/probar_comandos.py <IP_PULSERA>

Menu:
    l  VIBRAR_LEVE    (30 %, 1,5 s)
    m  VIBRAR_MEDIO   (60 %, 2,5 s)
    f  VIBRAR_FUERTE  (100 %, hasta nueva orden)
    a  APAGAR_TODO
    k  VERIFICACION con ventana de 7 s: la pulsera vibra K veces; contar y
       tocar el pad K. La respuesta aparece en pantalla (llega repetida:
       se muestra cada copia).
    K  VERIFICACION sin ventana (duracion_ms = 0): la pulsera usa la suya
       por defecto, 10 s
    d  descarte de repetidas: VIBRAR_FUERTE tres veces con el mismo id
    9  APAGADO_LIMPIO (tipo 99): la proxima perdida del enlace es esperada
    t  mostrar / ocultar la telemetria periodica
    q  salir
"""
import json
import random
import socket
import sys
import threading
import time

PUERTO_ORDENES = 5006      # la pulsera escucha aca (Pi -> ESP)
PUERTO_ESCUCHA = 5005      # la Pi escucha aca (ESP -> Pi): telemetria y respuestas

VIBRAR_LEVE, VIBRAR_MEDIO, VIBRAR_FUERTE = 1, 2, 3
SECUENCIA_ACK, APAGAR_TODO, APAGADO_LIMPIO = 8, 10, 99

VENTANA_PRUEBA_MS = 7000

_seguir = True
_ver_telemetria = False


def escuchar(sock_rx: socket.socket) -> None:
    """Hilo que muestra lo que llega de la pulsera."""
    while _seguir:
        try:
            datos, _origen = sock_rx.recvfrom(1024)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            obj = json.loads(datos.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            print(f"\n  <== datagrama que no es JSON: {datos[:60]!r}")
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("msg") == "ack":
            marca = "CORRECTO" if obj.get("secuencia_correcta") else "INCORRECTO"
            print(f"\n  <== RESPUESTA id_secuencia={obj.get('id_secuencia')} {marca}  "
                  f"tiempo_respuesta={obj.get('tiempo_respuesta_ms')} ms")
            print("cmd> ", end="", flush=True)
        elif obj.get("msg") == "telemetria" and _ver_telemetria:
            print(f"\n  <== telemetria bpm={obj.get('bpm')} bat={obj.get('bateria')}% "
                  f"id={obj.get('id_paquete')}")
            print("cmd> ", end="", flush=True)


def main() -> int:
    global _seguir, _ver_telemetria
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.20"

    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock_rx.bind(("0.0.0.0", PUERTO_ESCUCHA))
    except OSError as e:
        print(f"No se pudo escuchar en el puerto {PUERTO_ESCUCHA}: {e}")
        print("¿Esta corriendo main.py? Este banco y el nucleo no pueden convivir.")
        return 1
    sock_rx.settimeout(1.0)
    threading.Thread(target=escuchar, args=(sock_rx,), daemon=True).start()

    # Base aleatoria: la pulsera recuerda los ultimos 16 id_paquete. Si cada
    # ejecucion empezara en 1, las primeras ordenes de la segunda corrida se
    # descartarian como repetidas de la primera.
    id_paquete = random.randrange(1, 1 << 30)
    id_secuencia = 0

    def mandar(tipo, intensidad=0, duracion_ms=0, id_paq=None, id_sec=None):
        obj = {"v": 1, "tipo": tipo, "intensidad": intensidad,
               "duracion_ms": duracion_ms, "id_secuencia": id_sec,
               "id_paquete": id_paq}
        sock_tx.sendto(json.dumps(obj, separators=(",", ":")).encode(), (ip, PUERTO_ORDENES))
        print(f"  ==> enviado: {obj}")

    print(f"Banco de pruebas -> pulsera {ip}:{PUERTO_ORDENES}. "
          f"Respuestas en el puerto {PUERTO_ESCUCHA}.")
    print("Ordenes: l/m/f vibrar | a apagar | k/K verificacion | d repetidas | "
          "9 apagado limpio | t telemetria | q salir")

    while True:
        try:
            cmd = input("cmd> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("q", "Q"):
            break
        id_paquete += 1
        if cmd == "l":
            mandar(VIBRAR_LEVE, 30, 1500, id_paquete)
        elif cmd == "m":
            mandar(VIBRAR_MEDIO, 60, 2500, id_paquete)
        elif cmd == "f":
            mandar(VIBRAR_FUERTE, 100, 0, id_paquete)
        elif cmd == "a":
            mandar(APAGAR_TODO, id_paq=id_paquete)
        elif cmd in ("k", "K"):
            id_secuencia += 1
            ventana = VENTANA_PRUEBA_MS if cmd == "k" else 0
            print(f"  [verificacion] id_secuencia={id_secuencia}, ventana="
                  f"{ventana if ventana else 'la de la pulsera'} ms: contar los pulsos y tocar ese pad")
            mandar(SECUENCIA_ACK, 0, ventana, id_paquete, id_sec=id_secuencia)
        elif cmd == "d":
            print(f"  [repetidas] VIBRAR_FUERTE x3 con id_paquete={id_paquete}: debe vibrar UNA vez")
            for _ in range(3):
                mandar(VIBRAR_FUERTE, 100, 800, id_paquete)
                time.sleep(0.05)
        elif cmd == "9":
            for _ in range(3):
                mandar(APAGADO_LIMPIO, id_paq=id_paquete)
                time.sleep(0.05)
            print("  La pulsera no deberia avisar si ahora pierde el punto de acceso.")
            print("  Una orden posterior anula esta marca (firmware con el tipo 99 corregido).")
        elif cmd == "t":
            _ver_telemetria = not _ver_telemetria
            print(f"  telemetria {'visible' if _ver_telemetria else 'oculta'}")
        else:
            print("  orden desconocida (l m f a k K d 9 t q)")

    _seguir = False
    sock_tx.close()
    sock_rx.close()
    print("listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
