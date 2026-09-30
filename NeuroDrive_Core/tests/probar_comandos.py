#!/usr/bin/env python3
"""
Prueba del firmware con desafio ACK (Modulo 6).
Corre en la Pi. Manda comandos UDP a la pulsera Y escucha en el 5005 las
respuestas (ack y telemetria) que la pulsera devuelve, para ver el ciclo
completo del desafio cognitivo. Banco AISLADO: no usa el Core.
 
USO:
    python3 probar_comandos_ack.py <IP_DE_LA_PULSERA>
    # ejemplo:  python3 probar_comandos_ack.py 192.168.1.51
 
Menu:
    l  -> VIBRAR_LEVE (30%, 1.5s)
    m  -> VIBRAR_MEDIO (60%, 2.5s)
    f  -> VIBRAR_FUERTE (100%, continuo)
    a  -> APAGAR_TODO
    k  -> DESAFIO ACK: la pulsera vibra K veces, conta y toca el pad K.
          El script se queda escuchando la respuesta 'ack' que vuelve.
    d  -> dedup: VIBRAR_FUERTE x3 con el mismo id_paquete
    q  -> salir
 
Mientras corre, imprime en segundo plano toda telemetria/ack que llega.
"""
import socket
import sys
import json
import time
import threading
 
PUERTO_COMANDOS = 5006     # la pulsera escucha aca (Pi -> ESP)
PUERTO_ESCUCHA  = 5005     # la Pi escucha aca (ESP -> Pi): telemetria y ack
 
VIBRAR_LEVE, VIBRAR_MEDIO, VIBRAR_FUERTE = 1, 2, 3
SECUENCIA_ACK, APAGAR_TODO = 8, 10
 
_seguir = True

def escuchar(sock_rx):
    """Hilo que muestra todo lo que llega de la pulsera (ack y telemetria)."""
    while _seguir:
        try:
            datos, origen = sock_rx.recvfrom(1024)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            obj = json.loads(datos.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if obj.get("msg") == "ack":
            marca = "CORRECTO" if obj.get("secuencia_correcta") else "INCORRECTO"
            print(f"\n  <== ACK id_secuencia={obj.get('id_secuencia')} "
                  f"{marca}  tiempo_respuesta={obj.get('tiempo_respuesta_ms')}ms")
            print("cmd> ", end="", flush=True)
        # La telemetria llega cada 2s; la mostramos discreta para no tapar el menu.
        elif obj.get("msg") == "telemetria":
            pass  # descomentar para verla:  print("  (telemetria)", obj)

def main():
    global _seguir
    if len(sys.argv) < 2:
        print("Uso: python3 probar_comandos_ack.py <IP_DE_LA_PULSERA>")
        sys.exit(1)
    ip = sys.argv[1]
 
    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.bind(("0.0.0.0", PUERTO_ESCUCHA))
    sock_rx.settimeout(1.0)
 
    hilo = threading.Thread(target=escuchar, args=(sock_rx,), daemon=True)
    hilo.start()
 
    id_paquete = [0]
    id_secuencia = [0]
 
    def mandar(tipo, intensidad, duracion_ms, id_paq, id_sec=None):
        obj = {"v": 1, "tipo": tipo, "intensidad": intensidad,
               "duracion_ms": duracion_ms, "id_secuencia": id_sec,
               "id_paquete": id_paq}
        sock_tx.sendto(json.dumps(obj, separators=(",", ":")).encode(), (ip, PUERTO_COMANDOS))
        print(f"  ==> enviado: {obj}")
 
    print(f"Banco de pruebas ACK -> pulsera {ip}:{PUERTO_COMANDOS}")
    print("Escuchando respuestas en el {}.".format(PUERTO_ESCUCHA))
    print("Comandos: l/m/f=vibrar  a=apagar  k=DESAFIO ACK  d=dedup  q=salir")

    while True:
        try:
            cmd = input("cmd> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
 
        if cmd == "q":
            break
        elif cmd == "l":
            id_paquete[0] += 1; mandar(VIBRAR_LEVE, 30, 1500, id_paquete[0])
        elif cmd == "m":
            id_paquete[0] += 1; mandar(VIBRAR_MEDIO, 60, 2500, id_paquete[0])
        elif cmd == "f":
            id_paquete[0] += 1; mandar(VIBRAR_FUERTE, 100, 0, id_paquete[0])
        elif cmd == "a":
            id_paquete[0] += 1; mandar(APAGAR_TODO, 0, 0, id_paquete[0])
        elif cmd == "k":
            id_paquete[0] += 1; id_secuencia[0] += 1
            print(f"  [desafio] enviando SECUENCIA_ACK id_secuencia={id_secuencia[0]}")
            print("  -> conta las vibraciones y toca el pad correspondiente")
            mandar(SECUENCIA_ACK, 80, 0, id_paquete[0], id_sec=id_secuencia[0])
        elif cmd == "d":
            id_paquete[0] += 1; fijo = id_paquete[0]
            print(f"  [dedup] VIBRAR_FUERTE x3 con id_paquete={fijo}")
            for _ in range(3):
                mandar(VIBRAR_FUERTE, 100, 800, fijo); time.sleep(0.05)
        else:
            print("  comando desconocido (l/m/f/a/k/d/q)")

    _seguir = False
    sock_tx.close()
    sock_rx.close()
    print("listo.")
 
 
if __name__ == "__main__":
    main()