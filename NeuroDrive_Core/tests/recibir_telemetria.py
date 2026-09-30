#!/usr/bin/env python3
"""
Receptor de telemetria para probar el Modulo 4a del firmware.
Corre en la Pi, escucha el puerto 5005 (donde la pulsera manda el heartbeat) y
muestra cada latido que llega, verificando que el formato del protocolo sea el
esperado. Es un banco AISLADO: no usa el Core.

USO:
    python3 recibir_telemetria.py

Muestra cada telemetria y avisa si pasa demasiado tiempo sin recibir (posible
caida del heartbeat). Ctrl+C para salir.
"""
import socket
import json
import time

PUERTO_TELEMETRIA = 5005
CAMPOS_ESPERADOS = {"v", "msg", "bpm", "ack_recibido",
                    "secuencia_replicada", "bateria", "id_paquete"}
ALERTA_SIN_LATIDO_S = 5.0     # la Pi real corta a los ~10s; avisamos antes


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PUERTO_TELEMETRIA))
    sock.settimeout(1.0)

    print(f"Escuchando telemetria en el puerto {PUERTO_TELEMETRIA}...")
    print("(Ctrl+C para salir)\n")

    ultimo = time.monotonic()
    ultimo_id = None
    recibidos = 0

    while True:
        try:
            datos, origen = sock.recvfrom(1024)
        except socket.timeout:
            # Sin paquete en 1s: chequear si se cayo el heartbeat.
            if time.monotonic() - ultimo > ALERTA_SIN_LATIDO_S:
                print(f"  [!] {ALERTA_SIN_LATIDO_S:.0f}s sin heartbeat "
                      f"(la Pi real declararia la pulsera caida)")
                ultimo = time.monotonic()   # no spamear
            continue
        except KeyboardInterrupt:
            break

        ahora = time.monotonic()
        dt = ahora - ultimo
        ultimo = ahora
        recibidos += 1

        try:
            obj = json.loads(datos.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            print(f"  [X] paquete no-JSON de {origen[0]}: {e}")
            continue

        # Verificar formato
        faltan = CAMPOS_ESPERADOS - set(obj.keys())
        extra = set(obj.keys()) - CAMPOS_ESPERADOS
        estado = "OK"
        if obj.get("msg") != "telemetria":
            estado = f"msg inesperado: {obj.get('msg')}"
        elif faltan:
            estado = f"FALTAN campos: {faltan}"
        elif extra:
            estado = f"campos DE MAS: {extra}"

        # Detectar saltos/duplicados en id_paquete
        id_paq = obj.get("id_paquete")
        salto = ""
        if ultimo_id is not None and id_paq is not None:
            if id_paq == ultimo_id:
                salto = "  [dup id]"
            elif id_paq != ultimo_id + 1:
                salto = f"  [salto {ultimo_id}->{id_paq}]"
        ultimo_id = id_paq

        print(f"#{recibidos:04d} dt={dt:4.1f}s de {origen[0]}  "
              f"bpm={obj.get('bpm')} bat={obj.get('bateria')}% "
              f"ack={obj.get('ack_recibido')} id={id_paq}  [{estado}]{salto}")

    sock.close()
    print(f"\nlisto. Telemetrias recibidas: {recibidos}")


if __name__ == "__main__":
    main()
