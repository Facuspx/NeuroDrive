#!/usr/bin/env python3
"""
Receptor de telemetria de la pulsera, sin el nucleo.

Escucha el puerto 5005 y muestra cada mensaje que llega, comprobando que el
formato sea el del protocolo. No importa ningun modulo de NeuroDrive.

No correr junto con main.py: los dos escuchan en el puerto 5005.

USO:
    python3 herramientas/recibir_telemetria.py

Avisa si pasan mas de 5 s sin telemetria (el nucleo declara caida a la pulsera
a los 10 s). Las respuestas a una verificacion se muestran aparte. Ctrl+C para
salir.
"""
import json
import socket
import time

PUERTO_TELEMETRIA = 5005
CAMPOS_ESPERADOS = {"v", "msg", "bpm", "ack_recibido",
                    "secuencia_replicada", "bateria", "id_paquete"}
ALERTA_SIN_TELEMETRIA_S = 5.0


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", PUERTO_TELEMETRIA))
    sock.settimeout(1.0)

    print(f"Escuchando telemetria en el puerto {PUERTO_TELEMETRIA}...")
    print("(Ctrl+C para salir)\n")

    ultimo = time.monotonic()
    ultimo_id = None
    recibidos = 0

    try:
        while True:
            try:
                datos, origen = sock.recvfrom(1024)
            except socket.timeout:
                if time.monotonic() - ultimo > ALERTA_SIN_TELEMETRIA_S:
                    print(f"  [!] {ALERTA_SIN_TELEMETRIA_S:.0f} s sin telemetria "
                          f"(a los 10 s el nucleo declara caida a la pulsera)")
                    ultimo = time.monotonic()   # no repetir el aviso cada segundo
                continue

            ahora = time.monotonic()
            dt = ahora - ultimo

            try:
                obj = json.loads(datos.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                print(f"  [X] datagrama que no es JSON de {origen[0]}: {e}")
                continue
            if not isinstance(obj, dict):
                print(f"  [X] JSON que no es un objeto de {origen[0]}: {obj!r}")
                continue

            if obj.get("msg") == "ack":
                marca = "correcta" if obj.get("secuencia_correcta") else "incorrecta"
                print(f"      respuesta a la verificacion {obj.get('id_secuencia')}: {marca}, "
                      f"{obj.get('tiempo_respuesta_ms')} ms")
                continue

            ultimo = ahora
            recibidos += 1

            # Verificar formato
            faltan = CAMPOS_ESPERADOS - set(obj.keys())
            extra = set(obj.keys()) - CAMPOS_ESPERADOS
            estado = "OK"
            if obj.get("msg") != "telemetria":
                estado = f"msg inesperado: {obj.get('msg')}"
            elif faltan:
                estado = f"FALTAN campos: {sorted(faltan)}"
            elif extra:
                estado = f"campos DE MAS: {sorted(extra)}"

            # Saltos o repeticiones en id_paquete
            id_paq = obj.get("id_paquete")
            salto = ""
            if ultimo_id is not None and id_paq is not None:
                if id_paq == ultimo_id:
                    salto = "  [id repetido]"
                elif id_paq < ultimo_id:
                    salto = f"  [la pulsera se reinicio: {ultimo_id}->{id_paq}]"
                elif id_paq != ultimo_id + 1:
                    salto = f"  [salto {ultimo_id}->{id_paq}: se perdieron {id_paq - ultimo_id - 1}]"
            ultimo_id = id_paq

            bateria = obj.get("bateria")
            print(f"#{recibidos:04d} dt={dt:4.1f}s de {origen[0]}  "
                  f"bpm={obj.get('bpm')} bat={'--' if bateria is None else bateria}% "
                  f"id={id_paq}  [{estado}]{salto}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print(f"\nlisto. Telemetrias recibidas: {recibidos}")


if __name__ == "__main__":
    main()
