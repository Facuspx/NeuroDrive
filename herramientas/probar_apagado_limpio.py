#!/usr/bin/env python3
"""
Ensayo del apagado ordenado, sin el nucleo.

Envia a la pulsera la orden tipo 99, que en operacion normal transmite
ActuadorWearable.detener() cuando main.py se cierra de manera ordenada.
Sirve para comprobar que, tras recibirla, la pulsera NO emite el aviso remoto
al perder el punto de acceso.

USO (con la pulsera operando contra el punto de acceso):
    python3 herramientas/probar_apagado_limpio.py                # 192.168.4.20
    python3 herramientas/probar_apagado_limpio.py <IP_PULSERA>

Despues bajar el punto de acceso (./neurodrive_ap.sh down) y verificar en el
monitor serie de la pulsera el mensaje de perdida esperada, sin aviso.

Para el caso contrario (interrupcion imprevista) no hace falta ningun script:
alcanza con bajar el punto de acceso sin haber enviado esta orden.
"""
import json
import random
import socket
import sys
import time

ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.20"
PUERTO = 5006

# id_paquete aleatorio: con un valor fijo, la segunda ejecucion seguida seria
# descartada por la pulsera como repeticion de la primera.
msg = {"v": 1, "tipo": 99, "intensidad": 0, "duracion_ms": 0,
       "id_secuencia": None, "id_paquete": random.randrange(1, 1 << 30)}
datos = json.dumps(msg, separators=(",", ":")).encode()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Tres copias con el mismo id_paquete, como las ordenes criticas del nucleo.
for _ in range(3):
    s.sendto(datos, (ip, PUERTO))
    time.sleep(0.05)
s.close()
print(f"Enviado APAGADO_LIMPIO (tipo 99, id_paquete {msg['id_paquete']}) a {ip}:{PUERTO}")
print("Ahora bajar el punto de acceso:  ./neurodrive_ap.sh down")
print("La pulsera no deberia emitir el aviso remoto (perdida esperada).")
