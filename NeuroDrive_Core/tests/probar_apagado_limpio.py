#!/usr/bin/env python3
"""
Prueba del escenario "apagado limpio" de la Contingencia-b.
Manda a la pulsera el mensaje tipo=99 (apagado limpio) que normalmente
enviaria el ActuadorWearable.detener() de la Pi. Sirve para probar que, tras
recibirlo, la pulsera NO manda el Telegram al perder la conexion.

USO (en la Pi, con la pulsera OPERANDO con el AP):
    python3 probar_apagado_limpio.py
    # (opcional) python3 probar_apagado_limpio.py <IP_PULSERA>   default 192.168.4.20

Despues de correrlo, baja el AP de la Pi (./neurodrive_ap.sh down) y verifica
en el serial de la pulsera que dice "perdida LEGITIMA ... NO se avisa".
"""
import socket, sys, json, time

ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.20"
PUERTO = 5006

msg = {"v": 1, "tipo": 99, "intensidad": 0, "duracion_ms": 0,
       "id_secuencia": None, "id_paquete": 99001}
datos = json.dumps(msg, separators=(",", ":")).encode()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Lo mandamos 3 veces (como los criticos), por si se pierde alguno.
for _ in range(3):
    s.sendto(datos, (ip, PUERTO))
    time.sleep(0.05)
s.close()
print(f"Enviado APAGADO_LIMPIO (tipo=99) a {ip}:{PUERTO}")
print("Ahora baja el AP:  ./neurodrive_ap.sh down")
print("La pulsera NO deberia avisar por Telegram (perdida legitima).")
