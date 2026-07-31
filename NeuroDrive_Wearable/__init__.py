"""
NeuroDrive_Wearable - Comunicacion con la pulsera ESP32-S3.

Dos flujos:
  - Pi -> ESP32 (comandos): ActuadorWearable envia por UDP los ComandoActuador
    de vibracion / desafio ACK (implementa ActuadorBase, lo gobierna el
    Despachador).
  - ESP32 -> Pi (telemetria/ACK): ReceptorWearable escucha UDP, arma
    Envelope(EventoWearable | EventoAckWearable) y lo publica en la cola
    POSIX MQ del wearable, que el Gestor ya consume.

El formato en el cable (JSON) esta centralizado en protocolo.py.
El simulador_pulsera.py hace de ESP32 para testear todo el lado Pi sin
flashear nada.
"""
