#!/usr/bin/env python3
"""
NeuroDrive - Orquestador principal (main.py)
=============================================

Levanta el lado Core completo del sistema y lo deja corriendo:

    Gestor -> PreFSM -> FSM -> Despachador -> [Buzzer, Wearable]
                                                   ^
                          ReceptorWearable (UDP -> cola) alimenta al Gestor

La VISION corre como proceso aparte (tiene su propia camara). Este main NO
la lanza; la vision escribe en la cola /neurodrive_vision que el Gestor lee.

Uso tipico en la Raspberry Pi (dos o tres terminales):

    Terminal 1 - el Core (este script). Arrancar PRIMERO:
        cd ~/Desktop/NeuroDrive
        python main.py

    Terminal 2 - la vision real con la camara:
        python -m NeuroDrive_Vision.test_vision --mq-real

    Terminal 3 - la pulsera. Al principio, el SIMULADOR (sin ESP):
        python -m NeuroDrive_Wearable.simulador_pulsera --pi 127.0.0.1

Cuando tengas el ESP32 flasheado, no necesitas el simulador: el ESP se
conecta al AP de la Pi y ocupa su lugar.

Opciones:
    --duracion-max SEG    Detener tras SEG segundos (0 = sin limite)
    --periodo-resumen SEG Cada cuanto imprimir el resumen (default 5)
    --sin-buzzer          No registrar el buzzer (si no esta conectado)
    --buzzer-simulado     Usar un buzzer simulado (para probar sin GPIO)
    --sin-wearable        No levantar actuador ni receptor del wearable
    --wearable-ip IP      Sobrescribe config.red.ip_wearable (ej: 127.0.0.1
                          para probar contra el simulador local)
"""

from __future__ import annotations

import argparse
import logging
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from NeuroDrive_Core.config_loader import cargar_config
from NeuroDrive_Core.gestor_eventos import GestorEventos
from NeuroDrive_Core.pre_fsm import PreFSM
from NeuroDrive_Core.fsm import FSM, ConfigFSM
from NeuroDrive_Core.despachador import DespachadorComandos, ActuadorSimulado
from NeuroDrive_Core.orquestador import Orquestador
from NeuroDrive_Core.actuadores.buzzer import ActuadorBuzzer
from NeuroDrive_Wearable.actuador_wearable import ActuadorWearable
from NeuroDrive_Wearable.receptor_wearable import ReceptorWearable


def _construir_config_fsm(config) -> ConfigFSM:
    """
    Arma la ConfigFSM a partir del Config estructurado (los timeouts de ACK
    viven en [wearable], los umbrales de escalada en [fsm]). getattr con
    default protege contra claves que pudieran faltar en config.yaml.
    """
    fsm = config.fsm
    wea = config.wearable
    return ConfigFSM(
        tiempo_para_bajar_estado_seg=getattr(fsm, "tiempo_para_bajar_estado_seg", 60.0),
        timeout_ack_leve_seg=getattr(wea, "timeout_ack_leve_seg", 30.0),
        timeout_ack_medio_seg=getattr(wea, "timeout_ack_medio_seg", 20.0),
        timeout_ack_critico_seg=getattr(wea, "timeout_ack_critico_seg", 15.0),
        max_microsuenos_ventana_corta=getattr(fsm, "max_microsuenos_ventana_corta", 1),
        max_bostezos_ventana_corta=getattr(fsm, "max_bostezos_ventana_corta", 3),
        max_cabeceos_ventana_corta=getattr(fsm, "max_cabeceos_ventana_corta", 1),
        calentamiento_senales_seg=getattr(fsm, "calentamiento_senales_seg", 60.0),
        persistencia_senales_leves_seg=getattr(fsm, "persistencia_senales_leves_seg", 20.0),
        perclos_confirmado=getattr(fsm, "perclos_confirmado", 0.35),
        perclos_confirmado_sostenido_seg=getattr(fsm, "perclos_confirmado_sostenido_seg", 30.0),
        max_eventos_severos_ventana=getattr(fsm, "max_eventos_severos_ventana", 3),
        ventana_episodios_seg=getattr(fsm, "ventana_episodios_seg", 900.0),
        umbral_respuesta_lenta_ms=getattr(fsm, "umbral_respuesta_lenta_ms", 5000),
    )


def _configurar_logging(config) -> None:
    nivel = getattr(getattr(config, "logging", None), "nivel", "INFO")
    logging.basicConfig(
        level=getattr(logging, nivel, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NeuroDrive - Orquestador principal")
    parser.add_argument("--duracion-max", type=float, default=0.0)
    parser.add_argument("--periodo-resumen", type=float, default=5.0)
    parser.add_argument("--sin-buzzer", action="store_true")
    parser.add_argument("--buzzer-simulado", action="store_true")
    parser.add_argument("--sin-wearable", action="store_true")
    parser.add_argument("--wearable-ip", type=str, default=None)
    args = parser.parse_args(argv)

    print("=" * 64)
    print("  NeuroDrive - Sistema completo (Core + actuadores + wearable)")
    print("=" * 64)

    # ---- Config ----
    try:
        config = cargar_config()
    except Exception as e:
        print(f"ERROR cargando config.yaml: {e}")
        return 1
    _configurar_logging(config)
    log = logging.getLogger("NeuroDrive.main")

    ip_wearable = args.wearable_ip or config.red.ip_wearable
    print(f"  Cola vision:    {config.ipc.cola_vision}")
    print(f"  Cola wearable:  {config.ipc.cola_wearable}")
    print(f"  Buzzer GPIO:    {config.actuadores.buzzer_gpio_pin}")
    print(f"  Wearable:       {ip_wearable}:{config.red.puerto_udp_envio} (envio) "
          f"/ :{config.red.puerto_udp_escucha} (escucha)")
    print()

    # ---- Componentes del Core ----
    gestor = GestorEventos(config)
    pre_fsm = PreFSM(config)
    fsm = FSM(_construir_config_fsm(config))

    # ---- Despachador + actuadores ----
    despachador = DespachadorComandos(capacidad_cola=64)

    if args.buzzer_simulado:
        despachador.registrar_actuador(ActuadorSimulado("buzzer"))
        log.info("Buzzer SIMULADO registrado")
    elif not args.sin_buzzer:
        despachador.registrar_actuador(
            ActuadorBuzzer(pin=config.actuadores.buzzer_gpio_pin)
        )

    receptor = None
    if not args.sin_wearable:
        despachador.registrar_actuador(ActuadorWearable(
            ip_wearable=ip_wearable,
            puerto_envio=config.red.puerto_udp_envio,
            reenvios_criticos=config.red.reenvios_comandos_criticos,
            espaciado_reenvios_ms=config.red.espaciado_reenvios_ms,
        ))
        receptor = ReceptorWearable(
            puerto_escucha=config.red.puerto_udp_escucha,
            nombre_cola=config.ipc.cola_wearable,
            capacidad_cola=config.ipc.capacidad_cola,
            tamano_max_mensaje=config.ipc.tamano_max_mensaje_bytes,
            id_dispositivo=config.identificadores.id_wearable,
        )

    orq = Orquestador(gestor, pre_fsm, fsm, despachador, receptor)

    # ---- Correr ----
    try:
        orq.iniciar()
    except Exception as e:
        print(f"ERROR FATAL al iniciar el orquestador: {e}")
        import traceback; traceback.print_exc()
        return 1

    print("Sistema iniciado. Esperando eventos de vision y wearable...")
    print("(vision: python -m NeuroDrive_Vision.test_vision --mq-real)")
    print("(pulsera: python -m NeuroDrive_Wearable.simulador_pulsera --pi 127.0.0.1)")
    print("Ctrl+C para detener.\n")

    codigo = 0
    try:
        orq.correr(duracion_max=args.duracion_max,
                   periodo_resumen=args.periodo_resumen)
    except KeyboardInterrupt:
        print("\nInterrupcion por teclado.")
    except Exception as e:
        print(f"\nERROR en el bucle principal: {e}")
        import traceback; traceback.print_exc()
        codigo = 1
    finally:
        print("\nDeteniendo el sistema...")
        orq.detener()

    # ---- Reporte final ----
    print("\n" + "=" * 64)
    print("  REPORTE FINAL")
    print("=" * 64)
    print(f"  {orq.resumen()}")
    g = gestor.stats
    print(f"  Gestor: vision={g.mensajes_recibidos_vision} "
          f"wearable={g.mensajes_recibidos_wearable} "
          f"invalidos={g.mensajes_invalidos} "
          f"duplicados={g.duplicados_descartados} "
          f"fallos_emitidos={g.fallos_sensor_emitidos}")
    if receptor is not None:
        r = receptor.stats
        print(f"  Receptor wearable: recibidos={r.paquetes_recibidos} "
              f"telemetrias={r.telemetrias} acks={r.acks} "
              f"invalidos={r.invalidos} publicados={r.publicados}")
    print("=" * 64)
    return codigo


if __name__ == "__main__":
    sys.exit(main())
