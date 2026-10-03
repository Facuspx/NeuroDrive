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
    --sesion-nueva        No retomar la sesion anterior: iniciar en NORMAL
                          aunque haya un estado guardado reciente. Util para
                          que los ensayos partan siempre del mismo punto.

Estado entre sesiones:
    El nucleo guarda su estado en config.sesion.archivo_estado. Si se lo
    relanza dentro de config.sesion.ttl_sesion_seg y la sesion anterior dejo
    indicios de somnolencia, arranca en PRE_ALERTA y conserva los episodios
    severos y los bostezos recientes. Ver NeuroDrive_Core/persistencia_sesion.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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
from NeuroDrive_Core.persistencia_sesion import PersistenciaSesion, resolver_arranque
from NeuroDrive_Core.actuadores.buzzer import ActuadorBuzzer
from NeuroDrive_Wearable.actuador_wearable import ActuadorWearable
from NeuroDrive_Wearable.receptor_wearable import ReceptorWearable


# Raiz del proyecto: las rutas relativas del config se resuelven contra ella,
# no contra el directorio desde el que se lance el programa.
RAIZ_PROYECTO = Path(__file__).resolve().parent


def _ruta_estado_sesion(config) -> Path:
    ruta = Path(config.sesion.archivo_estado)
    return ruta if ruta.is_absolute() else RAIZ_PROYECTO / ruta


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
    parser.add_argument("--sesion-nueva", action="store_true",
                        help="No retomar la sesion anterior; iniciar en NORMAL")
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

    # ---- Sesion anterior ----
    config_fsm = ConfigFSM.desde_config(config)
    persistencia = PersistenciaSesion(
        _ruta_estado_sesion(config), ttl_seg=config.sesion.ttl_sesion_seg
    )
    if args.sesion_nueva:
        persistencia.borrar()
    arranque = resolver_arranque(
        persistencia.cargar(),
        ahora=time.time(),
        ventana_episodios_seg=config_fsm.ventana_episodios_seg,
        ventana_bostezos_seg=config.boca.ventana_bostezos_seg,
    )
    print(f"  Sesion:         {arranque.descripcion}")
    print()

    # ---- Componentes del Core ----
    gestor = GestorEventos(config)
    pre_fsm = PreFSM(config)
    pre_fsm.restaurar_bostezos(arranque.bostezos)
    fsm = FSM(
        config_fsm,
        estado_inicial=arranque.estado_inicial,
        episodios_previos=arranque.episodios_severos,
    )

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

    orq = Orquestador(gestor, pre_fsm, fsm, despachador, receptor,
                      persistencia=persistencia)

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
