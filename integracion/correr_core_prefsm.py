"""
correr_core_prefsm.py - Core hasta el Pre-FSM (Etapa III)
==========================================================

Levanta el lado Core del sistema SOLO hasta el Pre-FSM (sin FSM ni
actuadores) y muestra en consola lo que el Pre-FSM va detectando a
partir de los eventos reales que envia NeuroDrive_Vision.

Sirve para validar la Etapa III de integracion: que la vision real
(camara) produce eventos que el Pre-FSM interpreta correctamente
(parpadeos/min, bostezos, cabeceos, PERCLOS).

Arquitectura:
    GestorEventos  --(Envelope)-->  PreFSM  --(EventoProcesado)-->  consola

    El GestorEventos abre la cola POSIX, levanta sus hilos lectores, y
    entrega Envelopes ya validados y deduplicados. El PreFSM los
    transforma en EventoProcesado. Aca NO instanciamos la FSM: eso es
    la Etapa IV.

Como usarlo (dos terminales en la misma Raspberry Pi):

    Terminal 1 - el Core (este script). Arrancar PRIMERO:
        cd ~/NeuroDrive
        python -m integracion.correr_core_prefsm

    Terminal 2 - la vision real con la camara:
        cd ~/NeuroDrive
        python -m NeuroDrive_Vision.test_vision --mq-real

Despues, frente a la camara:
    - Parpadear normal: ver subir "parpadeos/min".
    - Bostezar (boca bien abierta > 1 s): ver "BOSTEZO detectado".
    - Bajar la cabeza > 1 s: ver "CABECEO detectado".
    - Cerrar los ojos varios segundos: ver "MICROSUENO" y subir PERCLOS.

Cortar con Ctrl+C: el script cierra todo de forma limpia.

Opciones:
    --duracion-max SEG   Detener automaticamente tras SEG segundos (0 = sin limite)
    --periodo-resumen SEG  Cada cuanto imprimir el resumen de metricas (default 5)
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from NeuroDrive_Core.config_loader import cargar_config
from NeuroDrive_Core.gestor_eventos import GestorEventos
from NeuroDrive_Core.pre_fsm import PreFSM
from common.contratos import EventoProcesado, NivelRiesgoBPM


# =============================================================================
# Acumulador de lo que detecta el Pre-FSM (para el resumen periodico)
# =============================================================================

class _Acumulador:
    """Lleva la cuenta de eventos discretos detectados desde el ultimo resumen."""

    def __init__(self) -> None:
        self.envelopes_recibidos = 0
        self.eventos_procesados = 0
        # Contadores de eventos discretos (acumulados de toda la corrida)
        self.total_parpadeos = 0
        self.total_bostezos = 0
        self.total_cabeceos = 0
        self.total_microsuenos = 0
        # Ultimas metricas continuas vistas
        self.ultimo_pp_min = None
        self.ultimo_perclos = None
        self.ultimo_bostezos_vl = 0
        self.ultima_vision_disp = True
        # Para detectar flancos (imprimir el evento UNA vez)
        self.frames_no_confiables = 0

    def integrar(self, ep: EventoProcesado) -> list[str]:
        """
        Integra un EventoProcesado. Devuelve una lista de mensajes de
        evento discreto para imprimir (vacia si no hubo eventos nuevos).
        """
        self.eventos_procesados += 1
        mensajes = []

        if ep.parpadeo:
            self.total_parpadeos += 1
        if ep.bostezo:
            self.total_bostezos += 1
            mensajes.append(f"  >> BOSTEZO detectado "
                             f"(total: {self.total_bostezos}, "
                             f"en ventana 15min: {ep.bostezos_ventana_larga})")
        if ep.cabeceo:
            self.total_cabeceos += 1
            mensajes.append(f"  >> CABECEO detectado (total: {self.total_cabeceos})")
        if ep.microsueno:
            self.total_microsuenos += 1
            mensajes.append(f"  >> MICROSUENO detectado "
                            f"(total: {self.total_microsuenos})")

        # Metricas continuas
        self.ultimo_pp_min = ep.parpadeos_por_minuto
        self.ultimo_perclos = ep.perclos
        self.ultimo_bostezos_vl = ep.bostezos_ventana_larga

        # Cambio de disponibilidad de vision
        if ep.vision_disponible != self.ultima_vision_disp:
            estado = "DISPONIBLE" if ep.vision_disponible else "NO DISPONIBLE"
            mensajes.append(f"  >> Vision ahora: {estado}")
            self.ultima_vision_disp = ep.vision_disponible

        if ep.ventana_no_confiable:
            self.frames_no_confiables += 1

        return mensajes

    def resumen(self) -> str:
        pp = f"{self.ultimo_pp_min:.1f}" if self.ultimo_pp_min is not None else "-"
        pc = f"{self.ultimo_perclos*100:.0f}%" if self.ultimo_perclos is not None else "-"
        return (
            f"[RESUMEN] envelopes={self.envelopes_recibidos} "
            f"procesados={self.eventos_procesados} | "
            f"parpadeos/min={pp} PERCLOS={pc} | "
            f"totales: parpadeos={self.total_parpadeos} "
            f"bostezos={self.total_bostezos} cabeceos={self.total_cabeceos} "
            f"microsuenos={self.total_microsuenos}"
        )


# =============================================================================
# Programa principal
# =============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="NeuroDrive Core - Pre-FSM (Etapa III de integracion)"
    )
    parser.add_argument("--duracion-max", type=float, default=0.0,
                        help="Detener tras N segundos (0 = sin limite)")
    parser.add_argument("--periodo-resumen", type=float, default=5.0,
                        help="Cada cuantos segundos imprimir el resumen")
    args = parser.parse_args(argv)

    print("=" * 64)
    print("  NeuroDrive Core - Pre-FSM (Etapa III)")
    print("  Consumiendo eventos de NeuroDrive_Vision via POSIX MQ")
    print("=" * 64)

    # ---- Cargar configuracion ----
    try:
        config = cargar_config()
    except Exception as e:
        print(f"ERROR cargando config.yaml: {e}")
        return 1

    print(f"  Cola de vision:  {config.ipc.cola_vision}")
    print(f"  Capacidad cola:  {config.ipc.capacidad_cola}")
    print(f"  Umbral cabeceo:  {config.cabeza.umbral_pitch_grados} grados "
          f"(absoluto; la vision envia el pitch ya normalizado)")
    print()

    # ---- Crear Gestor y Pre-FSM ----
    gestor = GestorEventos(config)
    pre_fsm = PreFSM(config)
    acum = _Acumulador()

    # ---- Iniciar el Gestor (abre la cola, levanta hilos lectores) ----
    try:
        gestor.iniciar()
    except Exception as e:
        print(f"ERROR FATAL al iniciar el GestorEventos: {e}")
        print("Verificar que posix_ipc este instalado y que la cola sea "
              "creable (revisar limites del kernel: fs.mqueue.msg_max).")
        return 1

    print("GestorEventos iniciado. Esperando eventos de la vision...")
    print("(Arranca la vision en otra terminal: "
          "python -m NeuroDrive_Vision.test_vision --mq-real)")
    print("Ctrl+C para detener.\n")

    ts_inicio = time.monotonic()
    ts_ultimo_resumen = ts_inicio
    ts_ultimo_evento = None

    codigo_salida = 0
    try:
        # El gestor maneja Ctrl+C internamente (registra handlers de senal):
        # cuando llega SIGINT, gestor.activo pasa a False y salimos del loop.
        while gestor.activo:
            # obtener_evento bloquea hasta 1s esperando un Envelope
            envelope = gestor.obtener_evento(timeout=1.0)

            if envelope is not None:
                acum.envelopes_recibidos += 1
                ts_ultimo_evento = time.monotonic()

                # El Pre-FSM transforma el Envelope en EventoProcesado
                evento_proc = pre_fsm.procesar(envelope)

                if evento_proc is not None:
                    mensajes = acum.integrar(evento_proc)
                    # Imprimir eventos discretos en el momento
                    for msg in mensajes:
                        print(msg)

            # Resumen periodico
            ahora = time.monotonic()
            if ahora - ts_ultimo_resumen >= args.periodo_resumen:
                print(acum.resumen())
                ts_ultimo_resumen = ahora
                # Aviso si hace rato que no llegan eventos
                if ts_ultimo_evento is None:
                    print("  (aun no llego ningun evento: la vision no arranco?)")
                elif ahora - ts_ultimo_evento > 5.0:
                    print(f"  (sin eventos hace {ahora - ts_ultimo_evento:.0f}s: "
                          f"la vision se detuvo?)")

            # Limite de duracion opcional
            if args.duracion_max > 0 and (ahora - ts_inicio) >= args.duracion_max:
                print(f"\nDuracion maxima ({args.duracion_max}s) alcanzada.")
                break

    except KeyboardInterrupt:
        # Por si el handler del gestor no lo capturo
        print("\nInterrupcion por teclado.")
    except Exception as e:
        print(f"\nERROR en el loop principal: {e}")
        import traceback
        traceback.print_exc()
        codigo_salida = 1
    finally:
        print("\nDeteniendo el GestorEventos...")
        gestor.detener()

    # ---- Reporte final ----
    print("\n" + "=" * 64)
    print("  REPORTE FINAL - Etapa III")
    print("=" * 64)
    duracion = time.monotonic() - ts_inicio
    print(f"  Duracion:               {duracion:.1f} s")
    print(f"  Envelopes recibidos:    {acum.envelopes_recibidos}")
    print(f"  Eventos procesados:     {acum.eventos_procesados}")
    print(f"  Parpadeos detectados:   {acum.total_parpadeos}")
    print(f"  Bostezos detectados:    {acum.total_bostezos}")
    print(f"  Cabeceos detectados:    {acum.total_cabeceos}")
    print(f"  Microsuenos detectados: {acum.total_microsuenos}")
    print(f"  Frames no confiables:   {acum.frames_no_confiables} "
          f"(frote de ojos / rostro perdido)")
    if acum.ultimo_pp_min is not None:
        print(f"  Ultima frec. parpadeo:  {acum.ultimo_pp_min:.1f} /min")
    if acum.ultimo_perclos is not None:
        print(f"  Ultimo PERCLOS:         {acum.ultimo_perclos*100:.0f}%")

    # Estadisticas del gestor (cuantos mensajes ley, descartes, etc.)
    stats = gestor.stats
    print(f"\n  --- Estadisticas del GestorEventos ---")
    print(f"  Mensajes de vision:     {stats.mensajes_recibidos_vision}")
    print(f"  Mensajes invalidos:     {stats.mensajes_invalidos}")
    print(f"  Duplicados descartados: {stats.duplicados_descartados}")
    print(f"  Cola interna llena:     {stats.cola_interna_llena_descartes}")
    print(f"  Errores lectura MQ:     {stats.errores_lectura_mq}")
    print("=" * 64)

    # ---- Validacion de la etapa ----
    print("\n--- Validacion de integracion (Etapa III) ---")
    fallas = []
    if acum.envelopes_recibidos == 0:
        fallas.append("no se recibio ningun evento de la vision "
                       "(verificar que la vision corrio con --mq-real)")
    if stats.mensajes_invalidos > acum.envelopes_recibidos * 0.1:
        fallas.append(f"demasiados mensajes invalidos: {stats.mensajes_invalidos}")

    if fallas:
        print("RESULTADO: FALLO")
        for f in fallas:
            print(f"  [FAIL] {f}")
        return 1

    if acum.envelopes_recibidos > 0:
        print("RESULTADO: OK - el Pre-FSM proceso eventos reales de la vision")
    return codigo_salida


if __name__ == "__main__":
    sys.exit(main())
