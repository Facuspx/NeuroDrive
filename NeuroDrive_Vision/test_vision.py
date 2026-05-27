"""
NeuroDrive Vision - Integrador / Main
=====================================

Punto de entrada del subsistema de vision. Orquesta los 7 modulos:
captura -> deteccion de rostro -> analisis (cabeza, ojos, boca) ->
deteccion de frote -> publicacion al Core por POSIX MQ.

Es a la vez:
  - El test de integracion de NeuroDrive_Vision.
  - El main real del subsistema cuando NeuroDrive corre en produccion.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Vision.test_vision

Controles durante la ejecucion:
    q   -> salir
    m   -> prender/apagar la ventana de malla (la negra futurista)
    ESC -> (durante calibracion) saltear y usar valores por defecto

Flujo:
  1. Arranque: instancia e inicia los 7 modulos.
  2. Calibracion: carga calibracion.json si existe, o calibra 60s.
  3. Loop principal: procesa frames hasta que se presione 'q'.
  4. Cierre: detiene modulos e imprime reporte.

Ventanas:
  - "NeuroDrive Vision": video en vivo con todos los overlays + panel de metricas.
  - "NeuroDrive - Malla": fondo negro con los landmarks (cara cyan, mano magenta).
    Se prende/apaga con la tecla 'm'.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np

from NeuroDrive_Core.config_loader import cargar_config, limpiar_cache
from NeuroDrive_Vision.captura_video import CapturaVideo
from NeuroDrive_Vision.detector_rostro import DetectorRostro
from NeuroDrive_Vision.analizador_cabeza import AnalizadorCabeza
from NeuroDrive_Vision.analizador_ojos import AnalizadorOjos
from NeuroDrive_Vision.analizador_boca import AnalizadorBoca
from NeuroDrive_Vision.detector_frote_ojos import DetectorFroteOjos
from NeuroDrive_Vision.calibrador import Calibrador, ResultadoCalibracion
from NeuroDrive_Vision.publicador_mq import PublicadorMQ


# Configuracion de logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger("NeuroDrive.TestVision")


# =============================================================================
# Constantes de visualizacion
# =============================================================================

RUTA_CALIBRACION = "calibracion.json"

# Colores BGR
COLOR_VERDE = (0, 255, 0)
COLOR_ROJO = (0, 0, 255)
COLOR_AMARILLO = (0, 255, 255)
COLOR_BLANCO = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)        # cara en la ventana de malla
COLOR_MAGENTA = (255, 0, 255)    # mano en la ventana de malla
COLOR_AZUL = (255, 0, 0)

# Conexiones del esqueleto de la mano (pares de indices de los 21 landmarks
# de MediaPipe Hands). Define que puntos se unen con lineas.
CONEXIONES_MANO = [
    # Pulgar
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Indice
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Medio
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Anular
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Meñique
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palma (base)
    (0, 17),
]


# =============================================================================
# Detector de cabeceo - SOLO VISUAL / DE PRUEBA
# =============================================================================
#
# IMPORTANTE: este detector NO es parte del modulo de vision de produccion.
# La deteccion oficial de cabeceo es responsabilidad de la FSM del Core,
# que recibe el pitch en grados desde la vision y aplica sus propias reglas.
#
# Esta clase existe SOLO para que en el integrador (test_vision) se pueda
# VER en pantalla cuando el pitch indicaria un cabeceo, util para validar
# visualmente y para las capturas del informe. Es una estimacion local,
# no la deteccion definitiva del sistema.

class DetectorCabeceoVisual:
    """
    Estimador visual de cabeceo para el integrador.

    Considera "cabeceo" cuando el pitch SUBE por encima de un umbral de
    forma sostenida.

    CONVENCION DE PITCH (verificada empiricamente con el hardware):
        - Cabeza mirando ABAJO  -> pitch POSITIVO (sube)  <- cabeceo de sueño
        - Cabeza mirando ARRIBA -> pitch NEGATIVO (baja)
    Por eso el cabeceo se detecta cuando el pitch SUPERA el umbral, no
    cuando cae por debajo.

    El umbral se mide RELATIVO al pitch neutro del conductor (obtenido
    de la calibracion), no en valor absoluto: si tu pose neutra es -10
    grados, un cabeceo es bajar la cabeza otros 15 grados -> pitch > +5.
    """

    # Cuantos grados por encima del neutro se considera cabeceo
    UMBRAL_RELATIVO_GRADOS = 15.0
    # Cuanto tiempo sostenido para contar el evento (segundos)
    DURACION_MIN_S = 1.2

    def __init__(self, pitch_neutro: float = 0.0) -> None:
        self.pitch_neutro = pitch_neutro
        # Cabeceo = cabeza abajo = pitch POR ENCIMA del neutro
        self.umbral_pitch = pitch_neutro + self.UMBRAL_RELATIVO_GRADOS
        self._ts_inicio_inclinacion = None
        self._evento_ya_contado = False
        self.cabeceos_contados = 0
        self.cabeceo_en_curso = False

    def actualizar(self, datos_cabeza, ts: float) -> None:
        """Procesa la pose de cabeza de un frame."""
        self.cabeceo_en_curso = False

        if datos_cabeza is None or not datos_cabeza.valido:
            # Sin pose valida: reseteamos el conteo en curso
            self._ts_inicio_inclinacion = None
            self._evento_ya_contado = False
            return

        # Cabeceo = cabeza inclinada hacia abajo = pitch por encima del umbral
        inclinado = datos_cabeza.pitch_deg > self.umbral_pitch

        if inclinado:
            if self._ts_inicio_inclinacion is None:
                self._ts_inicio_inclinacion = ts
                self._evento_ya_contado = False
            duracion = ts - self._ts_inicio_inclinacion
            if duracion >= self.DURACION_MIN_S:
                self.cabeceo_en_curso = True
                if not self._evento_ya_contado:
                    self.cabeceos_contados += 1
                    self._evento_ya_contado = True
        else:
            self._ts_inicio_inclinacion = None
            self._evento_ya_contado = False


# =============================================================================
# Ventana de malla futurista
# =============================================================================

def dibujar_ventana_malla(
    ancho: int,
    alto: int,
    datos_rostro,
    datos_frote,
) -> np.ndarray:
    """
    Construye la ventana negra futurista:
      - Fondo negro puro.
      - Rostro: 468 puntos cyan.
      - Manos: 21 puntos magenta + esqueleto (lineas entre articulaciones).
    """
    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)

    # ----- Rostro: solo puntos cyan -----
    if datos_rostro.rostro_presente and datos_rostro.puntos_pixeles is not None:
        for (x, y) in datos_rostro.puntos_pixeles:
            cv2.circle(lienzo, (int(x), int(y)), 1, COLOR_CYAN, -1)

    # ----- Manos: puntos + esqueleto magenta -----
    if datos_frote is not None and datos_frote.landmarks_manos:
        for mano in datos_frote.landmarks_manos:
            if len(mano) < 21:
                continue
            # Lineas del esqueleto (articulaciones)
            for (i, j) in CONEXIONES_MANO:
                p1 = (int(mano[i][0]), int(mano[i][1]))
                p2 = (int(mano[j][0]), int(mano[j][1]))
                cv2.line(lienzo, p1, p2, COLOR_MAGENTA, 1)
            # Puntos de las articulaciones
            for (x, y) in mano:
                cv2.circle(lienzo, (int(x), int(y)), 3, COLOR_MAGENTA, -1)

    # Titulo discreto
    cv2.putText(lienzo, "NeuroDrive - Malla", (10, alto - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    return lienzo


# =============================================================================
# Overlay de la ventana principal
# =============================================================================

def dibujar_panel_metricas(
    frame,
    fps: float,
    datos_rostro,
    datos_cabeza,
    datos_ojos,
    datos_boca,
    datos_frote,
    stats_mq: dict,
    detector_cabeceo=None,
) -> None:
    """
    Dibuja el panel de texto con todas las metricas en la esquina del frame.
    Modifica el frame in-place.
    """
    x = 10
    y = 25
    dy = 22

    def linea(texto, color=COLOR_BLANCO, escala=0.5):
        nonlocal y
        cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, color, 1, cv2.LINE_AA)
        y += dy

    # FPS y estado de rostro
    color_fps = COLOR_VERDE if fps >= 10 else COLOR_AMARILLO if fps >= 7 else COLOR_ROJO
    linea(f"FPS: {fps:.1f}", color_fps, 0.6)

    if not datos_rostro.rostro_presente:
        linea("SIN ROSTRO", COLOR_ROJO, 0.6)
        return

    # Ojos
    if datos_ojos is not None and datos_ojos.valido:
        estado_ojos = "CERRADOS" if datos_ojos.ojos_cerrados else "abiertos"
        col = COLOR_ROJO if datos_ojos.ojos_cerrados else COLOR_VERDE
        linea(f"EAR: {datos_ojos.ear_promedio:.3f}  ({estado_ojos})", col)
        linea(f"PERCLOS: {datos_ojos.perclos:.2f}   bpm: {datos_ojos.parpadeos_por_minuto:.1f}")
        if datos_ojos.evento_parpadeo:
            linea(f"  evento: {datos_ojos.evento_parpadeo}", COLOR_AMARILLO)

    # Boca
    if datos_boca is not None and datos_boca.valido:
        estado_boca = "ABIERTA" if datos_boca.boca_abierta else "cerrada"
        col = COLOR_AMARILLO if datos_boca.boca_abierta else COLOR_VERDE
        linea(f"MAR: {datos_boca.mar:.3f}  ({estado_boca})", col)
        if datos_boca.evento_bostezo:
            linea("  BOSTEZO detectado", COLOR_ROJO)

    # Cabeza
    if datos_cabeza is not None and datos_cabeza.valido:
        linea(f"pitch:{datos_cabeza.pitch_deg:+.0f} "
              f"yaw:{datos_cabeza.yaw_deg:+.0f} "
              f"roll:{datos_cabeza.roll_deg:+.0f}")

    # Cabeceo (estimacion visual de prueba; la deteccion real la hace el Core)
    if detector_cabeceo is not None:
        col = COLOR_ROJO if detector_cabeceo.cabeceo_en_curso else COLOR_VERDE
        estado_cab = "CABECEO" if detector_cabeceo.cabeceo_en_curso else "ok"
        linea(f"cabeceo: {estado_cab}  (contados: {detector_cabeceo.cabeceos_contados})", col)

    # Frote
    if datos_frote is not None and datos_frote.valido:
        if datos_frote.frote_en_curso:
            linea("FROTE DE OJOS", COLOR_ROJO, 0.6)
        linea(f"manos: {datos_frote.manos_detectadas}")

    # MQ
    linea(f"MQ enviados: {stats_mq.get('mensajes_enviados', 0)}  "
          f"descartados: {stats_mq.get('mensajes_descartados', 0)}",
          COLOR_BLANCO, 0.45)


# =============================================================================
# Fase de calibracion
# =============================================================================

def fase_calibracion(
    cap: CapturaVideo,
    detector_rostro: DetectorRostro,
    duracion_seg: float,
) -> ResultadoCalibracion:
    """
    Ejecuta la fase de calibracion mostrando una barra de progreso.
    Se puede saltear con ESC.

    Returns:
        ResultadoCalibracion (puede tener exito=False si se salteo o fallo).
    """
    calibrador = Calibrador(duracion_seg=duracion_seg)
    calibrador.iniciar()

    print(f"\n=== CALIBRACION ({duracion_seg:.0f}s) ===")
    print("Mira al frente con los ojos abiertos y la cabeza en posicion normal.")
    print("Presiona ESC para saltear y usar valores por defecto.\n")

    salteado = False

    while not calibrador.terminado:
        frame, _ = cap.leer()
        if frame is None:
            continue

        datos_rostro = detector_rostro.procesar(frame)
        calibrador.procesar(datos_rostro)

        # Visualizacion de la calibracion
        viz = frame.copy()
        progreso = calibrador.progreso
        alto, ancho = viz.shape[:2]

        # Barra de progreso
        margen = 40
        barra_y = alto - 60
        barra_w = ancho - 2 * margen
        cv2.rectangle(viz, (margen, barra_y), (margen + barra_w, barra_y + 25),
                      (60, 60, 60), -1)
        cv2.rectangle(viz, (margen, barra_y),
                      (margen + int(barra_w * progreso), barra_y + 25),
                      COLOR_VERDE, -1)
        cv2.rectangle(viz, (margen, barra_y), (margen + barra_w, barra_y + 25),
                      COLOR_BLANCO, 1)

        # Textos
        cv2.putText(viz, "CALIBRANDO", (margen, barra_y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_AMARILLO, 2)
        cv2.putText(viz, f"{calibrador.tiempo_restante_seg:.0f}s restantes",
                    (margen, barra_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLOR_BLANCO, 1)

        estado_rostro = "rostro OK" if datos_rostro.rostro_presente else "SIN ROSTRO"
        col_r = COLOR_VERDE if datos_rostro.rostro_presente else COLOR_ROJO
        cv2.putText(viz, estado_rostro, (margen, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_r, 2)
        cv2.putText(viz, "ESC para saltear", (margen, alto - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLANCO, 1)

        cv2.imshow("NeuroDrive Vision", viz)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            salteado = True
            break

    if salteado:
        print("Calibracion salteada por el usuario.")
        # Devolvemos un resultado fallido -> el sistema usa defaults
        return ResultadoCalibracion(exito=False, motivo_fallo="salteada por usuario")

    resultado = calibrador.finalizar()
    if resultado.exito:
        print(f"Calibracion exitosa: ear_base={resultado.ear_base:.3f}, "
              f"pitch_neutro={resultado.pitch_neutro:+.1f}")
        try:
            resultado.guardar(RUTA_CALIBRACION)
            print(f"Calibracion guardada en {RUTA_CALIBRACION}")
        except OSError as e:
            print(f"No se pudo guardar la calibracion: {e}")
    else:
        print(f"Calibracion fallida: {resultado.motivo_fallo}")
        print("Se usaran valores por defecto.")

    return resultado


# =============================================================================
# Main
# =============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NeuroDrive Vision - Integrador")
    parser.add_argument("--duracion-max", type=float, default=0.0,
                        help="Duracion maxima del loop en segundos (0 = sin limite)")
    parser.add_argument("--duracion-calibracion", type=float, default=60.0,
                        help="Duracion de la calibracion en segundos")
    parser.add_argument("--recalibrar", action="store_true",
                        help="Forzar recalibracion ignorando calibracion.json")
    parser.add_argument("--mq-real", action="store_true",
                        help="Usar la cola POSIX MQ real. Por defecto el "
                             "publicador corre en modo simulado, porque "
                             "mientras el NeuroDrive Core no exista nadie "
                             "consume la cola. Activar solo cuando se integre "
                             "el Core.")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  NeuroDrive Vision - Integrador")
    print("=" * 60)

    limpiar_cache()
    config = cargar_config()

    # -------------------------------------------------------------
    # 1) Instanciar modulos
    # -------------------------------------------------------------
    cap = CapturaVideo(config)
    detector_rostro = DetectorRostro()
    analizador_cabeza = AnalizadorCabeza()
    analizador_ojos = AnalizadorOjos()
    analizador_boca = AnalizadorBoca()
    detector_frote = DetectorFroteOjos()
    # El publicador corre en modo simulado salvo que se pida --mq-real.
    # Mientras el Core no exista, nadie consume la cola: el modo simulado
    # ejercita todo el publicador (construye EventoVision, Envelope, valida
    # tamanios) sin crear una cola real que se llenaria sin consumidor.
    #
    # Con --mq-real (integracion con el Core), el publicador NO debe tocar
    # el ciclo de vida de la cola: el Core es su duenio. Por eso, en modo
    # real, drenar_al_iniciar y eliminar_al_detener van en False.
    # En modo simulado esos flags son irrelevantes (no hay cola real).
    modo_integrado = args.mq_real
    publicador = PublicadorMQ(
        config,
        forzar_simulado=not args.mq_real,
        drenar_al_iniciar=not modo_integrado,
        eliminar_al_detener=not modo_integrado,
    )

    # Contadores del reporte final
    frames_totales = 0
    frames_con_rostro = 0
    frames_con_error = 0
    eventos_parpadeo = 0
    eventos_bostezo = 0
    eventos_frote = 0
    tiempos_frame = []

    mostrar_malla = True  # la ventana negra arranca prendida

    # -------------------------------------------------------------
    # 2) Iniciar modulos
    # -------------------------------------------------------------
    try:
        print("\nIniciando modulos...")
        cap.iniciar()
        detector_rostro.iniciar()
        detector_frote.iniciar()
        publicador.iniciar()
        print("Modulos iniciados correctamente.")
        if publicador.modo_simulado:
            print("MQ: modo simulado (el Core aun no consume la cola). "
                  "Usar --mq-real para la cola POSIX real.")
        else:
            print("MQ: modo real INTEGRADO con el Core.")
            print("  - La vision no drena ni elimina la cola (el Core es duenio).")
            print("  - Si la cola no existe aun, se crea con la capacidad del")
            print("    config.yaml; el Core se conecta a la misma cola.")
    except Exception as e:
        print(f"ERROR FATAL al iniciar modulos: {e}")
        _cerrar_todo(cap, detector_rostro, detector_frote, publicador)
        return 1

    # -------------------------------------------------------------
    # 3) Calibracion
    # -------------------------------------------------------------
    resultado_calib = None
    if not args.recalibrar:
        resultado_calib = ResultadoCalibracion.cargar(RUTA_CALIBRACION)
        if resultado_calib is not None and resultado_calib.exito:
            antiguedad = resultado_calib.antiguedad_dias()
            print(f"\nCalibracion previa encontrada (antiguedad: {antiguedad:.1f} dias).")
            print(f"  ear_base={resultado_calib.ear_base:.3f}")

    if resultado_calib is None or not resultado_calib.exito:
        try:
            resultado_calib = fase_calibracion(
                cap, detector_rostro, args.duracion_calibracion,
            )
        except Exception as e:
            print(f"Error durante la calibracion: {e}. Se usaran defaults.")
            resultado_calib = ResultadoCalibracion(exito=False, motivo_fallo=str(e))

    # Aplicar calibracion a los analizadores
    Calibrador.aplicar(resultado_calib, analizador_ojos, analizador_boca)

    # Pasar el pitch_neutro de la calibracion al publicador, para que
    # normalice el pitch antes de enviarlo al Core (Opcion A de integracion).
    # Si la calibracion fallo, queda en 0.0 (sin normalizacion).
    pitch_neutro_calib = (
        resultado_calib.pitch_neutro if resultado_calib.exito else 0.0
    )
    publicador.setear_pitch_neutro(pitch_neutro_calib)
    print(f"Publicador MQ: pitch_neutro = {pitch_neutro_calib:+.1f} grados "
          f"(el pitch se enviara normalizado al Core)")

    # Detector de cabeceo SOLO VISUAL (no es deteccion oficial, eso es del
    # Core). Usa el pitch_neutro de la calibracion como referencia: si la
    # calibracion fallo, pitch_neutro queda en 0.0 y el umbral es absoluto.
    detector_cabeceo = DetectorCabeceoVisual(pitch_neutro=pitch_neutro_calib)

    # -------------------------------------------------------------
    # 4) Loop principal
    # -------------------------------------------------------------
    print("\n=== OPERACION NORMAL ===")
    print("Controles: 'q' salir | 'm' prender/apagar ventana de malla\n")

    ts_inicio = time.monotonic()
    ts_ultimo_fps = ts_inicio
    contador_fps = 0
    fps_actual = 0.0

    try:
        while True:
            t_frame_0 = time.monotonic()

            # --- Captura ---
            try:
                frame, _ = cap.leer()
            except Exception as e:
                _log.error("Error de captura: %s", e)
                frames_con_error += 1
                if frames_con_error > 30:
                    print("Demasiados errores de captura. Abortando.")
                    break
                continue

            if frame is None:
                continue

            frames_totales += 1

            # --- Procesamiento (cada modulo protegido) ---
            datos_rostro = None
            datos_cabeza = None
            datos_ojos = None
            datos_boca = None
            datos_frote = None

            try:
                datos_rostro = detector_rostro.procesar(frame)
                if datos_rostro.rostro_presente:
                    frames_con_rostro += 1
                    datos_cabeza = analizador_cabeza.procesar(datos_rostro)
                    datos_ojos = analizador_ojos.procesar(datos_rostro)
                    datos_boca = analizador_boca.procesar(datos_rostro)

                # El frote se procesa siempre (necesita el frame)
                datos_frote = detector_frote.procesar(frame, datos_rostro)

                # Actualizar el estimador visual de cabeceo (solo para la
                # ventana; la deteccion oficial es del Core).
                detector_cabeceo.actualizar(datos_cabeza, time.monotonic())

                # Contar eventos
                if datos_ojos is not None and datos_ojos.evento_parpadeo:
                    eventos_parpadeo += 1
                if datos_boca is not None and datos_boca.evento_bostezo:
                    eventos_bostezo += 1
                if datos_frote is not None and datos_frote.evento_frote_iniciado:
                    eventos_frote += 1

                # --- Publicacion al Core ---
                publicador.publicar(
                    datos_rostro, datos_ojos, datos_boca,
                    datos_cabeza, datos_frote,
                )
            except Exception as e:
                _log.error("Error procesando frame %d: %s", frames_totales, e)
                frames_con_error += 1
                # No cortamos el loop: seguimos con el proximo frame

            # --- FPS ---
            contador_fps += 1
            ahora = time.monotonic()
            if ahora - ts_ultimo_fps >= 1.0:
                fps_actual = contador_fps / (ahora - ts_ultimo_fps)
                contador_fps = 0
                ts_ultimo_fps = ahora

            # --- Visualizacion ventana principal ---
            # Una sola copia del frame; todos dibujan sobre el mismo lienzo.
            viz = frame.copy()

            if datos_rostro is not None and datos_rostro.rostro_presente:
                # Landmarks del rostro
                if datos_rostro.puntos_pixeles is not None:
                    for (px, py) in datos_rostro.puntos_pixeles:
                        cv2.circle(viz, (int(px), int(py)), 1, COLOR_VERDE, -1)
                # Ejes 3D de cabeza
                if datos_cabeza is not None and datos_cabeza.valido:
                    viz = AnalizadorCabeza.dibujar_ejes(
                        viz, datos_rostro, datos_cabeza, analizador_cabeza,
                    )
                # Contornos de ojos
                if datos_ojos is not None:
                    viz = AnalizadorOjos.dibujar_ojos(viz, datos_rostro, datos_ojos)
                # Contorno de boca
                if datos_boca is not None:
                    viz = AnalizadorBoca.dibujar_boca(viz, datos_rostro, datos_boca)

            # Overlay de frote (regiones + puntas)
            if datos_frote is not None:
                viz = DetectorFroteOjos.dibujar(viz, datos_frote)

            # Panel de metricas
            if datos_rostro is not None:
                dibujar_panel_metricas(
                    viz, fps_actual, datos_rostro, datos_cabeza,
                    datos_ojos, datos_boca, datos_frote,
                    publicador.obtener_estadisticas(),
                    detector_cabeceo=detector_cabeceo,
                )

            cv2.imshow("NeuroDrive Vision", viz)

            # --- Visualizacion ventana de malla ---
            if mostrar_malla and datos_rostro is not None:
                alto, ancho = frame.shape[:2]
                malla = dibujar_ventana_malla(
                    ancho, alto, datos_rostro, datos_frote,
                )
                cv2.imshow("NeuroDrive - Malla", malla)

            # --- Tiempo de frame ---
            tiempos_frame.append((time.monotonic() - t_frame_0) * 1000.0)

            # --- Teclas ---
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                print("\nSalida solicitada por el usuario.")
                break
            elif tecla == ord("m"):
                mostrar_malla = not mostrar_malla
                if not mostrar_malla:
                    cv2.destroyWindow("NeuroDrive - Malla")
                print(f"Ventana de malla: {'ON' if mostrar_malla else 'OFF'}")

            # --- Limite de duracion (modo test) ---
            if args.duracion_max > 0 and (ahora - ts_inicio) >= args.duracion_max:
                print(f"\nLimite de duracion ({args.duracion_max}s) alcanzado.")
                break

    except KeyboardInterrupt:
        print("\nInterrumpido con Ctrl+C.")
    finally:
        cv2.destroyAllWindows()
        _cerrar_todo(cap, detector_rostro, detector_frote, publicador)

    # -------------------------------------------------------------
    # 5) Reporte final
    # -------------------------------------------------------------
    duracion_total = time.monotonic() - ts_inicio
    fps_promedio = frames_totales / duracion_total if duracion_total > 0 else 0.0
    tasa_rostro = (frames_con_rostro / frames_totales * 100) if frames_totales else 0.0
    t_frame_prom = sum(tiempos_frame) / len(tiempos_frame) if tiempos_frame else 0.0
    stats_mq = publicador.obtener_estadisticas()

    print("\n" + "=" * 60)
    print("  REPORTE FINAL")
    print("=" * 60)
    print(f"  Duracion:               {duracion_total:.1f} s")
    print(f"  Frames totales:         {frames_totales}")
    print(f"  Frames con rostro:      {frames_con_rostro} ({tasa_rostro:.1f}%)")
    print(f"  Frames con error:       {frames_con_error}")
    print(f"  FPS promedio:           {fps_promedio:.1f}")
    print(f"  Tiempo medio por frame: {t_frame_prom:.1f} ms")
    print(f"  Eventos de parpadeo:    {eventos_parpadeo}")
    print(f"  Eventos de bostezo:     {eventos_bostezo}")
    print(f"  Eventos de frote:       {eventos_frote}")
    print(f"  Cabeceos (estim. visual): {detector_cabeceo.cabeceos_contados}")
    print(f"  MQ enviados:            {stats_mq['mensajes_enviados']}")
    print(f"  MQ descartados:         {stats_mq['mensajes_descartados']}")
    print(f"  MQ modo simulado:       {stats_mq['modo_simulado']}")
    if stats_mq.get("mensajes_drenados", 0) > 0:
        print(f"  MQ cola vieja drenada:  {stats_mq['mensajes_drenados']} mensajes")
    print("=" * 60)

    # Nota sobre los descartes de MQ
    descartados = stats_mq["mensajes_descartados"]
    enviados = stats_mq["mensajes_enviados"]
    if (not stats_mq["modo_simulado"]) and descartados > enviados:
        print()
        print("  NOTA: se descartaron muchos mensajes MQ. Esto es ESPERADO")
        print("  mientras el NeuroDrive Core no este corriendo: nadie consume")
        print("  la cola, se llena, y la vision descarta (sin congelarse).")
        print("  Cuando el Core este integrado, el consumira los mensajes.")

    # -------------------------------------------------------------
    # Asserts globales del test de integracion
    # -------------------------------------------------------------
    print("\n--- Validacion de integracion ---")
    fallas = []

    if frames_totales < 30:
        fallas.append(f"muy pocos frames procesados: {frames_totales}")
    if fps_promedio < 10.0:
        fallas.append(f"FPS promedio bajo: {fps_promedio:.1f} (minimo 10)")
    if frames_con_error > frames_totales * 0.1:
        fallas.append(f"demasiados frames con error: {frames_con_error}")
    if tasa_rostro < 30.0:
        fallas.append(f"tasa de deteccion de rostro muy baja: {tasa_rostro:.1f}%")

    # Validacion del publicador: lo que importa es que pudo CONSTRUIR y
    # ENVIAR mensajes (los primeros entran antes de que la cola se llene).
    # Que despues descarte por cola llena NO es un fallo: es el
    # comportamiento correcto cuando no hay consumidor (Core).
    # Solo es fallo si no logro enviar NI UNO y tampoco descarto ninguno
    # (eso indicaria que el publicar() ni siquiera se ejecuto).
    total_mq = enviados + descartados
    if total_mq == 0:
        fallas.append("el publicador no proceso ningun mensaje (publicar no se ejecuto)")
    elif enviados == 0 and not stats_mq["modo_simulado"]:
        # Proceso mensajes pero no entro ninguno: la cola arranco llena.
        # Con el drenado al iniciar esto ya no deberia pasar; si pasa,
        # es un warning, no un fallo de integracion.
        print("  [AVISO] el publicador no logro encolar mensajes: "
              "la cola pudo haber arrancado llena. Revisar.")

    if fallas:
        print("RESULTADO: FALLO")
        for f in fallas:
            print(f"  [FAIL] {f}")
        return 1

    print("RESULTADO: OK - integracion validada")
    return 0


def _cerrar_todo(cap, detector_rostro, detector_frote, publicador) -> None:
    """Detiene todos los modulos en orden inverso, sin propagar errores."""
    for nombre, modulo in (
        ("publicador", publicador),
        ("detector_frote", detector_frote),
        ("detector_rostro", detector_rostro),
        ("captura", cap),
    ):
        try:
            modulo.detener()
        except Exception as e:
            _log.warning("Error al detener %s: %s", nombre, e)


if __name__ == "__main__":
    sys.exit(main())
