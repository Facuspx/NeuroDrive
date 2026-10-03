#!/usr/bin/env python3
"""
Diagnostico de la camara, sin NeuroDrive.

Captura 10 s con rpicam-vid a 1920x1080 en YUV420, muestra la imagen reducida
y la tasa de cuadros medida. Sirve para separar un problema de la camara o
del cable (no hay imagen, tasa baja) de un problema del programa de vision.

Requiere la camara conectada por CSI y un entorno grafico.

USO:
    python3 herramientas/diagnostico_camara.py
    (q para salir antes)
"""
import subprocess
import sys
import time

import cv2
import numpy as np

ANCHO, ALTO, FPS, DURACION_S = 1920, 1080, 15, 10

cmd = [
    "rpicam-vid", "-t", "0",
    "--width", str(ANCHO), "--height", str(ALTO),
    "--framerate", str(FPS),
    "--codec", "yuv420",
    "--nopreview",
    "-o", "-",
]

try:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
except FileNotFoundError:
    print("No se encontro rpicam-vid. Este diagnostico solo corre en la Raspberry Pi.")
    sys.exit(1)

bytes_por_cuadro = ANCHO * ALTO * 3 // 2      # YUV420: 1,5 bytes por pixel
alto_yuv = ALTO * 3 // 2

print(f"Capturando a {ANCHO}x{ALTO} YUV420 durante {DURACION_S} s...")
inicio = time.time()
n = 0
fps_medido = 0.0

try:
    while time.time() - inicio < DURACION_S:
        crudo = proc.stdout.read(bytes_por_cuadro)
        if len(crudo) != bytes_por_cuadro:
            break                              # rpicam-vid termino o no entrega cuadros
        yuv = np.frombuffer(crudo, dtype=np.uint8).reshape((alto_yuv, ANCHO))
        cuadro = cv2.resize(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420), (640, 480))
        n += 1
        fps_medido = n / (time.time() - inicio)
        cv2.putText(cuadro, f"{ANCHO}x{ALTO}->640x480 | FPS:{fps_medido:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Diagnostico de camara", cuadro)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    proc.terminate()
    proc.wait()
    cv2.destroyAllWindows()

if n == 0:
    print("FALLA: rpicam-vid no entrego ningun cuadro. Revisar el cable CSI y "
          "'rpicam-hello --list-cameras'.")
    sys.exit(1)
print(f"OK: {n} cuadros, {fps_medido:.1f} por segundo")
