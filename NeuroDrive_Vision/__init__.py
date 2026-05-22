"""
NeuroDrive_Vision - Modulo de vision por computadora.

Modulos:
  - captura_video: captura de frames desde Pi Camera CSI (Etapa 4.1)
  - detector_rostro: wrapper de MediaPipe FaceMesh (Etapa 4.2)
  - analizador_cabeza: angulos de Euler con solvePnP (Etapa 4.3, pendiente)
  - analizador_ojos: calculo de EAR, PERCLOS, parpadeos (Etapa 4.4, pendiente)
  - analizador_boca: calculo de MAR, bostezos (Etapa 4.5, pendiente)
  - detector_frote_ojos: MediaPipe Hands (Etapa 4.6, pendiente)
  - calibrador: calibracion personalizada de 30 seg (Etapa 4.7, pendiente)
  - publicador_mq: envio de Envelopes al Core via POSIX MQ (Etapa 4.8, pendiente)
"""

from NeuroDrive_Vision.captura_video import CapturaVideo, ErrorCaptura
from NeuroDrive_Vision.detector_rostro import (
    DetectorRostro,
    DatosRostro,
    ErrorDetectorRostro,
)
from NeuroDrive_Vision.analizador_cabeza import (
    AnalizadorCabeza,
    DatosCabeza,
    ErrorAnalizadorCabeza,
)
from NeuroDrive_Vision.analizador_ojos import (
    AnalizadorOjos,
    DatosOjos,
    ErrorAnalizadorOjos,
    OJO_IZQ_INDICES,
    OJO_DER_INDICES,
)
from NeuroDrive_Vision.analizador_boca import (
    AnalizadorBoca,
    DatosBoca,
    ErrorAnalizadorBoca,
    BOCA_INDICES,
)
from NeuroDrive_Vision.detector_frote_ojos import (
    DetectorFroteOjos,
    DatosFroteOjos,
    ErrorDetectorFroteOjos,
    TIPS_DEDOS,
)

__all__ = [
    "CapturaVideo",
    "ErrorCaptura",
    "DetectorRostro",
    "DatosRostro",
    "ErrorDetectorRostro",
    "AnalizadorCabeza",
    "DatosCabeza",
    "ErrorAnalizadorCabeza",
    "AnalizadorOjos",
    "DatosOjos",
    "ErrorAnalizadorOjos",
    "OJO_IZQ_INDICES",
    "OJO_DER_INDICES",
    "AnalizadorBoca",
    "DatosBoca",
    "ErrorAnalizadorBoca",
    "BOCA_INDICES",
    "DetectorFroteOjos",
    "DatosFroteOjos",
    "ErrorDetectorFroteOjos",
    "TIPS_DEDOS",
]
