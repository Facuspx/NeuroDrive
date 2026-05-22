"""
NeuroDrive_Vision - Modulo de vision por computadora.

Modulos:
  - captura_video: captura de frames desde Pi Camera CSI (Etapa 4.1)
  - detector_rostro: wrapper de MediaPipe FaceMesh (Etapa 4.2)
  - analizador_cabeza: angulos de Euler con solvePnP (Etapa 4.3)
  - analizador_ojos: calculo de EAR, PERCLOS, parpadeos (Etapa 4.4)
  - analizador_boca: calculo de MAR, bostezos (Etapa 4.5)
  - detector_frote_ojos: MediaPipe Hands (Etapa 4.6)
  - calibrador: calibracion personalizada de 60 seg (Etapa 4.7)
  - publicador_mq: envio de Envelopes al Core via POSIX MQ (Etapa 4.8)
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
from NeuroDrive_Vision.calibrador import (
    Calibrador,
    ResultadoCalibracion,
    ErrorCalibrador,
)
from NeuroDrive_Vision.publicador_mq import (
    PublicadorMQ,
    ErrorPublicadorMQ,
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
    "Calibrador",
    "ResultadoCalibracion",
    "ErrorCalibrador",
    "PublicadorMQ",
    "ErrorPublicadorMQ",
]
