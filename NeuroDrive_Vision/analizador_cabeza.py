"""
NeuroDrive Vision - Analizador de pose de cabeza
================================================

Calcula los angulos de Euler (pitch, yaw, roll) de la cabeza del conductor
usando el algoritmo Perspective-n-Point (PnP) de OpenCV sobre 6 landmarks
del rostro detectados por MediaPipe FaceMesh.

Decisiones tomadas (ver chat de planificacion para detalle):
  - 6 landmarks del modelo facial de Roth & Winkler (estandar literatura):
    nariz(1), menton(152), ojos exteriores(33, 263), boca exteriores(61, 291).
  - Matriz intrinseca de camara: aproximacion (focal=ancho, centro=medio).
    Suficiente para detectar cabeceos. Si se quiere precision absoluta,
    se pasa la matriz calibrada por parametro.
  - solvePnP con SOLVEPNP_SQPNP (mas estable). Fallback a ITERATIVE.
  - Extraccion de Euler con cv2.RQDecomp3x3 (evita errores de convencion).
  - Filtro EMA (exponential moving average) con alpha=0.5 para reducir
    jitter frame-a-frame.

Convencion de angulos (importante):
  - pitch: rotacion eje X -> cabeza arriba/abajo (cabeceo)
      > 0 = mirando hacia ABAJO (cabeceo de sueño), < 0 = mirando hacia arriba
      (convencion verificada empiricamente con el hardware: al bajar la
      cabeza el pitch crece hacia valores positivos)
  - yaw:   rotacion eje Y -> cabeza izquierda/derecha
      > 0 = mirando a la derecha del conductor, < 0 = a la izquierda
  - roll:  rotacion eje Z -> inclinacion lateral
      > 0 = oreja derecha hacia hombro derecho

ESTE MODULO NO DETECTA EVENTOS DE CABECEO. Solo extrae features (angulos).
La logica de "es cabeceo" vive en el Pre-FSM del Core (regla temporal).

API:
    analizador = AnalizadorCabeza(config)
    datos_cabeza = analizador.procesar(datos_rostro)
    if datos_cabeza.valido:
        print(f"pitch={datos_cabeza.pitch_deg:.1f}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from NeuroDrive_Vision.detector_rostro import DatosRostro

try:
    from NeuroDrive_Core.config_loader import Config
except ImportError:
    Config = None  # type: ignore


_log = logging.getLogger("NeuroDrive.AnalizadorCabeza")


# =============================================================================
# Modelo facial 3D de Roth & Winkler (unidades: milimetros)
# =============================================================================
#
# Estos son puntos de un rostro adulto promedio en el sistema de coordenadas
# del rostro (no de la camara). Cuando solvePnP recibe estos puntos junto a
# sus correspondientes 2D en pixel, calcula la rotacion y traslacion necesaria
# para que el modelo 3D se proyecte sobre los pixeles observados.
#
# El origen (0, 0, 0) es la punta de la nariz.
# +X = hacia la derecha del conductor (oreja derecha)
# +Y = hacia arriba (frente)
# +Z = hacia adelante (saliendo del rostro hacia el espectador)
#
# Indices de FaceMesh correspondientes a cada punto del modelo:
INDICES_PNP = np.array([
    1,    # punta de la nariz
    152,  # menton
    33,   # esquina exterior del ojo izquierdo (lado izquierdo del conductor)
    263,  # esquina exterior del ojo derecho
    61,   # esquina izquierda de la boca
    291,  # esquina derecha de la boca
], dtype=np.int32)

# Coordenadas 3D del modelo facial (mm), MISMO ORDEN que INDICES_PNP
MODELO_3D_MM = np.array([
    (   0.0,    0.0,    0.0),   # nariz
    (   0.0,  -63.6,  -12.5),   # menton
    ( -43.3,   32.7,  -26.0),   # ojo izquierdo
    (  43.3,   32.7,  -26.0),   # ojo derecho
    ( -28.9,  -28.9,  -24.1),   # boca izquierda
    (  28.9,  -28.9,  -24.1),   # boca derecha
], dtype=np.float64)


# =============================================================================
# Estructura de salida
# =============================================================================

@dataclass
class DatosCabeza:
    """
    Resultado del analisis de pose de cabeza para un frame.

    Si valido=False, los angulos son 0.0 y NO deben usarse.
    Si valido=True, todos los angulos estan en grados.
    """
    valido: bool

    # Angulos filtrados con EMA (recomendado para el Pre-FSM)
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0

    # Angulos crudos sin filtrar (para debug/calibracion)
    pitch_crudo: float = 0.0
    yaw_crudo: float = 0.0
    roll_crudo: float = 0.0

    # Vector de traslacion 3D (mm desde la camara al rostro)
    # Util para estimar distancia del conductor a la camara
    vector_traslacion: Optional[np.ndarray] = None  # shape (3,)

    # Diagnostico
    tiempo_procesamiento_ms: float = 0.0
    motivo_invalido: str = ""

    def __repr__(self) -> str:
        if not self.valido:
            return f"DatosCabeza(invalido: {self.motivo_invalido})"
        return (
            f"DatosCabeza(pitch={self.pitch_deg:+.1f}, "
            f"yaw={self.yaw_deg:+.1f}, roll={self.roll_deg:+.1f})"
        )


class ErrorAnalizadorCabeza(Exception):
    """Error tecnico en el analizador (NO incluye fallos de PnP)."""


# =============================================================================
# Clase principal
# =============================================================================

class AnalizadorCabeza:
    """
    Analizador de pose de cabeza usando solvePnP.

    Stateful: mantiene los angulos filtrados frame a frame para el EMA.
    No es thread-safe (una instancia por hilo).

    Lifecycle:
      - __init__: configura parametros y matrices.
      - procesar(datos_rostro): calcula angulos para este frame.
      - reset(): limpia el estado del filtro EMA.
    """

    # Si solvePnP devuelve un vector de traslacion con magnitud > este valor (mm),
    # algo salio mal (rostro a 100 metros de la camara). Marcamos como invalido.
    MAX_DISTANCIA_MM = 5000.0

    # Si solvePnP devuelve un vector con magnitud < este valor, tambien sospecha
    # (rostro pegado al lente). Aunque podria ser real, en practica indica error.
    MIN_DISTANCIA_MM = 50.0

    def __init__(
        self,
        config: Optional["Config"] = None,
        alpha_ema: float = 0.5,
        focal_length: Optional[float] = None,
        centro_optico: Optional[Tuple[float, float]] = None,
        matriz_camara: Optional[np.ndarray] = None,
        coefs_distorsion: Optional[np.ndarray] = None,
    ) -> None:
        """
        Parametros
        ----------
        config : Config | None
            Configuracion global (no usada aun, reservada).
        alpha_ema : float
            Coeficiente del filtro EMA en [0, 1].
            - 1.0 = sin filtrado (todo el valor nuevo)
            - 0.5 = mitad/mitad (default, balance jitter/latencia)
            - 0.1 = muy filtrado, mucho lag
        focal_length : float | None
            Distancia focal en pixeles. Si None, se calcula como ancho_frame
            cuando recibe el primer frame.
        centro_optico : (cx, cy) | None
            Centro optico en pixeles. Si None, se calcula como (ancho/2, alto/2).
        matriz_camara : np.ndarray | None
            Matriz intrinseca 3x3 completa. Si se pasa, sobrescribe
            focal_length y centro_optico (uso avanzado con camara calibrada).
        coefs_distorsion : np.ndarray | None
            Coeficientes de distorsion radial. Default = sin distorsion.
        """
        self.config = config

        if not (0.0 < alpha_ema <= 1.0):
            raise ValueError(f"alpha_ema debe estar en (0, 1], recibido {alpha_ema}")
        self.alpha_ema = float(alpha_ema)

        self._focal_length_init = focal_length
        self._centro_init = centro_optico
        self._matriz_camara_init = matriz_camara
        # Sin distorsion por default. Es lo correcto para una camara sin lente fish-eye.
        self.coefs_distorsion = (
            coefs_distorsion.astype(np.float64) if coefs_distorsion is not None
            else np.zeros(5, dtype=np.float64)
        )

        # La matriz de camara se construye en el primer procesamiento,
        # cuando ya conocemos las dimensiones reales del frame.
        self._matriz_camara: Optional[np.ndarray] = None

        # Estado del filtro EMA (None = aun no inicializado)
        self._pitch_filtrado: Optional[float] = None
        self._yaw_filtrado: Optional[float] = None
        self._roll_filtrado: Optional[float] = None

        # Que metodo usar para solvePnP. SQPNP es mas estable y nuevo.
        # Si no esta disponible, se hace fallback a ITERATIVE.
        # Se detecta la primera vez que se llama a solvePnP.
        self._metodo_pnp: Optional[int] = None

        # Indices y modelo 3D pre-cargados (no se recalculan)
        self._modelo_3d = MODELO_3D_MM.reshape(-1, 1, 3)  # shape (6, 1, 3)
        self._indices_pnp = INDICES_PNP

    # ------------------------------------------------------------------
    # Configuracion lazy de la matriz de camara
    # ------------------------------------------------------------------

    def _construir_matriz_camara(self, ancho: int, alto: int) -> np.ndarray:
        """
        Construye la matriz intrinseca de la camara.

        Si el usuario paso una matriz explicita en el constructor, la usa.
        Si no, usa la aproximacion estandar:
            focal_length = ancho_frame  (en pixeles)
            centro_optico = (ancho/2, alto/2)
        """
        if self._matriz_camara_init is not None:
            return self._matriz_camara_init.astype(np.float64)

        focal = self._focal_length_init if self._focal_length_init is not None else float(ancho)
        cx, cy = (
            self._centro_init if self._centro_init is not None
            else (ancho / 2.0, alto / 2.0)
        )

        matriz = np.array([
            [focal, 0.0,   cx],
            [0.0,   focal, cy],
            [0.0,   0.0,   1.0],
        ], dtype=np.float64)

        _log.info(
            "Matriz de camara construida: focal=%.1f px, centro=(%.1f, %.1f), frame=%dx%d",
            focal, cx, cy, ancho, alto,
        )
        return matriz

    # ------------------------------------------------------------------
    # solvePnP
    # ------------------------------------------------------------------

    def _detectar_metodo_pnp(self) -> int:
        """
        Detecta que metodo de solvePnP esta disponible.
        SQPNP es mejor pero solo en OpenCV >= 4.5. Sino, ITERATIVE.
        """
        try:
            metodo = cv2.SOLVEPNP_SQPNP
            _log.info("Usando SOLVEPNP_SQPNP")
            return int(metodo)
        except AttributeError:
            _log.info("SOLVEPNP_SQPNP no disponible, usando SOLVEPNP_ITERATIVE")
            return int(cv2.SOLVEPNP_ITERATIVE)

    # ------------------------------------------------------------------
    # Extraccion de Euler
    # ------------------------------------------------------------------

    @staticmethod
    def _matriz_rotacion_a_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
        """
        Convierte el vector de rotacion de Rodrigues a angulos de Euler
        en grados, usando la convencion estandar para pose de cabeza
        (rotacion intrinseca XYZ -> pitch, yaw, roll).

        Usamos cv2.RQDecomp3x3 que descompone la matriz de rotacion en
        sus angulos de Euler directamente, evitando errores de convencion
        comunes en implementaciones manuales.

        Returns:
            (pitch, yaw, roll) en grados.
        """
        # 1) Rodrigues: vector 3x1 -> matriz de rotacion 3x3
        rot_matrix, _ = cv2.Rodrigues(rvec)

        # 2) RQDecomp3x3: descompone R en sus angulos de Euler.
        # Firma OpenCV: retval, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(src)
        # El RETVAL (primer elemento) es la tupla (pitch_x, yaw_y, roll_z)
        # en grados. NO confundir con mtxQ (matriz ortogonal de la descomposicion).
        descomp = cv2.RQDecomp3x3(rot_matrix)
        euler_angles = descomp[0]  # tupla (3,) de grados

        # Para nuestra convencion: pitch=X, yaw=Y, roll=Z.
        # Aplanamos por las dudas, aunque deberia venir como tupla (3,).
        euler_flat = np.asarray(euler_angles).flatten()
        pitch_x = float(euler_flat[0])
        yaw_y = float(euler_flat[1])
        roll_z = float(euler_flat[2])

        # solvePnP devuelve la rotacion del sistema MUNDO -> CAMARA, que tiene
        # signo opuesto a la rotacion intuitiva del rostro respecto a la pose
        # frontal. Invertimos para obtener esta convencion (verificada
        # empiricamente con el hardware real):
        #   pitch > 0 = cabeza mirando hacia ABAJO (cabeceo de sueño)
        #   pitch < 0 = cabeza mirando hacia arriba
        #   yaw   > 0 = cabeza mirando hacia la DERECHA del conductor
        #   roll  > 0 = inclinacion oreja-DERECHA-hacia-hombro-DERECHO
        # Esta convencion es la que documentamos en el docstring del modulo
        # y la que va a consumir el Pre-FSM del Core: un cabeceo de somnolencia
        # se detecta como pitch creciendo hacia valores positivos.
        pitch_x = -pitch_x
        yaw_y = -yaw_y
        roll_z = -roll_z

        # Correccion estandar: RQDecomp3x3 a veces devuelve pitch con un offset
        # de 180 grados porque hay dos descomposiciones validas. Lo normalizamos
        # al rango [-90, 90] (rango fisico realista de cabeceo).
        if pitch_x > 90.0:
            pitch_x = 180.0 - pitch_x
        elif pitch_x < -90.0:
            pitch_x = -180.0 - pitch_x

        return pitch_x, yaw_y, roll_z

    # ------------------------------------------------------------------
    # Filtro EMA
    # ------------------------------------------------------------------

    def _aplicar_ema(self, pitch: float, yaw: float, roll: float) -> Tuple[float, float, float]:
        """
        Aplica filtro EMA a los 3 angulos.
        Si es el primer frame valido, inicializa con el valor crudo.
        """
        if self._pitch_filtrado is None:
            self._pitch_filtrado = pitch
            self._yaw_filtrado = yaw
            self._roll_filtrado = roll
            return pitch, yaw, roll

        a = self.alpha_ema
        self._pitch_filtrado = a * pitch + (1.0 - a) * self._pitch_filtrado
        self._yaw_filtrado = a * yaw + (1.0 - a) * self._yaw_filtrado
        self._roll_filtrado = a * roll + (1.0 - a) * self._roll_filtrado

        return self._pitch_filtrado, self._yaw_filtrado, self._roll_filtrado

    def reset(self) -> None:
        """Limpia el estado del filtro EMA. Usar al inicio de cada sesion."""
        self._pitch_filtrado = None
        self._yaw_filtrado = None
        self._roll_filtrado = None
        _log.info("AnalizadorCabeza: estado EMA reseteado")

    # ------------------------------------------------------------------
    # Procesamiento principal
    # ------------------------------------------------------------------

    def procesar(self, datos_rostro: DatosRostro) -> DatosCabeza:
        """
        Calcula los angulos de pose de cabeza para este frame.

        Parametros
        ----------
        datos_rostro : DatosRostro
            Salida del DetectorRostro (Etapa 4.2).

        Returns
        -------
        DatosCabeza
            Si datos_rostro.rostro_presente=False: valido=False.
            Si solvePnP falla: valido=False.
            Si todo OK: valido=True con todos los angulos.
        """
        t0 = time.monotonic()

        # 1) Si no hay rostro, no podemos calcular nada
        if not datos_rostro.rostro_presente:
            return DatosCabeza(
                valido=False,
                motivo_invalido="rostro no presente",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
            )

        if datos_rostro.puntos_pixeles is None:
            return DatosCabeza(
                valido=False,
                motivo_invalido="puntos_pixeles es None",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
            )

        # 2) Construir matriz de camara la primera vez
        if self._matriz_camara is None:
            ancho, alto = datos_rostro.resolucion
            if ancho <= 0 or alto <= 0:
                return DatosCabeza(
                    valido=False,
                    motivo_invalido=f"resolucion invalida: {datos_rostro.resolucion}",
                    tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
                )
            self._matriz_camara = self._construir_matriz_camara(ancho, alto)

        # 3) Detectar metodo PnP la primera vez
        if self._metodo_pnp is None:
            self._metodo_pnp = self._detectar_metodo_pnp()

        # 4) Extraer los 6 puntos 2D del rostro detectado
        try:
            puntos_2d = datos_rostro.puntos_pixeles[self._indices_pnp].astype(np.float64)
        except (IndexError, ValueError) as e:
            return DatosCabeza(
                valido=False,
                motivo_invalido=f"error al indexar landmarks: {e}",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
            )

        # solvePnP necesita shape (N, 1, 2) y dtype float64
        puntos_2d = puntos_2d.reshape(-1, 1, 2)

        # 5) Resolver PnP
        try:
            exito, rvec, tvec = cv2.solvePnP(
                self._modelo_3d,
                puntos_2d,
                self._matriz_camara,
                self.coefs_distorsion,
                flags=self._metodo_pnp,
            )
        except cv2.error as e:
            return DatosCabeza(
                valido=False,
                motivo_invalido=f"solvePnP lanzo excepcion: {e}",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
            )

        if not exito:
            return DatosCabeza(
                valido=False,
                motivo_invalido="solvePnP no convergio",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
            )

        # 6) Validar el vector de traslacion (sanity check)
        distancia = float(np.linalg.norm(tvec))
        if distancia > self.MAX_DISTANCIA_MM or distancia < self.MIN_DISTANCIA_MM:
            return DatosCabeza(
                valido=False,
                motivo_invalido=f"distancia improbable: {distancia:.0f} mm",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
            )

        # 7) Convertir rvec a angulos de Euler
        try:
            pitch_crudo, yaw_crudo, roll_crudo = self._matriz_rotacion_a_euler(rvec)
        except cv2.error as e:
            return DatosCabeza(
                valido=False,
                motivo_invalido=f"conversion a Euler fallo: {e}",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
            )

        # 8) Sanity check sobre los angulos: si vienen NaN o infinitos, descartar
        if not all(np.isfinite([pitch_crudo, yaw_crudo, roll_crudo])):
            return DatosCabeza(
                valido=False,
                motivo_invalido="angulos no finitos (NaN/inf)",
                tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
            )

        # 9) Aplicar filtro EMA
        pitch_f, yaw_f, roll_f = self._aplicar_ema(pitch_crudo, yaw_crudo, roll_crudo)

        return DatosCabeza(
            valido=True,
            pitch_deg=pitch_f,
            yaw_deg=yaw_f,
            roll_deg=roll_f,
            pitch_crudo=pitch_crudo,
            yaw_crudo=yaw_crudo,
            roll_crudo=roll_crudo,
            vector_traslacion=tvec.flatten().astype(np.float32),
            tiempo_procesamiento_ms=(time.monotonic() - t0) * 1000.0,
        )

    # ------------------------------------------------------------------
    # Visualizacion para debug (no usar en runtime)
    # ------------------------------------------------------------------

    @staticmethod
    def dibujar_ejes(
        frame_bgr: np.ndarray,
        datos_rostro: DatosRostro,
        datos_cabeza: DatosCabeza,
        analizador: "AnalizadorCabeza",
        longitud_mm: float = 60.0,
    ) -> np.ndarray:
        """
        Dibuja los 3 ejes (X rojo, Y verde, Z azul) saliendo de la punta de
        la nariz, segun la pose detectada. Util para visualizacion en vivo.

        Si datos_cabeza no es valido, devuelve el frame sin modificar.
        """
        if not datos_cabeza.valido or not datos_rostro.rostro_presente:
            return frame_bgr.copy()
        if analizador._matriz_camara is None or datos_rostro.puntos_pixeles is None:
            return frame_bgr.copy()

        # Recalcular rvec y tvec desde los angulos seria mas caro;
        # mejor lo hacemos pidiendo solvePnP de nuevo (ya tenemos todo).
        puntos_2d = datos_rostro.puntos_pixeles[INDICES_PNP].astype(np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            MODELO_3D_MM.reshape(-1, 1, 3),
            puntos_2d,
            analizador._matriz_camara,
            analizador.coefs_distorsion,
            flags=analizador._metodo_pnp or cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return frame_bgr.copy()

        # Definir los 3 ejes en coordenadas del rostro (mm)
        ejes_3d = np.array([
            [longitud_mm, 0, 0],   # +X (rojo)
            [0, longitud_mm, 0],   # +Y (verde)
            [0, 0, longitud_mm],   # +Z (azul, sale del rostro)
        ], dtype=np.float64).reshape(-1, 1, 3)

        # Proyectar al plano de imagen
        proyectados, _ = cv2.projectPoints(
            ejes_3d, rvec, tvec,
            analizador._matriz_camara, analizador.coefs_distorsion,
        )

        # Punto de origen: punta de la nariz (landmark 1)
        origen = tuple(int(v) for v in datos_rostro.puntos_pixeles[1])

        out = frame_bgr.copy()
        cv2.line(out, origen, tuple(int(v) for v in proyectados[0].flatten()), (0, 0, 255), 2)  # X rojo
        cv2.line(out, origen, tuple(int(v) for v in proyectados[1].flatten()), (0, 255, 0), 2)  # Y verde
        cv2.line(out, origen, tuple(int(v) for v in proyectados[2].flatten()), (255, 0, 0), 2)  # Z azul

        return out
