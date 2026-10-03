"""
NeuroDrive_Wearable - Protocolo en el cable (JSON sobre UDP)
============================================================

Fuente de verdad UNICA del formato de mensajes entre la Pi y el ESP32.
Lo usan el actuador (Pi->ESP), el receptor (ESP->Pi) y el simulador.
El firmware implementa el mismo formato del otro lado (red.c).

------------------------------------------------------------------
ORDENES  Pi -> ESP32  (la pulsera escucha en el puerto 5006)
------------------------------------------------------------------
    {
      "v": 1,
      "tipo": <int>,                # ver tabla de tipos
      "intensidad": <0-100>,
      "duracion_ms": <int>,         # significado segun el tipo (ver tabla)
      "id_secuencia": <int|null>,   # solo en el tipo 8
      "id_paquete": <int>           # identificador de envio (dedup en el ESP)
    }

    tipo  nombre           duracion_ms
    ----  ---------------  ---------------------------------------------
     1    VIBRAR_LEVE      duracion de la vibracion (0 = hasta nueva orden)
     2    VIBRAR_MEDIO     idem
     3    VIBRAR_FUERTE    idem
     8    SECUENCIA_ACK    VENTANA DE RESPUESTA: cuanto tiempo acepta la
                           pulsera el toque del conductor, contado desde el
                           ultimo pulso. 0 = usar el valor por defecto
                           (VENTANA_DESAFIO_DEFAULT_MS).
    10    APAGAR_TODO      sin uso. Detiene el motor y cancela el desafio.
    99    APAGADO_LIMPIO   sin uso. La unidad de procesamiento se detiene de
                           manera ordenada: la proxima perdida del enlace es
                           esperada y la pulsera no emite el aviso remoto.

Identificador de envio: las ordenes criticas viajan repetidas con el MISMO
id_paquete y la pulsera ejecuta una sola (recuerda los ultimos 16). El
contador no arranca en cero sino en una base aleatoria por sesion: si
arrancara siempre igual, tras reiniciar el nucleo las primeras ordenes
coincidirian con identificadores que la pulsera todavia recuerda de la sesion
anterior y se descartarian como duplicadas.

------------------------------------------------------------------
TELEMETRIA / RESPUESTA  ESP32 -> Pi  (la Pi escucha en el puerto 5005)
------------------------------------------------------------------
Telemetria periodica, cada ~2 s. Su sola llegada prueba que la pulsera opera:
    {
      "v": 1, "msg": "telemetria",
      "bpm": <int|null>,               # null si el MAX30102 no tiene lectura confiable
      "ack_recibido": <bool>,
      "secuencia_replicada": <bool>,
      "bateria": <int|null>,           # 0-100
      "id_paquete": <int>
    }

Respuesta del conductor a una verificacion. Viaja repetida (ACK_REENVIOS
copias con el mismo id_secuencia); el receptor de la Pi publica una sola:
    {
      "v": 1, "msg": "ack",
      "id_secuencia": <int>,           # el mismo id que vino en el tipo 8
      "secuencia_correcta": <bool>,    # toco el pad correcto (K pulsos -> pad K)
      "tiempo_respuesta_ms": <int>     # desde el ultimo pulso hasta el toque
    }
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from common.contratos import (
    ComandoActuador,
    EventoAckWearable,
    EventoWearable,
    TipoComandoActuador,
)

VERSION_PROTOCOLO = 1

# Tipo Pi->ESP fuera del enum de comandos de la FSM: apagado ORDENADO.
# No es un ComandoActuador (la FSM nunca lo emite); lo manda el
# ActuadorWearable en su detener(). El ESP lo usa para distinguir un apagado
# anunciado de una interrupcion imprevista.
TIPO_APAGADO_LIMPIO = 99

# Tipos de comando de la FSM que la pulsera entiende (el resto no se le envia).
# Es lo que ActuadorWearable.tipos_soportados() declara ante el Despachador.
TIPOS_PARA_WEARABLE = frozenset({
    TipoComandoActuador.VIBRAR_LEVE,
    TipoComandoActuador.VIBRAR_MEDIO,
    TipoComandoActuador.VIBRAR_FUERTE,
    TipoComandoActuador.SECUENCIA_ACK,
})

# ---------------------------------------------------------------------------
# Constantes de la verificacion de atencion. Son las del firmware
# (motor_pads.c): si se cambian alla, hay que cambiarlas aca.
# ---------------------------------------------------------------------------
DESAFIO_PULSOS_MAX = 4          # K se sortea entre 1 y 4
DESAFIO_PULSO_ON_MS = 200       # motor encendido por pulso
DESAFIO_PULSO_OFF_MS = 300      # pausa entre pulsos
VENTANA_DESAFIO_DEFAULT_MS = 10000   # ventana si la orden trae duracion_ms = 0

# Peor caso de la "presentacion" del desafio: lo que tarda la pulsera en
# emitir los K pulsos antes de abrir la ventana de respuesta. La Pi lo cubre
# con config.wearable.margen_presentacion_desafio_seg, para no declarar
# vencida una verificacion que la pulsera todavia acepta.
DESAFIO_PRESENTACION_MAX_MS = DESAFIO_PULSOS_MAX * (
    DESAFIO_PULSO_ON_MS + DESAFIO_PULSO_OFF_MS
)

# Repeticion de la respuesta del conductor (mismo criterio que las ordenes
# criticas, en sentido inverso).
ACK_REENVIOS = 3
ACK_ESPACIADO_MS = 50

# Cuantos identificadores de envio recuerda la pulsera para descartar
# repeticiones (DEDUP_N en red.c).
DEDUP_IDS_RECORDADOS = 16


class ErrorProtocolo(Exception):
    """El mensaje recibido no cumple el protocolo."""


# =============================================================================
#                     Pi -> ESP32 : serializar ordenes
# =============================================================================

def _serializar_orden(
    tipo: int,
    id_paquete: int,
    intensidad: int = 0,
    duracion_ms: int = 0,
    id_secuencia: Optional[int] = None,
) -> bytes:
    """Arma el JSON de una orden. Unico lugar donde se define su formato."""
    obj = {
        "v": VERSION_PROTOCOLO,
        "tipo": int(tipo),
        "intensidad": int(intensidad),
        "duracion_ms": int(duracion_ms),
        "id_secuencia": id_secuencia,
        "id_paquete": int(id_paquete),
    }
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def serializar_comando(comando: ComandoActuador, id_paquete: int) -> bytes:
    """Convierte un ComandoActuador en el JSON UDP que entiende la pulsera."""
    return _serializar_orden(
        tipo=int(comando.tipo),
        id_paquete=id_paquete,
        intensidad=comando.intensidad,
        duracion_ms=comando.duracion_ms,
        id_secuencia=comando.id_secuencia,
    )


def serializar_apagar(id_paquete: int) -> bytes:
    """Orden explicita de detencion (tipo 10): frena el motor y cancela el desafio."""
    return _serializar_orden(int(TipoComandoActuador.APAGAR_TODO), id_paquete)


def serializar_apagado_limpio(id_paquete: int) -> bytes:
    """Aviso de apagado ORDENADO (tipo 99). El ESP, al recibirlo, sabe que la
    proxima perdida de conexion es esperada y NO emite el aviso remoto."""
    return _serializar_orden(TIPO_APAGADO_LIMPIO, id_paquete)


# =============================================================================
#                     Decodificacion comun (ambos sentidos)
# =============================================================================

def _decodificar_objeto(datos: bytes) -> Dict[str, Any]:
    """
    bytes -> dict. Lanza ErrorProtocolo si no es UTF-8, no es JSON, o es JSON
    valido pero NO un objeto (una lista, un numero, un texto, null).

    Ese ultimo caso importa: json.loads(b"[1,2]") no falla, y sin esta
    verificacion el primer .get() posterior lanzaba AttributeError y terminaba
    el hilo receptor.
    """
    try:
        obj = json.loads(datos.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ErrorProtocolo(f"JSON invalido: {e}") from e
    if not isinstance(obj, dict):
        raise ErrorProtocolo(
            f"se esperaba un objeto JSON, llego {type(obj).__name__}"
        )
    return obj


def _entero(datos: Dict[str, Any], campo: str, default: Optional[int] = None) -> int:
    """
    Lee un campo entero. Si falta o es null usa `default`; si no hay default,
    o el valor no es un numero, lanza ErrorProtocolo.

    bool se rechaza a proposito: en Python True es instancia de int, y un
    `"id_secuencia": true` no es un identificador valido.
    """
    valor = datos.get(campo)
    if valor is None:
        if default is None:
            raise ErrorProtocolo(f"falta el campo obligatorio '{campo}'")
        return default
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ErrorProtocolo(f"campo '{campo}' no numerico: {valor!r}")
    return int(valor)


def _entero_opcional(datos: Dict[str, Any], campo: str) -> Optional[int]:
    """Como _entero, pero null/ausente se devuelve como None (dato no disponible)."""
    if datos.get(campo) is None:
        return None
    return _entero(datos, campo)


def parsear_comando(datos: bytes) -> Dict[str, Any]:
    """Lado ESP32/simulador: parsea una orden entrante. Valida lo minimo."""
    obj = _decodificar_objeto(datos)
    if "tipo" not in obj or "id_paquete" not in obj:
        raise ErrorProtocolo("faltan campos obligatorios (tipo/id_paquete)")
    # Se normalizan los dos campos que el simulador usa como enteros
    obj["tipo"] = _entero(obj, "tipo")
    obj["id_paquete"] = _entero(obj, "id_paquete")
    return obj


# =============================================================================
#                     ESP32 -> Pi : serializar telemetria / respuesta
# =============================================================================

def serializar_telemetria(
    bpm: Optional[int],
    bateria: Optional[int] = None,
    ack_recibido: bool = False,
    secuencia_replicada: bool = False,
    id_paquete: int = 0,
) -> bytes:
    """Lado ESP32/simulador: arma un mensaje de telemetria."""
    obj = {
        "v": VERSION_PROTOCOLO,
        "msg": "telemetria",
        "bpm": bpm,
        "ack_recibido": ack_recibido,
        "secuencia_replicada": secuencia_replicada,
        "bateria": bateria,
        "id_paquete": id_paquete,
    }
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def serializar_ack(
    id_secuencia: int,
    secuencia_correcta: bool,
    tiempo_respuesta_ms: int,
) -> bytes:
    """Lado ESP32/simulador: arma la respuesta a una verificacion."""
    obj = {
        "v": VERSION_PROTOCOLO,
        "msg": "ack",
        "id_secuencia": id_secuencia,
        "secuencia_correcta": secuencia_correcta,
        "tiempo_respuesta_ms": tiempo_respuesta_ms,
    }
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


# =============================================================================
#            ESP32 -> Pi : parsear en la Pi y construir el evento
# =============================================================================

def parsear_mensaje_pulsera(datos: bytes) -> Dict[str, Any]:
    """
    Lado Pi (receptor): parsea un datagrama de la pulsera y valida lo minimo.
    Devuelve el dict crudo. Lanza ErrorProtocolo si no cumple el protocolo.
    """
    obj = _decodificar_objeto(datos)
    msg = obj.get("msg")
    if msg not in ("telemetria", "ack"):
        raise ErrorProtocolo(f"campo 'msg' desconocido: {msg!r}")
    return obj


def construir_evento(datos_dict: Dict[str, Any], timestamp: float):
    """
    Convierte el dict parseado en el dataclass del contrato correspondiente.
    Devuelve un EventoWearable o un EventoAckWearable.

    Toda falla de contenido sale como ErrorProtocolo: campos faltantes, tipos
    equivocados (TypeError) y valores fuera de rango (ValueError del contrato).
    Asi el receptor tiene UNA sola excepcion que atender por datagrama.
    """
    msg = datos_dict.get("msg")
    try:
        if msg == "telemetria":
            return EventoWearable(
                timestamp=timestamp,
                bpm=_entero_opcional(datos_dict, "bpm"),
                ack_recibido=bool(datos_dict.get("ack_recibido", False)),
                secuencia_replicada=bool(datos_dict.get("secuencia_replicada", False)),
                bateria_porcentaje=_entero_opcional(datos_dict, "bateria"),
                id_paquete=_entero_opcional(datos_dict, "id_paquete"),
            )
        if msg == "ack":
            return EventoAckWearable(
                timestamp=timestamp,
                id_secuencia=_entero(datos_dict, "id_secuencia"),
                secuencia_correcta=bool(datos_dict.get("secuencia_correcta", False)),
                tiempo_respuesta_ms=_entero(datos_dict, "tiempo_respuesta_ms", default=0),
            )
    except (ValueError, TypeError) as e:
        raise ErrorProtocolo(f"contenido invalido en '{msg}': {e}") from e
    raise ErrorProtocolo(f"msg desconocido: {msg!r}")
