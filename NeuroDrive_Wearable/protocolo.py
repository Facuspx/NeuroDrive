"""
NeuroDrive_Wearable - Protocolo en el cable (JSON sobre UDP)
============================================================

Fuente de verdad UNICA del formato de mensajes entre la Pi y el ESP32.
Lo usan el actuador (Pi->ESP), el receptor (ESP->Pi) y el simulador.

------------------------------------------------------------------
COMANDOS  Pi -> ESP32  (puerto de envio, default 5006)
------------------------------------------------------------------
    {
      "v": 1,
      "tipo": <int>,          # TipoComandoActuador (1=VIBRAR_LEVE ... 8=SECUENCIA_ACK, 10=APAGAR_TODO)
      "intensidad": <0-100>,
      "duracion_ms": <int>,
      "id_secuencia": <int|null>,   # solo en SECUENCIA_ACK
      "id_paquete": <int>           # para deduplicar reenvios en el ESP
    }

Solo se envian los tipos que la pulsera entiende:
    VIBRAR_LEVE(1), VIBRAR_MEDIO(2), VIBRAR_FUERTE(3), SECUENCIA_ACK(8),
    APAGAR_TODO(10)  (este ultimo detiene el motor y cancela el desafio).

------------------------------------------------------------------
TELEMETRIA / ACK  ESP32 -> Pi  (puerto de escucha, default 5005)
------------------------------------------------------------------
Telemetria periodica (BPM + salud), cada ~2s (doble funcion de heartbeat):
    {
      "v": 1, "msg": "telemetria",
      "bpm": <int|null>,               # None si el MAX30102 no tiene lectura confiable
      "ack_recibido": <bool>,
      "secuencia_replicada": <bool>,
      "bateria": <int|null>,           # 0-100
      "id_paquete": <int>
    }

Respuesta a un desafio de confirmacion:
    {
      "v": 1, "msg": "ack",
      "id_secuencia": <int>,           # el mismo id que vino en el comando SECUENCIA_ACK
      "secuencia_correcta": <bool>,    # toco el pad correcto (K vibraciones -> pad K)
      "tiempo_respuesta_ms": <int>
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

# Tipos de comando que la pulsera entiende (el resto no se le envia)
TIPOS_PARA_WEARABLE = frozenset({
    TipoComandoActuador.VIBRAR_LEVE,
    TipoComandoActuador.VIBRAR_MEDIO,
    TipoComandoActuador.VIBRAR_FUERTE,
    TipoComandoActuador.SECUENCIA_ACK,
})


class ErrorProtocolo(Exception):
    """El mensaje recibido no cumple el protocolo."""


# =============================================================================
#                     Pi -> ESP32 : serializar comandos
# =============================================================================

def serializar_comando(comando: ComandoActuador, id_paquete: int) -> bytes:
    """Convierte un ComandoActuador en el JSON UDP que entiende la pulsera."""
    obj = {
        "v": VERSION_PROTOCOLO,
        "tipo": int(comando.tipo),
        "intensidad": comando.intensidad,
        "duracion_ms": comando.duracion_ms,
        "id_secuencia": comando.id_secuencia,
        "id_paquete": id_paquete,
    }
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def serializar_apagar(id_paquete: int) -> bytes:
    """Comando explicito de apagado (detiene motor y cancela desafio)."""
    obj = {
        "v": VERSION_PROTOCOLO,
        "tipo": int(TipoComandoActuador.APAGAR_TODO),
        "intensidad": 0,
        "duracion_ms": 0,
        "id_secuencia": None,
        "id_paquete": id_paquete,
    }
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def parsear_comando(datos: bytes) -> Dict[str, Any]:
    """Lado ESP32/simulador: parsea un comando entrante. Valida lo minimo."""
    try:
        obj = json.loads(datos.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ErrorProtocolo(f"JSON invalido: {e}") from e
    if "tipo" not in obj or "id_paquete" not in obj:
        raise ErrorProtocolo("faltan campos obligatorios (tipo/id_paquete)")
    return obj


# =============================================================================
#                     ESP32 -> Pi : serializar telemetria / ack
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
    """Lado ESP32/simulador: arma la respuesta a un desafio."""
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
    Devuelve el dict crudo. Lanza ErrorProtocolo si es basura.
    """
    try:
        obj = json.loads(datos.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ErrorProtocolo(f"JSON invalido: {e}") from e

    msg = obj.get("msg")
    if msg not in ("telemetria", "ack"):
        raise ErrorProtocolo(f"campo 'msg' desconocido: {msg!r}")
    return obj


def construir_evento(datos_dict: Dict[str, Any], timestamp: float):
    """
    Convierte el dict parseado en el dataclass del contrato correspondiente.
    Devuelve un EventoWearable o un EventoAckWearable.
    Lanza ErrorProtocolo (o ValueError del contrato) si los datos no validan.
    """
    msg = datos_dict.get("msg")
    if msg == "telemetria":
        return EventoWearable(
            timestamp=timestamp,
            bpm=datos_dict.get("bpm"),
            ack_recibido=bool(datos_dict.get("ack_recibido", False)),
            secuencia_replicada=bool(datos_dict.get("secuencia_replicada", False)),
            bateria_porcentaje=datos_dict.get("bateria"),
            id_paquete=datos_dict.get("id_paquete"),
        )
    elif msg == "ack":
        if "id_secuencia" not in datos_dict:
            raise ErrorProtocolo("ack sin id_secuencia")
        return EventoAckWearable(
            timestamp=timestamp,
            id_secuencia=int(datos_dict["id_secuencia"]),
            secuencia_correcta=bool(datos_dict.get("secuencia_correcta", False)),
            tiempo_respuesta_ms=int(datos_dict.get("tiempo_respuesta_ms", 0)),
        )
    raise ErrorProtocolo(f"msg desconocido: {msg!r}")
