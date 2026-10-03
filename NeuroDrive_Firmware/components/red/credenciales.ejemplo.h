#pragma once
// ---------------------------------------------------------------------------
// PLANTILLA de credenciales. Este archivo SI se versiona.
//
// Copiarlo en esta misma carpeta con el nombre  credenciales.h  y completar
// los valores reales. credenciales.h esta excluido del control de versiones
// (ver .gitignore): las claves no deben llegar al repositorio.
//
// Sin credenciales.h el proyecto no compila, y el error indica que falta.
// ---------------------------------------------------------------------------

// Red de la unidad de procesamiento (punto de acceso de la Raspberry Pi).
// Deben coincidir con los de neurodrive_ap.sh.
#define PI_SSID          "NeuroDrive_AP"
#define PI_PASS          "CLAVE_DEL_PUNTO_DE_ACCESO"

// Red de respaldo con salida a Internet (para el aviso remoto)
#define CAMION_SSID      "NOMBRE_DE_LA_RED_DE_RESPALDO"
#define CAMION_PASS      "CLAVE_DE_LA_RED_DE_RESPALDO"

// Servicio de mensajeria para el aviso remoto
#define TG_TOKEN         "TOKEN_DEL_BOT"
#define TG_CHAT_ID       "IDENTIFICADOR_DEL_DESTINATARIO"
