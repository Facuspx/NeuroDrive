#pragma once
#include <stdbool.h>

// Arranca la recuperación de bus I2C, inicializa el sensor y lanza la
// tarea de detección de BPM (corre en su propio task, independiente del
// loop principal).
void bpm_max30102_init(void);

// Getter no bloqueante para que "red" arme la telemetría.
// Retorna true si hay un BPM válido y confirmado (contacto real, no una
// superficie inerte); en ese caso escribe el valor en *bpm_out.
bool bpm_max30102_get(int *bpm_out);

// Pide encender (true) o apagar (false) el sensor. El sensor ARRANCA APAGADO y solo mide
// mientras se pida true (main.c: pulsera conectada a la AP de la Pi). No toca el chip: deja
// el pedido y la task del sensor ejecuta el init o el SHDN. Idempotente.
void bpm_max30102_set_activo(bool activo);

// Estado real del chip (true = midiendo). Util para logs/tests.
bool bpm_max30102_esta_activo(void);