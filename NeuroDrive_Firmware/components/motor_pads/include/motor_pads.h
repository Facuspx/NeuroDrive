#pragma once
#include <stdint.h>
#include <stdbool.h>

// Motor de vibracion (PWM, GP3), cuatro pads capacitivos (GP6, GP4, GP5, GP2),
// retroiluminacion de los pads (GP7) y verificacion de atencion.

void motor_pads_init(void);
void motor_pads_update(void);   // llamar en cada vuelta del ciclo principal; no bloquea

// Ejecuta una orden recibida de la Pi. Se registra con red_set_comando_handler().
// En la orden de verificacion (tipo 8), duracion_ms es la ventana de respuesta.
void motor_pads_ejecutar_comando(int tipo, int intensidad,
                                  uint32_t duracion_ms, int id_secuencia);
