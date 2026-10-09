#pragma once
#include <stdint.h>

// Medición de la batería (ADC en GP1, divisor 100K/100K) y dos indicadores
// RGB direccionables: conectividad (GP43) y batería (GP44). Los indicadores
// destellan; no quedan encendidos.
void bateria_led_init(void);
void bateria_led_update(void);   // llamar en cada vuelta del ciclo principal

// Milisegundos hasta el próximo encendido o apagado de un indicador
uint32_t bateria_led_ms_hasta_cambio(void);

// Carga estimada de la batería, 0-100. Se registra con red_set_bateria_provider().
int bateria_led_get_pct(void);
