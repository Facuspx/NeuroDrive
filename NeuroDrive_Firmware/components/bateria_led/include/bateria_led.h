#pragma once

// Medición de la batería (ADC en GP1, divisor 100K/100K) y dos indicadores
// RGB direccionables: conectividad (GP43) y batería (GP44).
void bateria_led_init(void);
void bateria_led_update(void);   // llamar en cada vuelta del ciclo principal

// Carga estimada de la batería, 0-100. Se registra con red_set_bateria_provider().
int bateria_led_get_pct(void);
