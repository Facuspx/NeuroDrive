#pragma once
#include <stdint.h>
#include "esp_timer.h"

// Milisegundos desde el arranque. Todas las esperas del firmware se hacen
// comparando contra este reloj, no con pausas: el ciclo principal nunca se
// detiene a esperar.
static inline int64_t millis(void) { return esp_timer_get_time() / 1000; }
