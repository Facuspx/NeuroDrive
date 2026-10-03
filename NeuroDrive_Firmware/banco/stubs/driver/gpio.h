#pragma once
#include "banco.h"
#define GPIO_MODE_OUTPUT 2
typedef struct { uint64_t pin_bit_mask; int mode; } gpio_config_t;
esp_err_t gpio_reset_pin(int pin);
esp_err_t gpio_config(const gpio_config_t *c);
esp_err_t gpio_set_level(int pin, uint32_t nivel);
int gpio_get_level(int pin);
