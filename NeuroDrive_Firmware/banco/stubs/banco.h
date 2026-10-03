// Banco de pruebas en PC: reemplaza el hardware y el Wi-Fi del ESP32 por
// funciones que el test controla. red.c y motor_pads.c se compilan SIN cambios.
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
typedef int esp_err_t;
#define ESP_OK 0
#define ESP_FAIL -1
const char *esp_err_to_name(esp_err_t e);
#define ESP_ERROR_CHECK(x) do { esp_err_t _e = (x); (void)_e; } while (0)
void banco_log(char nivel, const char *tag, const char *fmt, ...);
#define ESP_LOGE(tag, fmt, ...) banco_log('E', tag, fmt, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) banco_log('W', tag, fmt, ##__VA_ARGS__)
#define ESP_LOGI(tag, fmt, ...) banco_log('I', tag, fmt, ##__VA_ARGS__)
#define ESP_LOGD(tag, fmt, ...) banco_log('D', tag, fmt, ##__VA_ARGS__)
