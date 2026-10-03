#pragma once
#include "banco.h"
typedef const char *esp_event_base_t;
typedef void (*esp_event_handler_t)(void *arg, esp_event_base_t base, int32_t id, void *data);
extern esp_event_base_t WIFI_EVENT, IP_EVENT;
#define ESP_EVENT_ANY_ID (-1)
#define IP_EVENT_STA_GOT_IP 0
esp_err_t esp_event_loop_create_default(void);
esp_err_t esp_event_handler_register(esp_event_base_t base, int32_t id, esp_event_handler_t h, void *arg);
