#pragma once
#include "banco.h"
typedef struct { struct { uint8_t ssid[32]; uint8_t password[64]; } sta; } wifi_config_t;
typedef struct { int nada; } wifi_init_config_t;
#define WIFI_INIT_CONFIG_DEFAULT() { 0 }
#define WIFI_MODE_STA 1
#define WIFI_IF_STA 0
#define WIFI_EVENT_STA_DISCONNECTED 5
esp_err_t esp_wifi_init(const wifi_init_config_t *c);
esp_err_t esp_wifi_set_mode(int m);
esp_err_t esp_wifi_set_config(int ifx, wifi_config_t *c);
esp_err_t esp_wifi_start(void);
esp_err_t esp_wifi_connect(void);
esp_err_t esp_wifi_disconnect(void);
esp_err_t esp_wifi_stop(void);
