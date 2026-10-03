#pragma once
#include "banco.h"
typedef struct esp_netif_obj esp_netif_t;
typedef struct { uint32_t addr; } esp_ip4_addr_t;
typedef struct { esp_ip4_addr_t ip, netmask, gw; } esp_netif_ip_info_t;
esp_err_t esp_netif_init(void);
esp_netif_t *esp_netif_create_default_wifi_sta(void);
esp_err_t esp_netif_dhcpc_stop(esp_netif_t *n);
esp_err_t esp_netif_dhcpc_start(esp_netif_t *n);
esp_err_t esp_netif_set_ip_info(esp_netif_t *n, const esp_netif_ip_info_t *i);
esp_err_t esp_netif_set_hostname(esp_netif_t *n, const char *h);
void esp_netif_set_ip4_addr(esp_ip4_addr_t *a, uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3);
