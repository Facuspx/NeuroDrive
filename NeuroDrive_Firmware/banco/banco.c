// Implementacion del banco: el "hardware" que ve el firmware y los controles del test.
#include <stdarg.h>
#include <stdlib.h>
#include "banco.h"
#include "esp_timer.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_http_client.h"
#include "freertos/task.h"
#include "driver/ledc.h"
#include "driver/touch_pad.h"
#include "driver/gpio.h"

// ---------------- reloj ----------------
static int64_t reloj_us = 1000000;
int64_t esp_timer_get_time(void) { return reloj_us; }
void banco_avanzar_ms(int ms) { reloj_us += (int64_t)ms * 1000; }
int64_t banco_ahora_ms(void) { return reloj_us / 1000; }

// ---------------- registro ----------------
static char registro[65536]; static size_t reg_len = 0;
void banco_log(char nivel, const char *tag, const char *fmt, ...) {
    char linea[512]; va_list ap; va_start(ap, fmt);
    int n = snprintf(linea, sizeof(linea), "%c %s: ", nivel, tag);
    vsnprintf(linea + n, sizeof(linea) - n, fmt, ap); va_end(ap);
    size_t l = strlen(linea);
    if (reg_len + l + 2 < sizeof(registro)) { memcpy(registro + reg_len, linea, l); reg_len += l; registro[reg_len++] = '\n'; registro[reg_len] = 0; }
}
const char *banco_registro(void) { return registro; }
void banco_limpiar_registro(void) { reg_len = 0; registro[0] = 0; }
const char *esp_err_to_name(esp_err_t e) { return e == ESP_OK ? "ESP_OK" : "ESP_FAIL"; }

// ---------------- Wi-Fi ----------------
esp_event_base_t WIFI_EVENT = "WIFI_EVENT", IP_EVENT = "IP_EVENT";
static esp_event_handler_t manejador = NULL;
static char ssid_actual[33] = ""; static char clave_actual[65] = "";
static int ip_fija = -1, asociaciones_pedidas = 0, wifi_detenciones = 0;
esp_err_t esp_event_loop_create_default(void) { return ESP_OK; }
esp_err_t esp_event_handler_register(esp_event_base_t b, int32_t id, esp_event_handler_t h, void *a) { manejador = h; return ESP_OK; }
esp_err_t esp_netif_init(void) { return ESP_OK; }
esp_netif_t *esp_netif_create_default_wifi_sta(void) { return (esp_netif_t *)1; }
esp_err_t esp_netif_dhcpc_stop(esp_netif_t *n) { ip_fija = 1; return ESP_OK; }
esp_err_t esp_netif_dhcpc_start(esp_netif_t *n) { ip_fija = 0; return ESP_OK; }
esp_err_t esp_netif_set_ip_info(esp_netif_t *n, const esp_netif_ip_info_t *i) { return ESP_OK; }
esp_err_t esp_netif_set_hostname(esp_netif_t *n, const char *h) { return ESP_OK; }
void esp_netif_set_ip4_addr(esp_ip4_addr_t *a, uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) { a->addr = b0 | (b1 << 8) | (b2 << 16) | ((uint32_t)b3 << 24); }
esp_err_t esp_wifi_init(const wifi_init_config_t *c) { return ESP_OK; }
esp_err_t esp_wifi_set_mode(int m) { return ESP_OK; }
esp_err_t esp_wifi_set_config(int ifx, wifi_config_t *c) { strncpy(ssid_actual, (char *)c->sta.ssid, 32); strncpy(clave_actual, (char *)c->sta.password, 64); return ESP_OK; }
esp_err_t esp_wifi_start(void) { return ESP_OK; }
esp_err_t esp_wifi_connect(void) { asociaciones_pedidas++; return ESP_OK; }
esp_err_t esp_wifi_disconnect(void) { return ESP_OK; }
esp_err_t esp_wifi_stop(void) { wifi_detenciones++; return ESP_OK; }
const char *banco_ssid(void) { return ssid_actual; }
const char *banco_clave(void) { return clave_actual; }
int banco_ip_fija(void) { return ip_fija; }
int banco_asociaciones(void) { return asociaciones_pedidas; }
void banco_evento_conectado(void) { if (manejador) manejador(NULL, IP_EVENT, IP_EVENT_STA_GOT_IP, NULL); }
void banco_evento_desconectado(void) { if (manejador) manejador(NULL, WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, NULL); }

// ---------------- tareas ----------------
// xTaskCreate no ejecuta nada: guarda la funcion. El test decide cuando
// "corre" la tarea, para reproducir que el aviso tarda varios segundos.
static void (*tarea_pendiente)(void *) = NULL; static uint32_t tarea_pila = 0; static int tareas_creadas = 0;
BaseType_t xTaskCreate(void (*fn)(void *), const char *nombre, uint32_t pila, void *arg, UBaseType_t prio, TaskHandle_t *h) { tarea_pendiente = fn; tarea_pila = pila; tareas_creadas++; return pdPASS; }
void vTaskDelete(TaskHandle_t h) {}
static int pausas = 0;
void vTaskDelay(uint32_t ticks) { pausas++; reloj_us += (int64_t)ticks * 10000; }   // el firmware nuevo no deberia llamarla desde red/motor_pads
int banco_pausas(void) { return pausas; }
int banco_tareas_creadas(void) { return tareas_creadas; }
int banco_tarea_pendiente(void) { return tarea_pendiente != NULL; }
uint32_t banco_tarea_pila(void) { return tarea_pila; }
void banco_correr_tarea(void) { if (tarea_pendiente) { void (*f)(void *) = tarea_pendiente; tarea_pendiente = NULL; f(NULL); } }

// ---------------- HTTPS ----------------
static char http_url[256] = "", http_cuerpo[1024] = ""; static int http_envios = 0, http_resultado_ok = 1, http_verifica_nombre = -1, http_timeout = 0;
esp_http_client_handle_t esp_http_client_init(const esp_http_client_config_t *c) { strncpy(http_url, c->url, 255); http_verifica_nombre = !c->skip_cert_common_name_check; http_timeout = c->timeout_ms; return (esp_http_client_handle_t)1; }
esp_err_t esp_http_client_set_header(esp_http_client_handle_t c, const char *k, const char *v) { return ESP_OK; }
esp_err_t esp_http_client_set_post_field(esp_http_client_handle_t c, const char *d, int len) { snprintf(http_cuerpo, sizeof(http_cuerpo), "%.*s", len, d); return ESP_OK; }
esp_err_t esp_http_client_perform(esp_http_client_handle_t c) { http_envios++; return http_resultado_ok ? ESP_OK : ESP_FAIL; }
int esp_http_client_get_status_code(esp_http_client_handle_t c) { return http_resultado_ok ? 200 : 0; }
esp_err_t esp_http_client_cleanup(esp_http_client_handle_t c) { return ESP_OK; }
esp_err_t esp_crt_bundle_attach(void *conf) { return ESP_OK; }
int banco_http_envios(void) { return http_envios; }
const char *banco_http_cuerpo(void) { return http_cuerpo; }
const char *banco_http_url(void) { return http_url; }
int banco_http_verifica_nombre(void) { return http_verifica_nombre; }
int banco_http_timeout(void) { return http_timeout; }
void banco_http_resultado(int ok) { http_resultado_ok = ok; }

// ---------------- motor, pads, retroiluminacion ----------------
static uint32_t duty = 0, proximo_azar = 0; static int led_pads = 0, flancos_motor = 0; static uint32_t pad_raw[8] = {0};
esp_err_t ledc_timer_config(const ledc_timer_config_t *c) { return ESP_OK; }
esp_err_t ledc_channel_config(const ledc_channel_config_t *c) { return ESP_OK; }
esp_err_t ledc_set_duty(int m, int c, uint32_t d) { if (d > 0 && duty == 0) flancos_motor++; duty = d; return ESP_OK; }
esp_err_t ledc_update_duty(int m, int c) { return ESP_OK; }
esp_err_t touch_pad_init(void) { return ESP_OK; }
esp_err_t touch_pad_set_fsm_mode(int m) { return ESP_OK; }
esp_err_t touch_pad_set_voltage(int a, int b, int c) { return ESP_OK; }
esp_err_t touch_pad_config(touch_pad_t p) { return ESP_OK; }
esp_err_t touch_pad_fsm_start(void) { return ESP_OK; }
esp_err_t touch_pad_read_raw_data(touch_pad_t p, uint32_t *raw) { *raw = pad_raw[p & 7]; return ESP_OK; }
esp_err_t gpio_reset_pin(int pin) { return ESP_OK; }
esp_err_t gpio_config(const gpio_config_t *c) { return ESP_OK; }
esp_err_t gpio_set_level(int pin, uint32_t nivel) { if (pin == 7) led_pads = (int)nivel; return ESP_OK; }
int gpio_get_level(int pin) { return pin == 7 ? led_pads : 0; }
uint32_t esp_random(void) { return proximo_azar; }
void banco_fijar_k(int k) { proximo_azar = (uint32_t)(k - 1); }   // K = azar % 4 + 1
uint32_t banco_duty(void) { return duty; }
int banco_pulsos_motor(void) { return flancos_motor; }
void banco_reiniciar_pulsos(void) { flancos_motor = 0; }
int banco_led_pads(void) { return led_pads; }
// pad 1..4 -> canales tactiles 6, 4, 5, 2
void banco_tocar_pad(int pad, int tocado) { static const int canal[5] = {0, 6, 4, 5, 2}; pad_raw[canal[pad]] = tocado ? 50000 : 1000; }
