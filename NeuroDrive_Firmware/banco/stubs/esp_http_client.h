#pragma once
#include "banco.h"
typedef struct esp_http_client *esp_http_client_handle_t;
typedef enum { HTTP_METHOD_POST = 1 } esp_http_client_method_t;
typedef struct {
    const char *url; esp_http_client_method_t method; int timeout_ms;
    esp_err_t (*crt_bundle_attach)(void *conf);
    bool skip_cert_common_name_check;
} esp_http_client_config_t;
esp_http_client_handle_t esp_http_client_init(const esp_http_client_config_t *c);
esp_err_t esp_http_client_set_header(esp_http_client_handle_t c, const char *k, const char *v);
esp_err_t esp_http_client_set_post_field(esp_http_client_handle_t c, const char *d, int len);
esp_err_t esp_http_client_perform(esp_http_client_handle_t c);
int esp_http_client_get_status_code(esp_http_client_handle_t c);
esp_err_t esp_http_client_cleanup(esp_http_client_handle_t c);
