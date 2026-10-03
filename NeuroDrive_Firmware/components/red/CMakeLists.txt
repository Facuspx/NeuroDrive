idf_component_register(
    SRCS "red.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_wifi esp_event esp_netif lwip esp_http_client json esp_timer mbedtls
)