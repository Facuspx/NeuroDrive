idf_component_register(
    SRCS "bateria_led.c"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_timer red esp_adc
)