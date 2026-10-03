idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES red bpm_max30102 motor_pads bateria_led nvs_flash
)
