#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "red.h"
#include "bpm_max30102.h"
#include "motor_pads.h"
#include "bateria_led.h"

static const char *TAG = "NEURODRIVE";

// "red" pide el pulso sin conocer al sensor ni al bus I2C
static bool proveer_bpm(int *bpm_out) {
    return bpm_max30102_get(bpm_out);
}

// "red" pide la carga de la batería sin conocer el ADC
static int proveer_bateria(void) {
    return bateria_led_get_pct();
}

void app_main(void) {
    ESP_LOGI(TAG, "NeuroDrive pulsera - firmware ESP-IDF");

    // NVS requerido por esp_wifi
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    // La red primero: es lo crítico. El sensor de pulso se inicia al final
    // y en su propia tarea, de modo que no demora el arranque.
    red_set_comando_handler(motor_pads_ejecutar_comando);
    red_set_bpm_provider(proveer_bpm);
    red_set_bateria_provider(proveer_bateria);
    red_init();

    motor_pads_init();
    bateria_led_init();
    bpm_max30102_init();   // lanza su propio task, no bloquea

    // Ciclo principal. Cada módulo avanza un paso y vuelve en el acto:
    // ninguno espera. El sensor de pulso y el aviso remoto corren en tareas
    // propias.
    while (1) {
        red_update();
        motor_pads_update();
        bateria_led_update();

        // El MAX30102 solo mide mientras la pulsera esta conectada a la AP de la Pi
        // (estado OPERANDO_PI y WiFi con IP). Idempotente.
        bpm_max30102_set_activo(red_operando_con_pi());

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
