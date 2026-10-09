#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_pm.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#include "red.h"
#include "bpm_max30102.h"
#include "motor_pads.h"
#include "bateria_led.h"

static const char *TAG = "NEURODRIVE";

// ===========================================================================
// GESTION DE ENERGIA
//
// La pulsera pasa casi todo el tiempo sin nada que hacer: asociada a la Pi,
// esperando una orden y enviando la telemetria cada dos segundos. En ese
// estado, que aca se llama reposo, el procesador no hace falta.
//
//   ENERGIA_NIVEL 0  Sin gestion. El procesador queda siempre encendido a la
//                    frecuencia configurada y el ciclo principal da una
//                    vuelta cada CICLO_ACTIVO_MS. Es el comportamiento
//                    anterior a este cambio.
//   ENERGIA_NIVEL 1  En reposo el procesador baja a ENERGIA_FREC_MIN_MHZ y
//                    el ciclo principal espera CICLO_REPOSO_MS, o menos si
//                    llega una orden. El procesador no se detiene.
//   ENERGIA_NIVEL 2  Ademas, en reposo el microcontrolador entra en reposo
//                    ligero (light sleep): se detienen el procesador y los
//                    relojes, y se conservan la memoria y la asociacion
//                    Wi-Fi. Lo despiertan el vencimiento de una espera o la
//                    radio, cuando el punto de acceso anuncia trafico.
//
// Hay actividad, y por lo tanto no hay reposo, mientras el motor vibra, hay
// una verificacion en curso o la conectividad no esta estable. En esos casos
// el ciclo vuelve a CICLO_ACTIVO_MS y un bloqueo de energia mantiene los
// relojes: el PWM del motor los necesita.
//
// El bloqueo se toma tambien durante cada vuelta del ciclo, aun en reposo, y
// se suelta solo para esperar. Asi todo el codigo de los modulos se ejecuta
// siempre con los relojes con que se configuraron los perifericos, y lo
// unico que ocurre a frecuencia reducida o dormido es la espera.
//
// Los niveles 1 y 2 requieren en la configuracion del proyecto (menuconfig):
//     Component config > Power Management > Support for power management
//     Component config > FreeRTOS > Kernel > configUSE_TICKLESS_IDLE   (nivel 2)
// Si faltan, el firmware funciona igual, en el nivel que la configuracion
// permita, y lo informa al arrancar.
//
// Con el cable USB conectado a una computadora el microcontrolador no entra
// en reposo ligero (opcion "Don't enter the automatic light sleep when USB
// Serial/JTAG port is connected"): asi el monitor serie sigue funcionando.
// El consumo se mide alimentando la pulsera desde la bateria.
// ===========================================================================
#define ENERGIA_NIVEL          2

#define ENERGIA_FREC_MIN_MHZ   40      // frecuencia del procesador en reposo
#define CICLO_ACTIVO_MS        20      // vuelta del ciclo principal con actividad
#define CICLO_REPOSO_MS        100     // espera maxima del ciclo principal en reposo

#if CONFIG_PM_ENABLE && (ENERGIA_NIVEL > 0)
#define ENERGIA_GESTIONADA 1
#else
#define ENERGIA_GESTIONADA 0
#endif

#if ENERGIA_GESTIONADA
static esp_pm_lock_handle_t s_lock_actividad = NULL;
static bool s_hay_actividad = false;
#endif

// Toma o suelta el bloqueo que impide el reposo. Con el bloqueo tomado el
// procesador no baja de 80 MHz, que es la frecuencia a la que se configuran
// los perifericos, y el microcontrolador no duerme.
static void energia_marcar_actividad(bool actividad) {
#if ENERGIA_GESTIONADA
    if (!s_lock_actividad || actividad == s_hay_actividad) return;
    if (actividad) esp_pm_lock_acquire(s_lock_actividad);
    else           esp_pm_lock_release(s_lock_actividad);
    s_hay_actividad = actividad;
#else
    (void)actividad;
#endif
}

// Debe llamarse antes de iniciar los perifericos: deja tomado el bloqueo de
// actividad, de modo que todos se configuren con los relojes a pleno.
static void energia_init(void) {
#if ENERGIA_GESTIONADA
    if (esp_pm_lock_create(ESP_PM_APB_FREQ_MAX, 0, "actividad", &s_lock_actividad) != ESP_OK) {
        s_lock_actividad = NULL;
        ESP_LOGW(TAG, "[energia] no se pudo crear el bloqueo de actividad: sin gestion de energia");
        return;
    }
    energia_marcar_actividad(true);

    esp_pm_config_t cfg = {
        .max_freq_mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
        .min_freq_mhz = ENERGIA_FREC_MIN_MHZ,
        .light_sleep_enable = (ENERGIA_NIVEL >= 2),
    };
    esp_err_t err = esp_pm_configure(&cfg);
    if (err != ESP_OK && cfg.light_sleep_enable) {
        // Falta configUSE_TICKLESS_IDLE en la configuracion del proyecto
        ESP_LOGW(TAG, "[energia] reposo ligero no disponible (%s): se sigue en nivel 1",
                 esp_err_to_name(err));
        cfg.light_sleep_enable = false;
        err = esp_pm_configure(&cfg);
    }
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "[energia] nivel %d: en reposo %d MHz, reposo ligero %s",
                 cfg.light_sleep_enable ? 2 : 1, ENERGIA_FREC_MIN_MHZ,
                 cfg.light_sleep_enable ? "SI" : "NO");
    } else {
        ESP_LOGW(TAG, "[energia] no se pudo configurar (%s): sin gestion de energia",
                 esp_err_to_name(err));
    }
#else
    ESP_LOGI(TAG, "[energia] nivel 0: sin gestion de energia");
#endif
}

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

    energia_init();

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
    // propias. La única espera es la del final de la vuelta.
    while (1) {
        energia_marcar_actividad(true);

        red_update();
        motor_pads_update();
        bateria_led_update();

        // El MAX30102 solo mide mientras la pulsera esta conectada a la AP de la Pi
        // (estado OPERANDO_PI y WiFi con IP). Idempotente.
        bpm_max30102_set_activo(red_operando_con_pi());

        if (ENERGIA_GESTIONADA && red_en_reposo() && motor_pads_en_reposo()) {
            // Sin nada en curso: se espera una orden, o el próximo cambio de
            // un indicador, o CICLO_REPOSO_MS; lo que ocurra primero. Durante
            // esta espera, y solo durante ella, se permite el reposo.
            uint32_t espera_ms = bateria_led_ms_hasta_cambio();
            if (espera_ms > CICLO_REPOSO_MS) espera_ms = CICLO_REPOSO_MS;
            if (espera_ms < portTICK_PERIOD_MS) espera_ms = portTICK_PERIOD_MS;
            energia_marcar_actividad(false);
            red_esperar_orden(espera_ms);
        } else {
            vTaskDelay(pdMS_TO_TICKS(CICLO_ACTIVO_MS));
        }
    }
}
