#include "freertos/FreeRTOS.h"
#include "esp_random.h"
#include "esp_log.h"
#include "driver/ledc.h"
#include "driver/touch_pad.h"
#include "driver/gpio.h"

#include "motor_pads.h"
#include "red.h"            // red_enviar_ack() + red_marcar_secuencia_iniciada()
#include "protocolo_nd.h"   // tipos de orden y constantes de la verificacion
#include "reloj.h"          // millis()

static const char *TAG = "MOTOR_PADS";

// ===========================================================================
// MOTOR (M1) — LEDC PWM
//
// GP3 es un pin de configuración de arranque del ESP32-S3 (origen de JTAG).
// Tenerlo en cuenta si se modifica el circuito del motor: no debe forzar la
// línea durante el reinicio.
// ===========================================================================
#define PIN_MOTOR         3
#define PWM_FREQ_HZ       20000
#define PWM_RESOLUCION    LEDC_TIMER_8_BIT
#define PWM_TIMER         LEDC_TIMER_0
#define PWM_CANAL         LEDC_CHANNEL_0
#define PWM_MODO          LEDC_LOW_SPEED_MODE   // único modo disponible en S3

// LEDS_PADS — retroiluminación de los pads táctiles. Se enciende mientras
// hay una verificación en curso (vibrando o esperando el toque), para que el
// conductor ubique los pads en la oscuridad, y se apaga cuando termina:
// respuesta dada, ventana vencida o cancelación por APAGAR_TODO.
// Línea confirmada en la placa por continuidad: GPIO7.
#define PIN_LEDS_PADS     7

static bool     motor_activo = false;
static int64_t  motor_ts_fin_ms = 0;
static uint8_t  motor_intensidad = 0;

static uint32_t intensidad_a_duty(uint8_t v) {
    if (v > 100) v = 100;
    return (uint32_t)v * 255u / 100u;   // 8 bits -> max 255
}

static void motor_apagar(void) {
    ledc_set_duty(PWM_MODO, PWM_CANAL, 0);
    ledc_update_duty(PWM_MODO, PWM_CANAL);
    motor_activo = false;
    motor_ts_fin_ms = 0;
    motor_intensidad = 0;
}

static void motor_vibrar(uint8_t v, uint32_t dur_ms) {
    motor_intensidad = (v > 100) ? 100 : v;
    ledc_set_duty(PWM_MODO, PWM_CANAL, intensidad_a_duty(motor_intensidad));
    ledc_update_duty(PWM_MODO, PWM_CANAL);
    motor_activo = true;
    motor_ts_fin_ms = (dur_ms == 0) ? 0 : (millis() + dur_ms);
}

static void motor_update(void) {
    if (motor_activo && motor_ts_fin_ms != 0 && (millis() - motor_ts_fin_ms) >= 0) {
        motor_apagar();
    }
}

// LEDS_PADS es activa en alto (1 = encendida, 0 = apagada), verificado en la placa.
static inline void leds_pads_encender(void) { gpio_set_level(PIN_LEDS_PADS, 1); }
static inline void leds_pads_apagar(void)   { gpio_set_level(PIN_LEDS_PADS, 0); }

// ===========================================================================
// PADS CAPACITIVOS
//
// En el ESP32-S3 el número de canal táctil coincide con el número de GPIO
// (GPIOn -> TOUCH_PAD_NUMn). Asignación verificada en la placa:
//   Pad_1=GP6  Pad_2=GP4  Pad_3=GP5  Pad_4=GP2
//
// Umbral: valor ajustado en la placa sobre GP2 (~30000 en la escala de
// touch_pad_read_raw_data). Los cuatro pads comparten el umbral; si alguno
// diera toques falsos o no respondiera, corresponde ajustarlo por separado.
// ===========================================================================
static const touch_pad_t PAD_TOUCH_NUM[4] = {
    TOUCH_PAD_NUM6,   // pad1 = GP6
    TOUCH_PAD_NUM4,   // pad2 = GP4
    TOUCH_PAD_NUM5,   // pad3 = GP5
    TOUCH_PAD_NUM2,   // pad4 = GP2
};
#define PAD_UMBRAL          30000
#define PAD_ANTIRREBOTE_MS  60

static bool    pad_estaba_tocado[4] = {false, false, false, false};
static int64_t pad_ts_supero_ms[4]  = {0, 0, 0, 0};

// Devuelve 1..4 si detecta el flanco de un toque nuevo en ese pad, 0 si no.
static int pads_leer_flanco(void) {
    int det = 0;
    for (int i = 0; i < 4; i++) {
        uint32_t raw = 0;
        touch_pad_read_raw_data(PAD_TOUCH_NUM[i], &raw);
        bool sobre = (raw > PAD_UMBRAL);
        if (sobre) {
            if (pad_ts_supero_ms[i] == 0) pad_ts_supero_ms[i] = millis();
            if ((millis() - pad_ts_supero_ms[i]) >= PAD_ANTIRREBOTE_MS && !pad_estaba_tocado[i]) {
                pad_estaba_tocado[i] = true;
                det = i + 1;
                ESP_LOGI(TAG, "[pad] tocaste PAD %d (raw=%lu, umbral=%d)", det, (unsigned long)raw, PAD_UMBRAL);
            }
        } else {
            pad_ts_supero_ms[i] = 0;
            pad_estaba_tocado[i] = false;
        }
    }
    return det;
}

// ===========================================================================
// VERIFICACIÓN DE ATENCIÓN
// La pulsera sortea K entre 1 y DESAFIO_PULSOS_MAX, emite K pulsos y espera
// que el conductor toque el pad K. La Pi solo envía el identificador de la
// verificación y la ventana de respuesta.
// ===========================================================================
typedef enum { DES_INACTIVO, DES_VIBRANDO, DES_ESPERANDO } estado_desafio_t;

// Límites de la ventana que llega en la orden. Fuera de ellos el valor se
// toma como un error de transmisión y se usa la ventana de defecto.
#define DES_VENTANA_MIN_MS   1000
#define DES_VENTANA_MAX_MS  60000

static estado_desafio_t des_estado = DES_INACTIVO;
static int     des_id_secuencia = -1;
static int     des_K = 0;
static uint32_t des_ventana_ms = DESAFIO_VENTANA_DEFECTO_MS;
static int     des_pulsos_dados = 0;
static int64_t des_ts_evento_ms = 0;
static int64_t des_ts_fin_vibrar_ms = 0;
static bool    des_motor_on = false;

// ventana_ms: tiempo que tiene el conductor para tocar el pad, contado desde
// el último pulso. Lo fija la Pi según el nivel de alerta y es el mismo valor
// con el que ella calcula su plazo. 0 = la orden no lo trae.
static void desafio_iniciar(int id_secuencia, uint32_t ventana_ms) {
    if (id_secuencia < 0) {
        // Sin identificador la respuesta no se podría asociar a ningún pedido
        ESP_LOGW(TAG, "[ack] orden de verificacion sin id_secuencia: se ignora");
        return;
    }
    if (ventana_ms == 0) {
        des_ventana_ms = DESAFIO_VENTANA_DEFECTO_MS;
    } else if (ventana_ms < DES_VENTANA_MIN_MS || ventana_ms > DES_VENTANA_MAX_MS) {
        ESP_LOGW(TAG, "[ack] ventana %lu ms fuera de rango: se usa la de defecto",
                 (unsigned long)ventana_ms);
        des_ventana_ms = DESAFIO_VENTANA_DEFECTO_MS;
    } else {
        des_ventana_ms = ventana_ms;
    }
    des_id_secuencia = id_secuencia;
    des_K = (int)(esp_random() % DESAFIO_PULSOS_MAX) + 1;   // 1..4
    des_pulsos_dados = 0;
    des_estado = DES_VIBRANDO;
    des_motor_on = false;
    des_ts_evento_ms = millis();
    leds_pads_encender();
    red_marcar_secuencia_iniciada();       // lo informa la próxima telemetría
    ESP_LOGI(TAG, "[ack] desafio id=%d -> vibra K=%d (toca pad %d), ventana %lu ms",
             id_secuencia, des_K, des_K, (unsigned long)des_ventana_ms);
}

static void desafio_cancelar(void) {
    if (des_estado != DES_INACTIVO) ESP_LOGI(TAG, "[ack] desafio cancelado");
    des_estado = DES_INACTIVO;
    des_id_secuencia = -1;
    leds_pads_apagar();
}

static void desafio_update(void) {
    switch (des_estado) {
    case DES_INACTIVO:
        return;

    case DES_VIBRANDO:
        if ((millis() - des_ts_evento_ms) < 0) return;
        if (!des_motor_on) {
            if (des_pulsos_dados >= des_K) {
                motor_apagar();
                des_estado = DES_ESPERANDO;
                des_ts_fin_vibrar_ms = millis();
                pads_leer_flanco();   // limpia flancos acumulados durante la vibración
                ESP_LOGI(TAG, "[ack] fin vibracion -> esperando toque (%lu ms)",
                         (unsigned long)des_ventana_ms);
                return;
            }
            motor_vibrar(100, 0);
            des_motor_on = true;
            des_ts_evento_ms = millis() + DESAFIO_PULSO_ON_MS;
            ESP_LOGI(TAG, "[ack] pulso %d/%d ON", des_pulsos_dados + 1, des_K);
        } else {
            motor_apagar();
            des_motor_on = false;
            des_pulsos_dados++;
            des_ts_evento_ms = millis() + DESAFIO_PULSO_OFF_MS;
            ESP_LOGI(TAG, "[ack] pulso %d/%d OFF", des_pulsos_dados, des_K);
        }
        return;

    case DES_ESPERANDO: {
        int pad = pads_leer_flanco();
        if (pad > 0) {
            long tr = (long)(millis() - des_ts_fin_vibrar_ms);
            bool ok = (pad == des_K);
            ESP_LOGI(TAG, "[ack] pad %d, K=%d -> %s (tr=%ldms)",
                     pad, des_K, ok ? "CORRECTO" : "incorrecto", tr);
            red_enviar_ack(des_id_secuencia, ok, tr);
            des_estado = DES_INACTIVO;
            leds_pads_apagar();
            return;
        }
        if ((millis() - des_ts_fin_vibrar_ms) >= (int64_t)des_ventana_ms) {
            // Ventana cerrada: no se envía nada. La Pi da la verificación por
            // vencida con su propio plazo, que es esta ventana más un margen.
            ESP_LOGI(TAG, "[ack] sin respuesta en %lu ms -> no envia",
                     (unsigned long)des_ventana_ms);
            des_estado = DES_INACTIVO;
            leds_pads_apagar();
        }
        return;
    }
    }
}

// ===========================================================================
// API PÚBLICA
// ===========================================================================
void motor_pads_init(void) {
    ledc_timer_config_t timer_cfg = {
        .speed_mode      = PWM_MODO,
        .duty_resolution = PWM_RESOLUCION,
        .timer_num       = PWM_TIMER,
        .freq_hz         = PWM_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_cfg));

    ledc_channel_config_t ch_cfg = {
        .gpio_num   = PIN_MOTOR,
        .speed_mode = PWM_MODO,
        .channel    = PWM_CANAL,
        .timer_sel  = PWM_TIMER,
        .duty       = 0,
        .hpoint     = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ch_cfg));

    ESP_ERROR_CHECK(touch_pad_init());
    ESP_ERROR_CHECK(touch_pad_set_fsm_mode(TOUCH_FSM_MODE_TIMER));
    ESP_ERROR_CHECK(touch_pad_set_voltage(TOUCH_HVOLT_2V7, TOUCH_LVOLT_0V5, TOUCH_HVOLT_ATTEN_1V));
    for (int i = 0; i < 4; i++) {
        ESP_ERROR_CHECK(touch_pad_config(PAD_TOUCH_NUM[i]));
    }
    ESP_ERROR_CHECK(touch_pad_fsm_start());

    // LEDS_PADS: arranca apagada; se enciende en desafio_iniciar().
    gpio_reset_pin(PIN_LEDS_PADS);
    gpio_config_t leds_pads_cfg = {
        .pin_bit_mask = (1ULL << PIN_LEDS_PADS),
        .mode         = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&leds_pads_cfg));
    leds_pads_apagar();
    ESP_LOGI(TAG, "[leds_pads] nivel tras apagar = %d (esperado: 0, activa en alto)",
             gpio_get_level(PIN_LEDS_PADS));

    ESP_LOGI(TAG, "motor GP%d (LEDC) + pads GP6/4/5/2 (touch) + LEDS_PADS GP%d listos",
             PIN_MOTOR, PIN_LEDS_PADS);
}

void motor_pads_update(void) {
    motor_update();
    desafio_update();
    if (des_estado == DES_INACTIVO) {
        // Reafirma el nivel de LEDS_PADS en cada vuelta, no solo al arrancar
        leds_pads_apagar();
        // Mantiene al día el estado de cada pad (y registra los toques). Un
        // dedo apoyado desde antes de la verificación no cuenta después como
        // toque nuevo.
        pads_leer_flanco();
    }
}

void motor_pads_ejecutar_comando(int tipo, int intensidad,
                                  uint32_t duracion_ms, int id_secuencia) {
    switch (tipo) {
    case TIPO_VIBRAR_LEVE:
    case TIPO_VIBRAR_MEDIO:
    case TIPO_VIBRAR_FUERTE:
        // Durante una verificación el motor es de ella: una vibración ajena
        // alteraría la cuenta de pulsos que el conductor tiene que hacer.
        if (des_estado == DES_INACTIVO) {
            motor_vibrar((uint8_t)intensidad, duracion_ms);
            ESP_LOGI(TAG, "[cmd] VIBRAR tipo=%d int=%d dur=%lu",
                     tipo, intensidad, (unsigned long)duracion_ms);
        } else {
            ESP_LOGI(TAG, "[cmd] VIBRAR tipo=%d ignorada: hay una verificacion en curso", tipo);
        }
        break;
    case TIPO_SECUENCIA_ACK:
        desafio_iniciar(id_secuencia, duracion_ms);   // duracion_ms = ventana de respuesta
        break;
    case TIPO_APAGAR_TODO:
        desafio_cancelar();
        motor_apagar();
        ESP_LOGI(TAG, "[cmd] APAGAR_TODO");
        break;
    default:
        break;
    }
}