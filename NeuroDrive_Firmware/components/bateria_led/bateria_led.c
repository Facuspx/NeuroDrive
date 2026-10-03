#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/rmt_tx.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_log.h"
#include "bateria_led.h"
#include "red.h"     // estado de conectividad, para el color del indicador
#include "reloj.h"   // millis()

static const char *TAG = "BATERIA_LED";

// -----------------------------------------------------------------------
// Pines: batería en ADC GP1 (divisor 100K/100K + 0.1uF)
// LED batería = GP44 | LED conectividad = GP43  [WS2812/SK6812]
// GP43 y GP44 son, por defecto, TX y RX de UART0. Al configurar los
// indicadores esos pines dejan de ser el puerto serie: la consola del
// firmware se lee por USB (Serial/JTAG).
// -----------------------------------------------------------------------
#define PIN_BATERIA_ADC     1
#define PIN_LED_BATERIA     44
#define PIN_LED_CONEC       43

#define BAT_DIVISOR         2.0f
#define BAT_MUESTRAS        16
#define BAT_PERIODO_MS      5000

// -----------------------------------------------------------------------
// WS2812/SK6812 sobre RMT — timing estándar (resolución 10MHz = 100ns/tick)
//   bit 0: 300ns alto, 900ns bajo
//   bit 1: 600ns alto, 600ns bajo
// El reset (>80us bajo) queda satisfecho solo: entre una actualización y
// la siguiente pasan milisegundos, mucho más que el mínimo requerido.
// -----------------------------------------------------------------------
static rmt_channel_handle_t chan_conec = NULL, chan_bateria = NULL;
static rmt_encoder_handle_t enc_bytes  = NULL;

static rmt_channel_handle_t crear_canal_ws2812(int pin) {
    rmt_channel_handle_t chan = NULL;
    rmt_tx_channel_config_t cfg = {
        .clk_src           = RMT_CLK_SRC_DEFAULT,
        .gpio_num          = pin,
        .mem_block_symbols = 64,
        .resolution_hz     = 10 * 1000 * 1000,
        .trans_queue_depth = 4,
    };
    ESP_ERROR_CHECK(rmt_new_tx_channel(&cfg, &chan));
    ESP_ERROR_CHECK(rmt_enable(chan));
    return chan;
}

static void ws2812_set(rmt_channel_handle_t chan, uint8_t r, uint8_t g, uint8_t b) {
    if (!chan || !enc_bytes) return;

    // static: el transmisor lee este buffer mientras envía; no puede estar en
    // la pila de una función que podría haber retornado. Orden GRB.
    static uint8_t pixel[3];
    pixel[0] = g;
    pixel[1] = r;
    pixel[2] = b;

    rmt_transmit_config_t tx_cfg = { .loop_count = 0 };
    rmt_transmit(chan, enc_bytes, pixel, sizeof(pixel), &tx_cfg);
    rmt_tx_wait_all_done(chan, pdMS_TO_TICKS(20));
}

// -----------------------------------------------------------------------
// ADC batería
// -----------------------------------------------------------------------
static adc_oneshot_unit_handle_t adc_handle = NULL;
static adc_cali_handle_t adc_cali_handle = NULL;
static bool adc_calibrado = false;
static float bateria_voltaje = 0.0f;
static int   bateria_pct     = 100;
static int64_t bat_ts_prox   = 0;

static float bateria_leer_voltaje(void) {
    int64_t suma_mv = 0;
    int ultimo_raw = 0;
    for (int i = 0; i < BAT_MUESTRAS; i++) {
        int raw = 0;
        adc_oneshot_read(adc_handle, ADC_CHANNEL_0, &raw);   // GP1 = ADC1_CH0
        ultimo_raw = raw;
        int mv;
        if (adc_calibrado) {
            adc_cali_raw_to_voltage(adc_cali_handle, raw, &mv);   // curva real de fábrica, no lineal
        } else {
            // Fallback si la calibración no está disponible en este chip/eFuse
            mv = (raw * 3300) / 4095;
        }
        suma_mv += mv;
    }
    float v_pin = (suma_mv / (float)BAT_MUESTRAS) / 1000.0f;
    float v_bat = v_pin * BAT_DIVISOR;
    ESP_LOGI(TAG, "[bateria] raw=%d v_pin=%.3fV v_bat=%.3fV (calibrado=%d)",
             ultimo_raw, v_pin, v_bat, (int)adc_calibrado);
    return v_bat;
}

// -----------------------------------------------------------------------
// Curva de descarga Li-ion (403030, 400mAh, 3.7V nominal / 4.2V carga)
// Puntos tomados de la curva típica de descarga Li-ion, comprimidos al
// rango de corte seguro que ya usaba el proyecto (3.30V-4.20V) en vez del
// rango completo del gráfico (hasta 2.0V), para no sobredescargar la
// celda. Interpolación lineal por tramos entre puntos — mucho más fiel
// que una sola recta, porque la curva real es plana en el medio y cae
// fuerte en las puntas.
// -----------------------------------------------------------------------
typedef struct { float v; int pct; } curva_punto_t;
static const curva_punto_t CURVA_BATERIA[] = {
    { 4.20f, 100 },
    { 4.10f,  95 },
    { 4.02f,  90 },
    { 3.95f,  85 },
    { 3.90f,  80 },
    { 3.83f,  70 },
    { 3.79f,  60 },
    { 3.77f,  50 },
    { 3.75f,  40 },
    { 3.73f,  30 },
    { 3.71f,  20 },
    { 3.69f,  15 },
    { 3.63f,  10 },
    { 3.50f,   5 },
    { 3.30f,   0 },
};
#define N_CURVA_BATERIA (sizeof(CURVA_BATERIA) / sizeof(CURVA_BATERIA[0]))

static int bateria_v_a_pct(float v) {
    if (v >= CURVA_BATERIA[0].v) return 100;
    if (v <= CURVA_BATERIA[N_CURVA_BATERIA - 1].v) return 0;

    for (int i = 0; i < N_CURVA_BATERIA - 1; i++) {
        float v_alto = CURVA_BATERIA[i].v;
        float v_bajo = CURVA_BATERIA[i + 1].v;
        if (v <= v_alto && v >= v_bajo) {
            int pct_alto = CURVA_BATERIA[i].pct;
            int pct_bajo = CURVA_BATERIA[i + 1].pct;
            float frac = (v - v_bajo) / (v_alto - v_bajo);
            return pct_bajo + (int)(frac * (pct_alto - pct_bajo) + 0.5f);
        }
    }
    return 0;   // no debería llegar acá
}

// -----------------------------------------------------------------------
// Colores de los indicadores
// -----------------------------------------------------------------------
static void led_bateria_mostrar(int pct) {
    uint8_t r = 0, g = 0, b = 0;
    if (pct > 60) {
        g = 30;
    } else if (pct > 30) {
        r = 35; g = 25;
    } else if (pct >= 10) {
        r = 40;
    } else {
        bool on = (millis() / 400) % 2;
        r = on ? 45 : 0;
    }
    ws2812_set(chan_bateria, r, g, b);
}

static void led_conec_mostrar(void) {
    red_estado_conec_t conec = red_obtener_estado_conec();
    if (conec == RED_OPERANDO_PI) {
        ws2812_set(chan_conec, 0, 30, 0);          // verde
    } else if (conec == RED_MODO_TESTIGO) {
        ws2812_set(chan_conec, 25, 0, 35);         // violeta
    } else {
        ws2812_set(chan_conec, 25, 12, 0);         // ámbar (conectando/camión)
    }
}

// -----------------------------------------------------------------------
// API pública
// -----------------------------------------------------------------------
void bateria_led_init(void) {
    rmt_bytes_encoder_config_t enc_cfg = {
        .bit0 = { .duration0 = 3, .level0 = 1, .duration1 = 9, .level1 = 0 },
        .bit1 = { .duration0 = 6, .level0 = 1, .duration1 = 6, .level1 = 0 },
        .flags.msb_first = 1,
    };
    ESP_ERROR_CHECK(rmt_new_bytes_encoder(&enc_cfg, &enc_bytes));

    chan_conec   = crear_canal_ws2812(PIN_LED_CONEC);
    chan_bateria = crear_canal_ws2812(PIN_LED_BATERIA);

    adc_oneshot_unit_init_cfg_t adc_cfg = { .unit_id = ADC_UNIT_1 };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&adc_cfg, &adc_handle));
    adc_oneshot_chan_cfg_t chan_cfg = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten    = ADC_ATTEN_DB_12,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, ADC_CHANNEL_0, &chan_cfg));

    // Calibración real (curva de fábrica, no la formula lineal aproximada)
    adc_cali_curve_fitting_config_t cali_cfg = {
        .unit_id  = ADC_UNIT_1,
        .chan     = ADC_CHANNEL_0,
        .atten    = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    esp_err_t cal_err = adc_cali_create_scheme_curve_fitting(&cali_cfg, &adc_cali_handle);
    if (cal_err == ESP_OK) {
        adc_calibrado = true;
        ESP_LOGI(TAG, "[bateria] calibracion ADC (curve fitting) OK");
    } else {
        adc_calibrado = false;
        ESP_LOGW(TAG, "[bateria] calibracion ADC no disponible (err=%s) — usando formula lineal aproximada",
                 esp_err_to_name(cal_err));
    }

    // Apagados hasta la primera lectura/estado real
    ws2812_set(chan_conec, 0, 0, 0);
    ws2812_set(chan_bateria, 0, 0, 0);

    ESP_LOGI(TAG, "Bateria+LEDs listos (ADC GP%d, LED conec GP%d, LED bateria GP%d)",
             PIN_BATERIA_ADC, PIN_LED_CONEC, PIN_LED_BATERIA);
}

// Próximo refresco de los indicadores
static int64_t led_ts_prox = 0;

void bateria_led_update(void) {
    if ((millis() - bat_ts_prox) >= 0) {
        bat_ts_prox = millis() + BAT_PERIODO_MS;
        bateria_voltaje = bateria_leer_voltaje();
        bateria_pct = bateria_v_a_pct(bateria_voltaje);
    }

    // Los indicadores se refrescan cada 50 ms, no en cada vuelta del ciclo
    if ((millis() - led_ts_prox) >= 0) {
        led_ts_prox = millis() + 50;
        led_bateria_mostrar(bateria_pct);
        led_conec_mostrar();
    }
}

int bateria_led_get_pct(void) { return bateria_pct; }
