#include <stdio.h>
#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_rom_sys.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "bpm_max30102.h"

static const char *TAG = "BPM_MAX30102";

// Verificado contra esquemático/PCB reales: GP8=SCL, GP9=INT (libre), GP10=SDA
#define I2C_MASTER_SDA_IO           10
#define I2C_MASTER_SCL_IO           8
#define I2C_MASTER_NUM              I2C_NUM_0
#define I2C_MASTER_FREQ_HZ          100000
#define I2C_MASTER_TX_BUF_DISABLE   0
#define I2C_MASTER_RX_BUF_DISABLE   0

#define MAX30102_ADDR               0x57
#define REG_INTR_STATUS_1           0x00
#define REG_INTR_STATUS_2           0x01
#define REG_INTR_ENABLE_1           0x02
#define REG_INTR_ENABLE_2           0x03
#define REG_FIFO_WR_PTR             0x04
#define REG_OVF_COUNTER             0x05
#define REG_FIFO_RD_PTR             0x06
#define REG_FIFO_DATA               0x07
#define REG_FIFO_CONFIG             0x08
#define REG_MODE_CONFIG             0x09
#define REG_SPO2_CONFIG             0x0A
#define REG_LED1_PA                 0x0C
#define REG_LED2_PA                 0x0D

#define FINGER_ON_UMBRAL            20000
#define FINGER_OFF_UMBRAL           15000

#define DC_ALPHA                    0.9372
#define DC_WARMUP_SAMPLES           150

#define MA_SIZE                     12

#define PEAK_DECAY                  0.980
#define THRESHOLD_RATIO             0.35
#define THRESHOLD_MIN_ABS           30.0
#define ARTIFACT_CEILING            3000.0

#define REFRACTORY_US               500000ULL
#define CONTACT_CONFIRM_TIMEOUT_US  (12000000ULL)

#define BPM_ALPHA_BOOT              0.60
#define BPM_ALPHA_STEADY            0.10
#define BEATS_FOR_STEADY            6

#define MIN_BPM                     40.0f
#define MAX_BPM                     120.0f

#define MEDIAN_BUF_SIZE             16

// ---------------------------------------------------------------------
// Estado compartido con el getter público. Volatile alcanza: son
// escrituras de una sola palabra desde un único task productor, y un
// único consumidor (red_update, en otro task) que solo lee.
// ---------------------------------------------------------------------
static volatile float s_bpm_med_val = 0.0f;
static volatile bool  s_contacto_confirmado = false;
// Control de energia (el sensor solo mide mientras la pulsera esta conectada al AP de la Pi):
//   s_activo_pedido : lo que pide el resto del firmware (arranca en false -> sensor apagado)
//   s_hw_activo     : estado REAL del chip. Solo lo escribe max30102_task, asi el init y el
//                     SHDN nunca se pisan con una lectura de FIFO en curso.
static volatile bool  s_activo_pedido = false;
static volatile bool  s_hw_activo     = false;
static TaskHandle_t   s_task          = NULL;

bool bpm_max30102_get(int *bpm_out) {
    if (!s_contacto_confirmado) return false;
    if (bpm_out) *bpm_out = (int)(s_bpm_med_val + 0.5f);
    return true;
}

static esp_err_t max30102_write_reg(uint8_t reg_addr, uint8_t data) {
    uint8_t buf[2] = {reg_addr, data};
    return i2c_master_write_to_device(I2C_MASTER_NUM, MAX30102_ADDR,
                                      buf, 2, pdMS_TO_TICKS(100));
}
static esp_err_t max30102_read_reg(uint8_t reg_addr, uint8_t *out) {
    return i2c_master_write_read_device(I2C_MASTER_NUM, MAX30102_ADDR,
                                        &reg_addr, 1, out, 1, pdMS_TO_TICKS(100));
}
static esp_err_t max30102_read_fifo(uint32_t *red, uint32_t *ir) {
    uint8_t reg = REG_FIFO_DATA;
    uint8_t data[6];
    esp_err_t err = i2c_master_write_read_device(I2C_MASTER_NUM, MAX30102_ADDR,
                                                  &reg, 1, data, 6, pdMS_TO_TICKS(100));
    if (err == ESP_OK) {
        *red = ((uint32_t)(data[0] & 0x03) << 16) | ((uint32_t)data[1] << 8) | data[2];
        *ir  = ((uint32_t)(data[3] & 0x03) << 16) | ((uint32_t)data[4] << 8) | data[5];
    }
    return err;
}
static int max30102_samples_available(void) {
    uint8_t wr = 0, rd = 0;
    if (max30102_read_reg(REG_FIFO_WR_PTR, &wr) != ESP_OK) return -1;
    if (max30102_read_reg(REG_FIFO_RD_PTR, &rd) != ESP_OK) return -1;
    return (int)((wr - rd + 32) % 32);
}

static float mediana(float *arr, int n) {
    float tmp[MEDIAN_BUF_SIZE];
    memcpy(tmp, arr, n * sizeof(float));
    for (int i = 1; i < n; i++) {
        float key = tmp[i]; int j = i - 1;
        while (j >= 0 && tmp[j] > key) { tmp[j+1] = tmp[j]; j--; }
        tmp[j+1] = key;
    }
    if (n % 2 == 0) return (tmp[n/2-1] + tmp[n/2]) / 2.0f;
    return tmp[n/2];
}

static void max30102_init(void) {
    ESP_LOGI(TAG, "Inicializando MAX30102 (SDA:GP%d, SCL:GP%d)...",
             I2C_MASTER_SDA_IO, I2C_MASTER_SCL_IO);
    max30102_write_reg(REG_MODE_CONFIG,  0x40);
    vTaskDelay(pdMS_TO_TICKS(200));
    max30102_write_reg(REG_FIFO_WR_PTR,  0x00);
    max30102_write_reg(REG_OVF_COUNTER,  0x00);
    max30102_write_reg(REG_FIFO_RD_PTR,  0x00);
    max30102_write_reg(REG_FIFO_CONFIG,  0x3F);
    max30102_write_reg(REG_MODE_CONFIG,  0x03);
    max30102_write_reg(REG_SPO2_CONFIG,  0x67);
    // Validado en hardware real: 0xFF (51mA) causa brownouts en la ESP32-S3
    // SuperMini. 0xBF (~38mA) es estable y suficiente para muñeca.
    max30102_write_reg(REG_LED1_PA,      0xBF);
    max30102_write_reg(REG_LED2_PA,      0xBF);
    max30102_write_reg(REG_INTR_ENABLE_1, 0x40);
    max30102_write_reg(REG_INTR_ENABLE_2, 0x00);
    uint8_t dummy;
    max30102_read_reg(REG_INTR_STATUS_1, &dummy);
    max30102_read_reg(REG_INTR_STATUS_2, &dummy);
    ESP_LOGI(TAG, "Sensor listo. SR=50Hz efectivos, ADC=16384nA, LED=~38mA");
}

// No toca el chip: deja el pedido y despierta a la task, que es la unica que habla con el sensor.
void bpm_max30102_set_activo(bool activo) {
    if (activo == s_activo_pedido) return;   // idempotente
    s_activo_pedido = activo;
    // ESP_LOGx exige formato literal (concatena strings en compilacion): nada de "cond ? a : b".
    if (activo) ESP_LOGI(TAG, "Conectada a la AP -> se activa el MAX30102");
    else        ESP_LOGI(TAG, "Sin AP -> se apaga el MAX30102 (SHDN)");
    if (s_task) xTaskNotifyGive(s_task);     // no espera el timeout de 500 ms
}

bool bpm_max30102_esta_activo(void) { return s_hw_activo; }

// Pone el chip en SHDN (apaga LEDs y front-end analogico). ESP_OK solo si el chip lo acepto.
static esp_err_t max30102_apagar(void) {
    max30102_write_reg(REG_LED1_PA, 0x00);   // red de seguridad si SHDN no se aplicara
    max30102_write_reg(REG_LED2_PA, 0x00);
    return max30102_write_reg(REG_MODE_CONFIG, 0x80);
}

static void max30102_task(void *pv) {
    uint32_t ir_raw = 0, red_raw = 0;
    double dc_ir = 0.0; int warmup = 0; bool dc_ok = false;
    double ma_buf[MA_SIZE]; int ma_idx = 0; double ma_sum = 0.0; int ma_count = 0;
    memset(ma_buf, 0, sizeof(ma_buf));
    double det_buf[3] = {0,0,0}; int64_t ts_buf[3] = {0,0,0};
    double peak_env = 0.0;
    int64_t last_beat_us = 0;
    float bpm_ema = 0.0f, bpm_med_val = 0.0f;
    float med_buf[MEDIAN_BUF_SIZE]; int med_idx = 0, med_count = 0, beat_count = 0;
    memset(med_buf, 0, sizeof(med_buf));
    bool finger_present = false, contacto_confirmado = false;
    int64_t ts_primer_contacto_us = 0;
    int log_tick = 0, i2c_err_streak = 0, lockup_count = 0;

    while (1) {
        // ------------- control de energia (solo esta task toca el chip) -------------
        bool pedido = s_activo_pedido;
        if (pedido && !s_hw_activo) {
            max30102_init();                      // soft-reset + config completa, FIFO limpio
            s_hw_activo = true;
            finger_present = false; contacto_confirmado = false; s_contacto_confirmado = false;
            i2c_err_streak = 0; lockup_count = 0;
            ESP_LOGI(TAG, "MAX30102 ACTIVO: esperando dedo...");
        } else if (!pedido && s_hw_activo) {
            if (max30102_apagar() == ESP_OK) {
                s_hw_activo = false;
                finger_present = false; contacto_confirmado = false; s_contacto_confirmado = false;
                ESP_LOGI(TAG, "MAX30102 en SHDN.");
            } else {
                ESP_LOGE(TAG, "No pude poner SHDN, reintento.");
                vTaskDelay(pdMS_TO_TICKS(200));
                continue;                         // no se da por apagado hasta que el chip responda
            }
        }
        if (!s_hw_activo) {                       // apagado: sin I2C ni CPU hasta nueva orden
            ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(500));
            continue;
        }
        // -----------------------------------------------------------------------------

        int n = max30102_samples_available();
        if (n < 0) {
            i2c_err_streak++;
            if (i2c_err_streak == 1 || i2c_err_streak % 10 == 0)
                ESP_LOGE(TAG, "Error I2C (racha:%d) — esperando...", i2c_err_streak);
            if (i2c_err_streak >= 15) {
                ESP_LOGE(TAG, "Posible latch-up por brownout. Forzando reinicio del MAX30102...");
                max30102_init();
                i2c_err_streak = 0;
            }
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }
        i2c_err_streak = 0;
        if (n == 0) { vTaskDelay(pdMS_TO_TICKS(5)); continue; }

        int64_t ts_now = esp_timer_get_time();
        if (max30102_read_fifo(&red_raw, &ir_raw) != ESP_OK) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        if (ir_raw == 262143) {
            lockup_count++;
            if (lockup_count >= 50) {
                ESP_LOGE(TAG, "Bus bloqueado. Reiniciando sensor...");
                max30102_init();
                lockup_count = 0;
                finger_present = false;
            }
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        lockup_count = 0;

        double fir = (double)ir_raw;

        if (!finger_present) {
            if (fir >= FINGER_ON_UMBRAL) {
                finger_present = true;
                contacto_confirmado = false;
                s_contacto_confirmado = false;
                ts_primer_contacto_us = ts_now;
                dc_ir = fir; warmup = 0; dc_ok = false;
                ma_idx = 0; ma_sum = 0.0; ma_count = 0;
                memset(ma_buf, 0, sizeof(ma_buf));
                det_buf[0]=det_buf[1]=det_buf[2]=0.0;
                ts_buf[0]=ts_buf[1]=ts_buf[2]=0;
                peak_env = 0.0; last_beat_us = 0;
                bpm_ema = 0.0f; bpm_med_val = 0.0f;
                beat_count = 0; med_idx = 0; med_count = 0; log_tick = 0;
                memset(med_buf, 0, sizeof(med_buf));
                ESP_LOGI(TAG, "CONTACTO DETECTADO — Calibrando (~3s)...");
            } else {
                vTaskDelay(pdMS_TO_TICKS(20));
                continue;
            }
        } else {
            if (fir < FINGER_OFF_UMBRAL) {
                finger_present = false;
                contacto_confirmado = false;
                s_contacto_confirmado = false;
                ESP_LOGW(TAG, "Retirado. EMA:%.1f MED:%.1f", bpm_ema, bpm_med_val);
                vTaskDelay(pdMS_TO_TICKS(20));
                continue;
            }
            if (!contacto_confirmado &&
                (ts_now - ts_primer_contacto_us) >= (int64_t)CONTACT_CONFIRM_TIMEOUT_US) {
                ESP_LOGW(TAG, "Sin pulso fisiologico tras %llus — probable superficie. Reiniciando.",
                         (unsigned long long)(CONTACT_CONFIRM_TIMEOUT_US / 1000000ULL));
                finger_present = false;
                vTaskDelay(pdMS_TO_TICKS(20));
                continue;
            }
        }

        dc_ir = dc_ir * DC_ALPHA + fir * (1.0 - DC_ALPHA);
        if (!dc_ok) {
            if (++warmup >= DC_WARMUP_SAMPLES) {
                dc_ok = true;
                ESP_LOGI(TAG, "DC OK. IR_DC=%.0f — detectando latidos...", dc_ir);
            }
            ts_buf[0]=ts_buf[1]; ts_buf[1]=ts_buf[2]; ts_buf[2]=ts_now;
            continue;
        }

        double ir_ac = fir - dc_ir;
        ma_sum -= ma_buf[ma_idx]; ma_buf[ma_idx] = ir_ac; ma_sum += ir_ac;
        ma_idx = (ma_idx + 1) % MA_SIZE;
        if (ma_count < MA_SIZE) ma_count++;
        double ir_smooth = ma_sum / (double)ma_count;

        double abs_s = (ir_smooth > 0.0) ? ir_smooth : -ir_smooth;
        if (abs_s > ARTIFACT_CEILING) peak_env *= 0.95;
        else if (abs_s > peak_env) peak_env = abs_s;
        else peak_env *= PEAK_DECAY;

        double threshold = peak_env * THRESHOLD_RATIO;
        if (threshold < THRESHOLD_MIN_ABS) threshold = THRESHOLD_MIN_ABS;

        det_buf[0]=det_buf[1]; det_buf[1]=det_buf[2]; det_buf[2]=ir_smooth;
        ts_buf[0]=ts_buf[1]; ts_buf[1]=ts_buf[2]; ts_buf[2]=ts_now;

        double y0=det_buf[0], y1=det_buf[1], y2=det_buf[2];
        bool es_pico = (y1 > y0) && (y1 > y2);
        bool supera_umbral = (y1 >= threshold);
        bool fuera_refract = (last_beat_us == 0) ||
                              ((ts_buf[1]-last_beat_us) >= (int64_t)REFRACTORY_US);

        if (es_pico && supera_umbral && fuera_refract) {
            int64_t beat_time_us;
            double denom = y0 - 2.0*y1 + y2;
            if (fabs(denom) > 1e-9) {
                double delta = 0.5 * (y0 - y2) / denom;
                double dt = (double)(ts_buf[2]-ts_buf[0]) / 2.0;
                beat_time_us = ts_buf[1] + (int64_t)(delta * dt);
            } else beat_time_us = ts_buf[1];

            if (last_beat_us > 0) {
                int64_t intervalo_us = beat_time_us - last_beat_us;
                float bpm_inst = 60000000.0f / (float)intervalo_us;
                if (bpm_inst >= MIN_BPM && bpm_inst <= MAX_BPM) {
                    beat_count++;
                    med_buf[med_idx % MEDIAN_BUF_SIZE] = bpm_inst;
                    med_idx++;
                    if (med_count < MEDIAN_BUF_SIZE) med_count++;
                    float alpha = (beat_count <= BEATS_FOR_STEADY) ? BPM_ALPHA_BOOT : BPM_ALPHA_STEADY;
                    bpm_ema = (bpm_ema == 0.0f) ? bpm_inst : bpm_ema*(1.0f-alpha)+bpm_inst*alpha;
                    bpm_med_val = mediana(med_buf, med_count);

                    if (beat_count >= BEATS_FOR_STEADY) {
                        contacto_confirmado = true;
                        s_bpm_med_val = bpm_med_val;
                        s_contacto_confirmado = true;
                        ESP_LOGI(TAG, "MED:%5.1f EMA:%5.1f inst:%5.1f beat:%d",
                                 bpm_med_val, bpm_ema, bpm_inst, beat_count);
                    } else {
                        ESP_LOGI(TAG, "Cal %d/%d inst:%.1f", beat_count, BEATS_FOR_STEADY, bpm_inst);
                    }
                }
            }
            last_beat_us = beat_time_us;
        }

        if (++log_tick >= 250) {
            log_tick = 0;
            ESP_LOGI(TAG, ">> IR:%lu DC:%.0f EMA:%.1f MED:%.1f beats:%d",
                     (unsigned long)ir_raw, dc_ir, bpm_ema, bpm_med_val, beat_count);
        }
    }
}

static void i2c_bus_recovery(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << I2C_MASTER_SDA_IO) | (1ULL << I2C_MASTER_SCL_IO),
        .mode = GPIO_MODE_INPUT_OUTPUT_OD,
        .pull_up_en = GPIO_PULLUP_ENABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(I2C_MASTER_SDA_IO, 1);
    gpio_set_level(I2C_MASTER_SCL_IO, 1);
    vTaskDelay(pdMS_TO_TICKS(5));
    if (gpio_get_level(I2C_MASTER_SDA_IO) == 1) {
        ESP_LOGI(TAG, "I2C bus recovery: SDA ya libre.");
        return;
    }
    ESP_LOGW(TAG, "I2C bus recovery: clockeando SCL...");
    for (int i = 0; i < 9; i++) {
        gpio_set_level(I2C_MASTER_SCL_IO, 0); esp_rom_delay_us(5);
        gpio_set_level(I2C_MASTER_SCL_IO, 1); esp_rom_delay_us(5);
        if (gpio_get_level(I2C_MASTER_SDA_IO) == 1) break;
    }
    gpio_set_level(I2C_MASTER_SDA_IO, 0); esp_rom_delay_us(5);
    gpio_set_level(I2C_MASTER_SCL_IO, 1); esp_rom_delay_us(5);
    gpio_set_level(I2C_MASTER_SDA_IO, 1); esp_rom_delay_us(5);
    ESP_LOGI(TAG, "I2C bus recovery: SDA final = %d", gpio_get_level(I2C_MASTER_SDA_IO));
}

void bpm_max30102_init(void) {
    i2c_bus_recovery();

    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_NUM, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_MASTER_NUM, conf.mode,
                                       I2C_MASTER_RX_BUF_DISABLE,
                                       I2C_MASTER_TX_BUF_DISABLE, 0));
    // Arranca APAGADO: no se configura para medir. Despierta cuando haya AP (set_activo(true)).
    if (max30102_apagar() != ESP_OK)
        ESP_LOGW(TAG, "MAX30102 no respondio al SHDN inicial (se reintenta desde la task)");
    s_hw_activo = false;
    xTaskCreate(max30102_task, "max30102_task", 8192, NULL, 5, &s_task);
}