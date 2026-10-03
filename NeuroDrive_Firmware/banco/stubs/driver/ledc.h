#pragma once
#include "banco.h"
#define LEDC_TIMER_8_BIT 8
#define LEDC_TIMER_0 0
#define LEDC_CHANNEL_0 0
#define LEDC_LOW_SPEED_MODE 0
#define LEDC_AUTO_CLK 0
typedef struct { int speed_mode, duty_resolution, timer_num, freq_hz, clk_cfg; } ledc_timer_config_t;
typedef struct { int gpio_num, speed_mode, channel, timer_sel, duty, hpoint; } ledc_channel_config_t;
esp_err_t ledc_timer_config(const ledc_timer_config_t *c);
esp_err_t ledc_channel_config(const ledc_channel_config_t *c);
esp_err_t ledc_set_duty(int modo, int canal, uint32_t duty);
esp_err_t ledc_update_duty(int modo, int canal);
