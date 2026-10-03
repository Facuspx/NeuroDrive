#pragma once
#include "banco.h"
typedef int touch_pad_t;
enum { TOUCH_PAD_NUM2 = 2, TOUCH_PAD_NUM4 = 4, TOUCH_PAD_NUM5 = 5, TOUCH_PAD_NUM6 = 6 };
#define TOUCH_FSM_MODE_TIMER 0
#define TOUCH_HVOLT_2V7 0
#define TOUCH_LVOLT_0V5 0
#define TOUCH_HVOLT_ATTEN_1V 0
esp_err_t touch_pad_init(void);
esp_err_t touch_pad_set_fsm_mode(int m);
esp_err_t touch_pad_set_voltage(int a, int b, int c);
esp_err_t touch_pad_config(touch_pad_t p);
esp_err_t touch_pad_fsm_start(void);
esp_err_t touch_pad_read_raw_data(touch_pad_t p, uint32_t *raw);
