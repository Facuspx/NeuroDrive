#pragma once
#include <stdint.h>
typedef void *TaskHandle_t;
typedef int BaseType_t;
typedef unsigned int UBaseType_t;
#define pdPASS 1
#define pdTRUE 1
#define tskIDLE_PRIORITY 0
#define pdMS_TO_TICKS(ms) ((ms) / 10)
