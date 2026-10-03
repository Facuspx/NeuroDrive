#pragma once
#include "FreeRTOS.h"
BaseType_t xTaskCreate(void (*fn)(void *), const char *nombre, uint32_t pila,
                       void *arg, UBaseType_t prio, TaskHandle_t *h);
void vTaskDelete(TaskHandle_t h);
void vTaskDelay(uint32_t ticks);
