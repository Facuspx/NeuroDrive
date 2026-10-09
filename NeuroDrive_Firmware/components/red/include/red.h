#pragma once
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Modulo de red de la pulsera: Wi-Fi, maquina de conectividad, recepcion de
// ordenes, telemetria, respuesta a la verificacion y aviso remoto.
//
// "red" no conoce a los demas modulos. Recibe las ordenes y se las entrega a
// quien se haya registrado (motor_pads), y pide el pulso y la bateria a
// traves de funciones que main.c conecta. Asi la dependencia va en un solo
// sentido: motor_pads y bateria_led dependen de red, y no al reves.
// ---------------------------------------------------------------------------

// Orden entrante ya decodificada: (tipo, intensidad, duracion_ms, id_secuencia).
// id_secuencia vale -1 si la orden no lo trae.
typedef void (*red_comando_cb_t)(int tipo, int intensidad,
                                  uint32_t duracion_ms, int id_secuencia);
void red_set_comando_handler(red_comando_cb_t cb);

// Datos que van en la telemetria
typedef bool (*red_bpm_provider_t)(int *bpm_out);      // true si hay pulso valido
typedef int  (*red_bateria_provider_t)(void);           // carga de la bateria, 0-100
void red_set_bpm_provider(red_bpm_provider_t fn);
void red_set_bateria_provider(red_bateria_provider_t fn);

// Ciclo de vida
void red_init(void);      // inicia el Wi-Fi y la maquina de conectividad
void red_update(void);    // llamar en cada vuelta del ciclo principal; no bloquea

// true mientras la pulsera esta asociada a la red de la Pi
bool red_operando_con_pi(void);

// true cuando la conectividad esta estable y no queda nada por enviar: operando
// con la Pi, o en modo testigo entre dos revisiones. Mientras sea false (se
// esta asociando, reintentando o avisando) el ciclo principal no deja que el
// microcontrolador entre en reposo.
bool red_en_reposo(void);

// Espera hasta `ms` milisegundos, y vuelve antes si llega una orden. Es la
// pausa del ciclo principal cuando todo esta en reposo: el microcontrolador
// duerme mientras tanto y una orden lo despierta en el acto. No lee la
// orden; eso lo hace red_update() en la vuelta siguiente.
void red_esperar_orden(uint32_t ms);

// Respuesta del conductor a una verificacion. La transmite ACK_REENVIOS veces
// con el mismo id_secuencia: la primera copia sale en el acto y las demas
// desde red_update(), sin detener el ciclo principal.
void red_enviar_ack(int id_secuencia, bool correcta, long tiempo_respuesta_ms);

// motor_pads avisa que inicio una verificacion, para que la proxima
// telemetria lo informe (campos ack_recibido y secuencia_replicada).
void red_marcar_secuencia_iniciada(void);

// Estado de la maquina de conectividad (lo usa bateria_led para el indicador)
typedef enum {
    RED_CONECTANDO_PI,      // intentando asociarse a la red de la Pi
    RED_OPERANDO_PI,        // operacion normal
    RED_CONMUTAR_CAMION,    // perdio a la Pi: pasa a la red de respaldo y avisa
    RED_MODO_TESTIGO,       // en la red de respaldo; revisa si la Pi volvio
} red_estado_conec_t;
red_estado_conec_t red_obtener_estado_conec(void);
