#pragma once
// ---------------------------------------------------------------------------
// Protocolo entre la unidad de procesamiento (Raspberry Pi) y la pulsera.
// JSON sobre UDP. Es el mismo formato que define, del lado de la Pi,
// NeuroDrive_Wearable/protocolo.py: si se cambia un valor aca hay que
// cambiarlo alla, y viceversa.
// ---------------------------------------------------------------------------

#define PROTOCOLO_VERSION        1

// Puertos UDP
#define PUERTO_COMANDOS          5006   // la pulsera ESCUCHA ordenes aca
#define PUERTO_TELEMETRIA        5005   // la Pi escucha aca telemetria y respuestas

// Tipos de orden (campo "tipo")
#define TIPO_VIBRAR_LEVE         1
#define TIPO_VIBRAR_MEDIO        2
#define TIPO_VIBRAR_FUERTE       3
#define TIPO_SECUENCIA_ACK       8      // verificacion de atencion. El campo
                                        // "duracion_ms" trae la VENTANA DE
                                        // RESPUESTA (0 = usar la de defecto)
#define TIPO_APAGAR_TODO         10     // detiene el motor y cancela la verificacion
#define TIPO_APAGADO_LIMPIO      99     // la Pi se detiene de manera ordenada: la
                                        // proxima perdida del enlace es esperada

// Verificacion de atencion (los mismos valores figuran en protocolo.py)
#define DESAFIO_PULSOS_MAX       4      // K se sortea entre 1 y 4
#define DESAFIO_PULSO_ON_MS      200
#define DESAFIO_PULSO_OFF_MS     300
#define DESAFIO_VENTANA_DEFECTO_MS  10000   // si la orden trae duracion_ms = 0

// Repeticion de la respuesta del conductor (mismo id_secuencia en cada copia)
#define ACK_REENVIOS             3
#define ACK_ESPACIADO_MS         50

// Identificadores de envio que se recuerdan para descartar ordenes repetidas
#define DEDUP_N                  16
