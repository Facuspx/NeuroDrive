// ===========================================================================
// NeuroDrive - Pulsera: modulo de red
//
//   - Wi-Fi en modo estacion, con direccion fija en la red de la Pi
//   - Maquina de conectividad de cuatro estados (ver conectividad_update)
//   - Recepcion de ordenes por UDP, con descarte de repetidas
//   - Telemetria periodica y respuesta a la verificacion de atencion
//   - Aviso remoto cuando la Pi deja de responder sin haberlo anunciado
//
// Nada de este modulo detiene el ciclo principal. Las esperas (asociacion,
// reintentos, revision periodica) se resuelven con marcas de tiempo, y el
// envio del aviso remoto, que es una conexion HTTPS de varios segundos,
// corre en una tarea propia.
// ===========================================================================
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_http_client.h"
#include "esp_crt_bundle.h"
#include "cJSON.h"

#include "red.h"
#include "protocolo_nd.h"
#include "reloj.h"

// Claves de las redes y del servicio de mensajeria. El archivo no se
// versiona: ver credenciales.ejemplo.h.
#if __has_include("credenciales.h")
#include "credenciales.h"
#else
#error "Falta components/red/credenciales.h: copiar credenciales.ejemplo.h con ese nombre y completar los valores."
#endif

static const char *TAG = "RED";

// ===========================================================================
// PARAMETROS
// ===========================================================================
#define HOSTNAME_PULSERA       "neurodrive-pulsera"
#define PI_IP_RESPALDO         "192.168.4.1"   // destino hasta recibir la primera orden

#define PERIODO_TELEMETRIA_MS  2000

// Maquina de conectividad
#define INTENTO_TIMEOUT_MS     10000   // espera maxima por una asociacion a la Pi
#define ESPERA_REINTENTO_MS     3000   // pausa entre dos intentos
#define UMBRAL_PERDIDA_PI_MS   60000   // sin la Pi durante este tiempo: conmutar
#define REVISAR_PI_CADA_MS     20000   // en modo testigo, cada cuanto se la busca
#define CAMION_TIMEOUT_MS      15000   // espera maxima por la red de respaldo

// Aviso remoto
#define AVISO_PILA_BYTES        8192   // la negociacion TLS necesita pila propia
#define AVISO_HTTP_TIMEOUT_MS  10000
#define AVISO_ESPERA_MAX_MS    45000   // tope de espera de la maquina por la tarea

// Como mucho, cuantos datagramas se atienden por vuelta del ciclo principal.
// Las ordenes criticas llegan de a tres copias; con cuatro se despacha una
// rafaga entera sin que una inundacion de paquetes retenga al ciclo.
#define MAX_DATAGRAMAS_POR_VUELTA  4

// ===========================================================================
// ESTADO
// ===========================================================================
static red_comando_cb_t       s_comando_cb       = NULL;
static red_bpm_provider_t     s_bpm_provider     = NULL;
static red_bateria_provider_t s_bateria_provider = NULL;

// --- Maquina de conectividad ---
// Los cuatro estados son los de red_estado_conec_t. Dentro de cada uno, la
// "fase" indica que se esta esperando; es lo que reemplaza a las pausas.
typedef enum {
    FASE_ASOCIANDO_PI,       // se pidio la asociacion a la Pi; se espera el resultado
    FASE_ESPERA_REINTENTO,   // pausa entre dos intentos de asociacion a la Pi
    FASE_ASOCIANDO_CAMION,   // se pidio la asociacion a la red de respaldo
    FASE_AVISANDO,           // la tarea del aviso remoto esta trabajando
    FASE_REPOSO,             // modo testigo: se espera la proxima revision
} fase_t;

static red_estado_conec_t conec = RED_CONECTANDO_PI;
static fase_t  fase = FASE_ASOCIANDO_PI;
static int64_t ts_fase_ms = 0;            // instante en que comenzo la fase actual
static int64_t ts_desde_sin_pi_ms = 0;    // desde cuando no hay enlace con la Pi (0 = hay)

// true cuando la Pi anuncio su detencion (tipo 99): la perdida del enlace que
// sigue es esperada y no se avisa.
static bool apagado_limpio_recibido = false;
static bool aviso_ya_enviado = false;

// Lo escribe el manejador de eventos (otra tarea) y lo lee el ciclo principal
static volatile bool wifi_conectado = false;

// --- Aviso remoto (tarea propia) ---
typedef enum { AVISO_INACTIVO, AVISO_EN_CURSO, AVISO_OK, AVISO_FALLO } aviso_estado_t;
static volatile aviso_estado_t s_aviso = AVISO_INACTIVO;

// --- Sockets ---
static int sock_comandos = -1;     // recibe ordenes en PUERTO_COMANDOS
static int sock_telemetria = -1;   // envia telemetria y respuestas
static struct sockaddr_in pi_addr;
static bool pi_ip_conocida = false;

// --- Descarte de ordenes repetidas ---
static long    dedup_ids[DEDUP_N];
static uint8_t dedup_idx = 0;

// --- Telemetria ---
static int64_t tele_ts_prox_ms = 0;
static long    tele_id_paquete = 0;
static bool    flag_ack_recibido = false;
static bool    flag_secuencia_replicada = false;

// --- Respuesta a la verificacion: copias pendientes de envio ---
static char    ack_buf[128];
static int     ack_len = 0;
static int     ack_copias_pendientes = 0;
static int64_t ack_ts_prox_ms = 0;

static esp_netif_t *s_netif = NULL;

// ===========================================================================
// DESCARTE DE REPETIDAS
// Las ordenes criticas viajan tres veces con el mismo id_paquete; se ejecuta
// la primera copia. Se recuerdan los ultimos DEDUP_N identificadores.
// ===========================================================================
static void dedup_init(void) {
    for (int i = 0; i < DEDUP_N; i++) dedup_ids[i] = -1;
}
static bool dedup_ya_visto(long id) {
    for (int i = 0; i < DEDUP_N; i++) if (dedup_ids[i] == id) return true;
    return false;
}
static void dedup_marcar(long id) {
    dedup_ids[dedup_idx] = id;
    dedup_idx = (dedup_idx + 1) % DEDUP_N;
}

// ===========================================================================
// WI-FI
// ===========================================================================
static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_conectado = false;
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        wifi_conectado = true;
    }
}

static void udp_comandos_cerrar(void) {
    if (sock_comandos >= 0) { close(sock_comandos); sock_comandos = -1; }
}

static void wifi_stop(void) {
    udp_comandos_cerrar();
    esp_wifi_disconnect();
    esp_wifi_stop();
    // No se espera al evento de desconexion: desde aca el enlace anterior ya
    // no cuenta. Sin esta linea, la vuelta siguiente del ciclo podria leer un
    // "conectado" que corresponde a la red que se acaba de abandonar.
    wifi_conectado = false;
}

// Pide la asociacion a una red y vuelve en el acto. El resultado se conoce
// despues, por wifi_conectado.
//   usar_ip_fija = true  -> 192.168.4.20/24 (red de la Pi; sin asignacion automatica)
//   usar_ip_fija = false -> asignacion automatica (red de respaldo)
static void wifi_start_sta_ip_fija(const char *ssid, const char *pass,
                                    bool usar_ip_fija) {
    wifi_stop();

    if (usar_ip_fija) {
        esp_netif_dhcpc_stop(s_netif);          // desactivar la asignacion automatica
        esp_netif_ip_info_t ip_info = {0};
        esp_netif_set_ip4_addr(&ip_info.ip,      192, 168, 4, 20);
        esp_netif_set_ip4_addr(&ip_info.gw,      192, 168, 4, 1);
        esp_netif_set_ip4_addr(&ip_info.netmask, 255, 255, 255, 0);
        esp_netif_set_ip_info(s_netif, &ip_info);
    } else {
        esp_netif_dhcpc_start(s_netif);         // asignacion automatica para la otra red
    }

    esp_netif_set_hostname(s_netif, HOSTNAME_PULSERA);

    wifi_config_t wc = {0};
    strncpy((char *)wc.sta.ssid,     ssid, sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, pass, sizeof(wc.sta.password) - 1);
    esp_wifi_set_mode(WIFI_MODE_STA);           // modo estacion: cliente de un punto de acceso
    esp_wifi_set_config(WIFI_IF_STA, &wc);

    esp_err_t err = esp_wifi_start();
    if (err == ESP_OK) err = esp_wifi_connect();
    if (err != ESP_OK) {
        // No se aborta: la maquina de conectividad reintenta al vencer el plazo
        ESP_LOGW(TAG, "[wifi] no se pudo iniciar la asociacion a \"%s\": %s",
                 ssid, esp_err_to_name(err));
    }
}

bool red_operando_con_pi(void) {
    return conec == RED_OPERANDO_PI && wifi_conectado;
}

red_estado_conec_t red_obtener_estado_conec(void) { return conec; }

// ===========================================================================
// AVISO REMOTO
// Se envia cuando la Pi deja de responder sin haber anunciado su detencion.
// Es una peticion HTTPS: tarda segundos y necesita mucha pila, asi que corre
// en una tarea propia. La maquina de conectividad solo consulta s_aviso.
// ===========================================================================

// Codifica `src` para el cuerpo application/x-www-form-urlencoded. Deja pasar
// letras, digitos y "-_.~"; el resto va como %XX. Devuelve false si no entra.
static bool url_codificar(const char *src, char *dst, size_t tam_dst) {
    static const char HEX[] = "0123456789ABCDEF";
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        bool literal = (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                       (*p >= '0' && *p <= '9') ||
                       *p == '-' || *p == '_' || *p == '.' || *p == '~';
        size_t necesita = literal ? 1 : 3;
        if (o + necesita + 1 > tam_dst) return false;
        if (literal) {
            dst[o++] = (char)*p;
        } else {
            dst[o++] = '%';
            dst[o++] = HEX[*p >> 4];
            dst[o++] = HEX[*p & 0x0F];
        }
    }
    dst[o] = '\0';
    return true;
}

static bool enviar_telegram(const char *texto) {
    if (!wifi_conectado) return false;

    char url[160];
    snprintf(url, sizeof(url), "https://api.telegram.org/bot%s/sendMessage", TG_TOKEN);

    char texto_cod[400];
    if (!url_codificar(texto, texto_cod, sizeof(texto_cod))) {
        ESP_LOGW(TAG, "[tg] el texto del aviso no entra en el buffer");
        return false;
    }
    char body[480];
    snprintf(body, sizeof(body), "chat_id=%s&text=%s", TG_CHAT_ID, texto_cod);

    esp_http_client_config_t cfg = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = AVISO_HTTP_TIMEOUT_MS,
        // Verifica la cadena de certificados contra las autoridades que trae
        // ESP-IDF, y que el certificado corresponda al nombre del servidor.
        .crt_bundle_attach = esp_crt_bundle_attach,
    };
    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    if (!client) { ESP_LOGW(TAG, "[tg] no se pudo iniciar el cliente HTTPS"); return false; }

    esp_http_client_set_header(client, "Content-Type", "application/x-www-form-urlencoded");
    esp_http_client_set_post_field(client, body, strlen(body));

    esp_err_t err = esp_http_client_perform(client);
    int code = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);

    if (err == ESP_OK && code == 200) { ESP_LOGI(TAG, "[tg] aviso ENVIADO ok"); return true; }
    ESP_LOGW(TAG, "[tg] fallo (err=%s, code=%d)", esp_err_to_name(err), code);
    return false;
}

static bool avisar_perdida(void) {
    int bateria_pct = s_bateria_provider ? s_bateria_provider() : -1;
    char txt[160];
    snprintf(txt, sizeof(txt),
             "NeuroDrive: perdida de conexion con la unidad.\n"
             "Uptime dispositivo: %llds\n"
             "Ultima bateria conocida: %d%%",
             (long long)(millis() / 1000), bateria_pct);
    ESP_LOGW(TAG, "[con] perdida NO anunciada -> enviando aviso remoto");
    return enviar_telegram(txt);
}

static void tarea_aviso(void *pv) {
    s_aviso = avisar_perdida() ? AVISO_OK : AVISO_FALLO;
    vTaskDelete(NULL);
}

// Lanza la tarea del aviso si corresponde enviarlo. Devuelve true si quedo
// una tarea trabajando (y entonces hay que esperar su resultado).
static bool aviso_lanzar_si_corresponde(void) {
    if (apagado_limpio_recibido) {
        ESP_LOGI(TAG, "[con] perdida ANUNCIADA (apagado ordenado) -> no se avisa");
        return false;
    }
    if (aviso_ya_enviado) return false;
    if (s_aviso == AVISO_EN_CURSO) return true;      // quedo una tarea de antes
    if (!wifi_conectado) {
        ESP_LOGW(TAG, "[tg] sin red de respaldo: el aviso se reintenta en modo testigo");
        return false;
    }
    s_aviso = AVISO_EN_CURSO;
    if (xTaskCreate(tarea_aviso, "aviso_remoto", AVISO_PILA_BYTES, NULL,
                    tskIDLE_PRIORITY + 1, NULL) != pdPASS) {
        s_aviso = AVISO_INACTIVO;
        ESP_LOGE(TAG, "[tg] no se pudo crear la tarea del aviso");
        return false;
    }
    return true;
}

// Toma el resultado de la tarea del aviso, si ya termino.
static void aviso_recoger_resultado(void) {
    if (s_aviso == AVISO_OK) {
        aviso_ya_enviado = true;       // solo queda en true si realmente se envio
        s_aviso = AVISO_INACTIVO;
    } else if (s_aviso == AVISO_FALLO) {
        s_aviso = AVISO_INACTIVO;      // se reintenta en la proxima revision
    }
}

// ===========================================================================
// RECEPCION DE ORDENES
// ===========================================================================
static void udp_comandos_abrir(void) {
    if (sock_comandos >= 0) return;
    sock_comandos = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock_comandos < 0) { ESP_LOGE(TAG, "no se pudo crear el socket de ordenes"); return; }

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port   = htons(PUERTO_COMANDOS),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    if (bind(sock_comandos, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        ESP_LOGE(TAG, "no se pudo asociar el puerto de ordenes");
        close(sock_comandos);
        sock_comandos = -1;
    }
}

// Atiende un datagrama, si hay. Devuelve false cuando no queda ninguno.
static bool udp_recibir(void) {
    if (sock_comandos < 0) return false;
    uint8_t buf[256];
    struct sockaddr_in from;
    socklen_t fromlen = sizeof(from);

    // MSG_DONTWAIT: si no hay nada, retorna de inmediato
    int n = recvfrom(sock_comandos, buf, sizeof(buf) - 1, MSG_DONTWAIT,
                      (struct sockaddr *)&from, &fromlen);
    if (n <= 0) return false;
    buf[n] = 0;

    cJSON *doc = cJSON_ParseWithLength((const char *)buf, n);
    if (!cJSON_IsObject(doc)) {            // no es JSON, o no es un objeto
        cJSON_Delete(doc);
        return true;
    }

    cJSON *j_tipo  = cJSON_GetObjectItem(doc, "tipo");
    cJSON *j_idpaq = cJSON_GetObjectItem(doc, "id_paquete");
    int  tipo  = cJSON_IsNumber(j_tipo)  ? j_tipo->valueint  : -1;
    long idpaq = cJSON_IsNumber(j_idpaq) ? (long)j_idpaq->valueint : -1;
    if (tipo < 0) { cJSON_Delete(doc); return true; }

    // Aprender la direccion de la unidad de procesamiento a partir del
    // remitente. Se hace recien aca, con un mensaje que tiene forma de orden:
    // un datagrama cualquiera no debe poder redirigir la telemetria.
    pi_addr = from;
    pi_addr.sin_port = htons(PUERTO_TELEMETRIA);
    pi_ip_conocida = true;

    if (idpaq >= 0 && dedup_ya_visto(idpaq)) {      // esta orden ya se ejecuto
        ESP_LOGI(TAG, "[udp] dup id=%ld", idpaq);
        cJSON_Delete(doc);
        return true;
    }
    if (idpaq >= 0) dedup_marcar(idpaq);

    cJSON *j_int = cJSON_GetObjectItem(doc, "intensidad");
    cJSON *j_dur = cJSON_GetObjectItem(doc, "duracion_ms");
    cJSON *j_seq = cJSON_GetObjectItem(doc, "id_secuencia");
    int intensidad   = cJSON_IsNumber(j_int) ? j_int->valueint : 0;
    int duracion     = cJSON_IsNumber(j_dur) ? j_dur->valueint : 0;
    int id_secuencia = cJSON_IsNumber(j_seq) ? j_seq->valueint : -1;
    cJSON_Delete(doc);
    if (duracion < 0) duracion = 0;

    // Tipo 99: la Pi anuncia que se detiene. No involucra actuadores.
    if (tipo == TIPO_APAGADO_LIMPIO) {
        apagado_limpio_recibido = true;
        ESP_LOGI(TAG, "[cmd] APAGADO_LIMPIO -> la proxima perdida es esperada (no se avisa)");
        return true;
    }

    // Cualquier otra orden prueba que la Pi esta operando. Si antes habia
    // anunciado su detencion, ese anuncio caduca: fue relanzada sin que la
    // pulsera perdiera la red, y una caida posterior ya no seria esperada.
    if (apagado_limpio_recibido) {
        apagado_limpio_recibido = false;
        ESP_LOGI(TAG, "[cmd] la Pi volvio a operar -> se anula el anuncio de apagado");
    }

    if (s_comando_cb) s_comando_cb(tipo, intensidad, (uint32_t)duracion, id_secuencia);
    return true;
}

// Atiende los datagramas pendientes, hasta el tope por vuelta.
static void udp_atender(void) {
    for (int i = 0; i < MAX_DATAGRAMAS_POR_VUELTA; i++) {
        if (!udp_recibir()) break;
    }
}

// ===========================================================================
// TELEMETRIA Y RESPUESTA A LA VERIFICACION
// ===========================================================================
static int sock_out(void) {
    if (sock_telemetria < 0) {
        sock_telemetria = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    }
    return sock_telemetria;
}

static struct sockaddr_in destino_pi(void) {
    struct sockaddr_in d = {0};
    d.sin_family = AF_INET;
    d.sin_port   = htons(PUERTO_TELEMETRIA);
    if (pi_ip_conocida) {
        d.sin_addr = pi_addr.sin_addr;
    } else {
        d.sin_addr.s_addr = inet_addr(PI_IP_RESPALDO);
    }
    return d;
}

static bool enviar_a_pi(const char *datos, int len) {
    int s = sock_out();
    if (s < 0) return false;
    struct sockaddr_in d = destino_pi();
    return sendto(s, datos, len, 0, (struct sockaddr *)&d, sizeof(d)) >= 0;
}

void red_marcar_secuencia_iniciada(void) {
    flag_ack_recibido = true;
    flag_secuencia_replicada = true;
}

void red_enviar_ack(int id_secuencia, bool correcta, long tr_ms) {
    if (!red_operando_con_pi()) return;

    // Mismo formato que arma protocolo.py (serializar_ack)
    ack_len = snprintf(ack_buf, sizeof(ack_buf),
                       "{\"v\":%d,\"msg\":\"ack\",\"id_secuencia\":%d,"
                       "\"secuencia_correcta\":%s,\"tiempo_respuesta_ms\":%ld}",
                       PROTOCOLO_VERSION, id_secuencia,
                       correcta ? "true" : "false", tr_ms);
    if (ack_len <= 0 || ack_len >= (int)sizeof(ack_buf)) { ack_len = 0; return; }

    // Primera copia ahora; las demas salen desde ack_reenvio_update().
    bool ok = enviar_a_pi(ack_buf, ack_len);
    ack_copias_pendientes = ACK_REENVIOS - 1;
    ack_ts_prox_ms = millis() + ACK_ESPACIADO_MS;

    ESP_LOGI(TAG, "[ack] id=%d correcta=%d tr=%ldms (%s), %d copia(s) mas en camino",
             id_secuencia, correcta, tr_ms, ok ? "ok" : "FALLO", ack_copias_pendientes);
}

// Envia las copias restantes de la respuesta, de a una y espaciadas, sin
// detener el ciclo principal. La Pi publica una sola (las reconoce por el
// id_secuencia). Si un datagrama se pierde, la respuesta llega igual y un
// conductor que respondio bien no asciende de nivel por vencimiento.
static void ack_reenvio_update(void) {
    if (ack_copias_pendientes <= 0) return;
    if ((millis() - ack_ts_prox_ms) < 0) return;
    enviar_a_pi(ack_buf, ack_len);
    ack_copias_pendientes--;
    ack_ts_prox_ms = millis() + ACK_ESPACIADO_MS;
}

static void telemetria_update(void) {
    if ((millis() - tele_ts_prox_ms) < 0) return;
    tele_ts_prox_ms = millis() + PERIODO_TELEMETRIA_MS;
    tele_id_paquete++;

    int bpm = 0;
    bool bpm_valido = s_bpm_provider ? s_bpm_provider(&bpm) : false;
    int bateria_pct = s_bateria_provider ? s_bateria_provider() : 0;

    cJSON *doc = cJSON_CreateObject();
    if (!doc) return;
    cJSON_AddNumberToObject(doc, "v", PROTOCOLO_VERSION);
    cJSON_AddStringToObject(doc, "msg", "telemetria");
    // Sin medicion confiable se informa null. Repetir el ultimo valor seria
    // darle a la Pi una frecuencia que el sensor ya no respalda.
    if (bpm_valido) cJSON_AddNumberToObject(doc, "bpm", bpm);
    else            cJSON_AddNullToObject(doc, "bpm");
    cJSON_AddBoolToObject(doc, "ack_recibido", flag_ack_recibido);
    cJSON_AddBoolToObject(doc, "secuencia_replicada", flag_secuencia_replicada);
    cJSON_AddNumberToObject(doc, "bateria", bateria_pct);
    cJSON_AddNumberToObject(doc, "id_paquete", tele_id_paquete);

    char *out = cJSON_PrintUnformatted(doc);
    if (out) {
        enviar_a_pi(out, strlen(out));
        free(out);
    }
    cJSON_Delete(doc);

    flag_ack_recibido = false;
    flag_secuencia_replicada = false;
}

// ===========================================================================
// MAQUINA DE CONECTIVIDAD
//
//   CONECTANDO_PI    intenta asociarse a la Pi; reintenta cada
//                    INTENTO_TIMEOUT_MS + ESPERA_REINTENTO_MS
//   OPERANDO_PI      operacion normal
//   CONMUTAR_CAMION  tras UMBRAL_PERDIDA_PI_MS sin la Pi: se asocia a la red
//                    de respaldo y, si la perdida no fue anunciada, avisa
//   MODO_TESTIGO     permanece en la red de respaldo; cada REVISAR_PI_CADA_MS
//                    se desconecta un momento para ver si la Pi volvio
//
// Ninguna rama espera: cada una mira el reloj y vuelve.
// ===========================================================================
static void fase_entrar(fase_t nueva) {
    fase = nueva;
    ts_fase_ms = millis();
}

static int64_t fase_transcurrido_ms(void) {
    return millis() - ts_fase_ms;
}

static void detener_actuadores(void) {
    if (s_comando_cb) s_comando_cb(TIPO_APAGAR_TODO, 0, 0, -1);
}

static void asociar_a_pi(void) {
    wifi_start_sta_ip_fija(PI_SSID, PI_PASS, true);
    fase_entrar(FASE_ASOCIANDO_PI);
}

static void asociar_a_camion(void) {
    wifi_start_sta_ip_fija(CAMION_SSID, CAMION_PASS, false);
    fase_entrar(FASE_ASOCIANDO_CAMION);
}

static void entrar_operando_pi(void) {
    conec = RED_OPERANDO_PI;
    ts_desde_sin_pi_ms = 0;
    apagado_limpio_recibido = false;
    aviso_ya_enviado = false;
}

static void entrar_modo_testigo(void) {
    conec = RED_MODO_TESTIGO;
    fase_entrar(FASE_REPOSO);
}

// Espera comun a CONMUTAR_CAMION y a MODO_TESTIGO: asociarse a la red de
// respaldo y, si corresponde, enviar el aviso. Devuelve true cuando termino.
static bool respaldo_y_aviso_listos(void) {
    if (fase == FASE_ASOCIANDO_CAMION) {
        if (!wifi_conectado && fase_transcurrido_ms() < CAMION_TIMEOUT_MS) return false;
        if (wifi_conectado) ESP_LOGI(TAG, "[con] asociada a la red de respaldo");
        else                ESP_LOGW(TAG, "[con] no se pudo asociar a la red de respaldo");
        if (aviso_lanzar_si_corresponde()) {
            fase_entrar(FASE_AVISANDO);
            return false;
        }
        return true;
    }
    if (fase == FASE_AVISANDO) {
        if (s_aviso == AVISO_EN_CURSO && fase_transcurrido_ms() < AVISO_ESPERA_MAX_MS) return false;
        if (s_aviso == AVISO_EN_CURSO) ESP_LOGW(TAG, "[tg] el aviso demora demasiado; se sigue sin esperarlo");
        return true;
    }
    return true;
}

static void conectividad_update(void) {
    aviso_recoger_resultado();

    switch (conec) {
    case RED_CONECTANDO_PI:
        if (fase == FASE_ESPERA_REINTENTO) {
            if (fase_transcurrido_ms() >= ESPERA_REINTENTO_MS) asociar_a_pi();
            break;
        }
        // FASE_ASOCIANDO_PI
        if (wifi_conectado) {
            ESP_LOGI(TAG, "[con] OPERANDO con la Pi.");
            entrar_operando_pi();
        } else if (fase_transcurrido_ms() >= INTENTO_TIMEOUT_MS) {
            if (ts_desde_sin_pi_ms == 0) ts_desde_sin_pi_ms = millis();
            if ((millis() - ts_desde_sin_pi_ms) >= UMBRAL_PERDIDA_PI_MS) {
                ESP_LOGW(TAG, "[con] sin la Pi hace %d s -> red de respaldo",
                         UMBRAL_PERDIDA_PI_MS / 1000);
                conec = RED_CONMUTAR_CAMION;
                detener_actuadores();
                asociar_a_camion();
            } else {
                ESP_LOGI(TAG, "[con] intento fallido; reintenta en %d s",
                         ESPERA_REINTENTO_MS / 1000);
                fase_entrar(FASE_ESPERA_REINTENTO);
            }
        }
        break;

    case RED_OPERANDO_PI:
        if (!wifi_conectado) {
            ESP_LOGW(TAG, "[con] se perdio el enlace con la Pi -> reconectando");
            // Una desconexion en plena vibracion no debe dejar el motor encendido
            detener_actuadores();
            ack_copias_pendientes = 0;
            ts_desde_sin_pi_ms = millis();
            conec = RED_CONECTANDO_PI;
            asociar_a_pi();
        }
        break;

    case RED_CONMUTAR_CAMION:
        if (respaldo_y_aviso_listos()) entrar_modo_testigo();
        break;

    case RED_MODO_TESTIGO:
        switch (fase) {
        case FASE_REPOSO:
            if (fase_transcurrido_ms() >= REVISAR_PI_CADA_MS) {
                if (s_aviso == AVISO_EN_CURSO) break;   // no cortar la red en medio del aviso
                ESP_LOGI(TAG, "[con] testigo: revisando si volvio la Pi...");
                asociar_a_pi();
            }
            break;
        case FASE_ASOCIANDO_PI:
            if (wifi_conectado) {
                ESP_LOGI(TAG, "[con] la Pi VOLVIO -> operacion normal");
                entrar_operando_pi();
            } else if (fase_transcurrido_ms() >= INTENTO_TIMEOUT_MS) {
                asociar_a_camion();      // sigue sin aparecer: de vuelta al respaldo
            }
            break;
        default:   // FASE_ASOCIANDO_CAMION, FASE_AVISANDO
            // El aviso que no pudo salir antes se reintenta en cada revision
            if (respaldo_y_aviso_listos()) fase_entrar(FASE_REPOSO);
            break;
        }
        break;
    }
}

// ===========================================================================
// API PUBLICA
// ===========================================================================
void red_set_comando_handler(red_comando_cb_t cb) { s_comando_cb = cb; }
void red_set_bpm_provider(red_bpm_provider_t fn) { s_bpm_provider = fn; }
void red_set_bateria_provider(red_bateria_provider_t fn) { s_bateria_provider = fn; }

void red_init(void) {
    dedup_init();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_netif = esp_netif_create_default_wifi_sta();

    wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                                &wifi_event_handler, NULL));

    conec = RED_CONECTANDO_PI;
    asociar_a_pi();
}

void red_update(void) {
    conectividad_update();

    if (red_operando_con_pi()) {
        udp_comandos_abrir();
        udp_atender();
        ack_reenvio_update();
        telemetria_update();
    } else {
        udp_comandos_cerrar();
        ack_copias_pendientes = 0;
    }
}
