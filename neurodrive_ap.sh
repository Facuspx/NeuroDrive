#!/bin/bash
# ==========================================================================
#  NeuroDrive - Raspberry Pi 5 como Access Point WiFi (v2, compatible ESP32-S3)
# ==========================================================================
#  v2: canal 6 FIJO + WPA2 explicito (proto rsn / ccmp) en la CREACION.
#  El ESP32-S3 solo escanea canales 1-11 y habla WPA2 puro; con canal 0 (auto)
#  o cifrado sin definir, no encontraba/negociaba el AP. Ahora queda persistente.
#
#  BUG DE LA v1 CORREGIDO: 'up' borraba y recreaba la conexion cada vez, asi
#  perdia el canal y el cifrado. Ahora 'up' crea SOLO si falta y reaplica los
#  ajustes, sin borrar. Para borrar a proposito, usar 'recrear'.
#
#  Topologia:
#    Pi (AP) -> 192.168.4.1   (gateway; escucha telemetria en 5005)
#    Pulsera -> 192.168.4.20  (direccion fija que establece el firmware; no
#                              usa la asignacion automatica del AP. Escucha
#                              ordenes en 5006)
#
#  USO:
#    chmod +x neurodrive_ap.sh
#    ./neurodrive_ap.sh up | down | status | recrear
# ==========================================================================

SSID_NAME="NeuroDrive_AP"
WIFI_PASS="paquito2025"
CON_NAME="neurodrive-ap"
PI_IP="192.168.4.1"
WIFI_IFACE="wlan0"
CANAL="6"                 # 1-11 (el ESP32-S3 no ve canales altos ni 'auto')

aplicar_ajustes() {
    nmcli con modify "$CON_NAME" 802-11-wireless.mode ap
    nmcli con modify "$CON_NAME" 802-11-wireless.band bg
    nmcli con modify "$CON_NAME" 802-11-wireless.channel "$CANAL"
    nmcli con modify "$CON_NAME" 802-11-wireless-security.key-mgmt wpa-psk
    nmcli con modify "$CON_NAME" 802-11-wireless-security.psk "$WIFI_PASS"
    nmcli con modify "$CON_NAME" 802-11-wireless-security.proto rsn
    nmcli con modify "$CON_NAME" 802-11-wireless-security.pairwise ccmp
    nmcli con modify "$CON_NAME" 802-11-wireless-security.group ccmp
    nmcli con modify "$CON_NAME" 802-11-wireless-security.pmf 1
    nmcli con modify "$CON_NAME" ipv4.addresses "$PI_IP/24"
    nmcli con modify "$CON_NAME" ipv4.method shared
}

crear_si_no_existe() {
    if ! nmcli -t -f NAME connection show | grep -qx "$CON_NAME"; then
        echo "[*] Creando la conexion '$CON_NAME' por primera vez..."
        nmcli con add type wifi ifname "$WIFI_IFACE" con-name "$CON_NAME" \
            autoconnect no ssid "$SSID_NAME" || { echo "[X] fallo al crear"; exit 1; }
    fi
    aplicar_ajustes
}

levantar() {
    echo "[*] Levantando AP '$SSID_NAME' (canal $CANAL, WPA2)..."
    crear_si_no_existe
    if nmcli con up "$CON_NAME"; then
        echo "[ok] AP arriba.  SSID=$SSID_NAME  clave=$WIFI_PASS  IP=$PI_IP  canal=$CANAL"
        echo "    Firmware:  RED_SSID=\"$SSID_NAME\"  RED_PASS=\"$WIFI_PASS\"  PI_IP_RESPALDO=\"$PI_IP\""
    else
        echo "[X] No se pudo levantar. Revisar que $WIFI_IFACE soporte modo AP."
        exit 1
    fi
}

apagar() {
    echo "[*] Apagando el AP..."
    nmcli con down "$CON_NAME" 2>/dev/null
    echo "[ok] AP apagado."
}

recrear() {
    echo "[*] Borrando y recreando la conexion desde cero..."
    nmcli con down "$CON_NAME" 2>/dev/null
    nmcli connection delete "$CON_NAME" >/dev/null 2>&1
    crear_si_no_existe
    echo "[ok] recreada. Corre '$0 up' para levantarla."
}

estado() {
    echo "=== Estado del AP ==="
    if nmcli -t -f NAME,STATE connection show --active | grep -q "^$CON_NAME:activated"; then
        echo "AP ACTIVO ($SSID_NAME)"
        echo "Canal: $(nmcli -g 802-11-wireless.channel con show $CON_NAME)"
        echo "IP de la Pi: $(nmcli -g IP4.ADDRESS device show $WIFI_IFACE | head -1)"
        echo "Dispositivos conectados (la pulsera deberia aparecer):"
        ip neigh show dev "$WIFI_IFACE" | grep -v FAILED || echo "  (ninguno todavia)"
    else
        echo "AP NO activo."
    fi
}

case "$1" in
    up)      levantar ;;
    down)    apagar ;;
    status)  estado ;;
    recrear) recrear ;;
    *)       echo "Uso: $0 {up|down|status|recrear}" ; exit 1 ;;
esac
