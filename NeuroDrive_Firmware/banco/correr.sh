#!/bin/bash
# ==========================================================================
#  Banco de pruebas del firmware en PC (Linux o Raspberry Pi OS)
#
#  Compila red.c y motor_pads.c SIN modificar junto con banco.c, que ocupa el
#  lugar del hardware (Wi-Fi, motor, pads, reloj), y corre los casos de
#  banco_test.py contra los modulos reales de la Raspberry.
#
#  Requisitos: gcc, python con el entorno del proyecto, y cJSON:
#      sudo apt install libcjson-dev
#  (si esta definido IDF_PATH se usa el cJSON de ESP-IDF y no hace falta)
#
#  USO:   ./correr.sh            todos los casos
#         ./correr.sh <caso>     uno solo
#  No ejecutar junto con main.py: usan los mismos puertos (5005 y 5006).
# ==========================================================================
set -e
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FW="$(dirname "$AQUI")"

if [ -n "$IDF_PATH" ] && [ -f "$IDF_PATH/components/json/cJSON/cJSON.c" ]; then
    CJSON_INC="-I$IDF_PATH/components/json/cJSON"
    CJSON_SRC="$IDF_PATH/components/json/cJSON/cJSON.c"
    CJSON_LIB=""
elif [ -f /usr/include/cjson/cJSON.h ]; then
    CJSON_INC="-I/usr/include/cjson"
    CJSON_SRC=""
    CJSON_LIB="-lcjson"
else
    echo "[X] Falta cJSON. Instalar con:  sudo apt install libcjson-dev"
    exit 1
fi

# Las fuentes se copian a build/ junto con credenciales de relleno. Asi el
# banco nunca compila con credenciales.h real, aunque exista en el proyecto.
mkdir -p "$AQUI/build"
cp "$FW/components/red/red.c" "$FW/components/motor_pads/motor_pads.c" "$AQUI/build/"
cp "$AQUI/credenciales_banco.h" "$AQUI/build/credenciales.h"

# shellcheck disable=SC2086
gcc -shared -fPIC -O1 -Wall -Wextra -Wno-unused-parameter -include unistd.h \
    -I"$AQUI/stubs" -I"$FW/components/red/include" \
    -I"$FW/components/motor_pads/include" $CJSON_INC \
    "$AQUI/banco.c" "$AQUI/build/red.c" "$AQUI/build/motor_pads.c" $CJSON_SRC \
    $CJSON_LIB -o "$AQUI/build/libpulsera.so"

python "$AQUI/banco_test.py" "$@"
