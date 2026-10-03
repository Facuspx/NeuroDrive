"""
test_persistencia_sesion.py - Tests de la persistencia del estado entre sesiones.

Ejecutar:
    cd ~/NeuroDrive
    python -m NeuroDrive_Core.tests.test_persistencia_sesion

No requiere hardware ni colas: usa archivos temporales.

Cubre:
  1. Lectura y escritura: ida y vuelta, archivo ausente, danado o vencido.
  2. La regla de arranque (resolver_arranque), que es logica pura.
  3. El ciclo completo con la FSM real y el Orquestador: lo que el informe
     describe como "reinicio durante una sesion activa" (ensayo EME-8),
     incluido el reinicio imprevisto, en el que nadie llama a detener().
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from common.contratos import (
    Envelope,
    EstadoFSM,
    EstadoSesion,
    EventoAckWearable,
    EventoProcesado,
    OrigenEvento,
    TipoMensaje,
)
from NeuroDrive_Core.despachador import ActuadorSimulado, DespachadorComandos
from NeuroDrive_Core.fsm import FSM, ConfigFSM
from NeuroDrive_Core.orquestador import Orquestador
from NeuroDrive_Core.persistencia_sesion import (
    ArranqueSesion,
    PersistenciaSesion,
    capturar,
    resolver_arranque,
)

# Los casos de archivo danado y de disco inaccesible se provocan a proposito
logging.disable(logging.CRITICAL)

_resultados = []


def _test(nombre):
    def wrapper(func):
        try:
            func()
            _resultados.append((nombre, True, ""))
            print(f"  [OK]  {nombre}")
        except AssertionError as e:
            _resultados.append((nombre, False, str(e)))
            print(f"  [FAIL] {nombre}: {e}")
        except Exception as e:
            _resultados.append((nombre, False, f"{type(e).__name__}: {e}"))
            print(f"  [ERROR] {nombre}: {e}")
            traceback.print_exc()
        return func
    return wrapper


# Instante de referencia de toda la suite. Se toma del reloj real porque el
# Orquestador fecha cada guardado con time.time(); el "paso del tiempo" entre
# sesiones se simula pasando `ahora` a cargar() y a resolver_arranque().
AHORA = float(int(time.time()))
VENTANA_EPISODIOS = 900.0
VENTANA_BOSTEZOS = 900.0


def _sesion(estado=EstadoFSM.ALERTA_LEVE, hace=60.0, episodios=(), bostezos=()):
    return EstadoSesion(
        timestamp_guardado=AHORA - hace,
        estado_fsm=estado,
        bostezos_recientes=tuple(bostezos),
        episodios_severos=tuple(episodios),
        motivo_guardado="test",
    )


def _arranque(previa) -> ArranqueSesion:
    return resolver_arranque(previa, AHORA, VENTANA_EPISODIOS, VENTANA_BOSTEZOS)


# =============================================================================
print("\n--- Lectura y escritura ---")


@_test("guardar y cargar: ida y vuelta sin alteracion")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = PersistenciaSesion(Path(d) / "estado.json", ttl_seg=900)
        original = _sesion(episodios=(AHORA - 100, AHORA - 50), bostezos=(AHORA - 300,))
        assert p.guardar(original) is True
        assert p.cargar(ahora=AHORA) == original
        assert p.guardados_ok == 1


@_test("guardar crea la carpeta si no existe y no deja archivo temporal")
def _():
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "sub" / "carpeta" / "estado.json"
        p = PersistenciaSesion(ruta)
        assert p.guardar(_sesion())
        assert ruta.is_file()
        assert [f.name for f in ruta.parent.iterdir()] == ["estado.json"]


@_test("sin archivo, cargar devuelve None")
def _():
    with tempfile.TemporaryDirectory() as d:
        assert PersistenciaSesion(Path(d) / "no_existe.json").cargar(ahora=AHORA) is None


@_test("archivo danado o de otro formato: cargar devuelve None, sin excepcion")
def _():
    contenidos = (
        "esto no es json {{{",
        "",                                            # archivo vacio (corte al crear)
        "[1, 2, 3]",                                   # JSON valido, no es un objeto
        '{"estado_fsm": 2}',                           # falta timestamp_guardado
        '{"timestamp_guardado": 1.0, "estado_fsm": 99}',                    # estado inexistente
        json.dumps({"timestamp_guardado": AHORA, "estado_fsm": 2,           # contrato anterior
                    "microsuenos_recientes": [], "cabeceos_recientes": []}),
    )
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "estado.json"
        p = PersistenciaSesion(ruta)
        for texto in contenidos:
            ruta.write_text(texto, encoding="utf-8")
            assert p.cargar(ahora=AHORA) is None, f"deberia ignorarse: {texto[:40]!r}"


@_test("sesion mas vieja que el plazo: se ignora")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = PersistenciaSesion(Path(d) / "estado.json", ttl_seg=900)
        p.guardar(_sesion(hace=901))
        assert p.cargar(ahora=AHORA) is None
        p.guardar(_sesion(hace=899))
        assert p.cargar(ahora=AHORA) is not None


@_test("reloj por detras del guardado: la sesion se toma como reciente")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = PersistenciaSesion(Path(d) / "estado.json", ttl_seg=900)
        p.guardar(_sesion(hace=-3600))        # "guardado dentro de una hora"
        assert p.cargar(ahora=AHORA) is not None


@_test("si no se puede escribir, guardar devuelve False y no lanza")
def _():
    with tempfile.TemporaryDirectory() as d:
        estorbo = Path(d) / "soy_un_archivo"
        estorbo.write_text("x")
        # La "carpeta" de destino es en realidad un archivo: no se puede crear
        p = PersistenciaSesion(estorbo / "estado.json")
        assert p.guardar(_sesion()) is False
        assert p.guardados_fallidos == 1


@_test("un guardado fallido no destruye el estado anterior")
def _():
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "estado.json"
        p = PersistenciaSesion(ruta)
        buena = _sesion(estado=EstadoFSM.ALERTA_MEDIA)
        assert p.guardar(buena)

        class _NoSerializable:
            def to_json(self):
                raise ValueError("no se puede serializar")
        assert p.guardar(_NoSerializable()) is False
        assert p.cargar(ahora=AHORA) == buena
        assert not ruta.with_name("estado.json.tmp").exists()


@_test("borrar elimina el archivo y es inofensivo si no existe")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = PersistenciaSesion(Path(d) / "estado.json")
        p.borrar()                             # no existe: no debe fallar
        p.guardar(_sesion())
        p.borrar()
        assert p.cargar(ahora=AHORA) is None


# =============================================================================
print("\n--- Regla de arranque ---")


@_test("sin sesion anterior se inicia en NORMAL, sin indicadores")
def _():
    a = _arranque(None)
    assert a.estado_inicial == EstadoFSM.NORMAL and not a.retomada
    assert a.episodios_severos == () and a.bostezos == ()


@_test("sesion anterior en PRE_ALERTA o en cualquier alerta: se inicia en PRE_ALERTA")
def _():
    for estado in (EstadoFSM.PRE_ALERTA, EstadoFSM.ALERTA_LEVE,
                   EstadoFSM.ALERTA_MEDIA, EstadoFSM.CRITICO):
        a = _arranque(_sesion(estado=estado))
        assert a.estado_inicial == EstadoFSM.PRE_ALERTA, estado.name
        assert a.retomada


@_test("sesion anterior en NORMAL sin episodios: se inicia en NORMAL")
def _():
    a = _arranque(_sesion(estado=EstadoFSM.NORMAL))
    assert a.estado_inicial == EstadoFSM.NORMAL and a.retomada


@_test("sesion anterior en NORMAL pero con episodios vigentes: PRE_ALERTA")
def _():
    a = _arranque(_sesion(estado=EstadoFSM.NORMAL, episodios=(AHORA - 200,)))
    assert a.estado_inicial == EstadoFSM.PRE_ALERTA
    assert a.episodios_severos == (AHORA - 200,)


@_test("sesion anterior en MODO_DEGRADADO: NORMAL salvo que haya episodios")
def _():
    assert _arranque(_sesion(estado=EstadoFSM.MODO_DEGRADADO)).estado_inicial == EstadoFSM.NORMAL
    con = _arranque(_sesion(estado=EstadoFSM.MODO_DEGRADADO, episodios=(AHORA - 10,)))
    assert con.estado_inicial == EstadoFSM.PRE_ALERTA


@_test("los episodios y bostezos fuera de su ventana no se conservan")
def _():
    a = _arranque(_sesion(
        estado=EstadoFSM.NORMAL,
        episodios=(AHORA - 2000, AHORA - 100),
        bostezos=(AHORA - 950, AHORA - 800, AHORA - 5),
    ))
    assert a.episodios_severos == (AHORA - 100,)
    assert a.bostezos == (AHORA - 800, AHORA - 5)
    # Un episodio vencido no alcanza para arrancar en PRE_ALERTA
    solo_viejos = _arranque(_sesion(estado=EstadoFSM.NORMAL, episodios=(AHORA - 2000,)))
    assert solo_viejos.estado_inicial == EstadoFSM.NORMAL


@_test("marcas posteriores a la hora actual (reloj retrasado) se recortan a ahora")
def _():
    a = _arranque(_sesion(hace=-600, episodios=(AHORA + 500, AHORA + 100)))
    assert a.episodios_severos == (AHORA, AHORA)
    assert a.estado_inicial == EstadoFSM.PRE_ALERTA


# =============================================================================
print("\n--- Ciclo completo con la FSM y el Orquestador ---")


class _GestorNulo:
    activo = False
    def iniciar(self): pass
    def detener(self): pass
    def obtener_evento(self, timeout=0.0): return None


class _PreFSMPasante:
    """Entrega al Orquestador el EventoProcesado tal cual; el resto, None."""
    def __init__(self):
        self.bostezos = []
    def procesar(self, envelope):
        ev = envelope.evento
        if isinstance(ev, EventoProcesado):
            if ev.bostezo:
                self.bostezos.append(ev.timestamp)
            return ev
        return None
    def get_bostezos_recientes(self):
        return list(self.bostezos)
    def restaurar_bostezos(self, marcas):
        self.bostezos = sorted(marcas)


def _nucleo(persistencia, arranque: ArranqueSesion = ArranqueSesion()):
    """Arma FSM real + Orquestador con la persistencia, como lo hace main.py."""
    pre = _PreFSMPasante()
    pre.restaurar_bostezos(arranque.bostezos)
    fsm = FSM(ConfigFSM(), estado_inicial=arranque.estado_inicial,
              episodios_previos=arranque.episodios_severos)
    desp = DespachadorComandos()
    desp.registrar_actuador(ActuadorSimulado("todo"))
    orq = Orquestador(_GestorNulo(), pre, fsm, desp, persistencia=persistencia)
    orq.iniciar()
    return orq, fsm, pre


def _env(evento, seq):
    """Envelope con reloj de recepcion fijo (el Orquestador usa ese para la FSM)."""
    tipo = (TipoMensaje.EVENTO_ACK_WEARABLE if isinstance(evento, EventoAckWearable)
            else TipoMensaje.EVENTO_PROCESADO)
    return Envelope(
        tipo=tipo, origen=OrigenEvento.VISION, id_dispositivo="d", id_sesion="s",
        id_mensaje=f"m-{seq}", numero_secuencia=seq,
        timestamp_origen=evento.timestamp, timestamp_recepcion=evento.timestamp,
        evento=evento,
    )


@_test("capturar() arma el EstadoSesion con lo que saben la FSM y el Pre-FSM")
def _():
    pre = _PreFSMPasante()
    pre.restaurar_bostezos([AHORA - 40])
    fsm = FSM(ConfigFSM(), estado_inicial=EstadoFSM.PRE_ALERTA, episodios_previos=[AHORA - 90])
    s = capturar(fsm, pre, "prueba", ahora=AHORA)
    assert s.estado_fsm == EstadoFSM.PRE_ALERTA
    assert s.episodios_severos == (AHORA - 90,) and s.bostezos_recientes == (AHORA - 40,)
    assert s.timestamp_guardado == AHORA and s.motivo_guardado == "prueba"


@_test("el estado se guarda en cada transicion, sin esperar al apagado")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = PersistenciaSesion(Path(d) / "estado.json")
        orq, fsm, _ = _nucleo(p)
        assert p.cargar() is None, "antes del primer cambio no hay nada guardado"
        orq._procesar_envelope(_env(EventoProcesado(timestamp=AHORA, microsueno=True), 1))
        assert fsm.get_estado_actual() == EstadoFSM.ALERTA_LEVE
        guardado = p.cargar(ahora=AHORA)
        assert guardado is not None and guardado.estado_fsm == EstadoFSM.ALERTA_LEVE
        assert guardado.episodios_severos == (AHORA,)
        assert guardado.motivo_guardado == "evento_severo_desde_normal"
        orq.detener()


@_test("un evento sin novedades no escribe en disco")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = PersistenciaSesion(Path(d) / "estado.json")
        orq, _, _ = _nucleo(p)
        for i in range(20):
            orq._procesar_envelope(_env(EventoProcesado(timestamp=AHORA + i), i + 1))
        assert p.guardados_ok == 0
        orq.detener()                       # el apagado ordenado si guarda
        assert p.guardados_ok == 1


@_test("un bostezo confirmado se guarda aunque no haya transicion")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = PersistenciaSesion(Path(d) / "estado.json")
        orq, fsm, _ = _nucleo(p)
        orq._procesar_envelope(_env(
            EventoProcesado(timestamp=AHORA, bostezo=True, bostezos_ventana_larga=1), 1))
        assert fsm.get_estado_actual() == EstadoFSM.NORMAL
        guardado = p.cargar(ahora=AHORA)
        assert guardado is not None and guardado.bostezos_recientes == (AHORA,)
        orq.detener()


@_test("reinicio IMPREVISTO en plena alerta: la sesion siguiente retoma en PRE_ALERTA")
def _():
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "estado.json"
        # --- Sesion 1: microsueno, alerta, y se corta la alimentacion ---
        orq, fsm, _ = _nucleo(PersistenciaSesion(ruta))
        orq._procesar_envelope(_env(EventoProcesado(timestamp=AHORA, microsueno=True), 1))
        assert fsm.get_estado_actual() == EstadoFSM.ALERTA_LEVE
        orq.despachador.detener()           # se liberan hilos, pero NO se llama a orq.detener()

        # --- Sesion 2: 40 s despues ---
        p2 = PersistenciaSesion(ruta)
        arranque = resolver_arranque(p2.cargar(ahora=AHORA + 40), AHORA + 40,
                                     VENTANA_EPISODIOS, VENTANA_BOSTEZOS)
        assert arranque.estado_inicial == EstadoFSM.PRE_ALERTA
        assert arranque.episodios_severos == (AHORA,)
        orq2, fsm2, _ = _nucleo(p2, arranque)
        assert fsm2.get_estado_actual() == EstadoFSM.PRE_ALERTA
        assert fsm2.get_estado_interno().id_secuencia_ack_pendiente is None, (
            "la verificacion de la sesion anterior no se arrastra")
        orq2.detener()


@_test("la fatiga recurrente sobrevive al reinicio (EME-8)")
def _():
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "estado.json"
        # --- Sesion 1: tres episodios, cada uno respondido correctamente ---
        orq, fsm, _ = _nucleo(PersistenciaSesion(ruta))
        seq = 0
        for i in range(3):
            t = AHORA + i * 100
            seq += 1
            orq._procesar_envelope(_env(EventoProcesado(timestamp=t, microsueno=True), seq))
            id_seq = fsm.get_estado_interno().id_secuencia_ack_pendiente
            seq += 1
            orq._procesar_envelope(_env(EventoAckWearable(
                timestamp=t + 5, id_secuencia=id_seq,
                secuencia_correcta=True, tiempo_respuesta_ms=1500), seq))
        assert fsm.get_estado_actual() == EstadoFSM.PRE_ALERTA
        assert len(fsm.get_episodios_severos()) == 3
        orq.detener()                       # apagado ordenado

        # --- Sesion 2: dos minutos despues ---
        t2 = AHORA + 400
        p2 = PersistenciaSesion(ruta)
        arranque = resolver_arranque(p2.cargar(ahora=t2), t2,
                                     VENTANA_EPISODIOS, VENTANA_BOSTEZOS)
        assert arranque.estado_inicial == EstadoFSM.PRE_ALERTA
        assert len(arranque.episodios_severos) == 3
        orq2, fsm2, _ = _nucleo(p2, arranque)
        # El primer microsueno de la sesion nueva ya es el cuarto episodio:
        # va directo a ALERTA_MEDIA. Sin persistencia habria sido ALERTA_LEVE.
        orq2._procesar_envelope(_env(EventoProcesado(timestamp=t2 + 10, microsueno=True), 1))
        assert fsm2.get_estado_actual() == EstadoFSM.ALERTA_MEDIA
        orq2.detener()


@_test("pasado el plazo, la sesion siguiente arranca limpia")
def _():
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "estado.json"
        orq, _, _ = _nucleo(PersistenciaSesion(ruta, ttl_seg=900))
        orq._procesar_envelope(_env(EventoProcesado(timestamp=AHORA, microsueno=True), 1))
        orq.despachador.detener()
        p2 = PersistenciaSesion(ruta, ttl_seg=900)
        tarde = AHORA + 3600
        arranque = resolver_arranque(p2.cargar(ahora=tarde), tarde,
                                     VENTANA_EPISODIOS, VENTANA_BOSTEZOS)
        assert arranque.estado_inicial == EstadoFSM.NORMAL and not arranque.retomada


@_test("un fallo de disco al guardar no interrumpe el procesamiento")
def _():
    class _DiscoRoto:
        def guardar(self, estado):
            raise OSError("disco lleno")
    orq, fsm, _ = _nucleo(_DiscoRoto())
    orq._procesar_envelope(_env(EventoProcesado(timestamp=AHORA, microsueno=True), 1))
    assert fsm.get_estado_actual() == EstadoFSM.ALERTA_LEVE
    assert orq.stats.errores_procesamiento == 0
    orq.detener()


# =============================================================================
# RESUMEN
# =============================================================================

def _resumen() -> int:
    total = len(_resultados)
    pasaron = sum(1 for _, ok, _ in _resultados if ok)
    fallaron = total - pasaron

    print("\n" + "=" * 60)
    print(f"  Resumen: {pasaron}/{total} tests pasaron")
    if fallaron:
        print(f"  Fallaron {fallaron}:")
        for nombre, ok, err in _resultados:
            if not ok:
                print(f"    - {nombre}: {err}")
    print("=" * 60)
    return 0 if fallaron == 0 else 1


if __name__ == "__main__":
    print("=" * 60)
    print("  NeuroDrive - Tests de persistencia de sesion")
    print("=" * 60)
    sys.exit(_resumen())
