```mermaid
flowchart TB
    subgraph CAP["Captura — procesos productores"]
        VIS["NeuroDrive_Vision<br/><small>Cámara + MediaPipe (Pi 5)</small>"]
        WEA["NeuroDrive_Wearable<br/><small>ESP32-S3 · BPM + táctil</small>"]
    end

    subgraph IPC["IPC — Tramo 1 (POSIX Message Queues)"]
        MQV(["/neurodrive_vision"])
        MQW(["/neurodrive_wearable"])
    end

    subgraph CORE["NeuroDrive_Core — proceso central (hilos + queue.Queue)"]
        direction LR
        GES["Gestor<br/><small>valida · dedup · heartbeat</small>"]
        PRE["Pre-FSM<br/><small>ventanas · métricas</small>"]
        FSM["FSM<br/><small>decide el nivel</small>"]
        DES["Despachador<br/><small>genera comandos</small>"]
        GES --> PRE --> FSM --> DES
    end

    subgraph ACT["Actuadores — Tramo 3 (UDP + GPIO)"]
        BUZ["Buzzer (GPIO BCM)"]
        VOZ["Neuro_voz (Piper TTS)"]
        WEA2["Wearable (vibrar · ACK)"]
        SUP["Supervisor (notificación)"]
    end

    subgraph BASE["Capa transversal — leída por Visión y Core"]
        CFG["config/config.yaml<br/><small>única fuente de umbrales</small>"]
        CMN["common/contratos.py<br/><small>vocabulario compartido</small>"]
    end

    VIS --> MQV --> GES
    WEA --> MQW --> GES
    DES --> BUZ & VOZ & WEA2 & SUP

    CFG -.lee.-> CAP
    CFG -.lee.-> CORE
    CMN -.importan.-> CAP
    CMN -.importan.-> CORE
```

---

## 2. Diagrama de estados de la FSM


```mermaid
stateDiagram-v2
    [*] --> NORMAL

    state "Nivel Normal" as N {
        NORMAL: S0 · conductor alerta
        PRE_ALERTA: S1 · señales leves, observa
        NORMAL --> PRE_ALERTA: señales leves sostenidas
        PRE_ALERTA --> NORMAL: 60s sin eventos negativos
    }

    state "Nivel Advertencia" as A {
        ALERTA_LEVE: S2 · vibración suave + voz, espera ACK
        ALERTA_MEDIA: S3 · vibración fuerte + buzzer + táctil
        ALERTA_LEVE --> ALERTA_MEDIA: timeout ACK / 2º evento
        ALERTA_MEDIA --> ALERTA_LEVE: (sin transición directa)
    }

    state "Nivel Crítico" as C {
        CRITICO: S4 · vibración máx + alarma + supervisor
    }

    PRE_ALERTA --> ALERTA_LEVE: bostezo / microsueño
    ALERTA_LEVE --> PRE_ALERTA: ACK correcto
    ALERTA_MEDIA --> PRE_ALERTA: ACK correcto + BPM normal
    ALERTA_MEDIA --> CRITICO: timeout ACK + BPM crítico
    CRITICO --> PRE_ALERTA: ACK activo (≤15s)

    NORMAL --> MODO_DEGRADADO: fallo sensor (sev ≥ 2)
    PRE_ALERTA --> MODO_DEGRADADO: fallo sensor (sev ≥ 2)
    ALERTA_LEVE --> MODO_DEGRADADO: fallo sensor (sev ≥ 2)
    ALERTA_MEDIA --> MODO_DEGRADADO: fallo sensor (sev ≥ 2)
    CRITICO --> MODO_DEGRADADO: fallo sensor (sev ≥ 2)
    MODO_DEGRADADO --> NORMAL: sensor recuperado
```

---

Con esto tenés la vista general cerrada: el diagrama de bloques con las cuatro capas y los dos cimientos (`config`/`common`), y la FSM completa con el mapeo a los 3 niveles. Ambos los tenés en versión visual (para entender) y en Mermaid (para pegar en tu `.md`).

Cuando vayas analizando módulo por módulo y te surja una duda puntual, decime cuál estás mirando y la vemos en detalle. Un par de cosas que probablemente quieras tener a mano cuando lleguemos al nivel fino: la convención de pitch (cabeza abajo = pitch positivo, ya validada en hardware) y el detalle de que la Visión envía el `pitch_grados` ya normalizado restando el `pitch_neutro` de la calibración.
