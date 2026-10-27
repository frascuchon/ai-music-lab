# Audio2Midi

Plugin REAPER para transcribir audio a MIDI usando modelos de IA en la nube.

## Modelos soportados

| Modelo | Descripción | GPU | Licencia |
|--------|-------------|-----|---------|
| **Basic Pitch** (por defecto) | Un solo instrumento (guitarra, voz, piano). Se ejecuta en local (CPU), sin Modal. Devuelve una pista. | ninguna | Apache 2.0 |
| **MIROS** | Multi-instrumento con beat tracking. Mejor en música real. | A10G | Uso interno |
| **YourMT3+** | Multi-instrumento. Robusto, F1 77.5% Slakh. | T4+ | Apache 2.0 |

**Qué modelo elegir:** para un instrumento solo, Basic Pitch; para una mezcla con varios instrumentos, YourMT3+ o MIROS. En una guitarra distorsionada solista, Basic Pitch coincidió con una referencia pYIN en ~66 % de los fotogramas, frente a ~35 % (YourMT3+) y ~19 % (MIROS), y los dos modelos grandes repartían la guitarra en varias pistas.

**Limpieza de Basic Pitch:** con guitarra distorsionada el modelo añade "notas fantasma" una octava (o octava+quinta) por debajo de la nota real, y notas sueltas en la cola de eco tras terminar la música. Se eliminan automáticamente (en un clip real: 12 de 82 notas, sin perder ninguna correcta). Desmarcar **Beat tracking** hace que use el tempo del proyecto en lugar de estimarlo.

**Nivel de entrada:** antes de transcribir se normaliza el pico a -1 dBFS (máx. +30 dB de ganancia). Un render de guitarra con pico en -22 dBFS hacía que MIROS devolviera un MIDI vacío. Si no se detecta ninguna nota, el plugin lo indica como error.

Los modelos Modal aplican beat tracking (tempo map constante) para alinear las notas
al grid de REAPER. El flag "Beat tracking" del plugin lo controla (MIROS soporta
desactivarlo; YourMT3+ siempre aplica su propio procesamiento).

## Instalación

### 1. Estructura de carpetas requerida

```
plugins-and-extensions/
├── shared/          ← infraestructura común (uv, modal CLI, lib/)
└── Audio2Midi/      ← este plugin
    ├── Audio2Midi.lua
    ├── transcribe.py
    └── research/
        ├── research_basic_pitch.py     (local, uv run)
        ├── research_miros_modal.py
        └── research_yourmt3_modal.py
```

`shared/` debe ser hermano de `Audio2Midi/`. No cambies esta estructura.

### 2. Registrar los scripts en REAPER

En REAPER: **Actions → Load ReaScript** → añadir:
- `shared/Setup.lua` — wizard de configuración global
- `Audio2Midi/Audio2Midi.lua` — acción principal

### 3. Primera configuración (Setup global)

Ejecuta `shared/Setup.lua` en REAPER y completa los pasos:

1. **Entorno común** — instala `uv` si falta → instala deps (modal CLI) → haz login en Modal.
2. **Audio2Midi** — (opcional) precarga pesos MIROS o YourMT3+ en Modal Volumes
   para evitar la descarga en el primer uso (~8 GB MIROS, ~2 GB YourMT3+).

> Requisito de cuenta: necesitas cuenta gratuita en [modal.com](https://modal.com).
> Los primeros $30/mes de cómputo son gratuitos.

## Uso

1. En REAPER, **selecciona** una pista, un item de audio, o un split (sección recortada).
2. Abre **Audio2Midi.lua** desde el Action List.
3. Pulsa **R** para capturar la selección. La UI mostrará el archivo fuente y,
   si es una sección, el rango de tiempo en amarillo.
4. Elige **modelo** y **GPU**.
5. Pulsa **GENERAR MIDI**.

El progreso se muestra en tiempo real en la barra y el log.

### Resultado en REAPER

- **Un instrumento** → nueva pista MIDI nombrada `<fuente> [MIDI <modelo>]`
  en la posición del audio original.
- **Varios instrumentos** → carpeta de pistas nombradas,
  una por instrumento.

Ambos casos quedan como una operación deshacer (`Ctrl+Z`).

## Selección de input: tres niveles

| Nivel | Cómo | Comportamiento |
|-------|------|----------------|
| **Pista** | Seleccionar la pista en REAPER → R | Toma el primer item de audio activo |
| **Item** | Seleccionar un item → R | Transcribe el item completo |
| **Split** | Seleccionar un item recortado → R | Transcribe solo la sección recortada (`D_STARTOFFS` + `D_LENGTH`) |

El botón `...` permite elegir cualquier archivo WAV manualmente (sin capturar de REAPER).

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `Audio2Midi.lua` | Script REAPER (UI + integración) |
| `transcribe.py` | Backend CLI: invoca Modal, gestiona progreso |
| `research/research_miros_modal.py` | App Modal para MIROS |
| `research/research_yourmt3_modal.py` | App Modal para YourMT3+ |

## Notas técnicas

- El proceso de transcripción corre en la nube (Modal). La primera ejecución descarga
  los pesos si no están precargados (~1-2 min adicionales).
- `transcribe.py` crea un directorio temporal por ejecución para evitar conflictos de caché.
- El MIDI resultante se renombra a `<fuente>__<modelo>.mid` antes de importarse.
- Beat tracking: reescribe el tempo map del MIDI a un BPM constante detectado por
  librosa para que las notas queden alineadas al grid de REAPER.
- **MIROS normaliza la entrada a 16 kHz mono** antes de la inferencia (dentro del
  contenedor Modal, con `librosa.load(..., sr=16000, mono=True)`). Esto es necesario
  porque MIROS fue entrenado con audio 16 kHz mono y produce MIDI vacío con cualquier
  otro formato. La normalización es automática: el plugin funciona con clips REAPER de
  44.1/48 kHz estéreo sin intervención del usuario.
