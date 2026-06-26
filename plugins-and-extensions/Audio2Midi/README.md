# Audio2Midi

Plugin REAPER para transcribir audio a MIDI usando modelos de IA en la nube.

## Modelos soportados

| Modelo | Descripción | GPU | Licencia |
|--------|-------------|-----|---------|
| **MIROS** (por defecto) | Multi-instrumento con beat tracking. Mejor en música real. | A10G | Uso interno |
| **YourMT3+** | Multi-instrumento. Robusto, F1 77.5% Slakh. | T4+ | Apache 2.0 |

Ambos modelos aplican beat tracking (tempo map constante) para alinear las notas
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
