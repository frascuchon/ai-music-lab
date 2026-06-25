# Text2Audio — Evaluación de modelos

## Objetivo

Este directorio contiene los artefactos de evaluación de los modelos text-to-audio para el plugin
REAPER de generación de samples & loops. Sigue la misma convención de `Audio2Midi/evaluation/` y
`MidiGenerator/evaluation/`, adaptada a audio como output en lugar de MIDI.

## Diferencia clave respecto a Audio2Midi

Audio2Midi usa **mir_eval F1 contra ground truth MIDI 1:1** — hay una respuesta correcta para cada
audio. Text2Audio genera audio de forma **no determinista** a partir de texto: no hay un output
"correcto" único. Por eso las métricas son **distributivas y sin referencia directa**:

- **FAD (Fréchet Audio Distance)** — distancia estadística entre la distribución de los audios
  generados y un set de referencia de alta calidad. Calculado con `fadtk` (VGGish o MERT embeddings).
  _Menor es mejor._
- **CLAP score** — similitud coseno entre el embedding del prompt de texto y el embedding del audio
  generado, usando el modelo LAION-CLAP. Mide si el audio suena como describe el prompt.
  _Mayor es mejor._
- **Subjetivo en REAPER** (0–5) — escucha directa en el DAW. Dimensiones:
  - **Fidelidad al prompt** — ¿el audio suena como lo describe el texto?
  - **Calidad de audio** — ¿hay artefactos, distorsión, clipping?
  - **Musicalidad** — ¿la melodía/ritmo/armonía son coherentes?
  - **Loopability** — ¿el clip cierra limpiamente para usarlo en loop? (solo prompts rítmicos)
  - **Usabilidad** — ¿lo usarías en un proyecto real en REAPER?

## Estructura del directorio

```
evaluation/
├── README.md                         ← este archivo
├── prompts.json                      ← 12 prompts DAW curados, compartidos entre modelos
├── fetch_reference_set.sh            ← descarga set de referencia para FAD (~150 MB, gitignored)
├── compute_metrics.py                ← FAD + CLAP score por modelo/prompt
├── render_norm.sh                    ← output.wav → .mp3 normalizado (idempotente)
├── _reference/                       ← gitignored (cache binaria del set de referencia FAD)
├── stable_audio_open/                ← resultados de Stable Audio Open 1.0
│   ├── prompt01/
│   │   ├── output.wav                ← generado por research_stable_audio_open_modal.py
│   │   ├── output.mp3                ← renderizado por render_norm.sh
│   │   └── notes.txt                 ← observaciones + puntuación subjetiva
│   ├── prompt02/ … prompt12/
│   └── smoke/                        ← outputs de prueba libre (no parte de la eval formal)
├── stable_audio_open_small/          ← resultados de SAO Small (pendiente)
├── musicgen/                         ← resultados de MusicGen (pendiente)
├── magnet/                           ← resultados de MAGNeT (pendiente)
└── audiogen/                         ← resultados de AudioGen (pendiente)
```

## Cómo comparar modelos

Cada modelo usa los **mismos `prompts.json`** como input y escribe su salida en
`evaluation/<model>/promptNN/output.wav`. Esto permite:
- Comparación directa del mismo prompt entre modelos.
- Cálculo de FAD y CLAP sobre todos los modelos con un solo comando.
- Escucha side-by-side en REAPER (importar los output.mp3 de cada modelo en pistas paralelas).

## Plantilla notes.txt por prompt

```
Prompt NN — <category>: <primeras palabras del prompt>
Texto completo:  <prompt text>
BPM objetivo:    <bpm o N/A>
Duración:        <seconds> s (<bars> compases si aplica)
Modelo:          <nombre del modelo>
Fecha:           <YYYY-MM-DD>
GPU:             <A10G / T4 / etc.>

Resultado (output.wav):
  Tamaño:        <KB>
  Duration real: <s>

Métricas objetivas:
  CLAP score:    pendiente
  FAD:           pendiente (requiere fetch_reference_set.sh)

Métricas subjetivas (escucha en REAPER):
  Fidelidad al prompt:  /5
  Calidad de audio:     /5
  Musicalidad:          /5
  Loopability:          /5  (N/A si no es rítmico)
  Usabilidad:           /5
  Media:                /5

Observaciones:
  <texto libre>
```

## Flujo de trabajo completo

```bash
# 1. (Una vez) Aceptar licencia SAO y crear el secret de HF en Modal
#    Ver RESEARCH.md → "Requisitos previos"

# 2. Descargar pesos al Volume (una vez, ~3.4 GB):
cd research/
modal run research_stable_audio_open_modal.py::setup

# 3. Generar todos los prompts:
modal run research_stable_audio_open_modal.py::eval_all \
    --prompts-json ../evaluation/prompts.json \
    --model-dir stable_audio_open

# 4. Renderizar output.wav → .mp3 para escucha:
cd ../evaluation/
bash render_norm.sh

# 5. Calcular métricas objetivas:
#    (Opcional: primero descarga el set de referencia FAD)
bash fetch_reference_set.sh       # ~150 MB, gitignored
cd ../research/
uv run python ../evaluation/compute_metrics.py

# 6. Escucha en REAPER:
#    Abrir REAPER → arrastrar evaluation/<model>/promptNN/output.mp3 a una pista
#    Ajustar tempo del proyecto al BPM del prompt (ver prompts.json)
#    Rellenar notes.txt con puntuación subjetiva

# 7. Registrar resultados en RESEARCH.md
```
