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
├── prompts_official.json             ← prompts oficiales de cada modelo (smoke tests)
├── fetch_stable_audio_open_demos.sh  ← descarga referencia SAO 1.0 (gated, requiere HF_TOKEN)
├── fetch_foundation1_demos.sh        ← descarga demos Foundation-1 (públicos, sin token)
├── fetch_reference_set.sh            ← descarga set de referencia para FAD (~150 MB, gitignored)
├── compute_metrics.py                ← FAD + CLAP score por modelo/prompt
├── render_norm.sh                    ← output.wav → .mp3 normalizado (idempotente, recursivo)
├── _reference/                       ← gitignored (cache binaria del set de referencia FAD)
├── stable_audio_open/                ← resultados de Stable Audio Open 1.0
│   ├── reference_demos/              ← gitignored (audio oficial descargado por fetch_stable_audio_open_demos.sh)
│   ├── smoke/                        ← benchmark de verificación (3 prompts oficiales del model card)
│   │   ├── sao_smoke_01/
│   │   │   ├── output.wav            ← generado por ::smoke entrypoint
│   │   │   ├── output.mp3            ← renderizado por render_norm.sh
│   │   │   └── notes.txt             ← observaciones + CLAP + puntuación subjetiva
│   │   ├── sao_smoke_02/
│   │   └── sao_smoke_03/
│   ├── prompt01/
│   │   ├── output.wav                ← generado por research_stable_audio_open_modal.py::eval_all
│   │   ├── output.mp3                ← renderizado por render_norm.sh
│   │   └── notes.txt                 ← observaciones + puntuación subjetiva
│   └── prompt02/ … prompt12/
├── stable_audio_open_small/          ← resultados de SAO Small (pendiente)
├── musicgen/                         ← resultados de MusicGen (pendiente)
├── magnet/                           ← resultados de MAGNeT (pendiente)
├── audiogen/                         ← resultados de AudioGen (pendiente)
├── prompts_edit.json                 ← benchmark audio2audio: sources + 12 casos de edición/continuación
├── fetch_edit_sources.sh             ← descarga los 5 audios fuente compartidos (gitignored)
├── compute_metrics_edit.py           ← CLAP_text + CLAP_audio + chroma_corr por modelo/caso
└── edit/                             ← subtarea edición/versionado audio→audio
    ├── source_audio/                 ← gitignored (generado por fetch_edit_sources.sh)
    │   ├── bach.mp3
    │   ├── bolero_ravel.mp3
    │   ├── electronic.mp3
    │   ├── beatbox_loop_90bpm.wav
    │   └── sao_guitar_loop.wav
    ├── metrics_edit.json             ← generado por compute_metrics_edit.py
    ├── acestep15/                    ← resultados ACE-Step 1.5 (sesión E1)
    │   ├── smoke/
    │   │   ├── ace_smoke_01_source/output.wav
    │   │   └── ace_smoke_02_cover/output.wav
    │   ├── case01_bach_jazz/output.wav
    │   └── case02_bolero_country/output.wav  … (10 casos)
    ├── sao_a2a/                      ← resultados SAO 1.0 style transfer (sesión E2)
    ├── musicgen_melody/              ← resultados MusicGen-melody (sesión E3)
    ├── melodyflow/                   ← resultados MelodyFlow (sesión E4)
    ├── zeta_audioldm2/               ← resultados ZETA sobre AudioLDM2 (sesión E5)
    └── inspiremusic/                 ← resultados InspireMusic continuación (sesión E6)
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

## Archivos de prompts

| Archivo | Uso |
| --- | --- |
| `prompts.json` | 12 prompts DAW curados (frases en inglés). Usar para SAO 1.0, SAO Small, MusicGen, MAGNeT, AudioGen. |
| `prompts_official.json` | Prompts extraídos de los demos oficiales de cada modelo. Usar para **smoke test** al inicio de cada sesión. Foundation-1 usa SIEMPRE este archivo (formato de etiquetas, no frases). |
| `prompts_edit.json` | Benchmark de **edición/versionado audio→audio** (12 casos, 5 fuentes). Lo consumen los 6 scripts `research_*_edit_modal.py` y `compute_metrics_edit.py`. |

## Flujo estándar por sesión de edición (subtarea audio→audio)

```bash
# ── PASO 0: audios fuente compartidos (una vez) ───────────────────────────
cd plugins-and-extensions/Text2Audio/evaluation
bash fetch_edit_sources.sh    # → edit/source_audio/ (5 archivos, gitignored)


# ── PASO 1: smoke oficial (condiciones del paper replicadas) ─────────────
cd ../research
modal run research_<model>_edit_modal.py::setup   # descarga pesos (una vez)
modal run research_<model>_edit_modal.py::smoke   # → edit/<model>/smoke/*/output.wav
# Verificar que el smoke suena coherente con lo documentado en cada script.


# ── PASO 2: benchmark completo ─────────────────────────────────────────────
modal run research_<model>_edit_modal.py::eval_all
# → edit/<model>/<case_id>/output.wav
# Solo los casos soportados (cada script filtra por categoría).


# ── PASO 3: render + métricas ─────────────────────────────────────────────
bash render_norm.sh                                   # recursivo, cubre edit/ automáticamente
uv run python compute_metrics_edit.py                 # CLAP_text + CLAP_audio + chroma
uv run python compute_metrics_edit.py --only <model>  # solo un modelo
uv run python compute_metrics_edit.py --no-chroma     # solo CLAP (sin librosa)


# ── PASO 4: escucha en REAPER ─────────────────────────────────────────────
# 1. Abrir REAPER, importar source y output en pistas paralelas:
#    - edit/source_audio/<src> → pista 1 (referencia)
#    - edit/<model>/<case_id>/output.mp3 → pista 2 (versión editada)
# 2. Puntuar Adherencia a la edición / Preservación del tema / Calidad / Usabilidad (0-5)
# 3. Rellenar notes.txt (plantilla en RESEARCH.md → sección E1–E6)
```

## Flujo estándar por sesión (un modelo = una sesión)

```bash
# ── PASO 0: preparar benchmark de referencia ──────────────────────────────

# SAO 1.0 (gated — requiere HF_TOKEN; los pares *_sao_base.mp3 de Foundation-1 son alternativa pública):
HF_TOKEN=<token> bash fetch_stable_audio_open_demos.sh  # → stable_audio_open/reference_demos/

# Foundation-1 (público, sin token — incluye pares A/B con SAO 1.0 base):
bash fetch_foundation1_demos.sh        # → foundation_1/demos/*.mp3

# Para los demás (MusicGen, MAGNeT, AudioGen): usar el HF Space como referencia manual.


# ── PASO 1: smoke test — verificar que el script funciona ─────────────────

cd research/
modal run research_<model>_modal.py::setup   # descarga pesos (una vez)

# SAO 1.0: entrypoint dedicado (lee prompts_official.json automáticamente):
modal run research_stable_audio_open_modal.py::smoke
# → evaluation/stable_audio_open/smoke/sao_smoke_{01,02,03}/output.wav

# Foundation-1 y demás: smoke manual con los prompts oficiales:
modal run research_<model>_modal.py::main \
    --prompt "<prompt_oficial_del_modelo>" \
    --seconds <segundos> \
    --out-dir ../evaluation/<model>/smoke

# Comparar output.wav con referencia:
# - SAO 1.0: comparar con stable_audio_open/reference_demos/ (o con el HF Space)
# - Foundation-1: comparar con foundation_1/demos/*_foundation1.mp3
# - Resto: abrir el HF Space del modelo, generar el mismo prompt, comparar cualitativamente
# Si el output es razonable → script OK → continuar al paso 2


# ── PASO 2: evaluación completa ───────────────────────────────────────────

# SAO 1.0, SAO Small, MusicGen, MAGNeT, AudioGen → prompts.json
modal run research_<model>_modal.py::eval_all \
    --prompts-json ../evaluation/prompts.json \
    --model-dir <model>

# Foundation-1 → prompts_official.json (formato tag)
modal run research_foundation1_modal.py::eval_all \
    --prompts-json ../evaluation/prompts_official.json \
    --model-section foundation_1


# ── PASO 3: render + métricas ─────────────────────────────────────────────

bash render_norm.sh                                    # output.wav → .mp3
bash fetch_reference_set.sh                            # set de referencia FAD (~150 MB, gitignored)
uv run python compute_metrics.py                       # FAD + CLAP
uv run python compute_metrics.py --only <model>        # solo un modelo
uv run python compute_metrics.py --no-fad              # solo CLAP (sin _reference/)


# ── PASO 4: escucha en REAPER ─────────────────────────────────────────────

# 1. Abrir REAPER, crear proyecto vacío
# 2. Arrastrar evaluation/<model>/promptNN/output.mp3 a una pista
# 3. Ajustar tempo del proyecto al BPM del prompt (ver prompts.json → campo "bpm")
# 4. Comparar side-by-side con otros modelos en pistas paralelas
# 5. Rellenar notes.txt de cada prompt con la puntuación subjetiva


# ── PASO 5: registrar resultados ──────────────────────────────────────────

# Actualizar RESEARCH.md:
# - Rellenar "#### Resultados evaluación <Modelo>"
# - Actualizar la tabla "Estado de evaluación" con el veredicto
# - Actualizar "Tabla comparativa de candidatos" si hay nuevo ELEGIDO/DESCARTADO
```
