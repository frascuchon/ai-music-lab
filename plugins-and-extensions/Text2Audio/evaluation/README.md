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

## Archivos de prompts

| Archivo | Uso |
| --- | --- |
| `prompts.json` | 12 prompts DAW curados (frases en inglés). Usar para SAO 1.0, SAO Small, MusicGen, MAGNeT, AudioGen. |
| `prompts_official.json` | Prompts extraídos de los demos oficiales de cada modelo. Usar para **smoke test** al inicio de cada sesión. Foundation-1 usa SIEMPRE este archivo (formato de etiquetas, no frases). |

## Flujo estándar por sesión (un modelo = una sesión)

```bash
# ── PASO 0: preparar benchmark de referencia ──────────────────────────────

# Foundation-1 (único modelo con audio de referencia descargable):
bash fetch_foundation1_demos.sh        # → foundation_1/demos/*.mp3

# Para los demás: usar el HF Space del modelo (ver RESEARCH.md) como referencia manual.


# ── PASO 1: smoke test — verificar que el script funciona ─────────────────

cd research/
modal run research_<model>_modal.py::setup   # descarga pesos (una vez)

# Ejecutar 2-3 prompts oficiales (de prompts_official.json):
modal run research_<model>_modal.py::main \
    --prompt "<prompt_oficial_del_modelo>" \
    --seconds <segundos> \
    --out-dir ../evaluation/<model>/smoke

# Comparar output.wav con referencia:
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
