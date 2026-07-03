# Reference Demos — Audios oficiales de los proyectos

Para comparación cualitativa de nuestros outputs del benchmark con lo que los propios autores 
generaron con sus modelos bajo condiciones similares (o idénticas).

---

## ZETA (AudioLDM2) — `zeta_official/`

Fuente: https://hilamanor.github.io/AudioEditing/ (sección 1.1, texto-guiado)
Modelo: `cvssp/audioldm2-music`, método `ours` (DDPM edit-friendly inversion).
**NOTA**: los sources son tracks de MedleyDB (no los mismos que nuestros casos), pero el 
tipo de edición, los parámetros (tstart) y el backbone son idénticos.

| Archivo fuente | Archivo editado | Equivalente a nuestro caso |
|---|---|---|
| `source_hardrock_zeppelin.mp3` | `output_jazz_zeppelin_tstart100.mp3` | `case01_bach_jazz` (tstart=100, style→jazz) |
| `source_classical_vivaldi.mp3` | `output_arcade_vivaldi_tstart100.mp3` | `case05_bach_arcade` (tstart=130 en nuestro caso) |
| `source_jazz_modaljazz.mp3` | `output_country_modaljazz_tstart90.mp3` | `case02_bolero_country` (tstart=100, style→country) |

Prompt pattern oficial: "A recording of a <source genre>" → "A recording of a <target genre>"

---

## MelodyFlow — `melodyflow_official/`

Fuente: https://melodyflow.github.io/ (sección "Text-guided Music Editing", tabla principal)
Modelo: `facebook/melodyflow-t24-30secs`, método "our inv." (inversión propia del paper).
**NOTA**: sources son tracks de MusicCaps (no los mismos que nuestro benchmark). Los pares 
fuente→edición son diferentes a los nuestros, pero muestran la calidad máxima que el modelo
puede lograr en condiciones reportadas en el paper.

| Source (paper) | Target (paper) | Archivo source | Archivo editado |
|---|---|---|---|
| Synth electronic alternative | Oud Middle Eastern alternative | `source_0.mp3` | `melodyflow_ourinv_0.m4a` |
| Alternative rock, upbeat dance | Fun kids' tune, upbeat dance | `source_1.mp3` | `melodyflow_ourinv_1.m4a` |
| Soulful Latin pop | High-energy Latin rock anthem | `source_2.mp3` | `melodyflow_ourinv_2.m4a` |
| Up-tempo rock, sparkling guitars | Epic cinematic score, soaring strings | `source_3.mp3` | `melodyflow_ourinv_3.m4a` |
| Happy electric guitar rock | Happy synth-pop with pulsing synth | `source_4.mp3` | `melodyflow_ourinv_4.m4a` |
| Country rock, female vocals | Afrobeat, djembe drums | `source_5.mp3` | `melodyflow_ourinv_5.m4a` |
| Hipster mid-tempo rock | Acoustic-driven folk ballad | `source_6.mp3` | `melodyflow_ourinv_6.m4a` |
| Hip hop, heavy bounce | Classical tabla rhythms, Indian | `source_7.mp3` | `melodyflow_ourinv_7.m4a` |
| Indie rock, synth | Reggae, organ, wah-wah guitar | `source_8.mp3` | `melodyflow_ourinv_8.m4a` |

El más cercano a `case02_bolero_country`: `source_5/melodyflow_ourinv_5` (country rock → Afrobeat)
y `source_3/melodyflow_ourinv_3` (rock → cinematic strings, similar a case04_electronic_orchestral).

---

## MusicGen-melody — Sin audio de referencia descargable

El HF Space (https://huggingface.co/spaces/facebook/MusicGen) usa los prompts EXACTOS:
- `./assets/bach.mp3` + "An 80s driving pop song with heavy drums and synth pads" → nuestro `mgm_smoke_01`
- `./assets/bolero_ravel.mp3` + "A cheerful country song with acoustic guitars" → nuestro `case02_bolero_country`

Pero `cache_examples=False` en el código → no hay audio pre-generado descargable.
Para obtener el audio de referencia: ejecutar manualmente en el space https://huggingface.co/spaces/facebook/MusicGen
seleccionando `musicgen-stereo-melody` (el space usa stereo; nuestro script usa mono `musicgen-melody`).

---

## ACE-Step 1.5 — Sin audio de referencia disponible

El repo https://github.com/ace-step/ACE-Step-1.5 documenta los parámetros de la tarea cover 
pero no incluye audios de ejemplo pre-generados. El HF Space (https://huggingface.co/spaces/ACE-Step/ACE-Step)
permite generar interactivamente pero no tiene outputs cacheados.

---

## SAO 1.0 (Stable Audio Open) — Sin audio de referencia disponible

El model card (https://huggingface.co/stabilityai/stable-audio-open-1.0) incluye muestras 
de generación TEXT→AUDIO pero no de edición style transfer (init_audio). El gradio oficial
de stable-audio-tools solo demo genera desde texto.

---

## InspireMusic — Sin audio de referencia disponible

El README de https://github.com/FunAudioLLM/InspireMusic usa `audio_prompt.wav` como 
placeholder sin audio de ejemplo real. No hay demos descargables del modelo continuation.
