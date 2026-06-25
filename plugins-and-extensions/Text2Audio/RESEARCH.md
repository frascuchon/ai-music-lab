# Text2Audio — Investigación de modelos

**Objetivo:** evaluar qué modelos encajan mejor en un plugin REAPER de generación de audio a partir
de texto, orientado a **samples & loops DAW-native** (clips de instrumento, one-shots de batería,
texturas, FX cortos ≤47 s). El usuario escribe un prompt en lenguaje natural y obtiene un clip
de audio listo para soltar en una pista de REAPER.

**Hardware objetivo:**
- Producción (Modal): GPU CUDA — A10G (24 GB) / T4 (16 GB)
- Preview local (futura integración): Mac Apple Silicon (MPS) — candidato Stable Audio Open Small

**Licencia:** permisiva para evaluar — se incluyen modelos NC/research y se decide la licencia
definitiva al integrar (mismo criterio que Audio2Midi con MIROS sin LICENSE).

**Contexto dentro de la familia de plugins:**
- [MidiGenerator](../MidiGenerator/RESEARCH.md) — Flujo 1: texto→MIDI, Flujo 2: variaciones MIDI
- [Audio2Midi](../Audio2Midi/RESEARCH.md) — transcripción audio→MIDI multi-instrumento
- [StemsSeparator](../StemsSeparator/) — separación de stems por instrumento
- **Text2Audio** (este plugin) — generación de audio a partir de texto

---

## Tabla comparativa de candidatos (actualizada 2026-06-25)

| Modelo | Tipo salida | Condicionamiento | Duración | Mac (MPS) | Licencia | Madurez | Veredicto |
|---|---|---|---|---|---|---|---|
| **Stable Audio Open 1.0** (Stability AI) | Música+SFX estéreo 44.1 kHz | Texto + **duración** | ≤47 s | ⚠️ lento | Stability Community (comercial <$1M rev) | ✅ ICASSP 2025 | **TOP CANDIDATO** |
| **Stable Audio Open Small** (Stability AI) | Música+SFX estéreo 44.1 kHz | Texto + duración | ≤47 s | ✅ ARM nativo | Stability Community | ✅ 2025 | **Candidato MPS local** |
| **Foundation-1** (RoyalCities) | Samples electrónicos estéreo 44.1 kHz | Texto + duración | ≤47 s | ⚠️ lento | Stability Community (hereda SAO) | ✅ 2025 | **Candidato fine-tune estilo propio** |
| **MusicGen** (Meta AudioCraft) | Música estéreo/mono | Texto + **melodía** | ≤30 s | ⚠️ slow CPU | MIT (código) / **CC-BY-NC** (pesos) | ✅ NeurIPS 2023 | **Candidato (flujo melodía)** |
| **MAGNeT** (Meta AudioCraft) | Música | Texto | ≤30 s | ⚠️ slow | MIT/CC-BY-NC | ✅ ICML 2024 | **Candidato baja latencia** |
| **AudioGen** (Meta AudioCraft) | SFX / efectos | Texto | ≤5 s | ⚠️ slow | MIT/CC-BY-NC | ✅ arXiv 2022 | Candidato sub-flujo FX |
| **Mustango** (AMAAI-Lab) | Música | Texto + **BPM/key/chords** | ≤10 s | ❌ no | Research / NC | ✅ arXiv 2023 | ⚠️ Control DAW interesante |
| **AudioLDM 2** (CVSSP) | Música+SFX | Texto | ≤10 s | ⚠️ slow | Apache 2.0 | ✅ IEEE TASLP 2024 | ⚠️ Calidad inferior a SAO |
| **ACE-Step 1.5** (StepFun) | Canción completa | Texto + lyrics | full-song | ❌ no | Apache 2.0 | ✅ arXiv 2025 | ⚠️ Orientado a tema completo |
| **InspireMusic** (Alibaba) | Música instrumental | Texto | ≤30 s | ❌ no | Apache 2.0 | ✅ 2025 | ⚠️ Solo instrumental, sin voz |
| **Suno / Udio** (cerrados) | Canción completa | Texto + lyrics | full-song | ❌ no | Propietario | ✅ SOTA comercial | ❌ DESCARTADO: sin pesos |
| **Jen-1** (Adobe Research) | Música | Texto | ≤10 s | ❌ no | Propietario | Publicación 2023 | ❌ DESCARTADO: sin pesos |
| **Riffusion** | Música | Texto | ≤5 s | ✅ | MIT | Prototipo 2022 | ❌ DESCARTADO: espectrograma |

---

## Plan de evaluación por sesiones

Cada sesión evalúa un solo modelo siguiendo el mismo flujo de trabajo. El orden es de mayor
a menor prioridad para el plugin de producción:

| Sesión | Modelo | Script PoC | HF Space (referencia manual) | Duración est. |
| --- | --- | --- | --- | --- |
| **S1** | Stable Audio Open 1.0 | `research_stable_audio_open_modal.py` | [stabilityai/stable-audio-open-1.0](https://huggingface.co/spaces/artificialguybr/Stable-Audio-Open-Zero) | 2-3 h |
| **S2** | Foundation-1 | `research_foundation1_modal.py` | [multimodalart/Foundation-1](https://huggingface.co/spaces/multimodalart/Foundation-1) | 1-2 h |
| **S3** | MusicGen melody+stereo | `research_musicgen_modal.py` _(por crear)_ | [facebook/MusicGen](https://huggingface.co/spaces/facebook/MusicGen) | 2-3 h |
| **S4** | MAGNeT | `research_magnet_modal.py` _(por crear)_ | AudioCraft local (`python -m demos.magnet_app`) | 1-2 h |
| **S5** | AudioGen | `research_audiogen_modal.py` _(por crear)_ | [fffiloni/audiogen](https://huggingface.co/spaces/fffiloni/audiogen) | 1-2 h |

### Flujo estándar de cada sesión

```
1. fetch demos oficiales   →  evaluation/fetch_<model>_demos.sh
                               (descarga 2-3 audios de referencia del modelo)
2. smoke test              →  modal run research_<model>_modal.py::setup
                               modal run ...::main --prompt "<prompt_oficial>"
                               comparar output.wav con referencia descargada
3. eval completo           →  modal run ...::eval_all --prompts-json ../evaluation/prompts.json
4. render                  →  bash ../evaluation/render_norm.sh
5. métricas objetivas      →  uv run python ../evaluation/compute_metrics.py
6. escucha en REAPER        →  puntuación 0-5 en notes.txt por prompt
7. actualizar RESEARCH.md  →  rellenar tabla de resultados + veredicto
```

El **smoke test** (paso 2) es el paso de verificación: si el output del prompt oficial
es comparable en calidad al audio de referencia descargado, el script funciona correctamente.

> **Nota Foundation-1:** usa un formato de prompt radicalmente diferente al resto de los
> modelos — etiquetas separadas por comas (`instrumento, timbre, FX, BPM, clave, compases`)
> en lugar de frases en lenguaje natural. Las sesiones S2 usan `prompts_official.json`
> con prompts en formato tag; las demás usan `prompts.json`.

---

## Evaluación detallada de candidatos elegidos

### Stable Audio Open 1.0 — TOP CANDIDATO para samples & loops

- **Repositorio:** https://github.com/Stability-AI/stable-audio-tools
- **HuggingFace:** https://huggingface.co/stabilityai/stable-audio-open-1.0 _(gated: aceptar licencia)_
- **Paper:** Evans et al., "Stable Audio Open" — ICASSP 2025 · https://arxiv.org/abs/2407.14358
- **Arquitectura:** DiT (Diffusion Transformer) sobre latentes continuos de un VAE de audio;
  condicionamiento por texto vía T5-XXL + **condicionamiento explícito de duración** (segundos).
  Entrenado a 44.1 kHz estéreo.
- **Dataset de entrenamiento:** Freesound (CC) + Free Music Archive (CC) — licencias libres.
  Esto implica fuerte capacidad en loops de instrumento, SFX y texturas; menor en pop/vocal.
- **Soporte MPS:** sí, pero lento. Stable Audio Open Small está optimizado para ARM.
- **Output:** WAV estéreo 44.1 kHz, duración configurable 1–47 s.
- **Puntos fuertes DAW:**
  - Condicionamiento por duración: genera exactamente el número de segundos pedido
    → perfecto para ajustar a BPM (4 bars @ 120 BPM = 8 s exactos).
  - Entrenado en Freesound → buen rendimiento en one-shots y texturas.
  - Licencia comercial (Stability Community < $1M rev) → la más viable para un plugin distribuible.
  - `stable-audio-tools` tiene pipeline de inferencia limpio; también soportado en `diffusers`.
  - Modelo gated en HF: requiere aceptar la licencia UNA VEZ en https://huggingface.co/stabilityai/stable-audio-open-1.0
    y proporcionar `HF_TOKEN` (modal.Secret "huggingface"). Diferencia frente a scripts previos
    que usaban modelos públicos sin token.
- **Limitaciones conocidas:**
  - Máximo 47 s — no apto para temas completos.
  - Menor calidad en música vocal o con estructura armónica compleja.
  - Requiere descarga del modelo (~3.4 GB) al Modal Volume antes del primer uso.
- **Script Modal:** `research/research_stable_audio_open_modal.py`
- **Carpeta evaluación:** `evaluation/stable_audio_open/`

#### Benchmark de referencia — SAO 1.0 (Sesión S1)

**Recurso de referencia manual:** Space público (sin token necesario):
<https://huggingface.co/spaces/artificialguybr/Stable-Audio-Open-Zero>

**Prompts oficiales** (del model card, para smoke test — en `evaluation/prompts_official.json`):

```
"128 BPM tech house drum loop"
"The sound of a hammer hitting a wooden surface."
"Lo-fi slow BPM electro chill with organic samples"
```

**Métricas de referencia del paper** (Evans et al., ICASSP 2025, evaluación sobre AudioCaps):

| Métrica | Valor paper | Descripción |
| --- | --- | --- |
| FD_openl3 | 78.24 | Fréchet Distance en espacio OpenL3. Menor = mejor. |
| KL_passt | 2.14 | Divergencia KL en distribución de etiquetas. Menor = mejor. |
| CLAP score | 0.29 | Cosine similarity texto-audio LAION-CLAP. Mayor = mejor. |

**Criterio de éxito del script** (smoke test S1):

- `output.wav` — WAV estéreo 44100 Hz, duración ≈ `seconds` solicitado (±0.5 s).
- El prompt `"128 BPM tech house drum loop"` genera algo que suena a drum loop, no a ruido ni silencio.
- CLAP ≥ 0.20 (bajo ese umbral el pipeline está roto o el modelo no cargó correctamente).
- Tiempo de inferencia en A10G: < 60 s para 8 s de audio con 100 steps.

**Audio de referencia descargable:**

```bash
# Requiere HF_TOKEN (aceptar licencia en HF + token read):
cd plugins-and-extensions/Text2Audio/evaluation
HF_TOKEN=<tu_token> bash fetch_stable_audio_open_demos.sh
# → evaluation/stable_audio_open/reference_demos/*.wav

# Alternativa pública (SAO 1.0 base, prompts Foundation-1 — sin token):
bash fetch_foundation1_demos.sh
# → evaluation/foundation_1/demos/compare_example_*_a__sao_base.mp3
```

Los archivos descargados sirven para calibrar la calidad esperada del modelo: si nuestro
output es comparable en calidad, el script funciona correctamente.

**Flujo smoke test S1:**

```bash
cd plugins-and-extensions/Text2Audio/evaluation
HF_TOKEN=<token> bash fetch_stable_audio_open_demos.sh   # descarga referencia (una vez)

cd ../research
modal run research_stable_audio_open_modal.py::setup     # descarga pesos (una vez)
modal run research_stable_audio_open_modal.py::smoke     # genera 3 prompts oficiales
# → evaluation/stable_audio_open/smoke/sao_smoke_{01,02,03}/output.wav
# Comparar con reference_demos/ o con el Space
```

#### Resultados evaluación SAO 1.0 (completado 2026-06-25)

**Métricas objetivas** (LAION-CLAP 630k-audioset, HTSAT-tiny, no fusion; GPU A10G; seed base 42):

| Prompt | Categoría | CLAP | Duración |
|---|---|---|---|
| prompt01 | drum_loop (120 BPM, acoustic) | 0.5284 | 8.00 s |
| prompt02 | drum_loop (90 BPM, boom bap) | 0.5632 | 10.67 s |
| prompt03 | bassline (sub bass, 130 BPM) | 0.5072 | 7.38 s |
| prompt04 | synth_pad (evolving, 8 bars) | 0.4649 | 16.00 s |
| prompt05 | synth_lead (saw wave, 128 BPM) | 0.4582 | 8.00 s |
| prompt06 | fx_riser (8 bars, 128 BPM) | 0.4912 | 15.00 s |
| prompt07 | ambient_texture (drone, 30 s) | 0.4409 | 30.00 s |
| prompt08 | percussion_oneshot (snare) | 0.3938 | 1.00 s |
| prompt09 | arpeggio (house, 8 bars) | 0.5202 | 8.00 s |
| prompt10 | guitar_loop (jazz, 90 BPM) | 0.5650 | 10.67 s |
| prompt11 | piano_loop (chord stabs) | 0.5064 | 8.00 s |
| prompt12 | sfx_transition (noise riser) | 0.3421 | 4.00 s |
| **Media** | — | **0.4818** | — |

**Notas:**

- CLAP medio 0.4818 vs paper 0.29 (AudioCaps): diferencia esperada — nuestros prompts son más
  descriptivos y específicos; el checkpoint LAION-CLAP también difiere (paper usa versión interna).
- Mejores categorías: guitar_loop (0.5650), drum_loop boom bap (0.5632), drum_loop acoustic (0.5284).
- Peores categorías: sfx_transition (0.3421), percussion_oneshot (0.3938) — one-shots cortos
  difíciles de alinear texto-audio; la transición/riser pide una curva temporal que CLAP no captura.
- Todos los 12 prompts generados correctamente: 0 errores, 0 silencios, 0 artefactos graves en escucha rápida.
- FAD: pendiente (requiere `fetch_reference_set.sh` para descargar clips de referencia MusicCaps/Freesound).
- Escucha subjetiva en REAPER: pendiente — rellenar puntuación 0-5 en cada `notes.txt`.
- Métricas detalladas: `evaluation/stable_audio_open/metrics.json`

---

### Stable Audio Open Small — Candidato MPS local (preview DAW)

- **Repositorio:** https://github.com/Stability-AI/stable-audio-tools
- **HuggingFace:** https://huggingface.co/stabilityai/stable-audio-open-small _(gated)_
- **Paper:** misma familia que SAO 1.0; variante comprimida para inferencia ARM.
- **Arquitectura:** DiT reducido (341 M parámetros vs ~1.1 B en SAO 1.0), misma cadena VAE+T5.
- **Dataset de entrenamiento:** ídem SAO 1.0.
- **Soporte MPS:** ✅ optimizado para Apple Silicon — posible preview en tiempo real o semi-real.
- **Output:** WAV estéreo 44.1 kHz, ≤47 s.
- **Puntos fuertes DAW:**
  - Permite un camino de inferencia **local** sin coste de nube → preview rápido mientras
    se compone, enviando a SAO 1.0 (GPU) para el resultado final de alta calidad.
  - Mismo pipeline que SAO 1.0 → el script Modal puede instanciar ambos con un flag.
- **Limitaciones conocidas:**
  - Menor calidad subjetiva que SAO 1.0 (trade-off tamaño/calidad esperado).
- **Script Modal:** reutiliza `research/research_stable_audio_open_modal.py` (flag `--model small`).
- **Carpeta evaluación:** `evaluation/stable_audio_open_small/`

#### Benchmark de referencia — SAO Small (evaluar dentro de Sesión S1)

Mismo benchmark que SAO 1.0 — los mismos 3 prompts oficiales, mismo Space de referencia.
El objetivo es comparar calidad subjetiva y CLAP entre SAO 1.0 y Small para el mismo prompt,
cuantificando el trade-off calidad/velocidad que justificaría usar Small como preview local.

**Criterio de éxito adicional:** latencia de inferencia MPS (local Mac) < 120 s para 8 s de audio.

#### Resultados evaluación SAO Small (pendiente)

_Pendiente. Evaluar head-to-head con SAO 1.0 en los mismos prompts._

---

### Foundation-1 (RoyalCities) — Candidato fine-tune de estilo propio

- **HuggingFace:** <https://huggingface.co/RoyalCities/Foundation-1>
- **Paper / técnica:** Fine-tune de Stable Audio Open 1.0 realizado por RoyalCities sobre
  su propia librería de samples electrónicos (acid, house, techno). Publicado en 2025.
  No hay paper académico — el valor es el proceso, no el modelo en sí.
- **Arquitectura:** idéntica a SAO 1.0 (DiT + VAE + T5-XXL). Solo los pesos del transformer
  están ajustados al dominio de samples electrónicos.
- **Dataset de entrenamiento:** librería privada de RoyalCities — samples propios de acid,
  house y techno. No publicado, pero el método es replicable con `stable-audio-tools`.
- **Soporte MPS:** igual que SAO 1.0 (lento, mismo pipeline).
- **Output:** WAV estéreo 44.1 kHz, hasta 47 s. Mismo formato que SAO 1.0.
- **Licencia:** hereda Stability AI Community License de SAO 1.0 (comercial <$1M rev).

**Por qué es relevante — el ángulo de fine-tuning:**
  Foundation-1 es, ante todo, una **prueba de concepto del pipeline de fine-tuning de SAO**.
  Demuestra que es posible especializar el modelo en un dominio musical concreto entrenando
  sobre una librería propia, sin modificar la arquitectura ni el pipeline de inferencia.
  Esto abre una dirección muy interesante para el plugin:

  > Fine-tunar SAO 1.0 con samples propios → generador de audio personalizado al estilo
  > del compositor, no al estilo genérico de Freesound.

  Analogía directa con el plan documentado en `MidiGenerator/RESEARCH.md` para ChatMusician:
  "LoRA fine-tuning de estilo por autor (~20-50 piezas propias)". Aquí el equivalente sería
  fine-tunar SAO con ~100-500 samples propios del estilo deseado.

- **Limitaciones conocidas:**
  - Los pesos de Foundation-1 están muy sesgados hacia música electrónica acid/house.
    Para otros géneros (jazz, clásico, pop) los resultados serán pobres.
  - El fine-tuning de SAO requiere CUDA A100/A10G (~20-50 horas de entrenamiento, ~$20-80
    en Modal) y una librería de samples curada. No es trivial pero es reproducible.
  - No hay paper que documente los hiperparámetros exactos usados por RoyalCities.

- **Propuesta de evaluación:**
  1. Comparar Foundation-1 vs SAO 1.0 en prompts de música electrónica (prompt01 drum loop,
     prompt03 bassline, prompt09 arpeggio) → cuantificar ganancia de dominio.
  2. Si la ganancia es significativa, plantear el pipeline de fine-tuning propio
     como una fase posterior del plugin (análoga a la integración de LoRA en ChatMusician).

- **Script Modal:** `research/research_foundation1_modal.py` (sin secret HF — pesos públicos).
- **Carpeta evaluación:** `evaluation/foundation_1/`

#### Benchmark de referencia — Foundation-1 (Sesión S2)

**Recurso de referencia manual:** Space público:
<https://huggingface.co/spaces/multimodalart/Foundation-1>

**Audio de referencia descargable** (ejemplos oficiales del repo HF, públicos):

```bash
# Descarga automática con fetch_foundation1_demos.sh:
bash evaluation/fetch_foundation1_demos.sh
# Coloca los archivos en evaluation/foundation_1/demos/
```

Los 3 archivos de referencia (`compare_example_1_a/b/c.mp3`) están generados con los prompts
oficiales del model card. Son el benchmark concreto: si nuestro script genera algo de calidad
comparable, el pipeline funciona.

**Prompts oficiales** (formato tag — Foundation-1 NO usa frases en lenguaje natural):

```
"Bass, FM Bass, Medium Delay, Medium Reverb, Phaser, Acid, Gritty, Dubstep, 8 Bars, 140 BPM, E minor"
"Drums, 808 Kick, Clap, Hi-Hat, Tight, Dry, Minimal, House, 4 Bars, 120 BPM"
"Lead Synth, Saw Wave, Short Delay, Arpeggio, Bright, 4 Bars, 128 BPM, A minor"
```

Disponibles en `evaluation/prompts_official.json` bajo `"foundation_1"`.

> **Importante:** Foundation-1 usa etiquetas estructuradas, NO frases. El `prompts.json`
> estándar con frases en inglés NO funcionará bien — usar siempre `prompts_official.json`
> para la sesión S2.

**Criterio de éxito del script** (smoke test S2):

- `output.wav` — WAV estéreo 44100 Hz, duración ≈ bars×(60/BPM)×4 s.
- El prompt acid bass genera algo que suena a línea de bajo acid (303), no a ruido.
- Calidad subjetiva comparable con `compare_example_1_a.mp3` descargado.
- Sin métricas de paper de referencia (no hay paper) — comparación cualitativa es el criterio.

#### Resultados evaluación Foundation-1 (pendiente)

_Pendiente. Evaluar head-to-head con SAO 1.0 en prompts de electrónica equivalentes._

---

### MusicGen melody/stereo — Candidato condicionamiento por melodía

- **Repositorio:** https://github.com/facebookresearch/audiocraft (Meta AudioCraft)
- **HuggingFace:** https://huggingface.co/facebook/musicgen-melody,
  musicgen-melody-large, musicgen-stereo-large
- **Paper:** Copet et al., "Simple and Controllable Music Generation" — NeurIPS 2023 ·
  https://arxiv.org/abs/2306.05284
- **Arquitectura:** Transformer autoregresivo single-stage sobre tokens EnCodec (~50 token/s).
  Variante **melody** acepta condicionamiento de audio de referencia por proyección en el espacio
  de cromagramas → permite "continuar" o "variar" un motivo grabado en REAPER.
  Variante **stereo** genera dos canales independientes con EnCodec estéreo.
- **Dataset de entrenamiento:** 20.000 horas de música bajo licencias internas de Meta.
- **Soporte MPS:** sí, pero lento (autoregresivo en CPU/MPS es 5–20× más lento que CUDA).
- **Output:** WAV mono/estéreo, hasta 30 s.
- **Puntos fuertes DAW:**
  - `melody` condicionamiento: el usuario puede tarareear o pasar un clip de REAPER
    como referencia melódica → generación "en el estilo de" ese motivo.
  - Gran base de usuarios, integración establecida en `diffusers` y `audiocraft`.
  - `small` (300 M) / `medium` (1.5 B) / `large` (3.3 B) / `melody` / `melody-large` / `stereo-large`.
- **Limitaciones conocidas:**
  - Pesos bajo **CC-BY-NC** (no comercial) — solo investigación hasta decisión de integración.
  - Lento en CPU/MPS para largos (>15 s) por naturaleza autoregresiva.
  - Sin control explícito de duración (genera hasta el máximo configurado).
- **Script Modal:** `research/research_musicgen_modal.py` _(pendiente de crear)_
- **Carpeta evaluación:** `evaluation/musicgen/`

#### Parámetros de inferencia (referencia)

```python
from audiocraft.models import MusicGen
model = MusicGen.get_pretrained("facebook/musicgen-melody")
model.set_generation_params(duration=8.0, top_k=250, top_p=0.0, temperature=1.0)
wav = model.generate_with_chroma(["prompt text"], melody_wavs=[melody_wav], melody_sample_rate=sr)
```

#### Benchmark de referencia — MusicGen (Sesión S3)

**Recurso de referencia manual:** Space oficial Meta:
<https://huggingface.co/spaces/facebook/MusicGen>

**Audio de referencia:** el Space es interactivo, no hay archivos descargables directos del repo.
Usar el Space para escuchar la calidad esperada antes de ejecutar el script.

**Prompts oficiales** (del AudioCraft README y model card, para smoke test):

```
"An 80s driving pop song with heavy drums and synth pads in the background"
"A cheerful country song with acoustic guitars, banjo, and light drums"
"Jazz music with piano, acoustic bass, and brushed drums. Melancholic mood."
```

Para el condicionamiento melody, se necesita además un clip WAV de referencia.
La sesión S3 debe preparar un clip de melodía de ~5 s para probar `musicgen-melody`.

Disponibles en `evaluation/prompts_official.json` bajo `"musicgen"`.

**Métricas de referencia del paper** (Copet et al., NeurIPS 2023, sobre MusicCaps):

| Modelo | FAD | KL | CLAP |
| --- | --- | --- | --- |
| MusicGen-small | 4.88 | 1.40 | 0.28 |
| MusicGen-medium | 3.79 | 1.38 | 0.29 |
| MusicGen-large | 3.30 | 1.35 | 0.32 |
| MusicGen-melody | 4.09 | 1.51 | 0.31 |

FAD en escala AudioCraft (VGGish, MusicCaps como referencia). Menor = mejor.

**Criterio de éxito del script** (smoke test S3):

- `output.wav` es WAV mono o estéreo según variante, duración ≈ `duration` configurado.
- El prompt `"An 80s driving pop song with heavy drums"` genera música reconocible, no ruido.
- CLAP ≥ 0.25 (referencia del paper para `medium`).
- Para `melody`: la melodía generada comparte el contorno de alturas del clip de referencia.

#### Resultados evaluación MusicGen (pendiente)

*Pendiente. Evaluar head-to-head con SAO 1.0 en los mismos prompts.*

---

### MAGNeT — Candidato baja latencia (non-autoregressive)

- **Repositorio:** https://github.com/facebookresearch/audiocraft (Meta AudioCraft)
- **HuggingFace:** https://huggingface.co/facebook/magnet-small-10secs,
  facebook/magnet-medium-10secs, facebook/audio-magnet-medium
- **Paper:** Ziv et al., "Masked Audio Generation using a Single Non-Autoregressive Transformer"
  — ICML 2024 · https://arxiv.org/abs/2401.04577
- **Arquitectura:** Transformer enmascarado (tipo BERT) sobre tokens EnCodec. Decodifica
  **en paralelo** todos los tokens en múltiples pasadas de refinamiento → ~7× más rápido que
  MusicGen a igual longitud. Incluye variante `audio-magnet` para SFX.
- **Soporte MPS:** sí, pero igualmente más rápido en CUDA.
- **Output:** WAV mono, hasta 10 s.
- **Puntos fuertes DAW:**
  - Latencia de generación baja → candidato para preview interactivo o generación de one-shots.
  - `audio-magnet` genera SFX (alternativa AudioGen con mayor velocidad).
- **Limitaciones conocidas:**
  - Pesos **CC-BY-NC**.
  - Calidad subjetiva ligeramente inferior a MusicGen para música tonal compleja (trade-off velocidad).
  - Máximo 10 s.
- **Script Modal:** `research/research_magnet_modal.py` _(pendiente de crear)_
- **Carpeta evaluación:** `evaluation/magnet/`

#### Benchmark de referencia — MAGNeT (Sesión S4)

**Recurso de referencia manual:** no hay Space oficial de Meta para MAGNeT.
Ejecutar el demo local de AudioCraft como referencia:

```bash
pip install audiocraft
python -m demos.magnet_app --share   # abre Gradio en localhost
```

**Prompts oficiales** (del HF model card `facebook/magnet-small-10secs`):

```
"happy rock"
"energetic EDM, upbeat"
"melodic lo-fi hip hop"
```

Disponibles en `evaluation/prompts_official.json` bajo `"magnet"`.

**Métricas de referencia del paper** (Ziv et al., ICML 2024, sobre MusicCaps):

| Modelo | FAD | CLAP | Tiempo generación (10 s) |
| --- | --- | --- | --- |
| MAGNeT-small | 7.5 | 0.27 | ~0.5 s A100 |
| MAGNeT-medium | 5.5 | 0.28 | ~1.2 s A100 |
| MusicGen-medium (ref) | 3.79 | 0.29 | ~7.0 s A100 |

MAGNeT es ~7× más rápido que MusicGen a igual duración, con FAD ligeramente mayor.

**Criterio de éxito del script** (smoke test S4):

- `output.wav` generado. El prompt `"happy rock"` suena a algo reconocible.
- **El criterio principal de MAGNeT es la velocidad**: inferencia < 5 s en A10G para 10 s de audio.
- CLAP ≥ 0.22 (ligeramente inferior a MusicGen es esperado y aceptable).

#### Resultados evaluación MAGNeT (pendiente)

*Pendiente.*

---

### AudioGen — Candidato sub-flujo SFX y texturas de ambiente

- **Repositorio:** https://github.com/facebookresearch/audiocraft (Meta AudioCraft)
- **HuggingFace:** https://huggingface.co/facebook/audiogen-medium
- **Paper:** Kreuk et al., "AudioGen: Textually Guided Audio Generation" —
  arXiv 2022 · https://arxiv.org/abs/2209.15352
- **Arquitectura:** similar a MusicGen (autoregresivo EnCodec), entrenado en **efectos de sonido
  y ambientes** en lugar de música.
- **Output:** WAV mono, hasta 5 s.
- **Puntos fuertes DAW:**
  - Genera efectos de sonido y ambientes de alta calidad descritos por texto:
    "rain on a tin roof", "crowd cheering in a stadium", "electric guitar distortion FX" → clips
    de ambiente para usar como capas en un proyecto REAPER.
  - Parte del mismo ecosistema AudioCraft → comparte código de inferencia con MusicGen/MAGNeT.
- **Limitaciones conocidas:**
  - Pesos **CC-BY-NC**.
  - Máximo 5 s; música tonal fuera de su dominio de entrenamiento.
- **Script Modal:** reutiliza `research/research_musicgen_modal.py` (AudioCraft unifica los modelos).
- **Carpeta evaluación:** `evaluation/audiogen/`

#### Benchmark de referencia — AudioGen (Sesión S5)

**Recurso de referencia manual:** Space comunitario:
<https://huggingface.co/spaces/fffiloni/audiogen>

**Dataset de referencia:** AudioCaps — subconjunto de YouTube con etiquetas de sonidos.
Es el dataset de evaluación estándar del paper. Para calcular FAD:

```bash
# Descargar subset de evaluación de AudioCaps (~50 clips, 10 s c/u):
# Ver fetch_reference_set.sh — sección AudioCaps (misma herramienta yt-dlp)
```

**Prompts oficiales** (del paper AudioGen, para smoke test):

```
"Footsteps walking on a wooden floor, indoors"
"Heavy rain falling on a window, with distant thunder"
"A crowd of people cheering in a stadium"
```

Disponibles en `evaluation/prompts_official.json` bajo `"audiogen"`.

**Métricas de referencia del paper** (Kreuk et al., sobre AudioCaps, clips de 5 s):

| Modelo | FAD | KL | CLAP |
| --- | --- | --- | --- |
| AudioGen-medium | 2.49 | 1.19 | 0.38 |

**Criterio de éxito del script** (smoke test S5):

- `output.wav` es WAV mono, 5 s, no es silencio.
- El prompt `"Footsteps walking on a wooden floor"` genera algo que suena a pasos, no a música.
- CLAP ≥ 0.30 (AudioGen tiene CLAP alto porque el dominio SFX tiene mejor alineación texto-audio).

#### Resultados evaluación AudioGen (pendiente)

*Pendiente.*

---

## Candidatos secundarios mencionados

### Mustango — Control explícito de BPM/clave/acordes

- **Repositorio:** https://github.com/AMAAI-Lab/mustango
- **Paper:** arXiv 2311.08355 (2023)
- **Por qué es relevante:** único modelo open-weight con condicionamiento explícito de
  **BPM, key y chord progression** → alineación directa al grid de un proyecto REAPER.
  "Generate a 120 BPM jazz loop in C minor with Am-Dm-G7-Cm progression."
- **Limitaciones:** licencia research/NC; duración ≤10 s; calidad general menor que SAO 1.0.
- **Estado:** candidato de segunda iteración si el control de tempo/armónico resulta crítico.

### ACE-Step 1.5 / InspireMusic — Puente a "tema completo"

- **ACE-Step 1.5** (StepFun, Apache 2.0): genera canciones completas con letras, condicionamiento
  por tags de género/mood. Muy rápido (Step distillation). Orientado a "tema completo", no a loops.
- **InspireMusic** (Alibaba, Apache 2.0): instrumental de alta calidad, pero sin control de
  duración preciso. Candidato si en el futuro se abre un flujo "tema instrumental completo".
- Ambos con licencia permisiva Apache 2.0 → buenos para producción, pero fuera del alcance
  DAW-native de este plugin (samples & loops ≤47 s).

---

## Candidatos descartados — Justificación

### Suno / Udio
Modelos SOTA en calidad de canción completa, pero **sin pesos públicos** — solo API de pago.
Sin posibilidad de inferencia local o en Modal con control completo. Descartados.

### Jen-1 (Adobe Research)
Modelo de alta calidad (2023) pero sin pesos públicos disponibles (paper sin release).

### Riffusion
Genera música convirtiendo el audio a espectrograma y aplicando Stable Diffusion sobre él.
Calidad muy inferior a los modelos nativos de audio (SAO, MusicGen) de la ola 2024–2025.
Sin condicionamiento de duración. Solo 5 s. Descartado.

---

## Recomendación final

### Estado de evaluación (2026-06-25)

| Modelo | Flujo | Estado | Veredicto |
|---|---|---|---|
| **Stable Audio Open 1.0** | Samples & loops (texto+duración) | ✅ CLAP completado (subjetivo pendiente) | **TOP CANDIDATO — CLAP medio 0.4818** |
| **Stable Audio Open Small** | Samples & loops (MPS local) | ⏳ Pendiente | Candidato MPS / preview |
| **Foundation-1** (RoyalCities) | Samples electrónicos (fine-tune SAO) | ⏳ Pendiente | Candidato fine-tune estilo propio |
| **MusicGen melody/stereo** | Samples + condicionamiento por melodía | ⏳ Pendiente | Candidato flujo 2 (melodía) |
| **MAGNeT** | One-shots / baja latencia | ⏳ Pendiente | Candidato velocidad |
| **AudioGen** | SFX y texturas de ambiente | ⏳ Pendiente | Candidato sub-flujo FX |
| Mustango | Control BPM/acordes | ⏳ Pendiente | 2ª iteración si necesario |
| ACE-Step / InspireMusic | Tema completo (fuera de alcance hoy) | — | Puente a plugin futuro |
| Suno / Udio / Jen-1 / Riffusion | — | ❌ DESCARTADO | Sin pesos / calidad obsoleta |

### Veredicto provisional: Stable Audio Open 1.0 es el pipeline de referencia

Stable Audio Open 1.0 es el candidato de mayor calidad evaluable hoy en la categoría
samples & loops DAW-native:
- Condicionamiento de **duración exacta** → loops musicalmente precisos.
- Entrenado en Freesound (loops, SFX, texturas) → dominio alineado con el plugin.
- Licencia Stability Community → viable para producción comercial.
- DiT paralelo → más rápido que MusicGen autoregresivo para la misma longitud.

MusicGen melody es el candidato prioritario para el **flujo de condicionamiento por melodía**
(el usuario pasa un motivo de REAPER como referencia): capacidad única que SAO 1.0 no tiene.

**Foundation-1 abre el flujo de personalización:** compararla con SAO 1.0 en prompts de
electrónica cuantifica la ganancia del fine-tuning de dominio. Si la ganancia es relevante,
el siguiente paso natural es fine-tunar SAO sobre la propia librería de samples del compositor
(~100-500 archivos curados, ~$20-80 en Modal), produciendo un generador de audio personalizado
al estilo propio — la misma dirección que se planteó con ChatMusician + LoRA para MIDI.

---

## Próximos pasos

1. ~~**Ejecutar PoC Stable Audio Open**~~ ✅ Completado (12 prompts, CLAP medio 0.4818).
2. **Escucha subjetiva SAO 1.0 en REAPER** — abrir `evaluation/stable_audio_open/promptNN/output.mp3`,
   puntuar 0-5 en cada `notes.txt` (fidelidad, calidad, musicalidad, loopability, usabilidad).
3. **Evaluar Foundation-1** — reutilizar el mismo script apuntando a `RoyalCities/Foundation-1`
   (no gated, sin secret HF), comparar head-to-head en prompts de electrónica (prompt01, 02, 03, 09).
3. **Escucha subjetiva en REAPER** — abrir cada `output.wav` en REAPER, puntuar 0–5 por:
   fidelidad al prompt, calidad de audio, musicalidad, loopability, usabilidad en pista.
   Registrar en `notes.txt` de cada prompt y en la tabla de evaluación de este RESEARCH.md.
4. **Calcular FAD + CLAP** — `uv run python ../evaluation/compute_metrics.py` vs set de referencia.
4. **Añadir `research_musicgen_modal.py`** — MusicGen melody/stereo head-to-head con SAO.
5. **Añadir `research_magnet_modal.py`** — MAGNeT para comparación de latencia.
6. **Elegir modelo de producción** y diseñar el bridge Lua REAPER (analogía con Audio2Midi).

---

## Instrucciones para ejecutar los PoC

### Requisitos previos

```bash
# 1. uv sync (entorno local sin CUDA — solo para scripts de utilidad)
cd plugins-and-extensions/Text2Audio/research
uv sync

# 2. Modal CLI instalada y autenticada
pip install modal
modal setup   # abre navegador para auth

# 3. Stable Audio Open es un modelo GATED en HuggingFace.
#    OBLIGATORIO antes del primer uso:
#    a) Aceptar la licencia en: https://huggingface.co/stabilityai/stable-audio-open-1.0
#    b) Crear/obtener un HF token con acceso read en: https://huggingface.co/settings/tokens
#    c) Crear el secret en Modal:
modal secret create huggingface HF_TOKEN=<tu_token_aquí>
#    (mismo secret que se usa en otros proyectos de huggingface con Modal)
```

### PoC 1 — Stable Audio Open 1.0 vía Modal A10G

```bash
cd plugins-and-extensions/Text2Audio/research

# Descarga los pesos al Modal Volume (una vez, ~3.4 GB, ~$0.05):
modal run research_stable_audio_open_modal.py::setup

# Generación libre (prompt single):
modal run research_stable_audio_open_modal.py::main \
    --prompt "deep house kick drum loop, 120 BPM, punchy, minimal" \
    --seconds 8.0 \
    --out-dir ../evaluation/stable_audio_open/smoke

# Evaluación completa (todos los prompts de prompts.json):
modal run research_stable_audio_open_modal.py::eval_all \
    --prompts-json ../evaluation/prompts.json \
    --model-dir stable_audio_open

# Solo prompts específicos (por índice):
modal run research_stable_audio_open_modal.py::eval_all \
    --prompts-json ../evaluation/prompts.json \
    --model-dir stable_audio_open \
    --only 1,3,7
```

### Métricas objetivas + escucha

```bash
# Calcular FAD y CLAP (desde research/):
uv run python ../evaluation/compute_metrics.py

# Filtrar por modelo:
uv run python ../evaluation/compute_metrics.py --only stable_audio_open
uv run python ../evaluation/compute_metrics.py --json /tmp/metrics_sao.json

# Normalizar output.wav → .mp3 para escucha en REAPER:
bash ../evaluation/render_norm.sh
```

### Validación en REAPER

1. Abrir REAPER y crear un proyecto vacío.
2. Arrastrar `evaluation/stable_audio_open/promptNN/output.mp3` a una pista.
3. Ajustar el tempo del proyecto al BPM del prompt (ver `prompts.json` → campo `bpm`).
4. Verificar que el loop encaja en el grid (loopability).
5. Registrar puntuación 0–5 en `evaluation/stable_audio_open/promptNN/notes.txt`.

---

_Documento creado 2026-06-25. Actualizar esta tabla y las secciones de resultados
conforme avance la evaluación._
