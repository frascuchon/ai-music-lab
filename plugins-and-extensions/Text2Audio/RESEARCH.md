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

#### Resultados evaluación (pendiente)

*Pendiente de ejecución. Ver `evaluation/stable_audio_open/` y `evaluation/README.md`.*

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

#### Resultados evaluación (pendiente)

*Pendiente. Evaluar head-to-head con SAO 1.0 en los mismos prompts.*

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

#### Resultados evaluación (pendiente)

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

#### Resultados evaluación (pendiente)

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

#### Resultados evaluación (pendiente)

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
| **Stable Audio Open 1.0** | Samples & loops (texto+duración) | ⏳ Pendiente | **TOP CANDIDATO — evaluar primero** |
| **Stable Audio Open Small** | Samples & loops (MPS local) | ⏳ Pendiente | Candidato MPS / preview |
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

---

## Próximos pasos

1. **Ejecutar PoC Stable Audio Open** — `modal run research/research_stable_audio_open_modal.py::setup`
   y `::eval_all` → generar `output.wav` para todos los prompts del `evaluation/prompts.json`.
2. **Escucha subjetiva en REAPER** — abrir cada `output.wav` en REAPER, puntuar 0–5 por:
   fidelidad al prompt, calidad de audio, musicalidad, loopability, usabilidad en pista.
   Registrar en `notes.txt` de cada prompt y en la tabla de evaluación de este RESEARCH.md.
3. **Calcular FAD + CLAP** — `uv run python ../evaluation/compute_metrics.py` vs set de referencia.
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
