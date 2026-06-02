# MIDI Generator — Investigación de modelos

**Objetivo**: evaluar qué modelos encajan mejor en un plugin REAPER de generación MIDI con dos flujos:
1. Generación desde 0 (texto → MIDI)
2. Variaciones/arreglos sobre MIDI dado

**Hardware objetivo**: Mac Apple Silicon + Modal cloud fallback  
**Licencia**: uso personal/no comercial (cualquier licencia open source válida)

---

## Tabla comparativa de candidatos (actualizada 2026-06-02)

| Modelo | Flujo soportado | MIDI nativo | Mac (MPS/CPU) | Licencia | Multi-track | Madurez | Veredicto |
|---|---|---|---|---|---|---|---|
| **Text2midi** (AMAAI-Lab) | Text→MIDI | ✅ sí | ✅ MPS oficial | MIT | ✅ sí | ✅ AAAI 2025, HF | Evaluado: calidad 2/5 |
| **MIDI-LLM** (slSeanWU) | Text→MIDI | ✅ sí | ✅ MPS (bfloat16) | MIT | ✅ sí | ✅ NeurIPS AI4Music 2025 | **NUEVO candidato flujo 1** |
| **Aria** (EleutherAI) | Continuación piano | ✅ sí | ✅ MLX optimizado | Apache 2.0 | ❌ piano solo | ✅ ISMIR 2025, 60k h datos | **Candidato flujo 2 (piano)** |
| **Anticipatory MT** (Stanford) | Variaciones/infilling | ✅ sí | ⚠️ CPU (lento) | Apache 2.0 | ✅ sí | ✅ ICLR 2024, pesos abiertos | Evaluado: calidad 2/5 |
| **FIGARO** (ETH Zürich) | Fine-grained MIDI | ✅ sí | ✅ PyTorch CPU/MPS | ? | ✅ sí | ✅ ICLR 2022 | ⚠️ No acepta texto libre |
| **ChatMusician** (m-a-p) | Text→ABC notation | ❌ ABC (conv. necesaria) | ✅ MPS (LLaMA 2) | MIT | ✅ sí | ✅ 2024, HF Hub | ⚠️ Requiere ABC→MIDI extra |
| **MuseCoco** (Microsoft) | Text→MIDI con atributos | ✅ sí | ❌ fairseq pesado | MIT | ✅ sí | ✅ activo en muzic repo | **Script Modal listo** |
| **Amadeus** (AMAAI-Lab 2025) | Text→MIDI simbólico | ✅ sí | ? (por confirmar) | ? | ✅ sí | ⚠️ paper ago 2025, sin repo | Candidato v3 cuando salga |
| **Music Transformer** (Magenta) | Continuación piano | ✅ sí | ✅ TF2 CPU | Apache 2.0 | ❌ piano solo | ⚠️ legado (~2018) | Descartado (legado, mono) |
| **Foundation-1** (RoyalCities) | Text→Audio (MIDI post-hoc) | ❌ MIDI extraído | ❌ CUDA 7GB VRAM | Stability AI | ✅ multi-inst | ⚠️ requiere wrapper externo | Descartado (no MIDI nativo) |
| **MMM** (Multi-track Machine) | Infilling multi-track | ✅ sí | ✅ | ? | ✅ sí | ⚠️ antiguo (2020) | Descartado; superado por AMT |

---

## Evaluación detallada de candidatos elegidos

### Text2midi — Flujo 1: Generación desde 0

- **Repositorio**: https://github.com/AMAAI-Lab/Text2midi
- **Modelo HF**: `amaai-lab/text2midi`
- **Paper**: AAAI 2025, "Text2midi: Generating Symbolic Music from Captions"
- **Arquitectura**: encoder T5 (`google/flan-t5-base`) + decoder transformer autoregresivo
- **Dataset de entrenamiento**: SymphonyNet (pretraining) + MidiCaps (168k MIDIs con captions ricas: key, tempo, style, mood)
- **Soporte MPS**: explícito en `requirements-mac.txt` — primer ciudadano en Mac
- **Output**: archivos `.mid` vía `vocab_remi` (tokenización REMI)
- **Limitaciones conocidas**: pendiente verificación de si genera multi-track o solo piano; max 2000 tokens (~30-60s de música según densidad de notas)
- **Puntos fuertes**: MIT, MPS, end-to-end desde texto natural (no requiere atributos estructurados), HuggingFace Hub

#### Resultados PoC (ejecutado 2026-06-02, revisión manual 2026-06-02)

| Métrica | Valor |
|---|---|
| device | mps |
| tiempo carga (s) | 3.7 |
| tiempo inferencia (s) | 98.0 |
| RAM delta (MB) | 780 |
| MIDI válido | ✅ (6 pistas, 117 notas, instrumentos [0,1,28,33,35]) |
| Calidad subjetiva (0-5) | 2 — caótico pero guarda coherencia parcial |
| Notas | Multi-track confirmado (5 instrumentos). Resultado mejorable bajando temperatura (default 1.0) o aumentando max_len. Viable para el plugin. |

---

### Anticipatory Music Transformer — Flujo 2: Variaciones/arreglos

- **Repositorio**: https://github.com/jthickstun/anticipation
- **Modelo HF**: `stanford-crfm/music-medium-800k` (360M params)
- **Paper**: ICLR 2024, "Anticipatory Music Transformer"
- **Arquitectura**: transformer causal con anticipación de eventos de control intercalados
- **Dataset**: Lakh MIDI Dataset (~178k MIDIs, multi-instrumento, multi-track)
- **Capacidades únicas**:
  - **Infilling**: completa notas/compases dentro de un MIDI existente
  - **Continuation**: genera continuación temporal a partir de un segmento inicial
  - **Accompaniment**: genera acompañamiento dado una melodía como control fijo
- **Soporte MPS**: no optimizado (CUDA-first), funciona en CPU pero más lento
- **Output**: objetos `mido.MidiFile` via `events_to_midi()`
- **Puntos fuertes**: Apache 2.0, único diseñado para condicionamiento en MIDI de entrada, multi-track

#### Resultados PoC (ejecutado 2026-06-02, revisión manual 2026-06-02, script corregido 2026-06-02)

| Métrica | Valor |
|---|---|
| device | cpu (float32) |
| tiempo carga (s) | 0.8 |
| tiempo inferencia continuation (s) | 11.2 |
| tiempo inferencia accompaniment (s) | 8.2 (tras corrección) |
| RAM delta (MB) | 15 (pesos en caché HF) |
| MIDIs válidos | ✅ out_cont.mid · ✅ out_acc.mid (corregido, 2 pistas, instr [0,32]) |
| Calidad subjetiva continuation (0-5) | 2 — progresión armónica coherente pero sosa |
| Calidad subjetiva accompaniment (0-5) | pendiente de re-escucha tras corrección del script |
| Coherencia con melodía de entrada | pendiente de verificación en REAPER |
| Notas | Latencia excelente en CPU. Ver sección "Diagnóstico post-PoC" para detalles del bug corregido. |

---

### MIDI-LLM — Flujo 1 alternativo: Text→MIDI con LLM

- **Repositorio**: https://github.com/slSeanWU/MIDI-LLM
- **Modelo HF**: `slseanwu/MIDI-LLM_Llama-3.2-1B` (3.45 GB safetensors)
- **Paper**: NeurIPS AI4Music Workshop 2025, "MIDI-LLM: Adapting LLMs for Text-to-MIDI Generation"
- **Arquitectura**: Llama 3.2 1B con vocabulario extendido (+55k tokens MIDI de la librería `anticipation`)
- **Tokenización MIDI**: idéntica a AMT (eventos `anticipation`), rango [128256, 183282]
- **Dataset**: no publicado en paper (presumiblemente LakhMIDI / MidiCaps similares a Text2midi)
- **Soporte MPS**: bfloat16 funciona en MPS (confirmado PyTorch 2.0+ en M1+); script original hardcodea CUDA pero es adaptable
- **Ventajas vs Text2midi**: LLM decoder-only → mayor capacidad de seguir instrucciones textuales; 4-5× más rápido según benchmark del paper
- **Desventajas**: modelo 3.45 GB (vs ~900 MB Text2midi); primera descarga lenta

#### Resultados PoC (ejecutado 2026-06-02)

| Métrica | Jazz corto (1024 tok) | Pop (2046 tok) | Jazz largo (2046 tok) |
|---|---|---|---|
| device | mps | mps | mps |
| tiempo carga (s) | 346.6 (incl. descarga 3.4 GB) | **5.2** (modelo en caché) | **5.5** (modelo en caché) |
| tiempo inferencia (s) | 64.2 | 145.1 | 142.8 |
| RAM delta proceso (MB) | ~65 (pesos en GPU unified mem) | ~65 | ~65 |
| MIDI válido | ✅ (3 pistas, 339 notas, 8.4s) | ✅ (5 pistas, 682 notas, 28.5s) | ✅ (2 pistas, 681 notas, 24.2s) |
| Instrumentos | piano ×2, fretless bass (35) | piano, bass (33), vibraphone (11), strings (48), piano | piano ×2 (sin bass, sin batería real) |
| Calidad subjetiva (0-5) | — (demasiado corto para evaluar) | **3/5** — correcto, instrumentación coherente con el género | **2.5/5** — estructura razonable pero sin línea de contrabajo y batería atropellada |
| Notas | Demasiado corto, densidad muy alta. | Mejor resultado del modelo: pop mejor representado en el dataset de entrenamiento. | Falta contrabajo (ambas pistas asignadas a piano gm=0); batería poco orgánica. |

**Conclusión evaluación MIDI-LLM**: El modelo conoce mejor el género pop que el jazz. En pop logra instrumentación coherente (3/5); en jazz le falta la línea de bajo y la batería no suena natural (2.5/5). Mejor que Text2midi (2/5) en densidad y seguimiento del prompt, pero no suficiente para producción sin post-proceso.

**Comparativa directa con Text2midi** (mismo prompt "pop in C major"):

| | Text2midi | MIDI-LLM |
|---|---|---|
| Tokens generados | 512 | 2046 |
| Tiempo inferencia | 98s | 145s |
| Pistas | 6 | 5 |
| Notas totales | 117 | **682** |
| Duración MIDI | ~30s | 28.5s |
| Instrumentos | [0,1,28,33,35] | [0,33,11,48,0] |

---

### Aria (EleutherAI) — Flujo 2 alternativo: Continuación piano con MLX

- **Repositorio**: https://github.com/EleutherAI/aria
- **Modelo HF**: `loubb/aria-medium-base` / `loubb/aria-medium-gen`
- **Paper**: ISMIR 2025, "Scaling Self-Supervised Representation Learning for Symbolic Piano Performance"
- **Arquitectura**: Llama 3.2 1B entrenado en ~60k horas de piano MIDI expresivo
- **Soporte MPS**: ✅ MLX nativo para Apple Silicon (implementación optimizada, no solo PyTorch)
- **Capacidades**: continuación piano desde MIDI existente — reemplazaría AMT para flujo 2 piano
- **Limitación**: piano solo (no multi-track). Para acompañamiento multi-instrumento AMT sigue siendo la opción.
- **Nota**: no es text-to-zero, requiere MIDI de entrada como contexto

#### Resultados PoC (no ejecutado — depende de decisión sobre flujo 2)

Evaluar si la latencia de Aria en MLX es perceptiblemente mejor que AMT en CPU antes de añadir script.

---

### MuseCoco (Microsoft/muzic) — Flujo 1 en Modal GPU

- **Repositorio**: https://github.com/microsoft/muzic/tree/main/musecoco
- **Modelos HF**:
  - `XinXuNLPer/MuseCoco_text2attribute` — 1.35 GB (BERT fine-tuned para clasificación multi-label)
  - `XinXuNLPer/MuseCoco_attribute2music` — 14.5 GB (1B params, fairseq, linear attention)
- **Paper**: arXiv:2306.00110 (AAAI 2024), "MuseCoco: Generating Symbolic Music from Text"
- **Arquitectura**: dos etapas — (1) texto → atributos musicales estructurados, (2) atributos → tokens REMI2 → MIDI
- **Soporte Mac local**: ❌ — fairseq 0.10.2 + pytorch-fast-transformers requieren CUDA + Python 3.8
- **Ejecución**: Modal GPU (T4 16 GB, ~$0.009/generación, ~3 min)
- **Atributos controlables**: instrumento, ritmo, emoción, tonalidad, tempo, rango de pitch, compás, barras, firma de tiempo, artista, género
- **Ventaja clave**: control explícito multi-atributo; 947k MIDIs de entrenamiento; multi-track nativo (5-6 instrumentos)

#### Implementación Modal (research_musecoco_modal.py)
- **Image**: `pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel` + fairseq + pytorch-fast-transformers (compilado con NVCC)
- **Volume**: `musecoco-weights` — pesos persistentes (16 GB total, descarga única ~20-40 min)
- **Setup (una vez)**: `modal run research_musecoco_modal.py::setup_weights`
- **Inferencia**: `modal run research_musecoco_modal.py --prompt "..." --out out.mid`

#### Resultados PoC (pendiente ejecución)

| Métrica | Valor |
|---|---|
| device | Modal T4 GPU |
| tiempo total (s) | pendiente |
| pistas | pendiente |
| notas | pendiente |
| duración (s) | pendiente |
| instrumentos | pendiente |
| Calidad subjetiva (0-5) | pendiente |
| Notas | Control explícito de atributos → esperar mayor coherencia con el prompt |

---

## Diagnóstico post-PoC y correcciones de scripts

### Bug: `run_accompaniment` generaba solo REST tokens (resultado = melodía de entrada)

**Síntoma**: `out_acc.mid` (primera ejecución) sonaba exactamente igual que `fixtures/melody.mid`. Diagnóstico: 21 notas, todas instrumento 0 (piano). El modelo generaba únicamente REST tokens durante los 10 segundos → output = solo la melodía decodificada desde los controles de guía.

**Causa raíz (dos factores combinados)**:

1. **Fixture mono-canal**: `create_fixture_midi` generaba un MIDI type=0 con solo piano (canal 0, programa 0). Al llamar `extract_instruments(melody, [0])`, todos los eventos se convertían en controles y `remaining` quedaba vacío.

2. **Modelo arrancando en frío**: `generate(inputs=None, controls=piano)` — sin ningún historial de eventos (`inputs=[]`), AMT (entrenado en Lakh MIDI multi-track) generaba REST tokens para toda la ventana. El modelo nunca había visto en entrenamiento el estado de "cero eventos + solo controles".

   Verificación: el tqdm mostraba `950/1000` en 4.6s (~200 it/s, velocidad de REST tokens), vs ~150 it/s cuando genera notas reales.

**Correcciones aplicadas** (2026-06-02):

- `create_fixture_midi` → tipo 1 multi-track: piano (canal 0, prog 0) + bajo (canal 1, prog 32, progresión I-IV-V-I en C mayor, half-notes).
- `run_accompaniment` → `remaining, controls = extract_instruments(melody, [0])` en lugar de `_, controls = ...`. Se pasa `inputs=remaining` a `generate`. Ahora el bajo existente actúa como historial y el modelo genera notas reales de acompañamiento condicionadas por la melodía de piano.

**Resultado tras corrección**: `out_acc.mid` = 2 pistas, instrumentos [0, 32], 33 notas en 10s. El modelo genera notas reales.

### Limitación: Text2midi con temperatura 1.0 produce resultados irregulares

El resultado "caótico" no es un bug — es el comportamiento esperado con `temperature=1.0` (default). El modelo usa muestreo estocástico sin temperatura explícita en el script. Para producción se recomienda añadir control de temperatura (`--temperature 0.8`) y aumentar `max_len` (actualmente 512, máximo soportado 2000).

---

## Candidatos descartados — Justificación

### Foundation-1
Genera **audio** (fine-tune de Stable Audio Open), no MIDI nativo. El MIDI se obtiene como estimación post-hoc mediante transcripción de pitch sobre el audio generado — no son notas exactas editables. Requiere 7GB VRAM (CUDA), incompatible con Mac Apple Silicon para inferencia local. Pertenece a un plugin futuro de "AI sample/audio generation", no al plugin MIDI.

### Music Transformer (Magenta)
Modelo de 2018, piano solo (mono-track). La rama activa de Magenta (RealTime, 2025) genera audio, no MIDI. Superado por Text2midi y AMT en todos los criterios relevantes.

---

## Recomendación final

### Arquitectura del plugin: dos motores especializados

Igual que StemsSeparator usa **Demucs local + SAM via Modal**, el plugin MidiGenerator usará:

| Flujo | Motor | Ejecución | Invocación en plugin |
|---|---|---|---|
| Text → MIDI desde 0 | **Text2midi** | Local (MPS en Mac) | `python generate_text2midi.py --prompt "..." --out $TMPDIR/out.mid` |
| Variaciones/continuación/acompañamiento | **Anticipatory MT** | Local (CPU) o Modal si latencia inaceptable | `python generate_amt.py --mode continuation --input $ITEM_MIDI --out $TMPDIR/out.mid` |

### Plan B (si PoC revela problemas)

| Problema | Plan B |
|---|---|
| Text2midi no funciona en MPS | MuseCoco vía Modal (atributos explícitos + multi-track) |
| AMT demasiado lento en CPU (>60s) | AMT vía Modal (T4/A10G), igual que SAM en StemsSeparator |
| Text2midi genera solo piano | Evaluar Amadeus (2025, multi-track, SOTA) |

---

## Próximos pasos (siguiente iteración)

Una vez validados los PoCs y firmada la decisión:

1. Crear `plugins-and-extensions/MidiGenerator/` con la estructura completa del plugin:
   - `MidiGenerator.lua` — GUI principal (reutilizar `lib/gui.lua`, `lib/theme.lua`)
   - `Setup.lua` — wizard de primera ejecución (Python, uv, descarga de pesos HF)
   - `generate_text2midi.py` — backend CLI con protocolo `state|pct|msg`
   - `generate_amt.py` — backend CLI con protocolo `state|pct|msg`
   - `lib/common.lua` — copiar de StemsSeparator
   - `pyproject.toml` + `uv.lock` — dependencias de producción (subset de research)

2. Integración REAPER:
   - Lectura de item MIDI seleccionado: `reaper.GetMediaItemTake_Source` → `GetMediaSourceFileName()`
   - Escritura de resultado: `reaper.InsertMedia(midi_path, 0)` O `MIDI_InsertNote` vía helpers de `reaper-session-automation/lib/midi.lua`

3. Si AMT requiere Modal: adaptar `modal_sam_audio.py` de StemsSeparator como template

---

## Instrucciones para ejecutar los PoC

### Requisitos previos
```bash
# desde la carpeta research/
uv sync  # instala todas las dependencias (incluyendo AMT desde git)
```

### PoC 1a — Text2midi (referencia, ya ejecutado)
```bash
cd plugins-and-extensions/MidiGenerator/research
uv run research_text2midi.py --prompt "upbeat pop song in C major, 120 BPM, piano and strings" --out out_t2m.mid
```

### PoC 1c — MuseCoco vía Modal (nuevo candidato, pendiente evaluación)
```bash
# Primera vez: descarga ~16 GB de pesos al Volume de Modal (ejecutar una sola vez)
modal run research_musecoco_modal.py::setup_weights

# Inferencia en T4 GPU:
modal run research_musecoco_modal.py \
    --prompt "jazz piano trio, 120 BPM, C major, happy mood" \
    --out out_muse.mid

# Comparar con el mismo prompt usado en Text2midi y MIDI-LLM:
modal run research_musecoco_modal.py \
    --prompt "upbeat pop song in C major, 120 BPM, piano and strings" \
    --out out_muse_pop.mid
```

### PoC 1b — MIDI-LLM (nuevo candidato, pendiente evaluación)
```bash
cd plugins-and-extensions/MidiGenerator/research
# Primera ejecución: clona repo MIDI-LLM + descarga modelo 3.4 GB (una sola vez)
uv run research_midi_llm.py --prompt "upbeat jazz trio, 120 BPM, piano bass and drums" --out out_mllm.mid

# Probar con diferente prompt para comparar con Text2midi:
uv run research_midi_llm.py --prompt "upbeat pop song in C major, 120 BPM, piano and strings" --out out_mllm_pop.mid

# Si los tokens no pasan validación (excessive notes), bajar temperatura:
uv run research_midi_llm.py --prompt "..." --temperature 0.85 --out out_mllm.mid
```

### PoC 2 — Anticipatory Music Transformer (referencia, ya ejecutado)
```bash
uv run research_amt.py --mode both
```

### Validación en REAPER
1. Arrastrar cada `.mid` generado a una pista de REAPER
2. Escuchar y anotar calidad subjetiva (0-5) en las tablas de resultados arriba
3. Comparar `out_mllm.mid` vs `out_t2m.mid` con el mismo prompt
4. Para AMT: verificar que `out_acc.mid` suena coherente con `fixtures/melody.mid`

---

*Documento generado: 2026-06-01. PoC Text2midi+AMT: 2026-06-02. MIDI-LLM: 2026-06-02. MuseCoco Modal script: 2026-06-02. Pendiente: (1) escuchar out_mllm.mid en REAPER, (2) ejecutar setup_weights y PoC MuseCoco, (3) comparar calidad entre los tres.*
