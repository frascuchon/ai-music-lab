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
| **Text2midi** (AMAAI-Lab) | Text→MIDI | ✅ sí | ✅ MPS oficial | MIT | ✅ sí | ✅ AAAI 2025, HF | ❌ DESCARTADO: calidad 2/5, baseline académico |
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

#### Resultados evaluación completa (2026-06-12) ❌ DESCARTADO

Benchmark de 7 tests ejecutado en Modal A10G (CUDA float32, temperature=0.9, max_len=2000).
Referencias oficiales extraídas del demo branch del repositorio (output_A/B/C/D/E/F/4.mid).

| Test | Prompt | Seguimiento del prompt | Calidad |
|---|---|---|---|
| test1 | "A sad pop song with a strong piano presence." | ✅ parcial | 2/5 — Piano presente, progresión mecánica |
| test2 | "A rock song with strong drums and electric guitar." | ⚠️ débil | 1.5/5 — Sin drums reconocibles |
| test3 | "A soft love song on piano." | ❌ ignorado | 1.5/5 — Genera sax + armónica + bajo, no "solo piano" |
| test4 | "...trance... 138 BPM... A minor..." | ⚠️ parcial | 2/5 — Instrumentación aproximada |
| test5 | "A cheerful christmas song suitable for children." | ✅ aceptable | 2.5/5 — El más exitoso |
| test6 | "...C minor... brass... sax... 124 BPM... C7/E Eb6 Bbm6..." | ⚠️ parcial | 2/5 — Prompt detallado no produce pieza más larga |
| test7 | "A heavy metal song with strong drums and guitar." | ❌ débil | 1.5/5 — Sin agresividad del género |

**Limitaciones estructurales confirmadas**:
- 2000 tokens REMI = límite duro de ~30-60s independientemente del prompt
- Atributos específicos (BPM, tonalidad, acordes) raramente se reflejan con precisión
- Multi-track autoregresivo sin coherencia garantizada entre pistas
- Las referencias oficiales del paper también son decepcionantes — no es un problema de nuestro pipeline

**Contexto**: text2midi es un baseline académico cuya contribución real es el dataset MidiCaps
(168k pares MIDI-caption). El modelo existe para demostrar que el dataset es útil, no como
herramienta de producción. El listening study del paper reporta 4.62/7 en Musical Quality.

**Nota sobre ChatMusician**: arquitectónicamente más sólido (fine-tune LLaMA + ABC notation,
hereda comprensión semántica LLM real), pero ABC notation limita la polifonía y el multi-track —
no es el formato adecuado para integración DAW. El problema texto→MIDI multi-track coherente
sigue abierto a fecha de junio 2026.

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

### Estado de evaluación (actualizado 2026-06-12)

| Modelo | Flujo | Estado | Veredicto |
|---|---|---|---|
| **Text2midi** | Text→MIDI | ✅ Evaluado (7 tests, CUDA A10G) | ❌ **DESCARTADO** — calidad insuficiente para producción |
| **MIDI-LLM** | Text→MIDI | ✅ Evaluado (12 comparisons, CUDA A10G) | ⚠️ Mejor que text2midi pero insuficiente para producción sin postproceso |
| **Anticipatory MT** | Variaciones/acompañamiento | ✅ Evaluado (3 tests) | ⚠️ Viable para flujo 2, latencia aceptable en CPU |
| **MuseCoco** | Text→MIDI atributos | ✅ Evaluado (3 tests, Modal) | ⚠️ Control explícito pero interfaz rígida (atributos, no texto libre) |
| **ChatMusician** | Text→ABC notation | ✅ Evaluado (12 tests, CUDA A10G) | ⚠️ **CERRADO** — 7.5/12 (62%), mono-staff; nota fine-tuning de autor pendiente |
| **Amadeus** | Text→MIDI | ✅ Evaluado (8 tests, 16 MIDIs, CUDA A10G) | ⚠️ Instrumentación superior a text2midi/MIDI-LLM pero duración inconsistente; modelo-S limitado |

### Situación actual del flujo 1 (Text→MIDI)

Ningún modelo evaluado hasta la fecha alcanza calidad suficiente para uso productivo en un plugin
de REAPER. Tanto text2midi como MIDI-LLM son proofs-of-concept académicos con limitaciones
estructurales (token limit, seguimiento de atributos débil, coherencia multi-track insuficiente).

El espacio más prometedor para texto→MIDI multi-track coherente en 2026 está sin resolver.
Las alternativas son:
1. **Audio-first + transcripción**: usar MusicGen/Suno/Udio (audio generativo de alta calidad)
   y extraer MIDI via transcripción — introduce ruido pero la calidad de partida es superior.
2. **Esperar Amadeus** u otros modelos de nueva generación (2025-2026) con arquitecturas más grandes.
3. **Restringir el alcance del flujo 1**: en lugar de "texto libre → canción completa",
   generar motivos/frases cortas de un instrumento específico donde los modelos existentes
   son más fiables.

### Flujo 2 (variaciones/acompañamiento) — sigue siendo viable

**Anticipatory MT** sigue siendo la opción para flujo 2. La evaluación (test1-3) confirma
que continuation y accompaniment funcionan con calidad aceptable (2.5/5) para uso exploratorio.

### ChatMusician — Flujo 1 alternativo via ABC notation ✅ EVALUADO (2026-06-12)

- **Repositorio**: https://github.com/hf-lin/ChatMusician
- **Modelo HF**: `m-a-p/ChatMusician` (LLaMA 2 7B, ~13 GB safetensors fp16)
- **Paper**: ISMIR 2024, "ChatMusician: Understanding and Generating Music Intrinsically with LLM"
- **Arquitectura**: continual pretraining + SFT de LLaMA 2 7B sobre corpus MusicPile (4B tokens de ABC notation + teoría musical)
- **Output**: ABC notation → conversión a MIDI via `abc2midi` (herramienta `abcmidi`)
- **Capacidades únicas**: condicionamiento por acordes, formas musicales, motivos, armonización de melodías
- **Limitación estructural**: ABC notation es mono-staff (una sola pista); no multi-track

**Benchmark ejecutado** (12 tests, Modal A10G fp16, 2 outputs por test):

| Test | Categoría | Resultado | Calidad |
|------|-----------|-----------|---------|
| test01 | chord Am-F-C-G | ✅ 2/2 MIDIs | 2.5/5 — Am/G/Em (F ignorado) |
| test02 | chord Dm-C | ✅ 2/2 MIDIs | 3/5 — Dm/C/Gm/A7 (coherente) |
| test03 | chord D-G-C-B | ✅ 2/2 MIDIs | 3/5 — D/G/C (B ignorado, Dmix) |
| test04 | form Binary v1 | ✅ 2/2 MIDIs | 3.5/5 — Mejor output: binario real con :: |
| test05 | form Ternary | ❌ 0/2 | Modo ensayo (persistente a temp=0.5) |
| test06 | form Binary v2 | ✅ 2/2 MIDIs | 3/5 — Requirió temp=0.5 |
| test07 | motif AB + motivo | ⚠️ 1/2 MIDIs | 2/5 — Motivo reconocible, sin contraste A/B |
| test08 | motif Bin+motivo 3/4 | ✅ 2/2 MIDIs | 3/5 — Motivo presente, armonía densa |
| test09 | motif Only One Section | ❌ 0/2 | "Only One Section" → respuesta de décadas |
| test10 | harmonize G major | ❌ 0/2 | Output mínimo (1 barra sin X:) |
| test11 | harmonize D major | ✅ 2/2 MIDIs | 3/5 — v1: melodía original + D/Em/A7 |
| test12 | harmonize G+chords | ❌ 0/2 | Fragmento continuado del input |

**Score global: 7.5/12 (62.5%)**

**Calidad media de los exitosos: 2.9/5** — mejor que text2midi (1.5-2/5) para tareas estructuradas.

#### Hallazgos clave

1. **Chord conditioning funciona bien**: el modelo sigue los acordes del prompt con buena coherencia armónica. Los acordes generados son siempre musicalmente válidos aunque no idénticos al prompt.

2. **Form conditioning: sensible al phrasing**: "Binary + Verse/Chorus" con el verbo correcto genera piezas binarias reales (test04 ✅). Variantes abstractas o "Ternary" disparan modo ensayo — el MusicPile contiene muchos artículos de teoría musical y con `temperature=0.2` el modelo prefiere respuestas académicas.

3. **Motif conditioning: mixto**: el motivo musical es reproducible en algunas condiciones (test08 ✅) pero el modelo no garantiza contraste de secciones. "Only One Section" como modificador confunde el model (test09 ❌).

4. **Harmonization: funcionó parcialmente**: test11 v1 muestra armonización genuina (melodía preservada + acordes añadidos). test10 y test12 fallan — posiblemente la tonalidad G con pickup bar o los inputs más largos exceden la atención efectiva.

5. **temperature=0.2 demasiado baja para prompts abstractos**: subir a 0.5 convierte test06 de fallo en éxito. La configuración oficial del model card es conservadora para tareas creativas.

6. **ABC es mono-staff**: los outputs son melodía + anotaciones de acordes en una sola línea. No hay multi-track MIDI. Para un plugin DAW multi-instrumento, esto es una limitación fundamental.

#### Veredicto: ⚠️ CERRADO — CANDIDATO LIMITADO

ChatMusician supera a text2midi y MIDI-LLM en seguimiento de estructura musical (acordes, forma) pero su salida mono-staff y la sensibilidad al phrasing lo hacen inadecuado para generación de canción completa multi-track en un contexto DAW. **Caso de uso real: generación de melodías/frases cortas con armonización condicionada** (lead sheet style), no canciones completas.

**Prompt template** (verbatim de `model/infer/predict.py`):
```
Human: {instruction} </s> Assistant:
```

**GenerationConfig recomendado** (ajustado tras evaluación):
```python
temperature=0.5,  # 0.2 original es demasiado conservador para prompts abstractos
top_k=40, top_p=0.9, do_sample=True,
num_beams=1, repetition_penalty=1.1, max_new_tokens=1536
```

#### Nota pendiente: fine-tuning de estilo por autor

ChatMusician-Base admite LoRA/QLoRA con muy pocas piezas propias (~20-50) por una razón estructural: el modelo ya domina la sintaxis ABC por completo, así que el fine-tuning solo necesita desplazar la distribución hacia los patrones estilísticos del autor — tonalidades frecuentes, contornos melódicos, densidad armónica, patrones rítmicos. No requiere crear un dataset de pares instrucción/respuesta: los propios ficheros ABC del autor son el texto de entrenamiento (objetivo: causal LM).

El pipeline ya está casi completo: `tools/midi_abc.py` convierte las piezas del autor de MIDI a ABC, el script Modal del fine-tuning añadiría LoRA sobre `m-a-p/ChatMusician-Base`, y la inferencia usaría `research_chatmusician_modal.py` sin cambios. Coste estimado: < $1 en Modal A10G para 3-5 epochs sobre 50 piezas.

Lo que capturaría: estilo superficial (tonalidad, ritmo, fraseo). Lo que no: lógica compositiva profunda ni multi-track. Suficiente para generar "ideas en el estilo de X" como punto de partida en el DAW.

**Para explorar en una iteración futura**: `research_chatmusician_lora.py` — script Modal con QLoRA (rank=8, 4-bit), entrenado sobre las piezas del autor convertidas a ABC.

---

### Amadeus — Flujo 1 candidato: Text→MIDI con difusión bidireccional

- **Repositorio**: https://github.com/lingyu123-su/Amadeus
- **Modelo HF**: `longyu1315/Amadeus-S` (~2.5 GB checkpoint + vocab JSON)
- **Paper**: arXiv 2508.20665 (ago 2025), "Amadeus: Autoregressive Model with Bidirectional Attribute Modelling for Symbolic Music"
- **Arquitectura**: transformer autoregresivo para secuencias de notas + decoder de difusión bidireccional para atributos intra-nota (NB encoding). T5-base como encoder de texto.
  - **NB encoding**: los atributos de cada nota (pitch, duración, velocidad, instrumento, etc.) se modelan como un conjunto no ordenado — contrastando con el encoding secuencial de REMI/CP
  - **MLSDES**: contrastive learning para mejorar la calidad de representaciones intermedias
  - **CIEM**: módulo de atención para enriquecer el vector de latente de nota
- **Dataset de entrenamiento**: LakhALLFined (subconjunto filtrado del Lakh MIDI Dataset)
- **Ventaja clave**: 4× más rápido que modelos puramente autoregresivos (según el paper) con mejor calidad
- **Soporte GPU**: CUDA (A10G cómodo con ~2.5 GB del checkpoint)
- **Parámetros de generación** (verbatim de `demo/Amadeus_app_EN.py`):
  ```python
  threshold=0.99, temperature=1.25, generation_length=1024, sampling_method='top_p'
  text_encoder='google/flan-t5-base'  # Amadeus-S
  ```
- **Limitaciones conocidas**: el modelo liberado (Amadeus-S) podría no ser el más robusto del paper; los modelos M y L no están disponibles en HuggingFace Hub aún

#### Implementación Modal (`research_amadeus_modal.py`)
- **Image**: `debian_slim(python=3.10)` + `apt install fluidsynth fluid-soundfont-gm` + `git clone https://github.com/lingyu123-su/Amadeus /amadeus` + dependencias
- **Patch**: `midi2audio.py` parchado en build para apuntar a `/usr/share/sounds/sf2/FluidR3_GM.sf2` (soundfont de sistema) en lugar del path hardcodeado del autor
- **Volume**: `amadeus-weights` — checkpoint + vocab + T5 encoder cacheados
- **Setup (una vez)**: `modal run research/research_amadeus_modal.py::setup`
- **Inferencia**: `modal run research/research_amadeus_modal.py::main --prompt "..."`

#### Benchmark (8 tests en `evaluation/amadeus/`)

| Test | Categoría | Prompt |
|------|-----------|--------|
| test01 | official-example | Electronic ambient, E major, tubular bells, Andante |
| test02 | official-example | Electronic dreamy, B minor, drums, piano, brass, sax |
| test03 | official-example | Soothing pop, C major, piano, flute, violin |
| test04 | official-example | Rock/pop, A minor, pizzicato strings, 148 BPM |
| test05 | cross-model | = text2midi/test4: trance 138 BPM, A minor (structured) |
| test06 | cross-model | = text2midi/test6: C minor, 124 BPM, chord C7/E-Eb6-Bbm6 |
| test07 | cross-model | = text2midi/test1: "A sad pop song with a strong piano presence." |
| test08 | cross-model | = text2midi/test5: "A cheerful christmas song suitable for children." |

#### Resultados evaluación completa (2026-06-12, 8 tests × 2 variantes, CUDA A10G) ⚠️ EVALUADO

| Test | Prompt | Calidad v0 | Calidad v1 | Observaciones |
|------|--------|-----------|-----------|---------------|
| test01 | electronic ambient, E major, tubular bells | 1.5/5 | 2.5/5 | oboe overwhelm (v0=8.7s corto); v1 textura ambient razonable |
| test02 | electronic dreamy, B minor, drums, piano, brass, sax | 3/5 | 3/5 | brass+sax+bass presentes; v1 drums+piano+sax+brass ✓ |
| test03 | soothing pop, piano, flute, violin, guitar | **4/5** | 2.5/5 | v0 MATCH PERFECTO (todos los instrumentos); v1 sax inesperado |
| test04 | rock/pop, pizzicato strings, 148 BPM | 1/5 | 1.5/5 | FALLO: piccolo + organ, sin rock context |
| test05 | trance 138 BPM, drums, distortion guitar, flute, synth | 1.5/5 | 3/5 | v0 extremadamente corto (5.5s); v1 synth+bass+flute razonable |
| test06 | C minor, 124 BPM, brass, strings, tenor-sax, guitar, slap | **3.5/5** | 2.5/5 | v0 MATCH MUY BUENO — 9 instrumentos correctos |
| test07 | "A sad pop song with a strong piano presence." | **3.5/5** | **3.5/5** | Solo piano en ambas variantes — seguimiento perfecto |
| test08 | "A cheerful christmas song suitable for children." | 3/5 | 3/5 | Orquestal festivo con brass+strings+voices — apropiado |

**Métricas generales:**
- Velocidad: 18-19 tok/s en A10G → ~59s/output de 1024 tokens
- Notas por MIDI: exactamente 1022 notas (fijo, de 1024 tokens)
- Duración: **muy variable** (5.5s a 136.6s) — el modelo no controla duración por prompt
- Tracks por MIDI: 1 a 13 pistas (promedio ~6)

**Hallazgos clave:**

1. **Seguimiento de instrumentación sustancialmente mejor que text2midi y MIDI-LLM**: test03/v0 y test06/v0 son los mejores resultados vistos en toda la evaluación de modelos MIDI. La arquitectura NB con difusión bidireccional parece mejorar la distribución de instrumentos.

2. **Duración completamente inconsistente**: El modelo genera siempre ~1022 notas pero el spacing temporal varía enormemente (5.5s a 136s). El prompt "duration of 252 seconds" (test02) produjo 47s y 20s — el modelo ignora las referencias de duración.

3. **Prompts cortos funcionan bien**: test07 ("A sad pop song with a strong piano presence.") → solo piano en ambas variantes. Más robusto que text2midi ante prompts informales.

4. **"Rock" no se traduce a instrumentación eléctrica**: test04 (rock/pizzicato) → piccolo y organ. El dataset LakhALLFined probablemente subrepresenta el rock.

5. **Comparativa cross-model**: Amadeus supera a text2midi en 3 de 4 tests comparativos (test05 v1, test06 v0, test07). Es el mejor modelo evaluado para flujo 1 (text→MIDI) en términos de seguimiento de instrumentación.

6. **Limitación del modelo-S**: El modelo liberado (Amadeus-S, 280M params) es la variante más pequeña. Los modelos M y L del paper no están disponibles públicamente. La calidad mediocre en algunos tests puede ser consecuencia del tamaño reducido.

**Comparativa cross-model (mismos prompts):**

| Prompt estilo | text2midi | MIDI-LLM | Amadeus |
|---|---|---|---|
| Trance 138BPM (test05) | 2/5 | — | 3/5 (v1) |
| C minor 124BPM chords (test06) | 2/5 | — | 3.5/5 (v0) |
| "Sad pop + piano" (test07) | 2/5 | — | 3.5/5 |
| "Christmas for children" (test08) | 2.5/5 | — | 3/5 |

**Veredicto: ⚠️ MEJOR CANDIDATO PARA FLUJO 1 (entre los evaluados), pero insuficiente para producción**

Amadeus-S supera a text2midi y MIDI-LLM en seguimiento de instrumentación — la adición del decoder de difusión bidireccional marca la diferencia. Sin embargo, la inconsistencia en duración, los fallos en géneros eléctricos/percusivos (rock, EDM con drums reales), y el hecho de que sea el modelo más pequeño (sin los modelos M/L disponibles) lo hacen inadecuado para producción sin postproceso.

**Segunda batería — cross-model completo (2026-06-12, tests 09-23):**

| Test | Fuente | Prompt (resumen) | Calidad | Highlight |
|------|--------|-----------------|---------|-----------|
| test09 | t2m/test2 | Rock, drums, electric guitar, fast | 3/5 | electric-guitar+bass+drums ✓ |
| test10 | t2m/test3 | Soft love song on piano | 3.5/5 | solo piano, 291s, densidad baja |
| test11 | t2m/test7 | Heavy metal, drums, guitar | 2.5/5 | drums+distortion-guitar pero sin carácter metal |
| test12 | mllm/c1 | Soothing pop+rock, piano, pan flute, guitar, A minor | 3/5 | piano+guitar+strings+voices |
| test13 | mllm/c2 | Electronic rock, strings, voice, overdriven guitar, drums | 2/5 | strings+voices+drums; sax inesperado; 20s corto |
| test14 | mllm/c3 | Cinematic, viola+violin+cello, Andante, C major | 3.5/5 | **cuerdas perfectas** (viola+violin+cello) pero 12.7s corto |
| test15 | mllm/c4 | Classical, church organ + French horn, A minor, 361s | **4/5** | **church-organ solo, 214s** — mejor seguimiento de duración |
| test16 | mllm/c5 | Lively pop, strings+brass+choir+piano, D major, 129 BPM | 3/5 | brass+strings+drums+voices todos presentes |
| test17 | mllm/c6 | Rock/pop, pizzicato strings, 148 BPM, A minor | 1.5/5 | piano+strings+oboe — confirma fallo consistente de "rock" |
| test18 | mllm/c7 | Orchestral, piano+strings+guitar+voice+electric guitar, F major | 2.5/5 | strings+guitar+bass pero voices overwhelm |
| test19 | mllm/c8 | Electronic dark, electric guitar+tremolo strings+synth brass+drums, Eb minor | 3/5 | guitar+synth-brass+strings razonables |
| test20 | mllm/c9 | Electronic, synth lead+strings+bass+guitar+synth brass, C minor | 2.5/5 | strings dominant; sin synth-lead claro |
| test21 | mllm/c10 | Rock/pop, piano+distortion guitar+synth strings+drums, A minor | **3.5/5** | piano+drums+distortion-guitar ✓ — mejor rock de la evaluación |
| test22 | mllm/c11 | Instrumental rock, Bb minor, distorted guitar+synth pads+cello+choir | 2.5/5 | distortion-guitar+drums+bass; voices > choir |
| test23 | mllm/c12 | Christmas, Eb major, 122 BPM, piano+strings ensemble | 3.5/5 | piano+brass+drums festivo ✓ |

**Conclusiones consolidadas (31 MIDIs en total):**

1. **Instrumentación vs texto**: Amadeus-S supera a text2midi y MIDI-LLM en seguimiento de instrumentación. Funciona mejor con prompts MidiCaps-style que incluyen tonalidad, BPM y chord progression.

2. **Cuerdas y viento maderas**: El modelo tiene especial afinidad por violin/viola/cello, oboe, y flute — probablemente sobrerrepresentados en LakhALLFined.

3. **Rock/metal: punto débil consistente**: En 3 tests con contexto "rock/metal/heavy" (test09 siendo el mejor con 3/5), el modelo nunca genera guitarras distorsionadas con suficiente carácter. "Pizzicato strings" + "rock" → piano+strings (test04, test17: 1.5/5). Distortion guitar sí aparece (test11, test21) pero el ambiente es pop, no rock.

4. **Duración más larga cuando la densidad es baja**: test10 (291s) y test15 (214s) son los más largos — el modelo genera notas más separadas para prompts suaves/lentos. Coherente con el concepto musical.

5. **Test15 (church organ, 214s) y test14 (cuerdas, 3 instrumentos perfectos)**: los mejores resultados de la segunda batería.

6. **Voices/soprano-sax overrepresented**: En varios tests aparecen como instrumentos dominantes sin estar en el prompt — posible sesgo del dataset.

**Para explorar en futuras iteraciones:**
- Cuando los modelos Amadeus-M o Amadeus-L estén disponibles (esperados con más instrumentos y mayor coherencia)
- Postproceso de duración: ajustar el tempo MIDI para alcanzar duraciones objetivo

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

*Documento generado: 2026-06-01. PoC Text2midi+AMT: 2026-06-02. MIDI-LLM evaluado: 2026-06-12. MuseCoco evaluado: 2026-06-12. Text2midi evaluado y DESCARTADO: 2026-06-12. ChatMusician evaluado y CERRADO: 2026-06-12 (7.5/12, mono-staff; nota fine-tuning de autor pendiente). Próxima iteración: fine-tuning de estilo, Amadeus, o pivote audio-first.*
