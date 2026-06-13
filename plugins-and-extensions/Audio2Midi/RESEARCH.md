# Audio2Midi — Investigación de modelos

**Objetivo**: evaluar qué modelos/herramientas encajan mejor en un pipeline REAPER de transcripción automática de audio a MIDI multi-instrumento:
1. **Audio → MIDI multi-instrumento** (transcripción de una mezcla completa en pistas separadas por instrumento)
2. Caso especial: separación de stems + transcripción por pista (arquitectura compuesta)

**Hardware objetivo**: Mac Apple Silicon + Modal cloud fallback  
**Licencia**: uso personal/no comercial (cualquier licencia open source válida)

---

## Tabla comparativa de candidatos (actualizada 2026-06-13)

| Modelo | Enfoque | Multi-instrumento | MIDI nativo | Mac (MPS/CPU) | Licencia | Madurez | Veredicto |
|---|---|---|---|---|---|---|---|
| **YourMT3+** (QMUL) | End-to-end AMT | ✅ sí (incl. vocal) | ✅ sí | ⚠️ CPU (lento) | Apache 2.0 | ✅ MLSP 2024, GitHub | **TOP CANDIDATO** |
| **MT3** (Google Magenta) | End-to-end AMT | ✅ sí (6 datasets) | ✅ sí | ⚠️ CPU/JAX | Apache 2.0 | ✅ ICLR 2022, GitHub | Baseline de referencia |
| **PerceiverTF** (Lu/Wang) | End-to-end AMT | ✅ 12 clases + vocal | ✅ sí | ⚠️ CPU | MIT | ✅ ICASSP 2023 | Base de YourMT3+; integrado |
| **AMT Challenge 2025** (varios) | End-to-end AMT | ✅ sí | ✅ sí | ❓ depende impl. | variada | ✅ ICLR WS 2025 | Pendiente revisar repos |
| **MIDI-VALLE** (ISMIR 2025) | Neural codec LM | ✅ potencial | ✅ sí | ❓ desconocido | ? | ⚠️ ISMIR 2025 | Candidato emergente |
| **Two-branch clustering** | Ligero AMT | ✅ sí | ✅ sí | ❓ desconocido | ? | ⚠️ arXiv sep 2025 | Pendiente revisar |
| **Omnizart** (MCT Lab) | Toolbox modular | ✅ parcial (por módulo) | ✅ sí | ✅ CPU | Apache 2.0 | ⚠️ activo pero 2021 | ❌ DESCARTADO: arq. antigua |
| **Basic Pitch** (Spotify) | Mono/polif. instrument-agnostic | ❌ mono-instrumento | ✅ sí | ✅ MPS/CPU | Apache 2.0 | ✅ 2022, producción | Solo en pipeline compuesto |
| **Transkun V2** (ISMIR 2024) | Piano-only AMT | ❌ solo piano | ✅ sí | ✅ CPU | MIT | ✅ ISMIR 2024 | Solo en pipeline compuesto |
| **OaF Drums** (Magenta) | Drums-only | ❌ solo batería | ✅ sí | ✅ CPU/TF | Apache 2.0 | ✅ SOTA drums | Solo en pipeline compuesto |
| **ADTOF** (Zehren) | Drums-only CRNN | ❌ solo batería | ✅ sí | ✅ CPU | LGPL | ✅ ICASSP 2022 | Solo en pipeline compuesto |
| **Klangio** | SaaS por instrumento | ❌ un instrumento/app | ✅ MIDI+MXL+GP5 | ✅ (API REST) | Comercial | ✅ producción 2026 | ❌ DESCARTADO: solo 4/4 y 3/4 |
| **AnthemScore** | SaaS generalista | ⚠️ 1 pista | ✅ MIDI | ✅ (desktop) | Comercial | ✅ producción | ❌ DESCARTADO: mono, comercial |
| **NeuralNote** | Plugin VST/AU | ❌ mono (usa Basic Pitch) | ✅ sí | ✅ nativo | MIT | ⚠️ activo | Referencia de integración DAW |

---

## Evaluación detallada de candidatos elegidos

### YourMT3+ — Top candidato multi-instrumento end-to-end

- **Repositorio**: https://github.com/mimbres/YourMT3
- **Paper**: MLSP 2024, "YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation" (arXiv 2407.04822)
- **Arquitectura**: híbrido MT3 + PerceiverTF con Hierarchical Attention Transformer (HAT) en el dominio tiempo-frecuencia + Mixture of Experts (MoE). Nuevo método de decodificación multi-canal para anotaciones incompletas. Aumentación intra-stem y cross-stem para mezcla de datasets.
- **Dataset de entrenamiento**: mezcla de 10 datasets públicos (SOD, MAPS, MAESTRO, GuitarSet, URMP, MusicNet, Slakh, ComMU, EnsembleSet, HookTheory). Cross-dataset stem augmentation como técnica clave.
- **Soporte MPS**: no optimizado (CUDA-first). Funciona en CPU pero la inferencia será lenta para audios largos. Para producción: Modal CUDA.
- **Output**: archivos `.mid` con anotaciones multi-instrumento, incluyendo clase MIDI de instrumento por pista.
- **Capacidades únicas**: transcripción vocal directa (elimina la necesidad de pre-separador de voz), SOTA en 10 datasets, código completamente reproducible.
- **Limitaciones conocidas**: rendimiento bajo (<10%) en instrumentos no principales (excluyendo piano, bajo, vocal y batería) en música pop comercial cuando el entrenamiento se basa solo en datasets sintéticos. No reemplaza el fine-tuning con datos de pop real.
- **Puntos fuertes**: Apache 2.0, SOTA demostrado, benchmark exhaustivo, código público, incluye vocal, multi-track nativo.

#### Parámetros de inferencia (referencia, verbatim de `docs/` del repo)
```python
# Inferencia con checkpoint pre-entrenado
model_name = "YourMT3+"  # o variantes: "MT3", "PerceiverTF", etc.
checkpoint = "path/to/checkpoint.ckpt"
audio_file = "input.wav"  # 16kHz mono recomendado, o el modelo re-muestrea
# Output: .mid con múltiples tracks por instrumento
```

#### Resultados evaluación

*pendiente — PoC siguiente iteración*

---

### MT3 (Google Magenta) — Baseline de referencia

- **Repositorio**: https://github.com/magenta/mt3
- **Modelo HF**: checkpoints ICLR 2022 disponibles via repo
- **Paper**: ICLR 2022, "MT3: Multi-Task Multitrack Music Transcription" (arXiv 2111.03017)
- **Arquitectura**: T5-small (~60M params), encoder-decoder Transformer. Input: espectrograma Mel. Output: secuencia de tokens de eventos MIDI (onset, pitch, velocity, instrumento). Entrenado simultáneamente sobre 6 datasets en multi-task fashion.
- **Dataset**: MAESTRO (piano solo), GuitarSet, URMP, MusicNet, Slakh2100, SOD (6 datasets)
- **Soporte MPS**: framework JAX/T5X — no tiene soporte MPS nativo. Funciona en CPU pero dependencias JAX hacen la instalación más compleja en Mac.
- **Output**: archivos `.mid` multi-track con clases de instrumento.
- **Limitaciones**: T5-small es el modelo más pequeño de la familia T5 (60M params). YourMT3+ demuestra ganancias claras sobre MT3 en benchmarks modernos. Framework JAX complica el deployment fuera de TPU/GPU NVIDIA.
- **Puntos fuertes**: Apache 2.0, referencia académica muy citada, primer sistema general de AMT multi-task multi-track con Transformers.

#### Resultados evaluación

*pendiente — usado principalmente como baseline de comparación en el benchmark de YourMT3+*

---

### PerceiverTF — Arquitectura de referencia

- **Repositorio**: integrado en YourMT3 (`github.com/mimbres/YourMT3`)
- **Paper**: ICASSP 2023, "Multitrack Music Transcription with a Time-Frequency Perceiver" (arXiv 2306.10785)
- **Arquitectura**: augmenta Perceiver con una capa Transformer adicional para modelar coherencia temporal en el dominio tiempo-frecuencia. Entrenado para 12 clases instrumentales + vocal en multi-task learning.
- **Relación con YourMT3+**: YourMT3+ es un modelo híbrido que combina MT3 y PerceiverTF. PerceiverTF supera a MT3 y SpecTNT en los benchmarks del paper de 2023. YourMT3+ supera a ambos con <2.5% de parámetros adicionales respecto a MT3.
- **Nota práctica**: no usar de forma aislada — el checkpoint de YourMT3+ ya incluye su arquitectura mejorada. Documentado aquí como contexto histórico.

---

### AMT Challenge 2025 — Submissions recientes

- **Challenge**: 2025 Automatic Music Transcription Challenge (ai4musicians.org)
- **Paper**: "Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge" (arXiv 2603.27528, ICLR Workshop 2025)
- **Contexto**: primer benchmark público oficial de AMT multi-instrumento post-MT3. Varias submissions superaron el baseline MT3.
- **Hallazgos clave**: advances claros en accuracy, pero debilidades persistentes en polifonía densa, instrumentos tímbricamente similares, y datos de diversidad limitada. Foco en música clásica; jazz y pop previstos para futuras iteraciones.
- **Estado**: verificar qué submissions tienen código público y checkpoints disponibles. En particular buscar implementaciones que superen MT3 con soporte PyTorch (no JAX).

#### Resultados evaluación

*pendiente — revisar repos publicados de submissions*

---

### MIDI-VALLE — Candidato emergente (ISMIR 2025)

- **Paper**: ISMIR 2025, "MIDI-VALLE: Improving Expressive Piano Performance Synthesis Through Neural Codec Language Modelling"
- **Arquitectura**: trata MIDI como tokens discretos (neural codec LM), evitando piano rolls y espectrogramas tradicionales. Framework adaptable a síntesis y transcripción (incluyendo audio→MIDI).
- **Capacidades declaradas**: aplicable a score prediction, audio-to-MIDI transcription, music generation.
- **Estado**: código/checkpoint no verificado. Prioridad: buscar repo público tras ISMIR 2025.
- **Limitación conocida**: paper centrado en piano expresivo; capacidad multi-instrumento por confirmar.

#### Resultados evaluación

*pendiente — verificar disponibilidad de código y checkpoint*

---

### Two-branch contrastive clustering — Modelo ligero reciente

- **Paper**: arXiv 2509.12712 (septiembre 2025), "A Lightweight Two-Branch Architecture for Multi-instrument Transcription via Note-Level Contrastive Clustering"
- **Arquitectura**: two-branch para AMT multi-instrumento vía clustering contrastivo a nivel de nota. Diseñado para ser computacionalmente ligero.
- **Estado**: arquitectura reciente, sin evaluación propia. Interesante si el objetivo es inferencia local en Mac (CPU/MPS) sin Modal.
- **Prioridad**: segunda iteración tras YourMT3+.

#### Resultados evaluación

*pendiente*

---

## Arquitecturas compuestas: source separation + transcripción por stem

Las arquitecturas "end-to-end" (YourMT3+, MT3) intentan resolver el problema en un solo modelo. Una alternativa práctica es construir un pipeline en dos etapas:

```
[Audio mix]
    ↓
[Source separation: Demucs v4 / HTDemucs]
    ↓ (stems: vocals, bass, drums, other)
[Transcripción mono-instrumento por stem]
    ├── vocals → Basic Pitch  (melodía vocal)
    ├── bass   → Basic Pitch o Transkun (línea de bajo)
    ├── drums  → OaF Drums / ADTOF (batería → MIDI kit)
    └── other  → Basic Pitch (piano, guitarras, synths)
    ↓
[Merge de MIDI tracks]
    ↓
[MIDI multi-instrumento]
```

### Componentes del pipeline compuesto

#### Demucs v4 / HTDemucs (Meta)

- **Repositorio**: https://github.com/facebookresearch/demucs
- **Variantes**: `htdemucs` (4-stem por defecto), `htdemucs_6s` (6 stems: vocals+bass+drums+guitar+piano+other)
- **Output**: audio WAV separado por stem (vocals, bass, drums, other — o 6 stems)
- **Soporte Mac**: ✅ MPS nativo (PyTorch), rápido en Apple Silicon
- **Licencia**: MIT (v4), con excepciones por modelos pre-entrenados
- **Relevancia**: el proyecto ya tuvo `StemsSeparator/` basado en Demucs — la lógica de separación existe como referencia y puede reutilizarse directamente.

#### Basic Pitch (Spotify)

- **Repositorio**: https://github.com/spotify/basic-pitch
- **Arquitectura**: CNN ligera, instrument-agnostic, polyfónico. Incluye detección de pitch bends.
- **Limitación clave**: works best on one instrument at a time — optimizado para mono-instrumento. No mezclas.
- **Uso en pipeline compuesto**: ideal para stems individuales (bajo, vocals, other). Muy ligero, funciona en CPU/MPS sin CUDA.
- **Licencia**: Apache 2.0

#### Transkun V2 (ISMIR 2024)

- **Repositorio**: https://github.com/Yujia-Yan/Transkun
- **Arquitectura**: V2 = Transformer (reemplaza CNN de V1). Piano-only AMT, SOTA en MAESTRO.
- **Uso en pipeline compuesto**: para el stem "piano" o "other" en canciones dominadas por piano.
- **Limitación**: piano solo — no guitarra, no voz, no bajo.

#### OaF Drums (Magenta)

- **Repositorio**: https://magenta.tensorflow.org/oaf-drums
- **Dataset de entrenamiento**: Expanded Groove MIDI Dataset (E-GMD)
- **Capacidades**: velocidad, clasificación de hits de batería, timing preciso. SOTA perceptual en drum transcription según listening study.
- **Licencia**: Apache 2.0

#### ADTOF (Zehren)

- **Repositorio**: https://github.com/MZehren/ADTOF
- **Dataset**: 359h de música real anotada (no sintética) via rhythm game charts
- **Arquitectura**: CRNN con dataset escalado. Complementario a OaF Drums.
- **Ventaja vs OaF**: entrenado en datos reales (no solo MIDI sintético), mejor generalización a rock/metal.
- **Licencia**: LGPL

### Trade-offs: pipeline compuesto vs end-to-end

| Dimensión | End-to-end (YourMT3+) | Pipeline compuesto (Demucs + Basic Pitch) |
|---|---|---|
| **Calidad en pop real** | ⚠️ <10% en inst. secundarias | ⚠️ errores en cascada (leakage Demucs) |
| **Control granular** | ❌ black box | ✅ ajustable por stem |
| **Latencia total** | ✅ un solo modelo | ❌ separación + N transcripciones |
| **Complejidad de impl.** | ✅ menor | ❌ orquestación multi-paso |
| **Drums** | ⚠️ variable | ✅ ADTOF especializado |
| **Vocal transcripción** | ✅ YourMT3+ nativo | ⚠️ Basic Pitch en vocal funciona mal |
| **Reutilización StemsSeparator** | ❌ no aplica | ✅ código Demucs ya existente |
| **Soporte Mac MPS** | ❌ CUDA-first | ✅ Demucs + Basic Pitch = MPS nativo |

**Recomendación práctica**: probar YourMT3+ como primera opción (mayor calidad teórica). Si la calidad en batería o géneros específicos es insatisfactoria, implementar pipeline compuesto con Demucs + ADTOF para drums + Basic Pitch para otros stems — reutilizando la lógica de `StemsSeparator/`.

---

## Candidatos descartados — Justificación

### Omnizart (MCT Lab)
Toolbox modular multi-instrumento disponible desde 2021. Cubre piano solo, drum, vocal, chord recognition, beat tracking. Motivo de descarte: las arquitecturas subyacentes (U-Net, CRNN) son anteriores a la revolución Transformer en AMT. YourMT3+ y MT3 superan a Omnizart en todos los benchmarks donde se solapan. El mantenimiento activo del repo no compensa la brecha de calidad. Caso de uso potencial: referencia para módulos específicos (drums, vocal) si el pipeline compuesto requiere alternativas ligeras a OaF/ADTOF.

### Klangio (comercial)
Apps específicas por instrumento (piano, guitarra, bajo, vocal) con API REST. Produce MIDI + MusicXML + PDF + GP5. **Limitación crítica documentada en docs de septiembre 2025**: solo soporta 4/4 y 3/4 como metros. Inutilizable para música en 3/8, 6/8, 5/4, etc. Además, API comercial = dependencia externa incompatible con el modelo de plugin local. Alternativa si se necesita transcripción de alta calidad para instrumentos concretos en demos: válida para uso manual, no para pipeline automático.

### AnthemScore (comercial)
Software de escritorio AI de transcripción, single-model. Sólido y soporta todos los metros (a diferencia de Klangio), pero produce una sola pista de transcripción polifónica, no pistas por instrumento. Modelo comercial incompatible con integración automática en REAPER. Descartado.

### NeuralNote (plugin VST/AU)
Open source (MIT), envuelve internamente Basic Pitch. Interesante como precedente de integración DAW (exactamente lo que queremos construir). No es un candidato de transcripción multi-instrumento, sino una referencia de arquitectura para la capa de integración REAPER. Revisar su código cuando se diseñe el plugin REAPER final.

### ReconVAT (ACM MM 2021)
Framework semi-supervisado para AMT con datos escasos. Limitación estructural: no predice etiquetas de instrumento — produce un único piano-roll que combina todos los instrumentos. Descartado para multi-instrumento.

---

## Recomendación final

### Estado de evaluación (2026-06-13)

| Modelo | Tipo | Estado | Veredicto |
|---|---|---|---|
| **YourMT3+** | End-to-end AMT | *pendiente PoC* | **TOP CANDIDATO — evaluar primero** |
| MT3 | End-to-end AMT | *pendiente* | Baseline de comparación |
| AMT Challenge 2025 | End-to-end AMT | *pendiente revisar repos* | Candidato secundario |
| Pipeline Demucs + Basic Pitch | Compuesto | *pendiente* | Alternativa Mac-nativa |
| Pipeline Demucs + ADTOF | Compuesto (drums) | *pendiente* | Componente drums del pipeline |
| Omnizart | Toolbox modular | ❌ DESCARTADO | Arq. antigua |
| Klangio | SaaS comercial | ❌ DESCARTADO | Solo 4/4 y 3/4, comercial |
| AnthemScore | SaaS comercial | ❌ DESCARTADO | Mono, comercial |

### Primer PoC recomendado: YourMT3+

YourMT3+ es el estado del arte más sólido para transcripción multi-instrumento end-to-end:
- Supera a MT3 y PerceiverTF en todos los benchmarks públicos
- Transcripción vocal nativa (no requiere separador previo)
- Código completamente reproducible con 10 datasets
- Licencia Apache 2.0

**Limitación esperada en pop real**: los tests del paper muestran que el rendimiento en instrumentos secundarios de música pop comercial cae <10% cuando el entrenamiento es solo con datasets sintéticos/académicos. Para compensar: evaluar con audios sintéticos (MIDI→audio conocido = ground truth perfecto) y con mezclas reales de complejidad moderada.

**Métricas de evaluación a adoptar** (siguiendo el benchmark del AMT Challenge 2025):
- F1 nota (onset + pitch): métrica base
- F1 nota+offset (onset + pitch + duración): más estricta
- F1 nota+instrumento (onset + pitch + clase MIDI): métrica clave para multi-instrumento
- Calidad cualitativa subjetiva (0-5) como en evaluaciones previas del proyecto

### Segundo PoC: pipeline compuesto (Demucs + ADTOF + Basic Pitch)

Si YourMT3+ no alcanza calidad suficiente en batería o géneros específicos, implementar el pipeline compuesto reutilizando la lógica de `StemsSeparator/` (Demucs ya evaluado en el proyecto). ADTOF para drums es la pieza diferencial.

---

## Próximos pasos (siguiente iteración)

1. **Setup Modal para YourMT3+**: completar `research/research_yourmt3_modal.py` (skeleton listo, añadir lógica) y ejecutar `modal run research/research_yourmt3_modal.py::setup`
2. **Preparar fixtures**: 3 audios en `research/fixtures/` (ver `research/fixtures/README.md`)
3. **PoC inicial**: transcribir los fixtures con YourMT3+, guardar en `evaluation/yourmt3/`
4. **Evaluación cualitativa**: escuchar en REAPER, anotar F1 estimado visual + calidad 0-5 por instrumento
5. **Decidir**: ¿YourMT3+ es suficiente, o se necesita pipeline compuesto para drums/pop?
6. **Integración**: si YourMT3+ aprueba, diseñar el bridge Lua en REAPER (inspirado en `NeuralNote` para la capa de integración DAW)

---

## Instrucciones para ejecutar los PoC

### Requisitos previos
```bash
# desde la carpeta research/
cd plugins-and-extensions/Audio2Midi/research
uv sync
```

### PoC 1 — YourMT3+ vía Modal (skeleton listo, pendiente implementación)
```bash
# Primera vez: descarga pesos al Volume (ejecutar una sola vez)
modal run research/research_yourmt3_modal.py::setup

# Transcripción de un audio:
modal run research/research_yourmt3_modal.py::main \
    --audio-path research/fixtures/multitracks_short.wav \
    --out-dir evaluation/yourmt3/smoke

# Benchmark completo (cuando existan test*/input.wav):
modal run research/research_yourmt3_modal.py::eval_all \
    --eval-dir evaluation/yourmt3
```

### PoC 2 — Pipeline compuesto (pendiente implementación)
```bash
# Separación de stems con Demucs (reutilizar StemsSeparator/ como referencia)
# Transcripción por stem con Basic Pitch / ADTOF / OaF Drums
```

### Validación en REAPER
1. Arrastrar cada `.mid` generado a REAPER
2. Asignar instrumentos VST por pista y escuchar
3. Comparar pistas MIDI vs audio original
4. Anotar calidad por instrumento (batería, bajo, piano, cuerdas, voz) en las tablas de evaluación

---

*Documento generado: 2026-06-13. Research inicial: búsqueda de papers ICASSP/MLSP/ISMIR 2023-2025 y AMT Challenge 2025. Próxima iteración: PoC YourMT3+ en Modal A10G, evaluación con 3 fixtures.*
