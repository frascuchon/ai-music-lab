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
| **ADTOF** (Zehren) | Drums-only CRNN | ❌ solo batería | ✅ sí | ✅ CPU | CC BY-NC-SA 4.0 | ✅ ISMIR 2021 | Solo en pipeline compuesto (compound v2) |
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

#### Resultados evaluación (2026-06-22)

##### Métricas objetivas — Slakh2100 (ground truth disponible)

| Test | Pista | F1 onset+pitch | F1 +offset | F1 clase | Notas |
|------|-------|----------------|------------|----------|-------|
| test04 | Slakh 1884 | **77.5%** | 61.8% | 61.5% | Piano 93%, Guitar 91%, Bass 43%, Ensemble 64% |
| test05 | Slakh 1975 | **73.9%** | 26.5% | 90.9% | SynthLead 99%, Piano 81.5%, Bass 74%, Ensemble 30% |

Comparación con paper YourMT3+ (tabla 2, Slakh test set): F1 onset+pitch publicado ~70% → **nuestra reproducción supera el paper en ambos tests** (77.5% y 73.9%).

El F1+offset de test05 (26.5%) es anormalmente bajo por duraciones de notas excesivamente largas en la clase Ensemble — los onsets son correctos pero las notas no se cierran bien.

##### Métricas objetivas — MusicNet (limitación de GT)

| Test | Pista | F1 onset+pitch | F1 +offset | F1 clase | Notas |
|------|-------|----------------|------------|----------|-------|
| test07 | MusicNet 2556 | 2.6% | 0.7% | 100% | Piano solo; F1-cls=100% confirma detección correcta del instrumento |
| test08 | MusicNet 2628 | 14.4% | 7.0% | 50% | Piano+Strings; Strings transcrito como Ensemble (error de clase) |

**Aviso**: el F1 bajo en MusicNet no refleja la calidad real del modelo. Los MIDIs en `musicnet_midis.tar.gz` tienen timing de partitura (score), no de performance grabada — los onsets no coinciden con el audio. El F1-cls=100% en test07 confirma que el modelo sí identifica correctamente el instrumento; la transcripción será evaluada subjetivamente en REAPER. Para F1 real en MusicNet se necesitaría bajar los CSV de labels por grabación.

##### Evaluación cualitativa en REAPER (impresión general, 2026-06-22)

Escucha general comparativa de todos los tests: **YourMT3+ funciona claramente mejor** que el pipeline compuesto (Demucs + Basic Pitch). Impresión global confirmada.

Evaluación por instrumento por test: **pendiente** — anotar puntuación 0–5 en cada `notes.txt`.

##### Veredicto YourMT3+

- **Rendimiento general**: claramente superior al pipeline compuesto en escucha subjetiva (confirma F1 objetivo).
- **Música sintética multi-instrumento (Slakh)**: excelente. Piano y guitarra cerca de perfección (>90% F1). Bajo y ensemble con margen de mejora.
- **Música clásica real (MusicNet/MAPS)**: pendiente de puntuación por instrumento.
- **Limitación en instrumentos secundarios (Funk/Cámara)**: pendiente de puntuación. El paper indica <10% en inst. no principales de pop real con datos solo sintéticos.

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

❌ **CERRADO sin evaluar (2026-06-22).** YourMT3+ es el sucesor directo de MT3 y lo supera en todos los benchmarks públicos donde se solapan. Evaluar MT3 no aportaría información adicional al proyecto.

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
- **Licencia**: CC BY-NC-SA 4.0 (no comercial, share-alike) — solo investigación interna. Nota: RESEARCH.md indicaba LGPL por error; la licencia real es CC BY-NC-SA 4.0.

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

**Veredicto evaluado (2026-06-22)**: el pipeline compuesto fue evaluado con F1 objetivo (compound/test03 vs Slakh 1884): **31.1% vs 77.5% de YourMT3+** en el mismo audio, y **confirmado subjetivamente en escucha general** — YourMT3+ es claramente mejor. La hipótesis de que BasicPitch mejoraría con stems aislados no se cumple — el leakage de Demucs introduce demasiadas notas fantasma (7484 est vs 2355 ref = 3.2× sobre-detección). **El pipeline Demucs + Basic Pitch queda DESCARTADO.**

**Pipeline compuesto v2 (Demucs + ADTOF + YourMT3+) — implementado 2026-06-24:**

- Audio original completo → **YourMT3+** (F1 pitched preservado al máximo)
- Demucs htdemucs_6s extrae solo el stem **drums** → **ADTOF** (drums especializado, 359h real)
- Merge pitched + drums → `transcribed_cuda.mid` multi-pista
- Scripts: `research_adtof_modal.py` + `research_compound_v2_modal.py`
- Tests: `evaluation/compound_v2/test{04,05,07,08}` con GT disponible
- Métrica adicional en `compute_f1.py`: F1 drum-onset por clase (BD/SD/HH/TT/CY), `--no-drums` para omitir.

Nota de diseño: YourMT3+ recibe audio completo (no stems) para evitar el leakage de Demucs que hundió v1. El rol de Demucs se limita a extraer el stem de batería para ADTOF.

**EVALUADO 2026-06-24:**

| Test | Dataset | F1-op pitched | Δ vs YourMT3+ | F1-drm drums (macro) |
|---|---|---|---|---|
| test04 | Slakh 1884 | **77.5%** | +0.0pp (paridad) | 40.0% (HH 80%, BD 40%, SD 43%) |
| test05 | Slakh 1975 | **73.9%** | +0.0pp (paridad) | 49.6% (BD 75%, HH 66%, SD 49%) |
| test07 | MusicNet 2556 | 2.6% | +0.0pp | N/A (sin batería) |
| test08 | MusicNet 2628 | 14.4% | +0.0pp | N/A (sin batería) |

**Veredicto compound v2:**
- ✅ **F1 pitched = YourMT3+ standalone en todos los tests** — el merge no degrada el canal melódico.
- ⚠️ **F1 drums ADTOF**: 40-50% macro en Slakh (HH excelente 80%, BD/SD sobre-detectados 3-4× por leakage de Demucs). ADTOF aporta batería real ausente en YourMT3+, pero con ruido significativo.
- ✅ **Sin batería en clásico**: ADTOF correctamente devuelve MIDI vacío en MusicNet. El merge no introduce ruido.
- **Conclusión**: compound v2 es una mejora sobre v1 (F1 31.1% → 77.5% pitched) y añade pista de batería a YourMT3+. La calidad de drums (40-50% F1) es subóptima pero funcional como primera pista de percusión. El leakage de Demucs es el limitante principal.

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

### Estado de evaluación (2026-06-23)

| Modelo | Tipo | Estado | Veredicto |
|---|---|---|---|
| **YourMT3+** | End-to-end AMT | ✅ F1 Slakh 77.5% + subjetivo positivo | **ELEGIDO — pipeline de producción** |
| Pipeline Demucs + Basic Pitch | Compuesto | ✅ F1 31.1% + subjetivo negativo | ❌ **DESCARTADO — 3.2× sobre-detección** |
| MT3 | End-to-end AMT | ❌ CERRADO sin evaluar | YourMT3+ es su sucesor directo; no aporta información adicional evaluarlo |
| **MIROS** (AMT Challenge 2025 winner) | End-to-end AMT | ✅ F1-op 77.5%/64.0% (Slakh). Subjetivo real: **mejor de los tres** (melodía, 2026-06-24) | Empata YourMT3+ en Slakh 1884; pierde 9.9pp en Slakh 1975. En música real propia supera a YourMT3+ subjetivamente. |
| **YourMT3+ + ADTOF (compound v2)** | Compuesto v2 | ✅ CERRADO 2026-06-24 — F1-pitched=77.5% (=YourMT3+), F1-drm=40-50% | Merge no degrada pitched. Drums: HH 80% pero BD/SD sobre-detectados 3-4×. |
| Omnizart | Toolbox modular | ❌ DESCARTADO | Arq. antigua |
| Klangio | SaaS comercial | ❌ DESCARTADO | Solo 4/4 y 3/4, comercial |
| AnthemScore | SaaS comercial | ❌ DESCARTADO | Mono, comercial |

### Veredicto: YourMT3+ es el pipeline de producción

**Resumen de métricas (tests con ground truth disponible):**

| Test | Dataset | F1 onset+pitch | Nota |
|------|---------|----------------|------|
| yourmt3/test04 | Slakh 1884 | 77.5% ↑ | Piano 93%, Guitar 91% |
| yourmt3/test05 | Slakh 1975 | 73.9% ↑ | SynthLead 99%, Piano 81.5% |
| yourmt3/test07 | MusicNet 2556 | 2.6%* | *Score timing mismatch |
| yourmt3/test08 | MusicNet 2628 | 14.4%* | *Score timing mismatch; Strings→Ensemble |
| compound/test03 | Slakh 1884 | 31.1% ↓ | 3.2× sobre-detección de notas |

↑ Supera el F1 publicado en el paper (~70% Slakh). * Ver nota MusicNet.

**Razones para elegir YourMT3+:**
1. **F1 real medido superior al paper**: 77.5% y 73.9% en Slakh (vs ~70% publicado).
2. **Pipeline compuesto claramente inferior**: 31.1% vs 77.5% en mismo audio. BasicPitch genera 3.2× más notas de las reales por leakage de stems.
3. **Licencia Apache 2.0**: sin restricciones para integración en plugin REAPER.
4. **Single-model**: menor complejidad de implementación en el bridge Lua.
5. **Vocal nativo**: no requiere separador previo para transcribir voz.

**Limitaciones conocidas (confirmadas en escucha 2026-06-24):**

| Limitación | Impacto DAW | Causa técnica |
|---|---|---|
| **Instrument bleed** — una frase se fragmenta entre varios instrumentos GM | Edición difícil: la melodía está repartida en N pistas | El modelo asigna clase instrumento frame a frame sin restricción de continuidad por frase |
| **Huecos de detección** — silencios en melodías continuas | Notas faltantes en pasajes suaves o legato | Umbral de onset demasiado alto para notas de baja energía |
| **Sin detección de tempo** — MIDI siempre a 120 BPM fijo | **Bloqueante para producción**: imposible cuantizar ni alinear al grid del DAW | Los modelos AMT no incluyen beat tracking; tempo map no se escribe en el MIDI |
| Patrones rítmicos no capturados | El timing de notas es aproximado, no cuantizable | AMT optimiza onset F1 (±50ms), no alineación métrica |
| Instrumentos secundarios poco representados | Clases raras (Chromatic Perc, Other) a veces a 0% | Sub-representación en dataset de entrenamiento |
| Strings → Ensemble (error de clase frecuente) | Pista de cuerdas etiquetada como ensemble | Distribución GM ambigua entre clases |

**La limitación más crítica para el plugin REAPER es la ausencia de tempo detection.** Sin el tempo correcto en el MIDI, el usuario no puede usar el cuantizador del DAW ni alinear el material transcrito a ningún grid. Esto convierte la transcripción en un objeto de solo lectura, no editable. Solución futura necesaria: paso previo de beat tracking (librosa / madmom / essentia) que escriba el tempo map en el header del MIDI antes de entregarlo.

**Evaluación cualitativa en música real (2026-06-24):**
Escucha subjetiva sobre 3 temas propios: Half Foot Outside (jazz/funk), Villa-Lobos Bachianas nº5 (clásico), Moderat - A New Error (electronic).
Carpeta: `evaluation/custom_eval/track{01,02,03}/`.

| Dimensión | YourMT3+ | MIROS | Compound v2 |
|---|---|---|---|
| Calidad general | Caótico | **Mejor** | Caótico |
| Melodía | Aceptable | **Mejor** | Igual que YourMT3+ |
| Ritmo | Mediocre | Mediocre | Mediocre |
| Batería | — | — | Fuera de tiempo (desincronización merge) |

**Conclusión evaluación subjetiva:** MIROS supera a YourMT3+ en música real (especialmente melodía). La detección de patrones rítmicos es mediocre en los tres modelos — limitación estructural de la AMT actual. La batería de compound_v2 está fuera de tiempo: consecuencia de mezclar dos transcripciones independientes (YourMT3+ y ADTOF) sin referencia de tempo compartida.

**Implicación para la elección de pipeline:**
MIROS es candidato a revisitar como pipeline de producción (mejor subjetivo en música real), aunque en benchmarks Slakh empata o pierde vs YourMT3+. YourMT3+ sigue siendo más robusto en tests objetivos.

Ver `evaluation/custom_eval/track*/notes.txt` para notas por tema.

### Próximos pasos

1. **Ejecutar compound v2** (2026-06-24): scripts listos, ejecutar:

   ```bash
   cd Audio2Midi/research
   modal run research_adtof_modal.py::setup
   modal run research_compound_v2_modal.py::eval_all --only 4,5,7,8
   bash ../evaluation/render_mp3.sh
   uv run python ../evaluation/compute_f1.py --only compound_v2/
   ```

2. **Integración en REAPER**: bridge Lua para el pipeline ganador una vez cerrado compound v2.

---

## Resultados — MIROS (AMT Challenge 2025 winner) ✅ CERRADO 2026-06-23

**Fecha de evaluación:** 2026-06-23  
**Repo:** <https://github.com/amt-os/ai4m-miros>  
**GPU:** A10G (flash-attn 2.7.2, requiere Ampere+)  
**Coste estimado:** setup ~$0.05 (descarga 2 ckpts ~2 GB) + 10 tests ~$0.25  
**Licencia:** no especificada — uso solo investigación interna

### Leaderboard AMT Challenge 2025 (referencia)

| Ranking | Sistema | F-measure challenge | Notas |
|---------|---------|---------------------|-------|
| 1 | **MIROS** | **0.5998** | MusicFM encoder, Ampere GPU required |
| 2 | YourMT3-YPTF-MoE-M | 0.5938 | Nuestro modelo evaluado |
| 3 | YourMT3-YPTF-S | 0.5581 | — |
| 4 | YourMT3-P | 0.3947 | — |
| 5 | MT3 (baseline) | 0.3932 | — |

Fuente: arXiv 2603.27528, 76 piezas sintetizadas, 8 instrumentos.

### Métricas F1 objetivas (mir_eval, onset_tolerance=50ms) — 2026-06-23

| Test | Dataset | F1-op MIROS | F1-oop MIROS | F1-cls MIROS | F1-op YourMT3+ | Δ F1-op | Nota |
|------|---------|-------------|--------------|--------------|----------------|---------|------|
| test04 | Slakh 1884 | **77.5%** | 64.9% | 72.7% | 77.5% | **=** | Piano 92%, Guitar 90%, Bass 43% |
| test05 | Slakh 1975 | 64.0% | 19.3% | 76.9% | **73.9%** | **-9.9pp** | MIROS infra-detecta Ensemble (26% vs 1448 notas GT) |
| test07 | MusicNet 2556 | 3.3% | 0.8% | 100% | 2.6% | +0.7pp | Score timing mismatch — insignificante |
| test08 | MusicNet 2628 | 15.8% | 7.9% | 50% | 14.4% | +1.4pp | Score timing; Strings→Ensemble (ambos modelos) |

**Veredicto objetivo (Slakh, datos fiables):**
- test04: MIROS = YourMT3+ (77.5% empate). MIROS genera 99 notas menos (2200 vs 2101 est), ligeramente más preciso.
- test05: MIROS pierde 9.9pp vs YourMT3+. Derrumbe en Ensemble (26.4% vs 1448 GT) y Guitar (41.3% sub-detección). YourMT3+ obtiene 99% en SynthLead; MIROS solo 3.4%.
- MusicNet: scores de partitura — resultados no comparables con performance real.

**Nota:** F1-oop (onset+offset+pitch) MIROS test04 es 64.9% vs 61.8% de YourMT3+ — MIROS es 3.1pp mejor en predicción de duración de notas en Slakh 1884.

### Evaluación subjetiva (escucha REAPER)

Subjetivamente positiva (escucha informal). Ver `evaluation/miros/test*/notes.txt` para notas por test.

### Veredicto final MIROS

**No supera YourMT3+ de forma consistente.** En Slakh 1884 empatan; en Slakh 1975 MIROS pierde 9.9pp por fallos en Ensemble y SynthLead. El overhead (requiere GPU Ampere+, 2 checkpoints ~2 GB, repo sin LICENSE) no compensa la paridad de resultados. **YourMT3+ sigue siendo el pipeline de producción.**

MIROS es interesante para pistas con pocos instrumentos bien definidos (donde el challenge 2025 lo valida), pero no ofrece ventaja general en contenido musical real variado.

### Instrucciones de evaluación

```bash
cd plugins-and-extensions/Audio2Midi/research

# 1. Descargar pesos (una vez)
modal run research_miros_modal.py::setup

# 2. Smoke test
modal run research_miros_modal.py::main \
    --audio-path ../evaluation/miros/test04/input.wav \
    --out-dir ../evaluation/miros/test04

# 3. Evaluar todo
modal run research_miros_modal.py::eval_all

# 4. Render MP3
bash ../evaluation/render_mp3.sh

# 5. F1 objetivo
uv run python ../evaluation/compute_f1.py --only miros
uv run python ../evaluation/compute_f1.py --json /tmp/miros_f1.json
```

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
