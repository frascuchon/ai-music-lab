# MIDI Generator — Investigación de modelos

**Objetivo**: evaluar qué modelos encajan mejor en un plugin REAPER de generación MIDI con dos flujos:
1. Generación desde 0 (texto → MIDI)
2. Variaciones/arreglos sobre MIDI dado

**Hardware objetivo**: Mac Apple Silicon + Modal cloud fallback  
**Licencia**: uso personal/no comercial (cualquier licencia open source válida)

---

## Tabla comparativa de candidatos

| Modelo | Flujo soportado | MIDI nativo | Mac (MPS/CPU) | Licencia | Multi-track | Madurez | Veredicto |
|---|---|---|---|---|---|---|---|
| **Text2midi** (AMAAI-Lab) | Text→MIDI | ✅ sí | ✅ MPS oficial | MIT | ⚠️ ver notas | ✅ AAAI 2025, HF | **ELEGIDO flujo 1** |
| **Anticipatory MT** (Stanford) | Variaciones/infilling | ✅ sí | ⚠️ CPU (lento) | Apache 2.0 | ✅ sí | ✅ publicado, pesos abiertos | **ELEGIDO flujo 2** |
| **MuseCoco** (Microsoft) | Text→MIDI con atributos | ✅ sí | ⚠️ stack fairseq pesado | MIT | ✅ sí | ✅ activo en muzic repo | Alternativa flujo 1 (Modal) |
| **Music Transformer** (Magenta) | Continuación piano | ✅ sí | ✅ TF2 funciona en CPU | Apache 2.0 | ❌ piano solo | ⚠️ legado (~2018) | Descartado (legado, mono) |
| **Foundation-1** (RoyalCities) | Text→Audio (MIDI post-hoc) | ❌ MIDI extraído | ❌ CUDA 7GB VRAM | Stability AI | ✅ multi-inst | ⚠️ requiere wrapper externo | Descartado (no MIDI nativo) |
| **Amadeus** (2025) | Text→MIDI simbólico | ✅ sí | ? (por confirmar) | ? | ✅ sí | ⚠️ reciente, repo pendiente | Candidato v2 si Text2midi falla |
| **MMM** (Multi-track Machine) | Infilling multi-track | ✅ sí | ✅ | ? | ✅ sí | ⚠️ antiguo (2020) | Baseline; superado por AMT |

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

#### Resultados PoC (rellenar tras ejecutar research_text2midi.py)

| Métrica | Valor |
|---|---|
| device | ___ |
| tiempo carga (s) | ___ |
| tiempo inferencia (s) | ___ |
| RAM delta (MB) | ___ |
| MIDI válido | ✅ / ❌ |
| Calidad subjetiva (0-5) | ___ |
| Notas | ___ |

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

#### Resultados PoC (rellenar tras ejecutar research_amt.py)

| Métrica | Valor |
|---|---|
| device | ___ |
| tiempo carga (s) | ___ |
| tiempo inferencia continuation (s) | ___ |
| tiempo inferencia accompaniment (s) | ___ |
| RAM delta (MB) | ___ |
| MIDIs válidos | ✅ / ❌ |
| Calidad subjetiva continuation (0-5) | ___ |
| Calidad subjetiva accompaniment (0-5) | ___ |
| Coherencia con melodía de entrada | ___ |
| Notas | ___ |

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

### PoC 1 — Text2midi
```bash
cd plugins-and-extensions/MidiGenerator/research
uv run research_text2midi.py --prompt "upbeat pop song in C major, 120 BPM, piano and strings" --out out_t2m.mid
# Verificar: abrir out_t2m.mid en REAPER y escuchar
```

### PoC 2 — Anticipatory Music Transformer
```bash
# genera fixture + continuación + acompañamiento
uv run research_amt.py --mode both

# o por separado:
uv run research_amt.py --mode continuation --input fixtures/melody.mid --out out_cont.mid
uv run research_amt.py --mode accompaniment --input fixtures/melody.mid --out out_acc.mid
# Verificar: abrir los .mid en REAPER y escuchar
```

### Validación en REAPER
1. Arrastrar cada `.mid` generado a una pista de REAPER
2. Escuchar y anotar calidad subjetiva (0-5) en las tablas de resultados arriba
3. Para AMT: verificar que `out_acc.mid` suena coherente con `fixtures/melody.mid`

---

*Documento generado: 2026-06-01. Actualizar tras ejecutar los PoC.*
