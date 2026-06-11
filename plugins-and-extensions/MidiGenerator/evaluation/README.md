# MidiGenerator — Evaluación de scripts con referencias oficiales

Cada subcarpeta contiene:
- `prompt.txt` — descripción del test, fuente del prompt, comando exacto
- `reference_official.*` — audio o MIDI del proyecto/paper oficial (fuente de verdad)
- `input_fixture.*` — MIDI de entrada (solo AMT, para comparar input vs. output)
- `generated.mid` — MIDI generado por nuestro script con el mismo prompt
- `generated.mp3` — audio renderizado con MuseScore 4 (MS Basic.sf3)

---

## Estructura y fuentes de referencia

```
evaluation/
├── text2midi/             # MIDI demos de https://amaai-lab.github.io/Text2midi/
│   ├── test1/  "A sad pop song with a strong piano presence."        [Demo D]
│   ├── test2/  "A rock song with strong drums and electric guitar."  [Demo E]
│   └── test3/  "A soft love song on piano."                         [Demo C]
│
├── midi_llm/              # Prompts de assets/example_prompts.txt (repo oficial)
│   ├── test1/  Rock con sintetizadores en A menor, 4/4 — prompt #1
│   ├── test2/  Pieza clásica con órgano y corno francés — prompt #2
│   └── test3/  "Upbeat and playful jazz music with lively saxophones" — prompt #4
│
├── amt/                   # Demos de https://crfm.stanford.edu/2023/06/16/...
│   ├── test1/  Continuation (20s) — referencia del sitio: prompt/system4/0-clip-v0
│   ├── test2/  Accompaniment (20s) — referencia del sitio: demos/system0/0-clip-v0
│   └── test3/  Accompaniment (15s) — fixture jazz Bb, piano+walking bass
│
└── musecoco/              # Demos del paper MuseCoco (AAAI 2024)
    ├── test1/  "jazz piano trio, 120 BPM" — sin referencia de audio directa
    ├── test2/  ID 109 sax+drum+trombone — reference_official_109_v*.mp3
    └── test3/  ID 2273 piano+drum+bass  — reference_official_2273_v*.mp3
```

---

## Cómo comparar

**text2midi (test1-3)**: escuchar `reference_official.mp3` (generado por la implementación
oficial) y `generated.mp3` (nuestro script). Ambos derivan del mismo MIDI-to-audio pipeline
(MuseScore), por lo que la comparación es directa en estructura melódica y armónica.

**MIDI-LLM (test1-3)**: no hay audio de referencia pre-generado (el demo es interactivo).
Comparar que `generated.mid` es un MIDI válido, multi-track, y que el estilo del prompt
se refleja en la instrumentación y estructura.

**AMT (test1-3)**: el sitio oficial publica audio de demos con el modelo real. Comparar
`reference_official_continuation.mp3` / `reference_official_accompaniment.mp3` contra
`generated.mp3`. Los fixtures de entrada son distintos en cada test y relacionados con
la referencia correspondiente:
- test1: `input_fixture.mid` extraído del track `prompt` de `reference_official_continuation.mid`
  → MISMO input que usó el modelo oficial
- test2: `input_fixture.mid` extraído de los tracks `prompt` + `prompt_drums` de
  `reference_official_accompaniment.mid` → MISMO input que usó el modelo oficial
- test3: fixture jazz swing Bb mayor 100 BPM, distinto a test1 y test2

**MuseCoco (test2, test3)**: comparación directa — `reference_official_*.mp3` son audios
del dataset de entrenamiento original (IDs 109 y 2273 del paper). `generated.mp3` es el
MIDI que nuestro script Modal generó para la misma descripción textual.

---

## Resultados de ejecución

| Script | Test | Estado | Timing | Notas |
|--------|------|--------|--------|-------|
| text2midi | test1 | generado | ~400s (2000 tokens, MPS) | ver prompt.txt |
| text2midi | test2 | generado | ~400s | ver prompt.txt |
| text2midi | test3 | generado | ~400s | ver prompt.txt |
| midi_llm | test1 | generado | 148.4s (2046 tokens, MPS) | 5 pistas, 682 notas |
| midi_llm | test2 | generado | ~150s | ver prompt.txt |
| midi_llm | test3 | generado | ~150s | ver prompt.txt |
| amt | test1 | ✅ | carga 1.0s, inf 896.5s (CPU) | continuation 20s, fixture=prompt oficial (73 notas) |
| amt | test2 | ✅ (Modal A10G) | inf v0:151s v1:127s v2:231s | accompaniment 20s, 3 candidatos, input_fixture.mid (138 notas piano completo) |
| amt | test3 | ✅ (Modal A10G) | inf v0:1.4s v1:0.8s v2:0.7s | accompaniment 15s, 3 candidatos, fixture jazz Bb |
| musecoco | test1 | ✅ (Modal) | ~2 min (A100-40GB) | jazz piano |
| musecoco | test2 | ✅ (Modal) | ~2 min | ID 109 sax+drum |
| musecoco | test3 | ✅ (Modal) | ~2 min | ID 2273 piano+bass |

---

## Evaluación cualitativa — AMT

### Continuation (test1) ✅ Buena calidad
La continuación de 20s funciona bien. El resultado es coherente musicalmente con el prompt
de entrada y puede considerarse de calidad aceptable para uso en producción. El modelo
mantiene el estilo, la tonalidad y el ritmo del fragmento original.

### Accompaniment (test2) ✅ Re-generado con Modal A10G (2026-06-11)

**Dos bugs identificados y corregidos**:

1. `start_time=0` → `prompt_length=5`: el modelo requiere ≥5s de contexto previo de
   acompañamiento (confirmado por [issue #18](https://github.com/jthickstun/anticipation/issues/18)
   y [`humaneval/accompany.py`](https://github.com/jthickstun/anticipation/blob/main/humaneval/accompany.py)).

2. **Fixture incorrecto (primera corrección)**: `input_fixture_paper.mid` solo tenía los primeros
   5s de piano — la melodía desaparecía del output tras t=5s. Corregido usando `input_fixture.mid`
   (variante REAPER, 138 notas de piano durante los 29.3s completos como controles de melodía).

**Resultados** (3 candidatos, `input_fixture.mid`, Modal A10G, `music-medium-800k`):
- v0 (seed=0): 3 pistas | 748 notas | 20.0s | inferencia 151s
- v1 (seed=1): 3 pistas | 672 notas | 20.0s | inferencia 127s
- v2 (seed=2): 3 pistas | 1013 notas | 20.0s | inferencia 231s

**Valoración**: pendiente de escucha manual. Comparar `generated_v{0,1,2}.mp3` contra
`reference_official_accompaniment.mp3`.

### Accompaniment (test3) ✅ Re-generado con Modal A10G (2026-06-11)

Fixture jazz Bb mayor (piano + walking bass). Con `prompt_length=5` el walking bass proporciona
contexto de 5s para el modelo.

**Resultados** (3 candidatos, `input_fixture.mid`, Modal A10G, `music-medium-800k`):
- v0 (seed=0): 2 pistas | 45 notas | 15.0s | inferencia 1.4s
- v1 (seed=1): 2 pistas | 44 notas | 15.0s | inferencia 0.8s
- v2 (seed=2): 2 pistas | 45 notas | 15.0s | inferencia 0.7s

**Valoración**: pendiente de escucha manual. Comparar `generated_v{0,1,2}.mp3` contra
`reference_official_accompaniment.mp3` y el estilo jazz del fixture de entrada.

---

## Dependencias descubiertas (issues fijados durante evaluación)

- `text2midi`: faltaban `jsonlines`, `st-moe-pytorch`, `accelerate`, `spacy` en el `.venv`
  → Instalados via `uv pip install`. La centinela `.deps_installed` se había creado antes
  de instalar todas las deps; en producción habrá que incluirlas en `pyproject.toml`.
- `amt`: sin issues (CPU, float32, funciona en Mac)
- `midi_llm`: sin issues (MPS, bfloat16, funciona en M1+)
- `musecoco`: requiere Modal (CUDA, fairseq) — no ejecutable en Mac local
