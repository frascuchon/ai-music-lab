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
| amt | test2 | ✅ | carga 0.9s, inf 991.4s (~16.5min, CPU) | accompaniment 20s, fixture canónico (852 notas: piano+drums+acomp) |
| amt | test3 | ✅ | carga 0.7s, inf v0:8.5s v1:8.7s v2:8.7s (CPU) | accompaniment 15s, fixture jazz Bb, 3 candidatos |
| musecoco | test1 | ✅ (Modal) | ~2 min (A100-40GB) | jazz piano |
| musecoco | test2 | ✅ (Modal) | ~2 min | ID 109 sax+drum |
| musecoco | test3 | ✅ (Modal) | ~2 min | ID 2273 piano+bass |

---

## Evaluación cualitativa — AMT

### Continuation (test1) ✅ Buena calidad
La continuación de 20s funciona bien. El resultado es coherente musicalmente con el prompt
de entrada y puede considerarse de calidad aceptable para uso en producción. El modelo
mantiene el estilo, la tonalidad y el ritmo del fragmento original.

### Accompaniment (test2) ✅ Re-generado con fixture canónico (2026-06-11)

**Dos bugs identificados y corregidos**:

1. `start_time=0` → `prompt_length=5`: el modelo requiere ≥5s de contexto previo de
   acompañamiento (confirmado por [issue #18](https://github.com/jthickstun/anticipation/issues/18)
   y [`humaneval/accompany.py`](https://github.com/jthickstun/anticipation/blob/main/humaneval/accompany.py)).

2. **Fixture incorrecto**: el fixture original solo tenía piano (138 notas, melody) + 4 drum notes.
   Le faltaban las 60 notas del track `continuation` en t=3.5-5s (acompañamiento histórico).
   Sin este contexto rico, el modelo generaba output esparso (~2 notas/s). Con el fixture
   canónico (35 notas piano + 18 drums + 60 acompañamiento = 113 eventos), genera ~46 notas/s.

Pipeline correcto (usando `input_fixture_paper.mid` = primeros 5s de todos los instrumentos):
- `prompt_length=5`, `top_p=0.95`, `seed=0`
- `clip(combine(...), 0, clip_length)` para recortar correctamente

**Resultado**: `generated.mid` — 852 notas totales en 20s:
  - ch=0 prog=0: 35 notas (melody/controls 0-5s)
  - ch=9: 127 drum hits (batería densa durante todo el clip)
  - ch=2 prog=1: 690 notas de acompañamiento

**Valoración**: pendiente de escucha manual. Se espera calidad comparable al `reference_official_accompaniment.mp3`.
El archivo `generated_pre_fix.*` conserva el output del pipeline roto (143 notas) para comparación.

⚠️ **CPU: ~16.5min por clip de 20s** — Modal (GPU) es necesario para uso práctico en REAPER.

### Accompaniment (test3) 🔄 Re-generado con pipeline canónico (2026-06-11)

Fixture jazz Bb mayor (piano + walking bass). Con `prompt_length=5` el walking bass proporciona
contexto de 5s para el modelo. Tres candidatos con variaciones menores de seed.

3 candidatos (generated_v{0,1,2}.{mid,mp3}), todos con ~46 notas en 2 tracks (bass+piano), 15s.
**Valoración**: pendiente de escucha manual.

---

## Dependencias descubiertas (issues fijados durante evaluación)

- `text2midi`: faltaban `jsonlines`, `st-moe-pytorch`, `accelerate`, `spacy` en el `.venv`
  → Instalados via `uv pip install`. La centinela `.deps_installed` se había creado antes
  de instalar todas las deps; en producción habrá que incluirlas en `pyproject.toml`.
- `amt`: sin issues (CPU, float32, funciona en Mac)
- `midi_llm`: sin issues (MPS, bfloat16, funciona en M1+)
- `musecoco`: requiere Modal (CUDA, fairseq) — no ejecutable en Mac local
