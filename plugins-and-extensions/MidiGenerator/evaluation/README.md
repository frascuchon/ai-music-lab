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
├── midi_llm/              # 12 samples del demo oficial https://midi-llm-demo.vercel.app
│   ├── comparison_1..12/  prompts + refs oficiales + generated_cuda_v{0..3} + generated_local_mps_v{0..3}
│   ├── _orphan_sunday_picnic/  test3 anterior (jazz Sunday picnic) — sin ref oficial
│   ├── download_demo_references.sh  descarga los 12 sets desde Vercel
│   ├── build_prompts.py             genera prompt.txt canónico en cada carpeta
│   ├── regenerate_all.sh            regenera MIDIs locales vía research_midi_llm.py (MPS)
│   ├── render_cuda_mp3.sh           renderiza generated_cuda_v*.mid → mp3 con MuseScore
│   └── research_midi_llm_modal.py   genera en Modal A10G vía vLLM (pipeline idéntico al demo)
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

**MIDI-LLM (comparison_1-12)**: cada carpeta tiene 3 capas de audio:
- `reference_official_bf16.mp3` — generación del demo oficial (referencia principal)
- `generated_cuda_v{0..3}.mp3` — nuestras generaciones en Modal A10G vía vLLM (pipeline idéntico al demo)
- `generated_local_mps_v{0..3}.mp3` — generaciones en MPS bfloat16 (calidad inferior, para contraste)

La comparación relevante es `reference_official_bf16` vs `generated_cuda_v*`.
Las versiones MPS presentan colapso a percusión (drums solistas) por diferencias numéricas
entre bfloat16 CUDA y bfloat16 MPS — documentado en el diagnóstico de `research_midi_llm_modal.py`.

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
| midi_llm | comparison_1..12 | ✅ CUDA A10G | ~20s/prompt×4 (414s total) | 48 MIDIs, pipeline idéntico al demo |
| midi_llm | comparison_1..12 | ✅ MPS (ref) | ~120-180s/gen | generaciones locales para contraste |
| midi_llm | _orphan_sunday_picnic | archivado | — | sin ref oficial — jazz Sunday picnic |
| amt | test1 | ✅ | carga 1.0s, inf 896.5s (CPU) | continuation 20s, fixture=prompt oficial (73 notas) |
| amt | test2 | ✅ (Modal A10G) | inf v0:151s v1:127s v2:231s | accompaniment 20s, 3 candidatos, input_fixture.mid (138 notas piano completo) |
| amt | test3 | ✅ (Modal A10G) | inf v0:1.4s v1:0.8s v2:0.7s | accompaniment 15s, 3 candidatos, fixture jazz Bb |
| musecoco | test1 | ✅ (Modal) | ~2 min (A100-40GB) | jazz piano |
| musecoco | test2 | ✅ (Modal) | ~2 min | ID 109 sax+drum |
| musecoco | test3 | ✅ (Modal) | ~2 min | ID 2273 piano+bass |

---

## Evaluación cualitativa — MIDI-LLM ✅ Completada (2026-06-12)

### Pipeline correcto: vLLM + CUDA BF16 (Modal A10G)

Las generaciones CUDA (`generated_cuda_v*`) son comparables en calidad compositiva a
las referencias del demo (`reference_official_bf16.mp3`): instrumentación variada, coherencia
rítmica, y seguimiento del prompt (género, tonalidad, estado de ánimo).

### Diagnóstico: MPS bfloat16 vs CUDA bfloat16

Las generaciones MPS (`generated_local_mps_v*`) presentan colapso a percusión (drums
solistas repetidos) porque MPS implementa las operaciones bfloat16 de forma numéricamente
distinta a CUDA. El modelo fue entrenado en CUDA BF16 — las diferencias se acumulan en las
16 capas de atención y colapsan la distribución de probabilidad hacia los tokens de percusión.

El problema **no se resuelve** cambiando a float16 o float32 en MPS (ambos empeoran el resultado).
La única solución viable es CUDA, implementada vía Modal.

### Pipeline de referencia confirmado

```
vLLM 0.11.0 + torch 2.8.0 + CUDA BF16
temperature=1.0  top_p=0.98  max_tokens=2046  n=4
allowed_token_ids=[128256..183281]  (tokens MIDI únicamente)
```

### Valoración general del modelo

- Buena respuesta a prompts detallados (tonalidad, instrumentos, mood, tempo)
- La instrumentación específica (pizzicato strings, church organ, etc.) se refleja
  con fiabilidad cuando el pipeline es correcto (CUDA)
- Tempo "slow/relaxing" mejora notablemente respecto a MPS: duraciones más largas,
  notas más espaciadas
- Límite estructural: máx 682 tripletes (2046 tokens) → 20-100s según densidad.
  Piezas "361 seconds" del prompt son imposibles con el modelo actual
- No hay modelo más grande disponible públicamente (solo 1B)

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
- `midi_llm`: MPS bfloat16 produce colapso a percusión por diferencias numéricas con CUDA BF16
  (entorno de entrenamiento). Solución: Modal A10G vía `research_midi_llm_modal.py`.
  vLLM 0.6.4 incompatible con el tokenizer extendido (183K vocab) → usar vLLM 0.11.0.
- `musecoco`: requiere Modal (CUDA, fairseq) — no ejecutable en Mac local
