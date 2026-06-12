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
│   ├── test1/  "A sad pop song with a strong piano presence."           [output_D.mid]
│   ├── test2/  "A rock song with strong drums and electric guitar."     [output_E.mid]
│   ├── test3/  "A soft love song on piano."                            [output_C.mid]
│   ├── test4/  "An energetic electronic trance track... 138 BPM A min" [output_A.mid]
│   ├── test5/  "A cheerful christmas song suitable for children."      [output_B.mid]
│   ├── test6/  "This short electronic song in C minor..."              [output_4.mid]
│   ├── test7/  "A heavy metal song with strong drums and guitar."      [output_F.mid]
│   ├── regenerate_all.sh   genera MIDIs locales vía research_text2midi.py (MPS, n=2)
│   └── research_text2midi_modal.py  genera en Modal A10G (float32, CUDA, n=2)
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
├── musecoco/              # Demos del paper MuseCoco (AAAI 2024)
│   ├── test1/  "jazz piano trio, 120 BPM" — sin referencia de audio directa
│   ├── test2/  ID 109 sax+drum+trombone — reference_official_109_v*.mp3
│   └── test3/  ID 2273 piano+drum+bass  — reference_official_2273_v*.mp3
│
└── chatmusician/          # Demo oficial https://ezmonyi.github.io/ChatMusician/
    ├── test01-03/  Chord conditioning — progresiones Am-F-C-G, Dm-C, D-G-C-B
    ├── test04-06/  Form conditioning — Binary/Ternary + Verse/Chorus/Bridge
    ├── test07-09/  Motif + form — desarrollo de motivos con estructura formal
    ├── test10-12/  Melody harmonization — armonización de melodías dadas
    ├── test*/input_abc.txt   ABC notation de entrada (solo tests 07-12)
    ├── test*/reference_demo.png        PNG del output del demo oficial (sin audio)
    ├── test*/reference_demo_input.png  PNG del input del demo (tests 07-12)
    ├── regenerate_all.sh   genera vía Modal A10G (abc2midi incluido en container)
    └── README.md           documentación detallada + criterios de evaluación
    NOTA: No hay MIDIs/MP3s de referencia descargables — solo PNGs visuales.
          El modelo genera ABC notation; conversión a MIDI via abc2midi (abcmidi).
```

---

## Cómo comparar

**text2midi (test1-7)**: escuchar `reference_official.mp3` (MIDI oficial del demo branch) y
`generated_cuda_v*.mp3` (nuestras generaciones en Modal A10G, float32, temperature=0.9).
La comparación mide si nuestro pipeline reproduce fielmente el comportamiento del modelo oficial.
- Las referencias son los ficheros `output_{A,B,C,D,E,F,4}.mid` del demo branch del repositorio
  (https://github.com/AMAAI-Lab/Text2midi, rama `demo`, carpeta `samples/`)
- Parámetros oficiales (HuggingFace Space app.py): `temperature=0.9`, `max_len=500–2000`, float32, CUDA

**Bugs corregidos en research_text2midi.py** (descubiertos durante setup del benchmark):
1. `temperature=1.0` → corregido a `0.9` (default de la app oficial)
2. `max_len=512` → corregido a `2000` (default razonable para piezas completas)
3. `half_precision=True` → corregido a `False` (la app oficial no usa half; fp16 en MPS causa deriva)

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
| text2midi | test1 | ✅ CUDA A10G | ~80s/output | sad pop piano — ref 3314B, v0 3283B, v1 3332B |
| text2midi | test2 | ✅ CUDA A10G | ~80s/output | rock guitar — ref 3328B, v0 3518B, v1 3304B |
| text2midi | test3 | ✅ CUDA A10G | ~80s/output | soft love song piano — ref 2597B, v0 3448B, v1 3336B |
| text2midi | test4 | ✅ CUDA A10G | ~82s/output | trance 138 BPM — ref 3240B, v0 3101B, v1 3326B |
| text2midi | test5 | ✅ CUDA A10G | ~81s/output | christmas children — ref 3335B, v0 3624B, v1 3373B |
| text2midi | test6 | ✅ CUDA A10G | ~83s/output | C minor electrónico — ref **8580B**, v0 3545B, v1 3461B |
| text2midi | test7 | ✅ CUDA A10G | ~80s/output | heavy metal — ref 2476B, v0 3364B, v1 3254B |
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

## Evaluación cualitativa — text2midi ✅ Completada (2026-06-12) — MODELO DESCARTADO

### Pipeline de generación

```
Modal A10G (CUDA) · float32 · temperature=0.9 · max_len=2000
Modelo: amaai-lab/text2midi (pytorch_model.bin, 900 MB)
Referencia: demo branch samples/output_*.mid (7 MIDIs oficiales del paper)
```

### Hallazgo principal: los resultados son decepcionantes — incluyendo las referencias oficiales

La comparativa entre `reference_official.mp3` y `generated_cuda_v*.mp3` revela que ambos
conjuntos tienen calidad musical insuficiente para uso en producción. No es un problema de
nuestro pipeline: el propio paper reporta 4.62/7 en Musical Quality en su listening study
(vs 5.79 del ground truth MidiCaps). El modelo funciona correctamente — el límite es la
arquitectura y el enfoque.

### Observaciones por test

| Test | Prompt | Seguimiento | Observaciones |
|------|--------|-------------|---------------|
| test1 | "A sad pop song with a strong piano presence." | ✅ parcial | Piano presente, pero poco "pop" y poco "sad" — progresión mecánica |
| test2 | "A rock song with strong drums and electric guitar. The tempo is very fast." | ⚠️ débil | Sin drums reconocibles; "muy rápido" no se traduce en velocidad real |
| test3 | "A soft love song on piano." | ❌ ignorado | Genera piano + bajo + saxofón + armónica — no respeta "solo piano" |
| test4 | "...trance track... 138 BPM... A minor..." | ⚠️ parcial | Instrumentación aproximada, 138 BPM no garantizado |
| test5 | "A cheerful christmas song suitable for children." | ✅ aceptable | El más exitoso — evoca algo "navideño", aunque genérico |
| test6 | "...C minor... brass... sax... C7/E Eb6 Bbm6 124 BPM..." | ⚠️ parcial | Prompt muy detallado → pieza de ~3500B vs referencia de 8580B; el modelo no genera más contenido aunque el prompt sea más largo |
| test7 | "A heavy metal song with strong drums and guitar." | ❌ débil | Sin agresividad característica del metal; progresión monótona |

### Limitaciones estructurales del modelo

1. **Prompt corto = modelo libre = resultado genérico.** "A sad pop song" o "heavy metal" producen
   resultados intercambiables. El modelo no tiene masa suficiente para "interpretar" un género.

2. **Prompt largo ≠ pieza más larga.** test6 tiene el prompt más detallado (instrumentación
   explícita, BPM, progresión de acordes) pero genera ~3500 bytes igual que los demás.
   El límite de 2000 tokens REMI es estructural: corresponde a 20-60s de música según densidad,
   independientemente del prompt.

3. **Seguimiento débil de atributos específicos.** Tempo, tonalidad y progresión de acordes
   raramente se reflejan con precisión. CLAP score de 0.22 en el paper lo confirma objetivamente.

4. **Multi-track pero sin coherencia entre pistas.** Las pistas se generan token a token de forma
   autoregresiva — no hay mecanismo que garantice que el saxofón y el bajo estén en la misma tonalidad
   o toquen en el mismo tiempo.

### Contexto: ¿fue text2midi un modelo "serio"?

No en el sentido de herramienta de producción. Es una **contribución de investigación como baseline**:

- La aportación real del paper es el **dataset MidiCaps** (168k pares MIDI-caption), no el modelo.
  El modelo existe para demostrar que el dataset es útil para entrenamiento text-to-MIDI.
- El framing "primer modelo end-to-end text-to-MIDI" es posicionamiento de paper, no de producto.
  Cuando un paper establece un "primero", está abriendo la línea de investigación, no cerrándola.
- La arquitectura es deliberadamente modesta: encoder T5-base frozen + decoder custom de 18 capas,
  ~900MB total. No hay versiones más grandes disponibles públicamente.
- El listening study lo publica sin ambigüedad: 4.62/7. Es "aceptable" como baseline académico.

### ¿Qué habría que hacer para que funcionase bien?

La dirección correcta es la de **ChatMusician** (fine-tune de LLaMA con ABC notation) o
**MIDI-LLM** (Llama extendido con vocabulario MIDI): aprovechar la capacidad semántica de un LLM
real en lugar de construir una arquitectura ad hoc. ChatMusician es arquitectónicamente más sólido
que text2midi, pero ABC notation limita la polifonía y el multi-track — no es el formato adecuado
para una integración DAW seria. El problema de texto→MIDI multi-track coherente sigue abierto
a fecha de junio 2026.

### Veredicto: DESCARTADO ❌

text2midi queda descartado como componente del plugin MidiGenerator. No por fallos en nuestro
pipeline (que reproduce fielmente el comportamiento oficial) sino porque la calidad del modelo
es insuficiente para uso productivo incluso en condiciones óptimas.

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
