# Amadeus — Benchmark de evaluación

Modelo: [`longyu1315/Amadeus-S`](https://huggingface.co/longyu1315/Amadeus-S) (arXiv [2508.20665](https://arxiv.org/abs/2508.20665))

Repo: [lingyu123-su/Amadeus](https://github.com/lingyu123-su/Amadeus)  
Arquitectura: transformer autoregresivo (NB encoding, 768 dim, 16 layers) + decoder de difusión bidireccional para atributos intra-nota. T5-base como encoder de texto.  
Dataset: LakhALLFined (subconjunto filtrado del Lakh MIDI Dataset).

---

## Parámetros de generación

Verbatim de `demo/Amadeus_app_EN.py`:

```python
threshold = 0.99          # top_p cutoff
temperature = 1.25        # aleatoriedad (rango demo: 0.5-3.0)
generation_length = 1024  # tokens (rango demo: 256-3072)
sampling_method = "top_p"
text_encoder = "google/flan-t5-base"
```

---

## Tests

| Test | Categoría | Prompt (resumen) | Base para comparación |
|------|-----------|------------------|----------------------|
| test01 | official-example | Electronic ambient, E major, tubular bells, Andante | — |
| test02 | official-example | Electronic dreamy, B minor, drums, piano, brass, sax | — |
| test03 | official-example | Soothing pop, C major, piano, flute, violin, Andante | — |
| test04 | official-example | Rock/pop, A minor, pizzicato strings, 148 BPM | — |
| test05 | cross-model | Trance electrónico, A minor, 138 BPM, 4/4 (structured) | text2midi/test4 |
| test06 | cross-model | Electronic C minor, 124 BPM, chord C7/E-Eb6-Bbm6 | text2midi/test6 |
| test07 | cross-model | "A sad pop song with a strong piano presence." (short) | text2midi/test1 |
| test08 | cross-model | "A cheerful christmas song suitable for children." | text2midi/test5 |

**Hipótesis**: los tests 01-04 (prompts oficiales, estilo MidiCaps) deben mostrar la calidad máxima del modelo. Los tests 05-06 (prompts estructurados) deben ser comparables. Los tests 07-08 (prompts cortos/abstractos) testean la robustez ante prompts informales.

---

## Cómo ejecutar

### Prerrequisito: setup Modal (una vez)

```bash
cd MidiGenerator/research
modal run research_amadeus_modal.py::setup
```

### Benchmark completo (recomendado — batch único)

```bash
cd MidiGenerator/research
modal run research_amadeus_modal.py::eval_all \
    --eval-dir ../evaluation/amadeus \
    --n-outputs 2
```

### Test individual

```bash
cd MidiGenerator/research
modal run research_amadeus_modal.py::main \
    --prompt "A melodic electronic ambient song with a touch of darkness, set in the key of E major and a 4/4 time signature. Tubular bells, electric guitar, synth effects, synth pad, and oboe weave together to create an epic, space-like atmosphere. The tempo is a steady Andante, and the chord progression of A, B, and E forms the harmonic backbone of this captivating piece." \
    --out-dir ../evaluation/amadeus/test01
```

### Temperatura más alta (más variedad)

```bash
cd MidiGenerator/research
modal run research_amadeus_modal.py::eval_all \
    --eval-dir evaluation/amadeus \
    --temperature 1.5 \
    --n-outputs 2
```

### Script bash completo

```bash
bash evaluation/amadeus/regenerate_all.sh
```

---

## Criterios de evaluación

### Tests 01-04 (official examples)
- ¿La instrumentación mencionada en el prompt está presente?
- ¿La tonalidad y el tempo se reflejan en el MIDI?
- ¿La progresión de acordes es reconocible?
- ¿La duración es comparable a la pedida (test02: "252 seconds")?
- ¿La calidad musical es superior a text2midi y MIDI-LLM?

### Tests 05-06 (cross-model, structured)
- Comparar directamente con evaluation/text2midi/test4 y test6
- ¿Amadeus sigue mejor los atributos específicos (BPM, tonalidad, acordes)?
- ¿La calidad multi-track es más coherente?

### Tests 07-08 (cross-model, short prompts)
- ¿El modelo genera algo musicalmente coherente con el prompt?
- ¿La calidad supera a text2midi (2/5 y 2.5/5)?
- ¿Demuestra que Amadeus es más robusto ante prompts informales?

---

## Output por test

Cada test generará:
- `generated_cuda_v0.mid` — primera variante
- `generated_cuda_v1.mid` — segunda variante

Para renderizar MP3 localmente (requiere `fluidsynth`):
```bash
brew install fluidsynth fluid-soundfont-gm
fluidsynth -ni -g 0.5 /usr/local/share/sounds/sf2/GeneralUser\ GS.sf2 \
    evaluation/amadeus/test01/generated_cuda_v0.mid \
    -F evaluation/amadeus/test01/generated_cuda_v0.mp3
```

---

## Contexto en el benchmark cross-model

| Modelo | test05 equiv. | test06 equiv. | test07 equiv. | test08 equiv. |
|--------|--------------|--------------|--------------|--------------|
| text2midi | test4: 2/5 | test6: 2/5 | test1: 2/5 | test5: 2.5/5 |
| Amadeus | *pendiente* | *pendiente* | *pendiente* | *pendiente* |
