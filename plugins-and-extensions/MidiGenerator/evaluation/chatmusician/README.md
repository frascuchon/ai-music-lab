# ChatMusician — Benchmark de evaluación

Modelo: [`m-a-p/ChatMusician`](https://huggingface.co/m-a-p/ChatMusician) (LLaMA 2 7B, continual pretraining + SFT sobre ABC notation, ISMIR 2024)

Demo oficial: [ezmonyi.github.io/ChatMusician](https://ezmonyi.github.io/ChatMusician/)

---

## Particularidades del benchmark

### Formato de salida: ABC notation, no MIDI

ChatMusician no genera MIDI directamente. Genera **ABC notation** (representación textual de partitura), que luego se convierte a MIDI mediante `abc2midi` (herramienta `abcmidi`, la misma usada por el web demo oficial). Por eso cada test produce:

- `generated_cuda_v0.abc` — notación ABC generada por el modelo
- `generated_cuda_v0.mid` — MIDI resultado de `abc2midi score.abc`
- `generated_cuda_v0.mp3` — renderizado con MuseScore (si se usó `--render-mp3`)

### Referencias: solo visuales (sin audio descargable)

El demo oficial publica únicamente **imágenes PNG** del score renderizado, sin archivos MIDI ni MP3 descargables — a diferencia de Text2midi (que tenía MIDIs oficiales en el repo). Por tanto:

- `reference_demo.png` — captura de pantalla del output del demo oficial
- `reference_demo_input.png` (donde aplica) — input mostrado en el demo

La evaluación es **cualitativa** comparando la partitura ABC generada con la imagen de referencia.

### Tests con condicionante ABC

Los tests de categorías **motif-form** (test07-09) y **melody-harmonization** (test10-12) incluyen un `input_abc.txt` que se concatena al prompt antes de enviarlo al modelo (siguiendo el comportamiento del web demo).

---

## Tests

| Test | Categoría | Prompt (resumen) | Input ABC |
|------|-----------|------------------|-----------|
| test01 | chord-conditioning | `'Am', 'F', 'C', 'G'` | — |
| test02 | chord-conditioning | `'Dm', 'C', 'Dm', 'Dm', 'C', 'Dm', 'C', 'Dm'` | — |
| test03 | chord-conditioning | `'D', 'G', 'C', 'B', 'C', 'D', 'D', 'B'` | — |
| test04 | form-conditioning | Binary, Sectional: Verse/Chorus (v1) | — |
| test05 | form-conditioning | Ternary, Sectional: Verse/Chorus/Bridge | — |
| test06 | form-conditioning | Binary, Sectional: Verse/Chorus (v2) | — |
| test07 | motif-form | Alphabetic AB + motivo D mayor 4/4 | ✓ |
| test08 | motif-form | Binary Verse/Chorus + motivo G mayor 3/4 | ✓ |
| test09 | motif-form | Only One Section + motivo C mayor 4/4 | ✓ |
| test10 | melody-harmonization | Armonizar melodía G mayor (sin acordes) | ✓ |
| test11 | melody-harmonization | Armonizar melodía D mayor (sin acordes) | ✓ |
| test12 | melody-harmonization | Ampliar armonización G mayor (acordes parciales) | ✓ |

---

## Cómo ejecutar

### Prerrequisito: setup Modal (una vez)

```bash
cd MidiGenerator/research
modal run research_chatmusician_modal.py::setup
```

### Benchmark completo (recomendado — batch único)

```bash
cd MidiGenerator/research
modal run research_chatmusician_modal.py::eval_all \
    --eval-dir ../evaluation/chatmusician \
    --n-outputs 2
```

### Test individual

```bash
cd MidiGenerator/research
modal run research_chatmusician_modal.py::main \
    --prompt "Develop a musical piece using the given chord progression. 'Dm', 'C', 'Dm', 'Dm', 'C', 'Dm', 'C', 'Dm'" \
    --out-dir ../evaluation/chatmusician/test02
```

### Con input condicional (tests 07-12) — auto-detecta .abc o .mid

```bash
cd MidiGenerator/research

# Desde un fichero ABC:
modal run research_chatmusician_modal.py::main \
    --prompt "Formulate chord combinations to increase the harmonic complexity of the specified musical excerpt." \
    --input-file ../evaluation/chatmusician/test10/input_abc.txt \
    --out-dir ../evaluation/chatmusician/test10

# Desde un fichero MIDI (se convierte automáticamente a ABC antes de enviar al modelo):
modal run research_chatmusician_modal.py::main \
    --prompt "Formulate chord combinations to increase the harmonic complexity of the specified musical excerpt." \
    --input-file ../evaluation/text2midi/test1/reference_official.mid \
    --out-dir /tmp/cm_harmonize_t2m
```

### Script bash (test a test, con render MP3)

```bash
bash evaluation/chatmusician/regenerate_all.sh --render-mp3
# Solo tests específicos:
ONLY=1,4,10 bash evaluation/chatmusician/regenerate_all.sh
```

### Herramientas MIDI↔ABC locales

```bash
brew install abcmidi   # instalar una vez (provee abc2midi y midi2abc)

# CLI:
python research/tools/midi_abc.py midi-to-abc input.mid output.abc
python research/tools/midi_abc.py abc-to-midi generated.abc generated.mid

# O como librería Python:
from tools.midi_abc import midi_to_abc_text, abc_to_midi_bytes
abc_str   = midi_to_abc_text("input.mid")   # → string ABC
mid_bytes = abc_to_midi_bytes(abc_str)       # → bytes MIDI
```

**Nota**: el pipeline integra estas conversiones automáticamente — pasar `--input-file input.mid` y el pipeline convierte MIDI→ABC antes de enviar al modelo.

---

## Criterios de evaluación

### Chord conditioning (tests 01-03)
- ¿El ABC generado contiene anotaciones de acorde `"[Am]"` / `"^(Am)"` en los compases correctos?
- ¿La secuencia de acordes respeta el orden del prompt?
- ¿La pieza tiene sentido musical más allá de respetar los acordes?

### Form conditioning (tests 04-06)
- ¿El ABC tiene secciones claramente diferenciadas (part markers, doble barra, cambio temático)?
- ¿Los nombres de sección (Verse/Chorus/Bridge) son reconocibles estructuralmente?
- ¿La longitud de la pieza es suficiente para mostrar la forma?

### Motif + form (tests 07-09)
- ¿El motivo de `input_abc.txt` aparece reconocible en la pieza generada?
- ¿Se respeta la estructura formal pedida (AB, Binary, Single section)?
- ¿Hay desarrollo temático real o el modelo repite el motivo sin variación?

### Melody harmonization (tests 10-12)
- ¿El ABC generado incluye la melodía original con acordes añadidos?
- ¿Los acordes son musicalmente correctos respecto a la melodía?
- ¿Se evitan paralelas de 5ª/8ª y otros errores básicos de armonía?

---

## Parámetros de generación

Verbatim del model card y `chatmusician_web_demo.py`:

```python
GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    min_new_tokens=10,
    max_new_tokens=1536,
)
```

Prompt template (verbatim de `model/infer/predict.py`):

```
Human: {instruction} </s> Assistant:
```
