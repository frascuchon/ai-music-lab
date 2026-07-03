# MidiGenerator — Plugin REAPER

Genera MIDI desde texto o seed MIDI usando 6 modelos de IA evaluados en la nube (Modal CUDA).

## Instalación

1. Asegúrate de tener el directorio `shared/` junto a `MidiGenerator/`.
2. En REAPER: `Actions → Load ReaScript → MidiGenerator/MidiGenerator.lua`.
3. Ejecuta `shared/Setup.lua` para comprobar dependencias (uv, Modal CLI) y precargar pesos.

## Modelos disponibles

| Modelo | Tipo de input | Output | Notas |
|--------|--------------|--------|-------|
| **Amadeus** | Texto libre (MidiCaps recomendado) | Multi-track (1-13 pistas) | Mejor calidad multi-instrumento evaluada (~2.8/5) |
| **MIDI-LLM** | Texto libre | Multi-track (2-5 pistas) | Requiere CUDA — MPS colapsa a drums |
| **text2midi** | Texto libre | Multi-track (incoherente) | Baseline académico, calidad baja (~2/5) — incluido como referencia |
| **ChatMusician** | Instrucción + seed opcional | Mono-pista (lead sheet) | Limitado a melodía + acordes, sin multi-track |
| **MuseCoco** | Texto libre → atributos | Multi-track (5-6 pistas) | Stage-2 tarda ~11 min; instrumento override por clase |
| **Anticipatory** | Seed MIDI (obligatorio) | Multi-track (2-3 pistas) | Acompañamiento/continuación sobre MIDI existente |

Todos los modelos corren en Modal cloud (CUDA). No hay ejecución local.

## Uso básico

### Modelos de texto (Amadeus, MIDI-LLM, text2midi, ChatMusician, MuseCoco)

1. Selecciona el modelo en el dropdown.
2. Escribe el prompt en el campo de texto.
3. Para Amadeus/text2midi: rellena los campos opcionales (tonalidad, BPM, instrumentos, acordes) para resultados más predecibles.
4. Ajusta candidatos (1-4) y temperatura.
5. Pulsa **GENERAR MIDI**. El MIDI se importa automáticamente en la posición del cursor.

### Modo acompañamiento (Anticipatory)

1. Selecciona "Anticipatory" en el dropdown.
2. Prepara el seed MIDI:
   - Selecciona un item MIDI en REAPER y pulsa **R** (captura automática).
   - O pulsa **...** para cargar un fichero `.mid` desde disco.
3. El seed debe contener **≥5 segundos de historia** (no solo melodía) para resultados densos.
4. Selecciona el modo: **Acompañamiento** (fija la melodía y genera partes nuevas) o **Continuación** (extiende el clip).
5. Ajusta duración, pista de melodía (índice de canal) y número de candidatos.
6. Pulsa **GENERAR MIDI**.

### Armonización con ChatMusician

1. Selecciona "ChatMusician".
2. Escribe la instrucción (ej: `"Construct smooth-flowing chord progressions for the supplied music."`).
3. Activa **Armonizar seed MIDI** y carga un `.mid` o captúralo con R.
4. El modelo recibe el MIDI convertido a ABC notation como contexto.

## Traducción de prompts por modelo

El plugin aplica adaptaciones rule-based por modelo (sin LLM externo):

- **Amadeus / text2midi**: si rellenas los campos opcionales, el backend los ensambla en una caption estilo MidiCaps: `"<prompt>, featuring <instrumentos>, in <tonalidad>, at <BPM> BPM, with chord progression <acordes>."`.
- **MIDI-LLM**: passthrough — el script antepone `"You are a world-class composer..."` internamente.
- **ChatMusician**: passthrough — el script construye `"Human: {prompt} </s> Assistant: "`.
- **MuseCoco**: el backend infiere class-IDs de instrumentos del texto libre para el override de Stage-1 (ej: "piano" → clase 0, "sax" → clase 17).
- **Anticipatory**: sin prompt de texto; solo usa el seed MIDI.

## Parámetros Modal

Los modelos se ejecutan en Modal (cloud CUDA). Costes aproximados:

| GPU | Coste | Modelos |
|-----|-------|---------|
| A10G | ~$0.05/min | Amadeus, MIDI-LLM, text2midi, ChatMusician, Anticipatory |
| A100 | ~$0.14/min | MuseCoco (~11 min/generación) |

Precarga los pesos con `shared/Setup.lua → sección MidiGenerator` para evitar el tiempo de descarga en la primera generación.

## Estructura de ficheros

```
MidiGenerator/
├── MidiGenerator.lua      — Plugin REAPER (UI gfx + integración)
├── midigen.py             — Orquestador Python (lanza modal run)
├── prompt_adapters.py     — Traducción rule-based de prompts por modelo
├── research/
│   ├── research_amadeus_modal.py
│   ├── research_midi_llm_modal.py
│   ├── research_text2midi_modal.py
│   ├── research_chatmusician_modal.py
│   ├── research_musecoco_modal.py
│   ├── research_anticipatory_modal.py
│   └── ...
├── evaluation/            — Outputs y métricas del benchmark
└── RESEARCH.md            — Notas de evaluación comparativa
```

## Caveats

- **Amadeus**: la duración del MIDI generado no es controlable (5-291s para 1024 tokens).
- **ChatMusician**: salida mono-staff; no genera múltiples pistas para DAW.
- **MuseCoco**: ~11 min por generación usando spawn+poll interno.
- **Anticipatory + seed solo melodía**: si el seed no tiene historia de acompañamiento (≥5s), el resultado será esparso. Funciona mejor con seeds que incluyen melodía + batería + bajo en los primeros 5s.
