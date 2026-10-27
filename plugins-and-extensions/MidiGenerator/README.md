# MidiGenerator — Plugin REAPER

Genera MIDI desde texto (Amadeus) o a partir de un MIDI semilla (Anticipatory) usando modelos de IA en la nube (Modal CUDA).

## Instalación

1. Asegúrate de tener el directorio `shared/` junto a `MidiGenerator/`.
2. En REAPER: `Actions → Load ReaScript → MidiGenerator/MidiGenerator.lua`.
3. Ejecuta `shared/Setup.lua` para comprobar dependencias (uv, Modal CLI) y precargar pesos.

## Modelos disponibles

| Modelo | Tipo de input | Output | Notas |
|--------|--------------|--------|-------|
| **Amadeus** | Texto libre (MidiCaps recomendado) | Multi-track (1-13 pistas) | Mejor calidad multi-instrumento evaluada (~2.8/5) |
| **Anticipatory** | Seed MIDI (obligatorio) | Multi-track (2-3 pistas) | Acompañamiento/continuación sobre MIDI existente; también puede ejecutarse en local (CPU, lento) |

Amadeus corre en Modal cloud (CUDA); Anticipatory puede ejecutarse en Modal o en local.

> text2midi, MuseCoco, MIDI-LLM y ChatMusician se retiraron de la interfaz tras la evaluación (calidad baja, tiempos excesivos o salida limitada). Sus scripts y notas siguen en `research/` y `RESEARCH.md`.

## Uso básico

### Modelo de texto (Amadeus)

1. Selecciona Amadeus en el dropdown.
2. Escribe el prompt en el campo de texto.
3. Rellena los campos opcionales (tonalidad, instrumentos, acordes; el BPM sale del proyecto) para resultados más predecibles.
4. Ajusta candidatos (1-4) y temperatura.
5. Pulsa **GENERAR MIDI**. El MIDI se importa como pistas nuevas.

### Modo acompañamiento (Anticipatory)

1. Selecciona "Anticipatory" en el dropdown.
2. Prepara el seed MIDI:
   - Selecciona un item MIDI en REAPER y pulsa **R** (captura automática).
   - O pulsa **...** para cargar un fichero `.mid` desde disco.
3. El seed debe contener **≥5 segundos de historia** (no solo melodía) para resultados densos.
4. Selecciona el modo: **Acompañamiento** (fija la melodía y genera partes nuevas) o **Continuación** (extiende el clip).
5. Ajusta duración, pista de melodía (índice de canal) y número de candidatos.
6. Pulsa **GENERAR MIDI**.

## Traducción de prompts por modelo

El plugin aplica adaptaciones rule-based por modelo (sin LLM externo):

- **Amadeus**: si rellenas los campos opcionales, el backend los ensambla en una caption estilo MidiCaps: `"<prompt>, featuring <instrumentos>, in <tonalidad>, at <BPM> BPM, with chord progression <acordes>."`.
- **Anticipatory**: sin prompt de texto; solo usa el seed MIDI.

## Parámetros Modal

Los modelos se ejecutan en Modal (cloud CUDA). Costes aproximados:

| GPU | Coste | Modelos |
|-----|-------|---------|
| A10G | ~$0.05/min | Amadeus, Anticipatory |

Precarga los pesos con `shared/Setup.lua → sección MidiGenerator` para evitar el tiempo de descarga en la primera generación.

## Estructura de ficheros

```
MidiGenerator/
├── MidiGenerator.lua      — Plugin REAPER (UI gfx + integración)
├── midigen.py             — Orquestador Python (lanza modal run)
├── prompt_adapters.py     — Traducción rule-based de prompts por modelo
├── research/
│   ├── research_amadeus_modal.py
│   ├── research_anticipatory_modal.py
│   └── ...
├── evaluation/            — Outputs y métricas del benchmark
└── RESEARCH.md            — Notas de evaluación comparativa
```

## Caveats

- **Amadeus**: la duración del MIDI generado no es controlable (5-291s para 1024 tokens).
- **Anticipatory + seed solo melodía**: si el seed no tiene historia de acompañamiento (≥5s), el resultado será esparso. Funciona mejor con seeds que incluyen melodía + batería + bajo en los primeros 5s.
