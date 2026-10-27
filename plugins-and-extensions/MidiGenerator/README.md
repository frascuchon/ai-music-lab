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
| **Anticipatory** | Una melodía MIDI (basta con eso) | Acompañamiento multi-track (bajo, teclado y, a veces, más) | Genera el acompañamiento de tu melodía; Modal o local (CPU) |

Amadeus corre en Modal cloud (CUDA); Anticipatory puede ejecutarse en Modal o en local.

> text2midi, MuseCoco, MIDI-LLM y ChatMusician se retiraron de la interfaz tras la evaluación (calidad baja, tiempos excesivos o salida limitada). Sus scripts y notas siguen en `research/` y `RESEARCH.md`.

## Uso básico

### Modelo de texto (Amadeus)

1. Selecciona Amadeus en el dropdown.
2. Escribe el prompt en el campo de texto.
3. Rellena los campos opcionales (tonalidad, instrumentos, acordes; el BPM sale del proyecto) para resultados más predecibles.
4. Ajusta candidatos (1-4) y temperatura.
5. Pulsa **GENERAR MIDI**. El MIDI se importa como pistas nuevas.

### Acompañamiento de una melodía (Anticipatory)

1. Selecciona "Anticipatory" en el dropdown.
2. Selecciona el item MIDI con tu melodía en REAPER y pulsa **R** junto a «Melody».
3. Elige el **Style** del acompañamiento inicial (pop, pad o arpegio) y el número de candidatos.
4. Pulsa **GENERAR MIDI**. El resultado (solo el acompañamiento, no repite tu melodía) aparece en una carpeta justo debajo de la melodía, alineado con ella.

**Cómo funciona.** Anticipatory necesita unos segundos de acompañamiento previo para continuarlo; con una melodía sola arrancaba en frío y escribía casi nada (12 notas en una melodía de 17 s, 14 % de cobertura). El plugin construye ese arranque solo: detecta la tonalidad, elige un acorde cada dos tiempos y escribe un bajo y un teclado (`harmony.py`). Anticipatory lo continúa a lo largo de toda la melodía, siguiéndola. Con una melodía real: 89–99 % de cobertura y 4–6 notas por segundo.

- **Acompañamiento propio (opcional):** si añades pistas con **+** (por ejemplo un bajo que ya tengas), se usan como arranque en lugar del automático.
- **Duración:** sigue a la melodía; el silencio inicial se recorta y se restituye.
- **Cada pulsación da un resultado distinto** (semilla aleatoria).
- El acorde del arranque puede no ser el «correcto» (la armonía automática es un heurístico); el modelo lo corrige al continuar. Para alternativas, genera varios candidatos.

## Traducción de prompts por modelo

El plugin aplica adaptaciones rule-based por modelo (sin LLM externo):

- **Amadeus**: si rellenas los campos opcionales, el backend los ensambla en una caption estilo MidiCaps: `"<prompt>, featuring <instrumentos>, in <tonalidad>, at <BPM> BPM, with chord progression <acordes>."`.
- **Anticipatory**: sin prompt de texto; usa la melodía MIDI (y, opcionalmente, pistas de acompañamiento).

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
