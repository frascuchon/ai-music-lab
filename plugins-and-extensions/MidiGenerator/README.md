# MidiGenerator — Plugin REAPER

Genera MIDI desde texto (Amadeus) o a partir de un MIDI semilla (Anticipatory) usando modelos de IA en la nube (Modal CUDA).

## Instalación

Se instala con ReaPack como parte del plugin unificado **AI Music Lab** (pestaña MidiGenerator); ver la [guía de instalación](../../README.md#quick-install-about-10-minutes) del README principal. La configuración (uv, Modal, tokens) se hace desde el botón **Setup** del panel.

## Modelos disponibles

| Modelo | Tipo de input | Output | Notas |
|--------|--------------|--------|-------|
| **Amadeus** | Texto libre (MidiCaps recomendado) | Multi-track (1-13 pistas) | Mejor calidad multi-instrumento evaluada (~2.8/5) |
| **Anticipatory** | Una melodía MIDI (basta con eso) | Acompañamiento multi-track | Genera el acompañamiento de tu melodía; Modal o local (CPU) |
| **MIDI-GPT** | Una melodía MIDI | Acompañamiento multi-track | Alternativa con ritmo fijado por parte y partes que no se copian; solo Modal. **Pesos de uso no comercial** (acepta la licencia en https://huggingface.co/Metacreation/MIDI-GPT y guarda tu token en Setup) |

Amadeus y MIDI-GPT corren en Modal cloud (CUDA); Anticipatory puede ejecutarse en Modal o en local.

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

### Controles del acompañamiento

Son los mismos para Anticipatory y MIDI-GPT salvo donde se indica. Cada uno se midió en una melodía real de guitarra (17 s):

| Control | Qué hace | Efecto medido |
|---|---|---|
| **Parts** | Qué se escribe: bajo, teclado, cuerdas, guitarra, batería | Con batería aparecen ~5 golpes/s; banda completa (5 partes) en ambos motores |
| **Style** | Pop (acordes sincopados), pad (sostenido) o arpegio | Cambia el ritmo de bajo y teclado |
| **Density** | Escaso / normal / ocupado | Anticipatory 2,4 → 3,9 notas/s; MIDI-GPT 2,7 → 5,2 → 6,5 notas/s; la batería también sube |
| **Bass / Keys** | Timbre (programa General MIDI) | — |
| **Creativity** | Anticipatory: probabilidad acumulada (0,80–1,00); MIDI-GPT: temperatura (0,8–1,25) | Anticipatory: con valores altos el modelo añade instrumentos propios (2 → 3,5 instrumentos); MIDI-GPT: cambia sobre todo las notas elegidas, de forma más sutil |
| **Chords** (Anticipatory) | Tu propia progresión, p. ej. `C#m G# A# F`; *Beats/chord* fija cuánto dura cada acorde | Sustituye a la armonía detectada automáticamente, que es un heurístico |
| **Stay in the melody's notes** (MIDI-GPT) | Solo usa las notas de tu melodía | Con melodías pentatónicas apenas se nota: el modelo ya se queda dentro |
| **Prompt (s)** (Anticipatory) | Segundos de arranque que el modelo continúa | — |

En MIDI-GPT cada parte recibe su propio patrón rítmico: si dos partes comparten ritmo (o una lo tiene y otra no), el modelo hace que una sea copia transpuesta de la otra, y el acompañamiento suena a «dos instrumentos tocando lo mismo». Con ritmos distintos no hubo ninguna parte duplicada en 20 pruebas. El patrón supone compases de 4/4.

- **Acompañamiento propio (opcional, Anticipatory):** si añades pistas con **+** (por ejemplo un bajo que ya tengas), se usan como arranque en lugar del automático.
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

Precarga los pesos con el botón **Setup** para evitar el tiempo de descarga en la primera generación.

## Estructura de ficheros

```
MidiGenerator/
├── panel.lua              — Panel REAPER (UI gfx + integración)
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
