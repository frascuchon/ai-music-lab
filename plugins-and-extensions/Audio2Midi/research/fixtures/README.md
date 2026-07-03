# Fixtures de audio para evaluación Audio2Midi

Esta carpeta contiene los audios de referencia usados para los PoC de transcripción.

## Archivos esperados

No se incluyen audios en el repositorio (binarios grandes). Añadir manualmente:

| Archivo | Descripción | Por qué |
|---|---|---|
| `multitracks_short.wav` | Clip multi-instrumento ~30s (pop/rock o jazz) con batería, bajo, piano y guitarras | Caso general — evalúa capacidad multi-instrumento end-to-end |
| `synthetic_from_midi.wav` | Audio generado renderizando un MIDI conocido (ej. un MIDI de Lakh) con soundfont | Ground truth perfecto para métricas F1 — saber exactamente qué notas deben transcribirse |
| `piano_vocals.wav` | Voz acompañada de piano, ~30s | Probar transcripción vocal nativa de YourMT3+ (punto diferencial vs MT3) |

## Cómo generar `synthetic_from_midi.wav`

```bash
# Con FluidSynth y un soundfont GM:
fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 input.mid \
    -F synthetic_from_midi.wav -r 44100

# O con timidity:
timidity input.mid -Ow -o synthetic_from_midi.wav
```

## Formato esperado por los scripts

- Formato: WAV (PCM 16-bit o float32)
- Sample rate: cualquiera — los modelos re-muestrean internamente a 16kHz o 22kHz
- Canales: mono o estéreo — los modelos convierten a mono antes de procesar
- Duración: idealmente 20-60s para evaluación rápida; audios más largos en benchmarks formales
