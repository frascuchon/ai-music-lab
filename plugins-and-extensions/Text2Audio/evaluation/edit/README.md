# Audio2Audio Editing Benchmark — Resultados Finales

## Resumen ejecutivo

Benchmark completado (6/6 modelos, 48 outputs + 2 continuation). Evaluación cualitativa aceptable
en todos los casos. **SAO 1.0 elegido como candidato para integración en DAW** por su combinación
de CLAP_audio y chroma más altos, confirmada por escucha subjetiva.

## Resultados (métricas automáticas)

| Modelo | Técnica | n | CLAP_text | CLAP_audio | chroma | Ranking |
|---|---|---|---|---|---|---|
| **SAO 1.0** | SDEdit (style transfer) | 10 | 0.224 | **0.643** | **0.838** | **1º** |
| ZETA (AudioLDM2) | DDIM inversion (zero-shot) | 10 | **0.343** | 0.452 | 0.809 | 2º |
| MelodyFlow | flow matching (text-guided) | 10 | 0.320 | 0.446 | 0.708 | 3º |
| InspireMusic | continuation (audio prompt) | 2 | 0.136 | 0.421 | 0.715 | — |
| ACE-Step 1.5 | cover (DiT turbo) | 10 | 0.232 | 0.345 | 0.625 | 4º |
| MusicGen-melody | melody conditioning | 10 | 0.002 | 0.095 | 0.644 | 5º |

Métricas completas por caso en `metrics_edit.json`.

### Interpretación

- **CLAP_audio** (similaridad espectral fuente→salida): mide preservación del contenido musical.
  SAO 0.643 >> ZETA 0.452 ≈ MelodyFlow 0.446. ACE-Step y MusicGen generan música nueva, no editan.
- **CLAP_text** (alineación semántica prompt→salida): ZETA y MelodyFlow lidean mejor con prompts
  descriptivos. SAO bajo en CLAP_text porque SDEdit preserva tanto la fuente que el texto tiene
  poco efecto en rasgos de alto nivel.
- **chroma** (correlación melódica): SAO 0.838, ZETA 0.809. Ambos preservan bien la melodía.
  MusicGen-melody, pese a su bajo CLAP, preserva la curva melódica (chroma 0.644) — su problema
  es calidad tímbrica/textural, no seguimiento melódico.
- **MusicGen-melody** mal resultado: el modelo genera música nueva vagamente inspirada en la
  melodía de entrada; no es un editor audio2audio en el sentido de esta tarea.

### Por categoría (SAO vs ZETA, mejores dos)

| Categoría | SAO CLAP_audio | ZETA CLAP_audio | Ganador |
|---|---|---|---|
| style_transfer (cases 01–05) | 0.616 | 0.534 | SAO |
| instrumentation (cases 06–08) | 0.830 | 0.500 | SAO |
| mood/texture (cases 09–10) | 0.284 | 0.174 | SAO |

SAO gana en todas las categorías en CLAP_audio. ZETA es más fiable en CLAP_text (mejor adherencia
al prompt de texto).

## Casos de evaluación

### Style Transfer
| ID | Fuente | Prompt |
|---|---|---|
| case01_bach_jazz | bach.mp3 | jazz piano version, swing feel, relaxed tempo |
| case02_bolero_country | bolero_ravel.mp3 | A cheerful country song with acoustic guitars |
| case03_electronic_lofi | electronic.mp3 | lo-fi hip hop remix, chill, dusty vinyl texture, slow tempo |
| case04_electronic_orchestral | electronic.mp3 | orchestral symphonic arrangement |
| case05_bach_arcade | bach.mp3 | 8-bit chiptune arcade video game soundtrack version |

### Instrumentation
| ID | Fuente | Prompt |
|---|---|---|
| case06_bach_strings | bach.mp3 | the same melody played by a string quartet |
| case07_guitar_piano | sao_guitar_loop.wav | the same chord progression played on a grand piano, jazz voicings, warm and intimate |
| case08_beatbox_drumkit | beatbox_loop_90bpm.wav | acoustic drum kit playing the same rhythm, tight studio recording, 90 BPM |

### Mood / Texture
| ID | Fuente | Prompt |
|---|---|---|
| case09_guitar_dark | sao_guitar_loop.wav | dark ambient version, heavy reverb, ominous and haunting mood |
| case10_bolero_ambient | bolero_ravel.mp3 | soft dreamy ambient rendition with gentle synth pads |

### Continuation (InspireMusic)
| ID | Fuente | Prompt |
|---|---|---|
| case11_guitar_continue_jazz | sao_guitar_loop.wav | Continue to generate jazz music. |
| case12_electronic_continue | electronic.mp3 | Continue to generate energetic electronic dance music. |

## Estructura de archivos

```
evaluation/edit/
├── README.md                      (este archivo)
├── metrics_edit.json              (métricas CLAP + chroma, todos los casos)
├── prompts_edit.json              (prompts y metadata)
├── source_audio/                  (5 fuentes originales)
│   ├── bach.mp3
│   ├── bolero_ravel.mp3
│   ├── electronic.mp3
│   ├── beatbox_loop_90bpm.wav
│   └── sao_guitar_loop.wav
├── <modelo>/
│   └── <case_id>/
│       └── output.wav
└── reference_demos/               (audios de referencia de proyectos oficiales)
    ├── README.md                  (mapeo de archivos a casos del benchmark)
    ├── zeta_official/             (3 pares source+edited de hilamanor.github.io)
    └── melodyflow_official/       (9 pares source+ourinv de melodyflow.github.io)
```

## Audio de referencia para comparación cualitativa

Disponible en `reference_demos/` — ver `reference_demos/README.md` para el mapeo completo.

- **ZETA**: 3 pares descargados del demo oficial. Fuentes distintas a las nuestras, mismos
  parámetros (tstart=90–100, backbone AudioLDM2-music). Útil para calibrar expectativas del modelo.
- **MelodyFlow**: 9 pares del paper (MusicCaps tracks). Muestran la calidad máxima reportada.
- **ACE-Step, SAO, InspireMusic, MusicGen-melody**: sin audio de referencia descargable público.

## Conclusión y próximo paso

**SAO 1.0** (`stabilityai/stable-audio-open-1.0` vía `stable-audio-tools`) es el modelo
seleccionado para la integración en REAPER. Próximo paso: implementar plugin/skill de edición
audio2audio basado en SAO + SDEdit.
