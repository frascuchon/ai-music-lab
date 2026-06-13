# Evaluación Audio2Midi

Estructura de evaluación para los pipelines de transcripción automática audio → MIDI.
Sigue la misma convención de `MidiGenerator/evaluation/` adaptada a audio como input.

## Estructura

```
evaluation/
├── fetch_fixtures.sh           ← descarga los 10 audios de demo de YourMT3 HF Space
├── yourmt3/                    ← evaluación del modelo end-to-end YourMT3+
│   ├── test01/  Funk (MIRST)
│   ├── test02/  Chopin piano solo (MAPS)
│   ├── test03/  Schubert piano solo (MAPS)
│   ├── test04/  Slakh multi-instrumento #1884
│   ├── test05/  Slakh multi-instrumento #1975
│   ├── test06/  Cámara real (MIRST 493)
│   ├── test07/  Clásico real (MusicNet 2556)
│   ├── test08/  Clásico real (MusicNet 2628)
│   ├── test09/  Grabación real (RWC 087)
│   └── test10/  Grabación real (RWC 089, MP3)
└── compound/                   ← evaluación del pipeline Demucs + Basic Pitch
    ├── test01/  Funk (mismo que yourmt3/test01)
    ├── test02/  Piano solo (mismo que yourmt3/test02)
    ├── test03/  Slakh #1884 (mismo que yourmt3/test04)
    ├── test04/  RWC 087 (mismo que yourmt3/test09)
    └── test05/  MIRST 493 (mismo que yourmt3/test06)
```

Cada carpeta `test*/` contiene:
- `input.wav` (o `input.mp3`) — audio a transcribir (descargado por `fetch_fixtures.sh`)
- `notes.txt` — descripción del test, dataset de origen, métricas objetivo
- `transcribed_cuda.mid` — output generado (creado por `eval_all`)

## Flujo de trabajo

### 1. Descargar los audios de referencia

Los audios son los 10 demos oficiales del HuggingFace Space de YourMT3.
Vienen de los mismos datasets usados para entrenar/evaluar el modelo.

```bash
cd plugins-and-extensions/Audio2Midi/evaluation
bash fetch_fixtures.sh
```

### 2. Descargar el checkpoint YourMT3+ (una sola vez)

```bash
cd plugins-and-extensions/Audio2Midi/research
modal run research_yourmt3_modal.py::setup
```

Descarga `last.ckpt` (562 MB) al Modal Volume `yourmt3-weights`.
Coste: ~$0.02

### 3. Ejecutar evaluación YourMT3+

```bash
# Todos los tests (10 audios en una sola llamada Modal):
modal run research_yourmt3_modal.py::eval_all --eval-dir ../evaluation/yourmt3

# Solo los tests Slakh (multi-instrumento, los más representativos):
modal run research_yourmt3_modal.py::eval_all \
    --eval-dir ../evaluation/yourmt3 \
    --only 4,5

# Un test individual:
modal run research_yourmt3_modal.py::main \
    --audio-path ../evaluation/yourmt3/test04/input.wav \
    --out-dir ../evaluation/yourmt3/test04
```

### 4. Ejecutar evaluación pipeline compuesto

```bash
# Verificar entorno (primera vez):
modal run research_compound_pipeline_modal.py::setup

# Todos los tests (5 audios):
modal run research_compound_pipeline_modal.py::eval_all \
    --eval-dir ../evaluation/compound
```

### 5. Escuchar en REAPER

1. Abrir un proyecto nuevo en REAPER
2. Arrastrar `transcribed_cuda.mid` a la timeline
3. Cada pista MIDI tiene asignado un instrumento (canal MIDI)
4. Asignar VST por instrumento y escuchar
5. Comparar con el audio original

### 6. Anotar resultados

Editar `RESEARCH.md` en la sección "Resultados evaluación" de cada candidato.

Métricas subjetivas (0-5 por instrumento):
- 5 = transcripción perfecta, igual que el original
- 4 = muy buena, errores menores
- 3 = usable, pero notable degradación
- 2 = parcialmente correcto
- 1 = reconocible pero muy impreciso
- 0 = inútil

## Convención de archivos

| Archivo | Descripción |
|---------|-------------|
| `input.wav` | Audio de entrada (WAV PCM, cualquier sr) |
| `input.mp3` | Audio de entrada en MP3 (test10 RWC) |
| `transcribed_cuda.mid` | Salida del modelo (YourMT3+ o compound) |
| `notes.txt` | Metadatos del test y resultados |

## Datasets de origen (audios de referencia)

Los 10 audios provienen directamente de los datasets de evaluación de YourMT3+:

| Test | Dataset | Tipo |
|------|---------|------|
| test01 | MIRST | Grabación real multi-instrumento |
| test02 | MAPS (Kawai EX) | Piano Disklavier |
| test03 | MAPS (Yamaha DM) | Piano Disklavier |
| test04-05 | Slakh2100 | Sintético (MIDI+soundfont) |
| test06 | MIRST | Grabación real |
| test07-08 | MusicNet | Grabación clásica profesional |
| test09-10 | RWC Music DB | Grabación profesional multiuso |

Referencia: mimbres/YourMT3 HuggingFace Space, carpeta `/examples/`
