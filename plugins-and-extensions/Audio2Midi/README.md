# Audio2Midi

Plugin REAPER para transcribir audio a MIDI: en local para un instrumento solo, o en la nube para mezclas.

## Modelos soportados

| Modelo | Descripción | GPU | Licencia |
|--------|-------------|-----|---------|
| **Basic Pitch** (por defecto) | Un solo instrumento (guitarra, voz, piano). Se ejecuta en local (CPU), sin Modal. Devuelve una pista. | ninguna | Apache 2.0 |
| **MIROS** | Un solo instrumento, en Modal. El panel avisa de que no es para mezclas: con varios instrumentos usa YourMT3+. | A10G+ (no T4) | uso interno |
| **YourMT3+** | Multi-instrumento. Robusto, F1 77.5% Slakh. | T4+ | Apache 2.0 |

**Qué modelo elegir:** para un instrumento solo, Basic Pitch (local) o MIROS (Modal); para una mezcla con varios instrumentos, YourMT3+. En una guitarra distorsionada solista, Basic Pitch coincidió con una referencia pYIN en ~66 % de los fotogramas frente a ~35 % de YourMT3+, que además repartía la guitarra en varias pistas.

> MIROS se retiró de la interfaz: en esa misma guitarra apenas detectaba notas (~19 %). Su script y las notas de evaluación siguen en `research/` y `RESEARCH.md`.

**Pistas vacías:** tras transcribir, `transcribe.py` descarta del MIDI las pistas de instrumento que no contienen ninguna nota (no se importan como clips vacíos) y lo indica en el estado ("N empty discarded"). Si no queda ninguna nota, se informa de error.

**Limpieza de Basic Pitch:** con guitarra distorsionada el modelo añade "notas fantasma" una octava (o octava+quinta) por debajo de la nota real, y notas sueltas en la cola de eco tras terminar la música. Se eliminan automáticamente (en un clip real: 12 de 82 notas, sin perder ninguna correcta). Desmarcar **Estimate tempo** hace que use el tempo del proyecto en lugar de estimarlo. El control **Threshold** (0,3–0,9, por defecto 0,7) fija cuántas notas se aceptan: más alto, menos notas y más seguras. Sobre una guitarra real, subir de 0,5 a 0,7 llevó la F por nota (±50 ms, frente a pYIN) de 0,46 a 0,64.

**Nivel de entrada:** antes de transcribir se normaliza el pico a -1 dBFS (máx. +30 dB de ganancia). Un render de guitarra con pico en -22 dBFS hacía que los modelos devolvieran un MIDI vacío. Si no se detecta ninguna nota, el plugin lo indica como error.

**Importación:** el MIDI se lee y se crea pista a pista (`shared/lib/midi_import.lua`), sin `InsertMedia`. Los clips y pistas que ya hubiera en el proyecto no se envuelven en la carpeta nueva ni se mueven.

## Instalación

Se instala con ReaPack como parte del plugin unificado **AI Music Lab** (pestaña Audio2Midi); ver la [guía de instalación](../../README.md#quick-install-about-10-minutes) del README principal. La configuración (uv, Modal, tokens) se hace desde el botón **Setup** del panel.

## Uso

1. En REAPER, **selecciona** una pista, un item de audio, o un split (sección recortada).
2. Abre **AI Music Lab** desde el Action List y ve a la pestaña Audio2Midi.
3. Pulsa **R** para capturar la selección. La UI mostrará el archivo fuente y,
   si es una sección, el rango de tiempo en amarillo.
4. Elige **modelo** y **GPU**.
5. Pulsa **GENERAR MIDI**.

El progreso se muestra en tiempo real en la barra y el log.

### Resultado en REAPER

- **Un instrumento** → nueva pista MIDI nombrada `<fuente> [MIDI <modelo>]`
  en la posición del audio original.
- **Varios instrumentos** → carpeta de pistas nombradas,
  una por instrumento.

Ambos casos quedan como una operación deshacer (`Ctrl+Z`).

## Selección de input: tres niveles

| Nivel | Cómo | Comportamiento |
|-------|------|----------------|
| **Pista** | Seleccionar la pista en REAPER → R | Toma el primer item de audio activo |
| **Item** | Seleccionar un item → R | Transcribe el item completo |
| **Split** | Seleccionar un item recortado → R | Transcribe solo la sección recortada (`D_STARTOFFS` + `D_LENGTH`) |

El botón `...` permite elegir cualquier archivo WAV manualmente (sin capturar de REAPER).

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `panel.lua` | Script REAPER (UI + integración) |
| `transcribe.py` | Backend CLI: invoca Modal, gestiona progreso |
| `research/research_basic_pitch.py` | Transcripción local (Basic Pitch, `uv run`) |
| `research/research_miros_modal.py` | App Modal para MIROS |
| `research/research_yourmt3_modal.py` | App Modal para YourMT3+ |

## Notas técnicas

- El proceso de transcripción corre en la nube (Modal). La primera ejecución descarga
  los pesos si no están precargados (~1-2 min adicionales).
- `transcribe.py` crea un directorio temporal por ejecución para evitar conflictos de caché.
- El MIDI resultante se renombra a `<fuente>__<modelo>.mid` antes de importarse.
- Tempo: Basic Pitch escribe el MIDI con el BPM estimado por librosa (o el del proyecto
  si se desmarca *Estimate tempo*), para que las notas queden alineadas al grid de REAPER.
- La entrada se pasa a mono y se normaliza el pico a -1 dBFS antes de transcribir; el
  plugin funciona con clips REAPER de 44.1/48 kHz estéreo sin intervención del usuario.
