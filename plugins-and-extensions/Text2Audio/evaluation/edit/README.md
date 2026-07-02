# Audio2Audio Editing Benchmark

## Models Evaluated (5/6)

| Model | Task | Cases | Status |
|---|---|---|---|
| ACE-Step 1.5 | cover | 10/10 | ✅ |
| MelodyFlow | text-guided editing | 10/10 | ✅ |
| SAO 1.0 (stable-audio-tools) | style transfer | 10/10 | ✅ |
| ZETA (AudioLDM2) | zero-shot editing | 10/10 | ✅ |
| InspireMusic | continuation | 2/2 | ✅ |
| MusicGen-melody | melody-cond. versioning | 0/10 | ❌ (deps) |

## Structure

Each model folder contains:
- `output.wav` per case (e.g., `case01_bach_jazz/output.wav`)
- `source_audio/` - original source files
- `prompts_edit.json` - prompts and metadata

## Source Audio

- bach.mp3 (10s, classical baroque melody)
- bolero_ravel.mp3 (10s, orchestral bolero)
- electronic.mp3 (3s, electronic clip)
- beatbox_loop_90bpm.wav (10.67s, human beatbox)
- sao_guitar_loop.wav (9.6s, generated jazz guitar)

## Next Steps

1. Fix MusicGen-melody (xformers/torchaudio CUDA deps)
2. Run compute_metrics_edit.py
3. Subjective evaluation in REAPER
