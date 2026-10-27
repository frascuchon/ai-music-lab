#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "basic-pitch",
#     "onnxruntime",
#     "librosa",
#     "pretty_midi",
#     "soundfile",
#     "numpy",
#     "setuptools<81",  # resampy (via basic-pitch) imports pkg_resources
# ]
# ///
"""Local audio -> MIDI for a SINGLE instrument (guitar, voice, piano, bass...)
with Spotify's Basic Pitch. Runs on this machine (CPU), no Modal, no GPU.

    uv run research/research_basic_pitch.py --input guitar.wav --out out.mid

Why it exists: the multi-instrument models (YourMT3+, MIROS) split a solo
instrument across several tracks and need a well-leveled input. On a
distorted solo guitar, Basic Pitch matched a pYIN pitch reference on ~66 % of
frames versus ~35 % (YourMT3+) and ~19 % (MIROS), and returns one clean track.

Pipeline: mono mix -> peak-normalise (quiet recordings otherwise yield an
empty transcription) -> Basic Pitch -> optional BPM estimate so the MIDI grid
matches the audio -> one instrument track.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

TARGET_PEAK = 0.89          # -1 dBFS
MAX_GAIN_DB = 30.0          # never amplify noise floors by more than this
SILENCE_PEAK = 1e-4         # below this the file is treated as silent


def load_normalised(path: Path):
    """Return (mono float32 samples, sample rate, applied gain in dB)."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    if peak < SILENCE_PEAK:
        raise ValueError("the audio is silent (peak below -80 dBFS)")
    gain = min(TARGET_PEAK / peak, 10 ** (MAX_GAIN_DB / 20))
    return mono * gain, sr, 20 * np.log10(gain)


# Intervals (semitones) between a real note and the sub-harmonic "ghost" that
# Basic Pitch often adds below it on distorted guitar: 1 octave, octave+fifth
# (3rd harmonic), 2 octaves.
GHOST_INTERVALS = (12, 19, 24)
TAIL_DB = 14.0   # a note this far below the typical note level is reverb/noise tail


def _overlap(a, b) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def drop_ghost_notes(notes):
    """Remove sub-octave ghosts: a note is dropped when a louder-or-equal note
    exactly 12/19/24 semitones above it sounds over at least half of it."""
    kept = []
    for n in notes:
        dur = max(n.end - n.start, 1e-6)
        ghost = any(
            (m.pitch - n.pitch) in GHOST_INTERVALS
            and m.velocity >= n.velocity
            and _overlap(n, m) >= 0.5 * dur
            for m in notes if m is not n
        )
        if not ghost:
            kept.append(n)
    return kept


def drop_tail_notes(notes, y, sr):
    """Remove notes whose audio is far quieter than the typical note, i.e. the
    reverb/delay tail or noise after the performance ends."""
    import numpy as np

    if len(notes) < 8:
        return notes
    levels = []
    for n in notes:
        seg = y[int(n.start * sr):max(int(n.end * sr), int(n.start * sr) + 1)]
        levels.append(float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0)
    ref = float(np.median(levels))
    if ref <= 0:
        return notes
    floor = ref * 10 ** (-TAIL_DB / 20)
    return [n for n, lv in zip(notes, levels) if lv >= floor]


def estimate_bpm(y, sr) -> float:
    import librosa
    import numpy as np

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    if not (np.isfinite(bpm) and bpm > 0):
        return 120.0
    while bpm < 60:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return bpm


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True, help="Output .mid path.")
    p.add_argument("--onset-threshold", type=float, default=0.5, dest="onset")
    p.add_argument("--frame-threshold", type=float, default=0.4, dest="frame")
    p.add_argument("--min-note-ms", type=float, default=100.0, dest="min_ms")
    p.add_argument("--min-freq", type=float, default=70.0, dest="fmin",
                   help="Hz; default covers a guitar's low E (82 Hz) with margin.")
    p.add_argument("--max-freq", type=float, default=1400.0, dest="fmax")
    p.add_argument("--program", type=int, default=25,
                   help="General MIDI program of the output track (25 = steel guitar).")
    p.add_argument("--name", default="Transcription")
    p.add_argument("--no-beat-tracking", action="store_true", dest="no_beat")
    p.add_argument("--bpm", type=float, default=None,
                   help="Use this tempo instead of estimating it (e.g. the project's).")
    p.add_argument("--no-cleanup", action="store_true", dest="no_cleanup",
                   help="Keep sub-octave ghost notes and tail noise.")
    args = p.parse_args()

    in_path, out_path = Path(args.input), Path(args.out)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", flush=True)
        return 1

    print("[1/3] Reading and levelling audio...", flush=True)
    try:
        y, sr, gain_db = load_normalised(in_path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the panel log
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(f"[basic-pitch] input {len(y) / sr:.1f}s @ {sr} Hz, gain {gain_db:+.1f} dB",
          flush=True)

    import pretty_midi
    import soundfile as sf
    from basic_pitch.inference import predict

    print("[2/3] Transcribing (local CPU)...", flush=True)
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp_wav, y, sr, subtype="FLOAT")
        _, midi, _ = predict(
            tmp_wav,
            onset_threshold=args.onset,
            frame_threshold=args.frame,
            minimum_note_length=args.min_ms,
            minimum_frequency=args.fmin,
            maximum_frequency=args.fmax,
            melodia_trick=True,
        )
    finally:
        os.unlink(tmp_wav)

    notes = [n for inst in midi.instruments for n in inst.notes]
    if not notes:
        print("ERROR: no notes detected. Check that the selected item is the "
              "instrument you want to transcribe and that it is not silent.",
              flush=True)
        return 1

    print("[3/3] Writing MIDI...", flush=True)
    if not args.no_cleanup:
        raw = len(notes)
        notes = drop_tail_notes(drop_ghost_notes(notes), y, sr)
        print(f"[basic-pitch] cleanup removed {raw - len(notes)} of {raw} notes "
              "(sub-octave ghosts, tail noise)", flush=True)
        if not notes:
            print("ERROR: no notes left after cleanup.", flush=True)
            return 1
    if args.bpm and args.bpm > 0:
        bpm = args.bpm
    else:
        bpm = 120.0 if args.no_beat else estimate_bpm(y, sr)
    print(f"[basic-pitch] {len(notes)} notes, tempo {bpm:.1f} BPM", flush=True)

    # Re-create the file with the estimated tempo; note times stay in seconds.
    out = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    track = pretty_midi.Instrument(program=args.program, name=args.name)
    track.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    out.instruments.append(track)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(out_path))
    print(f"[basic-pitch] saved: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
