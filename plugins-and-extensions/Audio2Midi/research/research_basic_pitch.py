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
