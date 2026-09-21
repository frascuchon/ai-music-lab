#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for research_miros_modal.py's _apply_beat_tracking().

Context: a user reported that after transcribing a YouTube track, note
pitches were correct but playback was much faster than the source audio,
with resulting MIDI items too short in REAPER. One candidate cause was the
seconds→ticks re-encoding this function does when rewriting the MIDI with a
single global tempo (see its docstring). These tests confirm that encoding
is self-consistent: whatever BPM librosa detects, the tempo meta message
written into the file must match the tick spacing exactly, so a note's
absolute *wall-clock* position is preserved regardless of which BPM value
was chosen. (The actual bug turned out to be on the REAPER-import side —
see Audio2Midi/test_transcribe.py and panel.lua's import_midi — but a
regression here would independently break every transcription's timing,
so it's worth guarding directly.)
"""

import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import TestCase, mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_miros_modal as rm  # noqa: E402


def make_click_track(path: Path, bpm: float, duration_s: float, sr: int = 22050):
    """Write a simple metronome click WAV at the given BPM."""
    import numpy as np

    n = int(duration_s * sr)
    y = np.zeros(n, dtype=np.float32)
    beat_period = 60.0 / bpm
    t = 0.0
    click_len = int(0.01 * sr)
    while t < duration_s:
        start = int(t * sr)
        end = min(n, start + click_len)
        y[start:end] = 0.9
        t += beat_period
    pcm = (y * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def make_raw_midi_bytes(note_times: list[tuple[float, float, int]]) -> bytes:
    """Build a PrettyMIDI-readable MIDI (as bytes) with notes at explicit
    absolute (start_s, end_s, pitch) — mirrors what miros_transcribe's
    upstream model output looks like before beat-tracking rewrites it."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, name="test")
    for start, end, pitch in note_times:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        pm.write(f.name)
        data = Path(f.name).read_bytes()
    Path(f.name).unlink(missing_ok=True)
    return data


class TestApplyBeatTrackingPreservesAbsoluteTime(TestCase):
    """The core invariant: regardless of the detected BPM, a note's
    absolute position in seconds must survive the tick re-encoding.
    A regression here (e.g. tempo written to the file not matching the
    beat_duration used for seconds_to_ticks) would make every transcribed
    note play back scaled by whatever ratio the bug introduces — the
    "correct notes, wrong/faster tempo" symptom reported by the user."""

    def _check_round_trip(self, bpm_for_click, note_times):
        tmpdir = Path(tempfile.mkdtemp(prefix="beat_track_test_"))
        try:
            audio_path = tmpdir / "click.wav"
            make_click_track(audio_path, bpm=bpm_for_click, duration_s=8.0)
            audio_bytes = audio_path.read_bytes()
            midi_bytes = make_raw_midi_bytes(note_times)

            result_bytes = rm._apply_beat_tracking(audio_bytes, midi_bytes)

            out_path = tmpdir / "out.mid"
            out_path.write_bytes(result_bytes)

            import pretty_midi
            pm_out = pretty_midi.PrettyMIDI(str(out_path))
            self.assertEqual(len(pm_out.instruments), 1)
            out_notes = sorted(pm_out.instruments[0].notes, key=lambda n: n.start)
            self.assertEqual(len(out_notes), len(note_times))

            for (exp_start, exp_end, exp_pitch), got in zip(note_times, out_notes):
                self.assertEqual(got.pitch, exp_pitch)
                # Tick rounding at 480 PPQ is sub-millisecond at any
                # musically plausible tempo; 15ms is a generous bound that
                # would still catch a gross tempo-scaling bug (which
                # produces errors of hundreds of ms to seconds).
                self.assertAlmostEqual(got.start, exp_start, delta=0.015)
                self.assertAlmostEqual(got.end, exp_end, delta=0.015)
            return out_path
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_round_trip_at_moderate_tempo(self):
        self._check_round_trip(100.0, [(0.5, 1.0, 60), (2.0, 2.5, 64), (4.0, 4.8, 67)])

    def test_round_trip_at_fast_tempo(self):
        """Fast click track (160 BPM) — the regime most likely to expose a
        scaling bug like the one reported (playback much faster than the
        source)."""
        self._check_round_trip(160.0, [(0.3, 0.6, 60), (1.5, 1.9, 62), (5.0, 5.4, 65)])

    def test_round_trip_at_slow_tempo(self):
        self._check_round_trip(70.0, [(0.5, 1.2, 60), (3.0, 3.9, 64)])

    def test_forced_bpm_mismatch_with_source_still_preserves_time(self):
        """Even if librosa's detected BPM has nothing to do with the click
        track's real tempo (simulating an octave-error / bad detection),
        absolute note timing must still be preserved — the function's
        contract is "encode ticks against whatever BPM was chosen", not
        "detect the true tempo correctly". This isolates the encode/decode
        math from beat-detection accuracy."""
        note_times = [(0.5, 1.0, 60), (2.0, 2.5, 64), (4.0, 4.8, 67)]
        midi_bytes = make_raw_midi_bytes(note_times)
        audio_path = Path(tempfile.mkdtemp(prefix="beat_track_test_")) / "click.wav"
        make_click_track(audio_path, bpm=100.0, duration_s=8.0)
        audio_bytes = audio_path.read_bytes()

        with mock.patch(
            "librosa.beat.beat_track",
            return_value=([200.0], None),  # wildly wrong "detected" BPM
        ):
            result_bytes = rm._apply_beat_tracking(audio_bytes, midi_bytes)

        out_path = audio_path.parent / "out.mid"
        out_path.write_bytes(result_bytes)
        import pretty_midi
        pm_out = pretty_midi.PrettyMIDI(str(out_path))
        out_notes = sorted(pm_out.instruments[0].notes, key=lambda n: n.start)
        for (exp_start, exp_end, _), got in zip(note_times, out_notes):
            self.assertAlmostEqual(got.start, exp_start, delta=0.015)
            self.assertAlmostEqual(got.end, exp_end, delta=0.015)

    def test_embedded_tempo_matches_reported_bpm(self):
        """The tempo meta message written into the output file must equal
        the detected/clamped BPM used for tick spacing — this is exactly
        the value Audio2Midi/transcribe.py::get_midi_tempo() reads back to
        tell REAPER what project tempo to force before import. If these
        drift apart, forcing the project tempo would "fix" the import
        using the wrong number."""
        note_times = [(0.5, 1.0, 60), (2.0, 2.5, 64)]
        midi_bytes = make_raw_midi_bytes(note_times)
        audio_path = Path(tempfile.mkdtemp(prefix="beat_track_test_")) / "click.wav"
        make_click_track(audio_path, bpm=130.0, duration_s=8.0)
        audio_bytes = audio_path.read_bytes()

        with mock.patch("librosa.beat.beat_track", return_value=([130.0], None)):
            result_bytes = rm._apply_beat_tracking(audio_bytes, midi_bytes)

        import mido
        out_path = audio_path.parent / "out.mid"
        out_path.write_bytes(result_bytes)
        mid = mido.MidiFile(str(out_path))
        tempo_msgs = [
            m for track in mid.tracks for m in track
            if m.is_meta and m.type == "set_tempo"
        ]
        self.assertEqual(len(tempo_msgs), 1)
        self.assertAlmostEqual(mido.tempo2bpm(tempo_msgs[0].tempo), 130.0, places=1)


class TestApplyBeatTrackingClampsRange(TestCase):
    """BPM is folded into [60, 180] by doubling/halving (see docstring) —
    verify the clamp itself, since a broken clamp could independently
    cause an octave-scale (2x/0.5x) tempo error."""

    def _bpm_used(self, detected_bpm):
        note_times = [(0.5, 1.0, 60)]
        midi_bytes = make_raw_midi_bytes(note_times)
        audio_path = Path(tempfile.mkdtemp(prefix="beat_track_test_")) / "click.wav"
        make_click_track(audio_path, bpm=100.0, duration_s=4.0)
        audio_bytes = audio_path.read_bytes()
        with mock.patch("librosa.beat.beat_track", return_value=([detected_bpm], None)):
            result_bytes = rm._apply_beat_tracking(audio_bytes, midi_bytes)
        import mido
        out_path = audio_path.parent / "out.mid"
        out_path.write_bytes(result_bytes)
        mid = mido.MidiFile(str(out_path))
        for track in mid.tracks:
            for m in track:
                if m.is_meta and m.type == "set_tempo":
                    return mido.tempo2bpm(m.tempo)
        return None

    def test_too_slow_is_doubled_into_range(self):
        bpm = self._bpm_used(40.0)  # 40 -> 80
        self.assertAlmostEqual(bpm, 80.0, places=1)

    def test_too_fast_is_halved_into_range(self):
        bpm = self._bpm_used(300.0)  # 300 -> 150
        self.assertAlmostEqual(bpm, 150.0, places=1)

    def test_zero_bpm_falls_back_to_120(self):
        bpm = self._bpm_used(0.0)
        self.assertAlmostEqual(bpm, 120.0, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
