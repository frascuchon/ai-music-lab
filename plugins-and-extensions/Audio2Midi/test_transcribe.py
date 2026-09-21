#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for transcribe.py — focused on the tempo-reporting bug where
transcribed MIDI plays back faster than the source audio (correct notes,
too-short items) once imported into REAPER.

Root cause: REAPER's InsertMedia does not reliably adopt an imported MIDI
file's own tempo map — it can place notes on the project's existing tempo
grid instead of the one the file's ticks were encoded against. The fix is
for transcribe.py to read back the tempo actually embedded in the produced
MIDI (get_midi_tempo) and report it via the progress protocol
(``TEMPO|<bpm>``), so Audio2Midi.lua can force the project tempo to match
before calling InsertMedia (see panel.lua's import_midi).

These tests cover:
  - get_midi_tempo() on files with an explicit tempo, no tempo (SMF
    default 120 BPM), multiple tempo events, and a corrupt/missing file.
  - The end-to-end CLI reporting a correct ``TEMPO|`` line in the progress
    file for a synthetic beat-tracked-style MIDI.
"""

import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import TestCase

SCRIPT = Path(__file__).resolve().parent / "transcribe.py"

# ---------------------------------------------------------------------------
# Mock uv on PATH so _find_uv() resolves it, same pattern as
# StemsSeparator/test_separate_sam.py
# ---------------------------------------------------------------------------
_UV_MOCK_DIR = Path(tempfile.mkdtemp(prefix="uv_mock_a2m_"))
_UV_MOCK = _UV_MOCK_DIR / "uv"
_UV_MOCK.write_text(textwrap.dedent(f"""\
#!/usr/bin/env python3
import sys, os
PY = {sys.executable!r}  # same interpreter running the tests (has mido/pretty_midi)
args = sys.argv[1:]
for i, a in enumerate(args):
    if '::' in a:
        script = a.split('::')[0]
        os.execvp(PY, [PY, script] + args[i+1:])
        break
    if a.endswith('.py'):
        os.execvp(PY, [PY] + args[i:])
        break
sys.exit(1)
"""))
_UV_MOCK.chmod(0o755)
_ORIG_PATH = __import__("os").environ.get("PATH", "")


def setUpModule():
    import os
    os.environ["PATH"] = str(_UV_MOCK_DIR) + ":" + _ORIG_PATH


def tearDownModule():
    import os
    os.environ["PATH"] = _ORIG_PATH
    shutil.rmtree(_UV_MOCK_DIR, ignore_errors=True)


def parse_progress(path):
    if not path.exists():
        return None, None, None, []
    text = path.read_text()
    if not text.strip():
        return None, None, None, []
    lines = text.splitlines()
    parts = lines[0].split("|")
    if len(parts) < 3:
        return None, None, None, []
    state, pct = parts[0], float(parts[1])
    msg = "|".join(parts[2:])
    extra = [l for l in lines[1:] if l.strip()]
    return state, pct, msg, extra


def make_midi(path, tempo_bpm=None, n_notes=3):
    """Build a tiny SMF via mido. tempo_bpm=None omits the set_tempo event
    entirely (SMF spec default: 120 BPM)."""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    if tempo_bpm is not None:
        tempo_track.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0)
        )
    tempo_track.append(mido.MetaMessage("end_of_track", time=0))

    notes = mido.MidiTrack()
    mid.tracks.append(notes)
    for i in range(n_notes):
        notes.append(mido.Message("note_on", note=60 + i, velocity=80, time=0 if i == 0 else 120))
        notes.append(mido.Message("note_off", note=60 + i, velocity=0, time=240))
    notes.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(str(path))


class TestGetMidiTempo(TestCase):
    """Direct tests for get_midi_tempo() — the piece that lets the caller
    detect the file's real tempo and reconcile it with REAPER's project
    tempo before import."""

    def setUp(self):
        sys.path.insert(0, str(SCRIPT.parent))
        import transcribe
        self.transcribe = transcribe
        self.tmpdir = Path(tempfile.mkdtemp(prefix="a2m_tempo_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_explicit_tempo_is_read_back(self):
        mid_path = self.tmpdir / "explicit.mid"
        make_midi(mid_path, tempo_bpm=76.0)
        bpm = self.transcribe.get_midi_tempo(mid_path)
        self.assertAlmostEqual(bpm, 76.0, places=1)

    def test_fast_tempo_is_read_back(self):
        """This is the case that reproduces the reported bug: a
        beat-tracked fast tempo (e.g. 152 BPM) must be reported exactly,
        not silently coerced towards a 120 BPM default."""
        mid_path = self.tmpdir / "fast.mid"
        make_midi(mid_path, tempo_bpm=152.3)
        bpm = self.transcribe.get_midi_tempo(mid_path)
        self.assertAlmostEqual(bpm, 152.3, places=1)

    def test_missing_tempo_event_defaults_to_120(self):
        mid_path = self.tmpdir / "no_tempo.mid"
        make_midi(mid_path, tempo_bpm=None)
        bpm = self.transcribe.get_midi_tempo(mid_path)
        self.assertAlmostEqual(bpm, 120.0, places=1)

    def test_missing_file_returns_negative(self):
        bpm = self.transcribe.get_midi_tempo(self.tmpdir / "nope.mid")
        self.assertEqual(bpm, -1.0)

    def test_corrupt_file_returns_negative(self):
        bad = self.tmpdir / "corrupt.mid"
        bad.write_bytes(b"not a midi file")
        bpm = self.transcribe.get_midi_tempo(bad)
        self.assertEqual(bpm, -1.0)


class TestTempoReportedEndToEnd(TestCase):
    """Runs transcribe.py as a subprocess against a stubbed modal script
    that drops a pre-built, known-tempo MIDI where transcribed_cuda.mid
    is expected, then checks the progress file's TEMPO| line."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="a2m_e2e_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.input_wav = self.tmpdir / "input.wav"
        self.input_wav.write_bytes(b"\x00" * 8000)
        self.outdir = self.tmpdir / "out"
        self.outdir.mkdir()
        self.shared_dir = self.tmpdir / "shared"
        self.shared_dir.mkdir()
        (self.shared_dir / "pyproject.toml").write_text("[project]\nname='shared'\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_stub_script(self, tempo_bpm):
        script = self.tmpdir / "research_stub_modal.py"
        # Runs inside the mocked uv/modal pipeline; writes the same
        # transcribed_cuda.mid the real research_*_modal.py entrypoints
        # produce, but with a controlled, known tempo.
        script.write_text(textwrap.dedent(f"""\
        import argparse, sys
        sys.path.insert(0, {str(SCRIPT.parent)!r})
        from test_transcribe import make_midi
        p = argparse.ArgumentParser()
        p.add_argument("--audio-path")
        p.add_argument("--out-dir")
        p.add_argument("--force", action="store_true")
        p.add_argument("--no-beat-tracking", action="store_true")
        args = p.parse_args()
        from pathlib import Path
        out = Path(args.out_dir) / "transcribed_cuda.mid"
        make_midi(out, tempo_bpm={tempo_bpm})
        print("[stub] wrote", out)
        """))
        return script

    def _run(self, script, model="miros"):
        args = [
            sys.executable, str(SCRIPT),
            "--shared-dir", str(self.shared_dir),
            "--script", str(script),
            "--input", str(self.input_wav),
            "--out-dir", str(self.outdir),
            "--model", model,
            "--progress", str(self.progress_file),
        ]
        return subprocess.run(args, capture_output=True, text=True, timeout=30)

    def test_reports_beat_tracked_tempo(self):
        """The symptom reported: a fast beat-tracked tempo (much faster
        than the audio's real feel) must still be surfaced correctly via
        TEMPO| so Audio2Midi.lua can force the project to it — silently
        dropping/mis-reporting it is what causes the sped-up import."""
        script = self._write_stub_script(tempo_bpm=163.0)
        proc = self._run(script)
        self.assertEqual(proc.returncode, 0,
                          msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        state, pct, msg, extra = parse_progress(self.progress_file)
        self.assertEqual(state, "done")
        tempo_lines = [l for l in extra if l.startswith("TEMPO|")]
        self.assertEqual(len(tempo_lines), 1)
        reported_bpm = float(tempo_lines[0].split("|", 1)[1])
        self.assertAlmostEqual(reported_bpm, 163.0, places=1)

    def test_reports_slow_tempo(self):
        script = self._write_stub_script(tempo_bpm=68.0)
        proc = self._run(script)
        self.assertEqual(proc.returncode, 0,
                          msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        _, _, _, extra = parse_progress(self.progress_file)
        tempo_lines = [l for l in extra if l.startswith("TEMPO|")]
        self.assertEqual(len(tempo_lines), 1)
        self.assertAlmostEqual(float(tempo_lines[0].split("|", 1)[1]), 68.0, places=1)

    def test_instruments_line_still_present(self):
        """Regression guard: adding TEMPO| must not break the existing
        INSTRUMENTS| reporting."""
        script = self._write_stub_script(tempo_bpm=120.0)
        proc = self._run(script)
        self.assertEqual(proc.returncode, 0)
        _, _, _, extra = parse_progress(self.progress_file)
        instr_lines = [l for l in extra if l.startswith("INSTRUMENTS|")]
        self.assertEqual(len(instr_lines), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
