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


class TestGpuSelection(TestCase):
    """The panel's GPU combo must reach the Modal script: those scripts read
    MIROS_GPU / YOURMT3_GPU at import time, so transcribe.py has to export
    the chosen GPU into the subprocess environment."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="a2m_gpu_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.input_wav = self.tmpdir / "input.wav"
        self.input_wav.write_bytes(b"\x00" * 8000)
        self.outdir = self.tmpdir / "out"
        self.outdir.mkdir()
        self.shared_dir = self.tmpdir / "shared"
        self.shared_dir.mkdir()
        (self.shared_dir / "pyproject.toml").write_text("[project]\nname='shared'\n")
        self.script = self.tmpdir / "research_stub_modal.py"
        self.script.write_text(textwrap.dedent(f"""\
            import argparse, os, sys
            sys.path.insert(0, {str(SCRIPT.parent)!r})
            from test_transcribe import make_midi
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument("--audio-path"); p.add_argument("--out-dir")
            p.add_argument("--force", action="store_true")
            p.add_argument("--no-beat-tracking", action="store_true")
            a = p.parse_args()
            out = Path(a.out_dir)
            (out / "gpu_seen.txt").write_text(
                os.environ.get("MIROS_GPU", "-") + "," + os.environ.get("YOURMT3_GPU", "-"))
            make_midi(out / "transcribed_cuda.mid", tempo_bpm=120)
        """))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seen(self, model, gpu):
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--shared-dir", str(self.shared_dir), "--script", str(self.script),
             "--input", str(self.input_wav), "--out-dir", str(self.outdir),
             "--model", model, "--gpu", gpu, "--progress", str(self.progress_file)],
            capture_output=True, text=True, timeout=30)
        return (self.outdir / "gpu_seen.txt").read_text()

    def test_miros_gpu_is_exported(self):
        self.assertEqual(self._seen("miros", "A100"), "A100,-")

    def test_miros_on_t4_is_rejected_before_launching_modal(self):
        self._seen_no_output("miros", "T4")
        self.assertIn("cannot run on T4", self.progress_file.read_text())
        self.assertFalse((self.outdir / "gpu_seen.txt").exists())

    def _seen_no_output(self, model, gpu):
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--shared-dir", str(self.shared_dir), "--script", str(self.script),
             "--input", str(self.input_wav), "--out-dir", str(self.outdir),
             "--model", model, "--gpu", gpu, "--progress", str(self.progress_file)],
            capture_output=True, text=True, timeout=30)

    def test_yourmt3_gpu_is_exported(self):
        self.assertEqual(self._seen("yourmt3", "T4"), "-,T4")


class TestLocalModel(TestCase):
    """Basic Pitch runs through `uv run <script>` on this machine: no shared/
    project, no modal, no GPU env var. An empty result must be an error."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="a2m_local_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.input_wav = self.tmpdir / "guitar.wav"
        self.input_wav.write_bytes(b"\x00" * 8000)
        self.outdir = self.tmpdir / "out"
        self.outdir.mkdir()
        self.script = self.tmpdir / "research_basic_pitch.py"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_script(self, n_notes):
        self.script.write_text(textwrap.dedent(f"""\
            import argparse, os, sys
            sys.path.insert(0, {str(SCRIPT.parent)!r})
            from test_transcribe import make_midi
            p = argparse.ArgumentParser()
            p.add_argument("--input"); p.add_argument("--out")
            p.add_argument("--no-beat-tracking", action="store_true")
            a = p.parse_args()
            make_midi(a.out, tempo_bpm=100, n_notes={n_notes})
            print("[stub] gpu env:", os.environ.get("MIROS_GPU"), os.environ.get("YOURMT3_GPU"))
        """))

    def _run(self, extra=()):
        # --shared-dir deliberately points at a folder WITHOUT pyproject.toml
        return subprocess.run(
            [sys.executable, str(SCRIPT),
             "--shared-dir", str(self.tmpdir / "no_shared"), "--script", str(self.script),
             "--input", str(self.input_wav), "--out-dir", str(self.outdir),
             "--model", "basicpitch", "--progress", str(self.progress_file), *extra],
            capture_output=True, text=True, timeout=30)

    def test_local_model_runs_without_shared_project(self):
        self._write_script(n_notes=4)
        self._run()
        text = self.progress_file.read_text()
        self.assertTrue(text.startswith("done|"), msg=text)
        self.assertIn("TEMPO|100", text)
        self.assertIn("INSTRUMENTS|1", text)
        self.assertIn("guitar__basicpitch.mid", text)

    def test_local_model_does_not_export_gpu_env(self):
        self._write_script(n_notes=4)
        res = self._run(["--gpu", "A100"])
        self.assertIn("gpu env: None None", res.stdout)

    def test_bpm_is_forwarded_to_the_local_script(self):
        self._write_script(n_notes=3)
        self.script.write_text(self.script.read_text().replace(
            'p.add_argument("--no-beat-tracking", action="store_true")',
            'p.add_argument("--no-beat-tracking", action="store_true"); p.add_argument("--bpm", type=float, default=None)').replace(
            'tempo_bpm=100', 'tempo_bpm=(a.bpm or 100)'))
        self._run(["--bpm", "87"])
        self.assertIn("TEMPO|87", self.progress_file.read_text())

    def test_empty_transcription_is_an_error_not_completed(self):
        self._write_script(n_notes=0)
        self._run()
        text = self.progress_file.read_text()
        self.assertTrue(text.startswith("error|"), msg=text)
        self.assertIn("No notes were transcribed", text)


class TestBasicPitchCleanup(TestCase):
    """drop_ghost_notes / drop_tail_notes: the sub-octave ghosts and tail
    noise Basic Pitch produced on a distorted guitar render."""

    def setUp(self):
        sys.path.insert(0, str(SCRIPT.parent / "research"))
        import importlib
        self.bp = importlib.import_module("research_basic_pitch")
        import pretty_midi
        self.N = lambda p, s, e, v=80: pretty_midi.Note(velocity=v, pitch=p, start=s, end=e)

    def test_octave_ghost_under_louder_note_is_dropped(self):
        real, ghost = self.N(49, 8.05, 8.25, 69), self.N(37, 8.06, 8.25, 59)
        self.assertEqual(self.bp.drop_ghost_notes([real, ghost]), [real])

    def test_octave_plus_fifth_ghost_is_dropped(self):
        real, ghost = self.N(58, 9.21, 9.43, 79), self.N(39, 9.23, 9.38, 68)
        self.assertEqual(self.bp.drop_ghost_notes([real, ghost]), [real])

    def test_louder_low_note_is_kept(self):
        """A genuine octave doubling where the low note dominates survives."""
        low, high = self.N(49, 1.0, 1.3, 100), self.N(61, 1.0, 1.3, 60)
        self.assertEqual(len(self.bp.drop_ghost_notes([low, high])), 2)

    def test_non_overlapping_octave_is_kept(self):
        a, b = self.N(49, 1.0, 1.2, 70), self.N(61, 2.0, 2.2, 90)
        self.assertEqual(len(self.bp.drop_ghost_notes([a, b])), 2)

    def test_unrelated_interval_is_kept(self):
        a, b = self.N(49, 1.0, 1.3, 70), self.N(56, 1.0, 1.3, 90)  # a fifth
        self.assertEqual(len(self.bp.drop_ghost_notes([a, b])), 2)

    def test_quiet_tail_notes_are_dropped(self):
        import numpy as np
        sr = 16000
        y = np.zeros(sr * 10, dtype="float32")
        notes = [self.N(60 + i, float(i), i + 0.5) for i in range(8)]
        for n in notes:
            y[int(n.start * sr):int(n.end * sr)] = 0.5
        tail = self.N(64, 9.0, 9.5)
        y[int(9.0 * sr):int(9.5 * sr)] = 0.02            # ~28 dB below the notes
        kept = self.bp.drop_tail_notes(notes + [tail], y, sr)
        self.assertNotIn(tail, kept)
        self.assertEqual(len(kept), 8)

    def test_few_notes_are_never_filtered(self):
        import numpy as np
        notes = [self.N(60, 0.0, 0.5), self.N(62, 1.0, 1.5)]
        self.assertEqual(self.bp.drop_tail_notes(notes, np.zeros(16000 * 3, dtype="float32"), 16000), notes)


class TestLevelling(TestCase):
    """research_basic_pitch.load_normalised: quiet recordings (a Guitar Rig
    render peaking at -22 dBFS made MIROS return an empty MIDI) must be
    brought up to -1 dBFS; silence must be rejected."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="a2m_level_"))
        sys.path.insert(0, str(SCRIPT.parent / "research"))
        import importlib
        self.bp = importlib.import_module("research_basic_pitch")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _wav(self, amp, stereo=True, sr=48000):
        import numpy as np
        import soundfile as sf
        t = np.arange(sr) / sr
        y = amp * np.sin(2 * np.pi * 220 * t).astype("float32")
        data = np.stack([y, y], axis=1) if stereo else y
        path = self.tmpdir / f"t_{amp}.wav"
        sf.write(str(path), data, sr, subtype="PCM_24")
        return path

    def test_quiet_stereo_is_downmixed_and_peaks_at_minus_one_dbfs(self):
        y, sr, gain = self.bp.load_normalised(self._wav(0.077))
        self.assertEqual(y.ndim, 1)
        self.assertAlmostEqual(float(abs(y).max()), 0.89, places=2)
        self.assertGreater(gain, 20)

    def test_gain_is_capped(self):
        _, _, gain = self.bp.load_normalised(self._wav(0.001))
        self.assertLessEqual(gain, 30.01)

    def test_silence_is_rejected(self):
        with self.assertRaises(ValueError):
            self.bp.load_normalised(self._wav(0.0))


class TestErrorReporting(TestCase):
    """Covers the reported bug: when the Modal subprocess fails, the
    progress file used to only get "Modal failed with code 1. Check the
    log for details." — no hint of WHY. transcribe.py now keeps a rolling
    tail of the subprocess's own output and surfaces the real cause."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="a2m_err_"))
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

    def _write_failing_stub(self, error_line):
        script = self.tmpdir / "research_failing_modal.py"
        script.write_text(textwrap.dedent(f"""\
        import sys
        print("Loading model weights from Modal Volume...")
        print({error_line!r})
        sys.exit(1)
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

    def test_error_message_includes_real_cause_not_just_exit_code(self):
        script = self._write_failing_stub(
            "RuntimeError: CUDA out of memory on device A10G")
        proc = self._run(script)
        self.assertEqual(proc.returncode, 1)
        state, _, msg, extra = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("code 1", msg)
        self.assertIn("CUDA out of memory on device A10G", msg)
        self.assertNotEqual(
            msg.strip(),
            "Modal failed with code 1. Check the log for details.",
            "regression: error message must not collapse back to a bare "
            "exit-code-only message",
        )

    def test_extra_lines_contain_full_tail_for_full_log_view(self):
        script = self._write_failing_stub("ImportError: no module named foo")
        self._run(script)
        _, _, _, extra = parse_progress(self.progress_file)
        self.assertIn("--- last output lines ---", extra)
        self.assertTrue(any("ImportError: no module named foo" in l for l in extra))
        self.assertTrue(any("Loading model weights" in l for l in extra))

    def test_no_output_failure_still_gives_actionable_message(self):
        script = self.tmpdir / "research_silent_fail_modal.py"
        script.write_text("import sys; sys.exit(1)\n")
        proc = self._run(script)
        self.assertEqual(proc.returncode, 1)
        state, _, msg, _ = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("code 1", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
