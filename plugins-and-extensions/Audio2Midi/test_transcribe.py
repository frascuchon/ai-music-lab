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
sys.path.insert(0, str(SCRIPT.parent))
import transcribe  # noqa: E402

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

    def _run(self, script, model="yourmt3"):
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
    YOURMT3_GPU at import time, so transcribe.py has to export the chosen GPU
    into the subprocess environment."""

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
                os.environ.get("YOURMT3_GPU", "-"))
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

    def test_yourmt3_gpu_is_exported(self):
        self.assertEqual(self._seen("yourmt3", "T4"), "T4")

    def test_gpu_choice_reaches_the_modal_script(self):
        self.assertEqual(self._seen("yourmt3", "A100"), "A100")


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
            print("[stub] gpu env:", os.environ.get("YOURMT3_GPU"))
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
        self.assertIn("gpu env: None", res.stdout)

    def test_bpm_is_forwarded_to_the_local_script(self):
        self._write_script(n_notes=3)
        self.script.write_text(self.script.read_text().replace(
            'p.add_argument("--no-beat-tracking", action="store_true")',
            'p.add_argument("--no-beat-tracking", action="store_true"); p.add_argument("--bpm", type=float, default=None)').replace(
            'tempo_bpm=100', 'tempo_bpm=(a.bpm or 100)'))
        self._run(["--bpm", "87"])
        self.assertIn("TEMPO|87", self.progress_file.read_text())

    def test_onset_threshold_is_forwarded_to_the_local_script(self):
        self._write_script(n_notes=3)
        self.script.write_text(self.script.read_text().replace(
            'p.add_argument("--no-beat-tracking", action="store_true")',
            'p.add_argument("--no-beat-tracking", action="store_true"); p.add_argument("--onset-threshold", type=float)').replace(
            'tempo_bpm=100', 'tempo_bpm=int(a.onset_threshold * 100)'))
        self._run(["--onset-threshold", "0.8"])
        self.assertIn("TEMPO|80", self.progress_file.read_text())

    def test_empty_transcription_is_an_error_not_completed(self):
        self._write_script(n_notes=0)
        self._run()
        text = self.progress_file.read_text()
        self.assertTrue(text.startswith("error|"), msg=text)
        self.assertIn("No notes were transcribed", text)


class TestExecutionSelection(TestCase):
    """--execution picks local vs Modal per run. Basic Pitch supports both (the
    Modal one is CPU-only, so it never gets a GPU env var); YourMT3+ is
    Modal-only. Omitting the flag must keep the previous behaviour."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="a2m_exec_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.input_wav = self.tmpdir / "guitar.wav"
        self.input_wav.write_bytes(b"\x00" * 8000)
        self.outdir = self.tmpdir / "out"
        self.outdir.mkdir()
        self.shared_dir = self.tmpdir / "shared"
        self.shared_dir.mkdir()
        (self.shared_dir / "pyproject.toml").write_text("[project]\nname='shared'\n")
        # One stub that works as local (--input/--out) or Modal (--audio-path/
        # --out-dir) script, and records how it was called.
        self.script = self.tmpdir / "stub.py"
        self.script.write_text(textwrap.dedent(f"""\
            import argparse, json, os, sys
            sys.path.insert(0, {str(SCRIPT.parent)!r})
            from test_transcribe import make_midi
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument("--input"); p.add_argument("--out")
            p.add_argument("--audio-path"); p.add_argument("--out-dir")
            p.add_argument("--force", action="store_true")
            p.add_argument("--no-beat-tracking", action="store_true")
            p.add_argument("--bpm"); p.add_argument("--onset-threshold")
            a, unknown = p.parse_known_args()
            out = Path(a.out) if a.out else Path(a.out_dir) / "transcribed_cuda.mid"
            Path({str(self.tmpdir)!r}, "seen.json").write_text(json.dumps(dict(
                vars(a), unknown=unknown, mode="modal" if a.audio_path else "local",
                gpu=[os.environ.get(k) for k in ("YOURMT3_GPU", "BASICPITCH_GPU")])))
            make_midi(out, tempo_bpm=100)
        """))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, model, *extra, shared=None):
        res = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--shared-dir", str(shared or self.shared_dir), "--script", str(self.script),
             "--input", str(self.input_wav), "--out-dir", str(self.outdir),
             "--model", model, "--progress", str(self.progress_file), *extra],
            capture_output=True, text=True, timeout=30)
        seen = self.tmpdir / "seen.json"
        import json
        return res, (json.loads(seen.read_text()) if seen.exists() else None)

    def test_basicpitch_defaults_to_local(self):
        _, seen = self._run("basicpitch")
        self.assertEqual(seen["mode"], "local")

    def test_basicpitch_explicit_local(self):
        _, seen = self._run("basicpitch", "--execution", "local",
                            shared=self.tmpdir / "no_shared")
        self.assertEqual(seen["mode"], "local")

    def test_basicpitch_remote_goes_through_modal(self):
        res, seen = self._run("basicpitch", "--execution", "remote", "--gpu", "A100")
        self.assertEqual(seen["mode"], "modal", msg=res.stdout)
        self.assertTrue(seen["force"])
        self.assertTrue(self.progress_file.read_text().startswith("done|"))
        self.assertIn("guitar__basicpitch.mid", self.progress_file.read_text())
        self.assertIn("TEMPO|100", self.progress_file.read_text())

    def test_basicpitch_remote_gets_no_gpu_env(self):
        _, seen = self._run("basicpitch", "--execution", "remote", "--gpu", "A100")
        self.assertEqual(seen["gpu"], [None, None])

    def test_basicpitch_remote_forwards_tuning_options(self):
        _, seen = self._run("basicpitch", "--execution", "remote", "--onset-threshold", "0.8",
                            "--bpm", "87", "--no-beat-tracking")
        self.assertEqual(seen["onset_threshold"], "0.8")
        self.assertEqual(seen["bpm"], "87.0")
        self.assertTrue(seen["no_beat_tracking"])

    def test_basicpitch_remote_requires_the_shared_project(self):
        self._run("basicpitch", "--execution", "remote", shared=self.tmpdir / "no_shared")
        text = self.progress_file.read_text()
        self.assertTrue(text.startswith("error|"), msg=text)
        self.assertIn("shared/pyproject.toml", text)

    def test_yourmt3_remote_is_unchanged_and_ignores_tuning_options(self):
        _, seen = self._run("yourmt3", "--gpu", "T4", "--onset-threshold", "0.8", "--bpm", "90")
        self.assertEqual(seen["mode"], "modal")
        self.assertIsNone(seen["onset_threshold"])
        self.assertIsNone(seen["bpm"])
        self.assertEqual(seen["gpu"], ["T4", None])

    def test_yourmt3_cannot_run_locally(self):
        _, seen = self._run("yourmt3", "--execution", "local")
        self.assertIsNone(seen, "the script must not even be launched")
        text = self.progress_file.read_text()
        self.assertTrue(text.startswith("error|"), msg=text)
        self.assertIn("cannot run locally", text)

    def test_remote_empty_transcription_is_still_an_error(self):
        self.script.write_text(self.script.read_text().replace("tempo_bpm=100)", "tempo_bpm=100, n_notes=0)"))
        self._run("basicpitch", "--execution", "remote")
        text = self.progress_file.read_text()
        self.assertTrue(text.startswith("error|"), msg=text)
        self.assertIn("No notes were transcribed", text)


def make_multitrack_midi(path, n_empty=1, n_filled=1, tempo_bpm=100):
    """SMF with `n_filled` instruments holding notes and `n_empty` instruments
    that only carry a program change (they would import as empty clips)."""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    tt = mido.MidiTrack()
    tt.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    mid.tracks.append(tt)
    ch = 0
    for k in range(n_filled):
        tr = mido.MidiTrack()
        tr.append(mido.Message("program_change", program=24 + k, channel=ch, time=0))
        tr.append(mido.Message("note_on", note=60, velocity=80, channel=ch, time=0))
        tr.append(mido.Message("note_off", note=60, velocity=0, channel=ch, time=240))
        mid.tracks.append(tr)
        ch += 1
    for k in range(n_empty):
        tr = mido.MidiTrack()
        tr.append(mido.Message("program_change", program=40 + k, channel=ch, time=0))
        mid.tracks.append(tr)
        ch += 1
    mid.save(str(path))


class TestPruneEmptyInstruments(TestCase):
    """Instruments without notes must be discarded from the transcription so
    they never become empty clips/tracks in REAPER."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="a2m_prune_"))
        self.mid = self.tmpdir / "t.mid"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _count(self):
        import mido
        return len(mido.MidiFile(str(self.mid)).tracks)

    def test_empty_instruments_are_removed(self):
        make_multitrack_midi(self.mid, n_empty=2, n_filled=1)
        self.assertEqual(self._count(), 4)  # tempo + 1 filled + 2 empty
        self.assertEqual(transcribe.prune_empty_instruments(self.mid), 2)
        self.assertEqual(self._count(), 2)  # tempo + 1 filled

    def test_notes_and_tempo_survive_pruning(self):
        make_multitrack_midi(self.mid, n_empty=1, n_filled=2, tempo_bpm=93)
        transcribe.prune_empty_instruments(self.mid)
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(self.mid))
        self.assertEqual([len(i.notes) for i in pm.instruments], [1, 1])
        self.assertAlmostEqual(transcribe.get_midi_tempo(self.mid), 93, delta=0.5)

    def test_file_without_empty_instruments_is_untouched(self):
        make_multitrack_midi(self.mid, n_empty=0, n_filled=2)
        before = self.mid.read_bytes()
        self.assertEqual(transcribe.prune_empty_instruments(self.mid), 0)
        self.assertEqual(self.mid.read_bytes(), before)

    def test_all_empty_file_is_left_alone_for_the_error_path(self):
        make_multitrack_midi(self.mid, n_empty=2, n_filled=0)
        before = self.mid.read_bytes()
        self.assertEqual(transcribe.prune_empty_instruments(self.mid), 0)
        self.assertEqual(self.mid.read_bytes(), before)

    def test_unreadable_file_does_not_raise(self):
        self.mid.write_bytes(b"garbage")
        self.assertEqual(transcribe.prune_empty_instruments(self.mid), 0)


class TestMirosModel(TestCase):
    """MIROS (single instrument, Modal A10G+) is selectable again; it must keep
    its GPU env var, the T4 guard and --no-beat-tracking, and empty tracks in
    its output must be discarded."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="a2m_miros_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.input_wav = self.tmpdir / "gtr.wav"
        self.input_wav.write_bytes(b"\x00" * 8000)
        self.outdir = self.tmpdir / "out"
        self.outdir.mkdir()
        self.shared_dir = self.tmpdir / "shared"
        self.shared_dir.mkdir()
        (self.shared_dir / "pyproject.toml").write_text("[project]\nname='shared'\n")
        self.script = self.tmpdir / "stub.py"
        self.script.write_text(textwrap.dedent(f"""\
            import argparse, json, os, sys
            sys.path.insert(0, {str(SCRIPT.parent)!r})
            from test_transcribe import make_multitrack_midi
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument("--audio-path"); p.add_argument("--out-dir")
            p.add_argument("--force", action="store_true")
            p.add_argument("--no-beat-tracking", action="store_true")
            a = p.parse_args()
            Path({str(self.tmpdir)!r}, "seen.json").write_text(json.dumps(dict(
                no_bt=a.no_beat_tracking, miros_gpu=os.environ.get("MIROS_GPU"),
                yourmt3_gpu=os.environ.get("YOURMT3_GPU"))))
            print("Chunk 1/2: x", flush=True)
            make_multitrack_midi(Path(a.out_dir) / "transcribed_cuda.mid",
                                 n_empty=2, n_filled=1)
        """))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, *extra, model="miros"):
        res = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--shared-dir", str(self.shared_dir), "--script", str(self.script),
             "--input", str(self.input_wav), "--out-dir", str(self.outdir),
             "--model", model, "--progress", str(self.progress_file), *extra],
            capture_output=True, text=True, timeout=30)
        import json
        seen = self.tmpdir / "seen.json"
        return res, (json.loads(seen.read_text()) if seen.exists() else None)

    def test_miros_runs_on_modal_with_its_gpu_env(self):
        res, seen = self._run("--gpu", "A100")
        self.assertEqual(seen["miros_gpu"], "A100")
        self.assertIsNone(seen["yourmt3_gpu"])

    def test_miros_on_t4_is_rejected_before_launching_modal(self):
        _, seen = self._run("--gpu", "T4")
        self.assertIsNone(seen)
        self.assertIn("cannot run on T4", self.progress_file.read_text())

    def test_no_beat_tracking_is_forwarded_to_miros(self):
        _, seen = self._run("--gpu", "A10G", "--no-beat-tracking")
        self.assertTrue(seen["no_bt"])

    def test_beat_tracking_is_not_disabled_by_default(self):
        _, seen = self._run("--gpu", "A10G")
        self.assertFalse(seen["no_bt"])

    def test_empty_instruments_are_discarded_and_reported(self):
        self._run("--gpu", "A10G")
        state, _, msg, extra = parse_progress(self.progress_file)
        self.assertEqual(state, "done")
        self.assertIn("INSTRUMENTS|1", extra)
        self.assertIn("2 empty discarded", msg)
        import pretty_midi
        import mido
        mid = next(self.outdir.glob("*__miros.mid"))
        self.assertEqual(len(mido.MidiFile(str(mid)).tracks), 2)  # tempo + 1

    def test_yourmt3_output_is_pruned_too_and_keeps_its_env(self):
        self._run("--gpu", "T4", model="yourmt3")
        state, _, _, extra = parse_progress(self.progress_file)
        self.assertEqual(state, "done")
        self.assertIn("INSTRUMENTS|1", extra)


class TestPanelMirosWiring(TestCase):
    """panel.lua can't run without REAPER; check its static wiring instead."""

    PANEL = (SCRIPT.parent / "panel.lua").read_text()

    def test_miros_is_in_the_model_list_with_a_label_and_script(self):
        self.assertIn('{ "basicpitch", "miros", "yourmt3" }', self.PANEL)
        self.assertIn("MIROS  (single instrument", self.PANEL)
        self.assertIn("research_miros_modal.py", self.PANEL)

    def test_existing_models_are_still_present(self):
        self.assertIn("research_basic_pitch.py", self.PANEL)
        self.assertIn("research_yourmt3_modal.py", self.PANEL)
        self.assertIn("YourMT3+  (multi-instr.", self.PANEL)

    def test_single_instrument_models_warn_and_point_to_yourmt3(self):
        self.assertIn("A2M_SINGLE_INSTRUMENT = { basicpitch = true, miros = true }", self.PANEL)
        self.assertIn("use YourMT3+", self.PANEL)

    def test_default_model_is_still_basic_pitch(self):
        self.assertIn("model_idx       = 1", self.PANEL)
        self.assertTrue(self.PANEL.split("A2M_MODELS  = {")[1].lstrip().startswith('"basicpitch"'))

    def test_miros_defaults_to_a_gpu_that_supports_flash_attn(self):
        self.assertIn("miros = 1", self.PANEL)  # index 1 = A10G


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
    render peaking at -22 dBFS made a transcription model return an empty MIDI) must be
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

    def _run(self, script, model="yourmt3"):
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
