#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for separate_sam.py — validates CLI, uv-based command construction,
progress parsing, output file discovery, and error handling.

SAM Audio runs in the cloud via Modal, so we cannot do a real end-to-end test
without credentials.  Instead we mock the ``uv`` binary and verify:

  - The correct ``uv run --project DIR modal run tools/modal_sam_audio.py``
    command is assembled from all CLI flags.
  - Progress lines (``Chunk X/Y``) are parsed correctly.
  - ``write_progress`` writes the expected progress-file format.
  - Output files (``_target.wav`` / ``_residual.wav``) are discovered correctly.
  - Missing modal script and subprocess failures produce proper errors.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import TestCase

# ---------------------------------------------------------------------------
# SUT
# ---------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve().parent / "separate_sam.py"

# ---------------------------------------------------------------------------
# Mock uv on PATH so _find_uv() resolves it in all tests
# ---------------------------------------------------------------------------
_UV_MOCK_DIR = Path(tempfile.mkdtemp(prefix="uv_mock_global_"))
_UV_MOCK = _UV_MOCK_DIR / "uv"
_UV_MOCK.write_text(textwrap.dedent("""\
#!/usr/bin/env python3
import sys, os
# Mock uv: strip "run --project DIR", "modal run" args,
# find .py (no ::) or .py::func, exec python3 on the script.
args = sys.argv[1:]
for i, a in enumerate(args):
    if '::' in a:
        script = a.split('::')[0]
        os.execvp('python3', ['python3', script] + args[i+1:])
        break
    if a.endswith('.py'):
        os.execvp('python3', ['python3'] + args[i:])
        break
sys.exit(1)
"""))
_UV_MOCK.chmod(0o755)
_ORIG_PATH = os.environ.get("PATH", "")
os.environ["PATH"] = str(_UV_MOCK_DIR) + ":" + _ORIG_PATH


def setUpModule():
    pass


def tearDownModule():
    os.environ["PATH"] = _ORIG_PATH
    shutil.rmtree(_UV_MOCK_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def write_progress_py(path, state, pct, msg, files=None):
    with open(path, "w") as f:
        f.write(f"{state}|{pct:.3f}|{msg}\n")
        if files:
            for fp in files:
                f.write(fp + "\n")


def parse_progress(path):
    if not path.exists():
        return None, None, None
    text = path.read_text()
    if not text.strip():
        return None, None, None
    lines = text.splitlines()
    parts = lines[0].split("|")
    if len(parts) < 3:
        return None, None, None
    return parts[0], float(parts[1]), "|".join(parts[2:])


def dummy_modal_script(path):
    """Write a minimal modal_sam_audio.py that exits cleanly."""
    path.write_text(textwrap.dedent("""\
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--output-dir")
    p.add_argument("--prompt")
    p.add_argument("--model")
    p.add_argument("--chunk-dur", type=float)
    p.add_argument("--overlap", type=float)
    p.add_argument("--ode-steps", type=int)
    p.add_argument("--ode-method")
    p.add_argument("--confidence-threshold", type=float)
    p.add_argument("--reranking-candidates", type=int)
    p.add_argument("--predict-spans", action=argparse.BooleanOptionalAction)
    p.add_argument("--internal-candidates", type=int, default=2)
    args = p.parse_args()
    print("OK")
    sys.exit(0)
    """))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWriteProgress(TestCase):
    """Tests for the write_progress helper."""

    def test_basic_state(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            p = Path(f.name)
        try:
            write_progress_py(p, "running", 0.5, "Processing...")
            state, pct, msg = parse_progress(p)
            self.assertEqual(state, "running")
            self.assertAlmostEqual(pct, 0.5)
            self.assertEqual(msg, "Processing...")
        finally:
            p.unlink(missing_ok=True)

    def test_with_files(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            p = Path(f.name)
        try:
            write_progress_py(p, "done", 1.0, "OK", ["a.wav", "b.wav"])
            lines = p.read_text().splitlines()
            self.assertEqual(lines[0], "done|1.000|OK")
            self.assertIn("a.wav", lines)
            self.assertIn("b.wav", lines)
        finally:
            p.unlink(missing_ok=True)

    def test_no_path_is_noop(self):
        from separate_sam import write_progress
        write_progress("", "running", 0.5, "msg")


class TestArgumentParsing(TestCase):
    """Basic CLI validation."""

    def test_missing_input_fails(self):
        cmd = [sys.executable, str(SCRIPT), "--prompt", "vocals"]
        p = subprocess.run(cmd, capture_output=True, timeout=30)
        self.assertNotEqual(p.returncode, 0)

    def test_unknown_flag_fails(self):
        cmd = [sys.executable, str(SCRIPT), "--input", "x.wav", "--bogus"]
        p = subprocess.run(cmd, capture_output=True, timeout=30)
        self.assertNotEqual(p.returncode, 0)

    def test_defaults_help(self):
        cmd = [sys.executable, str(SCRIPT), "--help"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0)
        for flag in ("--input", "--prompt", "--sam-dir", "--model",
                     "--gpu", "--steps", "--ode-method", "--chunk", "--overlap",
                     "--confidence", "--candidates", "--outdir", "--progress",
                     "--predict-spans", "--no-predict-spans", "--internal-candidates"):
            self.assertIn(flag, p.stdout)


class TestUvDiscovery(TestCase):
    """Tests for the _find_uv helper."""

    def test_finds_uv_from_path(self):
        from separate_sam import _find_uv
        result = _find_uv()
        self.assertIsNotNone(result)
        self.assertTrue(result.name == "uv")

    def test_uv_not_found_returns_none(self):
        from separate_sam import _find_uv
        import unittest.mock
        orig_path = os.environ.get("PATH", "")
        try:
            os.environ["PATH"] = "/dev/null"
            with unittest.mock.patch.object(Path, "home",
                                             return_value=Path("/nonexistent_home")):
                result = _find_uv()
                self.assertIsNone(result)
        finally:
            os.environ["PATH"] = orig_path


class TestCommandConstruction(TestCase):
    """Verify that separate_sam.py builds the correct uv-based subprocess command."""

    SCRIPT_DIR = Path(tempfile.mkdtemp(prefix="test_sam_"))

    @classmethod
    def setUpClass(cls):
        dummy_modal_script(cls.SCRIPT_DIR / "modal_sam_audio.py")

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sam_test_"))
        self.input_wav = self.tmpdir / "test_input.wav"
        self.input_wav.touch()
        self.progress_file = self.tmpdir / "progress.txt"
        self.outdir = self.tmpdir / "output"
        self.outdir.mkdir()

    def _run(self, extra_args=None):
        args = [
            str(SCRIPT),
            "--sam-dir", str(self.SCRIPT_DIR),
            "--input", str(self.input_wav),
            "--prompt", "saxophone",
            "--model", "facebook/sam-audio-large",
            "--gpu", "A100",
            "--steps", "50",
            "--ode-method", "euler",
            "--chunk", "10.0",
            "--overlap", "1.5",
            "--confidence", "0.3",
            "--candidates", "2",
            "--outdir", str(self.outdir),
            "--progress", str(self.progress_file),
        ]
        if extra_args:
            args.extend(extra_args)
        proc = subprocess.run(
            [sys.executable] + args,
            capture_output=True, text=True, timeout=30,
        )
        return proc

    def test_command_includes_uv_run(self):
        proc = self._run()
        stdout = proc.stdout + proc.stderr
        self.assertIn("uv run", stdout)
        self.assertIn("--project", stdout)

    def test_command_includes_all_flags(self):
        proc = self._run()
        stdout = proc.stdout + proc.stderr
        self.assertIn("prompt saxophone", stdout)
        self.assertIn("model facebook/sam-audio-large", stdout)
        self.assertIn("ode-steps 50", stdout)
        self.assertIn("ode-method euler", stdout)
        self.assertIn("confidence-threshold 0.3", stdout)
        self.assertIn("reranking-candidates 2", stdout)
        self.assertIn("chunk-dur 10.0", stdout)
        self.assertIn("overlap 1.5", stdout)
        self.assertIn("internal-candidates 2", stdout)
        self.assertIn("predict-spans", stdout)
        self.assertIn("::main", stdout)

    def test_gpu_set_via_env_var(self):
        """--gpu is passed via SAM_AUDIO_GPU env var, not CLI flag."""
        proc = self._run()
        stdout = proc.stdout + proc.stderr
        self.assertNotIn("--gpu", stdout)

    def test_default_flags_omitted_when_not_set(self):
        """confidence=0 and candidates=1 should NOT produce CLI flags."""
        self.progress_file.write_text("")
        proc = self._run(extra_args=[
            "--confidence", "0.0",
            "--candidates", "1",
        ])
        running_line = next((l for l in (proc.stdout or "").splitlines()
                             if l.startswith("Running:")), "")
        self.assertNotIn("--confidence-threshold", running_line)
        self.assertNotIn("--reranking-candidates", running_line)

    def test_command_includes_project_flag(self):
        proc = self._run()
        stdout = proc.stdout + proc.stderr
        self.assertIn(f"--project {self.SCRIPT_DIR}", stdout)

    def test_command_includes_modal_run(self):
        proc = self._run()
        stdout = proc.stdout + proc.stderr
        self.assertIn("modal run", stdout)
        self.assertIn("modal_sam_audio.py", stdout)

    def test_missing_modal_script_errors(self):
        """If tools/modal_sam_audio.py does not exist, write 'error'."""
        empty_dir = Path(tempfile.mkdtemp(prefix="no_modal_"))
        args = [
            str(SCRIPT),
            "--sam-dir", str(empty_dir),
            "--input", str(self.input_wav),
            "--prompt", "vocals",
            "--outdir", str(self.tmpdir / "out2"),
            "--progress", str(self.progress_file),
        ]
        proc = subprocess.run(
            [sys.executable] + args,
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0)
        state, _, msg = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("No encontrado", msg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls.SCRIPT_DIR), ignore_errors=True)


class TestProgressParsing(TestCase):
    """Verify that the SUT correctly parses progress lines from modal output."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sam_progress_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.input_wav = self.tmpdir / "input.wav"
        self.input_wav.write_bytes(b"\x00" * 8000)
        self.outdir = self.tmpdir / "out"
        self.outdir.mkdir(exist_ok=True)

    def _run_with_modal_output(self, modal_lines):
        script_path = self.tmpdir / "modal_sam_audio.py"
        script_path.write_text(textwrap.dedent(f"""\
        import sys
        for line in {json.dumps(modal_lines)}:
            print(line)
        sys.exit(0)
        """))
        (self.outdir / "input_sam_50_target.wav").touch()
        (self.outdir / "input_sam_50_residual.wav").touch()
        args = [
            str(SCRIPT),
            "--sam-dir", str(self.tmpdir),
            "--input", str(self.input_wav),
            "--prompt", "vocals",
            "--outdir", str(self.outdir),
            "--progress", str(self.progress_file),
        ]
        proc = subprocess.run(
            [sys.executable] + args,
            capture_output=True, text=True, timeout=30,
        )
        return proc

    def test_chunk_progress_parsed(self):
        self._run_with_modal_output([
            "Chunk 1/10: processing",
            "Chunk 5/10: processing",
            "Chunk 10/10: processing",
        ])
        state, pct, msg = parse_progress(self.progress_file)
        self.assertEqual(state, "done")
        self.assertGreaterEqual(pct, 0.9)

    def test_single_chunk(self):
        self._run_with_modal_output(["Chunk 1/1: done"])
        state, pct, msg = parse_progress(self.progress_file)
        self.assertEqual(state, "done")

    def test_non_chunk_lines(self):
        self._run_with_modal_output([
            "Uploading audio...",
            "Starting inference...",
            "Done.",
        ])
        state, pct, msg = parse_progress(self.progress_file)
        self.assertEqual(state, "done")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestOutputDiscovery(TestCase):
    """Verify that separate_sam.py discovers _target.wav / _residual.wav files."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sam_output_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.input_wav = self.tmpdir / "test_track.wav"
        self.input_wav.write_bytes(b"\x00" * 8000)
        self.outdir = self.tmpdir / "out"
        self.outdir.mkdir()

    def _write_modal_stub(self):
        (self.tmpdir / "modal_sam_audio.py").write_text("import sys; sys.exit(0)\n")

    def _run(self):
        self._write_modal_stub()
        args = [
            str(SCRIPT),
            "--sam-dir", str(self.tmpdir),
            "--input", str(self.input_wav),
            "--prompt", "vocals",
            "--outdir", str(self.outdir),
            "--progress", str(self.progress_file),
        ]
        subprocess.run(
            [sys.executable] + args,
            capture_output=True, text=True, timeout=30,
        )

    def test_discovers_target_and_residual(self):
        (self.outdir / "test_track_sam_midpoint64_target.wav").touch()
        (self.outdir / "test_track_sam_midpoint64_residual.wav").touch()
        (self.outdir / "other.wav").touch()
        self._run()
        lines = self.progress_file.read_text().splitlines()
        listed_files = [l for l in lines[1:] if l.strip()]
        self.assertEqual(len(listed_files), 2)
        for f in listed_files:
            self.assertTrue(f.endswith("target.wav") or f.endswith("residual.wav"))

    def test_fallback_to_all_wavs(self):
        (self.outdir / "output_vocals.wav").touch()
        (self.outdir / "output_bass.wav").touch()
        self._run()
        lines = self.progress_file.read_text().splitlines()
        listed_files = [l for l in lines[1:] if l.strip()]
        self.assertEqual(len(listed_files), 2)

    def test_no_output_files_still_marks_done(self):
        self._run()
        state, pct, msg = parse_progress(self.progress_file)
        self.assertEqual(state, "done")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestErrorHandling(TestCase):
    """Error conditions that should produce non-zero exit + 'error' state."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sam_error_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.input_wav = self.tmpdir / "in.wav"
        self.input_wav.write_bytes(b"\x00" * 8000)
        self.outdir = self.tmpdir / "out"

    def _ensure_modal_stub(self, code=0):
        (self.tmpdir / "modal_sam_audio.py").write_text(
            f"import sys; sys.exit({code})\n")

    def _run(self):
        args = [
            str(SCRIPT),
            "--sam-dir", str(self.tmpdir),
            "--input", str(self.input_wav),
            "--prompt", "vocals",
            "--outdir", str(self.outdir),
            "--progress", str(self.progress_file),
        ]
        return subprocess.run(
            [sys.executable] + args,
            capture_output=True, text=True, timeout=30,
        )

    def test_subprocess_failure(self):
        self._ensure_modal_stub(42)
        proc = self._run()
        self.assertNotEqual(proc.returncode, 0)
        state, _, msg = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("fallo", msg)

    def test_missing_input_handled(self):
        self._ensure_modal_stub(0)
        args = [
            str(SCRIPT),
            "--sam-dir", str(self.tmpdir),
            "--input", "/nonexistent/file.wav",
            "--prompt", "vocals",
            "--outdir", str(self.outdir),
            "--progress", str(self.progress_file),
        ]
        proc = subprocess.run(
            [sys.executable] + args,
            capture_output=True, text=True, timeout=30,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertNotIn("Traceback", combined)

    def test_progress_written_even_on_error(self):
        self._ensure_modal_stub(1)
        self._run()
        self.assertTrue(self.progress_file.exists())
        content = self.progress_file.read_text()
        self.assertIn("error", content)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestIntegrationWithStemSeparatorLua(TestCase):
    """Verify the shell-command template that StemSeparator.lua builds
    actually passes all required arguments correctly.

    Uses the global mock ``uv`` on PATH (set at module level)
    so the SAM subprocess runs without cloud credentials."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sam_integration_"))
        self.progress_file = self.tmpdir / "progress.txt"

        import math, struct, wave
        self.input_wav = self.tmpdir / "integration_input.wav"
        nframes = int(44100 * 3)
        samples = []
        for i in range(nframes):
            val = int(round(16000 * math.sin(2 * math.pi * 440 * i / 44100)))
            val = max(-32768, min(32767, val))
            samples.append(struct.pack("<h", val))
        with wave.open(str(self.input_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"".join(samples))

        self.outdir = self.tmpdir / "output"
        self.outdir.mkdir()

        self.script_dir = self.tmpdir / "audio-processing"
        self.script_dir.mkdir()
        stub = self.script_dir / "modal_sam_audio.py"
        stub.write_text(textwrap.dedent("""\
        import argparse, json, sys
        p = argparse.ArgumentParser()
        p.add_argument("--input")
        p.add_argument("--output-dir")
        p.add_argument("--prompt")
        p.add_argument("--model")
        p.add_argument("--chunk-dur", type=float)
        p.add_argument("--overlap", type=float)
        p.add_argument("--ode-steps", type=int)
        p.add_argument("--ode-method")
        p.add_argument("--confidence-threshold", type=float, default=None)
        p.add_argument("--reranking-candidates", type=int, default=None)
        p.add_argument("--predict-spans", action=argparse.BooleanOptionalAction)
        p.add_argument("--internal-candidates", type=int, default=2)
        args = p.parse_args()
        with open(sys.argv[0].replace(".py", "_parsed.json"), "w") as f:
            json.dump(vars(args), f, default=str)
        from pathlib import Path
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        (out_dir / f"{stem}_sam_midpoint64_target.wav").touch()
        (out_dir / f"{stem}_sam_midpoint64_residual.wav").touch()
        print("OK")
        sys.exit(0)
        """))

    def test_lua_template_params_roundtrip(self):
        args = [
            str(SCRIPT),
            "--sam-dir", str(self.script_dir),
            "--input", str(self.input_wav),
            "--prompt", "jazz trumpet",
            "--model", "facebook/sam-audio-large",
            "--gpu", "A100",
            "--steps", "64",
            "--ode-method", "midpoint",
            "--chunk", "15.0",
            "--overlap", "2.0",
            "--confidence", "0.0",
            "--candidates", "1",
            "--outdir", str(self.outdir),
            "--progress", str(self.progress_file),
        ]
        proc = subprocess.run(
            [sys.executable] + args,
            capture_output=True, text=True, timeout=30,
            cwd=str(self.script_dir),
        )
        parsed_file = self.script_dir / "modal_sam_audio_parsed.json"
        self.assertTrue(
            parsed_file.exists(),
            msg=f"JSON not found. stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        parsed = json.loads(parsed_file.read_text())
        self.assertEqual(parsed["input"], str(self.input_wav))
        self.assertEqual(parsed["prompt"], "jazz trumpet")
        self.assertEqual(parsed["model"], "facebook/sam-audio-large")
        self.assertEqual(parsed["ode_steps"], 64)
        self.assertEqual(parsed["ode_method"], "midpoint")
        self.assertEqual(parsed["chunk_dur"], 15.0)
        self.assertEqual(parsed["overlap"], 2.0)
        self.assertEqual(parsed["internal_candidates"], 2)
        self.assertEqual(parsed["predict_spans"], True)

    def test_progress_written_on_success(self):
        args = [
            str(SCRIPT),
            "--sam-dir", str(self.script_dir),
            "--input", str(self.input_wav),
            "--prompt", "vocals",
            "--outdir", str(self.outdir),
            "--progress", str(self.progress_file),
        ]
        proc = subprocess.run(
            [sys.executable] + args,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(
            proc.returncode, 0,
            msg=f"rc={proc.returncode}\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
        )
        state, pct, msg = parse_progress(self.progress_file)
        self.assertEqual(state, "done")

    def test_progress_lists_output_files(self):
        args = [
            str(SCRIPT),
            "--sam-dir", str(self.script_dir),
            "--input", str(self.input_wav),
            "--prompt", "vocals",
            "--outdir", str(self.outdir),
            "--progress", str(self.progress_file),
        ]
        subprocess.run(
            [sys.executable] + args,
            capture_output=True, text=True, timeout=30,
        )
        lines = self.progress_file.read_text().splitlines()
        listed = [l for l in lines[1:] if l.strip()]
        self.assertGreaterEqual(len(listed), 1)
        for fp in listed:
            self.assertTrue("target" in fp or "residual" in fp,
                            f"Unexpected file: {fp}")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)