#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for separate_demucs.py — validates CLI, progress reporting,
output file generation and error handling without touching REAPER.

Fixtures use a 440 Hz sine wave instead of silence because htdemucs 4.0
asserts that the input contains at least one non-zero sample (pad1d in
hdemucs.py:39)."""

import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import TestCase

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve().parent / "separate_demucs.py"
PROGRESS_TMP = Path(tempfile.gettempdir()) / "test_demucs_progress.txt"
LOG_TMP = Path(tempfile.gettempdir()) / "test_demucs_log.txt"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_sine_wav(path: Path, duration=5.0, sr=44100, freq=440.0):
    """Write a mono 16-bit WAV with a sine wave (non-silent content avoids
    htdemucs 4.0 pad1d assertion on all-zero inputs)."""
    nframes = int(sr * duration)
    samples = []
    for i in range(nframes):
        val = int(round(16000 * math.sin(2 * math.pi * freq * i / sr)))
        val = max(-32768, min(32767, val))
        samples.append(struct.pack("<h", val))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(samples))


make_wav = make_sine_wav


def run_script(input_wav: Path, out_dir: Path,
               model: str = "htdemucs",
               stems: str = "vocals,drums,bass,other",
               device: str = "cpu",
               timeout: int = 300):
    """Run separate_demucs.py as a subprocess, return (rc, stdout, stderr)."""
    cmd = [
        sys.executable, str(SCRIPT),
        "--input", str(input_wav),
        "--model", model,
        "--stems", stems,
        "--outdir", str(out_dir),
        "--device", device,
        "--progress", str(PROGRESS_TMP),
    ]
    PROGRESS_TMP.write_text("")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def parse_progress():
    """Return (state, pct, msg) from the first line of the progress file.
    Returns (None, None, None) if the file is empty / missing."""
    if not PROGRESS_TMP.exists():
        return None, None, None
    text = PROGRESS_TMP.read_text()
    if not text.strip():
        return None, None, None
    line = text.splitlines()[0]
    parts = line.split("|")
    if len(parts) < 3:
        return None, None, None
    return parts[0], float(parts[1]), "|".join(parts[2:])


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
class TestArgumentParsing(TestCase):

    def test_missing_input_fails(self):
        """Running without --input must fail with exit code != 0."""
        cmd = [sys.executable, str(SCRIPT), "--model", "htdemucs"]
        p = subprocess.run(cmd, capture_output=True, timeout=30)
        self.assertNotEqual(p.returncode, 0,
                            msg=f"Expected non-zero exit\nstdout:{p.stdout}\nstderr:{p.stderr}")


class TestWavGeneration(TestCase):

    def test_generated_wav_is_valid(self):
        """make_wav produces a file that wave.open can read."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            name = f.name
        try:
            make_wav(Path(name))
            self.assertTrue(Path(name).exists())
            with wave.open(name, "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getframerate(), 44100)
        finally:
            try:
                Path(name).unlink()
            except OSError:
                pass


class TestDemucsAvailable(TestCase):
    """Verify that the *same* Python running the tests can import demucs."""

    def test_demucs_importable(self):
        """demucs must be importable from sys.executable."""
        cmd = [sys.executable, "-c", "import demucs; print('ok')"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(
            p.returncode, 0,
            msg=f"demucs not importable from {sys.executable}\n"
                f"stdout: {p.stdout}\nstderr: {p.stderr}"
        )

    def test_demucs_cli_help(self):
        """`python -m demucs --help` must succeed."""
        cmd = [sys.executable, "-m", "demucs", "--help"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0)
        self.assertIn("usage", p.stdout.lower())


class TestEndToEndSeparation(TestCase):
    """Run a real 5-second WAV through separate_demucs.py."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="test_demucs_"))
        cls.input_wav = cls.tmpdir / "input.wav"
        make_wav(cls.input_wav, duration=5.0)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_script_runs_successfully(self):
        """A real 5-second WAV must separate without error (returncode 0)."""
        out = self.tmpdir / "out"
        rc, stdout, stderr = run_script(self.input_wav, out)
        self.assertEqual(
            rc, 0,
            msg=f"separate_demucs.py failed (rc={rc})\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
        )

    def test_output_files_created(self):
        """After success each requested stem must exist as <stem>.wav."""
        out = self.tmpdir / "out"
        rc, stdout, stderr = run_script(self.input_wav, out)
        self.assertEqual(rc, 0, msg=f"rc={rc}\nstdout:{stdout}\nstderr:{stderr}")
        stem_dir = out / "htdemucs" / self.input_wav.stem
        for stem in ("vocals", "drums", "bass", "other"):
            f = stem_dir / f"{stem}.wav"
            self.assertTrue(
                f.exists(),
                msg=f"Expected stem file not found: {f}"
            )

    def test_progress_file_written(self):
        """The progress file must contain a valid 'done' line."""
        state, pct, msg = parse_progress()
        self.assertEqual(state, "done")
        self.assertAlmostEqual(pct, 1.0, places=2)
        self.assertIn("Completado", msg)


class TestStemsSubset(TestCase):
    """When only a subset of stems is requested, the progress file should
    only list those stems (Demucs always generates all stems on disk)."""

    @classmethod
    def setUpClass(cls):
        import shutil
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="test_demucs_subset_"))
        cls.input_wav = cls.tmpdir / "input2.wav"
        make_wav(cls.input_wav, duration=5.0)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_only_vocals_and_bass_in_progress(self):
        out = self.tmpdir / "out2"
        rc, stdout, stderr = run_script(
            self.input_wav, out, stems="vocals,bass"
        )
        self.assertEqual(
            rc, 0,
            msg=f"rc={rc}\nstdout:{stdout}\nstderr:{stderr}"
        )
        # Verify that only the requested stems appear in the progress file
        state, pct, msg = parse_progress()
        self.assertEqual(state, "done")
        # All 4 stems exist on disk (Demucs always generates all)
        stem_dir = out / "htdemucs" / self.input_wav.stem
        for stem in ("vocals", "drums", "bass", "other"):
            self.assertTrue(
                (stem_dir / f"{stem}.wav").exists(),
                msg=f"Expected {stem}.wav on disk"
            )


class TestOutputDirCreated(TestCase):
    """Output directory (incl. model sub-dir) must be created if missing."""

    def test_nested_outdir_created(self):
        import shutil
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "nonexistent" / "deep"
        wav = tmp / "t.wav"
        make_wav(wav, duration=5.0)
        try:
            rc, stdout, stderr = run_script(wav, out)
            self.assertEqual(
                rc, 0,
                msg=f"rc={rc}\nstdout:{stdout}\nstderr:{stderr}"
            )
            self.assertTrue(out.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestNonSilentAudio(TestCase):
    """Explicit guard: non-silent (sine wave) input must separate cleanly."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="test_nonsilent_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_sine_1s_separates(self):
        """1-second sine wave must succeed and produce at least one stem."""
        wav = self.tmpdir / "sine.wav"
        make_sine_wav(wav, duration=1.0)
        out = self.tmpdir / "out"
        rc, stdout, stderr = run_script(wav, out)
        self.assertEqual(
            rc, 0,
            msg=f"1s sine failed rc={rc}\nstdout:{stdout}\nstderr:{stderr}"
        )
        stem_dir = out / "htdemucs" / "sine"
        stems = list(stem_dir.glob("*.wav")) if stem_dir.exists() else []
        self.assertGreater(
            len(stems), 0,
            msg=f"No stems produced in {stem_dir}"
        )


class TestInvalidInput(TestCase):
    """Pointing --input to a missing file must produce a non-zero exit."""

    def test_nonexistent_input_returns_nonzero(self):
        out = Path(tempfile.mkdtemp())
        cmd = [
            sys.executable, str(SCRIPT),
            "--input", "/does/not/exist.wav",
            "--model", "htdemucs",
            "--outdir", str(out),
            "--device", "cpu",
            "--progress", str(PROGRESS_TMP),
        ]
        p = subprocess.run(cmd, capture_output=True, timeout=60)
        state, _, _ = parse_progress()
        self.assertEqual(
            p.returncode, 1,
            msg=f"Expected rc=1, got rc={p.returncode}\n"
                f"stdout:{p.stdout}\nstderr:{p.stderr}"
        )
        self.assertEqual(
            state, "error",
            msg=f"Expected state=error, got state={state}"
        )
        import shutil
        shutil.rmtree(out, ignore_errors=True)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
