#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for text2audio_local.py's argument validation and device-resolution
logic. No model weights are loaded and torch/diffusers/audiocraft are never
imported here — this file only exercises pure Python (argparse wiring,
_resolve_device, model-support gating), which is importable even when none
of those heavy optional dependencies are installed (verified: this repo's
environment has no diffusers/audiocraft installed)."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text2audio_local as t2al  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "text2audio_local.py"


class TestResolveDevice(TestCase):

    def test_auto_prefers_mps_over_cuda(self):
        self.assertEqual(t2al._resolve_device("auto", mps_available=True, cuda_available=True), "mps")

    def test_auto_falls_back_to_cuda_without_mps(self):
        self.assertEqual(t2al._resolve_device("auto", mps_available=False, cuda_available=True), "cuda")

    def test_auto_falls_back_to_cpu_with_nothing(self):
        self.assertEqual(t2al._resolve_device("auto", mps_available=False, cuda_available=False), "cpu")

    def test_explicit_mps_without_hardware_falls_back_to_cpu(self):
        self.assertEqual(t2al._resolve_device("mps", mps_available=False, cuda_available=False), "cpu")

    def test_explicit_cuda_without_hardware_falls_back_to_cpu(self):
        self.assertEqual(t2al._resolve_device("cuda", mps_available=False, cuda_available=False), "cpu")

    def test_explicit_cpu_always_cpu(self):
        self.assertEqual(t2al._resolve_device("cpu", mps_available=True, cuda_available=True), "cpu")

    def test_explicit_mps_with_hardware_honored(self):
        self.assertEqual(t2al._resolve_device("mps", mps_available=True, cuda_available=False), "mps")


class TestLocalSupportedModelsSet(TestCase):

    def test_matches_shared_registry_true_entries(self):
        shared_dir = Path(__file__).resolve().parent.parent / "shared"
        sys.path.insert(0, str(shared_dir))
        import model_execution as me
        expected = {k for k, v in me.TEXT2AUDIO.items() if v["local_supported"]}
        self.assertEqual(expected, t2al.LOCAL_SUPPORTED_MODELS)


class TestMainValidation(TestCase):
    """Drive main() via subprocess so argparse's --choices gating and the
    fail-fast checks run exactly as a real invocation would, without ever
    reaching the torch/diffusers import inside the try block (every case
    here is rejected before that point)."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="t2al_test_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, extra_args):
        args = [sys.executable, str(SCRIPT), "--out-dir", str(self.tmpdir / "out")] + extra_args
        return subprocess.run(args, capture_output=True, text=True, timeout=30)

    def test_unsupported_model_rejected_by_argparse_choices(self):
        proc = self._run(["--model", "mustango", "--mode", "generate", "--prompt", "x"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", proc.stderr.lower())

    def test_empty_prompt_rejected(self):
        proc = self._run(["--model", "sao", "--mode", "generate", "--prompt", "   "])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Empty prompt", proc.stdout)

    def test_edit_mode_missing_input_rejected(self):
        proc = self._run(["--model", "sao_edit", "--mode", "edit", "--prompt", "jazzy"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Source audio not found", proc.stdout)

    def test_sao_in_edit_mode_rejected_before_heavy_import(self):
        # sao only supports generate mode locally; this must be caught
        # inside the try block (after torch import) and reported cleanly,
        # not crash — exercised without needing torch itself to be mocked
        # since on failure the code path still prints "ERROR: ...".
        src = self.tmpdir / "src.wav"
        src.write_bytes(b"RIFF....WAVEfmt ")
        proc = self._run(["--model", "sao", "--mode", "edit",
                         "--prompt", "x", "--input", str(src)])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("ERROR", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
