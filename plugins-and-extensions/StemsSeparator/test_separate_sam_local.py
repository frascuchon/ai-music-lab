#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for separate_sam_local.py's argument validation and device-resolution
logic. No model weights are loaded and torch/sam_audio are never imported
here — this file only exercises pure Python (argparse wiring, _resolve_device,
model-support gating), which is importable even when sam_audio/torch are not
installed (verified: this repo's environment has neither installed)."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import separate_sam_local as ssl  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "separate_sam_local.py"


class TestResolveDevice(TestCase):

    def test_auto_prefers_mps_over_cuda(self):
        self.assertEqual(ssl._resolve_device("auto", mps_available=True, cuda_available=True), "mps")

    def test_auto_falls_back_to_cuda_without_mps(self):
        self.assertEqual(ssl._resolve_device("auto", mps_available=False, cuda_available=True), "cuda")

    def test_auto_falls_back_to_cpu_with_nothing(self):
        self.assertEqual(ssl._resolve_device("auto", mps_available=False, cuda_available=False), "cpu")

    def test_explicit_mps_without_hardware_falls_back_to_cpu(self):
        self.assertEqual(ssl._resolve_device("mps", mps_available=False, cuda_available=False), "cpu")

    def test_explicit_cuda_without_hardware_falls_back_to_cpu(self):
        self.assertEqual(ssl._resolve_device("cuda", mps_available=False, cuda_available=False), "cpu")

    def test_explicit_cpu_always_cpu(self):
        self.assertEqual(ssl._resolve_device("cpu", mps_available=True, cuda_available=True), "cpu")

    def test_explicit_mps_with_hardware_honored(self):
        self.assertEqual(ssl._resolve_device("mps", mps_available=True, cuda_available=False), "mps")


class TestLocalSupportedModelsSet(TestCase):

    def test_matches_shared_registry_true_entries_for_sam(self):
        shared_dir = Path(__file__).resolve().parent.parent / "shared"
        sys.path.insert(0, str(shared_dir))
        import model_execution as me
        expected = {"sam_large", "sam_base"}
        self.assertEqual(expected, ssl.LOCAL_SUPPORTED_MODELS)
        for key in expected:
            self.assertTrue(me.STEMSEPARATOR[key]["local_supported"])

    def test_model_hf_repo_mapping(self):
        self.assertEqual(ssl.MODEL_HF_REPO["sam_large"], "facebook/sam-audio-large")
        self.assertEqual(ssl.MODEL_HF_REPO["sam_base"], "facebook/sam-audio-base")


class TestMainValidation(TestCase):
    """Drive main() via subprocess so argparse's --choices gating and the
    fail-fast checks run exactly as a real invocation would, without ever
    reaching the torch/sam_audio import inside the try block."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ssl_test_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, extra_args):
        args = [sys.executable, str(SCRIPT), "--outdir", str(self.tmpdir / "out")] + extra_args
        return subprocess.run(args, capture_output=True, text=True, timeout=30)

    def test_unsupported_model_rejected_by_argparse_choices(self):
        proc = self._run(["--model", "sam_xlarge", "--input", "x.wav"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", proc.stderr.lower())

    def test_missing_input_file_rejected(self):
        proc = self._run(["--model", "sam_base", "--input", str(self.tmpdir / "missing.wav")])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("ERROR", proc.stdout)
        self.assertIn("does not exist", proc.stdout)

    def test_valid_input_reaches_heavy_import(self):
        # Reaches the torch import inside the try block and fails there
        # (torch/sam_audio not installed in this test environment) — proves
        # validation passed and execution got as far as intended, without
        # needing real hardware or weights.
        src = self.tmpdir / "src.wav"
        src.write_bytes(b"RIFF....WAVEfmt ")
        proc = self._run(["--model", "sam_base", "--input", str(src)])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("ERROR", proc.stdout)


class TestMissingSoundfileOnSectionExtraction(TestCase):
    """Same regression as separate_demucs.py/separate_sam.py: extract_section
    imports soundfile directly, which demucs 4.x no longer installs as a
    transitive dependency. Calls main() in-process (direct import) with
    extract_section mocked out, so it doesn't depend on whether soundfile
    actually happens to be installed in this test environment."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ssl_missing_sf_"))
        self.input_wav = self.tmpdir / "src.wav"
        self.input_wav.write_bytes(b"RIFF....WAVEfmt ")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_module_not_found_error_gives_actionable_message(self):
        argv = [
            "separate_sam_local.py",
            "--input", str(self.input_wav),
            "--model", "sam_base",
            "--start", "1.0",
            "--duration", "2.0",
            "--outdir", str(self.tmpdir / "out"),
        ]
        with patch.object(sys, "argv", argv), \
             patch.object(ssl, "extract_section",
                          side_effect=ModuleNotFoundError("No module named 'soundfile'", name="soundfile")), \
             patch("builtins.print") as mock_print:
            rc = ssl.main()
        self.assertEqual(rc, 1)
        printed = "\n".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("soundfile", printed)
        self.assertIn("Setup", printed)
        self.assertNotIn("Error extracting section: No module named", printed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
