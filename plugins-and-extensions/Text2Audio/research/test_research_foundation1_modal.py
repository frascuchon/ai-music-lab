#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for research_foundation1_modal.py's lazy weight download.

Context: the panel never calls `::setup` — it only invokes `::main` /
generate_batch directly (see GEN_SCRIPTS in Text2Audio/panel.lua). Unlike
audiogen/magnet/mustango, which lazily download weights on first use inside
get_pretrained()/Mustango(), Foundation-1's _load_pipeline() used to call
StableAudioPipeline.from_pretrained(..., local_files_only=True) against a
Volume that was never populated, crashing with:

    ValueError: The provided pretrained_model_name_or_path
    "/vol/weights/foundation1" is neither a valid local path nor a valid
    repo id.

_ensure_weights() now downloads on demand (same snapshot_download() setup()
used to do) and both setup() and _load_pipeline() call it, so a fresh
Volume self-heals on first real generation instead of requiring a manual
`modal run ...::setup` the plugin never performs.
"""

import sys
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_foundation1_modal as rm  # noqa: E402


class TestEnsureWeights(TestCase):

    def test_downloads_when_config_missing(self):
        with patch("os.path.exists", return_value=False), \
             patch("huggingface_hub.snapshot_download") as mock_dl, \
             patch("os.walk", return_value=[]), \
             patch.object(rm.weights_vol, "commit"):
            path = rm._ensure_weights()

        mock_dl.assert_called_once()
        self.assertEqual(mock_dl.call_args.kwargs["repo_id"], rm.MODEL_HF_REPO)
        self.assertEqual(path, f"{rm.WEIGHTS_MOUNT}/foundation1")

    def test_skips_download_when_already_present(self):
        with patch("os.path.exists", return_value=True), \
             patch("huggingface_hub.snapshot_download") as mock_dl:
            path = rm._ensure_weights()

        mock_dl.assert_not_called()
        self.assertEqual(path, f"{rm.WEIGHTS_MOUNT}/foundation1")


class TestLoadPipelineUsesEnsureWeights(TestCase):
    """_load_pipeline must never hand from_pretrained a path that hasn't
    been confirmed/downloaded via _ensure_weights() — that indirection is
    exactly what fixes the ValueError from a never-setup Volume."""

    def test_load_pipeline_calls_ensure_weights_before_from_pretrained(self):
        fake_pipe = MagicMock()
        fake_pipe.to.return_value = fake_pipe

        import types
        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.StableAudioPipeline = MagicMock()
        fake_diffusers.StableAudioPipeline.from_pretrained.return_value = fake_pipe
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"

        with patch.object(rm, "_ensure_weights", return_value="/vol/weights/foundation1") as mock_ensure, \
             patch.dict(sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}):
            rm._load_pipeline()

        mock_ensure.assert_called_once()
        fake_diffusers.StableAudioPipeline.from_pretrained.assert_called_once()
        self.assertEqual(
            fake_diffusers.StableAudioPipeline.from_pretrained.call_args.args[0],
            "/vol/weights/foundation1",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
