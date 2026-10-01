#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for research_magnet_modal.py's model device placement.

Same root cause/fix as research_audiogen_modal.py: MAGNeT (audiocraft's
BaseGenModel) has no .to() method. See test_research_audiogen_modal.py for
the full explanation — `model.to("cuda")` used to crash with:

    AttributeError: 'MAGNeT' object has no attribute 'to'

Device placement must happen via get_pretrained(variant, device="cuda").
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_magnet_modal as rm  # noqa: E402


def _zeros():
    import numpy as np
    return np.zeros((2, 10))


def _mock_modules(fake_model):
    fake_audiocraft_models = types.ModuleType("audiocraft.models")
    fake_audiocraft_models.MAGNeT = MagicMock()
    fake_audiocraft_models.MAGNeT.get_pretrained.return_value = fake_model
    fake_audiocraft = types.ModuleType("audiocraft")
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = MagicMock()
    fake_sf = types.ModuleType("soundfile")
    fake_sf.write = MagicMock()
    return {
        "audiocraft": fake_audiocraft,
        "audiocraft.models": fake_audiocraft_models,
        "torch": fake_torch,
        "soundfile": fake_sf,
    }, fake_audiocraft_models


class TestGenerateBatchDevicePlacement(TestCase):

    def _run(self, jobs, spec=None):
        fake_model = MagicMock(spec=spec) if spec else MagicMock()
        fake_model.sample_rate = 32000
        fake_wav = MagicMock()
        fake_wav.cpu.return_value = MagicMock(numpy=lambda: _zeros())
        fake_model.generate.return_value = [fake_wav]

        modules, fake_audiocraft_models = _mock_modules(fake_model)
        with patch.dict(sys.modules, modules):
            rm.generate_batch.local(jobs)
        return fake_audiocraft_models.MAGNeT.get_pretrained

    def test_get_pretrained_called_with_cuda_device_not_to(self):
        get_pretrained = self._run([{"text": "happy rock", "seconds": 8.0}])
        get_pretrained.assert_called_once()
        self.assertEqual(get_pretrained.call_args.kwargs.get("device"), "cuda")

    def test_model_to_is_never_called(self):
        """Regression guard: model.to(...) must not be called."""
        self._run(
            [{"text": "lo-fi hip hop", "seconds": 10.0}],
            spec=["sample_rate", "generate", "set_generation_params"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
