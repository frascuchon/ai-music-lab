#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for research_inspiremusic_gen_modal.py's per-job duration handling.

Context: "InspireMusic 1.5 no respeta el limite de duración definido (hace
más segundos)". Root cause: InspireMusicModel.inference() always generates
model.max_generate_audio_seconds of audio — a value fixed once at model
construction (from the env var INSPIREMUSIC_MAX_SECONDS, default 30.0) —
and truncates to model.max_generate_audio_length at the end. It does NOT
derive actual generated length from time_end (verified against
FunAudioLLM/InspireMusic's inspiremusic/cli/inference.py: model_input's
"duration_to_gen" is literally self.max_generate_audio_seconds). The old
code never looked at job["seconds"] at all, so every job silently
generated the fixed default regardless of what the UI asked for.

generate_batch now updates model.max_generate_audio_seconds /
max_generate_audio_length per job before calling inference(), clamped to
[model.min_generate_audio_seconds, MAX_GENERATE_SECONDS].
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_inspiremusic_gen_modal as rm  # noqa: E402


def _mock_model(output_sample_rate=48000, min_seconds=0.0, max_seconds=30.0):
    model = MagicMock()
    model.output_sample_rate = output_sample_rate
    model.min_generate_audio_seconds = min_seconds
    model.max_generate_audio_seconds = max_seconds
    model.max_generate_audio_length = int(output_sample_rate * max_seconds)

    def _inference(**kwargs):
        # Simulate InspireMusic writing output_{i}.wav into result_dir —
        # generate_batch locates it via Path(result_dir)/f"{output_fn}.wav".
        out_dir = Path(model._result_dir)
        (out_dir / f"{kwargs['output_fn']}.wav").write_bytes(b"\x00" * 100)

    model.inference.side_effect = _inference
    return model


def _run_jobs(jobs, model):
    model._result_dir = tempfile.mkdtemp(prefix="inspiremusic_test_")

    fake_inspiremusic_cli = types.ModuleType("inspiremusic.cli.inference")
    fake_inspiremusic_cli.InspireMusicModel = MagicMock(return_value=model)
    fake_inspiremusic_cli.env_variables = MagicMock()
    fake_inspiremusic_cli_pkg = types.ModuleType("inspiremusic.cli")
    fake_inspiremusic_pkg = types.ModuleType("inspiremusic")
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = MagicMock()
    fake_torchaudio = types.ModuleType("torchaudio")
    fake_torchaudio.set_audio_backend = MagicMock()

    # Path.glob is patched globally to an empty iterator so the yaml-path
    # patching loop (model_dir_path.glob("*.yaml")) is a no-op against a
    # Volume path that doesn't exist locally; _collect candidates in
    # generate_batch falls back to result_dir directly since the output
    # file is written at the exact expected path (output_fn.wav exists).
    with patch.object(Path, "glob", return_value=iter([])), \
         patch.dict(sys.modules, {
            "inspiremusic": fake_inspiremusic_pkg,
            "inspiremusic.cli": fake_inspiremusic_cli_pkg,
            "inspiremusic.cli.inference": fake_inspiremusic_cli,
            "torch": fake_torch,
            "torchaudio": fake_torchaudio,
         }), \
         patch.object(tempfile, "mkdtemp", return_value=model._result_dir), \
         patch.object(rm.weights_vol, "commit"):
        return rm.generate_batch.local(jobs)


class TestPerJobDuration(TestCase):

    def test_seconds_from_job_sets_model_duration_before_inference(self):
        model = _mock_model(max_seconds=30.0)
        _run_jobs([{"text": "ambient piano", "seconds": 8.0}], model)

        self.assertEqual(model.max_generate_audio_seconds, 8.0)
        self.assertEqual(model.max_generate_audio_length, int(48000 * 8.0))
        call_kwargs = model.inference.call_args.kwargs
        self.assertEqual(call_kwargs["time_end"], 8.0)

    def test_each_job_gets_its_own_duration_not_the_previous_jobs(self):
        model = _mock_model(max_seconds=30.0)
        _run_jobs(
            [{"text": "short clip", "seconds": 5.0},
             {"text": "long clip", "seconds": 25.0}],
            model,
        )
        calls = model.inference.call_args_list
        self.assertEqual(calls[0].kwargs["time_end"], 5.0)
        self.assertEqual(calls[1].kwargs["time_end"], 25.0)
        # Final state reflects the last job processed.
        self.assertEqual(model.max_generate_audio_seconds, 25.0)

    def test_missing_seconds_falls_back_to_max_generate_seconds(self):
        model = _mock_model(max_seconds=30.0)
        _run_jobs([{"text": "no duration specified"}], model)
        self.assertEqual(model.max_generate_audio_seconds, rm.MAX_GENERATE_SECONDS)

    def test_requested_seconds_above_cap_is_clamped(self):
        model = _mock_model(max_seconds=30.0)
        _run_jobs([{"text": "too long", "seconds": 999.0}], model)
        self.assertEqual(model.max_generate_audio_seconds, rm.MAX_GENERATE_SECONDS)

    def test_requested_seconds_below_model_minimum_is_clamped(self):
        model = _mock_model(min_seconds=10.0, max_seconds=30.0)
        _run_jobs([{"text": "too short", "seconds": 1.0}], model)
        self.assertEqual(model.max_generate_audio_seconds, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
