#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for research_acestep_gen_modal.py's inference-parameter clamping
and LoRA/LoKr adapter folder handling.

Context: the plugin now lets the user tune ACE-Step 1.5's inference
parameters (steps, guidance_scale, shift, CFG interval, LM temperature) and
load a LoRA/LoKr adapter from a local folder. The ranges mirror the official
ACE-Step Gradio UI for the turbo checkpoint this plugin ships (see
model_config.py::get_ui_control_config in the ace-step research repo). The
Modal container has no access to the user's local filesystem, so a LoRA
folder must be read into memory locally and shipped as part of the
.remote() call payload (_read_lora_dir / _materialize_lora_dir), matching
the existing pattern research_acestep_edit_modal.py already uses for a
single --source-audio file.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_acestep_gen_modal as rm  # noqa: E402


class TestClamp(TestCase):

    def test_within_range_unchanged(self):
        self.assertEqual(rm._clamp(5, 1, 20, "x"), 5)

    def test_below_range_clamped_to_min(self):
        self.assertEqual(rm._clamp(-3, 1, 20, "x"), 1)

    def test_above_range_clamped_to_max(self):
        self.assertEqual(rm._clamp(999, 1, 20, "x"), 20)

    def test_boundary_values_unchanged(self):
        self.assertEqual(rm._clamp(1, 1, 20, "x"), 1)
        self.assertEqual(rm._clamp(20, 1, 20, "x"), 20)


class TestResolveDitConfig(TestCase):
    """CFG (guidance_scale/use_adg/cfg_interval_*) only matters for the base
    checkpoint — turbo is CFG-distilled and generate_music() hard-overrides
    guidance_scale to 1.0 for it regardless of what's passed (verified
    empirically against a real Modal run). The Lua UI decides which controls
    to show based on this same turbo/base split, so the mapping here must
    stay exactly in sync with DIT_CONFIG_BY_VARIANT."""

    def test_turbo_maps_to_turbo_checkpoint(self):
        self.assertEqual(rm._resolve_dit_config("turbo"), "acestep-v15-turbo")

    def test_base_maps_to_base_checkpoint(self):
        self.assertEqual(rm._resolve_dit_config("base"), "acestep-v15-base")

    def test_case_insensitive(self):
        self.assertEqual(rm._resolve_dit_config("Base"), "acestep-v15-base")
        self.assertEqual(rm._resolve_dit_config("TURBO"), "acestep-v15-turbo")

    def test_unknown_variant_falls_back_to_turbo(self):
        self.assertEqual(rm._resolve_dit_config("sft"), "acestep-v15-turbo")
        self.assertEqual(rm._resolve_dit_config(""), "acestep-v15-turbo")
        self.assertEqual(rm._resolve_dit_config(None), "acestep-v15-turbo")


class TestInitHandlersPassesResolvedConfig(TestCase):
    """_init_handlers does `from acestep.handler import AceStepHandler`
    lazily inside the function body (the real package only exists inside
    the Modal container image, not in this local test env) — inject fake
    modules into sys.modules rather than patch.object'ing a real import
    path, so the import resolves without acestep actually being installed."""

    def test_initialize_service_receives_resolved_config_path(self):
        import types

        mock_dit_handler = MagicMock()
        fake_acestep = types.ModuleType("acestep")
        fake_handler_mod = types.ModuleType("acestep.handler")
        fake_handler_mod.AceStepHandler = MagicMock(return_value=mock_dit_handler)
        fake_llm_mod = types.ModuleType("acestep.llm_inference")
        fake_llm_mod.LLMHandler = MagicMock()

        with patch.dict(sys.modules, {
            "acestep": fake_acestep,
            "acestep.handler": fake_handler_mod,
            "acestep.llm_inference": fake_llm_mod,
        }):
            rm._init_handlers(dit_variant="base")

        mock_dit_handler.initialize_service.assert_called_once()
        self.assertEqual(
            mock_dit_handler.initialize_service.call_args.kwargs["config_path"],
            "acestep-v15-base",
        )


class TestReadLoraDir(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="lora_test_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_adapter(self, extra_files: dict[str, bytes] | None = None) -> Path:
        adapter = self.tmpdir / "my_adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text('{"peft_type": "LORA"}')
        (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 128)
        for rel, data in (extra_files or {}).items():
            p = adapter / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return adapter

    def test_reads_all_files_with_relative_paths(self):
        adapter = self._make_adapter()
        files = rm._read_lora_dir(str(adapter))
        self.assertEqual(set(files), {"adapter_config.json", "adapter_model.safetensors"})
        self.assertEqual(files["adapter_model.safetensors"], b"\x00" * 128)

    def test_skips_git_and_ds_store_and_pycache(self):
        adapter = self._make_adapter({
            ".git/HEAD": b"ref: refs/heads/main",
            ".DS_Store": b"junk",
            "__pycache__/x.pyc": b"junk",
        })
        files = rm._read_lora_dir(str(adapter))
        self.assertEqual(set(files), {"adapter_config.json", "adapter_model.safetensors"})

    def test_missing_folder_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            rm._read_lora_dir(str(self.tmpdir / "does_not_exist"))

    def test_empty_folder_raises_file_not_found(self):
        empty = self.tmpdir / "empty_adapter"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            rm._read_lora_dir(str(empty))

    def test_oversized_folder_raises_value_error(self):
        adapter = self._make_adapter()
        with patch.object(rm, "_LORA_MAX_TOTAL_BYTES", 64):
            with self.assertRaises(ValueError):
                rm._read_lora_dir(str(adapter))


class TestMaterializeLoraDir(TestCase):

    def test_none_returns_empty_string(self):
        self.assertEqual(rm._materialize_lora_dir(None), "")

    def test_empty_dict_returns_empty_string(self):
        self.assertEqual(rm._materialize_lora_dir({}), "")

    def test_writes_files_preserving_relative_structure(self):
        lora_dir = rm._materialize_lora_dir({
            "adapter_config.json": b'{"peft_type": "LORA"}',
            "nested/extra.bin": b"\x01\x02\x03",
        })
        try:
            self.assertTrue(Path(lora_dir, "adapter_config.json").is_file())
            self.assertEqual(Path(lora_dir, "adapter_config.json").read_bytes(),
                              b'{"peft_type": "LORA"}')
            self.assertEqual(Path(lora_dir, "nested", "extra.bin").read_bytes(),
                              b"\x01\x02\x03")
        finally:
            import shutil
            shutil.rmtree(lora_dir, ignore_errors=True)


class TestLoadLora(TestCase):
    """AceStepHandler's LoRA methods report failure as a returned "❌ ..."
    string rather than raising — _load_lora must turn that into a
    RuntimeError so the run fails loudly instead of silently generating
    without the adapter the user explicitly asked for."""

    def _handler(self, load_msg="✅ LoRA 'x' loaded", scale_msg="✅ LoRA scale set to 0.80",
                 enable_msg="✅ LoRA enabled"):
        h = MagicMock()
        h.load_lora.return_value = load_msg
        h.set_lora_scale.return_value = scale_msg
        h.set_use_lora.return_value = enable_msg
        return h

    def test_success_calls_all_three_steps_in_order(self):
        h = self._handler()
        rm._load_lora(h, "/some/path", 0.8)
        h.load_lora.assert_called_once_with("/some/path")
        h.set_lora_scale.assert_called_once_with(0.8)
        h.set_use_lora.assert_called_once_with(True)

    def test_load_failure_raises_runtime_error_with_handler_message(self):
        h = self._handler(load_msg="❌ LoRA path not found: /bad/path")
        with self.assertRaises(RuntimeError) as ctx:
            rm._load_lora(h, "/bad/path", 1.0)
        self.assertIn("LoRA path not found", str(ctx.exception))
        h.set_lora_scale.assert_not_called()
        h.set_use_lora.assert_not_called()

    def test_scale_failure_raises_runtime_error(self):
        h = self._handler(scale_msg="❌ Invalid LoRA scale: please provide a numeric value between 0 and 1.")
        with self.assertRaises(RuntimeError):
            rm._load_lora(h, "/some/path", 0.8)
        h.set_use_lora.assert_not_called()

    def test_enable_failure_raises_runtime_error(self):
        h = self._handler(enable_msg="❌ No adapter specified and no active adapter.")
        with self.assertRaises(RuntimeError):
            rm._load_lora(h, "/some/path", 0.8)

    def test_scale_is_clamped_before_being_passed_to_handler(self):
        h = self._handler()
        rm._load_lora(h, "/some/path", 1.7)  # out of [0,1] — LORA_SCALE_RANGE
        h.set_lora_scale.assert_called_once_with(1.0)


class TestGenerateBatchClampsParams(TestCase):
    """generate_batch is the last stop before the model actually receives
    these values — every param must be clamped there too, not just in the
    Lua UI, since the script can also be invoked directly."""

    def _run_with(self, **overrides):
        captured = {}

        def fake_generate_one(dit_handler, llm_handler, params_kwargs):
            captured.update(params_kwargs)
            return b"fake wav bytes"

        with patch.object(rm, "_init_handlers", return_value=(MagicMock(), MagicMock())), \
             patch.object(rm, "_generate_one", side_effect=fake_generate_one), \
             patch.object(rm.weights_vol, "commit"):
            kwargs = dict(
                inference_steps=8, guidance_scale=7.0, shift=3.0,
                cfg_interval_start=0.0, cfg_interval_end=1.0,
                thinking=False, lm_temperature=0.85,
            )
            kwargs.update(overrides)
            rm.generate_batch.local([{"text": "test prompt", "seconds": 10.0}], **kwargs)
        return captured

    def test_steps_out_of_range_clamped(self):
        captured = self._run_with(inference_steps=500)
        self.assertEqual(captured["inference_steps"], rm.STEPS_RANGE_BY_VARIANT["turbo"][1])

    def test_steps_range_depends_on_dit_variant(self):
        captured = self._run_with(dit_variant="base", inference_steps=150)
        self.assertEqual(captured["inference_steps"], 150)  # within base's (1, 200)
        captured = self._run_with(dit_variant="turbo", inference_steps=150)
        self.assertEqual(captured["inference_steps"], rm.STEPS_RANGE_BY_VARIANT["turbo"][1])  # clamped to 20

    def test_guidance_scale_out_of_range_clamped(self):
        captured = self._run_with(guidance_scale=0.1)
        self.assertEqual(captured["guidance_scale"], rm.GUIDANCE_SCALE_RANGE[0])

    def test_shift_out_of_range_clamped(self):
        captured = self._run_with(shift=100.0)
        self.assertEqual(captured["shift"], rm.SHIFT_RANGE[1])

    def test_cfg_interval_swapped_when_start_after_end(self):
        captured = self._run_with(cfg_interval_start=0.9, cfg_interval_end=0.1)
        self.assertLessEqual(captured["cfg_interval_start"], captured["cfg_interval_end"])
        self.assertAlmostEqual(captured["cfg_interval_start"], 0.1)
        self.assertAlmostEqual(captured["cfg_interval_end"], 0.9)

    def test_lm_temperature_out_of_range_clamped(self):
        captured = self._run_with(lm_temperature=9.0)
        self.assertEqual(captured["lm_temperature"], rm.LM_TEMPERATURE_RANGE[1])

    def test_values_within_range_pass_through_unchanged(self):
        captured = self._run_with(inference_steps=12, guidance_scale=8.5, shift=2.5)
        self.assertEqual(captured["inference_steps"], 12)
        self.assertEqual(captured["guidance_scale"], 8.5)
        self.assertEqual(captured["shift"], 2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
