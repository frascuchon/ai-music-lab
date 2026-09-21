#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for midigen.py's command-building and error-message selection —
covering the seed-file handling for ChatMusician's harmonize/simplify flow.

Regression context: a seed .mid that goes missing between being written by
smf_writer.lua and the command being built used to be silently dropped
(--input-file just omitted), leaving the run to proceed unseeded with no
warning at all. That silent path is now a hard FileNotFoundError, mirroring
the Anticipatory (seed_file) contract right below it in _build_cmd.
"""

import argparse
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midigen  # noqa: E402


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model="chatmusician", prompt="Simplify this melody.",
        n_outputs=2, temperature=0.2, gpu="A10G", force=False,
        seed_file="", seed="", mode="accompaniment",
        prompt_length=5, clip_length=20, top_p=0.95, melody_instrument=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildCmdSeedHandling(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="midigen_test_"))
        self.uv_bin = Path("/usr/bin/uv")
        self.shared_dir = self.tmpdir / "shared"
        self.script_path = self.tmpdir / "research_chatmusician_modal.py"
        self.out_dir = self.tmpdir / "out"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_chatmusician_existing_seed_file_is_included(self):
        seed = self.tmpdir / "seed.mid"
        seed.write_bytes(b"fake midi bytes")
        args = _make_args(seed_file=str(seed))
        cmd = midigen._build_cmd(
            self.uv_bin, self.shared_dir, self.script_path, self.out_dir,
            args, args.prompt, "",
        )
        self.assertIn("--input-file", cmd)
        self.assertEqual(cmd[cmd.index("--input-file") + 1], str(seed))

    def test_chatmusician_missing_seed_file_raises(self):
        missing = self.tmpdir / "does_not_exist.mid"
        args = _make_args(seed_file=str(missing))
        with self.assertRaises(FileNotFoundError):
            midigen._build_cmd(
                self.uv_bin, self.shared_dir, self.script_path, self.out_dir,
                args, args.prompt, "",
            )

    def test_chatmusician_no_seed_requested_omits_flag(self):
        args = _make_args(seed_file="")
        cmd = midigen._build_cmd(
            self.uv_bin, self.shared_dir, self.script_path, self.out_dir,
            args, args.prompt, "",
        )
        self.assertNotIn("--input-file", cmd)

    def test_anticipatory_missing_seed_raises(self):
        """Regression guard: the pre-existing Anticipatory contract must
        keep raising exactly like it did before this change."""
        args = _make_args(model="anticipatory", seed="")
        with self.assertRaises(FileNotFoundError):
            midigen._build_cmd(
                self.uv_bin, self.shared_dir, self.script_path, self.out_dir,
                args, args.prompt, "",
            )


class TestPickErrorMessage(TestCase):

    def test_nonzero_exit_prefers_captured_error_line(self):
        msg = midigen._pick_error_message(
            1, Path("/tmp/out"),
            last_error_line="midi2abc failed:\nno notes found",
            last_warn_line=None,
        )
        self.assertEqual(msg, "midi2abc failed:\nno notes found")

    def test_nonzero_exit_falls_back_to_generic_message(self):
        msg = midigen._pick_error_message(
            1, Path("/tmp/out"), last_error_line=None, last_warn_line=None,
        )
        self.assertIn("Modal failed (code 1)", msg)

    def test_zero_exit_no_outputs_appends_warn_line(self):
        msg = midigen._pick_error_message(
            0, Path("/tmp/out"),
            last_error_line=None,
            last_warn_line="input_file no existe: /tmp/seed.mid",
        )
        self.assertIn("No .mid files found in /tmp/out", msg)
        self.assertIn("input_file no existe: /tmp/seed.mid", msg)

    def test_zero_exit_no_outputs_prefers_error_over_warn_line(self):
        """Regression: research_chatmusician_modal.py can legitimately exit 0
        while every candidate failed internally (each logs its own captured
        "ERROR: ..." line but the overall process used to still report
        success). When both an ERROR: and a [warn] line were seen, the more
        specific ERROR: line must win."""
        msg = midigen._pick_error_message(
            0, Path("/tmp/out"),
            last_error_line="[1/1] v0 generation failed: No ABC notation found",
            last_warn_line="input_file no existe: /tmp/seed.mid",
        )
        self.assertIn("No .mid files found in /tmp/out", msg)
        self.assertIn("No ABC notation found", msg)
        self.assertNotIn("input_file no existe", msg)

    def test_zero_exit_no_outputs_no_warn_captured(self):
        msg = midigen._pick_error_message(
            0, Path("/tmp/out"), last_error_line=None, last_warn_line=None,
        )
        self.assertIn("No .mid files found in /tmp/out", msg)
        self.assertNotIn("Possible cause", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
