#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for midigen.py's command-building, error-message selection and
--execution local|remote routing (Amadeus and Anticipatory)."""

import argparse
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase, mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midigen  # noqa: E402

# Real shared/ dir (contains pyproject.toml and model_execution.py) — used
# by the --execution local tests below so midigen.py's
# `from model_execution import is_local_supported` import succeeds exactly
# as it would in production, without duplicating the registry in a fixture.
REAL_SHARED_DIR = Path(__file__).resolve().parent.parent / "shared"


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model="amadeus", prompt="A calm piano piece.",
        n_outputs=2, temperature=1.0, gpu="A10G", force=False,
        seed="", mode="accompaniment",
        prompt_length=5, clip_length=20, top_p=0.95, melody_instrument=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildCmd(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="midigen_test_"))
        self.uv_bin = Path("/usr/bin/uv")
        self.shared_dir = self.tmpdir / "shared"
        self.script_path = self.tmpdir / "research_amadeus_modal.py"
        self.out_dir = self.tmpdir / "out"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_amadeus_builds_text_dir_command(self):
        args = _make_args(model="amadeus", n_outputs=3, temperature=0.8)
        cmd = midigen._build_cmd(
            self.uv_bin, self.shared_dir, self.script_path, self.out_dir,
            args, args.prompt,
        )
        self.assertEqual(cmd[cmd.index("--prompt") + 1], args.prompt)
        self.assertEqual(cmd[cmd.index("--n-outputs") + 1], "3")
        self.assertEqual(cmd[cmd.index("--temperature") + 1], "0.8")
        self.assertIn("--force", cmd)
        self.assertNotIn("--gpu", cmd)

    def test_anticipatory_missing_seed_raises(self):
        """Anticipatory needs an existing seed MIDI; a missing one must be a
        hard error rather than a silently unseeded run."""
        args = _make_args(model="anticipatory", seed="")
        with self.assertRaises(FileNotFoundError):
            midigen._build_cmd(
                self.uv_bin, self.shared_dir, self.script_path, self.out_dir,
                args, args.prompt,
            )


class TestPickErrorMessage(TestCase):

    def test_nonzero_exit_prefers_captured_error_line(self):
        msg = midigen._pick_error_message(
            1, Path("/tmp/out"),
            last_error_line="seed conversion failed:\nno notes found",
            last_warn_line=None,
        )
        self.assertEqual(msg, "seed conversion failed:\nno notes found")

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
        """Regression: a research_*_modal.py script can legitimately exit 0
        while every candidate failed internally (each logs its own captured
        "ERROR: ..." line but the overall process used to still report
        success). When both an ERROR: and a [warn] line were seen, the more
        specific ERROR: line must win."""
        msg = midigen._pick_error_message(
            0, Path("/tmp/out"),
            last_error_line="[1/1] v0 generation failed: no notes generated",
            last_warn_line="input_file no existe: /tmp/seed.mid",
        )
        self.assertIn("No .mid files found in /tmp/out", msg)
        self.assertIn("no notes generated", msg)
        self.assertNotIn("input_file no existe", msg)

    def test_zero_exit_no_outputs_no_warn_captured(self):
        msg = midigen._pick_error_message(
            0, Path("/tmp/out"), last_error_line=None, last_warn_line=None,
        )
        self.assertIn("No .mid files found in /tmp/out", msg)
        self.assertNotIn("Possible cause", msg)


class TestBuildLocalCmd(TestCase):
    """--execution local command building (anticipatory only)."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="midigen_local_test_"))
        self.uv_bin = Path("/usr/bin/uv")
        self.out_dir = self.tmpdir / "out"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_anticipatory_missing_seed_raises(self):
        script = self.tmpdir / "research" / "research_anticipatory.py"
        args = _make_args(model="anticipatory", seed="", mode="accompaniment")
        with self.assertRaises(FileNotFoundError):
            midigen._build_local_cmd(self.uv_bin, script, self.out_dir, args, "")

    def test_anticipatory_accompaniment_builds_command_without_duration(self):
        script = self.tmpdir / "research" / "research_anticipatory.py"
        seed = self.tmpdir / "seed.mid"
        seed.write_bytes(b"fake midi bytes")
        args = _make_args(
            model="anticipatory", seed=str(seed), mode="accompaniment",
            prompt_length=5, clip_length=20, top_p=0.95,
            melody_instrument=0, n_outputs=3,
        )
        cmd = midigen._build_local_cmd(self.uv_bin, script, self.out_dir, args, "")

        self.assertIn(str(script), cmd)
        self.assertIn("--input", cmd)
        self.assertEqual(cmd[cmd.index("--input") + 1], str(seed))
        self.assertIn("--mode", cmd)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "accompaniment")
        self.assertIn("--multiplicity", cmd)
        self.assertEqual(cmd[cmd.index("--multiplicity") + 1], "3")
        self.assertIn("--clip-length", cmd)
        self.assertNotIn("--duration", cmd)
        self.assertNotIn("--gpu", cmd)

    def test_anticipatory_continuation_includes_duration(self):
        script = self.tmpdir / "research" / "research_anticipatory.py"
        seed = self.tmpdir / "seed.mid"
        seed.write_bytes(b"fake midi bytes")
        args = _make_args(
            model="anticipatory", seed=str(seed), mode="continuation",
            clip_length=15,
        )
        cmd = midigen._build_local_cmd(self.uv_bin, script, self.out_dir, args, "")

        self.assertIn("--duration", cmd)
        self.assertEqual(cmd[cmd.index("--duration") + 1], "15")

    def test_unsupported_model_raises_value_error(self):
        script = self.tmpdir / "research" / "research_amadeus.py"
        args = _make_args(model="amadeus")
        with self.assertRaises(ValueError):
            midigen._build_local_cmd(self.uv_bin, script, self.out_dir, args, "prompt")


class TestExecutionDispatch(TestCase):
    """main()'s --execution local|remote routing, exercised end-to-end
    through argument parsing (not just the _build_cmd/_build_local_cmd
    helpers in isolation)."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="midigen_dispatch_test_"))
        self.out_dir = self.tmpdir / "out"
        self.progress = self.tmpdir / "progress.txt"
        self.script = self.tmpdir / "research_amadeus_modal.py"
        self.script.write_text("# fake")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _argv(self, model, extra=None):
        argv = [
            "midigen.py",
            "--shared-dir", str(REAL_SHARED_DIR),
            "--script", str(self.script),
            "--model", model,
            "--out-dir", str(self.out_dir),
            "--progress", str(self.progress),
            "--prompt", "a test prompt",
        ]
        return argv + (extra or [])

    def test_local_execution_rejected_for_unsupported_model(self):
        """amadeus has local_supported=False in shared/model_execution.py —
        --execution local must fail fast with the registry's reason, and
        must never reach subprocess.Popen (Modal/local launch)."""
        with mock.patch.object(sys, "argv", self._argv("amadeus", ["--execution", "local"])), \
             mock.patch("subprocess.Popen", side_effect=AssertionError("must not launch a subprocess")):
            rc = midigen.main()

        self.assertEqual(rc, 1)
        progress_text = self.progress.read_text()
        self.assertIn("error", progress_text)
        self.assertIn("Local execution not supported for amadeus", progress_text)

    def test_omitting_execution_defaults_to_remote_build_cmd(self):
        """Regression guard: with --execution omitted, main() must still
        route through _build_cmd (the pre-existing Modal path) rather than
        _build_local_cmd — i.e. this change is purely additive."""
        with mock.patch.object(sys, "argv", self._argv("amadeus")), \
             mock.patch.object(midigen, "_find_uv", return_value=Path("/usr/bin/uv")), \
             mock.patch.object(midigen, "_build_cmd", side_effect=ValueError("stop here")) as m_remote, \
             mock.patch.object(midigen, "_build_local_cmd") as m_local:
            rc = midigen.main()

        self.assertEqual(rc, 1)  # ValueError from the mock short-circuits main()
        m_remote.assert_called_once()
        m_local.assert_not_called()

    def test_explicit_remote_execution_routes_to_build_cmd(self):
        with mock.patch.object(sys, "argv", self._argv("amadeus", ["--execution", "remote"])), \
             mock.patch.object(midigen, "_find_uv", return_value=Path("/usr/bin/uv")), \
             mock.patch.object(midigen, "_build_cmd", side_effect=ValueError("stop here")) as m_remote, \
             mock.patch.object(midigen, "_build_local_cmd") as m_local:
            rc = midigen.main()

        self.assertEqual(rc, 1)
        m_remote.assert_called_once()
        m_local.assert_not_called()

    def test_local_execution_for_supported_model_routes_to_build_local_cmd(self):
        seed = self.tmpdir / "seed.mid"
        seed.write_bytes(b"fake midi bytes")
        with mock.patch.object(sys, "argv", self._argv(
                 "anticipatory", ["--execution", "local", "--seed", str(seed)])), \
             mock.patch.object(midigen, "_find_uv", return_value=Path("/usr/bin/uv")), \
             mock.patch.object(midigen, "_build_cmd") as m_remote, \
             mock.patch.object(midigen, "_build_local_cmd", side_effect=ValueError("stop here")) as m_local:
            rc = midigen.main()

        self.assertEqual(rc, 1)
        m_local.assert_called_once()
        m_remote.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
