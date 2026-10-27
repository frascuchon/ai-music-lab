#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for setup_helpers.py: demucs check/install, subprocess streaming
and error reporting, and the main() crash guard (Setup tab)."""

import argparse
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_helpers as sh  # noqa: E402


class TestChkDemucs(TestCase):
    """demucs 4.x dropped its own `soundfile` dependency (switched to
    `sphn`), but separate_demucs.py/separate_sam.py still import soundfile
    directly for section extraction. A user could have demucs importable
    while soundfile is missing — reproducing "No module named 'soundfile'"
    at --start/--duration time. _chk_demucs must catch that case too, not
    just whether `import demucs` succeeds."""

    def test_both_importable_reports_ok(self):
        with patch.object(sh, "_run", side_effect=[(0, "4.0.1", ""), (0, "", "")]):
            status, detail = sh._chk_demucs()
        self.assertEqual(status, "ok")
        self.assertIn("demucs", detail)

    def test_demucs_missing_reports_missing(self):
        with patch.object(sh, "_run", return_value=(1, "", "ModuleNotFoundError: No module named 'demucs'")):
            status, detail = sh._chk_demucs()
        self.assertEqual(status, "missing")
        self.assertIn("demucs", detail)

    def test_soundfile_missing_reports_missing_even_though_demucs_ok(self):
        with patch.object(sh, "_run", side_effect=[
            (0, "4.0.1", ""),
            (1, "", "ModuleNotFoundError: No module named 'soundfile'"),
        ]):
            status, detail = sh._chk_demucs()
        self.assertEqual(status, "missing")
        self.assertIn("soundfile", detail)


class TestCmdInstallDemucsIncludesSoundfile(TestCase):
    """install-demucs must pull in soundfile explicitly since demucs 4.x no
    longer depends on it — otherwise a fresh install still leaves section
    extraction broken."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="install_demucs_test_"))
        self.pf = self.tmpdir / "progress.txt"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pip_install_command_includes_soundfile(self):
        args = argparse.Namespace(python="/usr/bin/python3", progress=str(self.pf))
        with patch.object(sh, "_stream") as mock_stream:
            sh.cmd_install_demucs(args)
        mock_stream.assert_called_once()
        cmd = mock_stream.call_args.args[1]
        self.assertEqual(
            cmd,
            ["/usr/bin/python3", "-m", "pip", "install", "--user",
             "demucs", "torch", "torchaudio", "soundfile"],
        )

    def test_defaults_to_sys_executable_when_no_python_given(self):
        args = argparse.Namespace(python=None, progress=str(self.pf))
        with patch.object(sh, "_stream") as mock_stream:
            sh.cmd_install_demucs(args)
        cmd = mock_stream.call_args.args[1]
        self.assertEqual(cmd[0], sys.executable)
        self.assertIn("soundfile", cmd)


class TestTailErrorMessage(TestCase):
    """_tail_error_message builds the descriptive text _stream() now writes
    on subprocess failure, replacing the old bare 'Error (code N)'."""

    def test_includes_command_name_and_code(self):
        msg = sh._tail_error_message(["/usr/bin/brew", "install", "demucs"], 1, [])
        self.assertIn("brew", msg)
        self.assertIn("code 1", msg)

    def test_includes_last_non_empty_output_line(self):
        tail = ["Downloading...", "", "Error: No available formula named foo"]
        msg = sh._tail_error_message(["brew"], 1, tail)
        self.assertIn("Error: No available formula named foo", msg)

    def test_skips_trailing_blank_lines_to_find_last_real_line(self):
        tail = ["real error line", "", "   "]
        msg = sh._tail_error_message(["cmd"], 1, tail)
        self.assertIn("real error line", msg)

    def test_no_output_still_actionable(self):
        msg = sh._tail_error_message(["cmd"], 1, [])
        self.assertIn("code 1", msg)
        self.assertIn("log", msg.lower())

    def test_never_produces_bare_exit_code_only_message(self):
        # Regression guard for the exact complaint: a message with nothing
        # but "(Exit code N)"/"Error (code N)" and no other context.
        for tail in ([], ["some real diagnostic output"]):
            msg = sh._tail_error_message(["cmd"], 1, tail)
            self.assertNotRegex(msg.strip(), r"^Error \(code \d+\)$")


class TestStream(TestCase):
    """_stream() drives a subprocess and writes the progress protocol.
    These tests exercise a REAL subprocess (no mocking of subprocess.Popen)
    to make sure the progress-file protocol (state|pct|msg\\n + extra lines)
    is preserved exactly, since panel_setup.lua's poll_simple parses it with
    a strict pattern."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="stream_test_"))
        self.pf = self.tmpdir / "progress.txt"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read(self) -> str:
        return self.pf.read_text() if self.pf.exists() else ""

    def test_success_writes_done_state_with_done_msg(self):
        rc = sh._stream(self.pf, [sys.executable, "-c", "print('hello')"],
                        done_msg="All good")
        self.assertEqual(rc, 0)
        content = self._read()
        self.assertTrue(content.startswith("done|1.000|All good"))

    def test_failure_writes_error_state_with_last_output_line(self):
        script = ("import sys; print('step 1'); print('boom: disk full'); "
                  "sys.exit(1)")
        rc = sh._stream(self.pf, [sys.executable, "-c", script])
        self.assertEqual(rc, 1)
        content = self._read()
        self.assertTrue(content.startswith("error|"))
        self.assertIn("code 1", content)
        self.assertIn("boom: disk full", content)
        # Old behavior produced only "Error (code 1)" with nothing else —
        # assert that's no longer the whole message.
        first_line = content.splitlines()[0]
        msg = first_line.split("|", 2)[2]
        self.assertNotEqual(msg.strip(), "Error (code 1)")

    def test_failure_includes_extra_tail_lines_for_full_log_view(self):
        script = ("import sys; [print(f'line {i}') for i in range(5)]; "
                  "sys.exit(2)")
        sh._stream(self.pf, [sys.executable, "-c", script])
        content = self._read()
        lines = content.splitlines()
        self.assertIn("--- last output lines ---", lines)
        self.assertIn("line 4", lines)

    def test_failure_with_no_output_still_writes_parseable_error_line(self):
        script = "import sys; sys.exit(3)"
        rc = sh._stream(self.pf, [sys.executable, "-c", script])
        self.assertEqual(rc, 3)
        content = self._read()
        first_line = content.splitlines()[0]
        state, pct, msg = first_line.split("|", 2)
        self.assertEqual(state, "error")
        self.assertIn("code 3", msg)

    def test_progress_protocol_still_parseable_by_read_progress_file(self):
        """Guards against the reformatting breaking common.read_progress_file
        (the Lua-side parser mirrors this pattern: state|pct|msg)."""
        import re
        script = "import sys; print('doing work'); sys.exit(1)"
        sh._stream(self.pf, [sys.executable, "-c", script])
        first_line = self._read().splitlines()[0]
        self.assertRegex(first_line, r"^[^|]+\|[^|]+\|.+$")
        state, pct_str, _ = first_line.split("|", 2)
        self.assertEqual(state, "error")
        float(pct_str)  # must parse as a float, like Lua's tonumber(r.pct)

    def test_launch_failure_for_nonexistent_binary_is_reported_not_raised(self):
        rc = sh._stream(self.pf, ["/no/such/binary/xyz123"])
        self.assertEqual(rc, 1)
        content = self._read()
        self.assertTrue(content.startswith("error|"))
        self.assertIn("Failed to launch", content)


class TestMainCrashGuard(TestCase):
    """Regression test for a real "Setup tab hangs forever" report: an
    uncaught exception in a dispatched subcommand (here, a resolved Python
    path that doesn't actually exist) used to crash the process without
    ever writing an 'error' line, leaving the progress file stuck on its
    initial 'running|0.00|...' forever — panel_setup.lua's poll_*()
    functions have no timeout, so they wait indefinitely. main() must now
    write an error line before re-raising."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="main_crash_test_"))
        self.pf = self.tmpdir / "progress.txt"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_uncaught_exception_still_writes_error_line(self):
        argv = ["setup_helpers.py", "check", "--progress", str(self.pf)]
        with patch.object(sys, "argv", argv), \
             patch.object(sh, "cmd_check", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                sh.main()
        content = self.pf.read_text()
        self.assertTrue(content.startswith("error|"))
        self.assertIn("boom", content)

    def test_exception_is_still_propagated_not_swallowed(self):
        argv = ["setup_helpers.py", "install-uv", "--progress", str(self.pf)]
        with patch.object(sys, "argv", argv), \
             patch.object(sh, "cmd_install_uv", side_effect=ValueError("x")):
            with self.assertRaises(ValueError):
                sh.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
