#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for setup_helpers.py's abcmidi check/install support (Setup tab).

Context: MidiGenerator's ChatMusician "harmonize/simplify with seed" flow
needs midi2abc/abc2midi (the abcmidi package) installed LOCALLY — the
MIDI<->ABC conversion runs on this machine before the Modal call, not inside
the Modal container. A real user hit "midi2abc not found in PATH" even
though `brew install abcmidi` had been run, because REAPER launched from
Finder/Dock inherits macOS's minimal launchd PATH, not the user's shell
PATH. These tests cover the PATH-tolerant lookup (_find_tool/_chk_abcmidi)
and the install dispatch (cmd_install_abcmidi) added to the Setup tab so the
user doesn't have to fix this from a terminal.
"""

import argparse
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_helpers as sh  # noqa: E402


class TestFindTool(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="find_tool_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_binary(self, dir_: Path, name: str) -> Path:
        dir_.mkdir(parents=True, exist_ok=True)
        p = dir_ / name
        p.write_text("#!/bin/sh\nexit 0\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return p

    def test_on_path_is_preferred(self):
        with patch.object(shutil, "which", return_value="/usr/bin/midi2abc"):
            self.assertEqual(sh._find_tool("midi2abc"), "/usr/bin/midi2abc")

    def test_not_on_path_falls_back_to_common_dirs(self):
        fake_dir = self.tmpdir / "homebrew_bin"
        fake_bin = self._make_fake_binary(fake_dir, "midi2abc")
        with patch.object(shutil, "which", return_value=None), \
             patch.object(sh, "_ABCMIDI_COMMON_DIRS", [str(fake_dir)]):
            self.assertEqual(sh._find_tool("midi2abc"), str(fake_bin))

    def test_not_found_anywhere_returns_none(self):
        with patch.object(shutil, "which", return_value=None), \
             patch.object(sh, "_ABCMIDI_COMMON_DIRS", [str(self.tmpdir / "nowhere")]):
            self.assertIsNone(sh._find_tool("midi2abc"))


class TestChkAbcmidi(TestCase):

    def test_both_present_reports_ok(self):
        with patch.object(sh, "_find_tool", side_effect=lambda n: f"/opt/homebrew/bin/{n}"):
            status, detail = sh._chk_abcmidi()
        self.assertEqual(status, "ok")
        self.assertIn("midi2abc", detail)

    def test_missing_one_reports_missing_with_name(self):
        def fake_find(name):
            return "/opt/homebrew/bin/midi2abc" if name == "midi2abc" else None
        with patch.object(sh, "_find_tool", side_effect=fake_find):
            status, detail = sh._chk_abcmidi()
        self.assertEqual(status, "missing")
        self.assertIn("abc2midi", detail)
        self.assertNotIn("midi2abc", detail)

    def test_missing_both_reports_missing_with_both_names(self):
        with patch.object(sh, "_find_tool", return_value=None):
            status, detail = sh._chk_abcmidi()
        self.assertEqual(status, "missing")
        self.assertIn("midi2abc", detail)
        self.assertIn("abc2midi", detail)


class TestTailErrorMessage(TestCase):
    """_tail_error_message builds the descriptive text _stream() now writes
    on subprocess failure, replacing the old bare 'Error (code N)'."""

    def test_includes_command_name_and_code(self):
        msg = sh._tail_error_message(["/usr/bin/brew", "install", "abcmidi"], 1, [])
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
    is preserved exactly, since panel_setup.lua's poll_simple/poll_prewarm
    parse it with a strict pattern."""

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


class TestCmdInstallAbcmidi(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="install_abcmidi_test_"))
        self.pf = self.tmpdir / "progress.txt"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _args(self):
        return argparse.Namespace(progress=str(self.pf))

    def _read_progress(self) -> str:
        return self.pf.read_text() if self.pf.exists() else ""

    def test_macos_without_homebrew_reports_actionable_error(self):
        with patch.object(sh.platform, "system", return_value="Darwin"), \
             patch.object(shutil, "which", return_value=None):
            sh.cmd_install_abcmidi(self._args())
        content = self._read_progress()
        self.assertTrue(content.startswith("error|"))
        self.assertIn("brew.sh", content)

    def test_macos_with_homebrew_invokes_brew_install(self):
        with patch.object(sh.platform, "system", return_value="Darwin"), \
             patch.object(shutil, "which", return_value="/opt/homebrew/bin/brew"), \
             patch.object(sh, "_stream") as mock_stream:
            sh.cmd_install_abcmidi(self._args())
        mock_stream.assert_called_once()
        cmd = mock_stream.call_args.args[1]
        self.assertEqual(cmd, ["/opt/homebrew/bin/brew", "install", "abcmidi"])

    def test_linux_reports_manual_sudo_instruction(self):
        with patch.object(sh.platform, "system", return_value="Linux"), \
             patch.object(shutil, "which", return_value="/usr/bin/apt-get"):
            sh.cmd_install_abcmidi(self._args())
        content = self._read_progress()
        self.assertTrue(content.startswith("error|"))
        self.assertIn("sudo", content)
        self.assertIn("abcmidi", content)

    def test_unsupported_platform_reports_manual_install(self):
        with patch.object(sh.platform, "system", return_value="Windows"):
            sh.cmd_install_abcmidi(self._args())
        content = self._read_progress()
        self.assertTrue(content.startswith("error|"))
        self.assertIn("manually", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
