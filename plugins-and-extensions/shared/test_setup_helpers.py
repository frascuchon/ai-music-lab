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
