#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for tools/midi_abc.py.

Covers the fix where midi2abc's default title ("T: from <full temp path>")
leaked the local filesystem path of exported seed files (e.g.
/var/folders/.../midigen_seed_3.mid) straight into the ABC text later
embedded in ChatMusician's prompt — wasted tokens and out-of-distribution
"title" text for the model. -title Seed now overrides it.

Requires the abcmidi package (midi2abc/abc2midi) — brew install abcmidi /
apt install abcmidi — same prerequisite already documented in
research_chatmusician_modal.py's docstring. Tests skip if unavailable.
"""

import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import midi_abc
from tools.midi_abc import midi_to_abc_text  # noqa: E402

_HAS_MIDI2ABC = shutil.which("midi2abc") is not None


def make_midi(path: Path, n_notes: int = 3):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    for i in range(n_notes):
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=60 + i, start=i * 0.5, end=i * 0.5 + 0.4))
    pm.instruments.append(inst)
    pm.write(str(path))


@unittest.skipUnless(_HAS_MIDI2ABC, "midi2abc not installed (brew/apt install abcmidi)")
class TestMidiToAbcText(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="midi_abc_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_title_does_not_leak_local_path(self):
        """Regression: the T: field must not contain the input file's
        filesystem path — it used to (midi2abc's default), burning prompt
        tokens on a meaningless absolute path and confusing the model with
        an out-of-distribution 'title'."""
        long_named_dir = self.tmpdir / "midigen_run_seed_export_3"
        long_named_dir.mkdir()
        seed = long_named_dir / "midigen_seed_3.mid"
        make_midi(seed)

        abc_text = midi_to_abc_text(str(seed))

        self.assertNotIn(str(seed), abc_text)
        self.assertNotIn("midigen_seed_3", abc_text)
        title_lines = [l for l in abc_text.splitlines() if l.startswith("T:")]
        self.assertEqual(len(title_lines), 1)
        self.assertIn("Seed", title_lines[0])

    def test_still_produces_valid_headers(self):
        seed = self.tmpdir / "seed.mid"
        make_midi(seed)
        abc_text = midi_to_abc_text(str(seed))
        self.assertIn("X:", abc_text)
        self.assertIn("K:", abc_text)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            midi_to_abc_text(str(self.tmpdir / "nope.mid"))


class TestResolveToolFallsBackToCommonDirs(TestCase):
    """Regression for a real user report: `midi2abc` was installed via
    `brew install abcmidi` and perfectly present on disk, but REAPER was
    launched from Finder/Dock (not a terminal) and so ran with macOS's
    minimal launchd PATH — a plain shutil.which("midi2abc") from inside
    REAPER's subprocess tree came up empty even though the binary existed.
    _resolve_tool must also check the well-known Homebrew/MacPorts dirs."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="resolve_tool_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_binary(self, dir_: Path, name: str) -> Path:
        dir_.mkdir(parents=True, exist_ok=True)
        p = dir_ / name
        p.write_text("#!/bin/sh\nexit 0\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return p

    def test_not_on_path_but_present_in_common_dir_is_found(self):
        fake_dir = self.tmpdir / "homebrew_bin"
        fake_bin = self._make_fake_binary(fake_dir, "midi2abc")
        with patch.object(shutil, "which", return_value=None), \
             patch.object(midi_abc, "_COMMON_TOOL_DIRS", [str(fake_dir)]):
            self.assertEqual(midi_abc._resolve_tool("midi2abc"), str(fake_bin))

    def test_not_found_anywhere_raises_with_actionable_message(self):
        with patch.object(shutil, "which", return_value=None), \
             patch.object(midi_abc, "_COMMON_TOOL_DIRS", [str(self.tmpdir / "nowhere")]):
            with self.assertRaises(RuntimeError) as ctx:
                midi_abc._resolve_tool("midi2abc")
            self.assertIn("brew install abcmidi", str(ctx.exception))

    def test_on_path_is_preferred_over_common_dirs(self):
        with patch.object(shutil, "which", return_value="/usr/bin/midi2abc"):
            self.assertEqual(midi_abc._resolve_tool("midi2abc"), "/usr/bin/midi2abc")


if __name__ == "__main__":
    unittest.main(verbosity=2)
