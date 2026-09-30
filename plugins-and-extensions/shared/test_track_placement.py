#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for shared/lib/track_placement.lua — the pure decision logic behind
where newly generated track(s)/folder(s) are inserted (StemsSeparator,
Audio2Midi, MidiGenerator, Text2Audio all had the bug of always appending at
reaper.CountTracks(0) instead of landing right below the source track/clip).

Runs shared/lib/test_harness_track_placement.lua (a plain assertion harness,
no reaper mocking needed since the module has no such dependency) via the
system `lua` interpreter.

Requires the Lua interpreter (`lua`) on PATH; skips otherwise.
"""

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import TestCase

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "lib" / "test_harness_track_placement.lua"

_HAS_LUA = shutil.which("lua") is not None


def run_harness():
    return subprocess.run(
        ["lua", str(HARNESS)],
        capture_output=True, text=True, cwd=str(HERE), timeout=30,
    )


@unittest.skipUnless(_HAS_LUA, "lua interpreter not found on PATH")
class TestTrackPlacement(TestCase):
    def test_all_cases_pass(self):
        result = run_harness()
        self.assertEqual(
            result.returncode, 0,
            msg=f"track_placement harness failed:\n{result.stdout}\n{result.stderr}",
        )
        self.assertNotIn("FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
