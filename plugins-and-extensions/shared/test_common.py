#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for shared/lib/common.lua's detect_reaper_python()/_find_versioned_python3().

Runs shared/lib/test_harness_common.lua (a plain assertion harness with a
stubbed `reaper` global) via the system `lua` interpreter — same approach
as test_process_control.py / test_collapse_state.py.

Requires the Lua interpreter (`lua`) on PATH; skips otherwise.
"""

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import TestCase

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "lib" / "test_harness_common.lua"

_HAS_LUA = shutil.which("lua") is not None


def run_harness():
    return subprocess.run(
        ["lua", str(HARNESS)],
        capture_output=True, text=True, cwd=str(HERE / "lib"), timeout=30,
    )


@unittest.skipUnless(_HAS_LUA, "lua interpreter not found on PATH")
class TestDetectReaperPython(TestCase):
    def test_all_cases_pass(self):
        result = run_harness()
        self.assertEqual(
            result.returncode, 0,
            msg=f"common.lua harness failed:\n{result.stdout}\n{result.stderr}",
        )
        self.assertNotIn("FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
