#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for widgets_extra.input_textarea scrolling (overlapping lines bug).

Runs shared/lib/test_harness_textarea.lua with the system `lua` interpreter,
like test_scroll_region.py. Skips when `lua` is not on PATH.
"""

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import TestCase

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "lib" / "test_harness_textarea.lua"

_HAS_LUA = shutil.which("lua") is not None


@unittest.skipUnless(_HAS_LUA, "lua interpreter not found on PATH")
class TestInputTextarea(TestCase):
    def test_all_cases_pass(self):
        result = subprocess.run(
            ["lua", str(HARNESS)],
            capture_output=True, text=True, cwd=str(HERE / "lib"), timeout=30,
        )
        self.assertEqual(
            result.returncode, 0,
            msg=f"input_textarea harness failed:\n{result.stdout}\n{result.stderr}",
        )
        self.assertNotIn("FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
