#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The ACE-Step Modal image must be reproducible.

Regression: the image installed torch/torchaudio/torchvision unpinned (the
+cu pins are stripped from ACE-Step's pyproject) and cloned ACE-Step's moving
HEAD. Modal caches images per account, so it worked on the account that built
it long ago, while a fresh build elsewhere resolved torch 2.15 (CUDA 13) next
to torchaudio 2.11 and failed with a torchaudio dependency error.
"""
import os
import re
import sys
from pathlib import Path
from unittest import TestCase, main

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import research_acestep_edit_modal as edit  # noqa: E402
import research_acestep_gen_modal as gen  # noqa: E402

SCRIPTS = {"gen": HERE / "research_acestep_gen_modal.py",
           "edit": HERE / "research_acestep_edit_modal.py"}


def _versions(pins):
    return {re.split(r"[=<>]", p)[0]: p.split("==")[1] for p in pins}


class TestAceStepImagePins(TestCase):

    def test_gen_and_edit_use_identical_pins(self):
        self.assertEqual(gen.TORCH_PINS, edit.TORCH_PINS)
        self.assertEqual(gen.ACESTEP_REF, edit.ACESTEP_REF)

    def test_every_torch_package_is_pinned_exactly(self):
        v = _versions(gen.TORCH_PINS)
        self.assertEqual(set(v), {"torch", "torchaudio", "torchvision", "torchcodec"})

    def test_torch_and_torchaudio_are_the_same_release(self):
        """torchaudio is built against one exact torch release."""
        v = _versions(gen.TORCH_PINS)
        self.assertEqual(v["torch"], v["torchaudio"])

    def test_acestep_ref_is_a_full_commit_sha(self):
        self.assertRegex(gen.ACESTEP_REF, r"^[0-9a-f]{40}$")

    def test_scripts_use_the_pins_and_checkout(self):
        for name, path in SCRIPTS.items():
            src = path.read_text()
            self.assertIn(".pip_install(*TORCH_PINS)", src, name)
            self.assertIn("checkout --detach {ACESTEP_REF}", src, name)
            self.assertNotRegex(src, r'pip_install\("torch"', f"{name}: unpinned torch")
            self.assertNotIn("--depth 1", src, f"{name}: shallow clone cannot checkout a sha")

    def test_pinned_set_resolves_together_on_linux(self):
        """The exact set must be installable together for the Modal container
        (Linux x86_64, Python 3.11) — no extra network needed beyond PyPI."""
        import shutil
        import subprocess
        # The real uv: other test modules put fake ones (uv_mock*) on PATH.
        uv = shutil.which("uv", path=os.pathsep.join(
            d for d in os.environ.get("PATH", "").split(os.pathsep) if "uv_mock" not in d))
        if not uv:
            self.skipTest("uv not installed")
        r = subprocess.run(
            [uv, "pip", "install", "--dry-run", "--python-platform",
             "x86_64-manylinux_2_28", "--python-version", "3.11", *gen.TORCH_PINS],
            capture_output=True, text=True)
        if r.returncode != 0 and any(w in r.stderr.lower() for w in
                                     ("dns", "connect", "network", "timed out")):
            self.skipTest("no network access to PyPI")
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    main()
