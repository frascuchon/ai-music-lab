#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for research_mustango_modal.py's container image setup.

Context: "ModuleNotFoundError: No module named 'pkg_resources'" — librosa
(pulled in by mustango's requirements.txt) imports pkg_resources at import
time (librosa/util/files.py), but recent Python/pip no longer bundle
setuptools (which provides pkg_resources) in a fresh venv by default. The
image now installs setuptools explicitly before requirements.txt, so
librosa (and therefore `from mustango import Mustango`) can import cleanly.

modal.Image's build steps only execute remotely inside a real Modal build
(no local Docker/network access here), so this asserts the explicit
setuptools step is present in the image definition's source rather than
actually building the image — consistent with how this fix must be
verified: the step must exist BEFORE the requirements.txt install (which is
what pulls in librosa).
"""

import re
import unittest
from pathlib import Path
from unittest import TestCase

SRC = Path(__file__).resolve().parent / "research_mustango_modal.py"


class TestMustangoImageInstallsSetuptools(TestCase):

    def setUp(self):
        self.src = SRC.read_text()

    def test_setuptools_explicitly_installed(self):
        self.assertIn(
            '.pip_install("setuptools")', self.src,
            "mustango image must explicitly install setuptools — librosa "
            "imports pkg_resources at import time and recent Python/pip "
            "no longer bundle it by default.",
        )

    def test_setuptools_installed_before_requirements_txt(self):
        setuptools_pos = self.src.index('.pip_install("setuptools")')
        req_txt_pos = self.src.index("req_mustango.txt")
        self.assertLess(
            setuptools_pos, req_txt_pos,
            "setuptools must be installed before requirements.txt (which "
            "pulls in librosa) so pkg_resources is available by the time "
            "librosa gets imported.",
        )

    def test_duration_fixed_at_10s_is_documented_as_a_model_limitation(self):
        """Mustango's UNet operates on a fixed-length mel-spectrogram
        latent trained for ~10s clips — this is a genuine architecture
        limitation of the pretrained checkpoint, not a plugin bug. The
        --seconds param is intentionally ignored; this just guards that the
        warning stays in place so it doesn't silently regress into a
        confusing no-op."""
        self.assertIn("ignorado", self.src)
        self.assertIn("10 s fijo", self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
