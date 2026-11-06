#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for research_basic_pitch_modal.py (Basic Pitch on Modal).

The Modal function reuses research_basic_pitch.main() instead of copying its
pipeline, so these tests check the glue: argv building, the bytes round trip,
failure handling, that the image pins what the local script needs, and that
the local script's CLI is unchanged by the main(argv) refactor.
"""

import re
import sys
import types
import unittest
from pathlib import Path
from unittest import TestCase, mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

modal = __import__("pytest").importorskip("modal")
import research_basic_pitch_modal as bpm  # noqa: E402


class TestBuildArgs(TestCase):
    def test_minimal(self):
        self.assertEqual(bpm.build_args("a.wav", "o.mid", 0.0, 0.0, False),
                         ["--input", "a.wav", "--out", "o.mid"])

    def test_all_options(self):
        argv = bpm.build_args("a.wav", "o.mid", 0.8, 87.5, True)
        self.assertEqual(argv, ["--input", "a.wav", "--out", "o.mid",
                                "--onset-threshold", "0.8", "--bpm", "87.5",
                                "--no-beat-tracking"])

    def test_argv_is_accepted_by_the_local_script_parser(self):
        import research_basic_pitch as local
        with mock.patch.object(local.Path, "exists", return_value=False):
            # input missing -> main returns 1 right after parsing: proves the
            # modal-built argv parses (an unknown flag would SystemExit(2)).
            self.assertEqual(local.main(bpm.build_args("a.wav", "o.mid", 0.8, 87.5, True)), 1)


class TestRemoteFunction(TestCase):
    def _fake_bp(self, rc, write=b"MThd-bytes"):
        calls = []

        def fake_main(argv):
            calls.append(argv)
            if write is not None:
                Path(argv[argv.index("--out") + 1]).write_bytes(write)
            return rc
        return types.SimpleNamespace(main=fake_main), calls

    def _run(self, fake, **kw):
        with mock.patch.dict(sys.modules, {"research_basic_pitch": fake}):
            return bpm.transcribe.get_raw_f()(b"RIFFdata", ".wav", **kw)

    def test_returns_midi_bytes_and_feeds_audio_to_the_pipeline(self):
        fake, calls = self._fake_bp(0)
        out = self._run(fake, onset_threshold=0.75, bpm=100.0)
        self.assertEqual(out, b"MThd-bytes")
        argv = calls[0]
        self.assertTrue(argv[argv.index("--input") + 1].endswith(".wav"))
        self.assertIn("--onset-threshold", argv)
        self.assertIn("--bpm", argv)

    def test_pipeline_failure_returns_empty(self):
        fake, _ = self._fake_bp(1, write=None)
        self.assertEqual(self._run(fake), b"")

    def test_exit_zero_without_output_returns_empty(self):
        fake, _ = self._fake_bp(0, write=None)
        self.assertEqual(self._run(fake), b"")


class TestImage(TestCase):
    src = (HERE / "research_basic_pitch_modal.py").read_text()

    def test_is_a_modal_image_and_app(self):
        self.assertIsInstance(bpm.image, modal.Image)
        self.assertIsInstance(bpm.app, modal.App)

    def test_image_pins_exact_versions(self):
        pins = re.findall(r'"([A-Za-z0-9_.-]+)(==|<)([^"]+)"', self.src)
        names = {n.lower().replace("_", "-") for n, _, _ in pins}
        for needed in ("basic-pitch", "numpy", "librosa", "pretty-midi", "soundfile", "scipy"):
            self.assertIn(needed, names)
        for n, op, v in pins:
            if n != "setuptools":
                self.assertEqual(op, "==", f"{n} must be pinned exactly")

    def test_pins_agree_with_the_local_lock(self):
        lock = (HERE / "uv.lock").read_text()
        pinned = {n.lower().replace("_", "-"): v
                  for n, v in re.findall(r'"([A-Za-z0-9_.-]+)==([\d.]+)"', self.src)}
        for name in ("basic-pitch", "numpy", "librosa", "pretty-midi", "scipy", "resampy"):
            self.assertIn(name, pinned)
            self.assertRegex(lock, rf'name = "{name}"\nversion = "{re.escape(pinned[name])}"')

    def test_local_script_is_shipped_into_the_image(self):
        self.assertTrue(bpm.LOCAL_SCRIPT.is_file())
        self.assertIn("add_local_file", self.src)

    def test_entrypoint_has_the_flags_transcribe_py_passes(self):
        import inspect
        params = set(inspect.signature(bpm.main.info.raw_f).parameters)
        self.assertTrue({"audio_path", "out_dir", "force", "onset_threshold", "bpm",
                         "no_beat_tracking"} <= params)


class TestLocalScriptUnchanged(TestCase):
    def test_main_still_defaults_to_sys_argv(self):
        import research_basic_pitch as local
        with mock.patch.object(sys, "argv", ["x", "--input", "/nonexistent.wav", "--out", "o.mid"]):
            self.assertEqual(local.main(), 1)


if __name__ == "__main__":
    unittest.main()
