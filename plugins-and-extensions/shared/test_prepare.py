#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for prepare.py (first-run preparation catalog + markers), the
`prepare` subcommand of setup_helpers.py and lib/prepared.lua."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare  # noqa: E402
import setup_helpers as sh  # noqa: E402


class TmpState(TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        env = patch.dict(os.environ, {prepare.STATE_ENV: str(self.tmp / "sub" / "prepared.txt")})
        env.start()
        self.addCleanup(env.stop)


class TestMarkers(TmpState):
    def test_unprepared_by_default(self):
        self.assertFalse(prepare.is_prepared("text2audio", "sao", "remote"))

    def test_mark_is_per_execution_and_idempotent(self):
        prepare.mark("text2audio", "sao", "remote")
        prepare.mark("text2audio", "sao", "remote")
        self.assertTrue(prepare.is_prepared("text2audio", "sao", "remote"))
        self.assertFalse(prepare.is_prepared("text2audio", "sao", "local"))
        self.assertEqual(len(prepare.load()), 1)


class TestConcurrentMarkers(TmpState):
    def test_many_processes_marking_at_once_lose_nothing(self):
        code = ("import sys; sys.path.insert(0, %r); import prepare; "
                "prepare.mark('text2audio', sys.argv[1], 'remote')" % str(HERE))
        procs = [subprocess.Popen([sys.executable, "-c", code, f"m{i}"], env=dict(os.environ))
                 for i in range(12)]
        for p in procs:
            self.assertEqual(p.wait(), 0)
        self.assertEqual(len(prepare.load()), 12)

    def test_group_of_links_acestep_gen_and_edit_only(self):
        gen = prepare.find_target("text2audio", "acestep_gen", "remote")
        edit = prepare.find_target("text2audio", "acestep", "remote")
        sao = prepare.find_target("text2audio", "sao", "remote")
        self.assertNotEqual(prepare.group_of(gen), "")
        self.assertEqual(prepare.group_of(gen), prepare.group_of(edit))
        self.assertEqual(prepare.group_of(sao), "")


class TestAppDir(TestCase):
    def test_state_lives_in_the_ai_musiclab_folder(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(prepare.STATE_ENV, None)
            self.assertEqual(prepare.state_file(), Path.home() / ".ai-musiclab" / "prepared.txt")

    def test_lua_and_python_agree_on_the_folder_name(self):
        if not shutil.which("lua"):
            self.skipTest("lua not on PATH")
        r = subprocess.run(["lua", "-e", f'package.path="{HERE}/lib/?.lua;"..package.path; '
                            'io.write(require("app_dirs").APP_DIR_NAME)'],
                           capture_output=True, text=True)
        self.assertEqual(r.stdout, prepare.APP_DIR_NAME)


class TestCatalog(TestCase):
    def test_every_target_script_and_backend_exist(self):
        for t in prepare.TARGETS:
            self.assertTrue(Path(t["script"]).is_file(), t["script"])
            self.assertTrue(Path(t["backend"]).is_file(), t["backend"])

    def test_targets_are_unique(self):
        keys = [prepare.key(t["domain"], t["model"], t["execution"]) for t in prepare.TARGETS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_local_targets_are_supported_by_the_execution_registry(self):
        from model_execution import is_local_supported
        domains = {"text2audio": "text2audio", "midigen": "midigen",
                   "stemseparator": "stemseparator"}
        for t in prepare.TARGETS:
            if t["execution"] == "local" and t["domain"] in domains:
                self.assertTrue(is_local_supported(domains[t["domain"]], t["model"])[0], t)

    def test_warmup_inputs_are_valid_files(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        import wave
        with wave.open(str(prepare.write_tone(tmp / "t.wav", 1.0)), "rb") as w:
            self.assertEqual(w.getnframes(), 16000)
        mid = prepare.write_melody_mid(tmp / "m.mid").read_bytes()
        self.assertTrue(mid.startswith(b"MThd"))
        try:
            import pretty_midi
        except ImportError:
            return
        self.assertEqual(len(pretty_midi.PrettyMIDI(str(tmp / "m.mid")).instruments[0].notes), 16)


class TestBuildCommand(TestCase):
    """The warm-up command must be accepted by the real backend's argparse."""

    def test_commands_only_use_flags_the_backends_define(self):
        import re
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for t in prepare.TARGETS:
            cmd = prepare.build_command(t, sys.executable, tmp, tmp / "pf.txt")
            src = Path(t["backend"]).read_text()
            for flag in (a for a in cmd if re.fullmatch(r"--[a-z0-9-]+", a)):
                self.assertIn(f'"{flag}"', src, f"{flag} not defined in {t['backend'].name}")

    def test_text2audio_edit_gets_an_input_file(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        cmd = prepare.build_command(prepare.find_target("text2audio", "sao_edit", "remote"),
                                    sys.executable, tmp, tmp / "pf.txt")
        self.assertTrue(Path(cmd[cmd.index("--input") + 1]).is_file())


class TestCmdPrepare(TmpState):
    def _run(self, target, backend_state):
        pf = self.tmp / "pf.txt"
        fake = self.tmp / "fake_backend.py"
        fake.write_text(
            "import sys\nargs = sys.argv\n"
            "pf = args[args.index('--progress') + 1]\n"
            f"open(pf, 'w').write({backend_state!r})\n")
        t = dict(prepare.find_target(*target.split(":")), backend=fake)
        with patch.object(prepare, "find_target", return_value=t), \
                patch.object(sh, "_chk_modal_auth", return_value=("ok", "")):
            sh.cmd_prepare(argparse.Namespace(target=target, progress=str(pf)))
        return pf.read_text().splitlines()[0]

    def test_success_marks_model_prepared(self):
        line = self._run("text2audio:sao:remote", "done|1.0|ok\n")
        self.assertTrue(line.startswith("done|"))
        self.assertTrue(prepare.is_prepared("text2audio", "sao", "remote"))

    def test_silent_synthetic_input_still_counts_as_prepared(self):
        line = self._run("audio2midi:miros:remote",
                         "error|0.5|No notes were transcribed. Check that the item is audible\n")
        self.assertTrue(line.startswith("done|"))
        self.assertTrue(prepare.is_prepared("audio2midi", "miros", "remote"))

    def test_other_errors_for_the_same_model_still_fail(self):
        line = self._run("audio2midi:miros:remote", "error|0.5|Image build failed\n")
        self.assertTrue(line.startswith("error|"))
        self.assertFalse(prepare.is_prepared("audio2midi", "miros", "remote"))

    def test_failure_reports_error_and_does_not_mark(self):
        line = self._run("text2audio:sao:remote", "error|0|Image build failed: boom\n")
        self.assertTrue(line.startswith("error|"))
        self.assertIn("boom", line)
        self.assertFalse(prepare.is_prepared("text2audio", "sao", "remote"))

    def test_already_prepared_is_skipped_without_running_the_backend(self):
        prepare.mark("text2audio", "sao", "remote")
        line = self._run("text2audio:sao:remote", "error|0|backend must not run\n")
        self.assertTrue(line.startswith("done|"))
        self.assertIn("skipped", line)

    def test_acestep_gen_and_edit_are_prepared_together(self):
        self._run("text2audio:acestep_gen:remote", "done|1.0|ok\n")
        self.assertTrue(prepare.is_prepared("text2audio", "acestep", "remote"))
        self.assertFalse(prepare.is_prepared("text2audio", "sao", "remote"))

    def test_unknown_target_is_an_error(self):
        pf = self.tmp / "pf.txt"
        sh.cmd_prepare(argparse.Namespace(target="x:y:remote", progress=str(pf)))
        self.assertTrue(pf.read_text().startswith("error|"))

    def test_remote_requires_modal_auth(self):
        pf = self.tmp / "pf.txt"
        with patch.object(sh, "_chk_modal_auth", return_value=("missing", "")):
            sh.cmd_prepare(argparse.Namespace(target="text2audio:sao:remote", progress=str(pf)))
        self.assertIn("Modal", pf.read_text())


class TestCheckEmitsPrepLines(TmpState):
    def test_prep_lines_reflect_markers(self):
        prepare.mark("text2audio", "sao", "remote")
        pf = self.tmp / "check.txt"
        with patch.object(sh, "_chk_python", return_value=("ok", "")), \
                patch.object(sh, "_chk_uv", return_value=("ok", "")), \
                patch.object(sh, "_chk_modal_cli", return_value=("ok", "")), \
                patch.object(sh, "_chk_modal_auth", return_value=("ok", "")), \
                patch.object(sh, "_chk_demucs", return_value=("ok", "")), \
                patch.object(sh, "_chk_hf_secret", return_value=("ok", "")):
            sh.cmd_check(argparse.Namespace(progress=str(pf)))
        lines = pf.read_text().splitlines()
        self.assertTrue(any(ln.startswith("PREP|text2audio|sao|remote|") and "|ready|" in ln
                            for ln in lines))
        self.assertTrue(any(ln.startswith("PREP|text2audio|sao|local|") and "|missing|normal||" in ln
                            for ln in lines))
        self.assertTrue(any(ln.startswith("PREP|stemseparator|sam_large|local|") and "|heavy|" in ln
                            for ln in lines))


@unittest.skipUnless(shutil.which("lua"), "lua interpreter not found on PATH")
class TestPreparedLua(TmpState):
    def test_lua_reads_and_writes_the_same_format(self):
        prepare.mark("midigen", "amadeus", "remote")
        script = (
            f'package.path = "{HERE}/lib/?.lua;" .. package.path\n'
            'local p = require("prepared")\n'
            'assert(p.is_prepared("midigen","amadeus","remote"))\n'
            'assert(not p.is_prepared("midigen","amadeus","local"))\n'
            'p.mark("audio2midi","miros","remote")\n'
            'local shown = {}\n'
            'local g = { text_colored = function(t) shown[#shown+1] = t end,\n'
            '            text_wrapped = function(t) shown[#shown+1] = t end,\n'
            '            spacing = function() end }\n'
            'assert(p.notice(g, "midigen", "midigpt", "remote") == true and #shown == 2)\n'
            'shown = {}\n'
            'assert(p.notice(g, "midigen", "amadeus", "remote") == false and #shown == 0)\n')
        r = subprocess.run(["lua", "-e", script], capture_output=True, text=True,
                           env=dict(os.environ))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(prepare.is_prepared("audio2midi", "miros", "remote"))


@unittest.skipUnless(shutil.which("lua"), "lua interpreter not found on PATH")
class TestPrepareQueueLua(TestCase):
    def test_scheduler_cases_pass(self):
        r = subprocess.run(["lua", str(HERE / "lib" / "test_harness_prepare_queue.lua")],
                           capture_output=True, text=True, cwd=str(HERE), timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("FAIL", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
