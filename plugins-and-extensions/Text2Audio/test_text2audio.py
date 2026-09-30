#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for text2audio.py's error reporting.

Context: the "más robusta gestión de errores" goal flagged that when the
Modal subprocess fails, the user only sees a message like "Modal failed
with code 1. Check the log for details." — no hint of what actually went
wrong (OOM, missing secret, bad prompt, etc). text2audio.py now keeps a
rolling tail of the subprocess's own stdout/stderr and surfaces the real
last line in the progress-file error message, plus the full tail as
"extra" lines (mirroring the pattern already used by separate_demucs.py,
separate_sam.py and midigen.py).

Uses the same mock-uv-on-PATH technique as Audio2Midi/test_transcribe.py
so the CLI can be driven end-to-end without a real Modal account.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import TestCase

SCRIPT = Path(__file__).resolve().parent / "text2audio.py"

# ---------------------------------------------------------------------------
# Mock uv on PATH — resolves "uv run --project <dir> modal run <script>::main
# <args...>" to just running <script> with the same interpreter as the tests.
# ---------------------------------------------------------------------------
_UV_MOCK_DIR = Path(tempfile.mkdtemp(prefix="uv_mock_t2a_"))
_UV_MOCK = _UV_MOCK_DIR / "uv"
_UV_MOCK.write_text(textwrap.dedent(f"""\
#!/usr/bin/env python3
import sys, os
PY = {sys.executable!r}
args = sys.argv[1:]
for i, a in enumerate(args):
    if '::' in a:
        script = a.split('::')[0]
        os.execvp(PY, [PY, script] + args[i+1:])
        break
    if a.endswith('.py'):
        os.execvp(PY, [PY] + args[i:])
        break
sys.exit(1)
"""))
_UV_MOCK.chmod(0o755)
_ORIG_PATH = os.environ.get("PATH", "")


def setUpModule():
    os.environ["PATH"] = str(_UV_MOCK_DIR) + ":" + _ORIG_PATH


def tearDownModule():
    os.environ["PATH"] = _ORIG_PATH
    shutil.rmtree(_UV_MOCK_DIR, ignore_errors=True)


def parse_progress(path):
    if not path.exists():
        return None, None, None, []
    text = path.read_text()
    if not text.strip():
        return None, None, None, []
    lines = text.splitlines()
    parts = lines[0].split("|")
    if len(parts) < 3:
        return None, None, None, []
    state, pct = parts[0], float(parts[1])
    msg = "|".join(parts[2:])
    extra = [l for l in lines[1:] if l.strip()]
    return state, pct, msg, extra


class TestErrorReporting(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="t2a_err_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.out_dir = self.tmpdir / "out"
        self.shared_dir = self.tmpdir / "shared"
        self.shared_dir.mkdir()
        (self.shared_dir / "pyproject.toml").write_text("[project]\nname='shared'\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_failing_stub(self, error_line):
        script = self.tmpdir / "research_failing_modal.py"
        script.write_text(textwrap.dedent(f"""\
        import sys
        print("Starting SAO inference...")
        print({error_line!r})
        sys.exit(1)
        """))
        return script

    def _run(self, script, model="sao", mode="generate", prompt="a calm piano loop"):
        args = [
            sys.executable, str(SCRIPT),
            "--shared-dir", str(self.shared_dir),
            "--script", str(script),
            "--model", model,
            "--mode", mode,
            "--out-dir", str(self.out_dir),
            "--prompt", prompt,
            "--progress", str(self.progress_file),
        ]
        return subprocess.run(args, capture_output=True, text=True, timeout=30)

    def test_error_message_includes_real_cause_not_just_exit_code(self):
        script = self._write_failing_stub(
            "modal.exception.AuthError: token missing or invalid")
        proc = self._run(script)
        self.assertEqual(proc.returncode, 1)
        state, _, msg, extra = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("code 1", msg)
        self.assertIn("token missing or invalid", msg)
        self.assertNotEqual(
            msg.strip(),
            "Modal failed with code 1. Check the log for details.",
            "regression: error message must not collapse back to a bare "
            "exit-code-only message",
        )

    def test_extra_lines_contain_full_tail_for_full_log_view(self):
        script = self._write_failing_stub("RuntimeError: CUDA out of memory")
        self._run(script)
        _, _, _, extra = parse_progress(self.progress_file)
        self.assertIn("--- last output lines ---", extra)
        self.assertTrue(any("CUDA out of memory" in l for l in extra))
        self.assertTrue(any("Starting SAO inference" in l for l in extra))

    def test_no_output_failure_still_gives_actionable_message(self):
        script = self.tmpdir / "research_silent_fail_modal.py"
        script.write_text("import sys; sys.exit(1)\n")
        proc = self._run(script)
        self.assertEqual(proc.returncode, 1)
        state, _, msg, _ = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("code 1", msg)

    def test_missing_script_reports_path_not_bare_error(self):
        missing = self.tmpdir / "does_not_exist_modal.py"
        proc = self._run(missing)
        self.assertEqual(proc.returncode, 1)
        state, _, msg, _ = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn(str(missing), msg)

    def test_empty_prompt_reports_actionable_message(self):
        script = self._write_failing_stub("unused")
        proc = self._run(script, prompt="   ")
        self.assertEqual(proc.returncode, 1)
        state, _, msg, _ = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("prompt", msg.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
