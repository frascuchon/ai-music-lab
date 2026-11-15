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

    def test_build_failure_reports_the_cause_not_modal_closing_line(self):
        """A failed image build ends with a generic Modal line; the real reason
        (here a torchaudio ImportError) is earlier and must be what the user sees."""
        script = self.tmpdir / "stub_build_fail.py"
        script.write_text(textwrap.dedent("""
            import sys
            print("Building image im-123")
            print("ImportError: libtorchaudio.so: undefined symbol: _ZN3c10", flush=True)
            print("Image build for im-123 failed.", flush=True)
            sys.exit(1)
        """))
        self._run(script)
        _, _, msg, extra = parse_progress(self.progress_file)
        self.assertIn("undefined symbol", msg)
        self.assertNotIn("Image build for", msg)
        self.assertTrue(any("Image build for im-123 failed" in l for l in extra))

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


class TestExecutionDispatch(TestCase):
    """--execution local|remote dispatch added alongside the local/remote
    toggle in panel.lua. remote (default) must be provably unchanged;
    local must route to text2audio_local.py for supported models and
    reject unsupported ones (e.g. acestep_gen, acestep) without touching uv/
    subprocess at all."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="t2a_exec_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.out_dir = self.tmpdir / "out"
        self.shared_dir = self.tmpdir / "shared"
        self.shared_dir.mkdir()
        (self.shared_dir / "pyproject.toml").write_text("[project]\nname='shared'\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, model, execution, mode="generate", prompt="a calm piano loop",
             script_name="research_sao_modal.py"):
        script = self.tmpdir / script_name
        if not script.exists():
            script.write_text("import sys; sys.exit(0)\n")
        args = [
            sys.executable, str(SCRIPT),
            "--shared-dir", str(self.shared_dir),
            "--script", str(script),
            "--model", model,
            "--mode", mode,
            "--out-dir", str(self.out_dir),
            "--prompt", prompt,
            "--progress", str(self.progress_file),
            "--execution", execution,
        ]
        return subprocess.run(args, capture_output=True, text=True, timeout=30)

    def test_local_unsupported_model_errors_without_subprocess(self):
        proc = self._run("acestep_gen", "local")
        self.assertEqual(proc.returncode, 1)
        state, _, msg, _ = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("Local execution not supported for acestep_gen", msg)
        # No output.wav should ever appear — nothing was launched.
        self.assertFalse((self.out_dir / "output.wav").exists())

    def test_local_unsupported_edit_model_errors(self):
        proc = self._run("acestep", "local", mode="edit")
        self.assertEqual(proc.returncode, 1)
        state, _, msg, _ = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("Local execution not supported for acestep", msg)

    def test_local_supported_model_routes_to_local_script(self):
        # text2audio_local.py will itself fail fast (no real model download),
        # but it MUST be the thing that ran — not "uv run ... modal run ...".
        proc = self._run("sao", "local")
        state, _, msg, extra = parse_progress(self.progress_file)
        self.assertEqual(state, "error")  # expected: no real GPU/weights here
        # The remote-path wording ("Modal failed...") must not appear —
        # confirms this went through the local-path error formatting.
        self.assertNotIn("Modal failed", msg)

    def test_remote_default_unchanged_for_known_bug(self):
        """Regression guard: omitting --execution entirely (as the OLD
        panel.lua command line did, before this feature existed) must still
        take the remote path byte-for-byte the same as before."""
        script = self._write_failing_stub("modal.exception.AuthError: token missing")
        proc = self._run_raw(script)  # no --execution flag at all
        self.assertEqual(proc.returncode, 1)
        state, _, msg, _ = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("Modal failed", msg)
        self.assertIn("token missing", msg)

    def test_remote_explicit_unchanged(self):
        script = self._write_failing_stub("RuntimeError: CUDA out of memory")
        proc = self._run("sao", "remote", script_name=script.name)
        self.assertEqual(proc.returncode, 1)
        state, _, msg, _ = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("Modal failed", msg)

    def _write_failing_stub(self, error_line):
        script = self.tmpdir / "research_failing_modal.py"
        script.write_text(textwrap.dedent(f"""\
        import sys
        print("Starting SAO inference...")
        print({error_line!r})
        sys.exit(1)
        """))
        return script

    def _run_raw(self, script):
        args = [
            sys.executable, str(SCRIPT),
            "--shared-dir", str(self.shared_dir),
            "--script", str(script),
            "--model", "sao",
            "--mode", "generate",
            "--out-dir", str(self.out_dir),
            "--prompt", "a calm piano loop",
            "--progress", str(self.progress_file),
        ]
        return subprocess.run(args, capture_output=True, text=True, timeout=30)


class TestAceStepCommandBuilding(TestCase):
    """The new ACE-Step controls must reach the Modal entrypoints."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="t2a_ace_"))
        self.out_dir = self.tmpdir / "out"
        self.shared_dir = self.tmpdir / "shared"
        self.shared_dir.mkdir()
        (self.shared_dir / "pyproject.toml").write_text("[project]\nname='shared'\n")
        self.script = self.tmpdir / "stub_modal.py"
        self.script.write_text(textwrap.dedent("""\
        import json, sys, pathlib
        a = sys.argv[1:]
        out = pathlib.Path(a[a.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        import os
        (out / "argv.json").write_text(json.dumps(a))
        (out / "gpu.txt").write_text(os.environ.get("ACESTEP_GPU", ""))
        (out / "output.wav").write_bytes(b"RIFF")
        """))
        self.lyrics = self.tmpdir / "lyrics.txt"
        self.lyrics.write_text("[verse]\nhello\n")
        self.src = self.tmpdir / "src.wav"
        self.src.write_bytes(b"RIFF")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, *extra, model, mode):
        import json
        cmd = [sys.executable, str(SCRIPT), "--shared-dir", str(self.shared_dir),
               "--script", str(self.script), "--model", model, "--mode", mode,
               "--out-dir", str(self.out_dir), "--prompt", "warm pads",
               "--progress", str(self.tmpdir / "p.txt"), *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return json.loads((self.out_dir / "argv.json").read_text())

    def test_generate_forwards_lyrics_and_metadata(self):
        argv = self._run("--lyrics-file", str(self.lyrics), "--bpm", "92",
                         "--keyscale", "A minor", "--timesignature", "4",
                         "--vocal-language", "en", model="acestep_gen", mode="generate")
        self.assertEqual(argv[argv.index("--lyrics-file") + 1], str(self.lyrics))
        self.assertEqual(argv[argv.index("--bpm") + 1], "92")
        self.assertEqual(argv[argv.index("--keyscale") + 1], "A minor")
        self.assertEqual(argv[argv.index("--timesignature") + 1], "4")
        self.assertEqual(argv[argv.index("--vocal-language") + 1], "en")

    def test_generate_without_controls_adds_no_metadata_flags(self):
        argv = self._run(model="acestep_gen", mode="generate")
        for flag in ("--lyrics-file", "--bpm", "--keyscale", "--timesignature",
                     "--vocal-language"):
            self.assertNotIn(flag, argv)

    def test_edit_cover_defaults_to_turbo_and_intensity_strength(self):
        argv = self._run("--input", str(self.src), model="acestep", mode="edit")
        self.assertEqual(argv[argv.index("--task") + 1], "cover")
        self.assertEqual(argv[argv.index("--dit-variant") + 1], "turbo")
        self.assertEqual(argv[argv.index("--strength") + 1], "0.7")

    def test_edit_repaint_forwards_region_and_explicit_strength(self):
        argv = self._run("--input", str(self.src), "--ace-task", "repaint",
                         "--repaint-start", "2.5", "--repaint-end", "6",
                         "--strength", "0.55", model="acestep", mode="edit")
        self.assertEqual(argv[argv.index("--task") + 1], "repaint")
        self.assertEqual(argv[argv.index("--start") + 1], "2.5")
        self.assertEqual(argv[argv.index("--end") + 1], "6.0")
        self.assertEqual(argv[argv.index("--strength") + 1], "0.55")

    def test_edit_lego_forwards_track_and_base_variant(self):
        argv = self._run("--input", str(self.src), "--ace-task", "lego",
                         "--dit-variant", "base", "--track-name", "bass",
                         model="acestep", mode="edit")
        self.assertEqual(argv[argv.index("--track-name") + 1], "bass")
        self.assertEqual(argv[argv.index("--dit-variant") + 1], "base")

    def test_edit_lego_requires_base_checkpoint(self):
        cmd = [sys.executable, str(SCRIPT), "--shared-dir", str(self.shared_dir),
               "--script", str(self.script), "--model", "acestep", "--mode", "edit",
               "--out-dir", str(self.out_dir), "--prompt", "x", "--input", str(self.src),
               "--ace-task", "lego", "--track-name", "bass", "--dit-variant", "turbo",
               "--progress", str(self.tmpdir / "p.txt")]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("base checkpoint", (self.tmpdir / "p.txt").read_text())

    def test_generate_forwards_xl_variant_and_lm_size(self):
        argv = self._run("--dit-variant", "xl_sft", "--thinking", "--lm-size", "4B",
                         model="acestep_gen", mode="generate")
        self.assertEqual(argv[argv.index("--dit-variant") + 1], "xl_sft")
        self.assertEqual(argv[argv.index("--lm-size") + 1], "4B")

    def test_lm_size_is_not_sent_without_thinking(self):
        argv = self._run("--lm-size", "4B", model="acestep_gen", mode="generate")
        self.assertNotIn("--lm-size", argv)

    def test_panel_gpu_reaches_the_acestep_script(self):
        self._run("--gpu", "A100-80GB", model="acestep_gen", mode="generate")
        self.assertEqual((self.out_dir / "gpu.txt").read_text(), "A100-80GB")

    def test_xl_base_generation_requires_a_lora(self):
        cmd = [sys.executable, str(SCRIPT), "--shared-dir", str(self.shared_dir),
               "--script", str(self.script), "--model", "acestep_gen", "--mode", "generate",
               "--out-dir", str(self.out_dir), "--prompt", "x", "--dit-variant", "xl_base",
               "--progress", str(self.tmpdir / "p.txt")]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("needs a LoRA", (self.tmpdir / "p.txt").read_text())

    def test_edit_lego_accepts_xl_base(self):
        argv = self._run("--input", str(self.src), "--ace-task", "lego",
                         "--dit-variant", "xl_base", "--track-name", "drums",
                         model="acestep", mode="edit")
        self.assertEqual(argv[argv.index("--dit-variant") + 1], "xl_base")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestAceStepBaseRequiresLora(TestCase):
    """ACE-Step base is only usable with a LoRA/LoKr adapter; sft/turbo never take one."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="t2a_lora_"))
        self.progress_file = self.tmpdir / "progress.txt"
        self.out_dir = self.tmpdir / "out"
        self.shared_dir = self.tmpdir / "shared"
        self.shared_dir.mkdir()
        (self.shared_dir / "pyproject.toml").write_text("[project]\nname='shared'\n")
        self.script = self.tmpdir / "research_acestep_gen_modal.py"
        self.script.write_text("import sys; print(sys.argv[1:]); sys.exit(1)\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, *extra):
        args = [sys.executable, str(SCRIPT), "--shared-dir", str(self.shared_dir),
                "--script", str(self.script), "--model", "acestep_gen",
                "--mode", "generate", "--out-dir", str(self.out_dir),
                "--prompt", "a calm piano loop", "--progress", str(self.progress_file), *extra]
        return subprocess.run(args, capture_output=True, text=True, timeout=30)

    def test_base_without_lora_is_rejected_before_launching_modal(self):
        proc = self._run("--dit-variant", "base")
        self.assertEqual(proc.returncode, 1)
        state, _, msg, _ = parse_progress(self.progress_file)
        self.assertEqual(state, "error")
        self.assertIn("needs a LoRA", msg)

    def test_base_with_missing_lora_folder_is_rejected(self):
        self._run("--dit-variant", "base", "--lora-path", str(self.tmpdir / "nope"))
        _, _, msg, _ = parse_progress(self.progress_file)
        self.assertIn("not found", msg)

    def test_base_with_lora_folder_forwards_it(self):
        lora = self.tmpdir / "adapter"; lora.mkdir()
        proc = self._run("--dit-variant", "base", "--lora-path", str(lora))
        self.assertNotIn("needs a LoRA", self.progress_file.read_text())
        self.assertIn("--lora-path", proc.stdout + self.progress_file.read_text() + proc.stderr)

    def test_sft_ignores_lora_and_is_not_blocked(self):
        lora = self.tmpdir / "adapter"; lora.mkdir()
        proc = self._run("--dit-variant", "sft", "--lora-path", str(lora))
        text = self.progress_file.read_text()
        self.assertNotIn("needs a LoRA", text)
        self.assertNotIn("--lora-path", text)
        self.assertIn("ignored", proc.stdout)
