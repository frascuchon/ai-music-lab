#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for research_chatmusician_modal.py's ABC-extraction and seed-handling
helpers — targeting a real user report: "No midi found in /var/folders/..."
when harmonizing a MIDI seed with ChatMusician (0 valid outputs, no crash).

Two concrete, code-level issues were found and are covered here:

1. _extract_abc's fallback regex only recognized a SUBSET of valid ABC 2.1
   information-field letters (missing A, D, I, V, W, X). This repo's own
   evaluation notes (evaluation/chatmusician/test10) already documented a
   real harmonization failure where the model answered with an "A:" field
   the old regex couldn't see at all. Broadening the letter class lets the
   fallback rescue any well-formed-but-atypical response.

2. _build_instruction concatenated the seed's ENTIRE ABC text into the
   prompt with no length cap. evaluation/chatmusician/test12's own notes
   already diagnosed a 0-output failure as "posiblemente el input más largo
   excede la ventana de atención efectiva" (LLaMA2 7B / ChatMusician has a
   4096-token context, minus max_new_tokens=1536 reserved for generation).
   panel.lua lets a user attach an arbitrarily long in-project MIDI take
   with no warning, unlike the Anticipatory seed path. _truncate_seed_abc
   caps this so a long/whole-song seed degrades gracefully instead of
   reliably blowing every one of the n_outputs attempts.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_chatmusician_modal as rm  # noqa: E402


class TestExtractAbcFieldLetterCoverage(TestCase):
    """_extract_abc's fallback must recognize every valid ABC 2.1
    information-field letter, not just a hand-picked subset."""

    def test_all_valid_abc_field_letters_are_covered(self):
        # abcnotation.com/wiki/abc:standard:v2.1 §3 information fields.
        expected = set("ABCDFGHIKLMNOPQRSTUVWXZ")
        self.assertEqual(set(rm._ABC_FIELD_LETTERS), expected)

    def test_response_starting_with_A_field_is_rescued(self):
        """Regression for the exact shape of the documented test10 failure
        (model answers with an 'A:' field) — as long as M:/K: follow, the
        fallback must now find it instead of raising."""
        response = "A: some area note\nK:G\nM:4/4\nL:1/8\nGABc dedB|\n"
        abc = rm._extract_abc(response)
        self.assertTrue(abc.startswith("X:1\n"))
        self.assertIn("K:G", abc)

    def test_response_starting_with_V_voice_field_is_rescued(self):
        """ChatMusician is marketed as multi-voice ('V:' headers); a
        response that echoes a voice header before X: used to be invisible
        to both regexes."""
        response = "V:1\nK:D\nM:4/4\nL:1/8\nDEFG ABcd|\n"
        abc = rm._extract_abc(response)
        self.assertTrue(abc.startswith("X:1\n"))

    def test_response_starting_with_D_or_I_or_W_field_is_rescued(self):
        for letter in ("D", "I", "W"):
            response = f"{letter}: some field\nK:C\nM:4/4\nCDEF GABc|\n"
            abc = rm._extract_abc(response)
            self.assertIn("K:C", abc)

    def test_still_prefers_canonical_X_header(self):
        """No regression: the primary, official regex path still wins when
        the model does produce a proper X: header."""
        response = "some chatter\nX:1\nT:Tune\nK:C\nCDEF|\n"
        abc = rm._extract_abc(response)
        self.assertTrue(abc.startswith("X:1\nT:Tune"))

    def test_field_without_M_or_K_still_raises(self):
        """No regression: a bare field with neither M: nor K: is still not
        usable ABC (abc2midi cannot determine pitches without a key) and
        must still raise, exactly like the documented test10 case where the
        model's whole answer was just 'A: G2 AB cBAG' with nothing else."""
        response = "A: G2 AB cBAG\n"
        with self.assertRaises(ValueError):
            rm._extract_abc(response)

    def test_pure_prose_still_raises(self):
        response = "I think a good chord progression would be C-G-Am-F."
        with self.assertRaises(ValueError):
            rm._extract_abc(response)


class TestTruncateSeedAbc(TestCase):

    def test_short_text_untouched(self):
        text = "X:1\nK:C\nCDEF GABc|\n"
        self.assertEqual(rm._truncate_seed_abc(text, max_chars=1000), text)

    def test_long_text_truncated_to_line_boundary(self):
        lines = [f"line{i} some notes here" for i in range(500)]
        text = "\n".join(lines)
        truncated = rm._truncate_seed_abc(text, max_chars=200)
        self.assertLessEqual(len(truncated), 200)
        # Must end exactly at a line boundary — no half-written line/note.
        self.assertIn(truncated, text)
        self.assertTrue(text.startswith(truncated))
        self.assertFalse(truncated.endswith("\n"))

    def test_default_cap_is_reasonable(self):
        """Regression guard on the actual cap used in production: it must
        be generous enough for a short melody but finite."""
        self.assertGreater(rm._MAX_SEED_ABC_CHARS, 500)
        self.assertLess(rm._MAX_SEED_ABC_CHARS, 10000)


class TestBuildInstructionCapsSeedLength(TestCase):
    """End-to-end (no GPU): a pathologically long seed .mid must not blow
    up the instruction sent to the model."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cm_seed_len_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_long_abc_seed(self) -> Path:
        # An .abc/.txt seed bypasses the local midi2abc dependency entirely,
        # letting this test run without abcmidi installed while still
        # exercising _build_instruction's truncation for the .abc branch.
        p = self.tmpdir / "input_abc.txt"
        lines = ["X:1", "T:Whole song", "M:4/4", "L:1/8", "K:C"]
        # ~200 bars worth of notes — a whole-song-length seed, matching the
        # "user attaches an entire track" scenario panel.lua doesn't guard
        # against for ChatMusician (unlike the Anticipatory seed picker).
        lines += ["CDEF GABc | dcBA GFED |" for _ in range(200)]
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_long_abc_seed_instruction_is_capped(self):
        seed = self._make_long_abc_seed()
        raw_len = len(seed.read_text(encoding="utf-8"))
        self.assertGreater(raw_len, rm._MAX_SEED_ABC_CHARS,
                            "test fixture must exceed the cap to be meaningful")

        instruction = rm._build_instruction("Harmonize this.", str(seed))
        # instruction = f"{prompt}\n{abc_text}" — abc_text portion must respect the cap.
        abc_portion = instruction.split(rm._SEED_BRIDGE + "\n", 1)[1]
        self.assertLessEqual(len(abc_portion), rm._MAX_SEED_ABC_CHARS)

    def test_short_seed_is_not_truncated(self):
        p = self.tmpdir / "short.abc"
        p.write_text("X:1\nK:C\nCDEF GABc|\n", encoding="utf-8")
        instruction = rm._build_instruction("Harmonize.", str(p))
        self.assertIn("CDEF GABc|", instruction)

    def test_seed_present_inserts_bridge_between_prompt_and_abc(self):
        """Regression, verified empirically against a real Modal run
        (2026-09-14): a short prompt with no explicit reference to attached
        material ("simplify the verses") plus the seed just concatenated
        with a bare newline made the model answer with clarifying questions
        ("What is the last note of this music?") instead of generating —
        the official web demo's own working prompts always end by
        referencing "the provided/specified musical excerpt" explicitly.
        _build_instruction must insert that anchor itself so any user
        prompt wording works, not just ones that happen to reference the
        seed."""
        p = self.tmpdir / "short.abc"
        p.write_text("X:1\nK:C\nCDEF GABc|\n", encoding="utf-8")
        instruction = rm._build_instruction("simplify the verses", str(p))
        expected = f"simplify the verses\n{rm._SEED_BRIDGE}\nX:1\nK:C\nCDEF GABc|"
        self.assertEqual(instruction, expected)

    def test_no_seed_prompt_has_no_bridge(self):
        self.assertNotIn(rm._SEED_BRIDGE, rm._build_instruction("Just a prompt."))

    def test_missing_seed_falls_back_to_prompt(self):
        instruction = rm._build_instruction("Just a prompt.", str(self.tmpdir / "nope.mid"))
        self.assertEqual(instruction, "Just a prompt.")

    def test_no_seed_returns_prompt_unchanged(self):
        self.assertEqual(rm._build_instruction("Prompt only."), "Prompt only.")


class TestBuildInstructionPropagatesConversionFailure(TestCase):
    """Regression for the real user report: a seed .mid the user explicitly
    picked (in-project MIDI item, exported by smf_writer.lua) that FAILS to
    convert MIDI→ABC (midi2abc missing from PATH, or the take is corrupt)
    must abort loudly — not silently fall back to an unseeded prompt that
    then fails ABC extraction/abc2midi three stages later with a misleading
    "No .mid files found" error and no clue about the real cause."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cm_seed_fail_"))
        # Content is irrelevant — midi_to_abc_text is mocked below — but the
        # file must exist so _build_instruction reaches the .mid branch.
        self.seed = self.tmpdir / "seed.mid"
        self.seed.write_bytes(b"not a real midi file")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_conversion_failure_raises_instead_of_falling_back(self):
        with patch("tools.midi_abc.midi_to_abc_text",
                   side_effect=RuntimeError(
                       "midi2abc failed:\nno notes found")) as mock_convert:
            with self.assertRaises(RuntimeError):
                rm._build_instruction("Simplify this melody.", str(self.seed))
            mock_convert.assert_called_once_with(str(self.seed))

    def test_main_exits_1_with_error_line_and_never_calls_generate(self):
        """main() must catch the RuntimeError, print an "ERROR: ..." line,
        exit(1), and never spend a GPU call on a run it knows is broken."""
        with patch("tools.midi_abc.midi_to_abc_text",
                   side_effect=RuntimeError("midi2abc failed:\nno notes found")), \
             patch.object(rm, "generate") as mock_generate, \
             patch("builtins.print") as mock_print:
            with self.assertRaises(SystemExit) as ctx:
                rm.main(prompt="Simplify this melody.", input_file=str(self.seed),
                        out_dir=str(self.tmpdir))
            self.assertEqual(ctx.exception.code, 1)
            mock_generate.remote.assert_not_called()
            printed = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
            self.assertTrue(any(p.startswith("ERROR:") for p in printed),
                             f"no ERROR: line printed, got: {printed}")


class TestPromptAdapterIntegration(TestCase):
    """The bridge phrase alone doesn't fix a user prompt that's too far from
    ChatMusician's fine-tuned instruction style (verified empirically:
    "simplify the verses" still failed even with the bridge). _build_
    instruction now runs the seeded prompt through _adapt_prompt_for_seed
    first — a small, self-hosted LLM (Qwen2.5-1.5B-Instruct, see
    adapt_prompt) running in its own Modal container, no external API.
    These tests cover the integration and the fallback contract; adapt_
    prompt's own model-loading internals are exercised only at the Modal
    level (same boundary as generate()/_infer_one — not unit-tested here)."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cm_adapt_"))
        self.seed = self.tmpdir / "seed.abc"
        self.seed.write_text("X:1\nK:C\nCDEF GABc|\n", encoding="utf-8")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_build_instruction_uses_adapted_prompt(self):
        with patch.object(rm, "_adapt_prompt_for_seed",
                           return_value="Add chords to the following musical excerpt.") as mock_adapt:
            instruction = rm._build_instruction("simplify the verses", str(self.seed))
        mock_adapt.assert_called_once_with("simplify the verses")
        self.assertTrue(instruction.startswith("Add chords to the following musical excerpt.\n"))
        self.assertNotIn("simplify the verses", instruction)

    def test_no_seed_never_calls_adapter(self):
        with patch.object(rm, "_adapt_prompt_for_seed") as mock_adapt:
            rm._build_instruction("simplify the verses")
        mock_adapt.assert_not_called()

    def test_adapter_container_failure_falls_back_to_original_prompt(self):
        with patch.object(rm.adapt_prompt, "remote",
                           side_effect=RuntimeError("container cold-start timeout")):
            result = rm._adapt_prompt_for_seed("simplify the verses")
        self.assertEqual(result, "simplify the verses")

    def test_adapter_empty_reply_falls_back_to_original_prompt(self):
        with patch.object(rm.adapt_prompt, "remote", return_value="   "):
            result = rm._adapt_prompt_for_seed("simplify the verses")
        self.assertEqual(result, "simplify the verses")

    def test_adapter_success_returns_rewritten_prompt(self):
        with patch.object(rm.adapt_prompt, "remote",
                           return_value="Add chord combinations to the following musical excerpt."):
            result = rm._adapt_prompt_for_seed("simplify the verses")
        self.assertEqual(result, "Add chord combinations to the following musical excerpt.")

    def test_end_to_end_fallback_still_produces_valid_instruction(self):
        """Even with the adapter container totally unavailable, a seeded
        request must still produce a well-formed instruction (original
        prompt + bridge + ABC) — the adapter is a booster, never a hard
        dependency."""
        with patch.object(rm.adapt_prompt, "remote", side_effect=RuntimeError("no GPU available")):
            instruction = rm._build_instruction("simplify the verses", str(self.seed))
        expected = f"simplify the verses\n{rm._SEED_BRIDGE}\nX:1\nK:C\nCDEF GABc|"
        self.assertEqual(instruction, expected)


class TestGeneratePrintsSingleLineErrorPerCandidate(TestCase):
    """The per-candidate failure line printed inside generate() is the ONLY
    thing that can tell the user the real cause (ABC extraction failure,
    abc2midi failure, ...) once main() exits non-zero — midigen.py greps
    stdout for a line starting with "ERROR:" verbatim. A multi-line
    exception message (e.g. abc2midi's stderr, or _extract_abc's
    response-preview ValueError) must be collapsed onto that single line,
    or everything after the first embedded newline becomes an invisible,
    unprefixed line that midigen.py's capture ignores."""

    def test_multiline_exception_collapses_to_one_error_line(self):
        with patch.object(rm, "_load_model", return_value=(object(), object())), \
             patch.object(rm, "_infer_one",
                           side_effect=ValueError("No ABC notation found in response:\n"
                                                   "some chatter\nmore chatter")), \
             patch("builtins.print") as mock_print:
            rm.generate.local(["Simplify this melody.\nX:1\nK:C\n"], n_outputs=1)

        printed = [str(c.args[0]) for c in mock_print.call_args_list if c.args]
        error_lines = [p for p in printed if p.startswith("ERROR:")]
        self.assertEqual(len(error_lines), 1, f"expected exactly 1 ERROR: line, got: {printed}")
        self.assertNotIn("\n", error_lines[0])
        self.assertIn("No ABC notation found in response:", error_lines[0])
        self.assertIn("more chatter", error_lines[0])


class TestMainFailsWhenAllCandidatesEmpty(TestCase):
    """Regression for the follow-up report: the seed DOES convert fine (real
    ABC embedded in the instruction, visible in the log), but every
    generated candidate still fails downstream — bad ABC extraction from the
    model's response, or abc2midi failing inside the Modal container.
    generate() already catches those per-candidate exceptions (so one bad
    candidate doesn't sink the others) and logs each with its own
    "ERROR: ..." line, but main() used to report "Completado" and exit 0
    even when EVERY candidate came back empty — which is exactly what
    produced the misleading, unrelated "No .mid files found" several stages
    later with no way to see the real cause. main() must now treat
    0/n_outputs saved as a hard failure."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cm_all_empty_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_exits_1_when_every_candidate_is_empty(self):
        with patch.object(rm, "generate") as mock_generate:
            mock_generate.remote.return_value = [[(b"", ""), (b"", "")]]
            with self.assertRaises(SystemExit) as ctx:
                rm.main(prompt="Simplify this melody.", out_dir=str(self.tmpdir),
                        n_outputs=2)
            self.assertEqual(ctx.exception.code, 1)
        self.assertEqual(list(self.tmpdir.glob("*.mid")), [])

    def test_succeeds_when_at_least_one_candidate_produced_output(self):
        with patch.object(rm, "generate") as mock_generate:
            mock_generate.remote.return_value = [[(b"", ""), (b"fake midi", "X:1\nK:C\n")]]
            rm.main(prompt="Simplify this melody.", out_dir=str(self.tmpdir),
                    n_outputs=2)  # must not raise — one real candidate is enough
        self.assertTrue((self.tmpdir / "generated_cuda_v1.mid").exists())
        self.assertFalse((self.tmpdir / "generated_cuda_v0.mid").exists())


class TestMainRetriesAtHigherTemperatureOnTotalFailure(TestCase):
    """ChatMusician's model-card default (temperature=0.2) occasionally makes
    the model answer with prose/noise instead of ABC notation for a seeded
    harmonization instruction — this file's own docstring already documented
    that raising temperature to 0.5-0.7 improves the success rate for
    exactly this failure mode. main() should retry once automatically at a
    higher temperature rather than immediately give up, but only when the
    original attempt used a conservative (<=0.5) temperature — a user who
    already dialed up temperature and still got nothing gains nothing from
    an identical retry."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cm_retry_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_retries_once_at_higher_temperature_and_succeeds(self):
        with patch.object(rm, "generate") as mock_generate:
            mock_generate.remote.side_effect = [
                [[(b"", ""), (b"", "")]],
                [[(b"fake midi", "X:1\nK:C\n"), (b"", "")]],
            ]
            rm.main(prompt="Simplify this melody.", out_dir=str(self.tmpdir),
                    n_outputs=2, temperature=0.2)
        self.assertEqual(mock_generate.remote.call_count, 2)
        self.assertAlmostEqual(mock_generate.remote.call_args_list[1].kwargs["temperature"], 0.7)
        self.assertTrue((self.tmpdir / "generated_cuda_v0.mid").exists())

    def test_no_retry_when_original_temperature_already_high(self):
        with patch.object(rm, "generate") as mock_generate:
            mock_generate.remote.return_value = [[(b"", ""), (b"", "")]]
            with self.assertRaises(SystemExit):
                rm.main(prompt="Simplify this melody.", out_dir=str(self.tmpdir),
                        n_outputs=2, temperature=1.0)
        self.assertEqual(mock_generate.remote.call_count, 1)

    def test_no_retry_when_first_attempt_already_succeeds(self):
        with patch.object(rm, "generate") as mock_generate:
            mock_generate.remote.return_value = [[(b"fake midi", "X:1\nK:C\n"), (b"", "")]]
            rm.main(prompt="Simplify this melody.", out_dir=str(self.tmpdir),
                    n_outputs=2, temperature=0.2)
        self.assertEqual(mock_generate.remote.call_count, 1)

    def test_retry_also_fails_still_exits_1(self):
        with patch.object(rm, "generate") as mock_generate:
            mock_generate.remote.return_value = [[(b"", ""), (b"", "")]]
            with self.assertRaises(SystemExit) as ctx:
                rm.main(prompt="Simplify this melody.", out_dir=str(self.tmpdir),
                        n_outputs=2, temperature=0.2)
            self.assertEqual(ctx.exception.code, 1)
        self.assertEqual(mock_generate.remote.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
