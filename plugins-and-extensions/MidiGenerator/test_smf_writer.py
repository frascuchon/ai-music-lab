#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for smf_writer.lua (write_midi_from_take / write_combined_midi) —
the Lua SMF exporter used for the ChatMusician "Harmonize seed" and the
Anticipatory melody/accompaniment seed when the source is an in-project
MIDI take rather than a .mid file on disk (panel.lua's _export_take_to_tmp
and the AMT seed-combining path).

Runs gen_test_smf.lua (a REAPER-API mock + harness, see that file) via the
system `lua` interpreter, then validates the resulting .mid with mido/
pretty_midi — the same real MIDI parsers ChatMusician's local pipeline
(tools/midi_abc.py -> midi2abc) and REAPER's InsertMedia ultimately rely on.

Requires the Lua interpreter (`lua`) on PATH; skips otherwise.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

HERE = Path(__file__).resolve().parent
GEN_SCRIPT = HERE / "gen_test_smf.lua"

_HAS_LUA = shutil.which("lua") is not None


def run_gen(case: str, bpm: float, out_path: Path, item_pos: float = 0.0):
    result = subprocess.run(
        ["lua", str(GEN_SCRIPT), case, str(bpm), str(out_path), str(item_pos)],
        capture_output=True, text=True, cwd=str(HERE), timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gen_test_smf.lua failed:\n{result.stdout}\n{result.stderr}")


@unittest.skipUnless(_HAS_LUA, "lua interpreter not found on PATH")
class TestWriteMidiFromTake(TestCase):
    """The 'simple' case in gen_test_smf.lua encodes 3 known-time notes:
    (0.5, 1.0, 60), (1.5, 2.25, 64), (3.0, 3.5, 67) — as absolute project
    seconds. write_midi_from_take must reproduce them, normalized to the
    take's own start (item_pos)."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="smf_writer_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _notes(self, out_path):
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(out_path))
        self.assertEqual(len(pm.instruments), 1)
        return sorted(pm.instruments[0].notes, key=lambda n: n.start)

    def test_item_at_project_start(self):
        out_path = self.tmpdir / "out.mid"
        run_gen("simple", 120.0, out_path, item_pos=0.0)
        notes = self._notes(out_path)
        expected = [(0.5, 1.0, 60), (1.5, 2.25, 64), (3.0, 3.5, 67)]
        self.assertEqual(len(notes), len(expected))
        for (exp_s, exp_e, exp_p), got in zip(expected, notes):
            self.assertAlmostEqual(got.start, exp_s, delta=0.01)
            self.assertAlmostEqual(got.end, exp_e, delta=0.01)
            self.assertEqual(got.pitch, exp_p)

    def test_item_offset_in_timeline_normalizes_to_item_start(self):
        """An in-project take sitting at e.g. 10.5s into the timeline must
        still export as if it started at t=0 — write_midi_from_take
        subtracts t_item_start for exactly this reason."""
        out_path = self.tmpdir / "out.mid"
        run_gen("simple", 120.0, out_path, item_pos=10.5)
        notes = self._notes(out_path)
        expected = [(0.5, 1.0, 60), (1.5, 2.25, 64), (3.0, 3.5, 67)]
        for (exp_s, exp_e, exp_p), got in zip(expected, notes):
            self.assertAlmostEqual(got.start, exp_s, delta=0.01)
            self.assertAlmostEqual(got.end, exp_e, delta=0.01)

    def test_tempo_embedded_matches_project_tempo(self):
        out_path = self.tmpdir / "out.mid"
        run_gen("simple", 96.0, out_path)
        import mido
        mid = mido.MidiFile(str(out_path))
        tempo_msgs = [m for t in mid.tracks for m in t if m.is_meta and m.type == "set_tempo"]
        self.assertEqual(len(tempo_msgs), 1)
        self.assertAlmostEqual(mido.tempo2bpm(tempo_msgs[0].tempo), 96.0, places=1)

    def test_output_is_parseable_abc_via_midi2abc(self):
        """Integration guard for the actual ChatMusician seed pipeline: the
        exported take must survive a real midi2abc pass (not just mido's
        lenient parser) and produce ABC with a key field, since abc2midi
        cannot determine pitches without one."""
        if not shutil.which("midi2abc"):
            self.skipTest("midi2abc not installed (brew/apt install abcmidi)")
        out_path = self.tmpdir / "out.mid"
        run_gen("simple", 120.0, out_path)
        abc_out = self.tmpdir / "out.abc"
        result = subprocess.run(
            ["midi2abc", str(out_path), "-o", str(abc_out)],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        abc_text = abc_out.read_text()
        self.assertIn("K:", abc_text)


@unittest.skipUnless(_HAS_LUA, "lua interpreter not found on PATH")
class TestWriteMidiFromTakeRejectsEmptyTake(TestCase):
    """Regression for a real user report: a seed exported from a REAPER item
    with zero MIDI notes (wrong item/track selected) used to be written as a
    'successful' but content-free .mid, which panel.lua then logged as
    'Seed exported: ...' — the ChatMusician run would later fail with a
    misleading 'No .mid files found' error with no hint the seed was empty.
    write_midi_from_take must now reject this at export time instead."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="smf_writer_empty_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_take_reports_no_notes_error(self):
        out_path = self.tmpdir / "out.mid"
        result = subprocess.run(
            ["lua", str(GEN_SCRIPT), "empty", "120.0", str(out_path), "0.0"],
            capture_output=True, text=True, cwd=str(HERE), timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("EMPTY_TAKE_ERROR: Selected item has no MIDI notes",
                       result.stdout)
        self.assertFalse(out_path.exists(),
                          "no file should be written for a rejected empty take")


@unittest.skipUnless(_HAS_LUA, "lua interpreter not found on PATH")
class TestWriteCombinedMidi(TestCase):
    """The 'combined' case puts the melody on channel 0 and one seed take
    on channel 1 (Anticipatory accompaniment path)."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="smf_writer_combined_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_two_channels_present_with_correct_notes(self):
        out_path = self.tmpdir / "combined.mid"
        run_gen("combined", 120.0, out_path)
        import mido
        mid = mido.MidiFile(str(out_path))
        channels_seen = {
            m.channel for t in mid.tracks for m in t
            if m.type in ("note_on", "note_off")
        }
        self.assertEqual(channels_seen, {0, 1})

        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(out_path))
        all_notes = sorted(
            ((n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes),
            key=lambda t: t[0],
        )
        # melody (channel 0): (0.5,1.0,60), (1.5,2.25,64), (3.0,3.5,67)
        # seed (channel 1):   (0.5,1.5,48), (2.0,3.0,52)
        self.assertEqual(len(all_notes), 5)
        pitches = sorted(p for _, _, p in all_notes)
        self.assertEqual(pitches, [48, 52, 60, 64, 67])


if __name__ == "__main__":
    unittest.main(verbosity=2)
