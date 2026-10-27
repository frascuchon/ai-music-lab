#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for harmony.py (key/chord detection and rendering) and accompaniment.py
(input preparation and output cleanup for Anticipatory accompaniment).

Regression context: the REAPER panel writes the melody on channel 0 and seed
tracks on channels 1..N with no program changes, so every note was program 0 and
Anticipatory treated all of it as the melody it must follow: no accompaniment
history reached the model and it started cold. These tests pin the channel ->
program mapping and the automatic accompaniment prompt for melody-only input.
"""

import io
import sys
import unittest
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import mido
    import pretty_midi
except ImportError:  # pragma: no cover
    mido = pretty_midi = None

import harmony  # noqa: E402

N = lambda s, e, p, v=80: (s, e, p, v)


def c_major_melody():
    """4 bars of 4/4 at 120 BPM (beat = 0.5 s): C | F | G | C arpeggios."""
    bars = [[60, 64, 67, 72], [65, 69, 72, 77], [67, 71, 74, 79], [72, 67, 64, 60]]
    return [N(b * 2.0 + i * 0.5, b * 2.0 + i * 0.5 + 0.45, p)
            for b, bar in enumerate(bars) for i, p in enumerate(bar)]


class TestHarmony(TestCase):

    def test_detect_key_major_and_minor(self):
        scale = [N(i * 0.5, i * 0.5 + 0.4, p) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72, 60, 64, 67])]
        self.assertEqual(harmony.detect_key(scale), (0, "major"))
        a_minor = [N(i * 0.5, i * 0.5 + 0.4, p) for i, p in enumerate([69, 71, 72, 74, 76, 77, 79, 69, 72, 76, 69])]
        self.assertEqual(harmony.detect_key(a_minor)[1], "minor")

    def test_chords_follow_the_melody(self):
        r = harmony.harmonize(c_major_melody(), 120.0, "pop")
        names = [c.name for _, _, c in r["chords"]]
        self.assertEqual(names[0], "C")
        self.assertIn("F", names)
        self.assertIn("G", names)
        self.assertEqual(names[-1], "C")

    def test_non_diatonic_chord_is_possible(self):
        """The key is only a soft preference: a clear Bb arpeggio must win."""
        mel = [N(i * 0.5, i * 0.5 + 0.45, p) for i, p in enumerate([70, 74, 77, 70, 74, 77, 70, 74])]
        r = harmony.harmonize(mel, 120.0, "pop")
        self.assertEqual(r["chords"][0][2].name, "A#")

    def test_silent_window_keeps_the_previous_chord(self):
        mel = [N(0.0, 0.45, 60), N(0.5, 0.95, 64), N(8.0, 8.45, 60)]  # 7 s of silence
        r = harmony.harmonize(mel, 120.0, "pop")
        names = [c.name for _, _, c in r["chords"]]
        self.assertEqual(set(names), {"C"}, msg=names)   # one chord held through the rest

    def test_render_ranges_and_bounds_for_every_style(self):
        for style in harmony.STYLES:
            r = harmony.harmonize(c_major_melody(), 120.0, style)
            self.assertTrue(r["bass"] and r["comp"], msg=style)
            end = r["chords"][-1][1]
            for s, e, p, v in r["bass"]:
                self.assertTrue(30 <= p <= 56, msg=(style, p))
            for s, e, p, v in r["comp"]:
                self.assertTrue(48 <= p <= 76, msg=(style, p))
            for s, e, p, v in r["bass"] + r["comp"]:
                self.assertTrue(0 <= s < e <= end + 0.01, msg=(style, s, e))
                self.assertTrue(1 <= v <= 127)

    def test_every_rendered_note_belongs_to_its_chord(self):
        """Regression: voicings were built on octave bases not aligned to C, so the
        piano played F major (A-C-F) over a C# chord."""
        for style in harmony.STYLES:
            r = harmony.harmonize(c_major_melody(), 120.0, style)
            for t0, t1, chord in r["chords"]:
                for s, e, p, v in r["comp"]:
                    if t0 - 1e-6 <= s < t1 - 1e-6:
                        self.assertIn(p % 12, chord.pcs, msg=(style, chord.name, p))
                for s, e, p, v in r["bass"]:
                    if t0 - 1e-6 <= s < t1 - 1e-6:
                        self.assertIn(p % 12, chord.pcs, msg=(style, chord.name, p))

    def test_voicings_for_every_chord_keep_their_pitch_classes(self):
        for chord in harmony.ALL_CHORDS:
            for center in (50.0, 60.0, 70.0):
                voicing = harmony._closest_voicing(chord, center, 52, 72)
                self.assertEqual({p % 12 for p in voicing}, set(chord.pcs), msg=(chord.name, voicing))
                self.assertTrue(all(52 <= p <= 72 for p in voicing))

    def test_until_limits_the_accompaniment(self):
        r = harmony.harmonize(c_major_melody(), 120.0, "pop", until=3.0)
        self.assertTrue(all(s < 3.0 and e <= 3.0 for s, e, _, _ in r["bass"] + r["comp"]))

    def test_unknown_style_and_empty_melody_are_errors(self):
        with self.assertRaises(ValueError):
            harmony.harmonize(c_major_melody(), 120.0, "polka")
        with self.assertRaises(ValueError):
            harmony.harmonize([], 120.0)


def make_midi(channels, programs=None, bpm=120):
    """channels: {channel: [(start_s, end_s, pitch)]} -> SMF bytes, no program changes
    unless `programs` ({channel: program}) says so (like the REAPER panel's writer)."""
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    for ch, prog in (programs or {}).items():
        tr.append(mido.Message("program_change", channel=ch, program=prog, time=0))
    events = []
    for ch, notes in channels.items():
        for s, e, p in notes:
            events.append((s, 1, "note_on", ch, p, 90))
            events.append((e, 0, "note_off", ch, p, 0))
    events.sort()
    prev = 0
    for t, _, kind, ch, p, v in events:
        tick = int(round(t * 480 * bpm / 60))
        tr.append(mido.Message(kind, channel=ch, note=p, velocity=v, time=tick - prev))
        prev = tick
    mid.tracks.append(tr)
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def programs_by_channel(midi_bytes):
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    progs, has_notes = {}, set()
    for track in mid.tracks:
        for m in track:
            if m.type == "program_change":
                progs[m.channel] = m.program
            elif m.type == "note_on" and m.velocity > 0:
                has_notes.add(m.channel)
    return progs, has_notes


@unittest.skipUnless(mido and pretty_midi, "mido / pretty_midi not installed")
class TestPrepare(TestCase):

    def setUp(self):
        import accompaniment
        self.acc = accompaniment
        self.melody = [(4.0 + i * 0.5, 4.0 + i * 0.5 + 0.4, p) for i, p in enumerate([60, 64, 67, 72] * 6)]

    def _instruments(self, prepared):
        pm = pretty_midi.PrettyMIDI(io.BytesIO(prepared.midi))
        return {i.name: i for i in pm.instruments}

    def test_melody_only_gets_an_automatic_accompaniment_prompt(self):
        prep = self.acc.prepare(make_midi({0: self.melody}), 0, 5.0)
        inst = self._instruments(prep)
        self.assertTrue(prep.auto_history)
        self.assertIn("Bass", inst)
        self.assertIn("Chords", inst)
        self.assertEqual(inst["Melody"].program, 0)
        self.assertNotIn(inst["Bass"].program, (0,))
        self.assertNotEqual(inst["Bass"].program, inst["Chords"].program)
        # the prompt only covers the first prompt_length seconds
        self.assertTrue(all(n.start < 5.0 for n in inst["Bass"].notes + inst["Chords"].notes))

    def test_auto_history_can_be_disabled(self):
        prep = self.acc.prepare(make_midi({0: self.melody}), 0, 5.0, auto_history=False)
        self.assertFalse(prep.auto_history)
        self.assertEqual(list(self._instruments(prep)), ["Melody"])

    def test_leading_silence_is_trimmed(self):
        prep = self.acc.prepare(make_midi({0: self.melody}), 0, 5.0)
        self.assertAlmostEqual(prep.lead, 4.0, places=2)
        self.assertAlmostEqual(min(n.start for n in self._instruments(prep)["Melody"].notes), 0.0, places=2)

    def test_seed_tracks_never_share_the_melody_program(self):
        """The regression: seeds without program changes used to be program 0 = melody."""
        seed1 = [(4.0, 4.5, 40), (5.0, 5.5, 43)]
        seed2 = [(4.0, 6.0, 52), (6.0, 8.0, 55)]
        prep = self.acc.prepare(make_midi({0: self.melody, 1: seed1, 2: seed2}), 0, 5.0)
        inst = self._instruments(prep)
        seeds = [v for k, v in inst.items() if k.startswith("Seed")]
        self.assertEqual(len(seeds), 2)
        programs = {i.program for i in seeds}
        self.assertNotIn(0, programs)
        self.assertEqual(len(programs), 2)
        self.assertFalse(prep.auto_history, "real seed tracks must be used instead of an automatic prompt")

    def test_drums_stay_drums(self):
        drums = [(4.0, 4.1, 36), (4.5, 4.6, 38)]
        prep = self.acc.prepare(make_midi({0: self.melody, 9: drums}), 0, 5.0)
        self.assertTrue(self._instruments(prep)["Drums"].is_drum)

    def test_melody_program_is_forced_to_melody_instrument(self):
        prep = self.acc.prepare(make_midi({0: self.melody}, programs={0: 25}), 0, 5.0)
        self.assertEqual(self._instruments(prep)["Melody"].program, 0)

    def test_missing_melody_is_an_error(self):
        with self.assertRaises(ValueError):
            self.acc.prepare(make_midi({1: self.melody}), 0, 5.0)

    def test_finalize_drops_melody_and_restores_the_offset(self):
        prep = self.acc.prepare(make_midi({0: self.melody}), 0, 5.0)
        out = self.acc.finalize(prep.midi, prep.lead, 0)
        pm = pretty_midi.PrettyMIDI(io.BytesIO(out))
        names = [i.name for i in pm.instruments]
        self.assertEqual(len(pm.instruments), 2, msg=names)
        self.assertTrue(all(p.program != 0 for p in pm.instruments))
        self.assertAlmostEqual(min(n.start for i in pm.instruments for n in i.notes), 4.0, places=2)

    def test_finalize_trims_notes_after_the_melody_ends(self):
        prep = self.acc.prepare(make_midi({0: self.melody}), 0, 5.0)
        # fake model output with an accompaniment note far beyond the melody
        pm = pretty_midi.PrettyMIDI(io.BytesIO(prep.midi))
        bass = [i for i in pm.instruments if i.name == "Bass"][0]
        bass.notes.append(pretty_midi.Note(velocity=80, pitch=40, start=prep.melody_end + 3, end=prep.melody_end + 4))
        buf = io.BytesIO(); pm.write(buf)
        out = self.acc.finalize(buf.getvalue(), prep.lead, 0, trim_after=prep.melody_end + 0.5)
        last = max(n.start for i in pretty_midi.PrettyMIDI(io.BytesIO(out)).instruments for n in i.notes)
        self.assertLessEqual(last, prep.lead + prep.melody_end + 0.5)

    def test_finalize_can_keep_the_melody(self):
        prep = self.acc.prepare(make_midi({0: self.melody}), 0, 5.0)
        out = self.acc.finalize(prep.midi, prep.lead, 0, keep_melody=True)
        self.assertEqual(len(pretty_midi.PrettyMIDI(io.BytesIO(out)).instruments), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
