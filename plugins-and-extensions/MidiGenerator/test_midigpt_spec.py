#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for midigpt_spec.build_spec: the panel's controls -> MIDI-GPT generation spec.

The rules come from measurements on a real melody: a part with a hard rhythm
schedule must start at tick 0 with unique positions inside a bar (MIDI-GPT rejects
anything else), and two parts must never share a schedule or MIDI-GPT makes one a
transposed copy of the other.
"""

import itertools
import sys
import unittest
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midigpt_spec as ms  # noqa: E402


def schedule(track):
    return tuple((r["pos"], r["polyphony"]) for r in (track.get("rhythm") or ()))


class TestBuildSpec(TestCase):

    def test_default_is_bass_and_keys_with_distinct_rhythms(self):
        spec = ms.build_spec()
        self.assertEqual([t["role"] for t in spec["tracks"]], ["Bass", "Keys"])
        self.assertNotEqual(schedule(spec["tracks"][0]), schedule(spec["tracks"][1]))

    def test_no_two_parts_ever_share_a_rhythm(self):
        all_parts = [p for p in ms.PARTS if p != "drums"]
        for style, density in itertools.product(ms.STYLES, (0, 1, 2)):
            for n in range(2, len(all_parts) + 1):
                for combo in itertools.combinations(all_parts, n):
                    spec = ms.build_spec(parts=combo, style=style, density=density)
                    scheds = [schedule(t) for t in spec["tracks"]]
                    self.assertEqual(len(set(scheds)), len(scheds), msg=(style, density, combo))

    def test_rhythms_satisfy_midigpt_constraints(self):
        for style, density, part in itertools.product(ms.STYLES, (0, 1, 2), ("bass", "keys", "strings", "guitar")):
            rhythm = ms.rhythm_for(part, style, density)
            positions = [r["pos"] for r in rhythm]
            self.assertEqual(positions[0], 0, msg=(style, density, part))        # every bar opens at tick 0
            self.assertEqual(len(set(positions)), len(positions))
            self.assertTrue(all(0 <= p < 48 for p in positions))
            self.assertTrue(all(r["polyphony"] >= 1 for r in rhythm))
            self.assertEqual(positions, sorted(positions))

    def test_density_changes_the_amount_of_notes(self):
        def onsets(part, density):
            return len(ms.rhythm_for(part, "pop", density))
        self.assertLess(onsets("bass", 0), onsets("bass", 2))
        self.assertLess(onsets("guitar", 0), onsets("guitar", 2))
        self.assertLess(onsets("keys", 0), onsets("keys", 2))

    def test_drums_use_the_density_attribute_not_a_rhythm_or_register(self):
        spec = ms.build_spec(parts=("bass", "drums"), density=2)
        drums = spec["tracks"][-1]
        self.assertEqual(drums["type"], "drum")
        self.assertEqual(drums["attributes"], {"note_density": ms.DRUM_DENSITY[2]})
        self.assertNotIn("rhythm", drums)
        self.assertNotIn("register", drums)
        levels = [ms.build_spec(parts=("drums",), density=d)["tracks"][0]["attributes"]["note_density"]
                  for d in (0, 1, 2)]
        self.assertEqual(levels, sorted(levels))

    def test_registers_keep_bass_below_keys(self):
        spec = ms.build_spec(parts=("bass", "keys", "strings", "guitar"))
        reg = {t["role"]: t["register"] for t in spec["tracks"]}
        self.assertLess(reg["Bass"][1], reg["Keys"][0] + 1)
        self.assertTrue(all(lo >= 33 for lo, hi in reg.values()))

    def test_programs_can_be_overridden(self):
        spec = ms.build_spec(parts=("bass", "keys"), programs={"bass": 32, "keys": 16})
        self.assertEqual([t["program"] for t in spec["tracks"]], [32, 16])

    def test_creativity_maps_to_temperature_inside_the_range(self):
        low, high = ms.TEMPERATURE_RANGE
        self.assertAlmostEqual(ms.build_spec(creativity=0.0)["temperature"], low)
        self.assertAlmostEqual(ms.build_spec(creativity=1.0)["temperature"], high)
        self.assertLess(ms.build_spec(creativity=0.2)["temperature"], ms.build_spec(creativity=0.8)["temperature"])

    def test_stay_in_key_flag_is_passed_through(self):
        self.assertTrue(ms.build_spec(stay_in_key=True)["pitch_classes"])
        self.assertFalse(ms.build_spec()["pitch_classes"])

    def test_bad_input_is_a_clear_error(self):
        for kwargs in ({"style": "polka"}, {"density": 5}, {"parts": ("kazoo",)}, {"parts": ()},
                       {"creativity": 1.5}):
            with self.assertRaises(ValueError, msg=kwargs):
                ms.build_spec(**kwargs)

    def test_spec_is_json_serialisable(self):
        import json
        json.dumps(ms.build_spec(parts=ms.PARTS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
