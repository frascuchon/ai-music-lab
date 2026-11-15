#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACE-Step edit entrypoint: tasks, checkpoints and lego guard rails."""
import sys
from pathlib import Path
from unittest import TestCase, main

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_acestep_edit_modal as edit  # noqa: E402


class TestAceStepEdit(TestCase):
    def test_tasks_cover_the_three_modes(self):
        self.assertEqual(edit.EDIT_TASKS, ("cover", "repaint", "lego"))

    def test_track_names_match_acestep_constants(self):
        self.assertEqual(set(edit.TRACK_NAMES), {
            "woodwinds", "brass", "fx", "synth", "strings", "percussion",
            "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"})

    def test_every_checkpoint_has_a_default_step_count(self):
        for variant in edit.DIT_CONFIG_BY_VARIANT:
            self.assertIn(variant, edit.STEPS_DEFAULT_BY_VARIANT)

    def test_base_is_the_lego_checkpoint(self):
        self.assertEqual(edit.DIT_CONFIG_BY_VARIANT["base"], "acestep-v15-base")


if __name__ == "__main__":
    main(verbosity=2)
