#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for lib/midi_import.lua — the explicit MIDI-to-tracks importer that
replaced reaper.InsertMedia in Audio2Midi and MidiGenerator.

Regression context: with MIDI clips already in the project, importing a
multi-track MIDI produced a folder that contained those pre-existing clips and
mixed/overwrote them, because the panels inferred which tracks InsertMedia had
created. The importer now creates every track and item itself, so the tests run
it against a mocked project that already holds tracks with clips and check that
none of them is moved, renamed, wrapped or modified.

Requires the Lua interpreter (`lua`) and mido; skips otherwise.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

HARNESS = Path(__file__).resolve().parent / "lib" / "test_harness_midi_import.lua"
_HAS_LUA = shutil.which("lua") is not None

try:
    import mido
except ImportError:  # pragma: no cover
    mido = None


def make_midi(path, instruments, bpm=100, tempo_changes=()):
    """instruments: list of (name, channel, program, [(start_tick, dur_ticks, pitch)])."""
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    last = 0
    for tick, new_bpm in sorted(tempo_changes):
        tempo.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(new_bpm), time=tick - last))
        last = tick
    mid.tracks.append(tempo)
    for name, ch, prog, notes in instruments:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if ch != 9:
            tr.append(mido.Message("program_change", channel=ch, program=prog, time=0))
        events = []
        for start, dur, pitch in notes:
            events.append((start, "note_on", pitch, 90))
            events.append((start + dur, "note_off", pitch, 0))
        events.sort(key=lambda e: (e[0], e[1] == "note_on"))
        t = 0
        for tick, kind, pitch, vel in events:
            tr.append(mido.Message(kind, channel=ch, note=pitch, velocity=vel, time=tick - t))
            t = tick
        mid.tracks.append(tr)
    mid.save(str(path))


def run_import(mid_path, existing, insert_at, always_folder=False, position=0.0):
    out = subprocess.run(
        ["lua", str(HARNESS), str(mid_path), str(existing), str(insert_at),
         "1" if always_folder else "0", str(position)],
        capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    result, tracks = None, []
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if parts[0] == "R":
            result = dict(first=int(parts[1]), count=int(parts[2]),
                          instruments=int(parts[3]), next=int(parts[4]))
        elif parts[0] == "ERR":
            result = dict(error=parts[1])
        elif parts[0] == "T":
            tracks.append(dict(
                idx=int(parts[1]), name=parts[2], depth=int(parts[3]), items=int(parts[4]),
                notes=int(parts[5]), ccs=int(parts[6]), pos=float(parts[7]), length=float(parts[8]),
                first_ppq=parts[9], first_pitch=parts[10]))
    return result, tracks


@unittest.skipUnless(_HAS_LUA and mido, "lua interpreter or mido not available")
class TestImportDoesNotTouchExistingTracks(TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="midi_import_"))
        self.multi = self.tmp / "multi.mid"
        make_midi(self.multi, [
            ("Guitar (clean)", 0, 24, [(0, 240, 60), (480, 240, 62), (960, 240, 64)]),
            ("Bass", 1, 33, [(0, 480, 40), (960, 480, 43)]),
            ("Drums", 9, 0, [(0, 120, 36), (480, 120, 38)]),
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_existing_untouched(self, tracks, removed_before, existing_total):
        """Every 'exist k' track still has exactly its own single clip/note and no folder state."""
        found = [t for t in tracks if t["name"].startswith("exist ")]
        self.assertEqual(len(found), existing_total)
        for t in found:
            self.assertEqual(t["depth"], 0, msg=f"{t['name']} became part of a folder")
            self.assertEqual((t["items"], t["notes"], t["ccs"]), (1, 1, 0),
                             msg=f"{t['name']} was modified")
            self.assertEqual(t["first_pitch"], "1")

    def test_mid_project_insert_wraps_only_the_new_tracks(self):
        res, tracks = run_import(self.multi, existing=5, insert_at=2)
        self.assertEqual((res["first"], res["count"], res["instruments"], res["next"]), (2, 4, 3, 6))
        self.assertEqual([t["name"] for t in tracks[:2]], ["exist 1", "exist 2"])
        folder, a, b, c = tracks[2:6]
        self.assertEqual((folder["name"], folder["depth"], folder["items"]), ("NEW", 1, 0))
        self.assertEqual([a["name"], b["name"], c["name"]], ["Guitar (clean)", "Bass", "Drums"])
        self.assertEqual([a["depth"], b["depth"], c["depth"]], [0, 0, -1])
        self.assertEqual([t["name"] for t in tracks[6:]], ["exist 3", "exist 4", "exist 5"])
        self._assert_existing_untouched(tracks, 0, 5)

    def test_insert_at_end(self):
        res, tracks = run_import(self.multi, existing=3, insert_at=3)
        self.assertEqual((res["first"], res["count"], res["next"]), (3, 4, 7))
        self.assertEqual(len(tracks), 7)
        self.assertEqual([t["depth"] for t in tracks[3:]], [1, 0, 0, -1])
        self._assert_existing_untouched(tracks, 0, 3)

    def test_insert_at_start(self):
        res, tracks = run_import(self.multi, existing=2, insert_at=0)
        self.assertEqual(tracks[0]["depth"], 1)
        self.assertEqual([t["name"] for t in tracks[4:]], ["exist 1", "exist 2"])
        self._assert_existing_untouched(tracks, 0, 2)

    def test_insert_index_beyond_track_count_is_clamped(self):
        res, tracks = run_import(self.multi, existing=2, insert_at=99)
        self.assertEqual(res["first"], 2)
        self._assert_existing_untouched(tracks, 0, 2)

    def test_importing_twice_keeps_both_results_separate(self):
        """A second import below the first must not merge into or overwrite it."""
        res, tracks = run_import(self.multi, existing=1, insert_at=1)
        res2, _ = run_import(self.multi, existing=1 + res["count"], insert_at=res["next"])
        self.assertEqual(res2["first"], res["next"])

    def test_single_instrument_is_one_track_without_folder(self):
        single = self.tmp / "single.mid"
        make_midi(single, [("Transcription", 0, 25, [(0, 240, 60), (480, 240, 64)])])
        res, tracks = run_import(single, existing=4, insert_at=1)
        self.assertEqual((res["count"], res["instruments"]), (1, 1))
        new = tracks[1]
        self.assertEqual((new["name"], new["depth"], new["items"], new["notes"]), ("NEW", 0, 1, 2))
        self._assert_existing_untouched(tracks, 0, 4)

    def test_always_folder_wraps_a_single_instrument(self):
        single = self.tmp / "single.mid"
        make_midi(single, [("Piano", 0, 0, [(0, 240, 60)])])
        res, tracks = run_import(single, existing=2, insert_at=2, always_folder=True)
        self.assertEqual(res["count"], 2)
        self.assertEqual([t["depth"] for t in tracks[2:]], [1, -1])

    def test_notes_land_at_the_right_time_and_item_starts_at_position(self):
        # 100 BPM: one beat (480 ticks) = 0.6 s; project position 8 s
        res, tracks = run_import(self.multi, existing=0, insert_at=0, position=8.0)
        guitar = tracks[1]
        self.assertEqual((guitar["notes"], guitar["first_pitch"]), (3, "60"))
        self.assertAlmostEqual(float(guitar["first_ppq"]), 8.0 * 1920, places=2)
        self.assertAlmostEqual(guitar["pos"], 8.0, places=3)
        self.assertGreater(guitar["length"], 1.5)

    def test_program_change_is_set_except_for_drums(self):
        res, tracks = run_import(self.multi, existing=0, insert_at=0)
        guitar, bass, drums = tracks[1], tracks[2], tracks[3]
        self.assertEqual((guitar["ccs"], bass["ccs"], drums["ccs"]), (1, 1, 0))

    def test_tempo_change_is_converted_to_seconds(self):
        # 120 BPM for the first beat (0.5 s), then 60 BPM; note at tick 960
        # starts at 0.5 + 1.0 = 1.5 s -> PPQ 1.5 * 1920 in the mock.
        f = self.tmp / "tempo.mid"
        make_midi(f, [("X", 0, 0, [(960, 240, 60)])], bpm=120, tempo_changes=[(480, 60)])
        res, tracks = run_import(f, existing=0, insert_at=0)
        self.assertAlmostEqual(float(tracks[0]["first_ppq"]), 1.5 * 1920, places=1)

    def test_file_without_notes_is_an_error_and_adds_no_tracks(self):
        f = self.tmp / "empty.mid"
        make_midi(f, [("Empty", 0, 0, [])])
        res, tracks = run_import(f, existing=3, insert_at=1)
        self.assertIn("error", res)
        self.assertEqual(len(tracks), 3)
        self._assert_existing_untouched(tracks, 0, 3)

    def test_running_status_and_note_on_zero_velocity(self):
        # MThd, one track: note-on, then running-status note-on vel 0 as note-off.
        trk = bytes([0x00, 0x90, 60, 100, 0x60, 60, 0, 0x00, 0xFF, 0x2F, 0x00])
        data = (b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
                + (1).to_bytes(2, "big") + (96).to_bytes(2, "big")
                + b"MTrk" + len(trk).to_bytes(4, "big") + trk)
        f = self.tmp / "running.mid"
        f.write_bytes(data)
        res, tracks = run_import(f, existing=0, insert_at=0)
        self.assertEqual((res["instruments"], tracks[0]["notes"]), (1, 1))

    def test_non_midi_file_is_an_error(self):
        f = self.tmp / "bad.mid"
        f.write_bytes(b"not a midi file at all")
        res, tracks = run_import(f, existing=2, insert_at=0)
        self.assertIn("error", res)
        self.assertEqual(len(tracks), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
