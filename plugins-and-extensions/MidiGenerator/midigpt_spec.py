#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate the panel's accompaniment controls into a MIDI-GPT generation spec.

Pure Python (no midigpt, no numpy) so it can run in REAPER's own Python and be
unit-tested. The spec is a JSON-able dict consumed by research_midigpt_modal.py:

    {"temperature": 1.0, "pitch_classes": False,
     "tracks": [{"role": "Bass", "program": 33, "register": [33, 52],
                 "rhythm": [{"pos": 0, "polyphony": 1}, ...],   # onsets inside a bar
                 "attributes": {...}},                         # MIDI-GPT track attributes
                ...]}

Why every part gets its OWN rhythm: measured on a guitar melody, when one track
has a rhythm constraint and the other does not (or two tracks are free), MIDI-GPT
tends to copy the other track's onsets and the parts end up doubling each other.
Distinct rhythms per part make them complement each other.

Rhythm positions are in the checkpoint's own ticks: 12 per quarter note, so a 4/4
bar has 48 ticks (beats at 0, 12, 24, 36). Time signatures other than 4/4 are not
handled here.
"""

from __future__ import annotations

PARTS = ("bass", "keys", "strings", "guitar", "drums")
STYLES = ("pop", "pad", "arp")

DEFAULT_PROGRAMS = {"bass": 33, "keys": 4, "strings": 48, "guitar": 25, "drums": 0}
REGISTERS = {"bass": (33, 52), "keys": (52, 76), "strings": (55, 79), "guitar": (40, 72)}
ROLE_NAMES = {"bass": "Bass", "keys": "Keys", "strings": "Strings", "guitar": "Guitar", "drums": "Drums"}

# note_density levels of MIDI-GPT's drum track (0-9) for sparse / normal / busy
DRUM_DENSITY = {0: 3, 1: 5, 2: 8}

# creativity 0..1 -> sampling temperature. Below ~0.8 the notes get short and choppy
# (measured: 0.7 gave 0.4 s notes vs 1.0 s at 1.0), so the range stays moderate.
TEMPERATURE_RANGE = (0.8, 1.25)


def _positions(*ticks_poly):
    return [{"pos": int(p), "polyphony": int(n)} for p, n in ticks_poly]


def _eighths(poly: int):
    return _positions(*[(t, poly) for t in range(0, 48, 6)])


def rhythm_for(part: str, style: str, density: int):
    """Onset schedule inside a 4/4 bar for one part (None = let the model decide)."""
    if part == "bass":
        if density == 0 or style == "pad":
            return _positions((0, 1))
        if density == 2:
            return _eighths(1)
        return _positions((0, 1), (24, 1)) if style == "pop" else _positions((0, 1), (12, 1), (24, 1), (36, 1))
    if part == "keys":
        if style == "pad" or density == 0:
            return _positions((0, 3), (24, 2))
        if style == "arp" or density == 2:
            return _eighths(1)
        return _positions((0, 3), (18, 2), (24, 3), (42, 2))            # pop stabs
    if part == "strings":
        return _positions((0, 3))                                        # sustained pad
    if part == "guitar":
        if density == 0:
            return _positions((0, 3), (24, 3))
        if density == 2:
            return _positions(*[(t, 3) for t in (0, 12, 18, 24, 36, 42)])
        return _positions((0, 3), (18, 3), (24, 3), (42, 3))            # syncopated strum
    return None                                                          # drums: density attribute


def build_spec(parts=("bass", "keys"), style: str = "pop", density: int = 1,
               programs: dict | None = None, creativity: float = 0.5,
               stay_in_key: bool = False) -> dict:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose one of {STYLES}")
    if density not in (0, 1, 2):
        raise ValueError("density must be 0 (sparse), 1 (normal) or 2 (busy)")
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise ValueError(f"unknown part(s) {unknown}; choose from {PARTS}")
    if not parts:
        raise ValueError("choose at least one part")
    if not 0.0 <= creativity <= 1.0:
        raise ValueError("creativity must be between 0 and 1")
    programs = programs or {}

    tracks = []
    for part in PARTS:                                  # fixed order: bass, keys, strings, guitar, drums
        if part not in parts:
            continue
        if part == "drums":
            tracks.append({"role": "Drums", "type": "drum", "program": programs.get("drums", 0),
                           "attributes": {"note_density": DRUM_DENSITY[density]}})
            continue
        lo, hi = REGISTERS[part]
        tracks.append({"role": ROLE_NAMES[part], "program": programs.get(part, DEFAULT_PROGRAMS[part]),
                       "register": [lo, hi], "rhythm": rhythm_for(part, style, density), "attributes": {}})

    # two parts must never share the same schedule (they would copy each other)
    seen = {}
    for t in tracks:
        key = tuple((r["pos"], r["polyphony"]) for r in (t.get("rhythm") or ()))
        if key and key in seen:
            # nudge the later part: drop its last onset (keeps it playable, makes it different)
            t["rhythm"] = t["rhythm"][:-1] if len(t["rhythm"]) > 1 else _positions((0, 2))
            key = tuple((r["pos"], r["polyphony"]) for r in t["rhythm"])
        if key:
            seen[key] = t["role"]

    low, high = TEMPERATURE_RANGE
    return {"temperature": round(low + (high - low) * creativity, 3),
            "pitch_classes": bool(stay_in_key), "tracks": tracks}
