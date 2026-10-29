#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input/output handling for Anticipatory accompaniment, so a melody alone is
enough.

Anticipatory (AMT) tells instruments apart by MIDI *program*, and in accompaniment
mode it moves every note of `melody_instrument` (program 0 by default) into the
"controls" it must follow. It also needs a few seconds of accompaniment before it
can continue ("prompt"); without them it starts cold and writes almost nothing.

The REAPER panel writes the melody on channel 0 and each seed track on channels
1..N without program changes, i.e. everything is program 0 and everything ends up
as a control: the model never sees any accompaniment. This module fixes that:

  prepare():  channel 0 is the melody (program = melody_instrument); every other
              channel becomes a distinct non-melody instrument; leading silence is
              trimmed; and if there is no accompaniment in the first
              `prompt_length` seconds one is built from the melody with harmony.py.
  finalize(): drops the melody from the model output (the user already has it),
              restores the leading silence and names the parts.

Needs mido and pretty_midi (both are in the Modal image).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import harmony


@dataclass
class Prepared:
    midi: bytes            # input for AMT (melody on melody_instrument, history, no leading silence)
    lead: float            # seconds of leading silence removed from the melody
    melody_end: float      # end of the melody after trimming
    bpm: float
    auto_history: bool     # True if the accompaniment prompt was generated here
    harmony: str           # description of the chords used for the prompt ("" if none)


def read_channels(midi_bytes: bytes):
    """-> ({channel: [(start, end, pitch, vel)]}, {channel: first program}, bpm).
    Times are in seconds, honouring the file's tempo map."""
    import mido

    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    bpm = 120.0
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta and msg.type == "set_tempo":
                bpm = mido.tempo2bpm(msg.tempo)
                break
        else:
            continue
        break

    notes, programs, open_ = {}, {}, {}
    now = 0.0
    for msg in mid:                    # iterating a MidiFile yields delta times in seconds
        now += msg.time
        if msg.is_meta:
            continue
        if msg.type == "program_change":
            programs.setdefault(msg.channel, msg.program)
        elif msg.type == "note_on" and msg.velocity > 0:
            open_.setdefault((msg.channel, msg.note), []).append((now, msg.velocity))
        elif msg.type in ("note_off", "note_on"):
            stack = open_.get((msg.channel, msg.note))
            if stack:
                start, vel = stack.pop(0)
                notes.setdefault(msg.channel, []).append((start, max(now, start + 0.01), msg.note, vel))
    for (ch, pitch), stack in open_.items():
        for start, vel in stack:
            notes.setdefault(ch, []).append((start, now, pitch, vel))
    for ch in notes:
        notes[ch].sort()
    return notes, programs, bpm


def _unused_program(taken: set, start: int) -> int:
    p = start % 128
    while p in taken:
        p = (p + 1) % 128
    taken.add(p)
    return p


def prepare(midi_bytes: bytes, melody_instrument: int = 0, prompt_length: float = 5.0,
            auto_history: bool = True, style: str = "pop",
            arrangement: "harmony.Arrangement | None" = None) -> Prepared:
    """`arrangement` (parts, density, instruments, own chords) shapes the automatic prompt;
    without it the prompt is bass + keys in `style`."""
    import pretty_midi

    chans, programs, bpm = read_channels(midi_bytes)
    melody = chans.get(0)
    if not melody:
        raise ValueError("no melody found: the melody must be on MIDI channel 0")

    lead = melody[0][0]
    shifted = lambda notes: [(max(s - lead, 0.0), e - lead, p, v) for s, e, p, v in notes if e - lead > 0]
    melody = shifted(melody)
    melody_end = max(n[1] for n in melody)

    taken = {melody_instrument}
    parts = []      # (name, program, is_drum, notes)
    for ch in sorted(c for c in chans if c != 0):
        notes = shifted(chans[ch])
        if not notes:
            continue
        if ch == 9:
            parts.append(("Drums", 0, True, notes))
            continue
        wanted = programs.get(ch)
        program = wanted if (wanted is not None and wanted not in taken) else _unused_program(taken, 1 + ch)
        taken.add(program)
        parts.append((f"Seed {ch}", program, False, notes))

    has_history = any(n[0] < prompt_length for _, _, _, notes in parts for n in notes)
    generated = False
    description = ""
    if not has_history and auto_history:
        arr = arrangement or harmony.Arrangement(style=style)
        result = harmony.harmonize(melody, bpm, arr.style, until=prompt_length, arrangement=arr)
        description = harmony.describe(result)
        for name, (notes, program, is_drum) in result["parts"].items():
            if not is_drum:
                program = _unused_program(taken, program)
            parts.append((harmony.PART_NAMES[name], program, is_drum, notes))
        generated = True

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for name, program, is_drum, notes in [("Melody", melody_instrument, False, melody)] + parts:
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
        for s, e, p, v in notes:
            if e > s:
                inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p), start=s, end=e))
        if inst.notes:
            pm.instruments.append(inst)

    buf = io.BytesIO()
    pm.write(buf)
    return Prepared(buf.getvalue(), lead, melody_end, bpm, generated, description)


def finalize(output_midi: bytes, lead: float, melody_instrument: int = 0,
             keep_melody: bool = False, trim_after: float | None = None) -> bytes:
    """Drop the melody (unless asked), restore the leading silence and name the parts.
    `trim_after` (prepared-time seconds) drops notes that start after it and clips
    the ones that end after it, so the accompaniment does not outlast the melody."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(io.BytesIO(output_midi))
    out = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for inst in pm.instruments:
        if not inst.is_drum and inst.program == melody_instrument and not keep_melody:
            continue
        name = "Drums" if inst.is_drum else pretty_midi.program_to_instrument_name(inst.program)
        new = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=name)
        for n in inst.notes:
            if trim_after is not None and n.start > trim_after:
                continue
            end = n.end if trim_after is None else min(n.end, trim_after + 0.5)
            new.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                              start=n.start + lead, end=end + lead))
        if new.notes:
            out.instruments.append(new)
    buf = io.BytesIO()
    out.write(buf)
    return buf.getvalue()
