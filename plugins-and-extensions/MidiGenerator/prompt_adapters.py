#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt adapters for each MIDI generation model.

Each model has a different input contract. This module translates the user's
free-text prompt into the form each model expects, with no external dependencies
(stdlib only).

Public API
----------
    adapt(model_key, raw_prompt, fields=None) -> str
        Returns the prompt ready to pass as --prompt to the Modal entrypoint.

    instruments_to_classids(text) -> list[int]
        Extracts MuseCoco instrument class-IDs from free text.
        Returns an empty list if none are detected (stage-1 decides on its own).

Models and their strategy
-------------------------
    amadeus      — MidiCaps-style caption: enriches with key/BPM/instruments
                   if the user provided them as optional fields.
                   No extra fields → passthrough.
    text2midi    — same as amadeus (same T5 encoder).
    midi_llm     — passthrough: the research script prepends its SYSTEM_PROMPT.
    chatmusician — passthrough: the research script builds "Human:/Assistant:".
    musecoco     — passthrough + recommended to use instruments_to_classids()
                   to derive --instruments (see midigen.py).
    anticipatory — no prompt; returns "" (unused).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# MuseCoco instrument map (28 classes, index = class-ID)
# Source: research_musecoco_modal.py:275-279
# ---------------------------------------------------------------------------
_MUSECOCO_CLASSES: list[str] = [
    "piano",       # 0
    "keyboard",    # 1
    "percussion",  # 2
    "organ",       # 3
    "guitar",      # 4
    "bass",        # 5
    "violin",      # 6
    "viola",       # 7
    "cello",       # 8
    "harp",        # 9
    "strings",     # 10
    "voice",       # 11
    "trumpet",     # 12
    "trombone",    # 13
    "tuba",        # 14
    "horn",        # 15
    "brass",       # 16
    "sax",         # 17
    "oboe",        # 18
    "bassoon",     # 19
    "clarinet",    # 20
    "piccolo",     # 21
    "flute",       # 22
    "pipe",        # 23
    "synthesizer", # 24
    "ethnic",      # 25
    "sound_effect",# 26
    "drum",        # 27
]

# Alias → class-ID (keywords in user free text)
_KEYWORD_TO_CLASS: dict[str, int] = {
    # piano / keyboard
    "piano": 0, "grand piano": 0, "upright piano": 0,
    "keyboard": 1, "electric piano": 1, "clavier": 1, "harpsichord": 1,
    "organ": 3, "hammond": 3, "church organ": 3,
    # plucked strings
    "guitar": 4, "electric guitar": 4, "acoustic guitar": 4,
    "bass guitar": 5, "bass": 5, "electric bass": 5, "upright bass": 5,
    "harp": 9,
    # bowed strings
    "violin": 6, "fiddle": 6,
    "viola": 7,
    "cello": 8,
    "strings": 10, "string ensemble": 10, "string quartet": 10,
    "orchestra": 10,
    # voice
    "voice": 11, "vocals": 11, "vocal": 11, "choir": 11,
    "soprano": 11, "singing": 11,
    # brass
    "trumpet": 12,
    "trombone": 13,
    "tuba": 14,
    "horn": 15, "french horn": 15,
    "brass": 16, "brass section": 16,
    # woodwinds
    "sax": 17, "saxophone": 17, "alto sax": 17, "tenor sax": 17,
    "soprano sax": 17,
    "oboe": 18,
    "bassoon": 19,
    "clarinet": 20,
    "piccolo": 21,
    "flute": 22,
    "pipe": 23, "bagpipe": 23,
    # synthesizers / electronics
    "synth": 24, "synthesizer": 24, "pad": 24, "synth pad": 24,
    "lead synth": 24, "arpeggiator": 24,
    # percussion
    "drum": 27, "drums": 27, "drumkit": 27, "drum kit": 27,
    "percussion": 2, "bongo": 2, "conga": 2, "marimba": 2, "xylophone": 2,
    "vibraphone": 2, "timpani": 2,
    # special
    "ethnic": 25, "sitar": 25, "shamisen": 25, "koto": 25,
    "sound effect": 26, "sfx": 26,
}


def instruments_to_classids(text: str) -> list[int]:
    """Detect MuseCoco instrument class-IDs in free text.

    Returns a list of unique IDs (order of appearance), or an empty list if
    none are detected (in that case MuseCoco stage-1 decides on its own).

    Example:
        >>> instruments_to_classids("jazz sax trio with piano and drums")
        [0, 17, 27]
    """
    text_lower = text.lower()
    seen: list[int] = []
    # Sort by descending length so "bass guitar" is tried before "bass" or "guitar".
    for kw in sorted(_KEYWORD_TO_CLASS, key=len, reverse=True):
        cid = _KEYWORD_TO_CLASS[kw]
        if kw in text_lower and cid not in seen:
            seen.append(cid)
    return seen


# ---------------------------------------------------------------------------
# Prompt adapters per model
# ---------------------------------------------------------------------------

def _adapt_midicaps(raw_prompt: str, fields: dict | None) -> str:
    """Build a MidiCaps-style caption.

    If the user provided no optional fields, returns the prompt as-is.
    If they provided key/BPM/instruments/chords, appends them at the end
    in a natural way to enrich the T5 encoder context for Amadeus/text2midi.
    """
    if not fields:
        return raw_prompt

    extras: list[str] = []
    key = (fields.get("key") or "").strip()
    bpm = (fields.get("bpm") or "").strip()
    instruments = (fields.get("instruments") or "").strip()
    chords = (fields.get("chords") or "").strip()

    if instruments:
        extras.append(f"featuring {instruments}")
    if key:
        extras.append(f"in {key}")
    if bpm:
        # Normalize "120 BPM", "120bpm", "120" → "120 BPM"
        bpm_clean = bpm.upper().replace("BPM", "").strip()
        if bpm_clean.isdigit():
            extras.append(f"at {bpm_clean} BPM")
        else:
            extras.append(bpm)
    if chords:
        extras.append(f"with chord progression {chords}")

    if not extras:
        return raw_prompt

    # Append extras at the end with clean punctuation
    base = raw_prompt.rstrip(".").rstrip()
    return base + ", " + ", ".join(extras) + "."


def adapt(model_key: str, raw_prompt: str, fields: dict | None = None) -> str:
    """Adapt a free-text prompt to the nomenclature of the specified model.

    Parameters
    ----------
    model_key : str
        Model key ("amadeus", "midi_llm", "text2midi", "chatmusician",
        "musecoco", "anticipatory").
    raw_prompt : str
        Free text as typed by the user in the REAPER UI.
    fields : dict | None
        Optional UI fields: {"key": str, "bpm": str,
        "instruments": str, "chords": str}. Used by amadeus/text2midi.

    Returns
    -------
    str
        Prompt ready to pass as --prompt to the Modal entrypoint.
        For "anticipatory" returns "" (no prompt used).

    Notes per model
    ---------------
    - amadeus / text2midi  : MidiCaps caption enriched with optional fields.
    - midi_llm             : passthrough — the research script prepends
                             "You are a world-class composer..." (SYSTEM_PROMPT).
    - chatmusician         : passthrough — the research script builds
                             "Human: {prompt} </s> Assistant: ".
    - musecoco             : passthrough — instruments are passed via
                             --instruments (inferred with instruments_to_classids).
    - anticipatory         : no prompt (seed MIDI + --mode); returns "".
    """
    key = model_key.lower()

    if key in ("amadeus", "text2midi"):
        return _adapt_midicaps(raw_prompt, fields)

    elif key in ("midi_llm", "chatmusician", "musecoco"):
        # Wrapping is done by the research script itself
        return raw_prompt.strip()

    elif key == "anticipatory":
        # Does not use --prompt; the entrypoint receives --input (seed MIDI)
        return ""

    else:
        # Unknown model: safe passthrough
        return raw_prompt.strip()


# ---------------------------------------------------------------------------
# Test CLI (standalone, without REAPER)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    tests = [
        ("amadeus",      "A melodic jazz ballad",
         {"key": "F minor", "bpm": "72", "instruments": "piano and double bass", "chords": "Fm-Db-Ab-Eb"}),
        ("amadeus",      "A chill lofi track",  None),
        ("text2midi",    "A sad pop song with a strong piano presence.", None),
        ("midi_llm",     "upbeat jazz trio, 120 BPM, piano bass and drums", None),
        ("chatmusician", "Develop a tune influenced by Bach's compositions.", None),
        ("musecoco",     "jazz piano trio, upbeat, C major, 120 BPM", None),
        ("anticipatory", "doesn't matter", None),
    ]

    print("=== prompt_adapters smoke test ===\n")
    for model, prompt, fields in tests:
        result = adapt(model, prompt, fields)
        cids = instruments_to_classids(prompt) if model == "musecoco" else []
        print(f"[{model}]")
        print(f"  input:  {prompt!r}")
        if fields:
            print(f"  fields: {json.dumps(fields)}")
        print(f"  output: {result!r}")
        if cids:
            names = [_MUSECOCO_CLASSES[i] for i in cids]
            print(f"  instruments class IDs: {cids}  ({', '.join(names)})")
        print()

    # Test instruments_to_classids
    test_phrases = [
        "jazz sax trio with piano and drums at 120 BPM",
        "orchestral piece with violin cello and french horn",
        "electronic track with synth pad and bass guitar",
        "no instruments here",
    ]
    print("=== instruments_to_classids ===\n")
    for phrase in test_phrases:
        cids = instruments_to_classids(phrase)
        names = [_MUSECOCO_CLASSES[i] for i in cids]
        print(f"  {phrase!r}")
        print(f"  → IDs: {cids}  names: {names}\n")
