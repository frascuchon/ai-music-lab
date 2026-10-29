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

Models and their strategy
-------------------------
    amadeus      — MidiCaps-style caption: enriches with key/BPM/instruments
                   if the user provided them as optional fields.
                   No extra fields → passthrough.
    anticipatory — no prompt; returns "" (unused).
"""

from __future__ import annotations


def _adapt_midicaps(raw_prompt: str, fields: dict | None) -> str:
    """Build a MidiCaps-style caption.

    If the user provided no optional fields, returns the prompt as-is.
    If they provided key/BPM/instruments/chords, appends them at the end
    in a natural way to enrich the T5 encoder context for Amadeus.
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
        Model key ("amadeus", "anticipatory").
    raw_prompt : str
        Free text as typed by the user in the REAPER UI.
    fields : dict | None
        Optional UI fields: {"key": str, "bpm": str,
        "instruments": str, "chords": str}. Used by amadeus.

    Returns
    -------
    str
        Prompt ready to pass as --prompt to the Modal entrypoint.
        For "anticipatory" returns "" (no prompt used).
    """
    key = model_key.lower()

    if key == "amadeus":
        return _adapt_midicaps(raw_prompt, fields)

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

    tests = [
        ("amadeus",      "A melodic jazz ballad",
         {"key": "F minor", "bpm": "72", "instruments": "piano and double bass", "chords": "Fm-Db-Ab-Eb"}),
        ("amadeus",      "A chill lofi track",  None),
        ("anticipatory", "doesn't matter", None),
    ]

    print("=== prompt_adapters smoke test ===\n")
    for model, prompt, fields in tests:
        result = adapt(model, prompt, fields)
        print(f"[{model}]")
        print(f"  input:  {prompt!r}")
        if fields:
            print(f"  fields: {json.dumps(fields)}")
        print(f"  output: {result!r}\n")
