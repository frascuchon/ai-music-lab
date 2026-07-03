#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adaptadores de prompt para cada modelo de generación MIDI.

Cada modelo tiene un contrato de entrada diferente. Este módulo traduce el
prompt de texto libre del usuario a la forma que cada modelo espera, sin
dependencias externas (solo stdlib).

API pública
-----------
    adapt(model_key, raw_prompt, fields=None) -> str
        Devuelve el prompt listo para pasar a --prompt del entrypoint Modal.

    instruments_to_classids(text) -> list[int]
        Extrae class-IDs de instrumentos MuseCoco a partir de texto libre.
        Devuelve lista vacía si no se detecta ninguno (stage-1 decide sola).

Modelos y su estrategia
-----------------------
    amadeus      — caption estilo MidiCaps: enriquece con key/BPM/instrumentos
                   si el usuario los proporcionó como campos opcionales.
                   Sin campos extra → passthrough.
    text2midi    — igual que amadeus (mismo encoder T5).
    midi_llm     — passthrough: el research script antepone su SYSTEM_PROMPT.
    chatmusician — passthrough: el research script construye "Human:/Assistant:".
    musecoco     — passthrough + se recomienda usar instruments_to_classids()
                   para derivar --instruments (ver midigen.py).
    anticipatory — no tiene prompt; devuelve "" (sin uso).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Mapa de instrumentos MuseCoco (28 clases, índice = class-ID)
# Fuente: research_musecoco_modal.py:275-279
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

# Alias → class-ID (keywords en texto libre del usuario)
_KEYWORD_TO_CLASS: dict[str, int] = {
    # piano / teclado
    "piano": 0, "grand piano": 0, "upright piano": 0,
    "keyboard": 1, "electric piano": 1, "clavier": 1, "harpsichord": 1,
    "organ": 3, "hammond": 3, "church organ": 3,
    # cuerdas pulsadas
    "guitar": 4, "electric guitar": 4, "acoustic guitar": 4,
    "bass guitar": 5, "bass": 5, "electric bass": 5, "upright bass": 5,
    "harp": 9,
    # cuerdas frotadas
    "violin": 6, "fiddle": 6,
    "viola": 7,
    "cello": 8,
    "strings": 10, "string ensemble": 10, "string quartet": 10,
    "orchestra": 10,
    # voz
    "voice": 11, "vocals": 11, "vocal": 11, "choir": 11,
    "soprano": 11, "singing": 11,
    # metal
    "trumpet": 12,
    "trombone": 13,
    "tuba": 14,
    "horn": 15, "french horn": 15,
    "brass": 16, "brass section": 16,
    # madera
    "sax": 17, "saxophone": 17, "alto sax": 17, "tenor sax": 17,
    "soprano sax": 17,
    "oboe": 18,
    "bassoon": 19,
    "clarinet": 20,
    "piccolo": 21,
    "flute": 22,
    "pipe": 23, "bagpipe": 23,
    # sintetizadores / electrónica
    "synth": 24, "synthesizer": 24, "pad": 24, "synth pad": 24,
    "lead synth": 24, "arpeggiator": 24,
    # percusión
    "drum": 27, "drums": 27, "drumkit": 27, "drum kit": 27,
    "percussion": 2, "bongo": 2, "conga": 2, "marimba": 2, "xylophone": 2,
    "vibraphone": 2, "timpani": 2,
    # especial
    "ethnic": 25, "sitar": 25, "shamisen": 25, "koto": 25,
    "sound effect": 26, "sfx": 26,
}


def instruments_to_classids(text: str) -> list[int]:
    """Detecta class-IDs de instrumentos MuseCoco en texto libre.

    Devuelve lista de IDs únicos (orden de aparición), o lista vacía si no
    detecta ninguno (en ese caso stage-1 de MuseCoco decide sola).

    Ejemplo:
        >>> instruments_to_classids("jazz sax trio with piano and drums")
        [0, 17, 27]
    """
    text_lower = text.lower()
    seen: list[int] = []
    # Ordenar por longitud descendente para que "bass guitar" se pruebe antes
    # que "bass" o "guitar" por separado.
    for kw in sorted(_KEYWORD_TO_CLASS, key=len, reverse=True):
        cid = _KEYWORD_TO_CLASS[kw]
        if kw in text_lower and cid not in seen:
            seen.append(cid)
    return seen


# ---------------------------------------------------------------------------
# Adaptadores de prompt por modelo
# ---------------------------------------------------------------------------

def _adapt_midicaps(raw_prompt: str, fields: dict | None) -> str:
    """Construye una caption estilo MidiCaps.

    Si el usuario no proporcionó campos opcionales, devuelve el prompt tal cual.
    Si proporcionó key/BPM/instrumentos/acordes, los añade al final de forma
    natural para enriquecer el contexto del encoder T5 de Amadeus/text2midi.
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
        # Normalizar "120 BPM", "120bpm", "120" → "120 BPM"
        bpm_clean = bpm.upper().replace("BPM", "").strip()
        if bpm_clean.isdigit():
            extras.append(f"at {bpm_clean} BPM")
        else:
            extras.append(bpm)
    if chords:
        extras.append(f"with chord progression {chords}")

    if not extras:
        return raw_prompt

    # Añadir extras al final, puntuando limpiamente
    base = raw_prompt.rstrip(".").rstrip()
    return base + ", " + ", ".join(extras) + "."


def adapt(model_key: str, raw_prompt: str, fields: dict | None = None) -> str:
    """Adapta un prompt de texto libre a la nomenclatura del modelo indicado.

    Parámetros
    ----------
    model_key : str
        Clave del modelo ("amadeus", "midi_llm", "text2midi", "chatmusician",
        "musecoco", "anticipatory").
    raw_prompt : str
        Texto libre tal como lo escribió el usuario en la UI de REAPER.
    fields : dict | None
        Campos opcionales de la UI: {"key": str, "bpm": str,
        "instruments": str, "chords": str}. Usados por amadeus/text2midi.

    Devuelve
    --------
    str
        Prompt listo para pasar como --prompt al entrypoint Modal.
        Para "anticipatory" devuelve "" (no usa prompt).

    Notas por modelo
    ----------------
    - amadeus / text2midi  : caption MidiCaps enriquecida con campos opcionales.
    - midi_llm             : passthrough — el research script antepone
                             "You are a world-class composer..." (SYSTEM_PROMPT).
    - chatmusician         : passthrough — el research script construye
                             "Human: {prompt} </s> Assistant: ".
    - musecoco             : passthrough — los instrumentos se pasan via
                             --instruments (inferidos con instruments_to_classids).
    - anticipatory         : sin prompt (seed MIDI + --mode); devuelve "".
    """
    key = model_key.lower()

    if key in ("amadeus", "text2midi"):
        return _adapt_midicaps(raw_prompt, fields)

    elif key in ("midi_llm", "chatmusician", "musecoco"):
        # El wrapping lo hace el propio research script
        return raw_prompt.strip()

    elif key == "anticipatory":
        # No usa --prompt; el entrypoint recibe --input (seed MIDI)
        return ""

    else:
        # Modelo desconocido: passthrough seguro
        return raw_prompt.strip()


# ---------------------------------------------------------------------------
# CLI de prueba (standalone, sin REAPER)
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
