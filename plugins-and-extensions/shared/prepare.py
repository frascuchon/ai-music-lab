#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First-run preparation of models — stdlib only.

The first inference with a model is slow: Modal has to build the container
image and download the weights into a Volume (remote), or uv has to create the
environment and download the weights (local). This module lets the Setup panel
pay that cost up front, one model at a time, and lets every panel know whether
it has already been paid.

* ``TARGETS``       — catalog of (domain, model, execution) combinations.
* ``build_command`` — the plugin backend CLI call for a minimal warm-up run
                      (a ~1 s generated tone / tiny MIDI as input).
* marker file       — ``~/.ai-musiclab/prepared.txt``, one ``domain|model|execution``
                      per line. Written by Setup after a successful warm-up and
                      by the panels after any successful real run
                      (lib/prepared.lua mirrors the format).
"""

from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path

PLUGINS_DIR = Path(__file__).resolve().parent.parent
SHARED_DIR = PLUGINS_DIR / "shared"

# Per-user folder for the app's shared resources (mirrors lib/app_dirs.lua).
APP_DIR_NAME = ".ai-musiclab"

STATE_ENV = "AI_MUSICLAB_PREPARED_FILE"


def app_dir() -> Path:
    return Path.home() / APP_DIR_NAME


def state_file() -> Path:
    override = os.environ.get(STATE_ENV)
    return Path(override) if override else app_dir() / "prepared.txt"


def key(domain: str, model: str, execution: str) -> str:
    return f"{domain}|{model}|{execution}"


def load() -> set[str]:
    try:
        lines = state_file().read_text().splitlines()
    except OSError:
        return set()
    return {ln.strip() for ln in lines if ln.count("|") == 2}


def is_prepared(domain: str, model: str, execution: str) -> bool:
    return key(domain, model, execution) in load()


def mark(domain: str, model: str, execution: str) -> None:
    """Append-only: several warm-ups finish concurrently, and a read-modify-
    write of the whole file could drop a marker. A single short O_APPEND write
    is atomic, and readers treat the file as a set (duplicates are harmless)."""
    entry = key(domain, model, execution)
    if entry in load():
        return
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(entry + "\n")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

T2A = PLUGINS_DIR / "Text2Audio"
MG = PLUGINS_DIR / "MidiGenerator"
A2M = PLUGINS_DIR / "Audio2Midi"
SS = PLUGINS_DIR / "StemsSeparator"

# (domain, model, execution, label, plugin folder, backend script, model script)
# "script" is the file handed to the backend via --script (None when the
# backend locates it itself).
TARGETS: list[dict] = []


def _add(domain, model, execution, label, plugin, backend, script, **extra):
    TARGETS.append({"domain": domain, "model": model, "execution": execution,
                    "label": label, "plugin": plugin, "backend": backend,
                    "script": script, **extra})


_T2A_PY = T2A / "text2audio.py"
for _m, _label, _script, _locals in (
    ("acestep_gen", "ACE-Step 1.5 (generate)", "research_acestep_gen_modal.py", False),
    ("sao", "Stable Audio Open 1.0", "research_stable_audio_open_modal.py", True),
    ("musicgen_gen", "MusicGen-medium", "research_musicgen_gen_modal.py", True),
    ("audiogen", "AudioGen-medium", "research_audiogen_modal.py", True),
):
    _add("text2audio", _m, "remote", _label, "Text2Audio", _T2A_PY,
         T2A / "research" / _script, mode="generate")
    if _locals:
        _add("text2audio", _m, "local", _label, "Text2Audio", _T2A_PY,
             T2A / "research" / _script, mode="generate")
_add("text2audio", "sao_edit", "remote", "SAO Style Transfer", "Text2Audio", _T2A_PY,
     T2A / "research" / "research_sao_edit_modal.py", mode="edit")
_add("text2audio", "sao_edit", "local", "SAO Style Transfer", "Text2Audio", _T2A_PY,
     T2A / "research" / "research_sao_edit_modal.py", mode="edit")
_add("text2audio", "acestep", "remote", "ACE-Step 1.5 (edit)", "Text2Audio", _T2A_PY,
     T2A / "research" / "research_acestep_edit_modal.py", mode="edit")

_MG_PY = MG / "midigen.py"
_add("midigen", "amadeus", "remote", "Amadeus", "MidiGenerator", _MG_PY,
     MG / "research" / "research_amadeus_modal.py")
_add("midigen", "anticipatory", "remote", "Anticipatory Music Transformer",
     "MidiGenerator", _MG_PY, MG / "research" / "research_anticipatory_modal.py")
_add("midigen", "anticipatory", "local", "Anticipatory Music Transformer",
     "MidiGenerator", _MG_PY, MG / "research" / "research_anticipatory.py")
_add("midigen", "midigpt", "remote", "MIDI-GPT (needs HF license)",
     "MidiGenerator", _MG_PY, MG / "research" / "research_midigpt_modal.py")

_A2M_PY = A2M / "transcribe.py"
_add("audio2midi", "basicpitch", "local", "Basic Pitch", "Audio2Midi", _A2M_PY,
     A2M / "research" / "research_basic_pitch.py")
_add("audio2midi", "basicpitch", "remote", "Basic Pitch", "Audio2Midi", _A2M_PY,
     A2M / "research" / "research_basic_pitch_modal.py")
_add("audio2midi", "miros", "remote", "MIROS", "Audio2Midi", _A2M_PY,
     A2M / "research" / "research_miros_modal.py")
_add("audio2midi", "yourmt3", "remote", "YourMT3+", "Audio2Midi", _A2M_PY,
     A2M / "research" / "research_yourmt3_modal.py")

_SAM_PY = SS / "separate_sam.py"
for _m, _repo, _label in (("sam_large", "facebook/sam-audio-large", "SAM Audio large"),
                          ("sam_base", "facebook/sam-audio-base", "SAM Audio base")):
    for _ex in ("remote", "local"):
        _add("stemseparator", _m, _ex, _label, "StemsSeparator", _SAM_PY,
             SS / "modal_sam_audio.py", repo=_repo)

# Rough size of the first-run work, shown next to each Prepare button.
HINTS = {
    ("acestep_gen", "remote"): "image build + ~10 GB weights",
    ("acestep", "remote"): "image build + ~10 GB weights",
    ("sam_large", "local"): "~15-20 GB download",
    ("sam_base", "local"): "several GB download",
    ("sam_large", "remote"): "image build + large weights",
    ("sam_base", "remote"): "image build + weights",
}


# Targets that run on the same Modal image and weights Volume: preparing one
# prepares the others, so "prepare all" never builds the same image twice.
SHARED_IMAGE = [
    {("text2audio", "acestep_gen", "remote"), ("text2audio", "acestep", "remote")},
]

# Too big for a one-click "prepare all local" (tens of GB): individual button only.
HEAVY = {("sam_large", "local"), ("sam_base", "local")}


def is_heavy(target: dict) -> bool:
    return (target["model"], target["execution"]) in HEAVY


def group_of(target: dict) -> str:
    """Id shared by the targets that run on one image ("" when alone)."""
    k = (target["domain"], target["model"], target["execution"])
    for i, group in enumerate(SHARED_IMAGE):
        if k in group:
            return f"image{i}"
    return ""


def mark_with_siblings(domain: str, model: str, execution: str) -> None:
    mark(domain, model, execution)
    for group in SHARED_IMAGE:
        if (domain, model, execution) in group:
            for d, m, e in group:
                mark(d, m, e)


# The warm-up input is synthetic, so a backend may legitimately end with an
# "error" that only says the model found nothing in it. The image, weights and
# inference all ran by then, which is all a warm-up is for.
BENIGN_ERRORS = {
    "audio2midi": "No notes were transcribed",
}


def is_benign_error(target: dict, message: str) -> bool:
    needle = BENIGN_ERRORS.get(target["domain"])
    return bool(needle) and needle in message


def available_targets() -> list[dict]:
    """Targets whose plugin folder is installed."""
    return [t for t in TARGETS if t["plugin"] and (PLUGINS_DIR / t["plugin"]).is_dir()]


def find_target(domain: str, model: str, execution: str) -> dict | None:
    return next((t for t in TARGETS
                 if (t["domain"], t["model"], t["execution"]) == (domain, model, execution)),
                None)


# ---------------------------------------------------------------------------
# Warm-up inputs (tiny, generated — no fixtures to ship)
# ---------------------------------------------------------------------------

def write_tone(path: Path, seconds: float = 2.0, sr: int = 16000) -> Path:
    """A quiet 440 Hz mono tone."""
    frames = b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / sr)))
                      for i in range(int(seconds * sr)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)
    return path


def write_melody_mid(path: Path) -> Path:
    """Format-0 SMF: 16 quarter notes (4 bars of 4/4, 8 s) of a C major scale,
    480 ppqn, 120 BPM. MIDI-GPT rejects scores shorter than its 4-bar window."""
    track = bytearray(b"\x00\xff\x51\x03\x07\xa1\x20")  # tempo 500000 µs/beat
    for pitch in (60, 62, 64, 65, 67, 69, 71, 72) * 2:
        track += b"\x00\x90" + bytes([pitch, 90])
        track += b"\x83\x60\x80" + bytes([pitch, 0])    # delta 480 ticks
    track += b"\x00\xff\x2f\x00"
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
    path.write_bytes(header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track))
    return path


# ---------------------------------------------------------------------------
# Warm-up command
# ---------------------------------------------------------------------------

def output_dir_name(target: dict) -> str:
    return f"{target['domain']}_{target['model']}_{target['execution']}"


def build_command(target: dict, python: str, workdir: Path, progress: Path) -> list[str]:
    """Backend CLI call that runs the smallest real job for the target."""
    out = workdir / "out"
    cmd = [python, str(target["backend"])]
    domain, model, ex = target["domain"], target["model"], target["execution"]
    script = str(target["script"])
    if domain == "text2audio":
        cmd += ["--shared-dir", str(SHARED_DIR), "--script", script, "--model", model,
                "--mode", target["mode"], "--prompt", "warm-up", "--gpu", "A10G",
                "--execution", ex, "--out-dir", str(out), "--progress", str(progress)]
        if target["mode"] == "generate":
            cmd += ["--seconds", "10" if model == "acestep_gen" else "4"]
        else:
            cmd += ["--input", str(write_tone(workdir / "tone.wav", 4.0)),
                    "--intensity", "moderate"]
    elif domain == "midigen":
        cmd += ["--shared-dir", str(SHARED_DIR), "--script", script, "--model", model,
                "--execution", ex, "--out-dir", str(out), "--progress", str(progress),
                "--gpu", "A10G", "--force"]
        if model == "amadeus":
            cmd += ["--prompt", "warm-up", "--n-outputs", "1"]
        else:
            cmd += ["--melody", str(write_melody_mid(workdir / "melody.mid")),
                    "--clip-length", "8"]
    elif domain == "audio2midi":
        cmd += ["--shared-dir", str(SHARED_DIR), "--script", script, "--model", model,
                "--execution", ex, "--input", str(write_tone(workdir / "tone.wav", 3.0)),
                "--out-dir", str(out), "--gpu", "A10G", "--progress", str(progress)]
    elif domain == "stemseparator":
        cmd += ["--sam-dir", str(SS), "--shared-dir", str(SHARED_DIR),
                "--input", str(write_tone(workdir / "tone.wav", 5.0)),
                "--prompt", "vocals", "--model", target["repo"], "--gpu", "A100-80GB",
                "--steps", "2", "--execution", ex,
                "--outdir", str(out), "--progress", str(progress)]
    else:
        raise ValueError(f"unknown domain {domain!r}")
    return cmd
