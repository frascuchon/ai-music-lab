#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend CLI for MidiGenerator.lua — generates MIDI via Modal cloud.

Called by MidiGenerator.lua as a background process:

    python3 midigen.py
        --shared-dir <shared/>
        --script     <research_*_modal.py>
        --model      <model-key>
        --out-dir    <output-directory>
        --progress   <progress-file>
        [--prompt    <free-text>]
        [--field-key <key>]  [--field-bpm <BPM>]
        [--field-instruments <list>]  [--field-chords <chords>]
        [--n-outputs <1-4>]
        [--temperature <float>]
        [--gpu   <A10G|A100|T4|L4>]
        [--force]
        # ChatMusician with harmonization only:
        [--seed-file  <path.mid>]
        # Anticipatory (AMT) only:
        [--seed       <path.mid>]
        [--mode       <accompaniment|continuation>]
        [--prompt-length   <int>]
        [--clip-length     <int>]
        [--top-p           <float>]
        [--multiplicity    <int>]
        [--melody-instrument <int>]

Progress protocol (--progress):
    state|pct|msg\\n          state = running | done | error
    [extra-lines]             .mid paths + "INSTRUMENTS|n"

Model entrypoint contracts (verified in research_*_modal.py)
-------------------------------------------------------------
text_dir  (amadeus, midi_llm, text2midi, chatmusician):
    ::main --prompt … --out-dir … [--n-outputs] [--temperature] [--gpu] [--force]
    Output: generated_cuda_v*.mid  (glob in out-dir)

text_file (musecoco):
    ::main --prompt … --out <out-dir>/generated.mid [--n-samples] [--instruments "0,5"]
    Output: generated.mid (n=1) or generated.mid + generated_2.mid … (n>1)

seed_file (anticipatory):
    ::main --input <seed.mid> --out <out-dir>/generated.mid
           --mode <accompaniment|continuation> --gpu …
           [--prompt-length] [--clip-length] [--top-p]
           [--multiplicity] [--melody-instrument]
    Output: generated.mid (mult=1) or generated_v0.mid … generated_v{N-1}.mid (mult>1)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import adapters (same directory)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
from prompt_adapters import adapt, instruments_to_classids  # noqa: E402


# ---------------------------------------------------------------------------
# Model contract type per model key
# ---------------------------------------------------------------------------
_MODEL_KIND: dict[str, str] = {
    "amadeus":      "text_dir",
    "midi_llm":     "text_dir",
    "text2midi":    "text_dir",
    "chatmusician": "text_dir",
    "musecoco":     "text_file",
    "anticipatory": "seed_file",
}

# Models whose ::main accepts --gpu (others ignore it)
_SUPPORTS_GPU = {"midi_llm", "anticipatory"}

# text_dir models that accept --force (midi_llm does not)
_SUPPORTS_FORCE = {"amadeus", "text2midi", "chatmusician"}


# ---------------------------------------------------------------------------
# Progress protocol
# ---------------------------------------------------------------------------

def write_progress(path: str, state: str, pct: float, msg: str,
                   extra: list[str] | None = None) -> None:
    if not path:
        return
    try:
        with open(path, "w") as f:
            f.write(f"{state}|{pct:.3f}|{msg}\n")
            if extra:
                for line in extra:
                    f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_uv() -> Path | None:
    uv = shutil.which("uv")
    if uv:
        return Path(uv)
    fallback = Path.home() / ".local" / "bin" / "uv"
    return fallback if fallback.is_file() else None


def count_instruments(mid_path: Path) -> int:
    """Count non-empty MIDI instruments; returns -1 if pretty_midi is unavailable."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(mid_path))
        return sum(1 for inst in pm.instruments if inst.notes)
    except Exception:
        return -1


def _collect_outputs(out_dir: Path) -> list[Path]:
    """Collect all generated .mid files in out_dir, sorted.
    Filters files that are too small (< 64 bytes) — usually empty or invalid
    MIDIs produced when the model generates text instead of ABC notation.
    For AMT (seed_file), expects the generated*.mid pattern."""
    mids = [m for m in sorted(out_dir.glob("*.mid")) if m.stat().st_size >= 64]
    return mids


# ---------------------------------------------------------------------------
# Build modal run command per model
# ---------------------------------------------------------------------------

def _build_cmd(
    uv_bin: Path,
    shared_dir: Path,
    script_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    adapted_prompt: str,
    instruments_str: str,
) -> list[str]:
    """Return the token list for the modal run command to execute."""

    model = args.model.lower()
    kind  = _MODEL_KIND.get(model, "text_dir")

    base = [
        str(uv_bin), "run", "--project", str(shared_dir),
        "modal", "run", f"{script_path}::main",
    ]

    if kind == "text_dir":
        cmd = base + [
            "--prompt",    adapted_prompt,
            "--out-dir",   str(out_dir),
            "--n-outputs", str(args.n_outputs),
            "--temperature", str(args.temperature),
        ]
        if model in _SUPPORTS_FORCE:
            cmd += ["--force"]
        if model == "midi_llm" and args.gpu:
            cmd += ["--gpu", args.gpu]
        # AMT (anticipatory) also accepts --gpu
        if model == "anticipatory" and args.gpu:
            cmd += ["--gpu", args.gpu]
        # ChatMusician: optional seed for harmonization
        if model == "chatmusician" and args.seed_file and Path(args.seed_file).exists():
            cmd += ["--input-file", args.seed_file]

    elif kind == "text_file":
        out_path = out_dir / "generated.mid"
        cmd = base + [
            "--prompt",   adapted_prompt,
            "--out",      str(out_path),
            "--n-samples", str(args.n_outputs),
        ]
        if instruments_str:
            cmd += ["--instruments", instruments_str]

    elif kind == "seed_file":
        if not args.seed or not Path(args.seed).exists():
            raise FileNotFoundError(
                f"Anticipatory requires a seed MIDI (--seed). "
                f"Specified path: {args.seed!r}"
            )
        out_path = out_dir / "generated.mid"
        cmd = base + [
            "--input",          args.seed,
            "--out",            str(out_path),
            "--mode",           args.mode,
            "--gpu",            args.gpu or "A10G",
            "--prompt-length",  str(args.prompt_length),
            "--top-p",          str(args.top_p),
            "--multiplicity",   str(args.n_outputs),
            "--melody-instrument", str(args.melody_instrument),
        ]
        # continuation uses --duration, accompaniment uses --clip-length
        if args.mode == "continuation":
            cmd += ["--duration", str(args.clip_length)]
        else:
            cmd += ["--clip-length", str(args.clip_length)]

    else:
        raise ValueError(f"Unknown contract type: {kind!r}")

    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate MIDI via Modal cloud (MidiGenerator.lua backend)."
    )
    # Infrastructure
    p.add_argument("--shared-dir", required=True, dest="shared_dir")
    p.add_argument("--script",     required=True)
    p.add_argument("--model",      required=True)
    p.add_argument("--out-dir",    required=True, dest="out_dir")
    p.add_argument("--progress",   default="")

    # Prompt and optional fields (for MidiCaps adapters)
    p.add_argument("--prompt",             default="")
    p.add_argument("--field-key",          default="", dest="field_key")
    p.add_argument("--field-bpm",          default="", dest="field_bpm")
    p.add_argument("--field-instruments",  default="", dest="field_instruments")
    p.add_argument("--field-chords",       default="", dest="field_chords")

    # Common generation parameters
    p.add_argument("--n-outputs",   type=int,   default=2,   dest="n_outputs")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gpu",         default="A10G")
    p.add_argument("--force",       action="store_true")

    # ChatMusician: optional seed for harmonization
    p.add_argument("--seed-file", default="", dest="seed_file")

    # Anticipatory (AMT): seed + mode parameters
    p.add_argument("--seed",             default="")
    p.add_argument("--mode",             default="accompaniment")
    p.add_argument("--prompt-length",    type=int,   default=5, dest="prompt_length")
    p.add_argument("--clip-length",      type=int,   default=20, dest="clip_length")
    p.add_argument("--top-p",            type=float, default=0.95, dest="top_p")
    p.add_argument("--melody-instrument",type=int,   default=0, dest="melody_instrument")

    args = p.parse_args()

    pf          = args.progress
    model_key   = args.model.lower()
    shared_dir  = Path(args.shared_dir)
    script_path = Path(args.script)
    out_dir     = Path(args.out_dir)

    write_progress(pf, "running", 0.02, "Preparing MIDI generation...")

    # --- Basic validations ---------------------------------------------------
    if not script_path.exists():
        write_progress(pf, "error", 0,
                       f"Modal script not found: {script_path}")
        return 1
    if not (shared_dir / "pyproject.toml").exists():
        write_progress(pf, "error", 0,
                       f"shared/pyproject.toml not found at: {shared_dir}")
        return 1

    kind = _MODEL_KIND.get(model_key)
    if kind is None:
        write_progress(pf, "error", 0, f"Unknown model: {model_key!r}")
        return 1

    if kind in ("text_dir", "text_file") and not args.prompt.strip():
        write_progress(pf, "error", 0,
                       "This model requires a text prompt (--prompt).")
        return 1

    if kind == "seed_file" and not args.seed:
        write_progress(pf, "error", 0,
                       "Anticipatory requires a seed MIDI (select a MIDI item "
                       "in REAPER or choose a .mid file).")
        return 1

    # --- Locate uv ----------------------------------------------------------
    uv_bin = _find_uv()
    if uv_bin is None:
        write_progress(pf, "error", 0,
                       "uv not found. Install with:\n"
                       "curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Adapt prompt + derive MuseCoco instruments -------------------------
    fields: dict | None = None
    if kind in ("text_dir", "text_file") and any([
        args.field_key, args.field_bpm, args.field_instruments, args.field_chords,
    ]):
        fields = {
            "key":         args.field_key,
            "bpm":         args.field_bpm,
            "instruments": args.field_instruments,
            "chords":      args.field_chords,
        }

    adapted_prompt  = adapt(model_key, args.prompt, fields)
    instruments_str = ""
    if model_key == "musecoco":
        cids = instruments_to_classids(args.prompt)
        if cids:
            instruments_str = ",".join(str(c) for c in cids)

    # --- Build and launch modal command ------------------------------------
    write_progress(pf, "running", 0.06,
                   f"Launching Modal ({model_key} · {args.gpu})...")

    try:
        cmd = _build_cmd(
            uv_bin, shared_dir, script_path,
            out_dir, args, adapted_prompt, instruments_str,
        )
    except (FileNotFoundError, ValueError) as exc:
        write_progress(pf, "error", 0, str(exc))
        return 1

    env = os.environ.copy()
    print(f"[midigen] Command: {' '.join(cmd)}", flush=True)
    print(f"[midigen] Model: {model_key}  kind={kind}", flush=True)
    if kind in ("text_dir", "text_file"):
        print(f"[midigen] Adapted prompt: {adapted_prompt[:120]}", flush=True)
    if instruments_str:
        print(f"[midigen] Instruments override: {instruments_str}", flush=True)

    import subprocess
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
    except (FileNotFoundError, PermissionError) as exc:
        write_progress(pf, "error", 0, f"Could not execute uv: {exc}")
        return 1

    # --- Parse stdout for progress -----------------------------------------
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)

        pct = 0.1
        m = re.search(r'\[(\d+)/(\d+)\]', line)
        if m:
            n, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                pct = 0.10 + (n / total) * 0.80
        elif "Chunk" in line and "/" in line:
            try:
                part = line.split("Chunk")[1].split(":")[0].strip()
                n, total = int(part.split("/")[0]), int(part.split("/")[1])
                if total > 0:
                    pct = 0.10 + (n / total) * 0.80
            except (ValueError, IndexError):
                pass
        # AMT/MuseCoco report progress differently; keep incremental pct
        elif "spawned" in line.lower() or "candidate" in line.lower():
            pct = 0.15
        elif "saved" in line.lower() or "→" in line:
            pct = 0.90

        write_progress(pf, "running", pct, line[:80])

    proc.wait()

    if proc.returncode != 0:
        write_progress(pf, "error", 0,
                       f"Modal failed (code {proc.returncode}). "
                       "Check the log for details.")
        return 1

    # --- Collect outputs ---------------------------------------------------
    write_progress(pf, "running", 0.95, "Collecting MIDI files...")

    mids = _collect_outputs(out_dir)
    if not mids:
        write_progress(pf, "error", 0,
                       f"No .mid files found in {out_dir}. "
                       "The process ended without error but produced no output.")
        return 1

    # Count instruments on the first candidate for the status message
    n_instr = count_instruments(mids[0])
    n_label = (f"{n_instr} instrument{'s' if n_instr != 1 else ''}"
               if n_instr >= 0 else "MIDI generated")
    done_msg = f"Completed — {len(mids)} candidate{'s' if len(mids)!=1 else ''}, {n_label}"

    extra = [str(m) for m in mids] + [f"INSTRUMENTS|{n_instr if n_instr >= 0 else '?'}"]
    write_progress(pf, "done", 1.0, done_msg, extra)
    print(f"[midigen] {done_msg}", flush=True)
    for m in mids:
        print(f"  → {m}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
