#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend CLI for MidiGenerator.lua — generates MIDI via Modal cloud, or
(for the models that support it — see shared/model_execution.py's MIDIGEN
registry) locally on this machine.

Called by MidiGenerator.lua as a background process:

    python3 midigen.py
        --shared-dir <shared/>
        --script     <research_*_modal.py>   (remote)
                      <research_*.py>         (local — plain, non-Modal script)
        --model      <model-key>
        --out-dir    <output-directory>
        --progress   <progress-file>
        [--execution local|remote]  (default: remote)
        [--prompt    <free-text>]
        [--field-key <key>]  [--field-bpm <BPM>]
        [--field-instruments <list>]  [--field-chords <chords>]
        [--n-outputs <1-4>]
        [--temperature <float>]
        [--gpu   <A10G|A100|T4|L4>]
        [--force]
        # Anticipatory (AMT) only:
        [--melody     <path.mid>]   (channel 0 = melody; other channels = optional accompaniment)
        [--style      <pop|pad|arp>]
        [--parts bass,keys,strings,guitar,drums] [--density 0|1|2]
        [--bass-program N] [--keys-program N] [--creativity 0..1]
        [--chords "C#m G# A# F"] [--beats-per-chord N]      (Anticipatory)
        [--stay-in-key]                                      (MIDI-GPT)
        [--prompt-length   <int>]
        [--clip-length     <int>]   (0 = length of the melody)
        [--top-p           <float>]
        [--multiplicity    <int>]
        [--melody-instrument <int>]

Progress protocol (--progress):
    state|pct|msg\\n          state = running | done | error
    [extra-lines]             .mid paths + "INSTRUMENTS|n"

Model entrypoint contracts (verified in research_*_modal.py)
-------------------------------------------------------------
text_dir  (amadeus):
    ::main --prompt … --out-dir … [--n-outputs] [--temperature] [--gpu] [--force]
    Output: generated_cuda_v*.mid  (glob in out-dir)

melody_file (anticipatory):
    ::main --input <melody.mid> --out <out-dir>/generated.mid --gpu … --style …
           --parts … --density … --bass-program … --keys-program … [--chords …]
           [--prompt-length] [--clip-length] [--top-p]
           [--multiplicity] [--melody-instrument]
    Output: only the accompaniment (the melody is not returned).

melody_file (midigpt):
    ::main --input <melody.mid> --out <out-dir>/generated.mid --spec <spec.json>
           --multiplicity N          (spec built by midigpt_spec.build_spec)
    Output: only the generated parts.
    Output: generated.mid (mult=1) or generated_v0.mid … generated_v{N-1}.mid (mult>1)

Local entrypoint contracts (--execution local; only anticipatory —
see shared/model_execution.py MIDIGEN registry for the full feasibility list)
-------------------------------------------------------------------------------
anticipatory (research_anticipatory.py, plain — no Modal, CPU-only):
    --input <melody.mid> --out <out-dir>/generated.mid --style …
           [--prompt-length] [--clip-length] [--top-p] [--multiplicity] [--melody-instrument]
    Output: generated.mid (mult=1) or generated_v0.mid … (mult>1)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import deque
from pathlib import Path

# How many trailing lines of the subprocess's raw stdout to keep and surface
# to the user on failure (see _pick_error_message / tail_lines below).
_LOG_TAIL_LINES = 20

# ---------------------------------------------------------------------------
# Import adapters (same directory)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
from prompt_adapters import adapt  # noqa: E402
import midigpt_spec  # noqa: E402


# ---------------------------------------------------------------------------
# Model contract type per model key
# ---------------------------------------------------------------------------
_MODEL_KIND: dict[str, str] = {
    "amadeus":      "text_dir",
    "anticipatory": "melody_file",
    "midigpt":      "melody_file",
}

# Models whose ::main accepts --gpu (others ignore it; MIDI-GPT's GPU is fixed in its script)
_SUPPORTS_GPU = {"anticipatory"}

# Accompaniment engines that take a melody MIDI, and the display name used in errors
_MELODY_ENGINES = {"anticipatory": "Anticipatory", "midigpt": "MIDI-GPT"}

# text_dir models that accept --force
_SUPPORTS_FORCE = {"amadeus"}

# text_dir models whose ::main accepts --base-seed
_SUPPORTS_BASE_SEED = {"amadeus"}


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
    For AMT (melody_file), expects the generated*.mid pattern."""
    mids = [m for m in sorted(out_dir.glob("*.mid")) if m.stat().st_size >= 64]
    return mids


def _pick_error_message(
    returncode: int, out_dir: Path,
    last_error_line: str | None, last_warn_line: str | None,
) -> str:
    """Build the final error message shown to the user, preferring a
    root-cause line captured from the subprocess's own stdout (e.g. the
    research_*_modal.py script's own "ERROR: ..." on seed-conversion
    failure) over the generic ones below.

    - Non-zero exit: an "ERROR:" line, if seen, names the actual failure
      (e.g. a failed Modal call); otherwise fall back to the exit-code message.
    - Zero exit but no outputs: a "[warn]" line, if seen, usually explains
      why the model was fed a degraded/unseeded prompt (e.g. an unrecognized
      seed extension); appended to the generic "no outputs" message.
    """
    if returncode != 0:
        if last_error_line:
            return last_error_line
        return f"Modal failed (code {returncode}). Check the log for details."

    msg = (f"No .mid files found in {out_dir}. "
           "The process ended without error but produced no output.")
    cause = last_error_line or last_warn_line
    if cause:
        msg += f" Possible cause: {cause}"
    return msg


# ---------------------------------------------------------------------------
# Accompaniment controls -> per-engine arguments
# ---------------------------------------------------------------------------

def _parts(args) -> list:
    return [p.strip() for p in args.parts.split(",") if p.strip()]


def _top_p(args) -> float:
    """Creativity 0..1 -> nucleus-sampling top_p 0.80..1.00 (measured: at 1.0 the model
    adds instruments of its own; at 0.8 it sticks to the ones in the prompt)."""
    return round(0.80 + 0.20 * args.creativity, 3)


def _anticipatory_flags(args) -> list:
    flags = [
        "--style",           args.style,
        "--parts",           ",".join(_parts(args)),
        "--density",         str(args.density),
        "--bass-program",    str(args.bass_program),
        "--keys-program",    str(args.keys_program),
        "--beats-per-chord", str(args.beats_per_chord),
        "--top-p",           str(_top_p(args)),
    ]
    if args.chords.strip():
        flags += ["--chords", args.chords]
    return flags


def _midigpt_spec(args) -> dict:
    return midigpt_spec.build_spec(
        parts=tuple(_parts(args)), style=args.style, density=args.density,
        programs={"bass": args.bass_program, "keys": args.keys_program},
        creativity=args.creativity, stay_in_key=args.stay_in_key,
    )


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
        if model in _SUPPORTS_BASE_SEED:
            # Fresh seed per run: a fixed seed made every generation of the
            # same prompt return the identical MIDI.
            cmd += ["--base-seed", str(random.randrange(2**31))]
        if model in _SUPPORTS_GPU and args.gpu:
            cmd += ["--gpu", args.gpu]

    elif kind == "melody_file":
        if not args.melody or not Path(args.melody).exists():
            raise FileNotFoundError(
                f"{_MELODY_ENGINES.get(model, model)} requires the melody MIDI (--melody). "
                f"Specified path: {args.melody!r}"
            )
        out_path = out_dir / "generated.mid"
        if model == "midigpt":
            spec_path = out_dir / "spec.json"
            spec_path.write_text(json.dumps(_midigpt_spec(args)))
            cmd = base + [
                "--input",        args.melody,
                "--out",          str(out_path),
                "--spec",         str(spec_path),
                "--multiplicity", str(args.n_outputs),
            ]
        else:
            cmd = base + [
                "--input",          args.melody,
                "--out",            str(out_path),
                "--gpu",            args.gpu or "A10G",
                "--prompt-length",  str(args.prompt_length),
                "--clip-length",    str(args.clip_length),
                "--multiplicity",   str(args.n_outputs),
                "--melody-instrument", str(args.melody_instrument),
            ] + _anticipatory_flags(args)

    else:
        raise ValueError(f"Unknown contract type: {kind!r}")

    return cmd


# ---------------------------------------------------------------------------
# Build local run command per model (--execution local)
# ---------------------------------------------------------------------------

# Models with a plain (non-Modal) local script — the only one with
# local_supported=True in shared/model_execution.py's MIDIGEN registry.
_LOCAL_SUPPORTED_MODELS = {"anticipatory"}


def _build_local_cmd(
    uv_bin: Path,
    script_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    adapted_prompt: str,
) -> list[str]:
    """Return the token list to run a model's plain local script directly
    (no `modal run`, no GPU). `script_path` points at the plain script
    (e.g. research_anticipatory.py) — its parent directory is the uv project
    (MidiGenerator/research/, with its own pyproject.toml/venv, distinct
    from the shared/ Modal-only project used by _build_cmd)."""

    model = args.model.lower()
    research_dir = script_path.parent

    base = [str(uv_bin), "run", "--project", str(research_dir), str(script_path)]

    if model == "anticipatory":
        if not args.melody or not Path(args.melody).exists():
            raise FileNotFoundError(
                f"Anticipatory requires the melody MIDI (--melody). "
                f"Specified path: {args.melody!r}"
            )
        out_path = out_dir / "generated.mid"
        cmd = base + [
            "--input",            args.melody,
            "--out",              str(out_path),
            "--prompt-length",    str(args.prompt_length),
            "--clip-length",      str(args.clip_length),
            "--melody-instrument", str(args.melody_instrument),
            "--multiplicity",     str(args.n_outputs),
        ] + _anticipatory_flags(args)

    else:
        # Defensive only: main() already rejects unsupported models via
        # shared/model_execution.py's is_local_supported() before reaching
        # here, so this should be unreachable.
        raise ValueError(f"Local execution not supported for model: {model!r}")

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
    p.add_argument("--execution",  choices=["local", "remote"], default="remote",
                   help="local runs the model's plain script directly on this "
                        "machine (only anticipatory supports it; see "
                        "shared/model_execution.py). remote (default) uses Modal, "
                        "unchanged from previous behavior.")

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

    # Anticipatory (AMT): melody file + accompaniment parameters
    p.add_argument("--melody",           default="")
    p.add_argument("--style",            choices=["pop", "pad", "arp"], default="pop")
    p.add_argument("--prompt-length",    type=int,   default=5, dest="prompt_length")
    p.add_argument("--clip-length",      type=int,   default=0, dest="clip_length")
    p.add_argument("--melody-instrument",type=int,   default=0, dest="melody_instrument")

    # Accompaniment controls (Anticipatory and MIDI-GPT)
    p.add_argument("--parts",            default="bass,keys",
                   help="comma list of: bass, keys, strings, guitar, drums")
    p.add_argument("--density",          type=int,   choices=[0, 1, 2], default=1,
                   help="0 sparse | 1 normal | 2 busy")
    p.add_argument("--bass-program",     type=int,   default=33, dest="bass_program")
    p.add_argument("--keys-program",     type=int,   default=4,  dest="keys_program")
    p.add_argument("--chords",           default="",
                   help="your own progression, e.g. 'C#m G# A# F' (Anticipatory only)")
    p.add_argument("--beats-per-chord",  type=int,   default=4,  dest="beats_per_chord")
    p.add_argument("--creativity",       type=float, default=0.5,
                   help="0..1: more variety (top_p for Anticipatory, temperature for MIDI-GPT)")
    p.add_argument("--stay-in-key",      action="store_true", dest="stay_in_key",
                   help="MIDI-GPT: only use the pitch classes of the melody")

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

    if kind == "text_dir" and not args.prompt.strip():
        write_progress(pf, "error", 0,
                       "This model requires a text prompt (--prompt).")
        return 1

    if kind == "melody_file" and not args.melody:
        write_progress(pf, "error", 0,
                       f"{_MELODY_ENGINES.get(model_key, model_key)} requires the melody MIDI "
                       "(select a MIDI item in REAPER and click R next to 'Melody').")
        return 1

    if args.execution == "local":
        sys.path.insert(0, str(shared_dir))
        from model_execution import is_local_supported  # noqa: E402
        supported, reason = is_local_supported("midigen", model_key)
        if not supported:
            write_progress(pf, "error", 0,
                           f"Local execution not supported for {model_key}: {reason}")
            return 1

    # --- Locate uv ----------------------------------------------------------
    uv_bin = _find_uv()
    if uv_bin is None:
        write_progress(pf, "error", 0,
                       "uv not found. Install with:\n"
                       "curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Adapt prompt -------------------------------------------------------
    fields: dict | None = None
    if kind == "text_dir" and any([
        args.field_key, args.field_bpm, args.field_instruments, args.field_chords,
    ]):
        fields = {
            "key":         args.field_key,
            "bpm":         args.field_bpm,
            "instruments": args.field_instruments,
            "chords":      args.field_chords,
        }

    adapted_prompt  = adapt(model_key, args.prompt, fields)

    # --- Build and launch run command ---------------------------------------
    if args.execution == "local":
        write_progress(pf, "running", 0.06,
                       f"Launching locally ({model_key})...")
    else:
        write_progress(pf, "running", 0.06,
                       f"Launching Modal ({model_key} · {args.gpu})...")

    try:
        if args.execution == "local":
            cmd = _build_local_cmd(uv_bin, script_path, out_dir, args, adapted_prompt)
        else:
            cmd = _build_cmd(
                uv_bin, shared_dir, script_path,
                out_dir, args, adapted_prompt,
            )
    except (FileNotFoundError, ValueError) as exc:
        write_progress(pf, "error", 0, str(exc))
        return 1

    env = os.environ.copy()
    print(f"[midigen] Command: {' '.join(cmd)}", flush=True)
    print(f"[midigen] Model: {model_key}  kind={kind}", flush=True)
    if kind == "text_dir":
        print(f"[midigen] Adapted prompt: {adapted_prompt[:120]}", flush=True)

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
    last_error_line: str | None = None
    last_warn_line: str | None = None
    # Rolling tail of raw subprocess output, written as "extra" progress
    # lines on failure so the REAPER log panel can show real context around
    # the failure instead of just one collapsed summary line — the single
    # captured ERROR:/[warn] line is a best-effort guess at THE cause, but
    # the surrounding lines (model response preview, per-candidate outcomes)
    # often matter too when the guess is incomplete.
    tail_lines: deque[str] = deque(maxlen=_LOG_TAIL_LINES)
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)
        tail_lines.append(line)

        stripped = line.strip()
        if stripped.startswith("ERROR:"):
            last_error_line = stripped[len("ERROR:"):].strip()
        elif stripped.startswith("[warn]"):
            last_warn_line = stripped[len("[warn]"):].strip()

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
        # AMT reports progress differently; keep incremental pct
        elif "spawned" in line.lower() or "candidate" in line.lower():
            pct = 0.15
        elif "saved" in line.lower() or "→" in line:
            pct = 0.90

        write_progress(pf, "running", pct, line[:80])

    proc.wait()

    if proc.returncode != 0:
        write_progress(pf, "error", 0,
                       _pick_error_message(proc.returncode, out_dir,
                                           last_error_line, last_warn_line),
                       extra=list(tail_lines))
        return 1

    # --- Collect outputs ---------------------------------------------------
    write_progress(pf, "running", 0.95, "Collecting MIDI files...")

    mids = _collect_outputs(out_dir)
    if not mids:
        write_progress(pf, "error", 0,
                       _pick_error_message(proc.returncode, out_dir,
                                           last_error_line, last_warn_line),
                       extra=list(tail_lines))
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
