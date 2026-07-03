#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend CLI for Audio2Midi.lua — transcribes audio to MIDI via Modal cloud.

Called by Audio2Midi.lua as a background process:
    python3 transcribe.py --shared-dir <shared/> --script <research_*.py>
        --input <audio> --out-dir <dir> --model <miros|yourmt3>
        --gpu <A10G|A100|T4> [--start S] [--duration D]
        [--no-beat-tracking] --progress <pf> >> <log> 2>&1 &

The script:
  1. Extracts a section (--start/--duration) if needed (via soundfile).
  2. Runs:  uv run --project <shared-dir> modal run <script>::main
              --audio-path <audio> --out-dir <out-dir> --force
             (adds --no-beat-tracking if the model supports it and it was requested)
  3. Renames the output  transcribed_cuda.mid  →  <src_stem>__<model>.mid
  4. Counts non-empty instruments (pretty_midi) and reports:
       done|1.0|... <midi-path>
       <midi-path>
       INSTRUMENTS|<n>

Progress protocol (written to --progress):
    state|pct|msg\n          state = running | done | error
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Models that accept --no-beat-tracking in their Modal entrypoint
SUPPORTS_NO_BEAT_TRACKING = {"miros"}


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
    """Locate the uv binary: check PATH first, then ~/.local/bin/uv."""
    uv = shutil.which("uv")
    if uv:
        return Path(uv)
    fallback = Path.home() / ".local" / "bin" / "uv"
    return fallback if fallback.is_file() else None


def extract_section(src_path: Path, start: float, duration: float,
                    dst_path: Path) -> None:
    """Write a time slice of src_path to dst_path using soundfile."""
    try:
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "soundfile is not installed. "
            "Install with: pip install soundfile"
        )
    with sf.SoundFile(str(src_path)) as f:
        sr = f.samplerate
        start_frame = int(start * sr)
        num_frames = int(duration * sr)
        f.seek(start_frame)
        data = f.read(num_frames, dtype="float32", always_2d=True)
    sf.write(str(dst_path), data, sr)


def count_instruments(mid_path: Path) -> int:
    """Count non-empty MIDI instruments (excludes drums on channel 9).
    Uses pretty_midi if available; returns -1 otherwise."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(mid_path))
        return sum(1 for inst in pm.instruments if inst.notes)
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Transcribe audio to MIDI via Modal cloud."
    )
    p.add_argument("--input",      required=True,
                   help="Path to the input audio file.")
    p.add_argument("--out-dir",    required=True, dest="out_dir",
                   help="Directory where the resulting MIDI is written.")
    p.add_argument("--shared-dir", required=True, dest="shared_dir",
                   help="shared/ directory containing pyproject.toml (modal CLI).")
    p.add_argument("--script",     required=True,
                   help="Path to the Modal script to run (research_*_modal.py).")
    p.add_argument("--model",      default="miros",
                   help="Model name (miros | yourmt3).")
    p.add_argument("--gpu",        default="A10G",
                   help="GPU to use on Modal (A10G | A100 | T4).")
    p.add_argument("--start",      type=float, default=None,
                   help="Start of the section to transcribe (seconds).")
    p.add_argument("--duration",   type=float, default=None,
                   help="Duration of the section (seconds).")
    p.add_argument("--no-beat-tracking", action="store_true",
                   dest="no_beat_tracking",
                   help="Disable beat tracking (only for models that support it).")
    p.add_argument("--progress",   default="",
                   help="Path to the progress file (state|pct|msg protocol).")
    args = p.parse_args()

    pf = args.progress
    write_progress(pf, "running", 0.02, "Preparing transcription...")

    input_path  = Path(args.input)
    out_dir     = Path(args.out_dir)
    shared_dir  = Path(args.shared_dir)
    script_path = Path(args.script)

    if not input_path.exists():
        write_progress(pf, "error", 0, f"File not found: {input_path}")
        return 1

    if not script_path.exists():
        write_progress(pf, "error", 0,
                       f"Modal script not found: {script_path}\n"
                       "Check the model path in Audio2Midi/research/.")
        return 1

    if not (shared_dir / "pyproject.toml").exists():
        write_progress(pf, "error", 0,
                       f"shared/pyproject.toml not found at: {shared_dir}\n"
                       "Check that shared/ is next to the plugins.")
        return 1

    # --- Locate uv -----------------------------------------------------------
    uv_bin = _find_uv()
    if uv_bin is None:
        write_progress(pf, "error", 0,
                       "uv not found in PATH or ~/.local/bin/uv.\n"
                       "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1

    # --- Extract section if specified ----------------------------------------
    temp_dir    = None
    process_path = input_path

    if args.start is not None and args.duration is not None:
        write_progress(pf, "running", 0.05,
                       f"Extracting section {args.start:.2f}s → "
                       f"{args.start + args.duration:.2f}s...")
        try:
            temp_dir = tempfile.mkdtemp(prefix="a2m_")
            section_file = Path(temp_dir) / input_path.name
            extract_section(input_path, args.start, args.duration, section_file)
            process_path = section_file
            print(f"[transcribe] Section extracted: {section_file}", flush=True)
        except Exception as exc:
            write_progress(pf, "error", 0, f"Error extracting section: {exc}")
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build modal run command ----------------------------------------------
    write_progress(pf, "running", 0.08,
                   f"Launching Modal ({args.model} · {args.gpu})...")

    cmd = [
        str(uv_bin), "run", "--project", str(shared_dir),
        "modal", "run", f"{script_path}::main",
        "--audio-path", str(process_path),
        "--out-dir",    str(out_dir),
        "--force",
    ]

    # --no-beat-tracking only if the model supports it
    if args.no_beat_tracking and args.model.lower() in SUPPORTS_NO_BEAT_TRACKING:
        cmd.append("--no-beat-tracking")

    env = os.environ.copy()

    print(f"[transcribe] Command: {' '.join(cmd)}", flush=True)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
    except FileNotFoundError:
        write_progress(pf, "error", 0, f"Could not execute: {uv_bin}")
        return 1
    except PermissionError:
        write_progress(pf, "error", 0, f"Permission denied: {uv_bin}")
        return 1

    # --- Parse stdout for progress -------------------------------------------
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)

        pct = 0.1
        # Pattern "[n/m]" in research_*_modal.py
        m = re.search(r'\[(\d+)/(\d+)\]', line)
        if m:
            n, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                pct = 0.1 + (n / total) * 0.8
        # Pattern "Chunk n/m" (MIROS)
        elif "Chunk" in line and "/" in line:
            try:
                part = line.split("Chunk")[1].split(":")[0].strip()
                n, total = int(part.split("/")[0]), int(part.split("/")[1])
                if total > 0:
                    pct = 0.1 + (n / total) * 0.8
            except (ValueError, IndexError):
                pass

        write_progress(pf, "running", pct, line[:80])

    proc.wait()

    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if proc.returncode != 0:
        write_progress(pf, "error", 0,
                       f"Modal failed with code {proc.returncode}. "
                       "Check the log for details.")
        return 1

    # --- Locate generated MIDI -----------------------------------------------
    raw_mid = out_dir / "transcribed_cuda.mid"
    if not raw_mid.exists():
        write_progress(pf, "error", 0,
                       f"MIDI not found at {out_dir}/transcribed_cuda.mid. "
                       "The process ended without error but produced no output.")
        return 1

    # Rename to <src_stem>__<model>.mid
    src_stem = input_path.stem
    final_name = f"{src_stem}__{args.model}.mid"
    final_mid  = out_dir / final_name
    try:
        raw_mid.rename(final_mid)
    except Exception:
        final_mid = raw_mid  # fallback: keep original name

    # Count non-empty instruments
    n_instruments = count_instruments(final_mid)
    instr_str = str(n_instruments) if n_instruments >= 0 else "?"

    n_label = (f"{n_instruments} instrument{'s' if n_instruments != 1 else ''}"
               if n_instruments >= 0 else "MIDI generated")
    done_msg = f"Completed — {n_label}"
    write_progress(pf, "done", 1.0, done_msg, [
        str(final_mid),
        f"INSTRUMENTS|{instr_str}",
    ])
    print(f"[transcribe] {done_msg}: {final_mid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
