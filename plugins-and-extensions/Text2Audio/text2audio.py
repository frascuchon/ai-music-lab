#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend CLI for Text2Audio.lua — generates/edits audio via Modal cloud.

Called by Text2Audio.lua as a background process:
    python3 text2audio.py
        --shared-dir <shared/>  --script <research_*.py>
        --model <sao|foundation1|sao_edit|acestep|musicgen|...>
        --mode  <generate|edit>
        --out-dir <dir>  --progress <pf>
        [--prompt <text>]
        [--seconds <float>]
        [--input <path>]                           # source audio (edit mode)
        [--start <float>] [--duration <float>]     # section of source
        [--intensity <subtle|moderate|strong>]     # edit mode
        [--gpu <A10G|A100|T4>]

Model entrypoint contracts (modal run <script>::main):
  text2audio (sao, foundation1, acestep_gen, inspiremusic_gen,
              mustango, audiogen, musicgen_gen, magnet):
      --prompt  --seconds  --out-dir  --force
  audio_edit_noise (sao_edit):
      --source-audio  --prompt  --noise-level  --out-dir  --force
  audio_edit_strength (acestep):
      --source-audio  --prompt  --strength  --out-dir  --force
  melody_edit (musicgen):
      --source-audio  --prompt  --seconds  --out-dir  --force
  audio_edit_flowstep (melodyflow):
      --source-audio  --prompt  --flowstep  --out-dir  --force
  audio_edit_tstart (zeta):
      --source-audio  --prompt  --tstart  --out-dir  --force
  audio_continuation (inspiremusic):
      --source-audio  --prompt  --out-dir  --force

Progress protocol (--progress):
    state|pct|msg\\n       state = running | done | error
    [extra-lines]          path to output .wav
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Model contract type per model key
# ---------------------------------------------------------------------------
MODEL_KIND: dict[str, str] = {
    # Generation (text → audio, no source)
    "sao":             "text2audio",
    "foundation1":     "text2audio",
    "acestep_gen":     "text2audio",
    "inspiremusic_gen":"text2audio",
    "mustango":        "text2audio",
    "audiogen":        "text2audio",
    "musicgen_gen":    "text2audio",
    "magnet":          "text2audio",
    # Editing (source audio + prompt → transformed audio)
    "sao_edit":    "audio_edit_noise",
    "acestep":     "audio_edit_strength",
    "musicgen":    "melody_edit",
    "melodyflow":  "audio_edit_flowstep",
    "zeta":        "audio_edit_tstart",
    "inspiremusic":"audio_continuation",
}

# Intensity → numeric value mapping
INTENSITY_MAP: dict[str, dict[str, float]] = {
    "sao_edit":   {"subtle": 0.4,  "moderate": 1.0,  "strong": 4.0},
    "acestep":    {"subtle": 0.9,  "moderate": 0.7,  "strong": 0.4},
    "melodyflow": {"subtle": 0.10, "moderate": 0.05, "strong": 0.0},
    "zeta":       {"subtle": 70,   "moderate": 100,  "strong": 130},
}


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


def extract_section(src_path: Path, start: float, duration: float,
                    dst_path: Path) -> None:
    try:
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "soundfile not installed. pip install soundfile"
        )
    with sf.SoundFile(str(src_path)) as f:
        sr = f.samplerate
        start_frame = int(start * sr)
        num_frames  = int(duration * sr)
        f.seek(start_frame)
        data = f.read(num_frames, dtype="float32", always_2d=True)
    sf.write(str(dst_path), data, sr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Generate or edit audio via Modal cloud.")
    p.add_argument("--shared-dir",  required=True, dest="shared_dir",
                   help="shared/ directory containing pyproject.toml.")
    p.add_argument("--script",      required=True,
                   help="Path to the Modal script to run.")
    p.add_argument("--model",       required=True,
                   help="Model key (sao|foundation1|acestep_gen|inspiremusic_gen|"
                        "mustango|audiogen|musicgen_gen|magnet|"
                        "sao_edit|acestep|musicgen|melodyflow|zeta|inspiremusic).")
    p.add_argument("--mode",        required=True, choices=["generate", "edit"],
                   help="Operation mode.")
    p.add_argument("--out-dir",     required=True, dest="out_dir",
                   help="Output directory for the generated audio.")
    p.add_argument("--progress",    default="",
                   help="Path to the progress file.")
    p.add_argument("--prompt",      default="",
                   help="Text prompt (description or change intent).")
    p.add_argument("--seconds",     type=float, default=8.0,
                   help="Target duration in seconds (generate mode).")
    p.add_argument("--input",       default="",
                   help="Path to the source audio (edit mode).")
    p.add_argument("--start",       type=float, default=None,
                   help="Start of the source section (seconds).")
    p.add_argument("--duration",    type=float, default=None,
                   help="Duration of the source section (seconds).")
    p.add_argument("--intensity",   default="moderate",
                   choices=["subtle", "moderate", "strong"],
                   help="Transformation intensity (edit mode).")
    p.add_argument("--gpu",         default="A10G",
                   help="GPU for Modal (A10G|A100|T4).")
    args = p.parse_args()

    pf = args.progress
    write_progress(pf, "running", 0.02, "Preparing...")

    shared_dir  = Path(args.shared_dir)
    script_path = Path(args.script)
    out_dir     = Path(args.out_dir)
    kind        = MODEL_KIND.get(args.model)

    # ---- Validations -------------------------------------------------------
    if not script_path.exists():
        write_progress(pf, "error", 0, f"Script not found: {script_path}")
        return 1

    if not (shared_dir / "pyproject.toml").exists():
        write_progress(pf, "error", 0,
                       f"shared/pyproject.toml not found at: {shared_dir}")
        return 1

    if kind is None:
        write_progress(pf, "error", 0, f"Unknown model: {args.model}")
        return 1

    if not args.prompt.strip():
        write_progress(pf, "error", 0, "Empty prompt. Write a description of the audio.")
        return 1

    uv_bin = _find_uv()
    if uv_bin is None:
        write_progress(pf, "error", 0,
                       "uv not found. "
                       "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1

    # ---- Extract source section if applicable ------------------------------
    temp_dir     = None
    process_path = Path(args.input) if args.input else None

    if process_path:
        if not process_path.exists():
            write_progress(pf, "error", 0,
                           f"Source audio not found: {process_path}")
            return 1
        if args.start is not None and args.duration is not None:
            write_progress(pf, "running", 0.05,
                           f"Extracting section {args.start:.2f}s → "
                           f"{args.start + args.duration:.2f}s...")
            try:
                temp_dir     = tempfile.mkdtemp(prefix="t2a_")
                section_file = Path(temp_dir) / process_path.name
                extract_section(process_path, args.start, args.duration, section_file)
                process_path = section_file
                print(f"[text2audio] Section extracted: {section_file}", flush=True)
            except Exception as exc:
                write_progress(pf, "error", 0, f"Error extracting section: {exc}")
                return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build modal run command -------------------------------------------
    write_progress(pf, "running", 0.08,
                   f"Launching Modal ({args.model} · {args.gpu})...")

    cmd = [
        str(uv_bin), "run", "--project", str(shared_dir),
        "modal", "run", f"{script_path}::main",
    ]

    if kind == "text2audio":
        cmd += [
            "--prompt",  args.prompt,
            "--seconds", str(args.seconds),
            "--out-dir", str(out_dir),
            "--force",
        ]
    elif kind == "audio_edit_noise":
        noise_level = INTENSITY_MAP["sao_edit"].get(args.intensity, 1.0)
        cmd += [
            "--source-audio", str(process_path),
            "--prompt",       args.prompt,
            "--noise-level",  str(noise_level),
            "--out-dir",      str(out_dir),
            "--force",
        ]
    elif kind == "audio_edit_strength":
        strength = INTENSITY_MAP["acestep"].get(args.intensity, 0.7)
        cmd += [
            "--source-audio", str(process_path),
            "--prompt",       args.prompt,
            "--strength",     str(strength),
            "--out-dir",      str(out_dir),
            "--force",
        ]
    elif kind == "melody_edit":
        cmd += [
            "--source-audio", str(process_path),
            "--prompt",       args.prompt,
            "--seconds",      str(args.seconds),
            "--out-dir",      str(out_dir),
            "--force",
        ]
    elif kind == "audio_edit_flowstep":
        flowstep = INTENSITY_MAP["melodyflow"].get(args.intensity, 0.05)
        cmd += [
            "--source-audio", str(process_path),
            "--prompt",       args.prompt,
            "--flowstep",     str(flowstep),
            "--out-dir",      str(out_dir),
            "--force",
        ]
    elif kind == "audio_edit_tstart":
        tstart = int(INTENSITY_MAP["zeta"].get(args.intensity, 100))
        cmd += [
            "--source-audio", str(process_path),
            "--prompt",       args.prompt,
            "--tstart",       str(tstart),
            "--out-dir",      str(out_dir),
            "--force",
        ]
    elif kind == "audio_continuation":
        cmd += [
            "--source-audio", str(process_path),
            "--prompt",       args.prompt,
            "--out-dir",      str(out_dir),
            "--force",
        ]

    # Propagate GPU to the environment (used by SAO and SAO-edit via env vars)
    env = os.environ.copy()
    env["STABLE_AUDIO_GPU"] = args.gpu

    print(f"[text2audio] Command: {' '.join(cmd)}", flush=True)

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

    # ---- Parse stdout for progress ----------------------------------------
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)

        pct = 0.1
        # Diffusion: "step N/M" (SAO, Foundation-1, MusicGen) or "[N/M]"
        m = re.search(r'step\s+(\d+)/(\d+)', line, re.IGNORECASE)
        if not m:
            m = re.search(r'\[(\d+)/(\d+)\]', line)
        if m:
            n, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                pct = 0.1 + (n / total) * 0.8

        write_progress(pf, "running", pct, line[:80])

    proc.wait()

    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if proc.returncode != 0:
        write_progress(pf, "error", 0,
                       f"Modal failed with code {proc.returncode}. "
                       "Check the log for details.")
        return 1

    # ---- Locate output.wav -------------------------------------------------
    raw_wav = out_dir / "output.wav"
    if not raw_wav.exists():
        write_progress(pf, "error", 0,
                       f"output.wav not found in {out_dir}. "
                       "The process ended without error but produced no output.")
        return 1

    # Rename to a descriptive name including a fragment of the prompt
    prompt_slug = re.sub(r'[^a-zA-Z0-9]+', '_', args.prompt.strip())[:40].strip('_').lower()
    if args.mode == "generate":
        final_name = f"generated__{args.model}__{prompt_slug}.wav"
    else:
        src_stem   = Path(args.input).stem if args.input else "audio"
        final_name = f"{src_stem}__{args.model}_edit__{prompt_slug}.wav"

    final_wav = out_dir / final_name
    try:
        raw_wav.rename(final_wav)
    except Exception:
        final_wav = raw_wav

    done_msg = "Completed — audio ready"
    write_progress(pf, "done", 1.0, done_msg, [str(final_wav)])
    print(f"[text2audio] {done_msg}: {final_wav}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
