#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM Audio (Modal cloud) CLI backend for Stem Separator.

Called via ``python separate_sam.py`` from StemSeparator.lua.
Internally uses ``uv run --project SAM_DIR modal run modal_sam_audio.py``
so uv manages the venv, dependencies (modal), and PATH — avoiding
the REAPER macOS GUI minimal-PATH problem entirely.

The target Modal script lives alongside this file in StemsSeparator/
and produces ``_target.wav`` + ``_residual.wav`` per run.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


# Name of the Modal entrypoint script (same directory as this file)
MODAL_SCRIPT = "modal_sam_audio.py"


def _find_uv():
    """Locate the uv binary. Check PATH first, then ~/.local/bin/uv."""
    uv = shutil.which("uv")
    if uv:
        return Path(uv)
    fallback = Path.home() / ".local" / "bin" / "uv"
    if fallback.is_file():
        return fallback
    return None


def write_progress(path, state, pct, msg, files=None):
    if not path:
        return
    try:
        with open(path, "w") as f:
            f.write(f"{state}|{pct:.3f}|{msg}\n")
            if files:
                for fp in files:
                    f.write(fp + "\n")
    except Exception:
        pass


def extract_section(src_path: Path, start: float, duration: float, dst_path: Path):
    """Write a time slice of src_path to dst_path using soundfile."""
    import soundfile as sf
    with sf.SoundFile(str(src_path)) as f:
        sr = f.samplerate
        start_frame = int(start * sr)
        num_frames = int(duration * sr)
        f.seek(start_frame)
        data = f.read(num_frames, dtype="float32", always_2d=True)
    sf.write(str(dst_path), data, sr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", default="vocals")
    p.add_argument("--sam-dir", default="",
                   help="Directory containing pyproject.toml and "
                        "modal_sam_audio.py (default: same directory as "
                        "this script).")
    p.add_argument("--model", default="facebook/sam-audio-large")
    p.add_argument("--gpu", default="A100")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--ode-method", default="midpoint", dest="ode_method")
    p.add_argument("--chunk", type=float, default=15.0)
    p.add_argument("--overlap", type=float, default=2.0)
    p.add_argument("--confidence", type=float, default=0.0)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--predict-spans", default=True, dest="predict_spans",
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--internal-candidates", type=int, default=2,
                   dest="internal_candidates")
    p.add_argument("--start", type=float, default=None,
                   help="Start time within source file (seconds)")
    p.add_argument("--duration", type=float, default=None,
                   help="Duration of section to process (seconds)")
    p.add_argument("--outdir", default="separated")
    p.add_argument("--progress", default="")
    args = p.parse_args()

    pf = args.progress
    write_progress(pf, "running", 0.02, "Preparando entorno Modal via uv...")

    input_path = Path(args.input)

    # Extract section to a temp file when start/duration are provided.
    temp_dir = None
    process_path = input_path
    if args.start is not None and args.duration is not None:
        write_progress(pf, "running", 0.01,
                       f"Extrayendo sección {args.start:.2f}s → "
                       f"{args.start + args.duration:.2f}s...")
        try:
            temp_dir = tempfile.mkdtemp(prefix="stemsep_")
            section_file = Path(temp_dir) / input_path.name
            extract_section(input_path, args.start, args.duration, section_file)
            process_path = section_file
            print(f"Section extracted to: {section_file}", flush=True)
        except Exception as exc:
            write_progress(pf, "error", 0, f"Error extrayendo sección: {exc}")
            return 1

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Snapshot existing wav files so we can detect only the new ones after the run
    existing_wavs = set(str(f) for f in out_path.glob("*.wav"))

    # --- locate uv binary ----------------------------------------------------
    uv_bin = _find_uv()
    if uv_bin is None:
        write_progress(pf, "error", 0,
                       "No encontrado: 'uv' no esta en el PATH ni en "
                       "~/.local/bin/uv. Instala uv con: "
                       "curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1

    # --- locate project root and Modal script --------------------------------
    sam_dir = Path(args.sam_dir) if args.sam_dir else Path.cwd()
    sam_script = sam_dir / MODAL_SCRIPT
    if not sam_script.exists():
        write_progress(pf, "error", 0,
                       f"No encontrado: {sam_script}. "
                       f"Revisa que SAM_DIR/{MODAL_SCRIPT} existe.")
        return 1

    # --- build command -------------------------------------------------------
    # --gpu is not a CLI flag; set SAM_AUDIO_GPU env var instead
    env = os.environ.copy()
    env["SAM_AUDIO_GPU"] = args.gpu
    cmd = [
        str(uv_bin), "run", "--project", str(sam_dir),
        "modal", "run", f"{sam_script}::main",
    ]
    # Typer args (--input, --output-dir, etc.)
    cmd += [
        "--input", str(process_path),
        "--output-dir", str(out_path),
        "--prompt", args.prompt,
        "--model", args.model,
        "--chunk-dur", str(args.chunk),
        "--overlap", str(args.overlap),
        "--ode-steps", str(args.steps),
        "--ode-method", args.ode_method,
    ]
    if args.confidence > 0:
        cmd += ["--confidence-threshold", str(args.confidence)]
    if args.candidates > 1:
        cmd += ["--reranking-candidates", str(args.candidates)]
    cmd += ["--predict-spans" if args.predict_spans else "--no-predict-spans"]
    cmd += ["--internal-candidates", str(args.internal_candidates)]

    # --- run -----------------------------------------------------------------
    print("Running:", " ".join(cmd), flush=True)
    write_progress(pf, "running", 0.05,
                   f"Lanzando uv run sobre {MODAL_SCRIPT}...")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
    except FileNotFoundError:
        write_progress(pf, "error", 0,
                       f"No encontrado: '{uv_bin}' no se pudo ejecutar.")
        return 1
    except PermissionError:
        write_progress(pf, "error", 0,
                       f"Permiso denegado: '{uv_bin}' no es ejecutable.")
        return 1

    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)
        pct = 0.1
        if "Chunk" in line and "/" in line:
            try:
                chunk_part = line.split("Chunk")[1].split(":")[0].strip()
                n, m = int(chunk_part.split("/")[0]), int(chunk_part.split("/")[1])
                pct = 0.1 + (n / m) * 0.8
            except (ValueError, IndexError):
                pass
        write_progress(pf, "running", pct, line[:80])

    proc.wait()
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if proc.returncode != 0:
        write_progress(pf, "error", 0,
                       f"SAM Audio fallo con codigo {proc.returncode}")
        return 1

    # --- discover output files -----------------------------------------------
    # Only report files created by this run (not leftovers from previous runs)
    all_wavs = set(str(f) for f in out_path.glob("*.wav"))
    output_files = sorted(all_wavs - existing_wavs)

    if not output_files:
        # Fallback: any wav matching this stem + prompt (avoids cross-run pollution)
        stem_name = process_path.stem
        prompt_tag = args.prompt.replace(" ", "_")
        output_files = sorted(
            str(f) for f in out_path.glob(f"{stem_name}_sam_*_{prompt_tag}_*.wav")
        )

    write_progress(pf, "done", 1.0,
                   f"Completado. {len(output_files)} archivos generados.",
                   output_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())