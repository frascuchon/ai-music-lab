#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demucs CLI backend for Stem Separator. Called by StemSeparator.lua."""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


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
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--stems", default="vocals,drums,bass,other")
    p.add_argument("--outdir", default="separated")
    p.add_argument("--device", default="cpu",
                   help="Device for demucs: cpu or cuda")
    p.add_argument("--start", type=float, default=None,
                   help="Start time within source file (seconds)")
    p.add_argument("--duration", type=float, default=None,
                   help="Duration of section to process (seconds)")
    p.add_argument("--progress")
    p.add_argument("--python",
                   help="Python executable to use for demucs subprocess. "
                        "Defaults to the interpreter running this script.")
    args = p.parse_args()

    pf = args.progress or ""

    input_path = Path(args.input)
    if not input_path.is_file():
        write_progress(pf, "error", 0,
                       f"Input file does not exist: {input_path}")
        return 1

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

    python_exe = args.python or sys.executable
    write_progress(pf, "running", 0.02, "Iniciando Demucs...")

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe, "-m", "demucs",
        "-n", args.model,
        "-o", str(out_path),
        "-d", args.device,
        str(process_path),
    ]
    print("Running:", " ".join(cmd), flush=True)

    write_progress(pf, "running", 0.05,
                   f"Separando con {args.model}...")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)
        pct = 0.5
        if "%" in line:
            try:
                pct = float(line.split("%")[0].split()[-1]) / 100.0
                pct = 0.05 + pct * 0.85
            except (ValueError, IndexError):
                pass
        write_progress(pf, "running", min(pct, 0.95), line[:80])

    proc.wait()
    if temp_dir:
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    if proc.returncode != 0:
        write_progress(
            pf, "error", 0,
            f"Demucs fallo con codigo {proc.returncode}",
        )
        return 1

    input_stem = process_path.stem
    stem_dir = out_path / args.model / input_stem
    selected = [s.strip() for s in args.stems.split(",")]
    output_files = []
    for s in selected:
        sf = stem_dir / f"{s}.wav"
        if sf.exists():
            output_files.append(str(sf))
        else:
            print(f"Aviso: no encontrado {sf}", flush=True)

    write_progress(
        pf, "done", 1.0,
        f"Completado. {len(output_files)} stems extraidos.",
        output_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
