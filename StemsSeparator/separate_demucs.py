#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demucs CLI backend for Stem Separator. Called by StemSeparator.lua."""
import argparse
import subprocess
import sys
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--stems", default="vocals,drums,bass,other")
    p.add_argument("--outdir", default="separated")
    p.add_argument("--device", default="cpu",
                   help="Device for demucs: cpu or cuda")
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

    python_exe = args.python or sys.executable
    write_progress(pf, "running", 0.02, "Iniciando Demucs...")

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe, "-m", "demucs",
        "-n", args.model,
        "-o", str(out_path),
        "-d", args.device,
        args.input,
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
    if proc.returncode != 0:
        write_progress(
            pf, "error", 0,
            f"Demucs fallo con codigo {proc.returncode}",
        )
        return 1

    input_stem = Path(args.input).stem
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
