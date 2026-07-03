#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend CLI para Text2Audio.lua — genera/edita audio vía Modal cloud.

Llamado por Text2Audio.lua como proceso en background:
    python3 text2audio.py
        --shared-dir <shared/>  --script <research_*.py>
        --model <sao|foundation1|sao_edit|acestep|musicgen>
        --mode  <generate|edit>
        --out-dir <dir>  --progress <pf>
        [--prompt <texto>]
        [--seconds <float>]
        [--input <path>]                           # audio fuente (modo edit)
        [--start <float>] [--duration <float>]     # sección del source
        [--intensity <subtle|moderate|strong>]     # modo edit
        [--gpu <A10G|A100|T4>]

Contratos de entrypoint por modelo (modal run <script>::main):
  text2audio (sao, foundation1):
      --prompt  --seconds  --out-dir  --force
  audio_edit_noise (sao_edit):
      --source-audio  --prompt  --noise-level  --out-dir  --force
  audio_edit_strength (acestep):
      --source-audio  --prompt  --strength  --out-dir  --force
  melody_edit (musicgen):
      --source-audio  --prompt  --seconds  --out-dir  --force

Protocolo de progreso (--progress):
    state|pct|msg\\n       state = running | done | error
    [extra-lines]          ruta .wav del output
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
# Tipos de contrato por modelo
# ---------------------------------------------------------------------------
MODEL_KIND: dict[str, str] = {
    "sao":         "text2audio",
    "foundation1": "text2audio",
    "sao_edit":    "audio_edit_noise",
    "acestep":     "audio_edit_strength",
    "musicgen":    "melody_edit",
}

# Mapeo intensity → valor numérico
INTENSITY_MAP: dict[str, dict[str, float]] = {
    "sao_edit": {"subtle": 0.4, "moderate": 1.0, "strong": 4.0},
    "acestep":  {"subtle": 0.9, "moderate": 0.7, "strong": 0.4},
}


# ---------------------------------------------------------------------------
# Protocolo de progreso
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
            "soundfile no instalado. pip install soundfile"
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
    p = argparse.ArgumentParser(description="Genera o edita audio vía Modal cloud.")
    p.add_argument("--shared-dir",  required=True, dest="shared_dir",
                   help="Directorio shared/ con pyproject.toml.")
    p.add_argument("--script",      required=True,
                   help="Ruta al script Modal a ejecutar.")
    p.add_argument("--model",       required=True,
                   help="Clave del modelo (sao|foundation1|sao_edit|acestep|musicgen).")
    p.add_argument("--mode",        required=True, choices=["generate", "edit"],
                   help="Modo de operación.")
    p.add_argument("--out-dir",     required=True, dest="out_dir",
                   help="Directorio de salida para el audio generado.")
    p.add_argument("--progress",    default="",
                   help="Ruta al archivo de progreso.")
    p.add_argument("--prompt",      default="",
                   help="Prompt de texto (descripción o intención del cambio).")
    p.add_argument("--seconds",     type=float, default=8.0,
                   help="Duración objetivo en segundos (modo generate).")
    p.add_argument("--input",       default="",
                   help="Ruta al audio fuente (modo edit).")
    p.add_argument("--start",       type=float, default=None,
                   help="Inicio de la sección del source (segundos).")
    p.add_argument("--duration",    type=float, default=None,
                   help="Duración de la sección del source (segundos).")
    p.add_argument("--intensity",   default="moderate",
                   choices=["subtle", "moderate", "strong"],
                   help="Intensidad de la transformación (modo edit).")
    p.add_argument("--gpu",         default="A10G",
                   help="GPU para Modal (A10G|A100|T4).")
    args = p.parse_args()

    pf = args.progress
    write_progress(pf, "running", 0.02, "Preparando...")

    shared_dir  = Path(args.shared_dir)
    script_path = Path(args.script)
    out_dir     = Path(args.out_dir)
    kind        = MODEL_KIND.get(args.model)

    # ---- Validaciones -------------------------------------------------------
    if not script_path.exists():
        write_progress(pf, "error", 0, f"Script no encontrado: {script_path}")
        return 1

    if not (shared_dir / "pyproject.toml").exists():
        write_progress(pf, "error", 0,
                       f"shared/pyproject.toml no encontrado en: {shared_dir}")
        return 1

    if kind is None:
        write_progress(pf, "error", 0, f"Modelo desconocido: {args.model}")
        return 1

    if not args.prompt.strip():
        write_progress(pf, "error", 0, "Prompt vacío. Escribe una descripción del audio.")
        return 1

    uv_bin = _find_uv()
    if uv_bin is None:
        write_progress(pf, "error", 0,
                       "uv no encontrado. "
                       "Instala con: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1

    # ---- Extraer sección del source si aplica ------------------------------
    temp_dir     = None
    process_path = Path(args.input) if args.input else None

    if process_path:
        if not process_path.exists():
            write_progress(pf, "error", 0,
                           f"Audio fuente no encontrado: {process_path}")
            return 1
        if args.start is not None and args.duration is not None:
            write_progress(pf, "running", 0.05,
                           f"Extrayendo sección {args.start:.2f}s → "
                           f"{args.start + args.duration:.2f}s...")
            try:
                temp_dir     = tempfile.mkdtemp(prefix="t2a_")
                section_file = Path(temp_dir) / process_path.name
                extract_section(process_path, args.start, args.duration, section_file)
                process_path = section_file
                print(f"[text2audio] Sección extraída: {section_file}", flush=True)
            except Exception as exc:
                write_progress(pf, "error", 0, f"Error extrayendo sección: {exc}")
                return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Construir comando modal run ----------------------------------------
    write_progress(pf, "running", 0.08,
                   f"Lanzando Modal ({args.model} · {args.gpu})...")

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

    # Propagar GPU al entorno (usado por SAO y SAO-edit via env vars)
    env = os.environ.copy()
    env["STABLE_AUDIO_GPU"] = args.gpu

    print(f"[text2audio] Comando: {' '.join(cmd)}", flush=True)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
    except FileNotFoundError:
        write_progress(pf, "error", 0, f"No se pudo ejecutar: {uv_bin}")
        return 1
    except PermissionError:
        write_progress(pf, "error", 0, f"Permiso denegado: {uv_bin}")
        return 1

    # ---- Parsear stdout para progreso --------------------------------------
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)

        pct = 0.1
        # Difusión: "step N/M" (SAO, Foundation-1, MusicGen) o "[N/M]"
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
                       f"Modal falló con código {proc.returncode}. "
                       "Revisa el log para más detalles.")
        return 1

    # ---- Localizar output.wav ----------------------------------------------
    raw_wav = out_dir / "output.wav"
    if not raw_wav.exists():
        write_progress(pf, "error", 0,
                       f"output.wav no encontrado en {out_dir}. "
                       "El proceso terminó sin error pero sin salida.")
        return 1

    # Renombrar a nombre descriptivo
    if args.mode == "generate":
        final_name = f"generated__{args.model}.wav"
    else:
        src_stem   = Path(args.input).stem if args.input else "audio"
        final_name = f"{src_stem}__{args.model}_edit.wav"

    final_wav = out_dir / final_name
    try:
        raw_wav.rename(final_wav)
    except Exception:
        final_wav = raw_wav

    done_msg = "Completado — audio listo"
    write_progress(pf, "done", 1.0, done_msg, [str(final_wav)])
    print(f"[text2audio] {done_msg}: {final_wav}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
