#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend CLI para Audio2Midi.lua — transcribe audio a MIDI via Modal cloud.

Llamado por Audio2Midi.lua como proceso en background:
    python3 transcribe.py --shared-dir <shared/> --script <research_*.py>
        --input <audio> --out-dir <dir> --model <miros|yourmt3>
        --gpu <A10G|A100|T4> [--start S] [--duration D]
        [--no-beat-tracking] --progress <pf> >> <log> 2>&1 &

El script:
  1. Extrae una sección (--start/--duration) si es necesario (vía soundfile).
  2. Lanza:  uv run --project <shared-dir> modal run <script>::main
              --audio-path <audio> --out-dir <out-dir> --force
             (añade --no-beat-tracking si el modelo lo soporta y se ha pedido)
  3. Renombra la salida  transcribed_cuda.mid  →  <src_stem>__<model>.mid
  4. Cuenta instrumentos no vacíos (pretty_midi) y reporta:
       done|1.0|... <ruta-midi>
       <ruta-midi>
       INSTRUMENTS|<n>

Protocolo de progreso (escrito en --progress):
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


# Modelos que aceptan --no-beat-tracking en su entrypoint Modal
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
    """Localiza el binario uv: primero en PATH, luego ~/.local/bin/uv."""
    uv = shutil.which("uv")
    if uv:
        return Path(uv)
    fallback = Path.home() / ".local" / "bin" / "uv"
    return fallback if fallback.is_file() else None


def extract_section(src_path: Path, start: float, duration: float,
                    dst_path: Path) -> None:
    """Escribe un fragmento de src_path en dst_path usando soundfile."""
    try:
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "soundfile no está instalado. "
            "Instala con: pip install soundfile"
        )
    with sf.SoundFile(str(src_path)) as f:
        sr = f.samplerate
        start_frame = int(start * sr)
        num_frames = int(duration * sr)
        f.seek(start_frame)
        data = f.read(num_frames, dtype="float32", always_2d=True)
    sf.write(str(dst_path), data, sr)


def count_instruments(mid_path: Path) -> int:
    """Cuenta instrumentos MIDI no vacíos (excluye batería en canal 9).
    Usa pretty_midi si está disponible; en caso contrario devuelve -1."""
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
        description="Transcribe audio a MIDI via Modal cloud."
    )
    p.add_argument("--input",      required=True,
                   help="Ruta al archivo de audio de entrada.")
    p.add_argument("--out-dir",    required=True, dest="out_dir",
                   help="Directorio donde se escribe el MIDI resultante.")
    p.add_argument("--shared-dir", required=True, dest="shared_dir",
                   help="Directorio shared/ que contiene pyproject.toml (modal CLI).")
    p.add_argument("--script",     required=True,
                   help="Ruta al script Modal a ejecutar (research_*_modal.py).")
    p.add_argument("--model",      default="miros",
                   help="Nombre del modelo (miros | yourmt3).")
    p.add_argument("--gpu",        default="A10G",
                   help="GPU a usar en Modal (A10G | A100 | T4).")
    p.add_argument("--start",      type=float, default=None,
                   help="Inicio de la sección a transcribir (segundos).")
    p.add_argument("--duration",   type=float, default=None,
                   help="Duración de la sección (segundos).")
    p.add_argument("--no-beat-tracking", action="store_true",
                   dest="no_beat_tracking",
                   help="Desactiva el beat tracking (solo modelos que lo soporten).")
    p.add_argument("--progress",   default="",
                   help="Ruta al archivo de progreso (protocolo state|pct|msg).")
    args = p.parse_args()

    pf = args.progress
    write_progress(pf, "running", 0.02, "Preparando transcripción...")

    input_path  = Path(args.input)
    out_dir     = Path(args.out_dir)
    shared_dir  = Path(args.shared_dir)
    script_path = Path(args.script)

    if not input_path.exists():
        write_progress(pf, "error", 0, f"Archivo no encontrado: {input_path}")
        return 1

    if not script_path.exists():
        write_progress(pf, "error", 0,
                       f"Script Modal no encontrado: {script_path}\n"
                       "Comprueba la ruta del modelo en Audio2Midi/research/.")
        return 1

    if not (shared_dir / "pyproject.toml").exists():
        write_progress(pf, "error", 0,
                       f"No encontrado shared/pyproject.toml en: {shared_dir}\n"
                       "Comprueba que shared/ está junto a los plugins.")
        return 1

    # --- Localizar uv --------------------------------------------------------
    uv_bin = _find_uv()
    if uv_bin is None:
        write_progress(pf, "error", 0,
                       "uv no encontrado en PATH ni en ~/.local/bin/uv.\n"
                       "Instala uv con: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1

    # --- Extraer sección si se especificó ------------------------------------
    temp_dir    = None
    process_path = input_path

    if args.start is not None and args.duration is not None:
        write_progress(pf, "running", 0.05,
                       f"Extrayendo sección {args.start:.2f}s → "
                       f"{args.start + args.duration:.2f}s...")
        try:
            temp_dir = tempfile.mkdtemp(prefix="a2m_")
            section_file = Path(temp_dir) / input_path.name
            extract_section(input_path, args.start, args.duration, section_file)
            process_path = section_file
            print(f"[transcribe] Sección extraída: {section_file}", flush=True)
        except Exception as exc:
            write_progress(pf, "error", 0, f"Error extrayendo sección: {exc}")
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Construir comando modal run -----------------------------------------
    write_progress(pf, "running", 0.08,
                   f"Lanzando Modal ({args.model} · {args.gpu})...")

    cmd = [
        str(uv_bin), "run", "--project", str(shared_dir),
        "modal", "run", f"{script_path}::main",
        "--audio-path", str(process_path),
        "--out-dir",    str(out_dir),
        "--force",
    ]

    # --no-beat-tracking solo si el modelo lo soporta
    if args.no_beat_tracking and args.model.lower() in SUPPORTS_NO_BEAT_TRACKING:
        cmd.append("--no-beat-tracking")

    env = os.environ.copy()

    print(f"[transcribe] Comando: {' '.join(cmd)}", flush=True)

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

    # --- Parsear stdout para progreso ----------------------------------------
    for line in iter(proc.stdout.readline, ""):
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)

        pct = 0.1
        # Patrón "[transcribe] [n/m]" en research_*_modal.py
        m = re.search(r'\[(\d+)/(\d+)\]', line)
        if m:
            n, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                pct = 0.1 + (n / total) * 0.8
        # Patrón "Chunk n/m" (MIROS)
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
                       f"Modal falló con código {proc.returncode}. "
                       "Revisa el log para más detalles.")
        return 1

    # --- Localizar MIDI generado ---------------------------------------------
    raw_mid = out_dir / "transcribed_cuda.mid"
    if not raw_mid.exists():
        write_progress(pf, "error", 0,
                       f"MIDI no encontrado en {out_dir}/transcribed_cuda.mid. "
                       "El proceso terminó sin error pero no produjo salida.")
        return 1

    # Renombrar a <src_stem>__<model>.mid
    src_stem = input_path.stem
    final_name = f"{src_stem}__{args.model}.mid"
    final_mid  = out_dir / final_name
    try:
        raw_mid.rename(final_mid)
    except Exception:
        final_mid = raw_mid  # fallback: dejar el nombre original

    # Contar instrumentos no vacíos
    n_instruments = count_instruments(final_mid)
    instr_str = str(n_instruments) if n_instruments >= 0 else "?"

    n_label = (f"{n_instruments} instrumento{'s' if n_instruments != 1 else ''}"
               if n_instruments >= 0 else "MIDI generado")
    done_msg = f"Completado — {n_label}"
    write_progress(pf, "done", 1.0, done_msg, [
        str(final_mid),
        f"INSTRUMENTS|{instr_str}",
    ])
    print(f"[transcribe] {done_msg}: {final_mid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
