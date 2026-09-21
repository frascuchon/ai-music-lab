"""
Conversión bidireccional MIDI ↔ ABC notation usando abcMIDI.

Herramienta oficial usada por el web demo de ChatMusician (abc2midi/midi2abc).
Proporciona tanto funciones importables como una interfaz CLI.

Instalación:
    brew install abcmidi       # macOS
    apt install abcmidi        # Debian/Ubuntu

Uso como librería:
    from tools.midi_abc import midi_to_abc_text, abc_to_midi_bytes

    abc_str  = midi_to_abc_text("input.mid")          # MIDI → ABC (string)
    mid_bytes = abc_to_midi_bytes(abc_str)             # ABC → MIDI (bytes)

Uso como CLI:
    python research/tools/midi_abc.py midi-to-abc input.mid output.abc
    python research/tools/midi_abc.py abc-to-midi input.abc output.mid

    # Output por defecto (mismo nombre, extensión cambiada):
    python research/tools/midi_abc.py midi-to-abc input.mid
    python research/tools/midi_abc.py abc-to-midi input.abc
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# REAPER launched from Finder/Dock (i.e. not from a terminal) inherits
# macOS's minimal launchd PATH, not the user's shell PATH — so `brew install
# abcmidi` can leave midi2abc/abc2midi perfectly present on disk yet
# invisible to a plain PATH lookup from inside REAPER's subprocess tree
# (panel.lua -> midigen.py -> here). Fall back to the well-known install
# locations before giving up.
_COMMON_TOOL_DIRS = [
    "/opt/homebrew/bin",  # Homebrew, Apple Silicon
    "/usr/local/bin",     # Homebrew, Intel Mac / manual installs
    "/opt/local/bin",     # MacPorts
]


# ---------------------------------------------------------------------------
# Funciones importables (núcleo del módulo)
# ---------------------------------------------------------------------------

def midi_to_abc_text(input_path: str) -> str:
    """
    Convierte un fichero MIDI a ABC notation y devuelve el texto.

    Raises:
        FileNotFoundError: si input_path no existe.
        RuntimeError: si midi2abc falla o no está en PATH.
    """
    tool = _resolve_tool("midi2abc")
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"MIDI file not found: {input_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_p = Path(tmpdir) / "score.abc"
        result = subprocess.run(
            # -title overrides midi2abc's default "T: from <input path>" — left
            # alone, that default embeds the full local temp-file path (e.g.
            # /var/folders/.../midigen_seed_3.mid) as the tune's title, which
            # burns prompt tokens on a meaningless string and hands the LLM
            # out-of-distribution "text" in a field it expects to be a title.
            [tool, str(in_p), "-o", str(out_p), "-title", "Seed"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"midi2abc failed:\n{result.stderr}")
        return out_p.read_text(encoding="utf-8")


def abc_to_midi_bytes(abc_text: str) -> bytes:
    """
    Convierte ABC notation a bytes MIDI.

    Raises:
        RuntimeError: si abc2midi falla o no está en PATH.
    """
    tool = _resolve_tool("abc2midi")
    with tempfile.TemporaryDirectory() as tmpdir:
        abc_path = Path(tmpdir) / "score.abc"
        midi_path = Path(tmpdir) / "score.mid"
        abc_path.write_text(abc_text, encoding="utf-8")

        result = subprocess.run(
            [tool, str(abc_path), "-o", str(midi_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"abc2midi failed:\n{result.stderr}")
        if not midi_path.exists():
            raise RuntimeError("abc2midi no produjo fichero MIDI")

        return midi_path.read_bytes()


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _resolve_tool(name: str) -> str:
    """Locate an abcMIDI binary, tolerating a restricted PATH (see
    _COMMON_TOOL_DIRS above for why). Returns the resolved absolute path.

    Raises:
        RuntimeError: not found in PATH nor in any common install location.
    """
    found = shutil.which(name)
    if found:
        return found
    for d in _COMMON_TOOL_DIRS:
        candidate = Path(d) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(
        f"'{name}' no encontrado ni en PATH ni en las ubicaciones habituales "
        f"({', '.join(_COMMON_TOOL_DIRS)}).\n"
        f"  macOS:  brew install abcmidi\n"
        f"  Linux:  apt install abcmidi\n"
        "  Si ya está instalado pero REAPER se abrió desde Finder/Dock: "
        "REAPER no hereda el PATH de tu shell — usa la pestaña Setup de "
        "AI Music Lab para verificar/instalar abcmidi."
    )


# ---------------------------------------------------------------------------
# CLI (wrappers sobre las funciones importables)
# ---------------------------------------------------------------------------

def _cli_midi_to_abc(input_path: str, output_path: str) -> None:
    abc_text = midi_to_abc_text(input_path)
    Path(output_path).write_text(abc_text, encoding="utf-8")
    print(f"→ {output_path}")


def _cli_abc_to_midi(input_path: str, output_path: str) -> None:
    in_p = Path(input_path)
    if not in_p.exists():
        print(f"ERROR: fichero no encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)
    abc_text = in_p.read_text(encoding="utf-8")
    midi_bytes = abc_to_midi_bytes(abc_text)
    Path(output_path).write_bytes(midi_bytes)
    print(f"→ {output_path}")


def _default_output(input_path: str, new_ext: str) -> str:
    return str(Path(input_path).with_suffix(new_ext))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conversión bidireccional MIDI ↔ ABC usando abcMIDI."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_m2a = sub.add_parser("midi-to-abc", help="Convierte MIDI → ABC notation")
    p_m2a.add_argument("input", help="Fichero MIDI de entrada (.mid)")
    p_m2a.add_argument("output", nargs="?", help="Fichero ABC de salida (.abc)")

    p_a2m = sub.add_parser("abc-to-midi", help="Convierte ABC notation → MIDI")
    p_a2m.add_argument("input", help="Fichero ABC de entrada (.abc)")
    p_a2m.add_argument("output", nargs="?", help="Fichero MIDI de salida (.mid)")

    args = parser.parse_args()

    try:
        if args.command == "midi-to-abc":
            out = args.output or _default_output(args.input, ".abc")
            _cli_midi_to_abc(args.input, out)
        elif args.command == "abc-to-midi":
            out = args.output or _default_output(args.input, ".mid")
            _cli_abc_to_midi(args.input, out)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
