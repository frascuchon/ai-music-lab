"""
Herramienta de conversión bidireccional MIDI ↔ ABC notation.

Usa abcMIDI (abc2midi / midi2abc), la misma herramienta empleada por el
web demo oficial de ChatMusician.

Instalación local:
    brew install abcmidi       # macOS
    apt install abcmidi        # Debian/Ubuntu

Uso:
    python research/tools/midi_abc.py midi-to-abc input.mid output.abc
    python research/tools/midi_abc.py abc-to-midi input.abc output.mid

    # Conversión in-place (output en la misma carpeta con extensión cambiada):
    python research/tools/midi_abc.py midi-to-abc input.mid
    python research/tools/midi_abc.py abc-to-midi input.abc

    # Leer de stdin / escribir a stdout:
    python research/tools/midi_abc.py midi-to-abc - -   # stdin→stdout
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def midi_to_abc(input_path: str, output_path: str) -> None:
    """Convierte MIDI → ABC usando midi2abc."""
    _check_tool("midi2abc")

    in_p = Path(input_path)
    if not in_p.exists():
        print(f"ERROR: fichero no encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_p = Path(output_path)
    result = subprocess.run(
        ["midi2abc", str(in_p), "-o", str(out_p)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR midi2abc:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"→ {out_p}")


def abc_to_midi(input_path: str, output_path: str) -> None:
    """Convierte ABC → MIDI usando abc2midi."""
    _check_tool("abc2midi")

    in_p = Path(input_path)
    if not in_p.exists():
        print(f"ERROR: fichero no encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_p = Path(output_path)
    result = subprocess.run(
        ["abc2midi", str(in_p), "-o", str(out_p)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR abc2midi:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"→ {out_p}")


def _check_tool(name: str) -> None:
    result = subprocess.run(["which", name], capture_output=True)
    if result.returncode != 0:
        print(
            f"ERROR: '{name}' no encontrado en PATH.\n"
            f"  macOS:  brew install abcmidi\n"
            f"  Linux:  apt install abcmidi",
            file=sys.stderr,
        )
        sys.exit(1)


def _default_output(input_path: str, new_ext: str) -> str:
    return str(Path(input_path).with_suffix(new_ext))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conversión bidireccional MIDI ↔ ABC usando abcMIDI."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_m2a = sub.add_parser("midi-to-abc", help="Convierte MIDI → ABC notation")
    p_m2a.add_argument("input", help="Fichero MIDI de entrada (.mid)")
    p_m2a.add_argument("output", nargs="?", help="Fichero ABC de salida (.abc); por defecto mismo nombre")

    p_a2m = sub.add_parser("abc-to-midi", help="Convierte ABC notation → MIDI")
    p_a2m.add_argument("input", help="Fichero ABC de entrada (.abc)")
    p_a2m.add_argument("output", nargs="?", help="Fichero MIDI de salida (.mid); por defecto mismo nombre")

    args = parser.parse_args()

    if args.command == "midi-to-abc":
        out = args.output or _default_output(args.input, ".abc")
        midi_to_abc(args.input, out)
    elif args.command == "abc-to-midi":
        out = args.output or _default_output(args.input, ".mid")
        abc_to_midi(args.input, out)


if __name__ == "__main__":
    main()
