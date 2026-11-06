"""
Modal.com inference app para Basic Pitch (Spotify, Apache 2.0) — mismo
pipeline que research_basic_pitch.py (nivelado de pico, limpieza de notas
fantasma/cola, tempo estimado o fijado), pero ejecutado en un contenedor
Modal en lugar de en esta máquina.

No duplica la lógica: el script local se copia al contenedor y su main() se
invoca con los mismos argumentos, así local y remoto producen el mismo MIDI.

Basic Pitch es una red pequeña (CPU basta): no hay elección de GPU.

    modal run research/research_basic_pitch_modal.py::main \\
        --audio-path guitar.wav --out-dir out/ --force

Salida: <out-dir>/transcribed_cuda.mid (mismo nombre que los demás modelos
Modal, que es el que espera transcribe.py).
"""

import os
import sys
import tempfile
from pathlib import Path

import modal

LOCAL_SCRIPT = Path(__file__).resolve().with_name("research_basic_pitch.py")
REMOTE_DIR = "/opt/basic_pitch_research"

# Exact version of every package the image contains (captured from the image that
# works; see tools/modal_freeze.py), applied through PIP_CONSTRAINT so a rebuild
# never drifts. The explicit pins below are the ones also locked in uv.lock.
_CONSTRAINTS = (Path(__file__).resolve().parent.parent.parent / "shared" / "modal_constraints"
                / "research_basic_pitch_modal.txt")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .add_local_file(str(_CONSTRAINTS), "/opt/modal_constraints.txt", copy=True)
    .env({"PIP_CONSTRAINT": "/opt/modal_constraints.txt"})
    .apt_install(["libsndfile1", "ffmpeg"])
    .pip_install(
        "basic-pitch==0.4.0",
        "numpy==1.26.4",
        "librosa==0.11.0",
        "pretty_midi==0.2.11",
        "soundfile==0.14.0",
        "scipy==1.15.3",
        "resampy==0.4.2",
        "setuptools<81",  # resampy (via basic-pitch) imports pkg_resources
    )
    .add_local_file(str(LOCAL_SCRIPT), f"{REMOTE_DIR}/research_basic_pitch.py")
)

app = modal.App("basic-pitch-inference", image=image)


def build_args(in_path: str, out_path: str, onset_threshold: float, bpm: float,
               no_beat_tracking: bool) -> list[str]:
    """argv for research_basic_pitch.main(). Unset options (<= 0) keep the
    local script's own defaults."""
    argv = ["--input", in_path, "--out", out_path]
    if onset_threshold > 0:
        argv += ["--onset-threshold", f"{onset_threshold}"]
    if bpm > 0:
        argv += ["--bpm", f"{bpm}"]
    if no_beat_tracking:
        argv.append("--no-beat-tracking")
    return argv


@app.function(timeout=1800, cpu=2.0)
def transcribe(audio: bytes, suffix: str, onset_threshold: float = 0.0,
               bpm: float = 0.0, no_beat_tracking: bool = False) -> bytes:
    """Transcribe one audio file (bytes) to MIDI (bytes); b"" on failure.
    The reason is printed (and streamed back by `modal run`) as `ERROR: ...`."""
    sys.path.insert(0, REMOTE_DIR)
    import research_basic_pitch as bp

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input" + suffix)
        out_path = os.path.join(tmp, "out.mid")
        Path(in_path).write_bytes(audio)
        rc = bp.main(build_args(in_path, out_path, onset_threshold, bpm,
                                no_beat_tracking))
        if rc != 0 or not os.path.exists(out_path):
            return b""
        return Path(out_path).read_bytes()


@app.local_entrypoint()
def main(
    audio_path: str = "",
    out_dir: str = ".",
    force: bool = False,
    onset_threshold: float = 0.0,
    bpm: float = 0.0,
    no_beat_tracking: bool = False,
):
    """Transcribe one audio file and save transcribed_cuda.mid in out-dir."""
    if not audio_path:
        print("ERROR: --audio-path requerido.")
        raise SystemExit(1)
    audio_p = Path(audio_path)
    if not audio_p.exists():
        print(f"ERROR: No existe: {audio_p}")
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_mid = out_p / "transcribed_cuda.mid"
    if out_mid.exists() and not force:
        print(f"[main] Ya existe {out_mid}. Usa --force para regenerar.")
        return

    print(f"[main] [1/2] Transcribiendo en Modal: {audio_p.name}")
    midi_bytes = transcribe.remote(audio_p.read_bytes(), audio_p.suffix or ".wav",
                                   onset_threshold, bpm, no_beat_tracking)
    if not midi_bytes:
        print("ERROR: Basic Pitch no devolvió notas (ver el mensaje anterior).")
        raise SystemExit(1)

    out_mid.write_bytes(midi_bytes)
    print(f"[main] [2/2] Guardado: {out_mid} ({len(midi_bytes)} bytes)")
