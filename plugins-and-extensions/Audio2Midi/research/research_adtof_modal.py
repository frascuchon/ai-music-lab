"""
Modal.com inference app para ADTOF (Automatic Drum Transcription with Onsets and Frames).

ADTOF es un modelo CRNN específico para transcripción de batería entrenado con el
dataset ADTOF (~17k tracks de YouTube). Usa TF/Keras con un encoder multi-scale
y un decoder frame-level + onset-level para drum events.

Repo:    https://github.com/MZehren/ADTOF
Paper:   "ADTOF: A large dataset of non-synthetic music for automatic drum transcription"
         Zehren et al., ISMIR 2021
Licencia: CC BY-NC-SA 4.0 (no comercial, share-alike) — solo investigación interna.

Notas de drums producidas (GM):
  35 → BD  (Bass Drum / Kick)
  38 → SD  (Snare)
  42 → HH  (Closed Hi-Hat)
  47 → TT  (Tom)
  49 → CY  (Cymbal + Ride)

Este script normaliza 35→36 (GM kick canónico) antes de guardar el MIDI.

Pesos: empaquetados en el wheel (package_data = adtof/models/*). No se necesita
Modal Volume — pip install /workspace/ADTOF incluye los pesos.

Setup (primera vez — solo verifica instalación):
    cd Audio2Midi/research
    modal run research_adtof_modal.py::setup

Smoke test (batería aislada):
    modal run research_adtof_modal.py::main \\
        --audio-path ../evaluation/yourmt3/test04/input.wav \\
        --out-dir /tmp/adtof_smoke

Benchmark completo (drums_only.mid por test):
    modal run research_adtof_modal.py::eval_all \\
        --eval-dir ../evaluation/compound_v2

GPU: T4 (16 GB, default). ADTOF es pequeño (~10-30 MB de pesos).
     Override: env ADTOF_GPU=A10G.

Coste estimado (T4):
  setup: ~$0.01
  4 tests × ~1-2 min/test: ~$0.05-0.10 total
"""

import os
import re
import sys
import time
import glob
import tempfile
import shutil
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_GPU = os.environ.get("ADTOF_GPU", "T4")

ADTOF_REPO_URL = "https://github.com/MZehren/ADTOF"
ADTOF_DIR = "/workspace/ADTOF"

# Mapa de remapeo de notas ADTOF → GM estándar
# ADTOF usa 35 (Acoustic Bass Drum); GM "canónico" es 36 (Bass Drum 1)
ADTOF_NOTE_REMAP = {35: 36}

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git", "ffmpeg", "sox", "libsndfile1",
        "build-essential", "libffi-dev",
    ])
    .run_commands(f"git clone {ADTOF_REPO_URL} {ADTOF_DIR}")
    # Cython + numpy deben instalarse antes de madmom (que compila extensiones C)
    .pip_install("Cython==3.0.10", "numpy==1.26.4")
    .run_commands(
        # madmom requiere Cython y numpy pre-instalados para compilar
        "pip install git+https://github.com/CPJKU/madmom",
        # Instalar ADTOF (incluye tapcorrect y los pesos empaquetados)
        f"pip install {ADTOF_DIR}",
    )
    .pip_install(
        "tensorflow-cpu>=2.14,<2.16",
        "librosa>=0.10",
        "pretty_midi>=0.2.9",
        "mir_eval",
        "ffmpeg-python",
        "scikit-learn>=1.3.2",
        "soundfile>=0.12",
    )
)

app = modal.App("adtof-inference", image=image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_model():
    """Carga el modelo ADTOF Frame_RNN con los pesos empaquetados."""
    from adtof.model.model import Model
    model, hparams = Model.modelFactory(modelName="Frame_RNN", scenario="adtofAll", fold=0)
    assert model.weightLoadedFlag, "ADTOF: pesos no cargados correctamente"
    return model, hparams


def _remap_notes(midi_path: str) -> None:
    """Normaliza notas ADTOF al estándar GM in-place (35→36 kick)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    changed = False
    for inst in pm.instruments:
        for note in inst.notes:
            if note.pitch in ADTOF_NOTE_REMAP:
                note.pitch = ADTOF_NOTE_REMAP[note.pitch]
                changed = True
    if changed:
        pm.write(midi_path)


def _transcribe_drums(audio_bytes: bytes) -> bytes:
    """
    Transcribe el audio al stem de batería con ADTOF.
    Devuelve bytes del MIDI de batería (is_drum=True, canal 9).
    """
    import pretty_midi

    suffix = ".wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        suffix = ".mp3"

    # Directorio temporal para audio de entrada y MIDI de salida
    work_dir = tempfile.mkdtemp()
    tmp_audio = os.path.join(work_dir, f"input{suffix}")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    try:
        with open(tmp_audio, "wb") as f:
            f.write(audio_bytes)

        model, hparams = _load_model()
        # predictFolder acepta glob o path; escribe MIDI en out_dir con el mismo stem name
        model.predictFolder(
            audioFiles=tmp_audio,
            outputLocation=out_dir,
            **hparams,
        )

        midi_files = glob.glob(os.path.join(out_dir, "*.mid"))
        if not midi_files:
            raise RuntimeError(f"ADTOF no generó MIDI en {out_dir}")

        midi_path = midi_files[0]

        # Normalizar notas (35→36) antes de leer
        _remap_notes(midi_path)

        # Marcar todos los instrumentos como drums (canal 9, is_drum=True)
        pm = pretty_midi.PrettyMIDI(midi_path)
        combined = pretty_midi.PrettyMIDI()
        for inst in pm.instruments:
            drum_inst = pretty_midi.Instrument(
                program=0,
                is_drum=True,
                name="Drums (ADTOF)",
            )
            drum_inst.notes = inst.notes
            combined.instruments.append(drum_inst)

        if not combined.instruments:
            raise RuntimeError("ADTOF generó MIDI pero sin notas")

        out_buf = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        out_buf.close()
        try:
            combined.write(out_buf.name)
            with open(out_buf.name, "rb") as f:
                return f.read()
        finally:
            os.unlink(out_buf.name)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Setup — verificación de entorno
# ---------------------------------------------------------------------------
@app.function(timeout=300)
def setup() -> None:
    """
    Verifica que ADTOF está instalado y los pesos cargan correctamente.

    Ejecutar una vez:
        modal run research_adtof_modal.py::setup
    """
    from adtof.model.model import Model
    print("[setup] Cargando modelo ADTOF Frame_RNN/adtofAll/fold=0 …")
    model, hparams = Model.modelFactory(modelName="Frame_RNN", scenario="adtofAll", fold=0)
    print(f"[setup] weightLoadedFlag = {model.weightLoadedFlag}")
    assert model.weightLoadedFlag, "Pesos no cargados"
    print(f"[setup] hparams = {hparams}")
    print("[setup] ADTOF OK — pesos cargados correctamente.")


# ---------------------------------------------------------------------------
# Función Modal principal — transcribe lista de audios → drums MIDI
# ---------------------------------------------------------------------------
@app.function(
    timeout=3600,
    gpu=DEFAULT_GPU,
)
def transcribe_batch(audio_payloads: list[bytes]) -> list[bytes]:
    """
    Transcribe cada audio a MIDI de batería con ADTOF.

    audio_payloads: bytes de cada archivo WAV/MP3
    Returns: list[midi_bytes] — b"" si la transcripción falló para ese audio
    """
    results = []
    for i, audio_bytes in enumerate(audio_payloads):
        t0 = time.time()
        try:
            midi_bytes = _transcribe_drums(audio_bytes)
            elapsed = time.time() - t0
            print(f"[adtof] [{i+1}/{len(audio_payloads)}] OK — {len(midi_bytes)} bytes ({elapsed:.1f}s)")
            results.append(midi_bytes)
        except Exception as exc:
            print(f"[adtof] [{i+1}/{len(audio_payloads)}] ERROR: {exc}")
            results.append(b"")

    return results


# ---------------------------------------------------------------------------
# Entrypoint: un audio → out-dir (escribe drums_only.mid para debugging)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    audio_path: str = "",
    out_dir: str = ".",
    force: bool = False,
):
    """
    Transcribe el stem de batería de un audio con ADTOF.

    Ejemplo:
        modal run research_adtof_modal.py::main \\
            --audio-path ../evaluation/yourmt3/test04/input.wav \\
            --out-dir /tmp/adtof_smoke
    """
    if not audio_path:
        print("ERROR: --audio-path requerido. Ejemplo:")
        print("  modal run research_adtof_modal.py::main \\")
        print("      --audio-path ../evaluation/yourmt3/test04/input.wav \\")
        print("      --out-dir /tmp/adtof_smoke")
        raise SystemExit(1)

    audio_p = Path(audio_path)
    if not audio_p.exists():
        print(f"ERROR: No existe: {audio_p}")
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_mid = out_p / "drums_only.mid"

    if out_mid.exists() and not force:
        print(f"[main] Ya existe {out_mid}. Usa --force para regenerar.")
        return

    print(f"[main] ADTOF: {audio_p.name} → {out_mid}")
    audio_bytes = audio_p.read_bytes()
    [midi_bytes] = transcribe_batch.remote([audio_bytes])

    if not midi_bytes:
        print("[main] ERROR: ADTOF devolvió vacío.")
        raise SystemExit(1)

    out_mid.write_bytes(midi_bytes)
    print(f"[main] Guardado: {out_mid} ({len(midi_bytes)} bytes)")


# ---------------------------------------------------------------------------
# Entrypoint: todos los tests del directorio de evaluación
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/compound_v2",
    force: bool = False,
    only: str = "",
):
    """
    Transcribe batería de todos los tests con ADTOF (drums_only.mid por test).

    Útil para debugging aislado de ADTOF antes de correr compound_v2 completo.

    Ejemplos:
        modal run research_adtof_modal.py::eval_all
        modal run research_adtof_modal.py::eval_all --only 4
    """
    eval_path = Path(eval_dir)

    test_dirs = sorted(
        eval_path.glob("test*"),
        key=lambda p: int(re.sub(r"\D", "", p.name) or "0"),
    )

    if only:
        only_nums = {int(x) for x in only.split(",")}
        test_dirs = [
            d for d in test_dirs
            if int(re.sub(r"\D", "", d.name) or "0") in only_nums
        ]

    if not test_dirs:
        print(f"ERROR: No se encontraron carpetas test* en {eval_dir}")
        sys.exit(1)

    audio_bytes_list = []
    valid_dirs = []
    for td in test_dirs:
        input_wav = td / "input.wav"
        if not input_wav.exists():
            input_mp3 = td / "input.mp3"
            if input_mp3.exists():
                input_wav = input_mp3
            else:
                print(f"[warn] Sin input.wav/mp3 en {td.name} — saltando.")
                continue

        out_mid = td / "drums_only.mid"
        if out_mid.exists() and not force:
            print(f"[skip] {td.name}: ya tiene drums_only.mid. Usa --force para regenerar.")
            continue

        audio_bytes_list.append(input_wav.read_bytes())
        valid_dirs.append((td, out_mid))
        print(f"[eval_all] Encolado: {td.name} ({input_wav.name}, {input_wav.stat().st_size / 1e6:.1f} MB)")

    if not audio_bytes_list:
        print("[eval_all] Nada que transcribir.")
        return

    print(f"\n[eval_all] Transcribiendo batería en {len(audio_bytes_list)} tests con ADTOF ({DEFAULT_GPU}) …\n")
    t0 = time.time()
    results = transcribe_batch.remote(audio_bytes_list)

    ok = 0
    for (td, out_mid), midi_bytes in zip(valid_dirs, results):
        if midi_bytes:
            out_mid.write_bytes(midi_bytes)
            print(f"[eval_all] ✓ {td.name} → {out_mid.name} ({len(midi_bytes)} bytes)")
            ok += 1
        else:
            print(f"[eval_all] ✗ {td.name} — ADTOF falló")

    elapsed = time.time() - t0
    print(f"\n[eval_all] Completado: {ok}/{len(audio_bytes_list)} OK en {elapsed:.0f}s")
    print(f"[eval_all] Outputs en: {eval_path.resolve()}")
