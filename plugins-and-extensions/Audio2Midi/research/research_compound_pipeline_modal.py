"""
Modal.com inference app para el pipeline compuesto de transcripción Audio→MIDI.

Arquitectura: Source Separation (Demucs) + Transcripción por stem (Basic Pitch)

Pipeline completo:
    [audio.wav  (mezcla completa)]
        ↓
    Demucs htdemucs_6s (Meta, MIT license)
        ↓ (6 stems: vocals / bass / drums / guitar / piano / other)
    Basic Pitch (Spotify, Apache 2.0)  ← por cada stem melódico
    [drums tratados por Basic Pitch — calidad limitada, ver notas]
        ↓
    Merge: todos los stems → MIDI multi-pista (canal por instrumento)
        ↓
    transcribed_cuda.mid  (multi-track)

Modelos:
    Demucs:      facebookresearch/demucs  (htdemucs_6s)
    Basic Pitch: spotify/basic-pitch      (ICASSP_2022_MODEL_PATH)

Repos:
    https://github.com/facebookresearch/demucs
    https://github.com/spotify/basic-pitch

Notas de calidad:
    ✅ Piano, guitarras, bajo: Basic Pitch sobre stem aislado funciona bien
    ✅ Voz (melódica): Basic Pitch detecta la línea melódica
    ⚠️  Batería (drums): Basic Pitch no está diseñado para drums no-pitched.
        Los resultados de drums serán ruidosos. Para mejores resultados de
        drums usar ADTOF (https://github.com/MZehren/ADTOF).
    ⚠️  htdemucs_6s: los stems guitar/piano son menos precisos que drums/bass/vocals.

Setup (primera vez):
    modal run research/research_compound_pipeline_modal.py::setup

Transcripción libre:
    modal run research/research_compound_pipeline_modal.py::main \\
        --audio-path research/fixtures/multitracks_short.wav \\
        --out-dir evaluation/compound/smoke

Benchmark completo:
    modal run research/research_compound_pipeline_modal.py::eval_all \\
        --eval-dir evaluation/compound

GPUs (--gpu, env COMPOUND_GPU):
    A10G     24 GB  (default) — Demucs MPS-capable pero aquí usamos CUDA
    T4       16 GB  — suficiente para Demucs + Basic Pitch

Coste estimado (A10G):
    - 1 audio de 30s: ~20-40s → ~$0.01-0.02
    - 5 tests × 1 output = ~$0.05-0.10 total
"""

import os
import re
import sys
import time
import tempfile
import uuid
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_GPU = os.environ.get("COMPOUND_GPU", "A10G")
DEMUCS_MODEL = "htdemucs_6s"
STEMS = ["vocals", "bass", "drums", "guitar", "piano", "other"]

# Asignación de canales MIDI por stem (GM program numbers, 0-indexed)
# drums usa canal 9 (percussion), el resto canales melódicos
STEM_MIDI_PROGRAM = {
    "vocals":  54,   # Synth Voice (GM 55) — aproximación para voz
    "bass":     33,  # Electric Bass (finger) (GM 34)
    "drums":     0,  # Canal 9 (percusión) — programa ignorado en canal 10
    "guitar":   25,  # Acoustic Guitar (steel) (GM 26)
    "piano":     0,  # Acoustic Grand Piano (GM 1)
    "other":    48,  # String Ensemble (GM 49)
}

STEM_MIDI_CHANNEL = {
    "vocals": 0,
    "bass":   1,
    "drums":  9,  # Canal MIDI de percusión (10 en 1-indexed)
    "guitar": 2,
    "piano":  3,
    "other":  4,
}

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "sox", "libsndfile1"])
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Todos en un paso para que pip resuelva conflictos de numpy/TF/demucs juntos
        "demucs>=4.0",
        "tensorflow-cpu>=2.14,<2.16",  # backend para basic-pitch (TFLite+numpy incompatibles)
        "basic-pitch>=0.4",
        "mido>=1.3",
        "pretty_midi>=0.2.10",
        "librosa>=0.10",
        "soundfile>=0.12",
        "numpy>=1.24",
        "psutil>=6.0",
    )
)

app = modal.App("compound-pipeline-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — verificación de entorno (no hay pesos persistentes que descargar)
# ---------------------------------------------------------------------------
@app.function(timeout=300, gpu=DEFAULT_GPU)
def setup():
    """
    Verifica que Demucs y Basic Pitch están instalados correctamente.
    Los pesos se descargan automáticamente en el primer uso (cacheados por Modal).

    Ejecutar una vez para verificar el entorno:
        modal run research/research_compound_pipeline_modal.py::setup
    """
    import demucs.api
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model as BPModel

    print("[setup] Demucs OK")

    print("[setup] Cargando Basic Pitch …")
    bp = BPModel(ICASSP_2022_MODEL_PATH)
    print("[setup] Basic Pitch OK")

    print("[setup] Verificando Demucs htdemucs_6s …")
    sep = demucs.api.Separator(model=DEMUCS_MODEL, device="cpu")
    stems = sep.model.sources
    print(f"[setup] Demucs htdemucs_6s stems: {stems}")
    print("[setup] Todo OK — pipeline listo.")


# ---------------------------------------------------------------------------
# Núcleo: separación + transcripción
# ---------------------------------------------------------------------------
def _separate_stems(audio_bytes: bytes) -> dict[str, bytes]:
    """
    Separa un audio en stems usando Demucs htdemucs_6s.
    Usa demucs.separate.main (compatible con 4.0.x stable y 4.1.x alpha).
    Returns: dict stem_name → WAV bytes
    """
    import shutil
    from demucs.separate import main as demucs_main

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    out_dir = tempfile.mkdtemp()
    try:
        demucs_main(["-n", DEMUCS_MODEL, "--out", out_dir,
                     "--device", "cuda", audio_path])

        base_name = Path(audio_path).stem
        stems_dir = Path(out_dir) / DEMUCS_MODEL / base_name
        if not stems_dir.exists():
            raise RuntimeError(f"Demucs no generó salida en {stems_dir}")

        result = {}
        for stem_file in sorted(stems_dir.glob("*.wav")):
            result[stem_file.stem] = stem_file.read_bytes()

        return result
    finally:
        os.unlink(audio_path)
        shutil.rmtree(out_dir, ignore_errors=True)


def _transcribe_stem(stem_name: str, stem_bytes: bytes) -> "pretty_midi.PrettyMIDI | None":
    """
    Transcribe un stem individual con Basic Pitch.
    Devuelve un PrettyMIDI con el canal e instrumento correcto para ese stem.
    """
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    import pretty_midi

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(stem_bytes)
        audio_path = tmp.name

    try:
        _, midi_data, _ = predict(
            audio_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58.0,   # ~1/16 a 120BPM
            minimum_frequency=None,
            maximum_frequency=None,
            multiple_pitch_bends=False,
            melodia_trick=True,
        )

        if midi_data is None or not midi_data.instruments:
            return None

        channel = STEM_MIDI_CHANNEL[stem_name]
        program = STEM_MIDI_PROGRAM[stem_name]
        is_drum = (stem_name == "drums")

        instr = pretty_midi.Instrument(
            program=program,
            is_drum=is_drum,
            name=stem_name,
        )
        instr.notes = midi_data.instruments[0].notes

        # Asignar canal MIDI correcto
        for note in instr.notes:
            note.channel = channel  # type: ignore[attr-defined]

        out = pretty_midi.PrettyMIDI()
        out.instruments.append(instr)
        return out

    except Exception as exc:
        print(f"[transcribe] ERROR en stem {stem_name}: {exc}")
        return None
    finally:
        os.unlink(audio_path)


def _merge_midi_tracks(stem_midis: dict[str, "pretty_midi.PrettyMIDI"]) -> bytes:
    """
    Fusiona todas las pistas MIDI de stems en un único archivo multi-track.
    Returns: bytes del MIDI combinado.
    """
    import pretty_midi

    combined = pretty_midi.PrettyMIDI()
    for stem_name in STEMS:
        midi = stem_midis.get(stem_name)
        if midi and midi.instruments:
            combined.instruments.extend(midi.instruments)

    buf = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    buf.close()
    try:
        combined.write(buf.name)
        with open(buf.name, "rb") as f:
            return f.read()
    finally:
        os.unlink(buf.name)


# ---------------------------------------------------------------------------
# Función Modal principal
# ---------------------------------------------------------------------------
@app.function(
    timeout=7200,
    gpu=DEFAULT_GPU,
    memory=16384,
)
def transcribe_batch(audio_payloads: list[bytes]) -> list[bytes]:
    """
    Transcribe cada audio al pipeline completo Demucs + Basic Pitch.

    audio_payloads: bytes de cada archivo WAV/MP3
    Returns: list[midi_bytes] — b"" si la transcripción falló
    """
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model as BPModel

    print("[pipeline] Pre-cargando Basic Pitch model …")
    _bp_model = BPModel(ICASSP_2022_MODEL_PATH)
    print("[pipeline] Basic Pitch listo.")

    results = []
    for i, audio_bytes in enumerate(audio_payloads):
        t0 = time.time()
        print(f"\n[pipeline] [{i+1}/{len(audio_payloads)}] Separando stems con Demucs …")
        try:
            stems = _separate_stems(audio_bytes)
            print(f"[pipeline]   Stems separados: {list(stems.keys())} ({time.time()-t0:.1f}s)")

            stem_midis = {}
            for stem_name, stem_bytes in stems.items():
                t1 = time.time()
                midi = _transcribe_stem(stem_name, stem_bytes)
                n_notes = len(midi.instruments[0].notes) if midi and midi.instruments else 0
                print(f"[pipeline]   {stem_name}: {n_notes} notas ({time.time()-t1:.1f}s)")
                if midi:
                    stem_midis[stem_name] = midi

            midi_bytes = _merge_midi_tracks(stem_midis)
            elapsed = time.time() - t0
            print(f"[pipeline] [{i+1}] OK — {len(midi_bytes)} bytes MIDI en {elapsed:.1f}s")
            results.append(midi_bytes)

        except Exception as exc:
            print(f"[pipeline] [{i+1}] ERROR: {exc}")
            results.append(b"")

    return results


# ---------------------------------------------------------------------------
# Entrypoint: un audio → out-dir
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    audio_path: str = "",
    out_dir: str = ".",
    force: bool = False,
):
    """
    Transcribe un audio con el pipeline Demucs+BasicPitch y guarda el MIDI.

    Ejemplo:
        modal run research/research_compound_pipeline_modal.py::main \\
            --audio-path research/fixtures/multitracks_short.wav \\
            --out-dir evaluation/compound/smoke
    """
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

    print(f"[main] Pipeline: {audio_p.name} → {out_mid}")
    [midi_bytes] = transcribe_batch.remote([audio_p.read_bytes()])

    if not midi_bytes:
        print("[main] ERROR: el pipeline devolvió vacío.")
        raise SystemExit(1)

    out_mid.write_bytes(midi_bytes)
    print(f"[main] Guardado: {out_mid} ({len(midi_bytes)} bytes)")


# ---------------------------------------------------------------------------
# Entrypoint: todos los tests
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/compound",
    force: bool = False,
    only: str = "",
):
    """
    Transcribe todos los tests del directorio de evaluación del pipeline compuesto.

    Ejemplo:
        modal run research/research_compound_pipeline_modal.py::eval_all \\
            --eval-dir evaluation/compound
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
                print(f"[warn] Sin input.wav/mp3 en {td.name} — saltando. Ejecuta fetch_fixtures.sh primero.")
                continue

        out_mid = td / "transcribed_cuda.mid"
        if out_mid.exists() and not force:
            print(f"[skip] {td.name}: ya tiene transcribed_cuda.mid.")
            continue

        audio_bytes_list.append(input_wav.read_bytes())
        valid_dirs.append((td, out_mid))
        print(f"[eval_all] Encolado: {td.name} ({input_wav.stat().st_size / 1e6:.1f} MB)")

    if not audio_bytes_list:
        print("[eval_all] Nada que transcribir.")
        return

    print(f"\n[eval_all] Procesando {len(audio_bytes_list)} tests …\n")
    t0 = time.time()
    results = transcribe_batch.remote(audio_bytes_list)

    ok = 0
    for (td, out_mid), midi_bytes in zip(valid_dirs, results):
        if midi_bytes:
            out_mid.write_bytes(midi_bytes)
            print(f"[eval_all] ✓ {td.name} → {out_mid.name} ({len(midi_bytes)} bytes)")
            ok += 1
        else:
            print(f"[eval_all] ✗ {td.name} — pipeline fallido")

    elapsed = time.time() - t0
    print(f"\n[eval_all] Completado: {ok}/{len(audio_bytes_list)} OK en {elapsed:.0f}s")
    print(f"[eval_all] Outputs en: {eval_path.resolve()}")
