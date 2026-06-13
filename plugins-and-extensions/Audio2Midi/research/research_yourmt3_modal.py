"""
Modal.com inference app para YourMT3+ (mimbres/YourMT3, arXiv 2407.04822).

YourMT3+ combina la arquitectura MT3 con un Hierarchical Attention Transformer
(HAT) en el dominio tiempo-frecuencia, Mixture of Experts (MoE), y aumentación
cross-stem para mezcla de múltiples datasets. Supera a MT3 y PerceiverTF en
10 benchmarks públicos. Transcripción vocal directa sin pre-separador.

Pipeline completo:
    [audio.wav  (cualquier sr, mono o estéreo)]
        ↓
    Re-muestreo → 16kHz mono (preproceso interno)
        ↓
    Log-mel spectrogram (frames de 2s, solapados)
        ↓
    YourMT3+ encoder (HAT + MoE)
        ↓
    Token decoder (multi-canal, por instrumento)
        ↓
    transcribed_cuda.mid  (multi-track, clases MIDI por pista)

Modelo:  mimbres/YourMT3  (Apache 2.0)
Repo:    https://github.com/mimbres/YourMT3
Paper:   https://arxiv.org/abs/2407.04822  (MLSP 2024)
Pesos:   verificar en HuggingFace Hub — varios checkpoints disponibles

Parámetros de inferencia (referencia; actualizar tras revisar docs del repo):
    # YourMT3+ con checkpoint pre-entrenado multi-dataset
    model = "YourMT3+"
    checkpoint = "path/to/checkpoint.ckpt"  # o HF model ID
    audio_file = "input.wav"

Setup (descarga pesos al Volume, ejecutar una vez):
    modal run research/research_yourmt3_modal.py::setup

Transcripción libre:
    modal run research/research_yourmt3_modal.py::main \\
        --audio-path research/fixtures/multitracks_short.wav \\
        --out-dir evaluation/yourmt3/smoke

Benchmark completo:
    modal run research/research_yourmt3_modal.py::eval_all \\
        --eval-dir evaluation/yourmt3 \\
        --n-outputs 1

GPUs disponibles (--gpu, por defecto A10G):
    A10G     24 GB, ~$1.10/hr  (default) — modelo ~500MB-2GB según variante
    T4       16 GB, ~$0.60/hr  — probablemente suficiente

Coste estimado (A10G):
    - 1 audio de 30s: ~15-45s → ~$0.01-0.02/transcripción
    - 10 tests × 1 output = ~$0.10-0.20 total
"""

import os
import re
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Volume — caché HuggingFace persistente
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("yourmt3-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"
HF_CACHE = f"{WEIGHTS_MOUNT}/hf_cache"

YOURMT3_REPO = "https://github.com/mimbres/YourMT3"
YOURMT3_DIR = "/yourmt3"

# TODO: actualizar MODEL_ID con el HF model ID correcto tras revisar el repo
MODEL_ID = "mimbres/YourMT3"  # placeholder — verificar HF Hub

DEFAULT_GPU = os.environ.get("YOURMT3_GPU", "A10G")

# ---------------------------------------------------------------------------
# Container image — CUDA 12 + PyTorch + dependencias YourMT3 + repo clonado
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "sox", "libsndfile1"])
    .run_commands(
        f"git clone {YOURMT3_REPO} {YOURMT3_DIR}",
    )
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "transformers>=4.44",
        "huggingface_hub>=0.24",
        "accelerate>=0.34",
        "safetensors>=0.4",
        # Arquitectura del modelo
        "einops>=0.7",
        "omegaconf>=2.3",
        # MIDI I/O
        "mido>=1.3",
        "pretty_midi>=0.2.10",
        # Audio
        "librosa>=0.10",
        "soundfile>=0.12",
        # Utilidades
        "psutil>=6.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # TODO: instalar deps adicionales de YourMT3 desde requirements.txt del repo
    # .run_commands(f"pip install -r {YOURMT3_DIR}/requirements.txt")
)

app = modal.App("yourmt3-inference", image=image)


# ---------------------------------------------------------------------------
# Pre-descarga de pesos al Volume
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=1800,
    gpu=DEFAULT_GPU,
)
def setup():
    """
    Descarga checkpoints de YourMT3+ al Volume persistente. Ejecutar una vez.

    TODO: actualizar con el proceso de descarga correcto del repo
    (snapshot_download de HF Hub o descarga manual de checkpoints via script
    del repo — verificar docs/README.md de YourMT3).
    """
    raise NotImplementedError(
        "PoC pendiente — revisar repo YourMT3 para proceso de descarga de pesos. "
        "Ver RESEARCH.md sección 'Próximos pasos'."
    )


# ---------------------------------------------------------------------------
# Núcleo de inferencia
# ---------------------------------------------------------------------------
def _load_model():
    """
    Carga el modelo YourMT3+ desde el Volume.

    TODO: implementar carga de modelo siguiendo el API del repo YourMT3.
    Ver scripts de inferencia en YourMT3/amt/src/transcription/
    """
    raise NotImplementedError("PoC pendiente")


def _infer_one(model, audio_path: str) -> bytes:
    """
    Transcribe un archivo de audio a MIDI.

    Returns: bytes del MIDI generado, o b"" si hay error.

    TODO: implementar invocando el pipeline de inferencia de YourMT3+.
    """
    raise NotImplementedError("PoC pendiente")


# ---------------------------------------------------------------------------
# Función Modal principal — transcribe N audios
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=7200,
    gpu=DEFAULT_GPU,
)
def transcribe(audio_paths: list[str]) -> list[bytes]:
    """
    Transcribe cada audio a MIDI multi-instrumento.

    audio_paths: rutas locales a archivos WAV/MP3 (en el container)
    Returns: list[midi_bytes] — b"" si la transcripción falló para ese audio
    """
    raise NotImplementedError("PoC pendiente")


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
    Transcribe un audio y guarda el MIDI resultante en out-dir.

    Ejemplos:
        modal run research/research_yourmt3_modal.py::main \\
            --audio-path research/fixtures/multitracks_short.wav \\
            --out-dir evaluation/yourmt3/smoke
    """
    raise NotImplementedError(
        "PoC pendiente — implementar tras completar setup() y transcribe(). "
        "Ver RESEARCH.md para contexto."
    )


# ---------------------------------------------------------------------------
# Entrypoint: todos los tests del directorio de evaluación
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/yourmt3",
    force: bool = False,
    only: str = "",
):
    """
    Transcribe todos los tests del directorio de evaluación.

    Cada carpeta test*/ debe contener:
      - input.wav  (audio a transcribir)
      - prompt.txt (descripción del test y parámetros — misma convención que MidiGenerator)

    Outputs por test: transcribed_cuda.mid

    Ejemplo:
        modal run research/research_yourmt3_modal.py::eval_all \\
            --eval-dir evaluation/yourmt3

        # Solo tests específicos:
        modal run research/research_yourmt3_modal.py::eval_all \\
            --eval-dir evaluation/yourmt3 \\
            --only 1,3,5
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

    # Filtrar tests ya procesados y tests sin input.wav
    audio_paths = []
    valid_dirs = []
    for td in test_dirs:
        input_wav = td / "input.wav"
        if not input_wav.exists():
            print(f"[warn] Sin input.wav en {td.name}, saltando")
            continue

        output_mid = td / "transcribed_cuda.mid"
        if output_mid.exists() and not force:
            print(f"[skip] {td.name}: ya tiene transcribed_cuda.mid. Usa --force para regenerar.")
            continue

        audio_paths.append(str(input_wav))
        valid_dirs.append(td)

    if not audio_paths:
        print("[eval_all] Nada que transcribir (todos los tests ya tienen outputs).")
        return

    raise NotImplementedError(
        f"PoC pendiente — {len(audio_paths)} audios listos para transcribir. "
        "Implementar transcribe() primero."
    )
