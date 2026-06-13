"""
Modal.com inference app para YourMT3+ (mimbres/YourMT3, arXiv 2407.04822).

YourMT3+ combina MT3 (T5-encoder) con PerceiverTF (HAT en tiempo-frecuencia),
Mixture of Experts (MoE), y aumentación cross-stem multi-dataset.
Supera MT3 y PerceiverTF en 10 benchmarks públicos con transcripción vocal
directa sin separador previo.

Modelo usado: YPTF.MoE+Multi (noPS)
  Checkpoint: mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt
  HF Hub:     mimbres/YourMT3
  Space:      huggingface.co/spaces/mimbres/YourMT3

Pipeline completo:
    [audio.wav  (cualquier sr, mono o estéreo)]
        ↓
    Re-muestreo → 16kHz mono (preproceso Space)
        ↓
    Log-mel spectrogram (frames de 2s, hop=300)
        ↓
    YourMT3+ encoder: PerceiverTF + HAT + MoE (26 capas)
        ↓
    Token decoder multi-canal (mc13_full_plus_256 tokenizer)
        ↓
    Detokenización + merge cross-channel
        ↓
    transcribed_cuda.mid  (multi-track, clases MIDI por pista)

Repo Space: https://huggingface.co/spaces/mimbres/YourMT3
Paper:      https://arxiv.org/abs/2407.04822  (MLSP 2024)
Pesos:      https://huggingface.co/mimbres/YourMT3  (2.77 GB total)

Checkpoint HF path (562 MB):
    logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt

Inferencia args (verbatim de app.py del Space, modelo YPTF.MoE+Multi noPS):
    [checkpoint_id@last.ckpt, '-p', '2024',
     '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
     '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1',
     '-ff', 'moe', '-wf', '4', '-nmoe', '8', '-kmoe', '2',
     '-act', 'silu', '-epe', 'rope', '-rp', '1',
     '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', '16']

Setup (descarga pesos al Volume, ejecutar una vez):
    modal run research/research_yourmt3_modal.py::setup

Transcripción libre:
    modal run research/research_yourmt3_modal.py::main \\
        --audio-path research/fixtures/multitracks_short.wav \\
        --out-dir evaluation/yourmt3/smoke

Benchmark completo:
    modal run research/research_yourmt3_modal.py::eval_all \\
        --eval-dir evaluation/yourmt3

GPUs disponibles (--gpu, env YOURMT3_GPU):
    A10G     24 GB, ~$1.10/hr  (default) — modelo ~2GB en VRAM con FP16
    T4       16 GB, ~$0.60/hr  — probablemente suficiente para FP16

Coste estimado (A10G, FP16):
    - Primer uso (setup): descarga 562 MB → ~$0.02
    - 1 audio de 30s: ~20-45s inferencia → ~$0.01-0.02
    - 10 tests × 1 output = ~$0.15-0.25 total
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
weights_vol = modal.Volume.from_name("yourmt3-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

SPACE_REPO = "https://huggingface.co/spaces/mimbres/YourMT3"
SPACE_DIR = "/yourmt3_space"

MODEL_HF_REPO = "mimbres/YourMT3"
EXP_ID = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops"
CKPT_FILE = "last.ckpt"
CKPT_HF_PATH = f"logs/2024/{EXP_ID}/checkpoints/{CKPT_FILE}"

DEFAULT_GPU = os.environ.get("YOURMT3_GPU", "A10G")
PRECISION = "16"

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "sox", "libsndfile1"])
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone {SPACE_REPO} {SPACE_DIR}"
    )
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "lightning>=2.2.1",
        "transformers==4.45.1",
        "numpy==1.26.4",
        "einops>=0.7",
        "librosa>=0.10",
        "soundfile>=0.12",
        "mido>=1.3",
        "pretty_midi>=0.2.10",
        "huggingface_hub>=0.24",
        "python-dotenv",
        "deprecated",
        "psutil>=6.0",
        "git+https://github.com/craffel/mir_eval.git",
    )
)

app = modal.App("yourmt3-inference", image=image)


# ---------------------------------------------------------------------------
# Helpers compartidos
# ---------------------------------------------------------------------------
def _setup_space_env():
    """Configura sys.path y el symlink logs/ → volume, y crea model_output/."""
    sys.path.insert(0, SPACE_DIR)
    os.chdir(SPACE_DIR)

    logs_link = f"{SPACE_DIR}/logs"
    if not os.path.lexists(logs_link):
        os.symlink(f"{WEIGHTS_MOUNT}/logs", logs_link)

    os.makedirs(f"{SPACE_DIR}/model_output", exist_ok=True)


def _model_args():
    """Devuelve los args de inferencia para YPTF.MoE+Multi (noPS)."""
    checkpoint_id = f"{EXP_ID}@{CKPT_FILE}"
    return [
        checkpoint_id,
        "-p", "2024",
        "-tk", "mc13_full_plus_256",
        "-dec", "multi-t5",
        "-nl", "26",
        "-enc", "perceiver-tf",
        "-sqr", "1",
        "-ff", "moe",
        "-wf", "4",
        "-nmoe", "8",
        "-kmoe", "2",
        "-act", "silu",
        "-epe", "rope",
        "-rp", "1",
        "-ac", "spec",
        "-hop", "300",
        "-atc", "1",
        "-pr", PRECISION,
    ]


# ---------------------------------------------------------------------------
# Setup — descarga checkpoint al Volume
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=1800,
)
def setup():
    """
    Descarga el checkpoint YPTF.MoE+Multi (noPS) de HuggingFace Hub
    al Volume persistente manteniendo la estructura de directorios esperada
    por initialize_trainer() del Space.

    Ejecutar una sola vez:
        modal run research/research_yourmt3_modal.py::setup
    """
    from huggingface_hub import hf_hub_download

    local_ckpt = f"{WEIGHTS_MOUNT}/{CKPT_HF_PATH}"

    if os.path.exists(local_ckpt):
        size_mb = os.path.getsize(local_ckpt) / 1e6
        print(f"[setup] Checkpoint ya descargado: {local_ckpt} ({size_mb:.0f} MB)")
        return

    print(f"[setup] Descargando {CKPT_HF_PATH} desde {MODEL_HF_REPO} …")
    t0 = time.time()
    hf_hub_download(
        repo_id=MODEL_HF_REPO,
        filename=CKPT_HF_PATH,
        local_dir=WEIGHTS_MOUNT,
        local_dir_use_symlinks=False,
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(local_ckpt) / 1e6
    print(f"[setup] OK — {size_mb:.0f} MB descargados en {elapsed:.0f}s")

    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Núcleo de inferencia (dentro del container)
# ---------------------------------------------------------------------------
def _load_model_once():
    """Carga modelo YourMT3+ en GPU. Llamar después de _setup_space_env()."""
    from model_helper import load_model_checkpoint

    print(f"[modal] Cargando modelo YPTF.MoE+Multi (noPS) en CPU …")
    t0 = time.time()
    model = load_model_checkpoint(args=_model_args(), device="cpu")
    print(f"[modal] Modelo cargado en CPU ({time.time()-t0:.1f}s), moviendo a CUDA …")
    model = model.cuda()
    print(f"[modal] Modelo en CUDA. Total: {time.time()-t0:.1f}s")
    return model


def _transcribe_one(model, audio_bytes: bytes, track_name: str) -> bytes:
    """
    Transcribe un audio (bytes) a MIDI (bytes).
    track_name debe ser único por llamada para evitar conflictos en model_output/.
    """
    from model_helper import transcribe as yt3_transcribe

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    try:
        audio_info = {"filepath": audio_path, "track_name": track_name}
        midi_path = yt3_transcribe(model, audio_info)
        with open(midi_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(audio_path)


# ---------------------------------------------------------------------------
# Función Modal principal — transcribe lista de audios
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=7200,
    gpu=DEFAULT_GPU,
)
def transcribe_batch(audio_payloads: list[bytes]) -> list[bytes]:
    """
    Transcribe cada audio a MIDI multi-instrumento.

    audio_payloads: bytes de cada archivo WAV/MP3 (leídos localmente)
    Returns: list[midi_bytes] — b"" si la transcripción falló para ese audio
    """
    _setup_space_env()
    model = _load_model_once()

    results = []
    for i, audio_bytes in enumerate(audio_payloads):
        track_name = f"track_{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        try:
            midi_bytes = _transcribe_one(model, audio_bytes, track_name)
            elapsed = time.time() - t0
            print(f"[transcribe] [{i+1}/{len(audio_payloads)}] OK — {len(midi_bytes)} bytes ({elapsed:.1f}s)")
            results.append(midi_bytes)
        except Exception as exc:
            print(f"[transcribe] [{i+1}/{len(audio_payloads)}] ERROR: {exc}")
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
    Transcribe un audio y guarda el MIDI resultante en out-dir.

    Ejemplo:
        modal run research/research_yourmt3_modal.py::main \\
            --audio-path research/fixtures/multitracks_short.wav \\
            --out-dir evaluation/yourmt3/smoke
    """
    if not audio_path:
        print("ERROR: --audio-path requerido. Ejemplo:")
        print("  modal run research/research_yourmt3_modal.py::main \\")
        print("      --audio-path research/fixtures/multitracks_short.wav \\")
        print("      --out-dir evaluation/yourmt3/smoke")
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

    print(f"[main] Transcribiendo: {audio_p.name} → {out_mid}")
    audio_bytes = audio_p.read_bytes()
    [midi_bytes] = transcribe_batch.remote([audio_bytes])

    if not midi_bytes:
        print("[main] ERROR: la transcripción devolvió vacío.")
        raise SystemExit(1)

    out_mid.write_bytes(midi_bytes)
    print(f"[main] Guardado: {out_mid} ({len(midi_bytes)} bytes)")


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
      - input.wav  (audio a transcribir — descargado con fetch_fixtures.sh)

    Output por test: transcribed_cuda.mid

    Ejemplos:
        modal run research/research_yourmt3_modal.py::eval_all \\
            --eval-dir evaluation/yourmt3

        # Solo tests 1, 4 y 5 (Slakh):
        modal run research/research_yourmt3_modal.py::eval_all \\
            --eval-dir evaluation/yourmt3 \\
            --only 1,4,5
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
            print(f"[skip] {td.name}: ya tiene transcribed_cuda.mid. Usa --force para regenerar.")
            continue

        audio_bytes_list.append(input_wav.read_bytes())
        valid_dirs.append((td, out_mid))
        print(f"[eval_all] Encolado: {td.name} ({input_wav.name}, {input_wav.stat().st_size / 1e6:.1f} MB)")

    if not audio_bytes_list:
        print("[eval_all] Nada que transcribir (todos los tests ya tienen outputs o faltan inputs).")
        return

    print(f"\n[eval_all] Transcribiendo {len(audio_bytes_list)} tests en Modal ({DEFAULT_GPU}) …\n")
    t0 = time.time()
    results = transcribe_batch.remote(audio_bytes_list)

    ok = 0
    for (td, out_mid), midi_bytes in zip(valid_dirs, results):
        if midi_bytes:
            out_mid.write_bytes(midi_bytes)
            print(f"[eval_all] ✓ {td.name} → {out_mid.name} ({len(midi_bytes)} bytes)")
            ok += 1
        else:
            print(f"[eval_all] ✗ {td.name} — transcripción fallida")

    elapsed = time.time() - t0
    print(f"\n[eval_all] Completado: {ok}/{len(audio_bytes_list)} OK en {elapsed:.0f}s")
    print(f"[eval_all] Outputs en: {eval_path.resolve()}")
    print("[eval_all] Siguiente paso: arrastrar los .mid a REAPER y escuchar.")
