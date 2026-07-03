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
# Ruta en HuggingFace Hub (sin amt/ prefix)
CKPT_HF_PATH = f"logs/2024/{EXP_ID}/checkpoints/{CKPT_FILE}"
# Ruta en el Volume: initialize_trainer() resuelve como amt/logs/... relativo a SPACE_DIR
CKPT_VOL_PATH = f"amt/{CKPT_HF_PATH}"

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
    .run_commands(
        # Audit de la estructura clonada — aparece en los logs del build
        f"echo '=== Space .py files ===' && find {SPACE_DIR} -name '*.py' "
        f"! -path '*/__pycache__/*' ! -path '*/logs/*' | sort | head -80"
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
        "wandb",          # importado en ymt3.py (top-level); solo necesario en runtime, no para log
        "omegaconf>=2.3", # usado por init_train
        "git+https://github.com/craffel/mir_eval.git",
    )
)

app = modal.App("yourmt3-inference", image=image)


# ---------------------------------------------------------------------------
# Diagnóstico — ejecutar si hay errores de checkpoint
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, timeout=120)
def debug_checkpoint():
    """
    Inspecciona el checkpoint descargado en el Volume.
    modal run research/research_yourmt3_modal.py::debug_checkpoint
    """
    ckpt = f"{WEIGHTS_MOUNT}/{CKPT_VOL_PATH}"
    print(f"Path:        {ckpt}")
    print(f"Exists:      {os.path.exists(ckpt)}")
    print(f"Is symlink:  {os.path.islink(ckpt)}")
    if os.path.islink(ckpt):
        target = os.readlink(ckpt)
        real = os.path.realpath(ckpt)
        print(f"Symlink →    {target}")
        print(f"Real path:   {real}")
        print(f"Real exists: {os.path.exists(real)}")

    if os.path.exists(ckpt):
        size = os.path.getsize(ckpt)
        print(f"Size:        {size} bytes ({size/1e6:.1f} MB)")
        with open(ckpt, "rb") as f:
            first = f.read(64)
        print(f"First hex:   {first.hex()}")
        print(f"First repr:  {first!r}")
        # PyTorch ZIP format starts with PK (0x504b)
        # PyTorch pickle format starts with \x80 (0x80)
        if first[:2] == b"PK":
            print("Formato:     PyTorch ZIP (moderno) ✓")
        elif first[0:1] == b"\x80":
            print("Formato:     PyTorch pickle (legacy) ✓")
        else:
            print("Formato:     DESCONOCIDO — posible LFS pointer o archivo corrupto")


# ---------------------------------------------------------------------------
# Helpers compartidos
# ---------------------------------------------------------------------------
def _setup_space_env():
    """Configura sys.path y el symlink logs/ → volume, y crea model_output/.

    Estructura real del Space (confirmada via build audit):
        /yourmt3_space/
            app.py, model_helper.py, html_helper.py
            amt/
                src/          ← aquí están model/, utils/, config/
                    model/
                    utils/
                    config/
            examples/
            logs/ (LFS)
    """
    for p in [SPACE_DIR, f"{SPACE_DIR}/amt/src"]:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(SPACE_DIR)

    # initialize_trainer resuelve el checkpoint como amt/logs/... relativo a CWD
    # El git clone (GIT_LFS_SKIP_SMUDGE=1) deja amt/logs/ como directorio real
    # con LFS pointer files. Lo reemplazamos por un symlink al volume.
    amt_logs_link = f"{SPACE_DIR}/amt/logs"
    if os.path.isdir(amt_logs_link) and not os.path.islink(amt_logs_link):
        import shutil
        shutil.rmtree(amt_logs_link)
        print(f"[setup_env] Eliminado directorio LFS: {amt_logs_link}")
    if not os.path.lexists(amt_logs_link):
        os.symlink(f"{WEIGHTS_MOUNT}/amt/logs", amt_logs_link)
        print(f"[setup_env] Symlink creado: {amt_logs_link} → {WEIGHTS_MOUNT}/amt/logs")

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
    Descarga el checkpoint YPTF.MoE+Multi (noPS) al Volume con requests directo
    (sin pasar por el caché de huggingface_hub que usa symlinks y se rompe entre containers).

    Ejecutar una sola vez:
        modal run research/research_yourmt3_modal.py::setup
    """
    import requests

    # Descargar a WEIGHTS_MOUNT/amt/logs/... para que el symlink amt/logs → WEIGHTS_MOUNT/amt/logs funcione
    local_ckpt = f"{WEIGHTS_MOUNT}/{CKPT_VOL_PATH}"

    if os.path.exists(local_ckpt) and not os.path.islink(local_ckpt):
        size_mb = os.path.getsize(local_ckpt) / 1e6
        if size_mb > 100:
            print(f"[setup] Checkpoint real ya existe: {local_ckpt} ({size_mb:.0f} MB)")
            return
        print(f"[setup] Archivo existente demasiado pequeño ({size_mb:.1f} MB), re-descargando…")
    elif os.path.islink(local_ckpt):
        print("[setup] Symlink detectado — eliminando para descargar archivo real…")
        os.unlink(local_ckpt)

    os.makedirs(os.path.dirname(local_ckpt), exist_ok=True)

    url = f"https://huggingface.co/{MODEL_HF_REPO}/resolve/main/{CKPT_HF_PATH}"
    print(f"[setup] Descargando desde {url} …")

    t0 = time.time()
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    total_bytes = 0
    with open(local_ckpt, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            total_bytes += len(chunk)
            if total_bytes % (100 * 1024 * 1024) < (8 * 1024 * 1024):
                print(f"[setup]   {total_bytes / 1e6:.0f} MB descargados…")

    elapsed = time.time() - t0
    size_mb = total_bytes / 1e6
    print(f"[setup] OK — {size_mb:.0f} MB en {elapsed:.0f}s ({size_mb/elapsed:.1f} MB/s)")

    # Verificar que es un archivo PyTorch válido
    with open(local_ckpt, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"PK" or magic[0:2] == b"\x80\x04":
        print(f"[setup] Verificación OK — magic bytes: {magic!r}")
    else:
        print(f"[setup] WARNING: magic bytes inesperados: {magic!r}. El checkpoint puede estar corrupto.")

    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Núcleo de inferencia (dentro del container)
# ---------------------------------------------------------------------------
def _load_model_once():
    """Carga modelo YourMT3+ en GPU. Llamar después de _setup_space_env()."""
    import torch
    from model_helper import load_model_checkpoint

    # Monkey-patch temporal: intercepta torch.load para loguear la ruta exacta
    _original_torch_load = torch.load
    def _patched_load(f, *args, **kwargs):
        path_str = str(f) if not isinstance(f, int) else f"<fd={f}>"
        print(f"[modal] torch.load → {path_str}")
        if isinstance(f, (str, os.PathLike)) and os.path.exists(f):
            size = os.path.getsize(f)
            with open(f, "rb") as fh:
                magic = fh.read(4)
            print(f"[modal]   size={size/1e6:.1f}MB  magic={magic!r}")
        return _original_torch_load(f, *args, **kwargs)
    torch.load = _patched_load

    print(f"[modal] Cargando modelo YPTF.MoE+Multi (noPS) en CPU …")
    t0 = time.time()
    model = load_model_checkpoint(args=_model_args(), device="cpu")
    torch.load = _original_torch_load  # restaurar
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
