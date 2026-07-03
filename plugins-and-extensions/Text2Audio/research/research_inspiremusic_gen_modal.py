"""
Modal.com inference app para InspireMusic 1.5B-Long — subtarea GENERACIÓN (text2music).

InspireMusic (Alibaba FunAudioLLM) genera música a partir de texto puro, sin audio de
referencia. Esta es la variante GENERACIÓN PURA (tarea "text2music"). Para continuación
condicionada por audio, usar research_inspiremusic_modal.py.

Punto de entrada DAW:
  - Input: prompt de texto + duración en segundos.
  - Output: WAV 48 kHz estéreo, hasta 240 s.
  - Ideal para: generación de largos fragmentos instrumentales.
  - Licencia: Apache 2.0 — apto para uso comercial.

Reutiliza el Volume inspiremusic-weights (pesos compartidos con research_inspiremusic_modal.py).

Condiciones oficiales (FunAudioLLM/InspireMusic README):
  model = InspireMusicModel(model_name="InspireMusic-1.5B-Long")
  model.inference("text-to-music", text="Generate jazz music.", audio_prompt=None)

Setup (descarga pesos al Volume, compartido con script de continuación):
    modal run research_inspiremusic_gen_modal.py::setup

Generación libre:
    modal run research_inspiremusic_gen_modal.py::main \\
        --prompt "Generate a relaxing ambient piano piece in C major" \\
        --seconds 30.0 \\
        --out-dir /tmp/inspiremusic_gen_test

Coste estimado (A10G): setup (~7 GB, compartido) ~$0.08 · 1 generación de 30 s ~$0.05-0.10
"""

import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config — reutiliza el mismo Volume que el script de continuación
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("inspiremusic-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

MODEL_NAME = os.environ.get("INSPIREMUSIC_MODEL", "InspireMusic-1.5B-Long")
MODEL_HF_REPO = f"FunAudioLLM/{MODEL_NAME}"
MODEL_LOCAL_DIR = f"{WEIGHTS_MOUNT}/{MODEL_NAME}"

REPO_URL = "https://github.com/FunAudioLLM/InspireMusic.git"
REPO_DIR = "/root/InspireMusic"

DEFAULT_GPU = os.environ.get("INSPIREMUSIC_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("INSPIREMUSIC_SEED", "42"))
MAX_GENERATE_SECONDS = float(os.environ.get("INSPIREMUSIC_MAX_SECONDS", "30.0"))

# ---------------------------------------------------------------------------
# Container image (idéntica a research_inspiremusic_modal.py)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "libsndfile1", "sox", "libsox-dev"])
    .run_commands(f"git clone --recursive {REPO_URL} {REPO_DIR}")
    .run_commands(
        f"grep -v 'flash[_-]attn' {REPO_DIR}/requirements.txt > /tmp/req_im.txt && "
        "pip install -r /tmp/req_im.txt"
    )
    .pip_install("soundfile>=0.12.1", "huggingface_hub>=0.24.0")
    .run_commands("pip install torchcodec")
    .run_commands("pip install 'peft>=0.17.0'")
    .run_commands(
        f"sed -i 's/attn_implementation=\"flash_attention_2\"/attn_implementation=\"sdpa\"/g'"
        f" {REPO_DIR}/inspiremusic/transformer/qwen_encoder.py"
    )
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache",
        "PYTHONPATH": f"{REPO_DIR}:{REPO_DIR}/third_party/Matcha-TTS",
    })
)

app = modal.App("inspiremusic-gen-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume (compartido con script de continuación)
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, timeout=3600)
def setup():
    """
    Descarga pesos de InspireMusic (~7 GB). Ejecutar una vez:
        modal run research_inspiremusic_gen_modal.py::setup
    Si ya ejecutaste research_inspiremusic_modal.py::setup, los pesos ya están listos.
    """
    from huggingface_hub import snapshot_download

    if os.path.exists(f"{MODEL_LOCAL_DIR}/llm.pt"):
        print(f"[setup] Modelo ya descargado en {MODEL_LOCAL_DIR}. Nada que hacer.")
        return

    print(f"[setup] Descargando {MODEL_HF_REPO} → {MODEL_LOCAL_DIR} …")
    t0 = time.time()
    snapshot_download(repo_id=MODEL_HF_REPO, local_dir=MODEL_LOCAL_DIR)
    print(f"[setup] Descarga completa en {time.time()-t0:.0f}s")
    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Función Modal principal — generación por lotes (text2music)
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=7200)
def generate_batch(jobs: list[dict], seed: int = DEFAULT_SEED) -> list[bytes]:
    """
    Cada job: {"text": str, "seconds": float}.
    Returns: list[bytes] WAV 48 kHz (b"" si falló).
    """
    import shutil
    import tempfile

    import torch
    import torchaudio

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **kw: None

    from inspiremusic.cli.inference import InspireMusicModel, env_variables

    env_variables()

    # Parchear rutas relativas en los YAML del modelo
    model_dir_path = Path(MODEL_LOCAL_DIR)
    for yaml_path in model_dir_path.glob("*.yaml"):
        content = yaml_path.read_text()
        rel_ref = f"../../pretrained_models/{MODEL_NAME}/"
        if rel_ref in content:
            content = content.replace(rel_ref, f"{MODEL_LOCAL_DIR}/")
            yaml_path.write_text(content)
            print(f"[patch] {yaml_path.name}: rutas relativas → absolutas")

    t0 = time.time()
    result_dir = tempfile.mkdtemp(prefix="inspiremusic_out_")
    model = InspireMusicModel(
        model_name=MODEL_NAME,
        model_dir=MODEL_LOCAL_DIR,
        max_generate_audio_seconds=MAX_GENERATE_SECONDS,
        output_sample_rate=48000,
        gpu=0,
        result_dir=result_dir,
        hub="huggingface",
    )
    weights_vol.commit()
    print(f"[load_model] {MODEL_NAME} listo ({time.time()-t0:.1f}s)")

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            torch.manual_seed(seed + i)

            output_fn = f"output_{i:02d}"
            model.inference(
                task="text-to-music",
                text=job.get("text") or "",
                audio_prompt=None,
                output_fn=output_fn,
                output_format="wav",
            )

            out_path = Path(result_dir) / f"{output_fn}.wav"
            if not out_path.exists():
                candidates = sorted(
                    Path(result_dir).glob("*.wav"),
                    key=lambda p: p.stat().st_mtime,
                )
                if not candidates:
                    raise RuntimeError(f"sin output en {result_dir}")
                out_path = candidates[-1]

            wav_bytes = out_path.read_bytes()
            print(
                f"[generate] [{i+1}/{len(jobs)}] OK — '{job.get('text', '')[:50]}' "
                f"→ {len(wav_bytes)//1024} KB, {time.time()-t0:.1f}s"
            )
            results.append(wav_bytes)
        except Exception as exc:
            import traceback
            print(f"[generate] [{i+1}/{len(jobs)}] ERROR: {exc}")
            print(traceback.format_exc())
            results.append(b"")

    return results


# ---------------------------------------------------------------------------
# Entrypoint: generación libre
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = "",
    seconds: float = 30.0,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Genera música desde un prompt de texto (InspireMusic text2music).

    Ejemplo:
        modal run research_inspiremusic_gen_modal.py::main \\
            --prompt "Generate a relaxing ambient piano piece in C major" \\
            --seconds 30.0 \\
            --out-dir /tmp/inspiremusic_gen_test
    """
    if not prompt:
        print("ERROR: --prompt es requerido.")
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_wav = out_p / "output.wav"
    if out_wav.exists() and not force:
        print(f"[main] Ya existe {out_wav}. Usa --force para regenerar.")
        return

    print(f"[main] Generando (InspireMusic text2music): '{prompt}' (~{seconds}s)")
    [wav_bytes] = generate_batch.remote(
        [{"text": prompt, "seconds": seconds}],
        seed=seed,
    )
    if not wav_bytes:
        print("[main] ERROR: la generación devolvió vacío.")
        raise SystemExit(1)
    out_p.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(wav_bytes)
    print(f"[main] Guardado: {out_wav} ({len(wav_bytes)//1024} KB)")
