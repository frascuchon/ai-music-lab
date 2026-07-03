"""
Modal.com inference app para Mustango (declare-lab/mustango) — subtarea GENERACIÓN.

Mustango (Melechovsky et al., NAACL 2024) genera música condicionada por texto con
conocimiento específico del dominio musical: extrae atributos de música (BPM, acorde,
tonalidad, estilo) del prompt con FLAN-T5 + MuBERT, luego genera con un UNet de difusión
sobre espectrogramas mel. Útil para prompts con instrucciones musicales precisas.

Punto de entrada DAW:
  - Input: prompt de texto (funciona mejor con atributos musicales: BPM, tonalidad, etc.).
  - Output: WAV mono 16 kHz, ~10 s (duración fija del modelo).
  - Ideal para: generación rápida con control de BPM/tonalidad, samples de 10 s.
  - Licencia: código Apache 2.0 / pesos CC-BY-NC-4.0 — solo uso no comercial.

⚠️  DURACIÓN FIJA: Mustango genera exactamente ~10 s independientemente del parámetro
    --seconds. El slider de duración de la UI se ignora; el label lo indica.

Condiciones oficiales (Melechovsky et al., NAACL 2024):
  https://github.com/AMAAI-Lab/mustango   ← repo real (no está en PyPI)
  https://huggingface.co/declare-lab/mustango
  - `from mustango import Mustango`
  - `model = Mustango("declare-lab/mustango")` (auto-descarga pesos)
  - `music = model.generate(prompt, steps=200, disable_progress=False)`
  - `music`: numpy array float32 (samples,) a 16000 Hz
  - Ejemplo oficial: "Generate a pleasant piano melody with chords."
  - Paper CLAP score: 0.274 en MusicCaps

Setup (descarga pesos al Volume, una vez, ~8 GB — FLAN-T5-Large + UNet diffusion):
    modal run research_mustango_modal.py::setup

Generación libre:
    modal run research_mustango_modal.py::main \\
        --prompt "Generate a jazz piano piece at 120 BPM in C major with walking bass" \\
        --out-dir /tmp/mustango_test

Coste estimado (A10G): setup (~8 GB) ~$0.08 · 1 generación (200 steps) ~$0.02-0.04
"""

import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("mustango-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

DEFAULT_GPU = os.environ.get("MUSTANGO_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("MUSTANGO_SEED", "42"))
DEFAULT_STEPS = int(os.environ.get("MUSTANGO_STEPS", "200"))
SAMPLE_RATE = 16000  # fijo del modelo

REPO_DIR = "/root/mustango"

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# Mustango NO está en PyPI — hay que clonar AMAAI-Lab/mustango e instalar su
# fork local de diffusers (incluido en el repo como subdirectorio).
# requirements.txt pina versiones antiguas de torch; las omitimos y usamos
# las versiones estándar de Modal (más recientes y con CUDA preinstalado).
_TORCH_SKIP = "|".join([
    "torch==", "torchaudio==", "torchvision==",
    "wandb",    # no necesario para inferencia
    "ssr_eval", # paquete roto en PyPI
])
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "libsndfile1"])
    # 1. Clonar repo
    .run_commands(f"git clone --depth 1 https://github.com/AMAAI-Lab/mustango {REPO_DIR}")
    # 2. Pre-instalar torch (satisface deps antes de requirements.txt)
    .pip_install("torch", "torchaudio", "torchvision")
    # 3. Instalar requirements.txt sin los pines de torch ni paquetes problemáticos
    .run_commands(
        f"grep -vE '{_TORCH_SKIP}' {REPO_DIR}/requirements.txt > /tmp/req_mustango.txt && "
        "pip install -r /tmp/req_mustango.txt"
    )
    # 4. Instalar fork local de diffusers (Mustango no funciona con el PyPI diffusers estándar)
    .run_commands(f"pip install -e {REPO_DIR}/diffusers")
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache",
        "TORCH_HOME": f"{WEIGHTS_MOUNT}/torch-cache",
        "PYTHONPATH": REPO_DIR,
    })
)

app = modal.App("mustango-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=3600)
def setup():
    """
    Descarga pesos de Mustango (~8 GB — FLAN-T5 + UNet). Ejecutar una vez:
        modal run research_mustango_modal.py::setup
    """
    from mustango import Mustango

    t0 = time.time()
    Mustango("declare-lab/mustango")
    print(f"[setup] Mustango descargado en {time.time()-t0:.0f}s")
    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Función Modal principal — generación por lotes
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=7200)
def generate_batch(
    jobs: list[dict],
    seed: int = DEFAULT_SEED,
) -> list[bytes]:
    """
    Cada job: {"text": str}.
    Returns: list[bytes] WAV (b"" si falló). Duración ~10 s fija.
    """
    import io

    import numpy as np
    import soundfile as sf
    import torch
    from mustango import Mustango

    torch.manual_seed(seed)

    t0 = time.time()
    model = Mustango("declare-lab/mustango")
    print(f"[load_model] Mustango listo ({time.time()-t0:.1f}s)")

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            audio = model.generate(job["text"], steps=DEFAULT_STEPS, disable_progress=False)
            # audio: numpy float32 (samples,)
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)
            audio = np.asarray(audio, dtype=np.float32)

            buf = io.BytesIO()
            sf.write(buf, audio, samplerate=SAMPLE_RATE, format="WAV", subtype="PCM_16")
            buf.seek(0)
            wav_bytes = buf.read()
            print(
                f"[generate] [{i+1}/{len(jobs)}] OK — '{job['text'][:50]}' "
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
    seconds: float = 10.0,  # ignorado — duración fija del modelo
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Genera música desde un prompt (con conocimiento de BPM/tonalidad/acorde).
    ⚠️  El parámetro --seconds se ignora: Mustango genera ~10 s siempre.

    Ejemplo:
        modal run research_mustango_modal.py::main \\
            --prompt "Generate a jazz piano piece at 120 BPM in C major with walking bass" \\
            --out-dir /tmp/mustango_test
    """
    if not prompt:
        print("ERROR: --prompt es requerido.")
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_wav = out_p / "output.wav"
    if out_wav.exists() and not force:
        print(f"[main] Ya existe {out_wav}. Usa --force para regenerar.")
        return

    if seconds != 10.0:
        print(f"[main] NOTA: --seconds={seconds} ignorado — Mustango genera ~10 s fijo.")
    print(f"[main] Generando (Mustango, ~10 s fijo): '{prompt}'")
    [wav_bytes] = generate_batch.remote(
        [{"text": prompt}],
        seed=seed,
    )
    if not wav_bytes:
        print("[main] ERROR: la generación devolvió vacío.")
        raise SystemExit(1)
    out_p.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(wav_bytes)
    print(f"[main] Guardado: {out_wav} ({len(wav_bytes)//1024} KB)")
