"""
Modal.com inference app para MAGNeT-medium-30secs (facebook/magnet-medium-30secs) —
subtarea GENERACIÓN.

MAGNeT (Meta AudioCraft, 2024) es un generador de música TEXT-TO-MUSIC no autoregresivo
basado en Masked Audio Generation. A diferencia de MusicGen (autoregresivo, lento),
MAGNeT genera en múltiples pases de desmascarado paralelos — mucho más rápido (×10-20)
con calidad comparable. Útil para prototipado rápido y generación interactiva.

Punto de entrada DAW:
  - Input: prompt de texto + duración en segundos.
  - Output: WAV estéreo 32 kHz, hasta 30 s.
  - Ideal para: generación rápida de ideas musicales, demos de producción.
  - Licencia: CC-BY-NC 4.0 — solo uso no comercial.

Condiciones oficiales (Ziv et al., 2024):
  https://github.com/facebookresearch/audiocraft/blob/main/docs/MAGNET.md
  - `MAGNeT.get_pretrained("facebook/magnet-medium-30secs")` (1.5B parámetros)
    Alternativa: "facebook/magnet-medium-10secs" para clips más cortos.
  - `model.set_generation_params(use_sampling=True, top_k=0, top_p=0.9)`
  - `model.generate([prompt])` → audio tensor (batch, channels, samples)
  - Sampling rate: 32000 Hz

Setup (descarga pesos al Volume, una vez, ~3 GB):
    modal run research_magnet_modal.py::setup

Generación libre:
    modal run research_magnet_modal.py::main \\
        --prompt "melodic synthwave with arpeggios, 120 BPM" \\
        --seconds 20.0 \\
        --out-dir /tmp/magnet_test

Coste estimado (A10G): setup (~3 GB) ~$0.04 · 1 generación de 20 s ~$0.01-0.02
(no-AR, mucho más rápido que MusicGen)
"""

import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("magnet-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

DEFAULT_VARIANT = os.environ.get("MAGNET_VARIANT", "facebook/magnet-medium-30secs")
DEFAULT_GPU = os.environ.get("MAGNET_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("MAGNET_SEED", "42"))
MAX_SECONDS = 30.0

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# MAGNeT no está en transformers — usa la librería audiocraft de Facebook.
# audiocraft requiere compilar av (PyAV) desde fuente → necesita las cabeceras
# de desarrollo de FFmpeg vía apt.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git", "ffmpeg", "libsndfile1", "pkg-config",
        "libavcodec-dev", "libavformat-dev", "libavdevice-dev",
        "libavutil-dev", "libswscale-dev", "libswresample-dev",
    ])
    .pip_install("torch", "torchaudio")
    .pip_install("audiocraft")
    .pip_install("soundfile>=0.12.1")
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache",
        "TORCH_HOME": f"{WEIGHTS_MOUNT}/torch-cache",
    })
)

app = modal.App("magnet-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=3600)
def setup(variant: str = DEFAULT_VARIANT):
    """
    Descarga pesos de MAGNeT-medium (~3 GB). Ejecutar una vez:
        modal run research_magnet_modal.py::setup
    """
    from audiocraft.models import MAGNeT

    t0 = time.time()
    MAGNeT.get_pretrained(variant)
    print(f"[setup] {variant} descargado en {time.time()-t0:.0f}s")
    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Función Modal principal — generación por lotes
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=7200)
def generate_batch(
    jobs: list[dict],
    variant: str = DEFAULT_VARIANT,
    seed: int = DEFAULT_SEED,
) -> list[bytes]:
    """
    Cada job: {"text": str, "seconds": float}.
    Returns: list[bytes] WAV (b"" si falló).
    """
    import io

    import soundfile as sf
    import torch
    from audiocraft.models import MAGNeT

    t0 = time.time()
    model = MAGNeT.get_pretrained(variant)
    model.to("cuda")
    sr_out = model.sample_rate  # 32000 Hz
    print(f"[load_model] {variant} listo ({time.time()-t0:.1f}s)")

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            seconds = min(float(job.get("seconds") or 15.0), MAX_SECONDS)
            torch.manual_seed(seed + i)

            model.set_generation_params(use_sampling=True, top_k=0, top_p=0.9,
                                        temperature=3.0, max_cfg_coef=10.0,
                                        min_cfg_coef=1.0, decoding_steps=[20, 10, 10, 10])
            wav = model.generate([job["text"]])
            # wav: (batch=1, channels, samples) — float32 CPU
            audio = wav[0].cpu()

            buf = io.BytesIO()
            sf.write(buf, audio.numpy().T, samplerate=sr_out, format="WAV", subtype="PCM_16")
            buf.seek(0)
            wav_bytes = buf.read()
            print(
                f"[generate] [{i+1}/{len(jobs)}] OK — '{job['text'][:50]}' "
                f"({seconds}s) → {len(wav_bytes)//1024} KB, {time.time()-t0:.1f}s"
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
    seconds: float = 15.0,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Genera música desde un prompt de texto (no-AR, rápido).

    Ejemplo:
        modal run research_magnet_modal.py::main \\
            --prompt "melodic synthwave with arpeggios, 120 BPM" \\
            --seconds 20.0 \\
            --out-dir /tmp/magnet_test
    """
    if not prompt:
        print("ERROR: --prompt es requerido.")
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_wav = out_p / "output.wav"
    if out_wav.exists() and not force:
        print(f"[main] Ya existe {out_wav}. Usa --force para regenerar.")
        return

    seconds = min(seconds, MAX_SECONDS)
    print(f"[main] Generando (MAGNeT no-AR): '{prompt}' ({seconds}s)")
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
