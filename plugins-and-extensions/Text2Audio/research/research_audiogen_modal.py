"""
Modal.com inference app para AudioGen-medium (facebook/audiogen-medium) — subtarea GENERACIÓN.

AudioGen (Meta AudioCraft) genera sonido y efectos de audio condicionados por texto.
A diferencia de MusicGen (orientado a música), AudioGen está entrenado en sonidos
del mundo real: pasos, lluvia, voces, objetos, entornos, etc. Útil para sound design.

Punto de entrada DAW:
  - Input: prompt de texto + duración en segundos.
  - Output: WAV mono 16 kHz, hasta 30 s.
  - Ideal para: sound design, efectos de ambiente, one-shots de percusión no tonal.
  - Licencia: CC-BY-NC 4.0 — solo uso no comercial.

Condiciones oficiales (Kreuk et al., 2022):
  https://github.com/facebookresearch/audiocraft
  - `AudioGen.get_pretrained("facebook/audiogen-medium")` (1.5B parámetros)
  - `model.set_generation_params(duration=seconds)`
  - `model.generate([prompt])` → audio tensor (batch, channels, samples)
  - Sampling rate: 16000 Hz, mono

Setup (descarga pesos al Volume, una vez, ~2 GB):
    modal run research_audiogen_modal.py::setup

Generación libre:
    modal run research_audiogen_modal.py::main \\
        --prompt "rain falling on a metal roof with distant thunder" \\
        --seconds 10.0 \\
        --out-dir /tmp/audiogen_test

Coste estimado (A10G): setup (~2 GB) ~$0.03 · 1 generación de 10 s ~$0.02-0.04
"""

import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("audiogen-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

DEFAULT_VARIANT = os.environ.get("AUDIOGEN_VARIANT", "facebook/audiogen-medium")
DEFAULT_GPU = os.environ.get("AUDIOGEN_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("AUDIOGEN_SEED", "42"))
MAX_SECONDS = 30.0

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "libsndfile1"])
    .pip_install(
        "torch", "torchaudio",
        "transformers>=4.39.0,<5.0",
        "soundfile>=0.12.1",
    )
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache",
        "TORCH_HOME": f"{WEIGHTS_MOUNT}/torch-cache",
    })
)

app = modal.App("audiogen-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=3600)
def setup(variant: str = DEFAULT_VARIANT):
    """
    Descarga pesos de AudioGen-medium (~2 GB). Ejecutar una vez:
        modal run research_audiogen_modal.py::setup
    """
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    t0 = time.time()
    # AudioGen usa la misma clase base que MusicGen en transformers
    AutoProcessor.from_pretrained(variant)
    MusicgenForConditionalGeneration.from_pretrained(variant)
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
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(variant)
    model = MusicgenForConditionalGeneration.from_pretrained(variant)
    model.to("cuda")
    sr_out = model.config.audio_encoder.sampling_rate  # 16000 Hz
    print(f"[load_model] {variant} listo ({time.time()-t0:.1f}s)")

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            seconds = min(float(job.get("seconds") or 10.0), MAX_SECONDS)
            torch.manual_seed(seed + i)

            inputs = processor(
                text=[job["text"]],
                return_tensors="pt",
            ).to("cuda")

            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3,
                max_new_tokens=int(seconds * 50),
            )
            audio = audio_values[0].cpu().to(torch.float32)

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
    seconds: float = 10.0,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Genera audio (efectos/sonidos) desde un prompt de texto.

    Ejemplo:
        modal run research_audiogen_modal.py::main \\
            --prompt "rain falling on a metal roof with distant thunder" \\
            --seconds 10.0 \\
            --out-dir /tmp/audiogen_test
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
    print(f"[main] Generando: '{prompt}' ({seconds}s)")
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
