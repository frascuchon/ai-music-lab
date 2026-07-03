"""
Modal.com inference app para MusicGen-medium (facebook/musicgen-medium) — subtarea GENERACIÓN.

MusicGen (Meta AudioCraft) genera música de alta calidad condicionada exclusivamente por
texto. Esta es la variante de GENERACIÓN PURA — sin audio de referencia. Para la variante
con condicionamiento melódico (melody edit), usar research_musicgen_melody_modal.py.

Punto de entrada DAW:
  - Input: prompt de texto + duración en segundos.
  - Output: WAV mono 32 kHz, hasta 30 s.
  - Ideal para: generar música de acompañamiento, loops de fondo, demos instrumentales.
  - Licencia: MIT (código) / CC-BY-NC (pesos) — solo investigación.

Condiciones oficiales (Copet et al., NeurIPS 2023):
  https://github.com/facebookresearch/audiocraft
  - `MusicGen.get_pretrained("facebook/musicgen-medium")` (1.5B parámetros)
  - `model.set_generation_params(duration=seconds)`
  - `model.generate([prompt])` → audio tensor (batch, channels, samples)
  - Sampling rate: 32000 Hz, mono

Setup (descarga pesos al Volume, una vez, ~3 GB):
    modal run research_musicgen_gen_modal.py::setup

Generación libre:
    modal run research_musicgen_gen_modal.py::main \\
        --prompt "upbeat jazz trio with piano, bass and drums" \\
        --seconds 15.0 \\
        --out-dir /tmp/musicgen_gen_test

Coste estimado (A10G): setup (~3 GB) ~$0.04 · 1 generación de 15 s ~$0.03-0.06
(autoregresivo, ~50 tokens/s a 32 kHz)
"""

import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("musicgen-gen-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

DEFAULT_VARIANT = os.environ.get("MUSICGEN_GEN_VARIANT", "facebook/musicgen-medium")
DEFAULT_GPU = os.environ.get("MUSICGEN_GEN_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("MUSICGEN_GEN_SEED", "42"))
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

app = modal.App("musicgen-gen-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=3600)
def setup(variant: str = DEFAULT_VARIANT):
    """
    Descarga pesos de MusicGen-medium (~3 GB). Ejecutar una vez:
        modal run research_musicgen_gen_modal.py::setup
    """
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    t0 = time.time()
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
    sr_out = model.config.audio_encoder.sampling_rate  # 32000 Hz
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
            # audio_values: (batch=1, channels, samples)
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
    Genera audio desde un prompt de texto.

    Ejemplo:
        modal run research_musicgen_gen_modal.py::main \\
            --prompt "upbeat jazz trio with piano, bass and drums" \\
            --seconds 15.0 \\
            --out-dir /tmp/musicgen_gen_test
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
