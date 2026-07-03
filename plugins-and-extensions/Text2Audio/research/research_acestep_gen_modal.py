"""
Modal.com inference app para ACE-Step 1.5 — subtarea GENERACIÓN (text2music).

ACE-Step 1.5 soporta generación de música de longitud arbitraria condicionada
exclusivamente por texto (tarea "text2music"). Esta es la variante de GENERACIÓN PURA.
Para la variante de edición (cover/style transfer), usar research_acestep_edit_modal.py.

Punto de entrada DAW:
  - Input: prompt de texto + duración en segundos.
  - Output: WAV estéreo 48 kHz.
  - Ideal para: canciones completas (full-song), generación de larga duración.
  - Licencia: MIT (código y pesos) — la más permisiva; apto para uso comercial.

Reutiliza el Volume acestep15-weights (pesos compartidos con research_acestep_edit_modal.py).

Condiciones oficiales (docs/en/INFERENCE.md, verificado 2026-07-02):
  https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
  - Checkpoint DiT: acestep-v15-turbo (2B, ~4.7 GB bf16)
  - Turbo: inference_steps=8, shift=3.0
  - Tarea text2music: caption + duration (segundos) → audio full-song

Setup (descarga pesos al Volume, una vez; compartido con script de edición):
    modal run research_acestep_gen_modal.py::setup

Generación libre:
    modal run research_acestep_gen_modal.py::main \\
        --prompt "uplifting electronic dance music with synthesizers and driving beat" \\
        --seconds 30.0 \\
        --out-dir /tmp/acestep_gen_test

Coste estimado (A10G): setup (~5 GB, compartido) ~$0.05 · 1 generación ~$0.01
(turbo, 8 steps, muy rápido)
"""

import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config — reutiliza el mismo Volume que el script de edición
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("acestep15-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

DIT_CONFIG = os.environ.get("ACESTEP_DIT", "acestep-v15-turbo")
DEFAULT_GPU = os.environ.get("ACESTEP_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("ACESTEP_SEED", "42"))

# ---------------------------------------------------------------------------
# Container image (idéntica a la del script de edición)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "ffmpeg", "libsndfile1"])
    .run_commands(
        "git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /root/ACEStep && "
        "sed -i '/+cu[0-9]/d; /nano-vllm/d' /root/ACEStep/pyproject.toml"
    )
    .pip_install("torch", "torchaudio", "torchvision")
    .run_commands("pip install /root/ACEStep")
    .pip_install("soundfile>=0.12.1")
    .env({"HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache"})
)

app = modal.App("acestep-gen-inference", image=image)


# ---------------------------------------------------------------------------
# Helpers (dentro del container)
# ---------------------------------------------------------------------------

def _init_handlers():
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    t0 = time.time()
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=WEIGHTS_MOUNT,
        config_path=DIT_CONFIG,
        device="cuda",
    )
    llm_handler = LLMHandler()  # sin .initialize() — fases LM omitidas
    print(f"[init] DiT {DIT_CONFIG} listo ({time.time()-t0:.1f}s), LM deshabilitado")
    return dit_handler, llm_handler


def _generate_one(dit_handler, llm_handler, params_kwargs: dict) -> bytes:
    import io
    import tempfile

    import soundfile as sf
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    params = GenerationParams(**params_kwargs)
    config = GenerationConfig(batch_size=1, audio_format="wav")

    with tempfile.TemporaryDirectory() as tmp:
        result = generate_music(dit_handler, llm_handler, params, config, save_dir=tmp)
        if not getattr(result, "success", True) or not result.audios:
            raise RuntimeError("generate_music sin audios (text2music)")

        audio = result.audios[0]
        tensor = audio["tensor"]
        sr = audio["sample_rate"]

        buf = io.BytesIO()
        sf.write(buf, tensor.numpy().T, samplerate=sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=3600)
def setup():
    """
    Descarga pesos de ACE-Step 1.5 (~5 GB) al Volume compartido. Ejecutar una vez:
        modal run research_acestep_gen_modal.py::setup
    Si ya ejecutaste research_acestep_edit_modal.py::setup, los pesos ya están listos.
    """
    _init_handlers()
    weights_vol.commit()
    print("[setup] Pesos descargados y Volume commiteado.")


# ---------------------------------------------------------------------------
# Función Modal principal — generación por lotes
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=7200)
def generate_batch(jobs: list[dict], seed: int = DEFAULT_SEED) -> list[bytes]:
    """
    Cada job: {"text": str, "seconds": float}.
    Returns: list[bytes] WAV 48 kHz (b"" si falló).
    """
    dit_handler, llm_handler = _init_handlers()
    weights_vol.commit()

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            wav_bytes = _generate_one(dit_handler, llm_handler, {
                "task_type": "text2music",
                "caption": job["text"],
                "duration": float(job.get("seconds") or 30.0),
                "thinking": False,
                "inference_steps": 8,
                "shift": 3.0,
                "seed": seed + i,
            })
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
    seconds: float = 30.0,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Genera música desde un prompt de texto (ACE-Step text2music).

    Ejemplo:
        modal run research_acestep_gen_modal.py::main \\
            --prompt "uplifting electronic dance music with synthesizers and driving beat" \\
            --seconds 30.0 \\
            --out-dir /tmp/acestep_gen_test
    """
    if not prompt:
        print("ERROR: --prompt es requerido.")
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_wav = out_p / "output.wav"
    if out_wav.exists() and not force:
        print(f"[main] Ya existe {out_wav}. Usa --force para regenerar.")
        return

    print(f"[main] Generando (ACE-Step text2music): '{prompt}' ({seconds}s)")
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
