"""
Modal.com inference app para Stable Audio Open 1.0 (stabilityai/stable-audio-open-1.0).

Stable Audio Open es un modelo DiT (Diffusion Transformer) sobre latentes continuos de
un VAE de audio estéreo 44.1 kHz. Condicionado por texto (T5-XXL) y duración explícita
(en segundos), entrenado en Freesound y Free Music Archive (CC).

Punto de entrada DAW:
  - Input: prompt de texto + duración en segundos + seed opcional.
  - Output: WAV estéreo 44.1 kHz, hasta 47 s.
  - Ideal para: loops de instrumento, one-shots de batería, texturas, FX.
  - Licencia: Stability AI Community License (comercial para ingresos <$1M/año).

⚠️  MODELO GATED — diferencia respecto a scripts previos (Audio2Midi, MidiGenerator):
  Los pesos de Stable Audio Open requieren aceptar la licencia en HuggingFace
  y autenticarse con un token. Pasos previos OBLIGATORIOS:
    1. Aceptar licencia en: https://huggingface.co/stabilityai/stable-audio-open-1.0
    2. Obtener token HF (read access): https://huggingface.co/settings/tokens
    3. Crear secret en Modal:
         modal secret create huggingface HF_TOKEN=<tu_token>
  Sin esto, setup() fallará con HTTP 401.

Pipeline completo:
    [prompt: str + seconds: float]
        ↓
    T5-XXL tokenizer + encoder (condicionamiento de texto)
        ↓
    Condicionamiento de duración (embedding numérico de segundos)
        ↓
    DiT diffusion (latentes audio estéreo) — N steps (default 100)
        ↓
    VAE decoder → waveform estéreo 44.1 kHz
        ↓
    output.wav  (int16, 44100 Hz, 2 canales)

Modelo: stabilityai/stable-audio-open-1.0
  HF Hub:  https://huggingface.co/stabilityai/stable-audio-open-1.0 (GATED)
  Paper:   Evans et al., "Stable Audio Open" — ICASSP 2025
           https://arxiv.org/abs/2407.14358
  Tamaño:  ~3.4 GB total (VAE + transformer + T5)

Setup (descarga pesos al Volume, ejecutar una vez):
    modal run research/research_stable_audio_open_modal.py::setup

Generación libre (single prompt):
    modal run research/research_stable_audio_open_modal.py::main \\
        --prompt "deep house kick drum loop, 120 BPM, punchy, minimal" \\
        --seconds 8.0 \\
        --out-dir evaluation/stable_audio_open/smoke

Benchmark completo (todos los prompts de prompts.json):
    modal run research/research_stable_audio_open_modal.py::eval_all \\
        --prompts-json ../evaluation/prompts.json \\
        --model-dir stable_audio_open

Solo prompts específicos (por id):
    modal run research/research_stable_audio_open_modal.py::eval_all \\
        --prompts-json ../evaluation/prompts.json \\
        --model-dir stable_audio_open \\
        --only prompt01,prompt03,prompt07

GPUs disponibles (env STABLE_AUDIO_GPU):
    A10G     24 GB, ~$1.10/hr  (default) — modelo ~6-8 GB VRAM con FP16
    T4       16 GB, ~$0.60/hr  — debería caber, más lento en difusión
    A100-40G 40 GB, ~$3.70/hr  — overkill, no necesario

Coste estimado (A10G, 100 diffusion steps):
    - Primer uso (setup): descarga ~3.4 GB → ~$0.05
    - 1 prompt × 8 s: ~20-40s de inferencia → ~$0.01-0.02
    - 12 prompts (eval_all): ~$0.15-0.25 total
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("stable-audio-open-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

MODEL_HF_REPO = "stabilityai/stable-audio-open-1.0"

# Variante: "1.0" (full) o "small" (solo si se quiere comparar SAO Small)
DEFAULT_MODEL_VARIANT = os.environ.get("STABLE_AUDIO_VARIANT", "1.0")

DEFAULT_GPU = os.environ.get("STABLE_AUDIO_GPU", "A10G")

# Número de pasos de difusión (quality vs speed)
# 100 = calidad completa (paper), 50 = ~2× más rápido con ligera pérdida de calidad
DEFAULT_STEPS = int(os.environ.get("STABLE_AUDIO_STEPS", "100"))

# Seed reproducible (-1 = aleatorio)
DEFAULT_SEED = int(os.environ.get("STABLE_AUDIO_SEED", "42"))

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# Nota: stable-audio-tools instala torch 2.x como dep; lo instalamos explícitamente
# con cu121 para asegurar la variante CUDA correcta.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "libsndfile1"])
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # stable-audio-tools: inferencia oficial de Stability AI
        # Incluye el pipeline de diffusers-compatible y el VAE propio
        "stable-audio-tools>=0.0.16",
        # diffusers como alternativa al pipeline oficial (StableAudioPipeline)
        "diffusers>=0.30.0",
        "transformers>=4.44.0",
        "accelerate>=0.30.0",
        # Audio I/O
        "soundfile>=0.12.1",
        "scipy>=1.13",
        # HuggingFace (necesario para download con token en setup)
        "huggingface_hub>=0.24.0",
    )
)

app = modal.App("stable-audio-open-inference", image=image)

# Secret con HF_TOKEN — OBLIGATORIO para modelo gated.
# Crear con: modal secret create huggingface HF_TOKEN=<tu_token>
hf_secret = modal.Secret.from_name("huggingface")


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    timeout=1800,
)
def setup():
    """
    Descarga los pesos de Stable Audio Open al Modal Volume.
    Ejecutar una sola vez (~3.4 GB, ~$0.05 en A10G):
        modal run research/research_stable_audio_open_modal.py::setup

    Prerequisito: aceptar licencia en HuggingFace y tener el secret "huggingface"
    creado con: modal secret create huggingface HF_TOKEN=<tu_token>
    """
    from huggingface_hub import snapshot_download
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN no encontrado. Crear secret: "
            "modal secret create huggingface HF_TOKEN=<tu_token>"
        )

    dest = f"{WEIGHTS_MOUNT}/stable-audio-open-1.0"

    # Check si ya existe (comprueba que hay config.json, no solo la carpeta)
    config_path = f"{dest}/config.json"
    if os.path.exists(config_path):
        size_gb = sum(
            os.path.getsize(os.path.join(d, f))
            for d, _, files in os.walk(dest)
            for f in files
        ) / 1e9
        print(f"[setup] Modelo ya descargado en {dest} ({size_gb:.1f} GB). Nada que hacer.")
        return

    print(f"[setup] Descargando {MODEL_HF_REPO} → {dest} …")
    print("[setup] (requiere que hayas aceptado la licencia en HuggingFace)")
    t0 = time.time()

    snapshot_download(
        repo_id=MODEL_HF_REPO,
        local_dir=dest,
        token=hf_token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )

    elapsed = time.time() - t0
    size_gb = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, files in os.walk(dest)
        for f in files
    ) / 1e9
    print(f"[setup] Descarga completa: {size_gb:.1f} GB en {elapsed:.0f}s")

    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Helpers de inferencia (dentro del container)
# ---------------------------------------------------------------------------

def _load_pipeline(variant: str = "1.0"):
    """
    Carga el pipeline de Stable Audio Open desde el Volume.

    Usa StableAudioPipeline de diffusers como interfaz principal.
    El pipeline incluye: AutoencoderKL (VAE), UNet (DiT), T5TextEncoder.
    """
    import torch
    from diffusers import StableAudioPipeline

    model_path = f"{WEIGHTS_MOUNT}/stable-audio-open-1.0"
    print(f"[load_pipeline] Cargando desde {model_path} …")
    t0 = time.time()

    pipe = StableAudioPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = pipe.to("cuda")
    print(f"[load_pipeline] Pipeline listo ({time.time()-t0:.1f}s)")
    return pipe


def _generate_one(
    pipe,
    text: str,
    seconds: float,
    seed: int = 42,
    num_inference_steps: int = 100,
) -> bytes:
    """
    Genera un WAV estéreo 44.1 kHz a partir de texto + duración.

    Devuelve bytes WAV (int16, 44100 Hz, 2 canales).
    """
    import torch
    import soundfile as sf
    import io

    generator = torch.Generator(device="cuda")
    if seed >= 0:
        generator.manual_seed(seed)

    output = pipe(
        text,
        seconds_total=seconds,
        num_inference_steps=num_inference_steps,
        audio_end_in_s=seconds,
        generator=generator,
    )
    # output.audios: tensor shape (batch, channels, samples)
    audio = output.audios[0]  # (channels, samples)
    # Convertir a numpy int16 para soundfile
    audio_np = audio.float().cpu().numpy().T  # (samples, channels)

    buf = io.BytesIO()
    sf.write(buf, audio_np, samplerate=44100, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Función Modal principal — genera batch de prompts
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    timeout=7200,
    gpu=DEFAULT_GPU,
)
def generate_batch(
    prompt_dicts: list[dict],
    seed: int = DEFAULT_SEED,
    num_inference_steps: int = DEFAULT_STEPS,
) -> list[bytes]:
    """
    Genera audio para cada prompt del batch.

    prompt_dicts: lista de dicts con campos:
        - "text": str — descripción del audio a generar
        - "seconds": float — duración objetivo en segundos

    Returns: list[bytes] — bytes WAV por prompt (b"" si falló)
    """
    pipe = _load_pipeline()

    results = []
    for i, pd in enumerate(prompt_dicts):
        text = pd["text"]
        seconds = float(pd.get("seconds", 8.0))
        prompt_seed = seed + i  # seed diferente por prompt para variedad

        t0 = time.time()
        try:
            wav_bytes = _generate_one(
                pipe, text, seconds, seed=prompt_seed,
                num_inference_steps=num_inference_steps,
            )
            elapsed = time.time() - t0
            print(
                f"[generate] [{i+1}/{len(prompt_dicts)}] OK — "
                f"{seconds}s, {len(wav_bytes)//1024} KB, {elapsed:.1f}s inferencia"
            )
            results.append(wav_bytes)
        except Exception as exc:
            print(f"[generate] [{i+1}/{len(prompt_dicts)}] ERROR: {exc}")
            results.append(b"")

    return results


# ---------------------------------------------------------------------------
# Entrypoint: un prompt → out-dir
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = "",
    seconds: float = 8.0,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Genera audio para un prompt y lo guarda en out-dir/output.wav.

    Ejemplo:
        modal run research/research_stable_audio_open_modal.py::main \\
            --prompt "deep house kick drum loop, 120 BPM, punchy, minimal" \\
            --seconds 8.0 \\
            --out-dir evaluation/stable_audio_open/smoke
    """
    if not prompt:
        print("ERROR: --prompt requerido. Ejemplo:")
        print('  modal run research/research_stable_audio_open_modal.py::main \\')
        print('      --prompt "deep house kick drum loop, 120 BPM" \\')
        print('      --seconds 8.0 \\')
        print('      --out-dir evaluation/stable_audio_open/smoke')
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_wav = out_p / "output.wav"

    if out_wav.exists() and not force:
        print(f"[main] Ya existe {out_wav}. Usa --force para regenerar.")
        return

    print(f"[main] Generando: {seconds}s · seed={seed}")
    print(f"[main] Prompt: {prompt}")

    [wav_bytes] = generate_batch.remote(
        [{"text": prompt, "seconds": seconds}],
        seed=seed,
    )

    if not wav_bytes:
        print("[main] ERROR: la generación devolvió vacío.")
        raise SystemExit(1)

    out_wav.write_bytes(wav_bytes)
    print(f"[main] Guardado: {out_wav} ({len(wav_bytes)//1024} KB)")


# ---------------------------------------------------------------------------
# Entrypoint: todos los prompts de prompts.json → evaluation/<model>/
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    prompts_json: str = "../evaluation/prompts.json",
    model_dir: str = "stable_audio_open",
    eval_base: str = "../evaluation",
    seed: int = DEFAULT_SEED,
    force: bool = False,
    only: str = "",
):
    """
    Genera output.wav para todos los prompts de prompts.json.

    Cada prompt genera: evaluation/<model_dir>/<prompt_id>/output.wav

    Ejemplos:
        modal run research/research_stable_audio_open_modal.py::eval_all \\
            --prompts-json ../evaluation/prompts.json \\
            --model-dir stable_audio_open

        # Solo prompts 1, 3 y 7:
        modal run research/research_stable_audio_open_modal.py::eval_all \\
            --prompts-json ../evaluation/prompts.json \\
            --model-dir stable_audio_open \\
            --only prompt01,prompt03,prompt07
    """
    prompts_path = Path(prompts_json).resolve()
    if not prompts_path.exists():
        print(f"ERROR: No existe {prompts_path}")
        raise SystemExit(1)

    with open(prompts_path) as f:
        data = json.load(f)

    prompts = data["prompts"]

    # Filtro --only
    if only:
        only_ids = {s.strip() for s in only.split(",")}
        prompts = [p for p in prompts if p["id"] in only_ids]

    if not prompts:
        print("ERROR: no hay prompts tras aplicar el filtro --only")
        raise SystemExit(1)

    eval_base_path = Path(eval_base).resolve()
    model_path = eval_base_path / model_dir

    # Detectar cuáles faltan (skip si ya existe y no --force)
    pending: list[dict] = []
    out_paths: list[Path] = []

    for p in prompts:
        out_dir = model_path / p["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / "output.wav"

        if out_wav.exists() and not force:
            print(f"[skip] {p['id']}: ya tiene output.wav. Usa --force para regenerar.")
            continue

        pending.append({"text": p["text"], "seconds": float(p.get("seconds", 8.0))})
        out_paths.append(out_wav)
        print(f"[eval_all] Encolado: {p['id']} ({p['category']}, {p.get('seconds',8.0)}s)")

    if not pending:
        print("[eval_all] Todos los prompts ya tienen output.wav. Usa --force para regenerar.")
        return

    print(f"\n[eval_all] Enviando {len(pending)} prompts a Modal ({DEFAULT_GPU}) …")
    t0 = time.time()

    wav_bytes_list = generate_batch.remote(pending, seed=seed)

    elapsed = time.time() - t0
    print(f"[eval_all] Batch completado en {elapsed:.0f}s")

    ok = 0
    for i, (wav_bytes, out_wav) in enumerate(zip(wav_bytes_list, out_paths)):
        if not wav_bytes:
            print(f"[eval_all] ERROR: prompt {i+1} devolvió vacío — saltando")
            continue
        out_wav.write_bytes(wav_bytes)
        print(f"[eval_all] Guardado: {out_wav.parent.name}/output.wav ({len(wav_bytes)//1024} KB)")
        ok += 1

    print(f"\n[eval_all] {ok}/{len(pending)} prompts generados con éxito.")
    print(f"\nSiguiente paso:")
    print(f"  bash ../evaluation/render_norm.sh               # output.wav → .mp3")
    print(f"  cd . && uv run python ../evaluation/compute_metrics.py  # FAD + CLAP")
