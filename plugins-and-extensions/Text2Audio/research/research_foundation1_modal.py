"""
Modal.com inference app para Foundation-1 (RoyalCities/Foundation-1).

Foundation-1 es un fine-tune de Stable Audio Open 1.0 entrenado por RoyalCities
sobre su propia librería de samples electrónicos (acid, house, techno, dubstep).
El objetivo de la evaluación es cuantificar la ganancia del fine-tuning de dominio
respecto a SAO 1.0 base en música electrónica.

⚠️  FORMATO DE PROMPT DIFERENTE AL RESTO DE MODELOS:
  Foundation-1 NO usa frases en lenguaje natural. Usa ETIQUETAS separadas por coma:
    instrumento, timbre, FX, genre tags, bars, BPM, key
  Ejemplo correcto:
    "Bass, FM Bass, Medium Delay, Medium Reverb, Phaser, Acid, Gritty, 8 Bars, 140 BPM, E minor"
  Usar siempre evaluation/prompts_official.json → sección "foundation_1".
  NO usar evaluation/prompts.json (diseñado para SAO 1.0 con frases naturales).

Pesos PÚBLICOS (no gated) — a diferencia de SAO 1.0, no requiere secret HF.

Pipeline completo (idéntico a SAO 1.0 — mismo VAE, solo pesos DiT distintos):
    [prompt en formato tag + seconds]
        ↓
    T5-XXL tokenizer + encoder
        ↓
    Condicionamiento de duración
        ↓
    DiT diffusion (pesos fine-tuned sobre samples electrónicos)
        ↓
    VAE decoder → waveform estéreo 44.1 kHz
        ↓
    output.wav  (int16, 44100 Hz, 2 canales)

Modelo: RoyalCities/Foundation-1
  HF Hub:  https://huggingface.co/RoyalCities/Foundation-1  (PÚBLICO, sin token)
  GitHub:  https://github.com/RoyalCities/RC-stable-audio-tools
  Demos:   https://huggingface.co/RoyalCities/Foundation-1/tree/main/examples
  Space:   https://huggingface.co/spaces/multimodalart/Foundation-1

Relación con SAO 1.0:
  - Misma arquitectura: DiT + VAE + T5-XXL a 44.1 kHz estéreo.
  - Fine-tune sobre samples de música electrónica → sesgado a acid/house/techno.
  - Para géneros no electrónicos (jazz, clásico, pop) los resultados serán pobres.
  - Pesos base SAO 1.0: ~3.4 GB. Foundation-1 sobrescribe solo el DiT → mismo tamaño.

Setup (descarga pesos al Volume, ejecutar una vez):
    modal run research/research_foundation1_modal.py::setup

Smoke test — verificación del script (prompt oficial del model card):
    modal run research/research_foundation1_modal.py::main \\
        --prompt "Bass, FM Bass, Medium Delay, Medium Reverb, Phaser, Acid, Gritty, Dubstep, 8 Bars, 140 BPM, E minor" \\
        --seconds 13.71 \\
        --out-dir evaluation/foundation_1/smoke

Benchmark completo (prompts oficiales en formato tag):
    modal run research/research_foundation1_modal.py::eval_all \\
        --prompts-json ../evaluation/prompts_official.json \\
        --model-section foundation_1

Comparación head-to-head con SAO 1.0 (mismos prompts DAW convertidos a formato tag):
    modal run research/research_foundation1_modal.py::eval_all \\
        --prompts-json ../evaluation/prompts_official.json \\
        --model-section foundation_1 \\
        --only f1_smoke_01,f1_smoke_02,f1_smoke_03

GPUs disponibles (env FOUNDATION1_GPU):
    A10G     24 GB, ~$1.10/hr  (default)
    T4       16 GB, ~$0.60/hr  — debería ser suficiente

Coste estimado (A10G, 100 diffusion steps):
    - Primer uso (setup): descarga ~3.4 GB → ~$0.05
    - 1 prompt × 8 s: ~20-40 s inferencia → ~$0.01-0.02
    - 3 smoke test prompts: ~$0.05-0.10 total
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("foundation1-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

MODEL_HF_REPO = "RoyalCities/Foundation-1"

DEFAULT_GPU = os.environ.get("FOUNDATION1_GPU", "A10G")
DEFAULT_STEPS = int(os.environ.get("FOUNDATION1_STEPS", "100"))
DEFAULT_SEED = int(os.environ.get("FOUNDATION1_SEED", "42"))

# ---------------------------------------------------------------------------
# Container image (idéntico a SAO 1.0, sin necesidad de secret HF)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "libsndfile1"])
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "stable-audio-tools>=0.0.16",
        "diffusers>=0.30.0",
        "transformers>=4.44.0",
        "accelerate>=0.30.0",
        "soundfile>=0.12.1",
        "scipy>=1.13",
        "huggingface_hub>=0.24.0",
    )
)

app = modal.App("foundation1-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume (sin token HF, repo público)
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=1800,
)
def setup():
    """
    Descarga los pesos de Foundation-1 al Modal Volume.
    Ejecutar una sola vez (~3.4 GB, ~$0.05):
        modal run research/research_foundation1_modal.py::setup

    No requiere HF_TOKEN — pesos públicos.
    """
    from huggingface_hub import snapshot_download

    dest = f"{WEIGHTS_MOUNT}/foundation1"
    config_path = f"{dest}/config.json"

    if os.path.exists(config_path):
        size_gb = sum(
            os.path.getsize(os.path.join(d, f))
            for d, _, files in os.walk(dest)
            for f in files
        ) / 1e9
        print(f"[setup] Foundation-1 ya descargado en {dest} ({size_gb:.1f} GB).")
        return

    print(f"[setup] Descargando {MODEL_HF_REPO} → {dest} …")
    t0 = time.time()

    snapshot_download(
        repo_id=MODEL_HF_REPO,
        local_dir=dest,
        ignore_patterns=["*.msgpack", "*.h5", "*.mp3", "examples/"],
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
# Helpers de inferencia
# ---------------------------------------------------------------------------

def _load_pipeline():
    import torch
    from diffusers import StableAudioPipeline

    model_path = f"{WEIGHTS_MOUNT}/foundation1"
    print(f"[load_pipeline] Cargando Foundation-1 desde {model_path} …")
    t0 = time.time()

    pipe = StableAudioPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = pipe.to("cuda")
    print(f"[load_pipeline] Pipeline listo ({time.time()-t0:.1f}s)")
    return pipe


def _generate_one(pipe, text: str, seconds: float, seed: int = 42,
                   num_inference_steps: int = 100) -> bytes:
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
    audio = output.audios[0]
    audio_np = audio.float().cpu().numpy().T

    buf = io.BytesIO()
    sf.write(buf, audio_np, samplerate=44100, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Función Modal principal
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
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

    prompt_dicts: lista de dicts con "text" (formato tag) y "seconds".
    Returns: list[bytes] WAV (b"" si falló).
    """
    pipe = _load_pipeline()

    results = []
    for i, pd in enumerate(prompt_dicts):
        text = pd["text"]
        seconds = float(pd.get("seconds", 8.0))
        prompt_seed = seed + i
        t0 = time.time()
        try:
            wav_bytes = _generate_one(pipe, text, seconds, seed=prompt_seed,
                                       num_inference_steps=num_inference_steps)
            elapsed = time.time() - t0
            print(f"[generate] [{i+1}/{len(prompt_dicts)}] OK — {seconds}s, "
                  f"{len(wav_bytes)//1024} KB, {elapsed:.1f}s")
            results.append(wav_bytes)
        except Exception as exc:
            print(f"[generate] [{i+1}/{len(prompt_dicts)}] ERROR: {exc}")
            results.append(b"")

    return results


# ---------------------------------------------------------------------------
# Entrypoint: un prompt
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
    Genera audio para un prompt Foundation-1 (formato tag).

    Smoke test con prompt oficial:
        modal run research/research_foundation1_modal.py::main \\
            --prompt "Bass, FM Bass, Medium Delay, Phaser, Acid, Gritty, 8 Bars, 140 BPM, E minor" \\
            --seconds 13.71 \\
            --out-dir evaluation/foundation_1/smoke
    """
    if not prompt:
        print("ERROR: --prompt requerido.")
        print("Usar formato TAG: 'Instrumento, Timbre, FX, Genre, N Bars, BPM, Key'")
        print("Ejemplo:")
        print('  --prompt "Bass, FM Bass, Phaser, Acid, 8 Bars, 140 BPM, E minor"')
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_wav = out_p / "output.wav"

    if out_wav.exists() and not force:
        print(f"[main] Ya existe {out_wav}. Usa --force para regenerar.")
        return

    print(f"[main] Foundation-1 — {seconds}s · seed={seed}")
    print(f"[main] Prompt (tag format): {prompt}")

    [wav_bytes] = generate_batch.remote([{"text": prompt, "seconds": seconds}], seed=seed)

    if not wav_bytes:
        print("[main] ERROR: la generación devolvió vacío.")
        raise SystemExit(1)

    out_wav.write_bytes(wav_bytes)
    print(f"[main] Guardado: {out_wav} ({len(wav_bytes)//1024} KB)")
    print(f"[main] Comparar con: evaluation/foundation_1/demos/compare_example_1_b__foundation1.mp3")
    print(f"[main] (Descargar demos con: bash evaluation/fetch_foundation1_demos.sh)")


# ---------------------------------------------------------------------------
# Entrypoint: todos los prompts de prompts_official.json → evaluation/foundation_1/
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    prompts_json: str = "../evaluation/prompts_official.json",
    model_section: str = "foundation_1",
    eval_base: str = "../evaluation",
    seed: int = DEFAULT_SEED,
    force: bool = False,
    only: str = "",
):
    """
    Genera output.wav para todos los prompts de la sección foundation_1 de prompts_official.json.

    Nota: usa prompts_official.json (formato tag), NO prompts.json (frases naturales).

    Ejemplos:
        modal run research/research_foundation1_modal.py::eval_all

        modal run research/research_foundation1_modal.py::eval_all \\
            --only f1_smoke_01,f1_smoke_02
    """
    prompts_path = Path(prompts_json).resolve()
    if not prompts_path.exists():
        print(f"ERROR: No existe {prompts_path}")
        raise SystemExit(1)

    with open(prompts_path) as f:
        data = json.load(f)

    if model_section not in data:
        print(f"ERROR: sección '{model_section}' no encontrada en {prompts_path}")
        print(f"Secciones disponibles: {list(data.keys())}")
        raise SystemExit(1)

    prompts = data[model_section]["prompts"]

    if only:
        only_ids = {s.strip() for s in only.split(",")}
        prompts = [p for p in prompts if p["id"] in only_ids]

    if not prompts:
        print("ERROR: no hay prompts tras aplicar el filtro --only")
        raise SystemExit(1)

    eval_base_path = Path(eval_base).resolve()
    model_path = eval_base_path / "foundation_1"

    pending: list[dict] = []
    out_paths: list[Path] = []

    for p in prompts:
        out_dir = model_path / p["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / "output.wav"

        if out_wav.exists() and not force:
            print(f"[skip] {p['id']}: ya tiene output.wav.")
            continue

        pending.append({"text": p["text"], "seconds": float(p.get("seconds", 8.0))})
        out_paths.append(out_wav)
        print(f"[eval_all] Encolado: {p['id']} ({p.get('seconds', 8.0)}s) — {p['text'][:60]}…")

    if not pending:
        print("[eval_all] Todos los prompts ya tienen output.wav.")
        return

    print(f"\n[eval_all] Enviando {len(pending)} prompts a Modal ({DEFAULT_GPU}) …")
    t0 = time.time()
    wav_bytes_list = generate_batch.remote(pending, seed=seed)
    print(f"[eval_all] Batch completado en {time.time()-t0:.0f}s")

    ok = 0
    for wav_bytes, out_wav in zip(wav_bytes_list, out_paths):
        if not wav_bytes:
            print(f"[eval_all] ERROR: {out_wav.parent.name} devolvió vacío")
            continue
        out_wav.write_bytes(wav_bytes)
        print(f"[eval_all] Guardado: {out_wav.parent.name}/output.wav ({len(wav_bytes)//1024} KB)")
        ok += 1

    print(f"\n[eval_all] {ok}/{len(pending)} prompts generados.")
    print(f"\nSiguiente paso:")
    print(f"  bash ../evaluation/render_norm.sh")
    print(f"  uv run python ../evaluation/compute_metrics.py --only foundation_1 --no-fad")
    print(f"  # Comparar cualitativamente con evaluation/foundation_1/demos/*.mp3")
