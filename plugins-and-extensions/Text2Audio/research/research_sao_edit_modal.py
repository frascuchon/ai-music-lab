"""
Modal.com inference app para Stable Audio Open 1.0 — subtarea EDICIÓN (style transfer audio2audio).

Stable Audio Open soporta oficialmente style transfer inicializando el ruido de la difusión
con un audio existente (técnica tipo SDEdit sobre latentes), vía `stable-audio-tools`:
    generate_diffusion_cond(..., init_audio=(sr, tensor), init_noise_level=N)
"Es posible hacer style-transfer inicializando el ruido con audio durante el sampling,
modificando la estética de una grabación existente según un prompt de texto, manteniendo
la estructura del audio de referencia" (README de stable-audio-tools).

⚠️  Diferencia con research_stable_audio_open_modal.py (text2audio, S1):
  Aquel usa diffusers.StableAudioPipeline, que NO soporta init_audio. Este script usa el
  pipeline oficial `stable-audio-tools` (mismo checkpoint, mismo Volume) porque es la vía
  documentada por Stability AI para audio-to-audio.

Punto de entrada DAW:
  - Input: audio fuente + prompt de texto + init_noise_level.
  - Output: WAV estéreo 44.1 kHz, misma duración que el source (≤47 s).
  - Ideal para: re-estilizar un loop conservando su groove/estructura.
  - Licencia: Stability Community (comercial <$1M rev) — la misma que el flujo text2audio.

Condiciones oficiales replicadas (model card + stable-audio-tools, verificado 2026-07-02):
  https://huggingface.co/stabilityai/stable-audio-open-1.0 (GATED — secret huggingface-secret)
  https://github.com/Stability-AI/stable-audio-tools
  - Sampling del model card: steps=100, cfg_scale=7, sigma_min=0.3, sigma_max=500,
    sampler_type="dpmpp-3m-sde".
  - Conditioning: [{"prompt": ..., "seconds_start": 0, "seconds_total": <dur source>}].
  - init_noise_level: default 1.0 (default de generate_diffusion_cond; el gradio oficial
    expone un slider 0.0–100.0). Mayor nivel de ruido = más transformación.

Mapeo strength_hint (prompts_edit.json) → init_noise_level:
  subtle → 0.4   (cerca del original)
  moderate → 1.0 (default de la función oficial)
  strong → 4.0   (transformación fuerte; >10 prácticamente regenera)

Setup (verifica/descarga el checkpoint stable-audio-tools en el Volume ya existente):
    modal run research_sao_edit_modal.py::setup

Smoke test oficial:
    modal run research_sao_edit_modal.py::smoke
    # Prompt del model card adaptado al BPM del source: "90 BPM tech house drum loop"
    # sobre el beatbox loop de 90 BPM (asset oficial de audiocraft).
    # → evaluation/edit/sao_a2a/smoke/sao_a2a_smoke_01/output.wav
    # Éxito: drum loop tech house que conserva el groove del beatbox.

Edición libre:
    modal run research_sao_edit_modal.py::main \\
        --source-audio ../evaluation/edit/source_audio/beatbox_loop_90bpm.wav \\
        --prompt "90 BPM tech house drum loop" \\
        --noise-level 1.0 \\
        --out-dir ../evaluation/edit/sao_a2a/manual

Benchmark completo:
    modal run research_sao_edit_modal.py::eval_all

Coste estimado (A10G): setup ~$0 (pesos ya en el Volume de S1) · 1 edición ~$0.01-0.02
· eval_all (10 casos) ~$0.15-0.25
"""

import json
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Mismo Volume que el flujo text2audio (S1) — el snapshot del repo HF incluye tanto el
# formato diffusers como el checkpoint stable-audio-tools (model.safetensors + model_config.json).
weights_vol = modal.Volume.from_name("stable-audio-open-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"
MODEL_LOCAL_DIR = f"{WEIGHTS_MOUNT}/stable-audio-open-1.0"

MODEL_HF_REPO = "stabilityai/stable-audio-open-1.0"

DEFAULT_GPU = os.environ.get("STABLE_AUDIO_GPU", "A10G")
DEFAULT_STEPS = int(os.environ.get("STABLE_AUDIO_STEPS", "100"))
DEFAULT_SEED = int(os.environ.get("STABLE_AUDIO_SEED", "42"))
DEFAULT_NOISE_LEVEL = float(os.environ.get("STABLE_AUDIO_INIT_NOISE", "1.0"))

MODEL_DIR_NAME = "sao_a2a"  # evaluation/edit/<model>/
SUPPORTED_CATEGORIES = {"style_transfer", "instrumentation", "mood_texture"}
STRENGTH_MAP = {"subtle": 0.4, "moderate": 1.0, "strong": 4.0}

SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# Container image
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
        "soundfile>=0.12.1",
        "huggingface_hub>=0.24.0",
        # stable-audio-tools/models/lora/callbacks.py require pytorch_lightning
        "pytorch_lightning>=2.0.0",
    )
    # torch==2.4.0 fue compilado contra numpy 1.x; stable-audio-tools instala numpy 2.x →
    # "numpy.dtype size changed" en runtime. Downgrade en paso separado para no deshacer torch.
    .run_commands("pip install 'numpy<2.0'")
)

app = modal.App("stable-audio-open-a2a-inference", image=image)

hf_secret = modal.Secret.from_name("huggingface-secret")


# ---------------------------------------------------------------------------
# Setup — verifica el checkpoint stable-audio-tools en el Volume
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, secrets=[hf_secret], timeout=1800)
def setup():
    """
    Verifica que model.safetensors + model_config.json (formato stable-audio-tools)
    están en el Volume compartido con el flujo text2audio; si faltan, los descarga.
        modal run research_sao_edit_modal.py::setup
    """
    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN")
    needed = ["model_config.json", "model.safetensors"]
    missing = [f for f in needed if not os.path.exists(f"{MODEL_LOCAL_DIR}/{f}")]

    if not missing:
        print(f"[setup] Checkpoint stable-audio-tools completo en {MODEL_LOCAL_DIR}. Nada que hacer.")
        return

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN no encontrado. Crear secret: "
            "modal secret create huggingface-secret HF_TOKEN=<tu_token>"
        )

    for fname in missing:
        print(f"[setup] Descargando {fname} …")
        hf_hub_download(
            repo_id=MODEL_HF_REPO,
            filename=fname,
            local_dir=MODEL_LOCAL_DIR,
            token=hf_token,
        )

    weights_vol.commit()
    print("[setup] Checkpoint completo y Volume commiteado.")


# ---------------------------------------------------------------------------
# Helpers de inferencia (dentro del container)
# ---------------------------------------------------------------------------

def _load_model():
    """Carga el modelo con el pipeline oficial stable-audio-tools desde el Volume."""
    import torch
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    t0 = time.time()
    with open(f"{MODEL_LOCAL_DIR}/model_config.json") as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(f"{MODEL_LOCAL_DIR}/model.safetensors"))
    model = model.to("cuda").eval().requires_grad_(False)
    print(f"[load_model] Modelo listo ({time.time()-t0:.1f}s)")
    return model, model_config


def _edit_one(
    model,
    model_config: dict,
    text: str,
    init_audio_bytes: bytes,
    init_noise_level: float,
    seconds: float | None,
    seed: int,
    steps: int,
) -> bytes:
    """
    Style transfer: genera con init_audio según las condiciones del model card.
    Devuelve bytes WAV estéreo 44.1 kHz recortado a `seconds` (duración del source
    si seconds es None).
    """
    import io

    import soundfile as sf
    import torch
    import torchaudio
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    # Cargar source → tensor (channels, samples) @ 44.1 kHz
    wav, sr = torchaudio.load(io.BytesIO(init_audio_bytes))
    if seconds is None:
        seconds = wav.shape[-1] / sr

    sample_rate = model_config.get("sample_rate", SAMPLE_RATE)
    sample_size = model_config["sample_size"]

    conditioning = [{
        "prompt": text,
        "seconds_start": 0,
        "seconds_total": seconds,
    }]

    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=7,  # model card
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,  # model card
        sigma_max=500,  # model card
        sampler_type="dpmpp-3m-sde",  # model card
        init_audio=(sr, wav),
        init_noise_level=init_noise_level,
        seed=seed,
        device="cuda",
    )

    # (batch, channels, samples) → (channels, samples), recorte a la duración del source
    audio = output[0].to(torch.float32)
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val
    n_samples = int(seconds * sample_rate)
    audio = audio[..., :n_samples].cpu().numpy().T

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Función Modal principal — batch de ediciones
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    gpu=DEFAULT_GPU,
    timeout=7200,
)
def generate_batch(jobs: list[dict], seed: int = DEFAULT_SEED, steps: int = DEFAULT_STEPS) -> list[bytes]:
    """
    Cada job: {"text", "init_audio_bytes", "init_noise_level", "seconds" (None = dur. source)}.
    Returns: list[bytes] WAV (b"" si falló).
    """
    model, model_config = _load_model()

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            wav_bytes = _edit_one(
                model,
                model_config,
                text=job["text"],
                init_audio_bytes=job["init_audio_bytes"],
                init_noise_level=float(job.get("init_noise_level", DEFAULT_NOISE_LEVEL)),
                seconds=job.get("seconds"),
                seed=seed + i,
                steps=steps,
            )
            print(
                f"[generate] [{i+1}/{len(jobs)}] OK — '{job['text'][:50]}' "
                f"(noise={job.get('init_noise_level', DEFAULT_NOISE_LEVEL)}) "
                f"→ {len(wav_bytes)//1024} KB, {time.time()-t0:.1f}s"
            )
            results.append(wav_bytes)
        except Exception as exc:
            print(f"[generate] [{i+1}/{len(jobs)}] ERROR: {exc}")
            results.append(b"")

    return results


# ---------------------------------------------------------------------------
# Helpers locales
# ---------------------------------------------------------------------------

def _load_prompts_edit(prompts_json: str) -> dict:
    prompts_path = Path(prompts_json).resolve()
    if not prompts_path.exists():
        print(f"ERROR: No existe {prompts_path}")
        raise SystemExit(1)
    with open(prompts_path) as f:
        return json.load(f)


def _read_source(eval_base: Path, sources: list[dict], source_id: str) -> tuple[bytes, dict]:
    src = next((s for s in sources if s["id"] == source_id), None)
    if src is None:
        raise ValueError(f"source_id desconocido: {source_id}")
    src_path = eval_base / src["file"]
    if not src_path.exists():
        raise FileNotFoundError(
            f"No existe {src_path}. Ejecuta antes: bash ../evaluation/fetch_edit_sources.sh"
        )
    return src_path.read_bytes(), src


# ---------------------------------------------------------------------------
# Entrypoint: edición libre
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    source_audio: str = "",
    prompt: str = "",
    noise_level: float = DEFAULT_NOISE_LEVEL,
    seconds: float = 0.0,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Style transfer de un audio fuente con un prompt.

    Ejemplo:
        modal run research_sao_edit_modal.py::main \\
            --source-audio ../evaluation/edit/source_audio/beatbox_loop_90bpm.wav \\
            --prompt "90 BPM tech house drum loop" \\
            --noise-level 1.0 \\
            --out-dir ../evaluation/edit/sao_a2a/manual
    """
    if not source_audio or not prompt:
        print("ERROR: --source-audio y --prompt son requeridos.")
        raise SystemExit(1)

    src_path = Path(source_audio).resolve()
    if not src_path.exists():
        print(f"ERROR: no existe {src_path}")
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_wav = out_p / "output.wav"
    if out_wav.exists() and not force:
        print(f"[main] Ya existe {out_wav}. Usa --force para regenerar.")
        return

    print(f"[main] Style transfer: '{prompt}' (init_noise_level={noise_level}) sobre {src_path.name}")
    [wav_bytes] = generate_batch.remote(
        [{
            "text": prompt,
            "init_audio_bytes": src_path.read_bytes(),
            "init_noise_level": noise_level,
            "seconds": seconds if seconds > 0 else None,
        }],
        seed=seed,
    )
    if not wav_bytes:
        print("[main] ERROR: la generación devolvió vacío.")
        raise SystemExit(1)
    out_p.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(wav_bytes)
    print(f"[main] Guardado: {out_wav} ({len(wav_bytes)//1024} KB)")


# ---------------------------------------------------------------------------
# Entrypoint: smoke test oficial
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def smoke(
    prompts_json: str = "../evaluation/prompts_edit.json",
    eval_base: str = "../evaluation",
    seed: int = DEFAULT_SEED,
    force: bool = False,
):
    """
    Smoke con el prompt del model card adaptado al BPM del source beatbox:
    "90 BPM tech house drum loop" sobre beatbox_loop_90bpm.wav (init_noise_level default).

    → evaluation/edit/sao_a2a/smoke/sao_a2a_smoke_01/output.wav
    Éxito: drum loop tech house que conserva el groove del beatbox de 90 BPM.
    """
    data = _load_prompts_edit(prompts_json)
    smoke_prompts = data["official_smoke"]["sao_a2a"]["prompts"]
    eval_base_path = Path(eval_base).resolve()
    smoke_dir = eval_base_path / "edit" / MODEL_DIR_NAME / "smoke"

    jobs: list[dict] = []
    out_paths: list[Path] = []
    for p in smoke_prompts:
        out_wav = smoke_dir / p["id"] / "output.wav"
        if out_wav.exists() and not force:
            print(f"[skip] {p['id']}: ya tiene output.wav. Usa --force para regenerar.")
            continue
        src_bytes, _src = _read_source(eval_base_path, data["sources"], p["source_id"])
        jobs.append({
            "text": p["target_prompt"],
            "init_audio_bytes": src_bytes,
            "init_noise_level": p.get("init_noise_level") or DEFAULT_NOISE_LEVEL,
            "seconds": p.get("seconds"),
        })
        out_paths.append(out_wav)
        print(f"[smoke] Encolado: {p['id']} — '{p['target_prompt']}'")

    if not jobs:
        print("[smoke] Nada pendiente. Usa --force para regenerar.")
        return

    wav_bytes_list = generate_batch.remote(jobs, seed=seed)
    ok = 0
    for wav_bytes, out_wav in zip(wav_bytes_list, out_paths):
        if not wav_bytes:
            print(f"[smoke] ERROR: {out_wav.parent.name} devolvió vacío")
            continue
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(wav_bytes)
        print(f"[smoke] Guardado: {out_wav} ({len(wav_bytes)//1024} KB)")
        ok += 1

    if ok == len(jobs):
        print("\n[smoke] Comparar con el source (groove intacto) y, si OK → eval_all:")
        print("            modal run research_sao_edit_modal.py::eval_all")


# ---------------------------------------------------------------------------
# Entrypoint: benchmark completo
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    prompts_json: str = "../evaluation/prompts_edit.json",
    eval_base: str = "../evaluation",
    seed: int = DEFAULT_SEED,
    force: bool = False,
    only: str = "",
):
    """
    Style transfer sobre todos los casos soportados de prompts_edit.json.
    Cada caso genera: evaluation/edit/sao_a2a/<case_id>/output.wav

    Ejemplos:
        modal run research_sao_edit_modal.py::eval_all
        modal run research_sao_edit_modal.py::eval_all --only case08_beatbox_drumkit
    """
    data = _load_prompts_edit(prompts_json)
    eval_base_path = Path(eval_base).resolve()
    model_path = eval_base_path / "edit" / MODEL_DIR_NAME

    cases = [c for c in data["cases"] if c["category"] in SUPPORTED_CATEGORIES]
    if only:
        only_ids = {s.strip() for s in only.split(",")}
        cases = [c for c in cases if c["id"] in only_ids]
    if not cases:
        print("ERROR: no hay casos tras aplicar los filtros")
        raise SystemExit(1)

    jobs: list[dict] = []
    out_paths: list[Path] = []
    for c in cases:
        out_wav = model_path / c["id"] / "output.wav"
        if out_wav.exists() and not force:
            print(f"[skip] {c['id']}: ya tiene output.wav. Usa --force para regenerar.")
            continue
        src_bytes, _src = _read_source(eval_base_path, data["sources"], c["source_id"])
        noise = STRENGTH_MAP.get(c.get("strength_hint") or "moderate", DEFAULT_NOISE_LEVEL)
        jobs.append({
            "text": c["target_prompt"],
            "init_audio_bytes": src_bytes,
            "init_noise_level": noise,
            "seconds": None,  # mantener duración del source
        })
        out_paths.append(out_wav)
        print(f"[eval_all] Encolado: {c['id']} ({c['category']}, noise={noise})")

    if not jobs:
        print("[eval_all] Todos los casos ya tienen output.wav. Usa --force para regenerar.")
        return

    print(f"\n[eval_all] Enviando {len(jobs)} ediciones a Modal ({DEFAULT_GPU}) …")
    t0 = time.time()
    wav_bytes_list = generate_batch.remote(jobs, seed=seed)
    print(f"[eval_all] Batch completado en {time.time()-t0:.0f}s")

    ok = 0
    for wav_bytes, out_wav in zip(wav_bytes_list, out_paths):
        if not wav_bytes:
            print(f"[eval_all] ERROR: {out_wav.parent.name} devolvió vacío — saltando")
            continue
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(wav_bytes)
        print(f"[eval_all] Guardado: {out_wav.parent.name}/output.wav ({len(wav_bytes)//1024} KB)")
        ok += 1

    print(f"\n[eval_all] {ok}/{len(jobs)} casos generados con éxito.")
    print("\nSiguiente paso:")
    print("  bash ../evaluation/render_norm.sh")
    print("  uv run python ../evaluation/compute_metrics_edit.py --only sao_a2a")
