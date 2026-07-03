"""
Modal.com inference app para MelodyFlow (facebook/melodyflow-t24-30secs) — subtarea EDICIÓN.

MelodyFlow (Meta) es un flow-matching DiT sobre latentes de un EnCodec sin cuantizar,
diseñado específicamente para EDICIÓN de música guiada por texto mediante inversión
latente regularizada (adaptación de ReNoise a flow matching). Edita audios reales sin
fine-tuning, con alta consistencia estructural.

⚠️  CANDIDATO NUEVO — no estaba en la tabla original de RESEARCH.md. Se incorpora porque
es el modelo abierto de referencia (Meta, 2024) para text-guided music editing.

⚠️  El código NO está en el repo principal de audiocraft: vive en el fork bundled del
Space oficial (https://huggingface.co/spaces/facebook/MelodyFlow). La imagen instala
audiocraft desde ese Space (es un repo git pip-instalable).

Punto de entrada DAW:
  - Input: audio fuente + descripción del resultado deseado.
  - Output: WAV estéreo 48 kHz, misma duración que el source (≤30 s).
  - Ideal para: edición fiel ("mismo tema, otro estilo") con mínima pérdida de calidad.
  - Licencia: MIT (código) / CC-BY-NC (pesos) — solo investigación hasta decidir integración.

Condiciones oficiales replicadas (verificado 2026-07-02):
  Paper: Le Lan et al., "High Fidelity Text-Guided Music Editing via Single-Stage Flow
  Matching" — https://arxiv.org/abs/2407.03648
  Demo oficial (demos/melodyflow_app.py del Space):
  - model = MelodyFlow.get_pretrained("facebook/melodyflow-t24-30secs")  (1B, 30 s)
  - Ejemplo de edición VERBATIM: assets/bolero_ravel.mp3 + "A cheerful country song with
    acoustic guitars", solver=euler, steps=125, target_flowstep=0.0, regularize=True,
    lambda_kl=0.2.
  - Preproceso: truncar el source a model.duration, convert_audio a 48 kHz estéreo,
    encode_audio → prompt_tokens; edit(prompt_tokens, descriptions, src_descriptions=[""]).

Mapeo strength_hint (prompts_edit.json) → target_flowstep (menor = edición más fuerte):
  subtle → 0.10
  moderate → 0.05
  strong → 0.0   (valor del ejemplo oficial)

Setup (descarga pesos al Volume, una vez):
    modal run research_melodyflow_modal.py::setup

Smoke test oficial:
    modal run research_melodyflow_modal.py::smoke
    # bolero_ravel.mp3 + "A cheerful country song with acoustic guitars" (par VERBATIM)
    # → evaluation/edit/melodyflow/smoke/mf_smoke_01/output.wav
    # Éxito: country reconocible que preserva la estructura del bolero.
    # Referencia manual: https://huggingface.co/spaces/facebook/MelodyFlow

Edición libre:
    modal run research_melodyflow_modal.py::main \\
        --source-audio ../evaluation/edit/source_audio/bolero_ravel.mp3 \\
        --prompt "A cheerful country song with acoustic guitars" \\
        --flowstep 0.0 \\
        --out-dir ../evaluation/edit/melodyflow/manual

Benchmark completo:
    modal run research_melodyflow_modal.py::eval_all

Coste estimado (A10G): setup (descarga ~4 GB) ~$0.05 · 1 edición (inversión + 125 steps)
~$0.02-0.05 · eval_all (10 casos) ~$0.30-0.50
"""

import json
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("melodyflow-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

MODEL_ID = os.environ.get("MELODYFLOW_MODEL", "facebook/melodyflow-t24-30secs")
DEFAULT_GPU = os.environ.get("MELODYFLOW_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("MELODYFLOW_SEED", "42"))

# Parámetros del ejemplo de edición oficial (demos/melodyflow_app.py)
DEFAULT_SOLVER = "euler"
DEFAULT_STEPS = 125
DEFAULT_FLOWSTEP = 0.0
DEFAULT_LAMBDA_KL = 0.2

MODEL_DIR_NAME = "melodyflow"  # evaluation/edit/<model>/
SUPPORTED_CATEGORIES = {"style_transfer", "instrumentation", "mood_texture"}
STRENGTH_MAP = {"subtle": 0.10, "moderate": 0.05, "strong": 0.0}

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# audiocraft con MelodyFlow instalado desde el Space oficial (repo git de HF).
image = (
    modal.Image.debian_slim(python_version="3.10")
    # PyAV (dep de audiocraft/encodec) requiere libav headers y pkg-config.
    # GIT_LFS_SKIP_SMUDGE=1 evita que pip falle al clonar el Space HF que tiene archivos LFS.
    .apt_install([
        "git", "ffmpeg", "libsndfile1",
        "pkg-config",
        "libavformat-dev", "libavdevice-dev", "libavcodec-dev",
        "libavutil-dev", "libswresample-dev", "libavfilter-dev",
    ])
    # pip install git+<hf_space_url> usa internamente git clone --filter=blob:none que
    # el servidor git de HF Spaces no soporta (exit 128). Solución: clonar primero
    # manualmente (sin --filter) e instalar desde la ruta local.
    # GIT_LFS_SKIP_SMUDGE=1 evita la descarga de archivos LFS del Space.
    .run_commands(
        "GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/facebook/MelodyFlow /root/MelodyFlow && "
        "pip install /root/MelodyFlow"
    )
    .pip_install("soundfile>=0.12.1")
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache",
        "TORCH_HOME": f"{WEIGHTS_MOUNT}/torch-cache",
    })
)

app = modal.App("melodyflow-edit-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume
# ---------------------------------------------------------------------------
def _patch_torch_load_weights_only():
    """
    Torch ≥ 2.6 usa weights_only=True por defecto en torch.load, lo que rechaza los
    tipos omegaconf serializados en los checkpoints de audiocraft/MelodyFlow.
    Monkey-patch a weights_only=False (seguro porque los checkpoints son de Meta/HF).
    Más robusto que enumerar todos los tipos omegaconf necesarios.
    """
    import torch
    _orig = torch.load
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)
    torch.load = _load


@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=3600)
def setup():
    """
    Descarga los pesos de MelodyFlow al Volume (una vez):
        modal run research_melodyflow_modal.py::setup
    Pesos públicos en HF — sin token.
    """
    _patch_torch_load_weights_only()
    from audiocraft.models import MelodyFlow

    t0 = time.time()
    MelodyFlow.get_pretrained(MODEL_ID)
    print(f"[setup] {MODEL_ID} descargado y cargado en {time.time()-t0:.0f}s")
    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Función Modal principal — batch de ediciones
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=7200)
def generate_batch(jobs: list[dict], seed: int = DEFAULT_SEED) -> list[bytes]:
    """
    Cada job: {"text", "source_bytes", "target_flowstep", "solver", "steps", "lambda_kl"}.
    Flujo de edición del demo oficial: encode_audio → edit con inversión regularizada.
    Returns: list[bytes] WAV estéreo 48 kHz (b"" si falló).
    """
    import io

    import soundfile as sf
    import torch
    import torchaudio
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.models import MelodyFlow

    _patch_torch_load_weights_only()
    t0 = time.time()
    model = MelodyFlow.get_pretrained(MODEL_ID)
    weights_vol.commit()  # persistir pesos si el setup no se ejecutó antes
    print(f"[load_model] {MODEL_ID} listo ({time.time()-t0:.1f}s)")

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            torch.manual_seed(seed + i)

            # Preproceso según demos/melodyflow_app.py
            melody, sr = torchaudio.load(io.BytesIO(job["source_bytes"]))
            if melody.dim() == 2:
                melody = melody[None]
            if melody.shape[-1] > int(sr * model.duration):
                melody = melody[..., : int(sr * model.duration)]
            melody = convert_audio(melody, sr, model.sample_rate, model.audio_channels)
            prompt_tokens = model.encode_audio(melody.to(model.device))

            model.set_editing_params(
                solver=job.get("solver", DEFAULT_SOLVER),
                steps=int(job.get("steps", DEFAULT_STEPS)),
                target_flowstep=float(job.get("target_flowstep", DEFAULT_FLOWSTEP)),
                regularize=True,
                lambda_kl=float(job.get("lambda_kl", DEFAULT_LAMBDA_KL)),
            )

            out = model.edit(
                prompt_tokens=prompt_tokens,
                descriptions=[job["text"]],
                src_descriptions=[""],
                return_tokens=False,
            )
            audio = out[0].to(torch.float32).cpu()  # (channels, samples)
            max_val = audio.abs().max()
            if max_val > 1.0:
                audio = audio / max_val

            buf = io.BytesIO()
            sf.write(buf, audio.numpy().T, samplerate=model.sample_rate, format="WAV", subtype="PCM_16")
            buf.seek(0)
            wav_bytes = buf.read()
            print(
                f"[generate] [{i+1}/{len(jobs)}] OK — '{job['text'][:50]}' "
                f"(flowstep={job.get('target_flowstep', DEFAULT_FLOWSTEP)}) "
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


def _run_jobs_and_save(jobs: list[dict], out_paths: list[Path], seed: int, label: str) -> int:
    print(f"\n[{label}] Enviando {len(jobs)} ediciones a Modal ({DEFAULT_GPU}) …")
    t0 = time.time()
    wav_bytes_list = generate_batch.remote(jobs, seed=seed)
    print(f"[{label}] Batch completado en {time.time()-t0:.0f}s")

    ok = 0
    for wav_bytes, out_wav in zip(wav_bytes_list, out_paths):
        if not wav_bytes:
            print(f"[{label}] ERROR: {out_wav.parent.name} devolvió vacío — saltando")
            continue
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(wav_bytes)
        print(f"[{label}] Guardado: {out_wav.parent.name}/output.wav ({len(wav_bytes)//1024} KB)")
        ok += 1
    print(f"\n[{label}] {ok}/{len(jobs)} generados con éxito.")
    return ok


# ---------------------------------------------------------------------------
# Entrypoint: edición libre
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    source_audio: str = "",
    prompt: str = "",
    flowstep: float = DEFAULT_FLOWSTEP,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Edición de un audio fuente con un prompt.

    Ejemplo:
        modal run research_melodyflow_modal.py::main \\
            --source-audio ../evaluation/edit/source_audio/bolero_ravel.mp3 \\
            --prompt "A cheerful country song with acoustic guitars" \\
            --flowstep 0.0 \\
            --out-dir ../evaluation/edit/melodyflow/manual
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

    print(f"[main] Edición: '{prompt}' (flowstep={flowstep}) sobre {src_path.name}")
    [wav_bytes] = generate_batch.remote(
        [{
            "text": prompt,
            "source_bytes": src_path.read_bytes(),
            "target_flowstep": flowstep,
            "steps": steps,
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
    Replica el ejemplo de edición VERBATIM del demo oficial:
    bolero_ravel.mp3 + "A cheerful country song with acoustic guitars"
    (euler, 125 steps, target_flowstep 0.0, regularize, lambda_kl 0.2).

    → evaluation/edit/melodyflow/smoke/mf_smoke_01/output.wav
    Éxito: country reconocible que preserva la estructura del bolero.
    Comparar con el Space oficial: https://huggingface.co/spaces/facebook/MelodyFlow
    """
    data = _load_prompts_edit(prompts_json)
    smoke_prompts = data["official_smoke"]["melodyflow"]["prompts"]
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
            "source_bytes": src_bytes,
            "target_flowstep": p.get("target_flowstep", DEFAULT_FLOWSTEP),
            "solver": p.get("solver", DEFAULT_SOLVER),
            "steps": p.get("steps", DEFAULT_STEPS),
            "lambda_kl": p.get("lambda_kl", DEFAULT_LAMBDA_KL),
        })
        out_paths.append(out_wav)
        print(f"[smoke] Encolado: {p['id']} — '{p['target_prompt'][:60]}'")

    if not jobs:
        print("[smoke] Nada pendiente. Usa --force para regenerar.")
        return

    ok = _run_jobs_and_save(jobs, out_paths, seed, "smoke")
    if ok == len(jobs):
        print("\nSi la edición es comparable al Space oficial → eval_all:")
        print("    modal run research_melodyflow_modal.py::eval_all")


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
    Edita todos los casos soportados de prompts_edit.json.
    Cada caso genera: evaluation/edit/melodyflow/<case_id>/output.wav

    Ejemplos:
        modal run research_melodyflow_modal.py::eval_all
        modal run research_melodyflow_modal.py::eval_all --only case02_bolero_country
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
        flowstep = STRENGTH_MAP.get(c.get("strength_hint") or "moderate", DEFAULT_FLOWSTEP)
        jobs.append({
            "text": c["target_prompt"],
            "source_bytes": src_bytes,
            "target_flowstep": flowstep,
        })
        out_paths.append(out_wav)
        print(f"[eval_all] Encolado: {c['id']} ({c['category']}, flowstep={flowstep})")

    if not jobs:
        print("[eval_all] Todos los casos ya tienen output.wav. Usa --force para regenerar.")
        return

    _run_jobs_and_save(jobs, out_paths, seed, "eval_all")
    print("\nSiguiente paso:")
    print("  bash ../evaluation/render_norm.sh")
    print("  uv run python ../evaluation/compute_metrics_edit.py --only melodyflow")
