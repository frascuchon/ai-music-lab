"""
Modal.com inference app para MusicGen-melody (facebook/musicgen-melody) — subtarea VERSIONADO.

MusicGen-melody (Meta AudioCraft) genera música condicionada por TEXTO + MELODÍA de un audio
de referencia (proyección del cromagrama). No "edita" el source: crea una VERSIÓN nueva que
sigue el contorno melódico del audio de referencia en el estilo descrito por el texto —
exactamente el flujo "versión de este tema en otro estilo" del plugin.

Punto de entrada DAW:
  - Input: audio de referencia (motivo/melodía de REAPER) + descripción de estilo.
  - Output: WAV mono 32 kHz, hasta 30 s.
  - Ideal para: "este riff, pero como pop de los 80" / "esta melodía en jazz".
  - Licencia: MIT (código) / CC-BY-NC (pesos) — solo investigación hasta decidir integración.

Condiciones oficiales replicadas (verificado 2026-07-02):
  https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md
  https://huggingface.co/facebook/musicgen-melody
  - API oficial: MusicGen.get_pretrained("facebook/musicgen-melody") +
    set_generation_params(duration=…, top_k=250) + generate_with_chroma(...).
  - Audio de referencia del demo oficial: assets/bach.mp3 (el mismo que descarga
    fetch_edit_sources.sh como src01_bach).
  - Métricas del paper (Copet et al., NeurIPS 2023, MusicCaps): FAD 4.09, CLAP 0.31
    para musicgen-melody.

Nota strength_hint: MusicGen-melody NO tiene knob de intensidad — el condicionamiento por
cromagrama es fijo. El campo strength_hint de prompts_edit.json se ignora.

Setup (descarga pesos al Volume, una vez, ~11 GB melody):
    modal run research_musicgen_melody_modal.py::setup

Smoke test oficial:
    modal run research_musicgen_melody_modal.py::smoke
    # bach.mp3 + "An 80s driving pop song with heavy drums and synth pads in the background"
    # (combinación del demo oficial de MusicGen-melody)
    # → evaluation/edit/musicgen_melody/smoke/mgm_smoke_01/output.wav
    # Éxito: pop 80s cuyo contorno melódico sigue al bach.mp3. CLAP ≥ 0.25.

Versión libre:
    modal run research_musicgen_melody_modal.py::main \\
        --source-audio ../evaluation/edit/source_audio/sao_guitar_loop.wav \\
        --prompt "lo-fi hip hop with mellow keys" \\
        --seconds 10.0 \\
        --out-dir ../evaluation/edit/musicgen_melody/manual

Benchmark completo:
    modal run research_musicgen_melody_modal.py::eval_all

Coste estimado (A10G): setup (descarga ~11 GB) ~$0.10 · 1 versión de 10 s ~$0.02-0.04
(autoregresivo, ~50 tokens/s) · eval_all (10 casos) ~$0.25-0.40
"""

import json
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("musicgen-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

# Variante oficial de la tabla RESEARCH.md (mono 32 kHz). Alternativa estéreo:
# facebook/musicgen-stereo-melody (mismo pipeline, flag --variant).
DEFAULT_VARIANT = os.environ.get("MUSICGEN_VARIANT", "facebook/musicgen-melody")

DEFAULT_GPU = os.environ.get("MUSICGEN_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("MUSICGEN_SEED", "42"))

MODEL_DIR_NAME = "musicgen_melody"  # evaluation/edit/<model>/
SUPPORTED_CATEGORIES = {"style_transfer", "instrumentation", "mood_texture"}

MAX_SECONDS = 30.0  # límite del modelo

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    # PyAV (dep de audiocraft/encodec) requiere libav headers y pkg-config para compilar.
    .apt_install([
        "git", "ffmpeg", "libsndfile1",
        "pkg-config",
        "libavformat-dev", "libavdevice-dev", "libavcodec-dev",
        "libavutil-dev", "libswresample-dev", "libavfilter-dev",
    ])
    .pip_install("audiocraft>=1.3.0", "soundfile>=0.12.1")
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache",
        "TORCH_HOME": f"{WEIGHTS_MOUNT}/torch-cache",
    })
)

app = modal.App("musicgen-melody-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=3600)
def setup(variant: str = DEFAULT_VARIANT):
    """
    Descarga los pesos de MusicGen-melody al Volume (una vez, ~11 GB):
        modal run research_musicgen_melody_modal.py::setup
    Pesos públicos en HF — sin token.
    """
    from audiocraft.models import MusicGen

    t0 = time.time()
    MusicGen.get_pretrained(variant)
    print(f"[setup] {variant} descargado y cargado en {time.time()-t0:.0f}s")
    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Función Modal principal — batch de versiones
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=7200)
def generate_batch(
    jobs: list[dict],
    variant: str = DEFAULT_VARIANT,
    seed: int = DEFAULT_SEED,
) -> list[bytes]:
    """
    Cada job: {"text", "melody_bytes", "seconds"}.
    Params oficiales del README: top_k=250 (resto por defecto).
    Returns: list[bytes] WAV mono 32 kHz (b"" si falló).
    """
    import io

    import soundfile as sf
    import torch
    import torchaudio
    from audiocraft.models import MusicGen

    t0 = time.time()
    model = MusicGen.get_pretrained(variant)
    weights_vol.commit()  # persistir pesos si el setup no se ejecutó antes
    print(f"[load_model] {variant} listo ({time.time()-t0:.1f}s)")

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            seconds = min(float(job.get("seconds") or 10.0), MAX_SECONDS)
            model.set_generation_params(duration=seconds, top_k=250)
            torch.manual_seed(seed + i)

            melody, sr = torchaudio.load(io.BytesIO(job["melody_bytes"]))
            wav = model.generate_with_chroma(
                descriptions=[job["text"]],
                melody_wavs=[melody],
                melody_sample_rate=sr,
            )
            audio = wav[0].to(torch.float32).cpu()  # (channels, samples)
            max_val = audio.abs().max()
            if max_val > 1.0:
                audio = audio / max_val

            buf = io.BytesIO()
            sf.write(buf, audio.numpy().T, samplerate=model.sample_rate, format="WAV", subtype="PCM_16")
            buf.seek(0)
            wav_bytes = buf.read()
            print(
                f"[generate] [{i+1}/{len(jobs)}] OK — '{job['text'][:50]}' "
                f"({seconds}s) → {len(wav_bytes)//1024} KB, {time.time()-t0:.1f}s"
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


def _run_jobs_and_save(jobs: list[dict], out_paths: list[Path], seed: int, label: str):
    print(f"\n[{label}] Enviando {len(jobs)} trabajos a Modal ({DEFAULT_GPU}) …")
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
# Entrypoint: versión libre
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    source_audio: str = "",
    prompt: str = "",
    seconds: float = 10.0,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Genera una versión del audio de referencia en el estilo del prompt.

    Ejemplo:
        modal run research_musicgen_melody_modal.py::main \\
            --source-audio ../evaluation/edit/source_audio/bach.mp3 \\
            --prompt "An 80s driving pop song with heavy drums" \\
            --seconds 10.0 \\
            --out-dir ../evaluation/edit/musicgen_melody/manual
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

    print(f"[main] Versión: '{prompt}' ({seconds}s) con melodía de {src_path.name}")
    [wav_bytes] = generate_batch.remote(
        [{"text": prompt, "melody_bytes": src_path.read_bytes(), "seconds": seconds}],
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
    Replica el demo oficial de MusicGen-melody: bach.mp3 + descripción del model card.

    → evaluation/edit/musicgen_melody/smoke/mgm_smoke_01/output.wav
    Éxito: pop 80s cuyo contorno melódico sigue al bach.mp3 (comparar croma en
    compute_metrics_edit.py). Referencia manual: https://huggingface.co/spaces/facebook/MusicGen
    """
    data = _load_prompts_edit(prompts_json)
    smoke_prompts = data["official_smoke"]["musicgen_melody"]["prompts"]
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
        jobs.append({"text": p["target_prompt"], "melody_bytes": src_bytes, "seconds": p.get("seconds", 10.0)})
        out_paths.append(out_wav)
        print(f"[smoke] Encolado: {p['id']} — '{p['target_prompt'][:60]}'")

    if not jobs:
        print("[smoke] Nada pendiente. Usa --force para regenerar.")
        return

    ok = _run_jobs_and_save(jobs, out_paths, seed, "smoke")
    if ok == len(jobs):
        print("\nSi el resultado sigue la melodía del source → eval_all:")
        print("    modal run research_musicgen_melody_modal.py::eval_all")


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
    Genera versiones para todos los casos soportados de prompts_edit.json.
    Cada caso genera: evaluation/edit/musicgen_melody/<case_id>/output.wav
    La duración de cada versión = duración del source (limitada a 30 s).

    Ejemplos:
        modal run research_musicgen_melody_modal.py::eval_all
        modal run research_musicgen_melody_modal.py::eval_all --only case01_bach_jazz
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
        src_bytes, src = _read_source(eval_base_path, data["sources"], c["source_id"])
        jobs.append({
            "text": c["target_prompt"],
            "melody_bytes": src_bytes,
            "seconds": src.get("duration_s", 10.0),
        })
        out_paths.append(out_wav)
        print(f"[eval_all] Encolado: {c['id']} ({c['category']})")

    if not jobs:
        print("[eval_all] Todos los casos ya tienen output.wav. Usa --force para regenerar.")
        return

    _run_jobs_and_save(jobs, out_paths, seed, "eval_all")
    print("\nSiguiente paso:")
    print("  bash ../evaluation/render_norm.sh")
    print("  uv run python ../evaluation/compute_metrics_edit.py --only musicgen_melody")
