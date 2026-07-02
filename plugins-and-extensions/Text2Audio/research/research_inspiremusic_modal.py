"""
Modal.com inference app para InspireMusic (Alibaba FunAudioLLM) — subtarea CONTINUACIÓN.

InspireMusic genera música instrumental y soporta CONTINUACIÓN guiada por texto: recibe
un audio prompt + una instrucción y genera cómo sigue el tema. No edita el source (a
diferencia de cover/style transfer): lo continúa — útil como ASISTENTE DE COMPOSICIÓN
("¿cómo seguiría este loop?").

Punto de entrada DAW:
  - Input: audio prompt (se usan los primeros 5 s por defecto) + texto opcional.
  - Output: WAV 48 kHz (modelo *-Long), hasta 30 s por defecto en este script.
  - Ideal para: continuar un sample/loop de REAPER en un estilo indicado.
  - Licencia: Apache 2.0 (código y pesos) — apto para producción.

Condiciones oficiales replicadas (verificado 2026-07-02):
  https://github.com/FunAudioLLM/InspireMusic  (README quickstart):
    model = InspireMusicModel(model_name="InspireMusic-1.5B-Long")
    model.inference("continuation", "Continue to generate jazz music.", "audio_prompt.wav")
  - Checkpoint: FunAudioLLM/InspireMusic-1.5B-Long (48 kHz, hasta varios minutos).
  - max_audio_prompt_length=5.0 s (default oficial — solo los primeros 5 s del source
    condicionan la continuación).
  - El repo NO incluye un audio de ejemplo real (audio_prompt.wav del README es
    placeholder); el smoke usa nuestro loop de guitarra jazz como audio prompt.

Nota: el output de InspireMusic en continuation INCLUYE el audio prompt al inicio
seguido de la continuación. compute_metrics_edit.py recorta la duración del prompt
antes de medir el segmento continuado.

Setup (descarga pesos al Volume, una vez, ~7 GB):
    modal run research_inspiremusic_modal.py::setup

Smoke test oficial:
    modal run research_inspiremusic_modal.py::smoke
    # sao_guitar_loop.wav + "Continue to generate jazz music." (frase VERBATIM del README)
    # → evaluation/edit/inspiremusic/smoke/im_smoke_01/output.wav
    # Éxito: la continuación arranca coherente (tonalidad/tempo) y sigue en jazz.

Continuación libre:
    modal run research_inspiremusic_modal.py::main \\
        --source-audio ../evaluation/edit/source_audio/electronic.mp3 \\
        --prompt "Continue to generate energetic electronic dance music." \\
        --out-dir ../evaluation/edit/inspiremusic/manual

Benchmark (solo casos category == "continuation" de prompts_edit.json):
    modal run research_inspiremusic_modal.py::eval_all

Coste estimado (A10G): setup (descarga ~7 GB) ~$0.08 · 1 continuación de 30 s ~$0.03-0.06
· eval_all (2 casos) ~$0.10
"""

import json
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("inspiremusic-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

MODEL_NAME = os.environ.get("INSPIREMUSIC_MODEL", "InspireMusic-1.5B-Long")
MODEL_HF_REPO = f"FunAudioLLM/{MODEL_NAME}"
MODEL_LOCAL_DIR = f"{WEIGHTS_MOUNT}/{MODEL_NAME}"

REPO_URL = "https://github.com/FunAudioLLM/InspireMusic.git"
REPO_DIR = "/root/InspireMusic"

DEFAULT_GPU = os.environ.get("INSPIREMUSIC_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("INSPIREMUSIC_SEED", "42"))
MAX_GENERATE_SECONDS = float(os.environ.get("INSPIREMUSIC_MAX_SECONDS", "30.0"))

MODEL_DIR_NAME = "inspiremusic"  # evaluation/edit/<model>/
SUPPORTED_CATEGORIES = {"continuation"}

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# El repo se usa como paquete vía PYTHONPATH (patrón CosyVoice/FunAudioLLM: clone
# --recursive para el submódulo third_party/Matcha-TTS + requirements.txt).
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "libsndfile1", "sox", "libsox-dev"])
    .run_commands(f"git clone --recursive {REPO_URL} {REPO_DIR}")
    .run_commands(f"pip install -r {REPO_DIR}/requirements.txt")
    .pip_install("soundfile>=0.12.1", "huggingface_hub>=0.24.0")
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache",
        "PYTHONPATH": f"{REPO_DIR}:{REPO_DIR}/third_party/Matcha-TTS",
    })
)

app = modal.App("inspiremusic-continuation-inference", image=image)


# ---------------------------------------------------------------------------
# Setup — descarga pesos al Volume
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, timeout=3600)
def setup():
    """
    Descarga los pesos de InspireMusic al Volume (una vez, ~7 GB):
        modal run research_inspiremusic_modal.py::setup
    Pesos públicos en HF (Apache 2.0) — sin token.
    """
    from huggingface_hub import snapshot_download

    if os.path.exists(f"{MODEL_LOCAL_DIR}/llm.pt"):
        print(f"[setup] Modelo ya descargado en {MODEL_LOCAL_DIR}. Nada que hacer.")
        return

    print(f"[setup] Descargando {MODEL_HF_REPO} → {MODEL_LOCAL_DIR} …")
    t0 = time.time()
    snapshot_download(repo_id=MODEL_HF_REPO, local_dir=MODEL_LOCAL_DIR)
    print(f"[setup] Descarga completa en {time.time()-t0:.0f}s")
    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Función Modal principal — batch de continuaciones
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=7200)
def generate_batch(jobs: list[dict], seed: int = DEFAULT_SEED) -> list[bytes]:
    """
    Cada job: {"text" (opcional), "source_bytes", "source_name"}.
    API oficial: model.inference("continuation", text, audio_prompt_path).
    Returns: list[bytes] WAV 48 kHz (b"" si falló). El output incluye el audio
    prompt al inicio, seguido de la continuación.
    """
    import shutil
    import tempfile

    import torch

    from inspiremusic.cli.inference import InspireMusicModel, env_variables

    env_variables()
    t0 = time.time()
    result_dir = tempfile.mkdtemp(prefix="inspiremusic_out_")
    model = InspireMusicModel(
        model_name=MODEL_NAME,
        model_dir=MODEL_LOCAL_DIR,
        max_generate_audio_seconds=MAX_GENERATE_SECONDS,
        output_sample_rate=48000,
        gpu=0,
        result_dir=result_dir,
        hub="huggingface",
    )
    weights_vol.commit()  # persistir pesos si el setup no se ejecutó antes
    print(f"[load_model] {MODEL_NAME} listo ({time.time()-t0:.1f}s)")

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        workdir = tempfile.mkdtemp(prefix="inspiremusic_in_")
        try:
            torch.manual_seed(seed + i)
            suffix = Path(job.get("source_name", "src.wav")).suffix or ".wav"
            src_path = os.path.join(workdir, f"audio_prompt{suffix}")
            with open(src_path, "wb") as f:
                f.write(job["source_bytes"])

            output_fn = f"output_{i:02d}"
            model.inference(
                task="continuation",
                text=job.get("text") or None,
                audio_prompt=src_path,
                output_fn=output_fn,
                output_format="wav",
            )

            out_path = Path(result_dir) / f"{output_fn}.wav"
            if not out_path.exists():
                candidates = sorted(Path(result_dir).glob("*.wav"), key=lambda p: p.stat().st_mtime)
                if not candidates:
                    raise RuntimeError(f"sin output en {result_dir}")
                out_path = candidates[-1]

            wav_bytes = out_path.read_bytes()
            print(
                f"[generate] [{i+1}/{len(jobs)}] OK — '{(job.get('text') or '(sin texto)')[:50]}' "
                f"→ {len(wav_bytes)//1024} KB, {time.time()-t0:.1f}s"
            )
            results.append(wav_bytes)
        except Exception as exc:
            print(f"[generate] [{i+1}/{len(jobs)}] ERROR: {exc}")
            results.append(b"")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

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
    print(f"\n[{label}] Enviando {len(jobs)} continuaciones a Modal ({DEFAULT_GPU}) …")
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
# Entrypoint: continuación libre
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    source_audio: str = "",
    prompt: str = "",
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Continúa un audio prompt (texto opcional).

    Ejemplo:
        modal run research_inspiremusic_modal.py::main \\
            --source-audio ../evaluation/edit/source_audio/sao_guitar_loop.wav \\
            --prompt "Continue to generate jazz music." \\
            --out-dir ../evaluation/edit/inspiremusic/manual
    """
    if not source_audio:
        print("ERROR: --source-audio es requerido (--prompt es opcional).")
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

    print(f"[main] Continuación: '{prompt or '(sin texto)'}' sobre {src_path.name}")
    [wav_bytes] = generate_batch.remote(
        [{"text": prompt, "source_bytes": src_path.read_bytes(), "source_name": src_path.name}],
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
    Smoke con la frase VERBATIM del README: "Continue to generate jazz music."
    sobre nuestro loop de guitarra jazz (el repo no incluye audio de ejemplo real).

    → evaluation/edit/inspiremusic/smoke/im_smoke_01/output.wav
    Éxito: la continuación arranca coherente con el loop (tonalidad/tempo) y
    sigue en estilo jazz, sin silencio ni ruido.
    """
    data = _load_prompts_edit(prompts_json)
    smoke_prompts = data["official_smoke"]["inspiremusic"]["prompts"]
    eval_base_path = Path(eval_base).resolve()
    smoke_dir = eval_base_path / "edit" / MODEL_DIR_NAME / "smoke"

    jobs: list[dict] = []
    out_paths: list[Path] = []
    for p in smoke_prompts:
        out_wav = smoke_dir / p["id"] / "output.wav"
        if out_wav.exists() and not force:
            print(f"[skip] {p['id']}: ya tiene output.wav. Usa --force para regenerar.")
            continue
        src_bytes, src = _read_source(eval_base_path, data["sources"], p["source_id"])
        jobs.append({
            "text": p["target_prompt"],
            "source_bytes": src_bytes,
            "source_name": Path(src["file"]).name,
        })
        out_paths.append(out_wav)
        print(f"[smoke] Encolado: {p['id']} — '{p['target_prompt']}'")

    if not jobs:
        print("[smoke] Nada pendiente. Usa --force para regenerar.")
        return

    ok = _run_jobs_and_save(jobs, out_paths, seed, "smoke")
    if ok == len(jobs):
        print("\nSi la continuación es coherente → eval_all:")
        print("    modal run research_inspiremusic_modal.py::eval_all")


# ---------------------------------------------------------------------------
# Entrypoint: benchmark (casos continuation)
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
    Ejecuta los casos category == "continuation" de prompts_edit.json.
    Cada caso genera: evaluation/edit/inspiremusic/<case_id>/output.wav

    Ejemplos:
        modal run research_inspiremusic_modal.py::eval_all
        modal run research_inspiremusic_modal.py::eval_all --only case11_guitar_continue_jazz
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
            "source_bytes": src_bytes,
            "source_name": Path(src["file"]).name,
        })
        out_paths.append(out_wav)
        print(f"[eval_all] Encolado: {c['id']} ({c['category']})")

    if not jobs:
        print("[eval_all] Todos los casos ya tienen output.wav. Usa --force para regenerar.")
        return

    _run_jobs_and_save(jobs, out_paths, seed, "eval_all")
    print("\nSiguiente paso:")
    print("  bash ../evaluation/render_norm.sh")
    print("  uv run python ../evaluation/compute_metrics_edit.py --only inspiremusic")
