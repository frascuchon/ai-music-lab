"""
Modal.com inference app para ZETA (zero-shot text-based audio editing) — subtarea EDICIÓN.

ZETA (Manor & Michaeli, ICML 2024) edita audio real SIN fine-tuning aplicando inversión
DDPM edit-friendly sobre un modelo de difusión text-to-audio preentrenado. Es la técnica
que cubre la fila AudioLDM2 de la tabla de RESEARCH.md para esta subtarea, y además
soporta Stable Audio Open 1.0 como backbone (añadido oficialmente en octubre 2024).

Punto de entrada DAW:
  - Input: audio fuente + prompt del contenido original (opcional) + prompt objetivo.
  - Output: WAV (16 kHz mono con AudioLDM2; 44.1 kHz estéreo con SAO).
  - Ideal para: ediciones semánticas zero-shot ("hard rock → jazz") con backbone abierto.
  - Licencia: código MIT; pesos AudioLDM2 CC-BY-SA 4.0 (SAO: Stability Community).

Condiciones oficiales replicadas (verificado 2026-07-02):
  https://github.com/HilaManor/AudioEditingCode  (defaults de code/main_run.py — los del
  user study del paper):
  - model_id default: cvssp/audioldm2-music
  - num_diffusion_steps=200, tstart=100, cfg_src=3, cfg_tar=12, mode=ours
  - Para SAO el repo recomienda cfg_src=1 (añadido en el CHANGELOG).
  - El repo NO incluye audios (los del paper vienen de MedleyDB, descarga aparte);
    los prompts siguen el patrón de la página oficial de ejemplos
    (https://hilamanor.github.io/AudioEditing/): "A recording of X" → "A recording of Y".

Mapeo strength_hint (prompts_edit.json) → tstart (sobre 200 steps; mayor = más cambio):
  subtle → 70
  moderate → 100 (default del user study)
  strong → 130

Setup (descarga pesos del backbone al Volume, una vez):
    modal run research_zeta_edit_modal.py::setup

Smoke test oficial:
    modal run research_zeta_edit_modal.py::smoke
    # electronic.mp3: "A recording of an electronic music track…" →
    #                 "A recording of a jazz song with piano and double bass"
    # (patrón de la página oficial, defaults del user study)
    # → evaluation/edit/zeta_audioldm2/smoke/zeta_smoke_01/output.wav
    # Éxito: jazz que conserva la estructura rítmica del original.

Edición libre:
    modal run research_zeta_edit_modal.py::main \\
        --source-audio ../evaluation/edit/source_audio/electronic.mp3 \\
        --source-prompt "A recording of an electronic music track" \\
        --prompt "A recording of a jazz song with piano and double bass" \\
        --out-dir ../evaluation/edit/zeta_audioldm2/manual

Benchmark completo:
    modal run research_zeta_edit_modal.py::eval_all

Backbone SAO 1.0 (gated — requiere secret huggingface-secret):
    modal run research_zeta_edit_modal.py::eval_all --model-id stabilityai/stable-audio-open-1.0
    (outputs → evaluation/edit/zeta_sao/; cfg_src se fuerza a 1 según el repo)

Coste estimado (A10G): setup (AudioLDM2-music ~5 GB) ~$0.05 · 1 edición (inversión 200
steps + sampling) ~$0.03-0.06 · eval_all (10 casos) ~$0.40-0.60
"""

import json
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("zeta-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

REPO_URL = "https://github.com/HilaManor/AudioEditingCode.git"
REPO_DIR = "/root/AudioEditingCode"

DEFAULT_MODEL_ID = os.environ.get("ZETA_MODEL_ID", "cvssp/audioldm2-music")
DEFAULT_GPU = os.environ.get("ZETA_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("ZETA_SEED", "42"))

# Defaults oficiales del repo (user study del paper)
DEFAULT_STEPS = 200
DEFAULT_TSTART = 100
DEFAULT_CFG_SRC = 3.0
DEFAULT_CFG_TAR = 12.0

SUPPORTED_CATEGORIES = {"style_transfer", "instrumentation", "mood_texture"}
STRENGTH_MAP = {"subtle": 70, "moderate": 100, "strong": 130}


def _model_dir_name(model_id: str) -> str:
    """cvssp/audioldm2-music → zeta_audioldm2 · stabilityai/stable-audio-open-1.0 → zeta_sao"""
    return "zeta_sao" if "stable-audio" in model_id else "zeta_audioldm2"


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# Clona el repo oficial e instala sus requirements (torch 2.x, diffusers>=0.26, …).
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "libsndfile1"])
    .run_commands(f"git clone {REPO_URL} {REPO_DIR}")
    .run_commands(f"pip install -r {REPO_DIR}/requirements.txt")
    # requirements.txt instala diffusers 0.38.0 + transformers 5.x, incompatibles entre sí.
    # diffusers 0.27.0 usa cached_download (eliminado de huggingface_hub ≥ 0.24).
    # diffusers 0.31.0 (Oct 2024) + transformers 4.46.3: sin cached_download y AutoImageProcessor
    # en su ubicación original. VQModel/UNet2DModel siguen disponibles en 0.31.x.
    .run_commands("pip install 'diffusers==0.31.0' 'transformers==4.46.3'")
    # torchaudio >= 2.1 requiere torchcodec para torchaudio.load()
    .run_commands("pip install torchcodec")
    .pip_install("soundfile>=0.12.1")
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache",
        # main_run.py llama wandb.login() a nivel de módulo; WANDB_DISABLED evita el
        # prompt de autenticación en el contenedor.
        "WANDB_DISABLED": "true",
        "WANDB_MODE": "offline",
    })
)

app = modal.App("zeta-edit-inference", image=image)

# Solo necesario para el backbone SAO (gated); inocuo para AudioLDM2.
hf_secret = modal.Secret.from_name("huggingface-secret")


# ---------------------------------------------------------------------------
# Setup — descarga pesos del backbone al Volume
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, secrets=[hf_secret], timeout=3600)
def setup(model_id: str = DEFAULT_MODEL_ID):
    """
    Descarga el backbone al cache HF del Volume (una vez):
        modal run research_zeta_edit_modal.py::setup
        modal run research_zeta_edit_modal.py::setup --model-id stabilityai/stable-audio-open-1.0
    """
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") if "stable-audio" in model_id else None
    print(f"[setup] Descargando {model_id} al cache HF del Volume …")
    t0 = time.time()
    snapshot_download(repo_id=model_id, token=token)
    print(f"[setup] Descarga completa en {time.time()-t0:.0f}s")
    weights_vol.commit()
    print("[setup] Volume commiteado.")


# ---------------------------------------------------------------------------
# Función Modal principal — batch de ediciones vía CLI oficial
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[hf_secret],
    gpu=DEFAULT_GPU,
    timeout=7200,
)
def generate_batch(jobs: list[dict], model_id: str = DEFAULT_MODEL_ID, seed: int = DEFAULT_SEED) -> list[bytes]:
    """
    Ejecuta code/main_run.py del repo oficial por cada job:
      {"source_bytes", "source_name", "source_prompt", "target_prompt",
       "tstart", "cfg_src", "cfg_tar", "num_diffusion_steps"}
    Returns: list[bytes] WAV editado (b"" si falló).
    """
    import shutil
    import subprocess
    import tempfile

    results = []
    code_dir = f"{REPO_DIR}/code"

    for i, job in enumerate(jobs):
        t0 = time.time()
        workdir = tempfile.mkdtemp(prefix="zeta_")
        try:
            suffix = Path(job.get("source_name", "src.wav")).suffix or ".wav"
            # ZETA usa wave.open() (Python stdlib) para get_duration, que solo acepta
            # RIFF WAV. Convertir siempre a mono 22050 Hz WAV via ffmpeg (instalado).
            tmp_in = os.path.join(workdir, f"input{suffix}")
            with open(tmp_in, "wb") as f:
                f.write(job["source_bytes"])
            src_path = os.path.join(workdir, "source.wav")
            subprocess.run(
                ["ffmpeg", "-i", tmp_in, "-ac", "1", "-ar", "22050", src_path, "-y"],
                check=True, capture_output=True,
            )

            results_path = os.path.join(workdir, "out")
            os.makedirs(results_path, exist_ok=True)  # asegurar que el directorio existe
            cfg_src = job.get("cfg_src", DEFAULT_CFG_SRC)
            if "stable-audio" in model_id:
                cfg_src = 1.0  # recomendación oficial del repo para SAO

            cmd = [
                "python", "main_run.py",
                "--mode", "ours",
                "--model_id", model_id,
                "--init_aud", src_path,
                "--target_prompt", job["target_prompt"],
                "--tstart", str(int(job.get("tstart", DEFAULT_TSTART))),
                "--cfg_src", str(cfg_src),
                "--cfg_tar", str(job.get("cfg_tar", DEFAULT_CFG_TAR)),
                "--num_diffusion_steps", str(int(job.get("num_diffusion_steps", DEFAULT_STEPS))),
                "--results_path", results_path,
            ]
            if job.get("source_prompt"):
                cmd += ["--source_prompt", job["source_prompt"]]

            print(f"[generate] [{i+1}/{len(jobs)}] {' '.join(cmd[:8])} …")
            proc = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"main_run.py falló:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
                )

            # main_run.py antepone './' a results_path (que es un mkdtemp absoluto y único
            # por job), por lo que el WAV editado cae en:
            #   code_dir/<results_path_sin_slash_inicial>/…/cfg_e_*.wav
            # Buscar acotado al subárbol único de este job para evitar contaminación cruzada.
            nested_root = Path(code_dir) / str(results_path).lstrip("/")
            edited = [p for p in nested_root.rglob("*.wav") if p.name != "orig.wav"]
            if not edited:
                raise RuntimeError(
                    f"sin WAV editado en {nested_root}.\n"
                    f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
                )
            edited_path = max(edited, key=lambda p: p.stat().st_mtime)
            wav_bytes = edited_path.read_bytes()
            print(
                f"[generate] [{i+1}/{len(jobs)}] OK — '{job['target_prompt'][:50]}' "
                f"→ {len(wav_bytes)//1024} KB, {time.time()-t0:.1f}s"
            )
            results.append(wav_bytes)
        except Exception as exc:
            print(f"[generate] [{i+1}/{len(jobs)}] ERROR: {exc}")
            results.append(b"")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    weights_vol.commit()  # persistir cache HF descargado on-the-fly
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


def _run_jobs_and_save(jobs: list[dict], out_paths: list[Path], model_id: str, seed: int, label: str) -> int:
    print(f"\n[{label}] Enviando {len(jobs)} ediciones a Modal ({DEFAULT_GPU}, backbone={model_id}) …")
    t0 = time.time()
    wav_bytes_list = generate_batch.remote(jobs, model_id=model_id, seed=seed)
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
    source_prompt: str = "",
    tstart: int = DEFAULT_TSTART,
    model_id: str = DEFAULT_MODEL_ID,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Edición ZETA de un audio fuente.

    Ejemplo:
        modal run research_zeta_edit_modal.py::main \\
            --source-audio ../evaluation/edit/source_audio/electronic.mp3 \\
            --source-prompt "A recording of an electronic music track" \\
            --prompt "A recording of a jazz song with piano and double bass" \\
            --out-dir ../evaluation/edit/zeta_audioldm2/manual
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

    print(f"[main] ZETA: '{prompt}' (tstart={tstart}) sobre {src_path.name}")
    [wav_bytes] = generate_batch.remote(
        [{
            "source_bytes": src_path.read_bytes(),
            "source_name": src_path.name,
            "source_prompt": source_prompt,
            "target_prompt": prompt,
            "tstart": tstart,
        }],
        model_id=model_id,
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
    model_id: str = DEFAULT_MODEL_ID,
    seed: int = DEFAULT_SEED,
    force: bool = False,
):
    """
    Smoke con defaults oficiales del repo (user study): electronic.mp3,
    "A recording of an electronic music track…" → "A recording of a jazz song with
    piano and double bass", tstart=100, cfg_src=3, cfg_tar=12, 200 steps.

    → evaluation/edit/zeta_audioldm2/smoke/zeta_smoke_01/output.wav
    Éxito: jazz que conserva la estructura rítmica del original.
    """
    data = _load_prompts_edit(prompts_json)
    smoke_prompts = data["official_smoke"]["zeta_audioldm2"]["prompts"]
    eval_base_path = Path(eval_base).resolve()
    smoke_dir = eval_base_path / "edit" / _model_dir_name(model_id) / "smoke"

    jobs: list[dict] = []
    out_paths: list[Path] = []
    for p in smoke_prompts:
        out_wav = smoke_dir / p["id"] / "output.wav"
        if out_wav.exists() and not force:
            print(f"[skip] {p['id']}: ya tiene output.wav. Usa --force para regenerar.")
            continue
        src_bytes, src = _read_source(eval_base_path, data["sources"], p["source_id"])
        jobs.append({
            "source_bytes": src_bytes,
            "source_name": Path(src["file"]).name,
            "source_prompt": p.get("source_prompt", src["source_prompt"]),
            "target_prompt": p["target_prompt"],
            "tstart": p.get("tstart", DEFAULT_TSTART),
            "cfg_src": p.get("cfg_src", DEFAULT_CFG_SRC),
            "cfg_tar": p.get("cfg_tar", DEFAULT_CFG_TAR),
            "num_diffusion_steps": p.get("num_diffusion_steps", DEFAULT_STEPS),
        })
        out_paths.append(out_wav)
        print(f"[smoke] Encolado: {p['id']} — '{p['target_prompt'][:60]}'")

    if not jobs:
        print("[smoke] Nada pendiente. Usa --force para regenerar.")
        return

    ok = _run_jobs_and_save(jobs, out_paths, model_id, seed, "smoke")
    if ok == len(jobs):
        print("\nSi la edición es razonable (comparar con ejemplos de")
        print("https://hilamanor.github.io/AudioEditing/) → eval_all:")
        print("    modal run research_zeta_edit_modal.py::eval_all")


# ---------------------------------------------------------------------------
# Entrypoint: benchmark completo
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    prompts_json: str = "../evaluation/prompts_edit.json",
    eval_base: str = "../evaluation",
    model_id: str = DEFAULT_MODEL_ID,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    only: str = "",
):
    """
    Edita todos los casos soportados de prompts_edit.json.
    Cada caso genera: evaluation/edit/<zeta_audioldm2|zeta_sao>/<case_id>/output.wav
    ZETA usa el source_prompt del source (descripción del original) + target_prompt del caso.

    Ejemplos:
        modal run research_zeta_edit_modal.py::eval_all
        modal run research_zeta_edit_modal.py::eval_all --model-id stabilityai/stable-audio-open-1.0
    """
    data = _load_prompts_edit(prompts_json)
    eval_base_path = Path(eval_base).resolve()
    model_path = eval_base_path / "edit" / _model_dir_name(model_id)

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
        tstart = STRENGTH_MAP.get(c.get("strength_hint") or "moderate", DEFAULT_TSTART)
        jobs.append({
            "source_bytes": src_bytes,
            "source_name": Path(src["file"]).name,
            "source_prompt": src["source_prompt"],
            "target_prompt": c["target_prompt"],
            "tstart": tstart,
        })
        out_paths.append(out_wav)
        print(f"[eval_all] Encolado: {c['id']} ({c['category']}, tstart={tstart})")

    if not jobs:
        print("[eval_all] Todos los casos ya tienen output.wav. Usa --force para regenerar.")
        return

    _run_jobs_and_save(jobs, out_paths, model_id, seed, "eval_all")
    print("\nSiguiente paso:")
    print("  bash ../evaluation/render_norm.sh")
    print(f"  uv run python ../evaluation/compute_metrics_edit.py --only {_model_dir_name(model_id)}")
