"""
Modal.com inference app para ACE-Step 1.5 — subtarea EDICIÓN/VERSIONADO (cover + repaint).

ACE-Step 1.5 (ace-step/ACE-Step-1.5, MIT) es un modelo de generación musical full-song
con soporte nativo de tareas de edición sobre un audio existente:
  - cover:   audio fuente + caption de estilo → versión del tema en el estilo pedido.
             `audio_cover_strength` controla la fidelidad al original (1.0 = muy fiel,
             0.1 = interpretación libre).
  - repaint: regenera un segmento [start, end] del audio fuente según el caption.

Punto de entrada DAW:
  - Input: audio fuente (wav/mp3) + caption de estilo + strength.
  - Output: WAV estéreo 48 kHz.
  - Ideal para: "haz una versión jazz de este loop", "convierte este sample a orquesta".
  - Licencia: MIT (código y pesos) — la más permisiva de todos los candidatos de edición.

Condiciones oficiales replicadas (docs/en/INFERENCE.md, verificado 2026-07-02):
  https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
  - Checkpoint DiT: acestep-v15-turbo (2B, ~4.7 GB bf16) — cabe holgado en A10G.
  - Turbo: inference_steps=8, shift=3.0 (recomendado para turbo), guidance_scale
    auto-corregido a 1.0.
  - LM handler NO inicializado → generate_music() omite las fases LM (thinking) de forma
    controlada (llm_handler.llm_initialized == False). Evita instalar/cargar vLLM.
  - Ejemplo cover verbatim de la doc: caption "orchestral symphonic arrangement",
    audio_cover_strength=0.7.
  - Duración soportada: 10 s – 600 s. Con sources < 10 s el output puede quedar
    extendido a la duración mínima — documentado como observación en la evaluación.

Mapeo strength_hint (prompts_edit.json) → audio_cover_strength:
  subtle → 0.9   (muy fiel al original)
  moderate → 0.7 (valor del ejemplo oficial de la doc)
  strong → 0.4   (interpretación libre)

Setup (descarga pesos al Volume, una vez):
    modal run research_acestep_edit_modal.py::setup

Smoke test oficial (2 pasos autocontenidos — el repo no incluye audios de ejemplo):
    modal run research_acestep_edit_modal.py::smoke
    # 1) text2music: "calm ambient music with soft piano and strings" (ejemplo doc)
    # 2) cover del paso 1: "orchestral symphonic arrangement", strength 0.7 (ejemplo doc)
    # → evaluation/edit/acestep15/smoke/ace_smoke_{01_source,02_cover}/output.wav
    # Éxito: el cover mantiene estructura/tempo del source con timbre orquestal.

Edición libre (un source + un caption):
    modal run research_acestep_edit_modal.py::main \\
        --source-audio ../evaluation/edit/source_audio/sao_guitar_loop.wav \\
        --prompt "jazz piano version, swing feel" \\
        --strength 0.7 \\
        --out-dir ../evaluation/edit/acestep15/manual

Benchmark completo (casos de prompts_edit.json — categorías con soporte cover):
    modal run research_acestep_edit_modal.py::eval_all

Coste estimado (A10G):
    - setup: descarga ~5 GB → ~$0.05
    - 1 cover con turbo (8 steps): segundos de inferencia → ~$0.01
    - eval_all (10 casos de edición): ~$0.10-0.15
"""

import json
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("acestep15-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

DIT_CONFIG = os.environ.get("ACESTEP_DIT", "acestep-v15-turbo")
DEFAULT_GPU = os.environ.get("ACESTEP_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("ACESTEP_SEED", "42"))

MODEL_DIR_NAME = "acestep15"  # evaluation/edit/<model>/

# Categorías de prompts_edit.json que la tarea cover cubre
SUPPORTED_CATEGORIES = {"style_transfer", "instrumentation", "mood_texture"}

# strength_hint → audio_cover_strength (1.0 = fiel al original, según doc oficial)
STRENGTH_MAP = {"subtle": 0.9, "moderate": 0.7, "strong": 0.4}

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# ACE-Step 1.5 requiere Python 3.11-3.12 (README).
# Instalamos desde repo clonado, eliminando nano-vllm (paquete interno de StepFun que
# no existe en PyPI para Linux pero es dependencia de ace-step; solo se necesita para
# las fases LLM que este script deshabilita explícitamente).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "ffmpeg", "libsndfile1"])
    # Pasos de instalación separados para claridad y para que Modal cachee cada capa:
    # 1. Clonar repo y filtrar deps incompatibles con el mirror de Modal:
    #    - nano-vllm: paquete interno de StepFun, no en PyPI; solo necesario para fases LLM
    #      que este script deshabilita.
    #    - torch==X.Y+cuNNN: CUDA build custom que no existe en pypi-mirror.modal.local;
    #      pre-instalamos torch estándar antes.
    # 2. Pre-instalar torch (satisface la dep antes de que pip intente el cu-build custom).
    # 3. Instalar ace-step desde el repo local (torch ya satisfecho, nano-vllm eliminado).
    .run_commands(
        "git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /root/ACEStep && "
        # Eliminar deps problemáticas del pyproject.toml antes de instalar:
        #   +cu<NNN>: pinnings CUDA custom (torch, torchaudio, torchvision) que no existen en
        #             el mirror de Modal; instalamos torch estándar en el paso siguiente.
        #   nano-vllm: paquete interno de StepFun no disponible en PyPI; solo para fases LLM.
        "sed -i '/+cu[0-9]/d; /nano-vllm/d' /root/ACEStep/pyproject.toml"
    )
    .pip_install("torch", "torchaudio", "torchvision")  # pre-instalar antes de ace-step
    .run_commands("pip install /root/ACEStep")
    .pip_install("soundfile>=0.12.1")
    .env({"HF_HOME": f"{WEIGHTS_MOUNT}/hf-cache"})
)

app = modal.App("acestep-edit-inference", image=image)


# ---------------------------------------------------------------------------
# Helpers (dentro del container)
# ---------------------------------------------------------------------------

def _init_handlers():
    """
    Inicializa el handler DiT según la doc oficial (docs/en/API.md + INFERENCE.md).
    El LLMHandler se crea SIN inicializar: generate_music() detecta
    llm_initialized == False y omite las fases LM (thinking).
    """
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
    """Ejecuta generate_music con GenerationParams(**params_kwargs) y devuelve bytes WAV."""
    import io
    import tempfile

    import soundfile as sf
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    params = GenerationParams(**params_kwargs)
    config = GenerationConfig(batch_size=1, audio_format="wav")

    with tempfile.TemporaryDirectory() as tmp:
        result = generate_music(dit_handler, llm_handler, params, config, save_dir=tmp)
        if not getattr(result, "success", True) or not result.audios:
            raise RuntimeError(f"generate_music sin audios (task={params_kwargs.get('task_type')})")

        audio = result.audios[0]
        tensor = audio["tensor"]  # (channels, samples), float32 CPU
        sr = audio["sample_rate"]

        buf = io.BytesIO()
        sf.write(buf, tensor.numpy().T, samplerate=sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# Setup — fuerza la descarga de pesos al Volume
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=3600)
def setup():
    """
    Inicializa el handler una vez para forzar la descarga de los checkpoints
    (auto-download desde HuggingFace en el primer uso, según README) y commitea
    el Volume. Ejecutar una sola vez:
        modal run research_acestep_edit_modal.py::setup
    """
    _init_handlers()
    weights_vol.commit()
    print("[setup] Pesos descargados y Volume commiteado.")


# ---------------------------------------------------------------------------
# Función Modal principal — batch de trabajos de edición
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=7200)
def generate_batch(jobs: list[dict], seed: int = DEFAULT_SEED) -> list[bytes]:
    """
    Ejecuta un batch de trabajos. Cada job es un dict:
      - "task_type": "cover" | "repaint" | "text2music"
      - "caption": str
      - "src_audio_bytes": bytes | None  — audio fuente (cover/repaint)
      - "src_audio_name": str            — nombre original (extensión correcta)
      - "audio_cover_strength": float    — solo cover
      - "repainting_start"/"repainting_end": float — solo repaint
      - "duration"/"bpm"/"keyscale":     — solo text2music
    Returns: list[bytes] WAV 48 kHz (b"" si falló).
    """
    import tempfile

    dit_handler, llm_handler = _init_handlers()
    weights_vol.commit()  # persistir pesos si el setup no se ejecutó antes

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            params_kwargs = {
                "task_type": job["task_type"],
                "caption": job["caption"],
                "thinking": False,  # sin LM handler
                "inference_steps": 8,  # turbo (doc oficial)
                "shift": 3.0,  # recomendado para turbo (doc oficial)
                "seed": seed + i,
            }
            tmp_src = None
            if job.get("src_audio_bytes"):
                suffix = Path(job.get("src_audio_name", "src.wav")).suffix or ".wav"
                tmp_src = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp_src.write(job["src_audio_bytes"])
                tmp_src.flush()
                params_kwargs["src_audio"] = tmp_src.name

            if job["task_type"] == "cover":
                params_kwargs["audio_cover_strength"] = float(job.get("audio_cover_strength", 0.7))
            elif job["task_type"] == "repaint":
                params_kwargs["repainting_start"] = float(job.get("repainting_start", 0.0))
                params_kwargs["repainting_end"] = float(job.get("repainting_end", -1.0))
            elif job["task_type"] == "text2music":
                for k in ("duration", "bpm", "keyscale"):
                    if job.get(k) is not None:
                        params_kwargs[k] = job[k]

            wav_bytes = _generate_one(dit_handler, llm_handler, params_kwargs)
            print(
                f"[generate] [{i+1}/{len(jobs)}] OK — {job['task_type']} "
                f"'{job['caption'][:50]}' → {len(wav_bytes)//1024} KB, {time.time()-t0:.1f}s"
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


def _read_source(eval_base: Path, sources: list[dict], source_id: str) -> tuple[bytes, str]:
    src = next((s for s in sources if s["id"] == source_id), None)
    if src is None:
        raise ValueError(f"source_id desconocido: {source_id}")
    src_path = eval_base / src["file"]
    if not src_path.exists():
        raise FileNotFoundError(
            f"No existe {src_path}. Ejecuta antes: bash ../evaluation/fetch_edit_sources.sh"
        )
    return src_path.read_bytes(), src_path.name


def _save_outputs(wav_bytes_list: list[bytes], out_paths: list[Path]) -> int:
    ok = 0
    for wav_bytes, out_wav in zip(wav_bytes_list, out_paths):
        if not wav_bytes:
            print(f"[save] ERROR: {out_wav.parent.name} devolvió vacío — saltando")
            continue
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(wav_bytes)
        print(f"[save] Guardado: {out_wav.parent.name}/output.wav ({len(wav_bytes)//1024} KB)")
        ok += 1
    return ok


# ---------------------------------------------------------------------------
# Entrypoint: edición libre (un source + un caption)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    source_audio: str = "",
    prompt: str = "",
    strength: float = 0.7,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Cover de un audio fuente con un caption de estilo.

    Ejemplo:
        modal run research_acestep_edit_modal.py::main \\
            --source-audio ../evaluation/edit/source_audio/sao_guitar_loop.wav \\
            --prompt "jazz piano version, swing feel" \\
            --strength 0.7 \\
            --out-dir ../evaluation/edit/acestep15/manual
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

    print(f"[main] Cover: '{prompt}' (strength={strength}) sobre {src_path.name}")
    [wav_bytes] = generate_batch.remote(
        [{
            "task_type": "cover",
            "caption": prompt,
            "src_audio_bytes": src_path.read_bytes(),
            "src_audio_name": src_path.name,
            "audio_cover_strength": strength,
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
# Entrypoint: repaint (regenerar un segmento del source)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def repaint(
    source_audio: str = "",
    prompt: str = "",
    start: float = 0.0,
    end: float = -1.0,
    seed: int = DEFAULT_SEED,
    out_dir: str = ".",
    force: bool = False,
):
    """
    Regenera el segmento [start, end] del audio fuente según el caption.
    end = -1 → hasta el final del archivo (convención de la doc oficial).

    Ejemplo:
        modal run research_acestep_edit_modal.py::repaint \\
            --source-audio ../evaluation/edit/source_audio/bolero_ravel.mp3 \\
            --prompt "smooth transition with piano solo" \\
            --start 4.0 --end 8.0 \\
            --out-dir ../evaluation/edit/acestep15/repaint_test
    """
    if not source_audio or not prompt:
        print("ERROR: --source-audio y --prompt son requeridos.")
        raise SystemExit(1)

    src_path = Path(source_audio).resolve()
    out_p = Path(out_dir)
    out_wav = out_p / "output.wav"
    if out_wav.exists() and not force:
        print(f"[repaint] Ya existe {out_wav}. Usa --force para regenerar.")
        return

    print(f"[repaint] '{prompt}' en [{start}, {end}] de {src_path.name}")
    [wav_bytes] = generate_batch.remote(
        [{
            "task_type": "repaint",
            "caption": prompt,
            "src_audio_bytes": src_path.read_bytes(),
            "src_audio_name": src_path.name,
            "repainting_start": start,
            "repainting_end": end,
        }],
        seed=seed,
    )
    if not wav_bytes:
        print("[repaint] ERROR: la generación devolvió vacío.")
        raise SystemExit(1)
    out_p.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(wav_bytes)
    print(f"[repaint] Guardado: {out_wav} ({len(wav_bytes)//1024} KB)")


# ---------------------------------------------------------------------------
# Entrypoint: smoke test oficial (2 pasos: text2music → cover)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def smoke(
    prompts_json: str = "../evaluation/prompts_edit.json",
    eval_base: str = "../evaluation",
    seed: int = DEFAULT_SEED,
    force: bool = False,
):
    """
    Replica los ejemplos VERBATIM de docs/en/INFERENCE.md en dos pasos:
      1. text2music: "calm ambient music with soft piano and strings" (bpm 80, C Major)
      2. cover del paso 1: "orchestral symphonic arrangement", strength 0.7

    → evaluation/edit/acestep15/smoke/ace_smoke_01_source/output.wav
    → evaluation/edit/acestep15/smoke/ace_smoke_02_cover/output.wav

    Criterios de éxito:
      - Ambos WAV 48 kHz estéreo, sin silencio.
      - El cover conserva estructura/tempo del source con timbre orquestal.
      - Inferencia turbo: pocos segundos por audio en A10G.
    """
    data = _load_prompts_edit(prompts_json)
    smoke_cfg = data["official_smoke"]["acestep15"]["steps"]
    step_t2m = next(s for s in smoke_cfg if s["task_type"] == "text2music")
    step_cover = next(s for s in smoke_cfg if s["task_type"] == "cover")

    smoke_dir = Path(eval_base).resolve() / "edit" / MODEL_DIR_NAME / "smoke"
    out_source = smoke_dir / step_t2m["id"] / "output.wav"
    out_cover = smoke_dir / step_cover["id"] / "output.wav"

    if out_cover.exists() and not force:
        print(f"[smoke] Ya existe {out_cover}. Usa --force para regenerar.")
        return

    # Paso 1 — text2music (source)
    if out_source.exists() and not force:
        print(f"[smoke] Source ya existe: {out_source}")
        source_bytes = out_source.read_bytes()
    else:
        print(f"[smoke] Paso 1 — text2music: '{step_t2m['caption']}'")
        [source_bytes] = generate_batch.remote(
            [{
                "task_type": "text2music",
                "caption": step_t2m["caption"],
                "duration": step_t2m.get("duration"),
                "bpm": step_t2m.get("bpm"),
                "keyscale": step_t2m.get("keyscale"),
            }],
            seed=seed,
        )
        if not source_bytes:
            print("[smoke] ERROR: text2music devolvió vacío.")
            raise SystemExit(1)
        out_source.parent.mkdir(parents=True, exist_ok=True)
        out_source.write_bytes(source_bytes)
        print(f"[smoke] Source guardado: {out_source} ({len(source_bytes)//1024} KB)")

    # Paso 2 — cover del source
    print(f"[smoke] Paso 2 — cover: '{step_cover['caption']}' (strength={step_cover['audio_cover_strength']})")
    [cover_bytes] = generate_batch.remote(
        [{
            "task_type": "cover",
            "caption": step_cover["caption"],
            "src_audio_bytes": source_bytes,
            "src_audio_name": "source.wav",
            "audio_cover_strength": step_cover["audio_cover_strength"],
        }],
        seed=seed,
    )
    if not cover_bytes:
        print("[smoke] ERROR: cover devolvió vacío.")
        raise SystemExit(1)
    out_cover.parent.mkdir(parents=True, exist_ok=True)
    out_cover.write_bytes(cover_bytes)
    print(f"[smoke] Cover guardado: {out_cover} ({len(cover_bytes)//1024} KB)")

    print("\n[smoke] Escuchar ambos en REAPER y verificar que el cover mantiene la")
    print("        estructura del source con timbre orquestal. Si OK → eval_all:")
    print("            modal run research_acestep_edit_modal.py::eval_all")


# ---------------------------------------------------------------------------
# Entrypoint: benchmark completo (casos de prompts_edit.json)
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
    Ejecuta la tarea cover sobre todos los casos soportados de prompts_edit.json
    (categorías: style_transfer, instrumentation, mood_texture).

    Cada caso genera: evaluation/edit/acestep15/<case_id>/output.wav

    Ejemplos:
        modal run research_acestep_edit_modal.py::eval_all
        modal run research_acestep_edit_modal.py::eval_all --only case01_bach_jazz,case07_guitar_piano
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

        src_bytes, src_name = _read_source(eval_base_path, data["sources"], c["source_id"])
        strength = STRENGTH_MAP.get(c.get("strength_hint") or "moderate", 0.7)
        jobs.append({
            "task_type": "cover",
            "caption": c["target_prompt"],
            "src_audio_bytes": src_bytes,
            "src_audio_name": src_name,
            "audio_cover_strength": strength,
        })
        out_paths.append(out_wav)
        print(f"[eval_all] Encolado: {c['id']} ({c['category']}, strength={strength})")

    if not jobs:
        print("[eval_all] Todos los casos ya tienen output.wav. Usa --force para regenerar.")
        return

    print(f"\n[eval_all] Enviando {len(jobs)} covers a Modal ({DEFAULT_GPU}) …")
    t0 = time.time()
    wav_bytes_list = generate_batch.remote(jobs, seed=seed)
    print(f"[eval_all] Batch completado en {time.time()-t0:.0f}s")

    ok = _save_outputs(wav_bytes_list, out_paths)
    print(f"\n[eval_all] {ok}/{len(jobs)} casos generados con éxito.")
    print("\nSiguiente paso:")
    print("  bash ../evaluation/render_norm.sh")
    print("  uv run python ../evaluation/compute_metrics_edit.py --only acestep15")
