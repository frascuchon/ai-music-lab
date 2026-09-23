"""
Modal.com inference app para ACE-Step 1.5 — subtarea GENERACIÓN (text2music).

ACE-Step 1.5 soporta generación de música de longitud arbitraria condicionada
exclusivamente por texto (tarea "text2music"). Esta es la variante de GENERACIÓN PURA.
Para la variante de edición (cover/style transfer), usar research_acestep_edit_modal.py.

Punto de entrada DAW:
  - Input: prompt de texto + duración en segundos + parámetros de inferencia
    opcionales (steps, guidance_scale, shift, seed, ADG, CFG interval,
    thinking/LM temperature) + adapter LoRA/LoKr opcional desde una carpeta
    local.
  - Output: WAV estéreo 48 kHz.
  - Ideal para: canciones completas (full-song), generación de larga duración.
  - Licencia: MIT (código y pesos) — la más permisiva; apto para uso comercial.

Reutiliza el Volume acestep15-weights (pesos compartidos con research_acestep_edit_modal.py).

Condiciones oficiales (docs/en/INFERENCE.md, verificado 2026-07-02):
  https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
  - Checkpoint DiT: acestep-v15-turbo (2B, ~4.7 GB bf16)
  - Turbo: inference_steps=8, shift=3.0 (defaults; ajustables dentro de STEPS_RANGE_BY_VARIANT[variant]/SHIFT_RANGE)
  - Tarea text2music: caption + duration (segundos) → audio full-song
  - guidance_scale / use_adg / cfg_interval_start-end: el checkpoint turbo
    (CFG-distilled) los ignora; se aceptan y clampan igualmente por si
    --dit-variant base apunta a un checkpoint base/SFT.

Límites de parámetros — mirror de los sliders de la Gradio UI oficial para
el checkpoint turbo (acestep/ui/gradio/events/generation/model_config.py):
  inference_steps  1–20      guidance_scale  1.0–15.0
  shift             1.0–5.0  cfg_interval_*  0.0–1.0
  lm_temperature    0.0–2.0  lora_scale      0.0–1.0
Todo valor fuera de rango se clampa (no se rechaza), tanto en main() como
dentro de generate_batch — ver STEPS_RANGE_BY_VARIANT/GUIDANCE_SCALE_RANGE/etc. abajo.

Adapter LoRA/LoKr desde carpeta local: la carpeta debe contener un adapter
PEFT (adapter_config.json + adapter_model.safetensors) o un artefacto LoKr
(lokr_weights.safetensors) — mismo formato que produce el entrenamiento con
Side-Step. El contenedor Modal no tiene acceso al filesystem local, así que
main() lee la carpeta entera en memoria y la envía como parte de la llamada
remota (ver _read_lora_dir / _materialize_lora_dir); límite de 2 GB para
evitar subir por error un directorio de checkpoint de entrenamiento completo.

Setup (descarga pesos al Volume, una vez; compartido con script de edición):
    modal run research_acestep_gen_modal.py::setup

Generación libre:
    modal run research_acestep_gen_modal.py::main \\
        --prompt "uplifting electronic dance music with synthesizers and driving beat" \\
        --seconds 30.0 \\
        --out-dir /tmp/acestep_gen_test

Con parámetros de inferencia y un adapter LoRA:
    modal run research_acestep_gen_modal.py::main \\
        --prompt "funk reinterpretation with syncopated bass" \\
        --seconds 60.0 --steps 12 --guidance-scale 7.5 --shift 3.0 \\
        --lora-path /path/to/my_adapter --lora-scale 0.8 \\
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

DEFAULT_DIT_VARIANT = os.environ.get("ACESTEP_DIT_VARIANT", "turbo")
DEFAULT_GPU = os.environ.get("ACESTEP_GPU", "A10G")
DEFAULT_SEED = int(os.environ.get("ACESTEP_SEED", "42"))

# ---------------------------------------------------------------------------
# Checkpoint variants + inference-parameter limits — mirrors the ranges the
# official ACE-Step 1.5 Gradio UI enforces per checkpoint type, see
# acestep/ui/gradio/events/generation/model_config.py::get_ui_control_config
# (is_turbo branch vs. else) and generation_advanced_dit_controls.py's slider
# bounds. Values outside these are silently clamped rather than rejected,
# both here and in the Lua UI, so a stale/hand-edited command line can't push
# the model into an unsupported or nonsensical configuration.
#
# CFG (guidance_scale/use_adg/cfg_interval_start-end) is meaningless for
# turbo — it's a CFG-distilled checkpoint, generate_music() itself hard-
# overrides guidance_scale to 1.0 for turbo (verified empirically: passing
# guidance_scale=8.0 to turbo logs "overriding guidance_scale 8.0 -> 1.0").
# The Lua UI hides those controls entirely for turbo rather than just
# graying them out, since setting them has zero effect there.
# ---------------------------------------------------------------------------
DIT_CONFIG_BY_VARIANT = {
    "turbo": "acestep-v15-turbo",
    "base": "acestep-v15-base",
}
STEPS_RANGE_BY_VARIANT = {
    "turbo": (1, 20),
    "base": (1, 200),
}
STEPS_DEFAULT_BY_VARIANT = {"turbo": 8, "base": 32}
GUIDANCE_SCALE_RANGE = (1.0, 15.0)  # base/SFT only — ignored (hard-overridden) for turbo
SHIFT_RANGE = (1.0, 5.0)
CFG_INTERVAL_RANGE = (0.0, 1.0)      # base/SFT only
LM_TEMPERATURE_RANGE = (0.0, 2.0)
LORA_SCALE_RANGE = (0.0, 1.0)       # set_lora_scale() clamps to this anyway; matched here for clear logging


def _clamp(value, lo, hi, name: str):
    clamped = max(lo, min(hi, value))
    if clamped != value:
        print(f"[params] {name}={value} out of range [{lo}, {hi}] — clamped to {clamped}")
    return clamped


def _resolve_dit_config(variant: str) -> str:
    """Map a "turbo"/"base" variant to its ACE-Step config_path, falling
    back to turbo (with a warning) for anything unrecognized — never lets a
    typo'd/stale variant string reach initialize_service()."""
    key = (variant or "").strip().lower()
    if key not in DIT_CONFIG_BY_VARIANT:
        print(f"[params] Unknown ACE-Step variant {variant!r} — falling back to 'turbo'")
        key = "turbo"
    return DIT_CONFIG_BY_VARIANT[key]

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

def _init_handlers(dit_variant: str = "turbo", lora_path: str = "", lora_scale: float = 1.0):
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    dit_config = _resolve_dit_config(dit_variant)
    t0 = time.time()
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=WEIGHTS_MOUNT,
        config_path=dit_config,
        device="cuda",
    )
    llm_handler = LLMHandler()  # sin .initialize() — fases LM omitidas
    print(f"[init] DiT {dit_config} listo ({time.time()-t0:.1f}s), LM deshabilitado")

    if lora_path:
        _load_lora(dit_handler, lora_path, lora_scale)

    return dit_handler, llm_handler


def _load_lora(dit_handler, lora_path: str, lora_scale: float) -> None:
    """Load a LoRA/LoKr adapter from a local folder into dit_handler and
    activate it for the rest of this container's generations.

    AceStepHandler's LoRA methods (load_lora/set_lora_scale/set_use_lora)
    report failure as a returned "❌ ..." string rather than raising — this
    wraps that contract into a RuntimeError so callers (generate_batch/main)
    can fail the run loudly instead of silently generating without the
    adapter the user explicitly asked for.
    """
    msg = dit_handler.load_lora(lora_path)
    if not msg.startswith("✅"):
        raise RuntimeError(f"Failed to load LoRA adapter from {lora_path!r}: {msg}")
    print(f"[lora] {msg}")

    scale = _clamp(lora_scale, *LORA_SCALE_RANGE, "lora_scale")
    scale_msg = dit_handler.set_lora_scale(scale)
    if not scale_msg.startswith("✅"):
        raise RuntimeError(f"Failed to set LoRA scale: {scale_msg}")
    print(f"[lora] {scale_msg}")

    enable_msg = dit_handler.set_use_lora(True)
    if not enable_msg.startswith("✅"):
        raise RuntimeError(f"Failed to enable LoRA adapter: {enable_msg}")
    print(f"[lora] {enable_msg}")


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
def setup(dit_variant: str = "turbo"):
    """
    Descarga pesos de ACE-Step 1.5 al Volume compartido. Ejecutar una vez por variante:
        modal run research_acestep_gen_modal.py::setup                       # turbo (~5 GB)
        modal run research_acestep_gen_modal.py::setup --dit-variant base    # base (larger)
    Si ya ejecutaste research_acestep_edit_modal.py::setup, turbo ya está listo.
    """
    _init_handlers(dit_variant=dit_variant)
    weights_vol.commit()
    print(f"[setup] Pesos ({dit_variant}) descargados y Volume commiteado.")


def _materialize_lora_dir(lora_files: dict[str, bytes] | None) -> str:
    """Write a LoRA/LoKr adapter's files (shipped as {relative_path: bytes}
    by main(), since the container has no access to the user's local
    filesystem) into a temp directory inside the container, and return its
    path — or "" if no adapter was requested."""
    if not lora_files:
        return ""
    import tempfile

    lora_dir = tempfile.mkdtemp(prefix="lora_adapter_")
    for rel_path, data in lora_files.items():
        dest = Path(lora_dir) / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print(f"[lora] Materialized {len(lora_files)} file(s) → {lora_dir}")
    return lora_dir


# ---------------------------------------------------------------------------
# Función Modal principal — generación por lotes
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, gpu=DEFAULT_GPU, timeout=7200)
def generate_batch(
    jobs: list[dict],
    seed: int = DEFAULT_SEED,
    dit_variant: str = "turbo",
    inference_steps: int = 8,
    guidance_scale: float = 7.0,
    shift: float = 3.0,
    use_adg: bool = False,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    thinking: bool = False,
    lm_temperature: float = 0.85,
    lora_files: dict[str, bytes] | None = None,
    lora_scale: float = 1.0,
) -> list[bytes]:
    """
    Cada job: {"text": str, "seconds": float}.
    Inference params apply to every job in the batch (matches how `seed` is
    already a per-batch base offset by job index below). All are clamped to
    STEPS_RANGE_BY_VARIANT/GUIDANCE_SCALE_RANGE/etc. — see their definitions
    above — before use, regardless of what the caller passed. guidance_scale/
    use_adg/cfg_interval_* are accepted for any dit_variant but only have an
    effect on "base" — the turbo checkpoint hard-overrides them internally.
    Returns: list[bytes] WAV 48 kHz (b"" si falló).
    """
    steps_range = STEPS_RANGE_BY_VARIANT.get(dit_variant, STEPS_RANGE_BY_VARIANT["turbo"])
    inference_steps = int(_clamp(inference_steps, *steps_range, "inference_steps"))
    guidance_scale = _clamp(guidance_scale, *GUIDANCE_SCALE_RANGE, "guidance_scale")
    shift = _clamp(shift, *SHIFT_RANGE, "shift")
    cfg_interval_start = _clamp(cfg_interval_start, *CFG_INTERVAL_RANGE, "cfg_interval_start")
    cfg_interval_end = _clamp(cfg_interval_end, *CFG_INTERVAL_RANGE, "cfg_interval_end")
    if cfg_interval_start > cfg_interval_end:
        print(f"[params] cfg_interval_start ({cfg_interval_start}) > cfg_interval_end "
              f"({cfg_interval_end}) — swapping")
        cfg_interval_start, cfg_interval_end = cfg_interval_end, cfg_interval_start
    lm_temperature = _clamp(lm_temperature, *LM_TEMPERATURE_RANGE, "lm_temperature")

    lora_path = _materialize_lora_dir(lora_files)
    dit_handler, llm_handler = _init_handlers(
        dit_variant=dit_variant, lora_path=lora_path, lora_scale=lora_scale)
    weights_vol.commit()

    results = []
    for i, job in enumerate(jobs):
        t0 = time.time()
        try:
            wav_bytes = _generate_one(dit_handler, llm_handler, {
                "task_type": "text2music",
                "caption": job["text"],
                "duration": float(job.get("seconds") or 30.0),
                "thinking": thinking,
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "use_adg": use_adg,
                "cfg_interval_start": cfg_interval_start,
                "cfg_interval_end": cfg_interval_end,
                "lm_temperature": lm_temperature,
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
# LoRA folder → bytes (runs locally; the container has no access to the
# user's filesystem, so the adapter's files travel as part of the .remote()
# call payload — same pattern research_acestep_edit_modal.py already uses
# for a single --source-audio file, extended to a whole directory).
# ---------------------------------------------------------------------------
_LORA_SKIP_NAMES = {".git", ".ds_store", "__pycache__", ".ipynb_checkpoints"}
_LORA_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB — generous for an inference adapter,
                                                  # guards against accidentally pointing at a
                                                  # full training checkpoint dir (optimizer state etc.)


def _read_lora_dir(lora_path: str) -> dict[str, bytes]:
    root = Path(lora_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LoRA adapter folder not found: {root}")

    files: dict[str, bytes] = {}
    total = 0
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if any(part.lower() in _LORA_SKIP_NAMES for part in f.relative_to(root).parts):
            continue
        size = f.stat().st_size
        total += size
        if total > _LORA_MAX_TOTAL_BYTES:
            raise ValueError(
                f"LoRA adapter folder exceeds {_LORA_MAX_TOTAL_BYTES // (1024**3)} GB "
                f"({root}) — this looks like a full training checkpoint directory rather "
                "than an inference-ready adapter (expected: adapter_config.json + "
                "adapter_model.safetensors, or a lokr_weights.safetensors)."
            )
        files[str(f.relative_to(root))] = f.read_bytes()

    if not files:
        raise FileNotFoundError(f"LoRA adapter folder is empty: {root}")
    print(f"[lora] Read {len(files)} file(s), {total/1024/1024:.1f} MB from {root}")
    return files


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
    dit_variant: str = "turbo",
    steps: int = -1,  # -1 = use STEPS_DEFAULT_BY_VARIANT[dit_variant]
    guidance_scale: float = 7.0,
    shift: float = 3.0,
    use_adg: bool = False,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    thinking: bool = False,
    lm_temperature: float = 0.85,
    lora_path: str = "",
    lora_scale: float = 1.0,
):
    """
    Genera música desde un prompt de texto (ACE-Step text2music).

    Ejemplo:
        modal run research_acestep_gen_modal.py::main \\
            --prompt "uplifting electronic dance music with synthesizers and driving beat" \\
            --seconds 30.0 \\
            --dit-variant turbo --steps 12 --guidance-scale 7.5 --shift 3.0 \\
            --lora-path /path/to/my_adapter --lora-scale 0.8 \\
            --out-dir /tmp/acestep_gen_test

    dit_variant: "turbo" (fast, 8-20 steps, no CFG — the default) or "base"
    (32-200 steps, CFG-capable via guidance_scale/use_adg/cfg_interval_*).
    steps=-1 (default) picks the variant's own default (8 for turbo, 32 for
    base) — see STEPS_DEFAULT_BY_VARIANT above.

    Todos los parámetros de inferencia se clampan a rangos válidos (ver
    STEPS_RANGE_BY_VARIANT/GUIDANCE_SCALE_RANGE/etc. arriba) tanto aquí como dentro de
    generate_batch — un valor fuera de rango nunca llega al modelo, solo se
    loguea el clamp aplicado.
    """
    if not prompt:
        print("ERROR: --prompt es requerido.")
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_wav = out_p / "output.wav"
    if out_wav.exists() and not force:
        print(f"[main] Ya existe {out_wav}. Usa --force para regenerar.")
        return

    dit_variant_key = (dit_variant or "").strip().lower()
    if dit_variant_key not in DIT_CONFIG_BY_VARIANT:
        print(f"[main] Unknown --dit-variant {dit_variant!r} — falling back to 'turbo'")
        dit_variant_key = "turbo"
    if steps is None or steps < 0:
        steps = STEPS_DEFAULT_BY_VARIANT[dit_variant_key]

    lora_files = None
    if lora_path:
        try:
            lora_files = _read_lora_dir(lora_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            raise SystemExit(1)

    print(f"[main] Generando (ACE-Step text2music, {dit_variant_key}): '{prompt}' ({seconds}s)"
          f"{' + LoRA ' + lora_path if lora_path else ''}")
    [wav_bytes] = generate_batch.remote(
        [{"text": prompt, "seconds": seconds}],
        seed=seed,
        dit_variant=dit_variant_key,
        inference_steps=steps,
        guidance_scale=guidance_scale,
        shift=shift,
        use_adg=use_adg,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        thinking=thinking,
        lm_temperature=lm_temperature,
        lora_files=lora_files,
        lora_scale=lora_scale,
    )
    if not wav_bytes:
        print("[main] ERROR: la generación devolvió vacío.")
        raise SystemExit(1)
    out_p.mkdir(parents=True, exist_ok=True)
    out_wav.write_bytes(wav_bytes)
    print(f"[main] Guardado: {out_wav} ({len(wav_bytes)//1024} KB)")
