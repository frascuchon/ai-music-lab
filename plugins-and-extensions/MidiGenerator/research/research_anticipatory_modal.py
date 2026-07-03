"""
Modal.com inference app para Anticipatory Music Transformer (Stanford CRFM, Apache 2.0).

Modos:
  - continuation:   genera continuación libre de un MIDI existente
  - accompaniment:  genera acompañamiento dado una melodía como control

Modelos disponibles (HuggingFace):
  stanford-crfm/music-small-800k    (~100M params,  ~400 MB)
  stanford-crfm/music-medium-800k   (~360M params, ~1.4 GB)  [default]
  stanford-crfm/music-large-800k    (~760M params, ~3.0 GB)

GPUs disponibles (--gpu):
  T4       16 GB, $0.59/hr  — barato, válido para small/medium
  L4       24 GB, $0.80/hr  — equilibrado
  A10G     24 GB, $1.10/hr  — recomendado (default)
  A100-40GB 40 GB, $2.10/hr — máxima velocidad

Coste estimado por clip de 20s (A10G):
  music-small:  ~30s  → ~$0.009
  music-medium: ~60s  → ~$0.018
  music-large:  ~120s → ~$0.037

IMPORTANTE: este script tiene múltiples local_entrypoints. Siempre especificar ::main
para inferencia. modal run sin ::suffix falla cuando hay más de un entrypoint.

Setup (pre-descarga de pesos al Volume, opcional):
    modal run research_anticipatory_modal.py::setup
    modal run research_anticipatory_modal.py::setup --model stanford-crfm/music-large-800k

Inferencia (modo accompaniment):
    modal run research_anticipatory_modal.py::main \
        --input ../evaluation/anticipatory/test2/input_fixture.mid \
        --mode accompaniment --prompt-length 5 --clip-length 20 \
        --out ../evaluation/anticipatory/test2/generated.mid

    # con modelo grande y GPU T4:
    modal run research_anticipatory_modal.py::main \
        --model stanford-crfm/music-large-800k --gpu T4 \
        --input ../evaluation/anticipatory/test2/input_fixture.mid \
        --mode accompaniment --clip-length 20 \
        --out /tmp/generated_large.mid

    # 3 candidatos (multiplicity):
    modal run research_anticipatory_modal.py::main \
        --input ../evaluation/anticipatory/test2/input_fixture.mid \
        --mode accompaniment --multiplicity 3 \
        --out ../evaluation/anticipatory/test2/generated.mid

Inferencia (modo continuation):
    modal run research_anticipatory_modal.py::main \
        --input ../evaluation/anticipatory/test1/input_fixture.mid \
        --mode continuation --duration 20 \
        --out ../evaluation/anticipatory/test1/generated.mid

Recovery (si el cliente cae durante spawn+poll):
    modal run research_anticipatory_modal.py::recover

Referencias:
  Repo:  https://github.com/jthickstun/anticipation
  Paper: arXiv:2306.03128 (ICML 2023)
  Issue #18 (prompt_length): https://github.com/jthickstun/anticipation/issues/18
"""

import json
import os
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Volume — caché HuggingFace persistente entre ejecuciones (~400MB–3GB según modelo)
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("anticipatory-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"
HF_CACHE = f"{WEIGHTS_MOUNT}/hf_cache"

DEFAULT_MODEL = "stanford-crfm/music-medium-800k"
DEFAULT_GPU = os.environ.get("ANTICIPATORY_GPU", "A10G")
LAST_CALL_FILE = "/tmp/.anticipatory_last_call.json"

# ---------------------------------------------------------------------------
# Container image — PyTorch CUDA 12.1 + anticipation desde GitHub
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git"])  # requerido para pip install git+...
    .pip_install(
        "torch",
        "transformers>=4.44",
        "huggingface_hub>=0.21",
        "mido>=1.3",
        "pretty_midi>=0.2.10",
        "numpy",
        "psutil",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "pip install 'git+https://github.com/jthickstun/anticipation.git'",
    )
)

app = modal.App("amt-inference", image=image)


# ---------------------------------------------------------------------------
# Núcleo de inferencia — función regular (no Modal), llamada por run()
# ---------------------------------------------------------------------------
def _inference_impl(
    midi_bytes: bytes,
    model_name: str,
    mode: str,
    duration: int,
    prompt_length: int,
    clip_length: int,
    top_p: float,
    melody_instrument: int,
    seed: int,
    temperature: float = 1.0,
) -> bytes:
    """Carga el modelo, ejecuta continuation o accompaniment, retorna bytes MIDI."""
    import time

    import numpy as np
    import torch
    import transformers.safetensors_conversion as _sc
    from transformers import AutoModelForCausalLM

    _sc.auto_conversion = lambda *a, **kw: None  # silencia thread de conversión HF

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}  model={model_name}  mode={mode}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # AMT no es compatible con fp16
        low_cpu_mem_usage=True,
        use_safetensors=False,
        cache_dir=HF_CACHE,
    )
    model = model.to(device)
    model.eval()
    print(f"[timing] carga modelo: {time.time() - t0:.1f}s")

    # Escribir MIDI de entrada en archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        f.write(midi_bytes)
        input_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        output_path = f.name

    from anticipation.convert import events_to_midi, midi_to_events

    t1 = time.time()
    with torch.no_grad():
        if mode == "continuation":
            from anticipation.ops import clip, sort
            from anticipation.sample import generate

            events = midi_to_events(input_path)
            if not events:
                raise ValueError(
                    "El seed MIDI no contiene eventos. "
                    "Asegúrate de que el item seleccionado tiene notas MIDI."
                )
            # Usa prompt_length para la ventana de historia (no hardcoded 5s)
            history = clip(events, 0, prompt_length)
            if not history:
                raise ValueError(
                    f"El seed MIDI no tiene eventos en los primeros {prompt_length}s. "
                    f"Usa un item más largo o reduce prompt_length ({prompt_length}s)."
                )
            new_events = generate(model, prompt_length, prompt_length + duration,
                                  inputs=history, top_p=top_p, temperature=temperature)
            if not new_events:
                raise ValueError("AMT no generó ningún evento MIDI. Prueba con otro seed o aumenta duration.")
            # Solo exportar la continuación (sin la historia) para no solapar
            # la pista original al importar en REAPER. La historia ya existe.
            midi_out = events_to_midi(new_events)

        elif mode == "accompaniment":
            from anticipation.ops import clip, combine
            from anticipation.sample import generate
            from anticipation.tokenize import extract_instruments

            np.random.seed(seed)
            torch.manual_seed(seed)

            events = midi_to_events(input_path)
            events, controls = extract_instruments(events, [melody_instrument])

            if not events:
                print("[warning] sin eventos non-melody — AMT generará REST tokens.")

            prompt = clip(events, 0, prompt_length, clip_duration=False)
            generated = generate(
                model,
                prompt_length,
                clip_length,
                inputs=prompt,
                controls=controls,
                top_p=top_p,
            )
            output = clip(combine(generated, controls), 0, clip_length)
            midi_out = events_to_midi(output)

        else:
            raise ValueError(f"modo desconocido: {mode!r}")

    t_infer = time.time() - t1
    print(f"[timing] inferencia: {t_infer:.1f}s")

    midi_out.save(output_path)

    # Continuación: desplazar notas a t=0 (AMT genera desde prompt_length en adelante)
    if mode == "continuation":
        import pretty_midi
        pm_raw = pretty_midi.PrettyMIDI(output_path)
        pm_shifted = pretty_midi.PrettyMIDI()
        for instr in pm_raw.instruments:
            ni = pretty_midi.Instrument(program=instr.program, is_drum=instr.is_drum, name=instr.name)
            for note in instr.notes:
                ni.notes.append(pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=max(0.0, note.start - prompt_length),
                    end=max(0.001, note.end - prompt_length),
                ))
            if ni.notes:
                pm_shifted.instruments.append(ni)
        pm_shifted.write(output_path)
        n_notes = sum(len(i.notes) for i in pm_shifted.instruments)
        print(f"[continuation] desplazado -{prompt_length}s → {n_notes} notas, {len(pm_shifted.instruments)} pistas")

    # Persistir pesos descargados (best-effort: xet puede tener log files abiertos)
    try:
        weights_vol.commit()
    except Exception as e:
        print(f"[warning] weights_vol.commit() ignorado: {e}")

    # Análisis rápido del output
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(output_path)
        n_notes = sum(len(i.notes) for i in pm.instruments)
        print(f"[output] {len(pm.instruments)} pistas | {n_notes} notas | {pm.get_end_time():.1f}s")
    except Exception:
        pass

    with open(output_path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# GPU variant única — el tipo se fija vía DEFAULT_GPU (env var ANTICIPATORY_GPU) y se
# puede sobreescribir en runtime con run.with_options(gpu=...) desde main().
# ---------------------------------------------------------------------------
_FN_KWARGS = dict(cpu=2, memory=6144, timeout=3600, volumes={WEIGHTS_MOUNT: weights_vol})


@app.function(gpu=DEFAULT_GPU, **_FN_KWARGS)
def run(midi_bytes: bytes, model_name: str = DEFAULT_MODEL, mode: str = "accompaniment",
        duration: int = 10, prompt_length: int = 5, clip_length: int = 20,
        top_p: float = 0.95, melody_instrument: int = 0, seed: int = 0,
        temperature: float = 1.0) -> bytes:
    return _inference_impl(midi_bytes, model_name, mode, duration=duration,
                           prompt_length=prompt_length, clip_length=clip_length,
                           top_p=top_p, melody_instrument=melody_instrument, seed=seed,
                           temperature=temperature)


# ---------------------------------------------------------------------------
# Setup — pre-descarga del modelo al Volume (opcional)
# ---------------------------------------------------------------------------
@app.function(
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={WEIGHTS_MOUNT: weights_vol},
)
def download_model(model_name: str = DEFAULT_MODEL) -> None:
    """Descarga el modelo Anticipatory Music Transformer al Volume. Ejecutar una sola vez por modelo."""
    import time
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = HF_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)
    weights_vol.reload()

    print(f"Descargando {model_name} a {HF_CACHE}...")
    t0 = time.time()
    snapshot_download(
        repo_id=model_name,
        cache_dir=HF_CACHE,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
    )
    print(f"Descarga completada en {time.time() - t0:.0f}s")
    weights_vol.commit()
    print("Volume sincronizado.")


@app.local_entrypoint()
def setup(model: str = DEFAULT_MODEL) -> None:
    """Pre-descarga un modelo Anticipatory Music Transformer al Volume (evita cold-start en inferencia).

    Ejemplo:
        modal run research_anticipatory_modal.py::setup
        modal run research_anticipatory_modal.py::setup --model stanford-crfm/music-large-800k
    """
    print(f"Pre-descargando {model} al Volume 'anticipatory-weights'...")
    download_model.remote(model_name=model)
    print("Setup completado.")


# ---------------------------------------------------------------------------
# Entrypoint principal de inferencia
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input: str = "",
    out: str = "out_anticipatory.mid",
    mode: str = "accompaniment",
    model: str = DEFAULT_MODEL,
    gpu: str = "A10G",
    # continuation
    duration: int = 10,
    amt_temperature: float = 1.0,
    # accompaniment
    prompt_length: int = 5,
    clip_length: int = 20,
    top_p: float = 0.95,
    seed: int = 0,
    melody_instrument: int = 0,
    multiplicity: int = 1,
) -> None:
    """
    Genera MIDI con Anticipatory Music Transformer en Modal GPU.

    Ejemplos:
        modal run research_anticipatory_modal.py \\
            --input ../evaluation/anticipatory/test2/input_fixture.mid \\
            --mode accompaniment --clip-length 20 --out /tmp/generated.mid

        modal run research_anticipatory_modal.py \\
            --model stanford-crfm/music-large-800k --gpu L4 \\
            --input ../evaluation/anticipatory/test2/input_fixture.mid \\
            --mode accompaniment --multiplicity 3 \\
            --out ../evaluation/anticipatory/test2/generated.mid
    """
    import time

    _VALID_GPUS = {"T4", "L4", "A10G", "A100-40GB"}
    gpu_key = gpu.upper()
    if gpu_key not in _VALID_GPUS:
        raise SystemExit(f"[error] GPU desconocida: {gpu!r}. Opciones: {', '.join(sorted(_VALID_GPUS))}")

    if not input:
        raise SystemExit("[error] Debes especificar --input <path.mid>")

    input_path = Path(input)
    if not input_path.exists():
        raise SystemExit(f"[error] Archivo no encontrado: {input_path}")

    out_path = Path(out)
    midi_bytes = input_path.read_bytes()
    fn = run.with_options(gpu=gpu_key)

    mode_info = (
        f"prompt_length={prompt_length}s, clip_length={clip_length}s, "
        f"top_p={top_p}, seed={seed}"
        if mode == "accompaniment"
        else f"duration={duration}s, temp={amt_temperature}"
    )
    print(
        f"[modal] gpu={gpu_key}  model={model}  mode={mode}  {mode_info}"
        + (f"  multiplicity={multiplicity}" if multiplicity > 1 else "")
    )

    def _versioned(base: Path, i: int, total: int) -> Path:
        if total == 1:
            return base
        return base.with_stem(f"{base.stem}_v{i}")

    for i in range(multiplicity):
        out_i = _versioned(out_path, i, multiplicity)
        seed_i = seed + i

        if multiplicity > 1:
            print(f"\n[candidate {i}] seed={seed_i}  →  {out_i}")

        call = fn.spawn(
            midi_bytes=midi_bytes,
            model_name=model,
            mode=mode,
            duration=duration,
            prompt_length=prompt_length,
            clip_length=clip_length,
            top_p=top_p,
            melody_instrument=melody_instrument,
            seed=seed_i,
            temperature=amt_temperature,
        )
        print(f"[spawned] object_id={call.object_id}")

        # Persistir estado para recovery si el cliente cae
        _state = {
            "call_id": call.object_id,
            "out": str(out_i),
            "gpu": gpu_key,
            "model": model,
            "candidate": i,
        }
        Path(LAST_CALL_FILE).write_text(json.dumps(_state, indent=2))

        # Poll cada 30s (dentro de la ventana de caché de 5min)
        result_bytes = None
        dots = 0
        t_start = time.time()
        while result_bytes is None:
            try:
                result_bytes = call.get(timeout=30)
            except TimeoutError:
                print(".", end="", flush=True)
                dots += 1
            except Exception as e:
                msg = str(e).lower()
                if any(k in msg for k in ("deadline", "timed out")):
                    print(".", end="", flush=True)
                    dots += 1
                else:
                    raise
        if dots:
            print()

        t_total = time.time() - t_start
        out_i.write_bytes(result_bytes)
        print(f"[saved] {out_i.resolve()}  ({len(result_bytes) / 1024:.1f} KB, {t_total:.0f}s)")

        _report_midi(out_i)

    # Limpiar archivo de recovery al completar
    Path(LAST_CALL_FILE).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Recovery — retoma un spawn huérfano
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def recover(call_id: str = "", out: str = "") -> None:
    """
    Recupera el resultado de un spawn() previo que quedó huérfano.
    Sin argumentos lee .anticipatory_last_call.json del directorio actual.

    Ejemplo:
        modal run research_anticipatory_modal.py::recover
        modal run research_anticipatory_modal.py::recover --call-id fc-XXXX --out /tmp/out.mid
    """
    import time

    if not call_id:
        state_file = Path(LAST_CALL_FILE)
        if not state_file.exists():
            print("[recover] No hay /tmp/.anticipatory_last_call.json y no se pasó --call-id")
            return
        state = json.loads(state_file.read_text())
        call_id = state["call_id"]
        if not out:
            out = state.get("out", "out_recovered.mid")
        print(f"[recover] Leyendo estado: call_id={call_id}  out={out}")
    else:
        print(f"[recover] call_id={call_id}  out={out or 'out_recovered.mid'}")

    out_path = Path(out or "out_recovered.mid")
    call = modal.functions.FunctionCall.from_id(call_id)

    result_bytes = None
    dots = 0
    t_start = time.time()
    while result_bytes is None:
        try:
            result_bytes = call.get(timeout=30)
        except TimeoutError:
            print(".", end="", flush=True)
            dots += 1
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("deadline", "timed out")):
                print(".", end="", flush=True)
                dots += 1
            else:
                raise
    if dots:
        print()

    out_path.write_bytes(result_bytes)
    print(f"[saved] {out_path.resolve()}  ({len(result_bytes) / 1024:.1f} KB, {time.time() - t_start:.0f}s)")
    _report_midi(out_path)
    Path(LAST_CALL_FILE).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Utilidad local — muestra resumen del MIDI generado
# ---------------------------------------------------------------------------
def _report_midi(path: Path) -> None:
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(path))
        n_notes = sum(len(i.notes) for i in pm.instruments)
        instrs = [(i.program, i.is_drum) for i in pm.instruments]
        print(f"  pistas: {len(pm.instruments)}  notas: {n_notes}  "
              f"duración: {pm.get_end_time():.1f}s  instrs: {instrs}")
    except ImportError:
        pass
    except Exception as e:
        print(f"  (no se pudo analizar MIDI: {e})")
