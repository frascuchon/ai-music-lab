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
    modal run research_amt_modal.py::setup
    modal run research_amt_modal.py::setup --model stanford-crfm/music-large-800k

Inferencia (modo accompaniment):
    modal run research_amt_modal.py::main \
        --input ../evaluation/amt/test2/input_fixture.mid \
        --mode accompaniment --prompt-length 5 --clip-length 20 \
        --out ../evaluation/amt/test2/generated.mid

    # con modelo grande y GPU T4:
    modal run research_amt_modal.py::main \
        --model stanford-crfm/music-large-800k --gpu T4 \
        --input ../evaluation/amt/test2/input_fixture.mid \
        --mode accompaniment --clip-length 20 \
        --out /tmp/generated_large.mid

    # 3 candidatos (multiplicity):
    modal run research_amt_modal.py::main \
        --input ../evaluation/amt/test2/input_fixture.mid \
        --mode accompaniment --multiplicity 3 \
        --out ../evaluation/amt/test2/generated.mid

Inferencia (modo continuation):
    modal run research_amt_modal.py::main \
        --input ../evaluation/amt/test1/input_fixture.mid \
        --mode continuation --duration 20 \
        --out ../evaluation/amt/test1/generated.mid

Recovery (si el cliente cae durante spawn+poll):
    modal run research_amt_modal.py::recover

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
weights_vol = modal.Volume.from_name("amt-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"
HF_CACHE = f"{WEIGHTS_MOUNT}/hf_cache"

DEFAULT_MODEL = "stanford-crfm/music-medium-800k"
LAST_CALL_FILE = ".amt_last_call.json"

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
# Núcleo de inferencia — función regular (no Modal), llamada por las variantes GPU
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
            history = clip(events, 0, 5)
            new_events = generate(model, 5, 5 + duration, inputs=history, top_p=0.98)
            combined = sort(history + new_events)
            midi_out = events_to_midi(combined)

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
# GPU variants — factory que registra la misma firma con distintos GPU specs.
# Modal necesita funciones en scope de módulo con nombres distintos; se asignan
# via __name__/__qualname__ antes de pasar a app.function().
# ---------------------------------------------------------------------------
_FN_KWARGS = dict(cpu=2, memory=6144, timeout=3600, volumes={WEIGHTS_MOUNT: weights_vol})


def _make_gpu_fn(gpu_str: str):
    fn_name = "run_" + gpu_str.lower().replace("-", "_")

    def _wrapper(midi_bytes: bytes, model_name: str = DEFAULT_MODEL, mode: str = "accompaniment",
                 duration: int = 10, prompt_length: int = 5, clip_length: int = 20,
                 top_p: float = 0.95, melody_instrument: int = 0, seed: int = 0) -> bytes:
        return _inference_impl(midi_bytes, model_name, mode, duration=duration,
                               prompt_length=prompt_length, clip_length=clip_length,
                               top_p=top_p, melody_instrument=melody_instrument, seed=seed)

    _wrapper.__name__ = fn_name
    _wrapper.__qualname__ = fn_name
    return app.function(gpu=gpu_str, **_FN_KWARGS)(_wrapper)


run_t4        = _make_gpu_fn("T4")
run_l4        = _make_gpu_fn("L4")
run_a10g      = _make_gpu_fn("A10G")
run_a100_40gb = _make_gpu_fn("A100-40GB")

_gpu_fns = {"T4": run_t4, "L4": run_l4, "A10G": run_a10g, "A100-40GB": run_a100_40gb}


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
    """Descarga el modelo AMT al Volume. Ejecutar una sola vez por modelo."""
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
    """Pre-descarga un modelo AMT al Volume (evita cold-start en inferencia).

    Ejemplo:
        modal run research_amt_modal.py::setup
        modal run research_amt_modal.py::setup --model stanford-crfm/music-large-800k
    """
    print(f"Pre-descargando {model} al Volume 'amt-weights'...")
    download_model.remote(model_name=model)
    print("Setup completado.")


# ---------------------------------------------------------------------------
# Entrypoint principal de inferencia
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input: str = "",
    out: str = "out_amt.mid",
    mode: str = "accompaniment",
    model: str = DEFAULT_MODEL,
    gpu: str = "A10G",
    # continuation
    duration: int = 10,
    # accompaniment
    prompt_length: int = 5,
    clip_length: int = 20,
    top_p: float = 0.95,
    seed: int = 0,
    melody_instrument: int = 0,
    multiplicity: int = 1,
) -> None:
    """
    Genera MIDI con AMT en Modal GPU.

    Ejemplos:
        modal run research_amt_modal.py \\
            --input ../evaluation/amt/test2/input_fixture.mid \\
            --mode accompaniment --clip-length 20 --out /tmp/generated.mid

        modal run research_amt_modal.py \\
            --model stanford-crfm/music-large-800k --gpu L4 \\
            --input ../evaluation/amt/test2/input_fixture.mid \\
            --mode accompaniment --multiplicity 3 \\
            --out ../evaluation/amt/test2/generated.mid
    """
    import time

    gpu_key = gpu.upper()
    if gpu_key not in _gpu_fns:
        valid = ", ".join(_gpu_fns.keys())
        raise SystemExit(f"[error] GPU desconocida: {gpu!r}. Opciones: {valid}")

    if not input:
        raise SystemExit("[error] Debes especificar --input <path.mid>")

    input_path = Path(input)
    if not input_path.exists():
        raise SystemExit(f"[error] Archivo no encontrado: {input_path}")

    out_path = Path(out)
    midi_bytes = input_path.read_bytes()
    fn = _gpu_fns[gpu_key]

    mode_info = (
        f"prompt_length={prompt_length}s, clip_length={clip_length}s, "
        f"top_p={top_p}, seed={seed}"
        if mode == "accompaniment"
        else f"duration={duration}s"
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
    Sin argumentos lee .amt_last_call.json del directorio actual.

    Ejemplo:
        modal run research_amt_modal.py::recover
        modal run research_amt_modal.py::recover --call-id fc-XXXX --out /tmp/out.mid
    """
    import time

    if not call_id:
        state_file = Path(LAST_CALL_FILE)
        if not state_file.exists():
            print("[recover] No hay .amt_last_call.json y no se pasó --call-id")
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
