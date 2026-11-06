"""
Modal.com inference app para Anticipatory Music Transformer (Stanford CRFM, Apache 2.0).

Genera un acompañamiento para una melodía MIDI (canal 0). No hace falta aportar
acompañamiento previo: si el MIDI solo trae la melodía, se construye
automáticamente un "prompt" de bajo + acordes con harmony.py (ver accompaniment.py).

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

Los pesos se descargan automáticamente (y se cachean en el Volume,
best-effort commit — ver _inference_impl) en la primera llamada real; no
requiere un paso de setup manual.

Inferencia:
    modal run research_anticipatory_modal.py::main \
        --input melody.mid --clip-length 0 --out generated.mid

    # estilo del prompt automático (pop | pad | arp) y 3 candidatos:
    modal run research_anticipatory_modal.py::main \
        --input melody.mid --style arp --multiplicity 3 --out generated.mid

    # con acompañamiento propio en otros canales (sin prompt automático):
    modal run research_anticipatory_modal.py::main \
        --input melody_y_semilla.mid --no-auto-history --out generated.mid

    # modelo grande y GPU T4:
    modal run research_anticipatory_modal.py::main \
        --model stanford-crfm/music-large-800k --gpu T4 --input melody.mid --out generated.mid

    --clip-length 0 (por defecto en el plugin) = duración de la melodía.

Recovery (si el cliente cae durante spawn+poll):
    modal run research_anticipatory_modal.py::recover

Referencias:
  Repo:  https://github.com/jthickstun/anticipation
  Paper: arXiv:2306.08620 (TMLR)
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
# Exact version of every package this model's image contains (captured from the image
# that works; see tools/modal_freeze.py). Applied to every pip call through
# PIP_CONSTRAINT so a rebuild never drifts to newer releases of torch & co.
_CONSTRAINTS = (Path(__file__).resolve().parent.parent.parent / "shared" / "modal_constraints"
                / "research_anticipatory_modal.txt")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(str(_CONSTRAINTS), "/opt/modal_constraints.txt", copy=True)
    .env({"PIP_CONSTRAINT": "/opt/modal_constraints.txt"})
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
        "pip install 'git+https://github.com/jthickstun/anticipation.git@af37397922665a0fb8d474d7988b0f3755a38d45'",
    )
    # Melody-only accompaniment: automatic prompt + channel->instrument mapping
    .add_local_file(str(Path(__file__).resolve().parent.parent / "harmony.py"),
                    "/root/harmony.py")
    .add_local_file(str(Path(__file__).resolve().parent.parent / "accompaniment.py"),
                    "/root/accompaniment.py")
)

app = modal.App("amt-inference", image=image)


def _arrangement_kwargs(a: dict | None) -> dict:
    """JSON-friendly arrangement dict -> harmony.Arrangement kwargs."""
    a = a or {}
    kw = {}
    if a.get("parts"):
        kw["parts"] = tuple(a["parts"])
    if "density" in a:
        kw["density"] = int(a["density"])
    if a.get("programs"):
        kw["programs"] = tuple((k, int(v)) for k, v in a["programs"].items())
    if a.get("chords"):
        kw["chords"] = a["chords"]
    if a.get("beats_per_chord"):
        kw["beats_per_chord"] = int(a["beats_per_chord"])
    return kw


# ---------------------------------------------------------------------------
# Núcleo de inferencia — función regular (no Modal), llamada por run()
# ---------------------------------------------------------------------------
def _inference_impl(
    midi_bytes: bytes,
    model_name: str,
    prompt_length: int,
    clip_length: int,
    top_p: float,
    melody_instrument: int,
    seed: int,
    auto_history: bool = True,
    style: str = "pop",
    arrangement: dict | None = None,
) -> bytes:
    """Carga el modelo, genera el acompañamiento y retorna bytes MIDI (solo el acompañamiento)."""
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
    print(f"[info] device={device}  model={model_name}")

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
        from anticipation.ops import clip, combine
        from anticipation.sample import generate
        from anticipation.tokenize import extract_instruments

        import sys
        sys.path.insert(0, "/root")
        import accompaniment

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Channel 0 = melody (-> melody_instrument), other channels = accompaniment
        # history; builds the history from the melody when there is none.
        import harmony
        arr = harmony.Arrangement(style=style, **_arrangement_kwargs(arrangement))
        prep = accompaniment.prepare(midi_bytes, melody_instrument, prompt_length,
                                     auto_history=auto_history, style=style, arrangement=arr)
        lead, melody_end = prep.lead, prep.melody_end
        with open(input_path, "wb") as f:
            f.write(prep.midi)
        if prep.auto_history:
            print(f"[prompt] acompañamiento automático ({style}, partes: {', '.join(arr.parts)}, "
                  f"densidad {arr.density}): {prep.harmony}")
        if lead > 0.05:
            print(f"[input] silencio inicial recortado: {lead:.2f}s")

        # A short melody cannot supply a full prompt plus something to generate.
        eff_prompt = min(float(prompt_length), max(2.0, melody_end * 0.35))
        eff_clip = clip_length if clip_length > 0 else int(melody_end) + 2
        eff_clip = max(eff_clip, int(eff_prompt) + 2)
        print(f"[input] melodía {melody_end:.1f}s  prompt={eff_prompt:.1f}s  clip={eff_clip}s")

        events = midi_to_events(input_path)
        events, controls = extract_instruments(events, [melody_instrument])

        if not events:
            print("[warning] sin eventos non-melody — AMT generará REST tokens.")

        prompt = clip(events, 0, eff_prompt, clip_duration=False)
        generated = generate(
            model,
            eff_prompt,
            eff_clip,
            inputs=prompt,
            controls=controls,
            top_p=top_p,
        )
        output = clip(combine(generated, controls), 0, eff_clip)
        midi_out = events_to_midi(output)

    t_infer = time.time() - t1
    print(f"[timing] inferencia: {t_infer:.1f}s")

    midi_out.save(output_path)

    # Only the accompaniment goes back (the user already has the melody),
    # shifted to the melody's original start time and not outlasting it.
    with open(output_path, "rb") as f:
        final = accompaniment.finalize(f.read(), lead, melody_instrument,
                                       trim_after=melody_end + 0.5)
    with open(output_path, "wb") as f:
        f.write(final)

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
def run(midi_bytes: bytes, model_name: str = DEFAULT_MODEL,
        prompt_length: int = 5, clip_length: int = 0,
        top_p: float = 0.95, melody_instrument: int = 0, seed: int = 0,
        auto_history: bool = True, style: str = "pop", arrangement: dict | None = None) -> bytes:
    return _inference_impl(midi_bytes, model_name,
                           prompt_length=prompt_length, clip_length=clip_length,
                           top_p=top_p, melody_instrument=melody_instrument, seed=seed,
                           auto_history=auto_history, style=style, arrangement=arrangement)


# ---------------------------------------------------------------------------
# Entrypoint principal de inferencia
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input: str = "",
    out: str = "out_anticipatory.mid",
    model: str = DEFAULT_MODEL,
    gpu: str = "A10G",
    prompt_length: int = 5,
    clip_length: int = 0,
    top_p: float = 0.95,
    seed: int = -1,
    melody_instrument: int = 0,
    multiplicity: int = 1,
    auto_history: bool = True,
    style: str = "pop",
    parts: str = "bass,keys",
    density: int = 1,
    bass_program: int = 33,
    keys_program: int = 4,
    chords: str = "",
    beats_per_chord: int = 4,
) -> None:
    """
    Genera MIDI con Anticipatory Music Transformer en Modal GPU.

    Ejemplos:
        modal run research_anticipatory_modal.py \\
            --input melody.mid --out /tmp/generated.mid

        modal run research_anticipatory_modal.py \\
            --model stanford-crfm/music-large-800k --gpu L4 \\
            --input melody.mid --multiplicity 3 --out /tmp/generated.mid

    seed < 0 (por defecto) elige una semilla aleatoria; con una semilla fija el
    resultado es reproducible. El candidato i usa la semilla base + i.
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

    if seed < 0:
        import random
        seed = random.randrange(2**31)
    arrangement = {
        "parts": [p.strip() for p in parts.split(",") if p.strip()],
        "density": density,
        "programs": {"bass": bass_program, "keys": keys_program},
        "chords": chords,
        "beats_per_chord": beats_per_chord,
    }
    print(
        f"[modal] gpu={gpu_key}  model={model}  style={style}  "
        f"prompt_length={prompt_length}s, clip_length={clip_length}s, top_p={top_p}, seed={seed}"
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
            prompt_length=prompt_length,
            clip_length=clip_length,
            top_p=top_p,
            melody_instrument=melody_instrument,
            seed=seed_i,
            auto_history=auto_history,
            style=style,
            arrangement=arrangement,
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
