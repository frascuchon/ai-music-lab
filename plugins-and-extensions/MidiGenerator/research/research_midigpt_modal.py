"""
Prueba de investigación: MIDI-GPT (Metacreation Lab, AAAI 2025) en Modal.

Objetivo: comprobar si MIDI-GPT genera un acompañamiento razonable para una
melodía MIDI (pista nueva condicionada por la melodía) y compararlo con Anticipatory.

Requisitos: haber aceptado la licencia de https://huggingface.co/Metacreation/MIDI-GPT
(pesos de uso no comercial) y tener el secret `huggingface-secret` (HF_TOKEN) en Modal.

    modal run research_midigpt_modal.py::api                     # imprime la API real
    modal run research_midigpt_modal.py::main --input melody.mid --out out.mid
"""

import os
from pathlib import Path

import modal

weights_vol = modal.Volume.from_name("midigpt-weights", create_if_missing=True)
HF_CACHE = "/vol/weights/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("midigpt[inference]", "huggingface_hub", "pretty_midi", "mido", "numpy")
)
app = modal.App("midigpt-research", image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(volumes={"/vol/weights": weights_vol}, secrets=[hf_secret], timeout=900)
def inspect_api() -> str:
    import inspect

    import midigpt
    from midigpt import Score
    from midigpt.inference import GenerationRequest, InferenceConfig, InferenceEngine, TrackPrompt

    os.environ["HF_HOME"] = HF_CACHE
    out = [f"midigpt version: {getattr(midigpt, '__version__', '?')}", f"dir(midigpt): {dir(midigpt)}"]
    for obj in (Score, TrackPrompt, GenerationRequest, InferenceConfig, InferenceEngine):
        out.append(f"\n=== {obj.__name__} ===\n{inspect.signature(obj) if callable(obj) else ''}")
        out.append((inspect.getdoc(obj) or "")[:1500])
        out.append("public: " + ", ".join(n for n in dir(obj) if not n.startswith("_")))
    # does the download work with this token?
    try:
        engine = InferenceEngine.from_pretrained("yellow_medium")
        out.append("\nfrom_pretrained('yellow_medium'): OK")
        out.append("engine public: " + ", ".join(n for n in dir(engine) if not n.startswith("_")))
    except Exception as exc:  # noqa: BLE001
        out.append(f"\nfrom_pretrained FAILED: {type(exc).__name__}: {exc}")
    weights_vol.commit()
    return "\n".join(out)


@app.function(volumes={"/vol/weights": weights_vol}, secrets=[hf_secret], timeout=900)
def inspect_types() -> str:
    import inspect

    import midigpt
    from midigpt import Bar, Note, Score, Track
    from midigpt.inference import InferenceConfig

    out = []
    for obj in (Track, Bar, Note):
        out.append(f"=== {obj.__name__} ===\n{inspect.signature(obj)}\n{(inspect.getdoc(obj) or '')[:800]}")
        out.append("public: " + ", ".join(n for n in dir(obj) if not n.startswith("_")))
    out.append("\n=== midigpt.attributes ===\n" + ", ".join(n for n in dir(midigpt.attributes) if not n.startswith("_")))
    for n in dir(midigpt.attributes):
        if not n.startswith("_"):
            try:
                out.append(f"{n}: {inspect.getdoc(getattr(midigpt.attributes, n))[:300] if inspect.getdoc(getattr(midigpt.attributes, n)) else getattr(midigpt.attributes, n)}")
            except Exception:
                pass
    out.append("\n=== InferenceConfig doc ===\n" + (inspect.getdoc(InferenceConfig) or ""))
    out.append("\n=== InferenceConfig fields ===\n" + str(inspect.signature(InferenceConfig)))
    out.append("\n=== Score.from_midi doc ===\n" + str(inspect.getdoc(Score.from_midi)))
    out.append("\n=== InferenceEngine.session doc ===\n" + str(inspect.getdoc(__import__("midigpt.inference", fromlist=["InferenceEngine"]).InferenceEngine.session)))
    return "\n".join(out)


@app.function(volumes={"/vol/weights": weights_vol}, secrets=[hf_secret], timeout=900)
def model_attributes(model: str = "yellow_medium") -> str:
    """Which attributes the checkpoint understands and their value ranges."""
    import json

    os.environ["HF_HOME"] = HF_CACHE
    from midigpt.attributes import ATTRIBUTE_REGISTRY
    from midigpt.inference import InferenceEngine

    out = [f"registry: {sorted(ATTRIBUTE_REGISTRY)}"]
    engine = InferenceEngine.from_pretrained(model)
    out.append("engine fields: " + ", ".join(vars(engine)))
    for name, holder in vars(engine).items():
        if name.endswith("model"):
            continue
        out.append(f"\n## engine.{name} ({type(holder).__name__})")
        for n in dir(holder):
            if n.startswith("__"):
                continue
            try:
                v = getattr(holder, n)
            except Exception:
                continue
            if not callable(v):
                out.append(f"  {n} = {str(v)[:900]}")
            else:
                out.append(f"  {n}()")
    return "\n".join(out)


@app.function(volumes={"/vol/weights": weights_vol}, secrets=[hf_secret], timeout=900)
def dump_source(rel: str = "inference/config.py", start: int = 0, count: int = 400) -> str:
    """Print a slice of a file of the installed midigpt package (to read semantics, not guess them)."""
    import pathlib

    import midigpt

    root = pathlib.Path(midigpt.__file__).parent
    if rel == "?":
        return "\n".join(sorted(str(p.relative_to(root)) for p in root.rglob("*.py")))
    lines = (root / rel).read_text().splitlines()
    return "\n".join(f"{i + start + 1:4d} {l}" for i, l in enumerate(lines[start:start + count]))


BASS_PROGRAM = 33     # electric bass (finger)
KEYS_PROGRAM = 4      # electric piano 1

# One entry per generated part. `attributes` are MIDI-GPT track attributes (quantized
# levels 0-9: note_density, min/max_polyphony, min/max_note_duration); `controls` are
# hard/soft constraints (pitch_mask register + pitch classes, rhythm_mask grid).
DEFAULT_SPEC = {
    "temperature": 1.0,
    "pitch_classes": False,                       # restrict to the melody's pitch classes
    "tracks": [
        {"role": "Bass", "program": BASS_PROGRAM, "register": [33, 52], "attributes": {}, "grid": None},
        {"role": "Keys", "program": KEYS_PROGRAM, "register": [52, 76], "attributes": {}, "grid": None},
    ],
}


def _generate(engine, midi_bytes: bytes, spec: dict, seed: int) -> bytes:
    """Melody -> the parts described by `spec`, as MIDI containing ONLY the new tracks, on the
    melody's original timeline."""
    import io
    import tempfile

    import pretty_midi
    from midigpt import Bar, Score, Track
    from midigpt.inference import GenerationRequest, InferenceConfig, TrackPrompt

    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    bpm = pm.get_tempo_changes()[1][0]
    melody = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    lead = min(n.start for n in melody)
    # leading silence is trimmed (bars would otherwise start with empty music) and restored later
    trimmed = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=80, name="Melody")      # 80 = square lead
    for n in melody:
        inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                           start=n.start - lead, end=n.end - lead))
    trimmed.instruments.append(inst)
    pcs = sorted({n.pitch % 12 for n in melody})
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        trimmed.write(f.name)
        path = f.name

    score = Score.from_midi(path)
    n_bars = len(score.tracks[0].bars)
    prompts = [TrackPrompt(id=0, bars=[])]                              # melody = context
    for i, t in enumerate(spec["tracks"], start=1):
        drum = t.get("type") == "drum"
        score.tracks.append(Track(bars=[Bar(notes=[]) for _ in range(n_bars)],
                                  instrument=t["program"], track_type="drum" if drum else "melodic"))
        controls = {}
        if not drum:
            lo, hi = t["register"]
            mask = {"shape": {"type": "uniform", "min": lo, "max": hi}}
            if spec.get("pitch_classes"):
                mask["pitch_classes"] = pcs
            controls["pitch_mask"] = mask
        if t.get("rhythm"):                    # exact onset schedule within the bar
            controls["rhythm_mask"] = {"positions": t["rhythm"]}
        elif t.get("grid"):
            controls["rhythm_mask"] = {"grid": {"unit": t["grid"], "strength": 0.9}}
        prompts.append(TrackPrompt(id=i, bars=list(range(n_bars)), autoregressive=True,
                                   attributes=dict(t.get("attributes") or {}), controls=controls))

    request = GenerationRequest(
        tracks=prompts,
        config=InferenceConfig(temperature=spec.get("temperature", 1.0), seed=seed, model_dim=4,
                               tracks_per_step=1),
    )
    result = engine.session(score, request).run()
    out_path = path + ".out.mid"
    result.to_midi(out_path)

    melody_end = max(n.end for n in melody) - lead
    gen = pretty_midi.PrettyMIDI(out_path)
    final = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    roles = {t["program"]: t["role"] for t in spec["tracks"] if t.get("type") != "drum"}
    for inst in gen.instruments:
        if not inst.is_drum and inst.program == 80:
            continue                                     # the melody itself
        new = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum,
                                     name="Drums" if inst.is_drum else roles.get(inst.program, inst.name))
        for n in inst.notes:
            if n.start > melody_end + 0.5:       # do not outlast the melody
                continue
            new.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                              start=n.start + lead,
                                              end=min(n.end, melody_end + 0.5) + lead))
        if new.notes:
            final.instruments.append(new)
    buf = io.BytesIO()
    final.write(buf)
    return buf.getvalue()


@app.function(volumes={"/vol/weights": weights_vol}, secrets=[hf_secret], timeout=3000, gpu="A10G")
def accompany(midi_bytes: bytes, jobs: list, model: str = "yellow_medium") -> list:
    """Run several (label, spec, seed) jobs against ONE loaded model.
    Returns [(label, seed, midi_bytes)]."""
    from midigpt.inference import InferenceEngine

    os.environ["HF_HOME"] = HF_CACHE
    engine = InferenceEngine.from_pretrained(model)
    enc = getattr(engine._tokenizer, "_encoder", None)
    print("[midigpt] encoder fields:", [n for n in dir(enc) if not n.startswith("_")][:40] if enc else None)
    for n in ("resolution", "ticks_per_quarter", "config"):
        if enc is not None and hasattr(enc, n):
            print(f"[midigpt] encoder.{n} =", str(getattr(enc, n))[:300])
    out = []
    for label, spec, seed in jobs:
        try:
            out.append((label, seed, _generate(engine, midi_bytes, spec, seed)))
            print(f"[midigpt] {label} seed={seed} ok")
        except Exception as exc:  # noqa: BLE001
            print(f"[midigpt] {label} seed={seed} FAILED: {type(exc).__name__}: {exc}")
    weights_vol.commit()
    return out


@app.local_entrypoint()
def main(input: str = "", out: str = "out_midigpt.mid", model: str = "yellow_medium",
         seed: int = -1, spec: str = "", multiplicity: int = 1):
    """--spec is a JSON file with the parts (see DEFAULT_SPEC); defaults to bass + keys."""
    import json
    import random

    the_spec = json.loads(Path(spec).read_text()) if spec else DEFAULT_SPEC
    base = seed if seed >= 0 else random.randrange(2**31)
    jobs = [("candidate", the_spec, base + i) for i in range(multiplicity)]
    for i, (_, sd, data) in enumerate(accompany.remote(Path(input).read_bytes(), jobs, model)):
        target = Path(out) if multiplicity == 1 else Path(out).with_stem(f"{Path(out).stem}_v{i}")
        target.write_bytes(data)
        print(f"[saved] {target} (seed {sd})")


@app.local_entrypoint()
def sweep(input: str = "", outdir: str = "sweep", model: str = "yellow_medium", seeds: int = 2,
          configs: str = ""):
    """Run every named config in the JSON file `configs` ({name: spec}) with `seeds` seeds."""
    import json

    cfgs = json.loads(Path(configs).read_text())
    jobs = [(name, spec, sd) for name, spec in cfgs.items() for sd in range(seeds)]
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    for label, sd, data in accompany.remote(Path(input).read_bytes(), jobs, model):
        (d / f"{label}_s{sd}.mid").write_bytes(data)
    print(f"[saved] {len(jobs)} jobs -> {d}")


@app.local_entrypoint()
def api(types: bool = False, src: str = "", start: int = 0, count: int = 400, attrs: str = ""):
    if attrs:
        print(model_attributes.remote(attrs))
    elif src:
        print(dump_source.remote(src, start, count))
    else:
        print((inspect_types if types else inspect_api).remote())
