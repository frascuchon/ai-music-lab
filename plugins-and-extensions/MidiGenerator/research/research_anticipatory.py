"""
Acompañamiento local (CPU) con Anticipatory Music Transformer (Stanford CRFM).

Misma lógica que research_anticipatory_modal.py, sin Modal: el canal 0 del MIDI de
entrada es la melodía y, si no hay acompañamiento previo, se construye un "prompt"
automático de bajo + acordes (ver ../accompaniment.py y ../harmony.py).

Uso:
    uv run research_anticipatory.py --input melody.mid --out out_acc.mid
    uv run research_anticipatory.py --input melody.mid --style arp --multiplicity 3

En CPU es lento (≈16 min para 20 s de música); para uso real, Modal.

Referencia: https://github.com/jthickstun/anticipation
Modelo HF: stanford-crfm/music-medium-800k (360M params, Apache 2.0)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "stanford-crfm/music-medium-800k"


def run_accompaniment(
    model,
    input_midi_path: str,
    out_path: str,
    prompt_length: int = 5,
    clip_length: int = 0,
    top_p: float = 0.95,
    melody_instrument: int = 0,
    seed: int = 0,
    auto_history: bool = True,
    style: str = "pop",
):
    """Genera el acompañamiento de la melodía (canal 0) y guarda solo el acompañamiento."""
    import tempfile

    import numpy as np
    import torch
    from anticipation.convert import events_to_midi, midi_to_events
    from anticipation.ops import clip, combine
    from anticipation.sample import generate
    from anticipation.tokenize import extract_instruments

    import accompaniment

    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(input_midi_path, "rb") as f:
        prep = accompaniment.prepare(f.read(), melody_instrument, prompt_length,
                                     auto_history=auto_history, style=style)
    if prep.auto_history:
        print(f"[prompt] acompañamiento automático ({style}): {prep.harmony}")

    eff_prompt = min(float(prompt_length), max(2.0, prep.melody_end * 0.35))
    eff_clip = clip_length if clip_length > 0 else int(prep.melody_end) + 2
    eff_clip = max(eff_clip, int(eff_prompt) + 2)
    print(f"[accompaniment] melodía {prep.melody_end:.1f}s  prompt={eff_prompt:.1f}s  "
          f"clip={eff_clip}s  top_p={top_p}  seed={seed}")

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp.write(prep.midi)
        prepared_path = tmp.name
    try:
        events = midi_to_events(prepared_path)
    finally:
        os.unlink(prepared_path)
    events, controls = extract_instruments(events, [melody_instrument])
    if not events:
        print("[warning] sin eventos non-melody — AMT generará REST tokens.")

    prompt = clip(events, 0, eff_prompt, clip_duration=False)
    t0 = time.time()
    generated = generate(model, eff_prompt, eff_clip, inputs=prompt, controls=controls, top_p=top_p)
    t_infer = time.time() - t0
    print(f"[timing] inferencia: {t_infer:.1f}s")

    mid = events_to_midi(clip(combine(generated, controls), 0, eff_clip))
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        raw_path = tmp.name
    try:
        mid.save(raw_path)
        with open(raw_path, "rb") as f:
            final = accompaniment.finalize(f.read(), prep.lead, melody_instrument,
                                           trim_after=prep.melody_end + 0.5)
    finally:
        os.unlink(raw_path)
    with open(out_path, "wb") as f:
        f.write(final)
    print(f"[output] MIDI guardado en: {os.path.abspath(out_path)}")
    return t_infer


def _versioned_path(base_path: str, idx: int, total: int) -> str:
    """Retorna base_path si total==1, o base_path con sufijo _v{idx} si total>1."""
    if total == 1:
        return base_path
    root, ext = os.path.splitext(base_path)
    return f"{root}_v{idx}{ext}"


def main():
    parser = argparse.ArgumentParser(description="Acompañamiento local con Anticipatory Music Transformer")
    parser.add_argument("--input", required=True, help="MIDI con la melodía en el canal 0")
    parser.add_argument("--out", default="out_acc.mid")
    parser.add_argument("--style", choices=["pop", "pad", "arp"], default="pop",
                        help="Estilo del prompt automático de bajo + acordes (default: pop)")
    parser.add_argument("--prompt-length", type=int, default=5,
                        help="Segundos de acompañamiento usados como prompt (default: 5)")
    parser.add_argument("--clip-length", type=int, default=0,
                        help="Duración total en segundos; 0 = la de la melodía (default: 0)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling (default: 0.95)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Semilla; negativa = aleatoria en cada ejecución (default: -1)")
    parser.add_argument("--melody-instrument", type=int, default=0,
                        help="Programa MIDI asignado a la melodía (default: 0)")
    parser.add_argument("--multiplicity", type=int, default=1,
                        help="Número de candidatos (default: 1). Cada uno usa seed+i; "
                             "salidas out_acc_v0.mid, out_acc_v1.mid, ...")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[error] archivo MIDI de entrada no encontrado: {args.input}")
        raise SystemExit(1)

    import random

    import torch
    import transformers.safetensors_conversion as _sc
    from transformers import AutoModelForCausalLM
    _sc.auto_conversion = lambda *a, **kw: None  # silencia el thread de conversión HF

    seed = args.seed if args.seed >= 0 else random.randrange(2**31)

    # AMT requiere float32 (en fp16 genera tokens fuera de vocabulario)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}  dtype=float32  (en CPU es lento)")

    t0 = time.time()
    print(f"[setup] cargando {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=False,
    ).to(device)
    model.eval()
    print(f"[timing] carga modelo: {time.time() - t0:.1f}s")

    with torch.no_grad():
        for i in range(args.multiplicity):
            run_accompaniment(
                model, args.input, _versioned_path(args.out, i, args.multiplicity),
                prompt_length=args.prompt_length, clip_length=args.clip_length,
                top_p=args.top_p, melody_instrument=args.melody_instrument,
                seed=seed + i, style=args.style,
            )


if __name__ == "__main__":
    main()
