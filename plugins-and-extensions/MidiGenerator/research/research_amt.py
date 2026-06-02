"""
PoC para evaluar Anticipatory Music Transformer (Stanford CRFM) en Mac Apple Silicon.

Modos:
  - continuation:   genera continuación de un MIDI existente
  - accompaniment:  genera acompañamiento dado una melodía como control

Uso:
    uv run research_amt.py --mode continuation --input fixtures/melody.mid --out out_cont.mid
    uv run research_amt.py --mode accompaniment --input fixtures/melody.mid --out out_acc.mid
    uv run research_amt.py  # crea fixture + genera continuación

Referencia: https://github.com/jthickstun/anticipation
Modelo HF: stanford-crfm/music-medium-800k (360M params, Apache 2.0)
"""

import argparse
import os
import time

import psutil

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "melody.mid")
MODEL_NAME = "stanford-crfm/music-medium-800k"


def _mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    # AMT usa CUDA principalmente; en CPU/MPS cargamos en CPU
    return "cpu"


def create_fixture_midi(path: str):
    """Genera un MIDI simple de 8 compases (melodía en C mayor, 4/4, 120 BPM)."""
    import mido

    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))  # 120 BPM
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.Message("program_change", program=0, channel=0, time=0))  # piano

    # escala de C mayor: C D E F G A B C, 2 octavas, quarter notes
    notes = [60, 62, 64, 65, 67, 69, 71, 72,   # C4..C5
             72, 71, 69, 67, 65, 64, 62, 60,   # C5..C4
             60, 64, 67, 72, 60, 64, 67, 72,   # arpegios
             72, 67, 64, 60, 72, 67, 64, 60]   # arpegios inversos

    quarter = 480
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=80, channel=0, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, channel=0, time=quarter))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    mid.save(path)
    print(f"[fixture] MIDI de prueba generado en: {os.path.abspath(path)}")


def run_continuation(model, input_midi_path: str, out_path: str, duration: int = 10):
    from anticipation.convert import events_to_midi, midi_to_events
    from anticipation.ops import clip, sort
    from anticipation.sample import generate

    print("[continuation] cargando MIDI de entrada...")
    events = midi_to_events(input_midi_path)
    # usar los primeros 5 segundos como "historia" y generar 10s de continuación
    history = clip(events, 0, 5)

    print(f"[continuation] generando {duration}s de continuación...")
    t0 = time.time()
    new_events = generate(
        model,
        start_time=5,
        end_time=5 + duration,
        inputs=history,
        top_p=0.98,
    )
    t_infer = time.time() - t0
    print(f"[timing] inferencia: {t_infer:.1f}s")

    # combine(a, b) resta CONTROL_OFFSET a b — sólo válido cuando b son controles.
    # history y new_events son eventos normales; basta concatenar y ordenar.
    combined = sort(history + new_events)
    mid = events_to_midi(combined)
    mid.save(out_path)
    print(f"[output] MIDI guardado en: {os.path.abspath(out_path)}")
    return t_infer


def run_accompaniment(model, input_midi_path: str, out_path: str, duration: int = 10):
    from anticipation.convert import events_to_midi, midi_to_events
    from anticipation.ops import clip, combine
    from anticipation.sample import generate
    from anticipation.tokenize import extract_instruments

    print("[accompaniment] cargando MIDI de entrada como melodía de control...")
    events = midi_to_events(input_midi_path)
    melody = clip(events, 0, duration)
    # extract_instruments devuelve (remaining_events, controls_con_CONTROL_OFFSET)
    _, controls = extract_instruments(melody, [0])

    print(f"[accompaniment] generando {duration}s de acompañamiento...")
    t0 = time.time()
    accompaniment = generate(
        model,
        start_time=0,
        end_time=duration,
        controls=controls,
        top_p=0.98,
    )
    t_infer = time.time() - t0
    print(f"[timing] inferencia: {t_infer:.1f}s")

    # combine(events, controls): resta CONTROL_OFFSET a controls y mezcla con events
    combined = combine(accompaniment, controls)
    mid = events_to_midi(combined)
    mid.save(out_path)
    print(f"[output] MIDI guardado en: {os.path.abspath(out_path)}")
    return t_infer


def main():
    parser = argparse.ArgumentParser(description="PoC Anticipatory Music Transformer")
    parser.add_argument("--mode", choices=["continuation", "accompaniment", "both"],
                        default="both")
    parser.add_argument("--input", default=FIXTURE_PATH)
    parser.add_argument("--out", default=None,
                        help="Ruta de salida. Si mode=both se ignora y usa nombres fijos.")
    parser.add_argument("--duration", type=int, default=10,
                        help="Segundos a generar (default: 10)")
    parser.add_argument("--no-half", action="store_true",
                        help="Desactivar float16 (más RAM, más preciso)")
    args = parser.parse_args()

    # crear fixture si no existe
    if not os.path.isfile(args.input):
        if args.input == FIXTURE_PATH:
            create_fixture_midi(FIXTURE_PATH)
        else:
            print(f"[error] archivo MIDI de entrada no encontrado: {args.input}")
            raise SystemExit(1)

    import torch
    from transformers import AutoModelForCausalLM
    import transformers.safetensors_conversion as _sc
    _sc.auto_conversion = lambda *a, **kw: None  # silencia el thread de conversión HF

    # AMT requiere float32: en fp16 genera tokens fuera de vocabulario (assert negativo)
    if not args.no_half:
        print("[warning] AMT no es compatible con float16; usando float32 automáticamente")
    dtype = torch.float32

    device = _device()
    print(f"[info] device={device}  dtype=float32  (AMT optimizado para CUDA; en CPU es más lento)")

    # --- carga del modelo ---
    t0 = time.time()
    mem_before = _mem_mb()

    print(f"[setup] cargando {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=False,
    )
    model = model.to(device)
    model.eval()

    t_load = time.time() - t0
    mem_load = _mem_mb() - mem_before
    print(f"[timing] carga modelo: {t_load:.1f}s | RAM delta: {mem_load:.0f} MB")

    # --- inferencia ---
    with torch.no_grad():
        if args.mode in ("continuation", "both"):
            out = args.out or "out_cont.mid"
            t_cont = run_continuation(model, args.input, out, args.duration)
        if args.mode in ("accompaniment", "both"):
            out = args.out or "out_acc.mid"
            t_acc = run_accompaniment(model, args.input, out, args.duration)

    # resumen para RESEARCH.md
    print("\n--- RESULTADOS (copiar en RESEARCH.md) ---")
    print(f"  modelo:          {MODEL_NAME}")
    print(f"  device:          {device}  (dtype: float32)")
    print(f"  carga (s):       {t_load:.1f}")
    if args.mode in ("continuation", "both"):
        print(f"  inferencia cont (s): {t_cont:.1f}")
    if args.mode in ("accompaniment", "both"):
        print(f"  inferencia acc  (s): {t_acc:.1f}")
    print(f"  RAM delta (MB):  {mem_load:.0f}")


if __name__ == "__main__":
    main()
