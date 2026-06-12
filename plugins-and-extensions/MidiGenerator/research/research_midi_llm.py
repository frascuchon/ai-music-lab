"""
PoC para evaluar MIDI-LLM (slSeanWU / NeurIPS AI4Music 2025) en Mac Apple Silicon.

Modelo: slseanwu/MIDI-LLM_Llama-3.2-1B  — Llama 3.2 1B con vocabulario extendido (MIDI tokens)
Tokenización MIDI: misma que Anticipatory Music Transformer (librería `anticipation`)
Referencia: https://github.com/slSeanWU/MIDI-LLM  |  https://arxiv.org/abs/2511.03942

Uso:
    uv run research_midi_llm.py --prompt "upbeat jazz trio, 120 BPM, piano bass drums" --out out_mllm.mid
    uv run research_midi_llm.py  # prompt por defecto

Dependencias extra vs research_text2midi.py:
    - ninguna (torch, transformers, anticipation ya están en pyproject.toml)

Notas de compatibilidad Mac:
    - bfloat16 soportado en MPS desde PyTorch 2.0 (M1+)
    - El código original hardcodea device="cuda"; aquí usamos auto-detect MPS/CPU
    - 3.4 GB modelo → se descarga una sola vez a ~/.cache/huggingface/
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = os.path.join(os.path.dirname(__file__), "midi_llm_repo")
REPO_URL = "https://github.com/slSeanWU/MIDI-LLM.git"
MODEL_ID = "slseanwu/MIDI-LLM_Llama-3.2-1B"

SYSTEM_PROMPT = (
    "You are a world-class composer. Please compose some music "
    "according to the following description: "
)

DEFAULT_PROMPT = (
    "upbeat jazz trio, 120 BPM, piano bass and drums, happy mood, "
    "swinging feel, 30 seconds"
)


def _clone_repo_if_needed():
    if not os.path.isdir(os.path.join(REPO_DIR, "midi_llm")):
        print(f"[setup] Clonando {REPO_URL} → {REPO_DIR} ...")
        subprocess.check_call(
            ["git", "clone", "--depth=1", REPO_URL, REPO_DIR]
        )


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mem_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024




def generate(
    prompt: str,
    out_path: str,
    temperature: float = 1.0,
    top_p: float = 0.98,
    max_tokens: int = 2046,
    n_outputs: int = 4,
    device_override: str | None = None,
):
    """
    Generate n_outputs MIDI files from a single prompt.

    If n_outputs == 1, writes exactly to out_path.
    If n_outputs > 1, writes to out_path (v0) and out_path_v1/v2/... siblings,
    replacing any existing _vN suffix so the naming stays clean.
    """
    _clone_repo_if_needed()

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from midi_llm.utils import (
        AMT_GPT2_BOS_ID,
        LLAMA_VOCAB_SIZE,
        save_generation,
    )

    device = device_override if device_override else _device()
    dtype = torch.bfloat16

    print(f"[info] device={device}  dtype={dtype}  model={MODEL_ID}")

    # --- carga del modelo (una sola vez) ---
    t0 = time.time()
    mem_before = _mem_mb()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, pad_token="<|eot_id|>")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)
    model.eval()

    t_load = time.time() - t0
    mem_load = _mem_mb() - mem_before
    print(f"[timing] carga modelo: {t_load:.1f}s | RAM delta: {mem_load:.0f} MB")

    # --- preparar input ids (solo una vez) ---
    full_prompt = SYSTEM_PROMPT + prompt + " "
    llama_input = tokenizer(full_prompt, return_tensors="pt", padding=False)
    input_ids_base = llama_input["input_ids"]

    midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]])
    input_ids_base = torch.cat([input_ids_base, midi_bos], dim=1).to(device)

    # --- output paths ---
    p = Path(out_path)
    stem_clean = p.stem
    # strip existing _vN suffix so we can re-run cleanly
    import re
    stem_clean = re.sub(r"_v\d+$", "", stem_clean)

    def nth_path(i: int) -> Path:
        if n_outputs == 1:
            return p
        return p.parent / f"{stem_clean}_v{i}{p.suffix}"

    # --- generación secuencial (evita x4 KV-cache en MPS) ---
    saved = []
    for i in range(n_outputs):
        t1 = time.time()
        print(f"\n[gen {i+1}/{n_outputs}] generating...")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids_base,
                do_sample=True,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        t_infer = time.time() - t1
        print(f"[timing] inferencia: {t_infer:.1f}s")

        prompt_len = input_ids_base.shape[1]
        generated = outputs[0, prompt_len:]
        midi_tokens = (generated - LLAMA_VOCAB_SIZE).cpu().tolist()

        out_dir = p.parent / "_mllm_tmp"
        ok = save_generation(
            tokens=midi_tokens,
            prompt=prompt,
            output_dir=out_dir,
            generation_idx=1,
            synthesize=False,
            validate=True,
        )

        if ok:
            tmp_mid = out_dir / "gen_1.mid"
            dest = nth_path(i)
            shutil.move(str(tmp_mid), str(dest))
            for f in out_dir.iterdir():
                f.unlink(missing_ok=True)
            out_dir.rmdir()
            print(f"[output] MIDI guardado en: {dest.absolute()}")
            saved.append(dest)

            try:
                import pretty_midi
                midi_obj = pretty_midi.PrettyMIDI(str(dest))
                n_inst = len(midi_obj.instruments)
                n_notes = sum(len(ins.notes) for ins in midi_obj.instruments)
                instrs = [ins.program for ins in midi_obj.instruments]
                dur = midi_obj.get_end_time()
                print(f"[midi]   {n_inst} pistas | {n_notes} notas | {dur:.1f}s | instrumentos: {instrs}")
            except Exception as e:
                print(f"[midi]   no se pudo inspeccionar: {e}")
        else:
            print(f"[warn] generación {i+1} falló validación — omitida")

    print(f"\n[done] {len(saved)}/{n_outputs} outputs guardados")
    print(f"  prompt: {prompt[:100]}...")
    print(f"  device: {device} | temperatura: {temperature} | top_p: {top_p}")
    print(f"  outputs: {[str(s) for s in saved]}")


def main():
    parser = argparse.ArgumentParser(description="PoC MIDI-LLM en Mac")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", default="out_mllm_v0.mid",
                        help="Ruta del primer output (v0); los siguientes se numeran _v1, _v2...")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperatura de muestreo (default 1.0 — igual que el script oficial)")
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_tokens", type=int, default=2046,
                        help="Tokens MIDI a generar (default 2046)")
    parser.add_argument("--n_outputs", type=int, default=4,
                        help="Número de MIDIs a generar (default 4)")
    parser.add_argument("--device", default=None,
                        help="Forzar dispositivo: cpu, mps, cuda (auto si se omite)")
    args = parser.parse_args()

    generate(
        args.prompt,
        args.out,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n_outputs=args.n_outputs,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
