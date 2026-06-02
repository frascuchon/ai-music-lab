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
import subprocess
import sys
import time

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
    temperature: float = 0.95,
    top_p: float = 0.98,
    max_tokens: int = 1024,
    device_override: str | None = None,
):
    _clone_repo_if_needed()

    # exponer midi_llm/ al path
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from midi_llm.utils import (
        AMT_GPT2_BOS_ID,
        LLAMA_VOCAB_SIZE,
        save_generation,
    )
    from pathlib import Path

    device = device_override if device_override else _device()
    dtype = torch.bfloat16  # soportado en MPS y CUDA; float32 en CPU puro si falla

    print(f"[info] device={device}  dtype={dtype}  model={MODEL_ID}")

    # --- carga del modelo ---
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

    # --- preparar input ---
    full_prompt = SYSTEM_PROMPT + prompt + " "
    llama_input = tokenizer(full_prompt, return_tensors="pt", padding=False)
    input_ids = llama_input["input_ids"]

    # añadir token BOS de MIDI (igual que en el script original)
    midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]])
    input_ids = torch.cat([input_ids, midi_bos], dim=1).to(device)

    # --- inferencia ---
    t1 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    t_infer = time.time() - t1
    print(f"[timing] inferencia: {t_infer:.1f}s")

    # --- convertir tokens → MIDI ---
    prompt_len = input_ids.shape[1]
    generated = outputs[0, prompt_len:]
    midi_tokens = (generated - LLAMA_VOCAB_SIZE).cpu().tolist()

    # save_generation espera un directorio; usamos un tmp subdir
    out_dir = Path(out_path).parent / "_mllm_tmp"
    ok = save_generation(
        tokens=midi_tokens,
        prompt=prompt,
        output_dir=out_dir,
        generation_idx=1,
        synthesize=False,
        validate=True,
    )

    if ok:
        import shutil
        tmp_mid = out_dir / "gen_1.mid"
        shutil.move(str(tmp_mid), out_path)
        # limpiar archivos auxiliares del tmp dir
        for f in out_dir.iterdir():
            f.unlink(missing_ok=True)
        out_dir.rmdir()
        print(f"[output] MIDI guardado en: {os.path.abspath(out_path)}")
    else:
        print("[error] save_generation falló — los tokens generados no pasaron validación.")
        print("        Prueba con --temperature 0.9 o --max_tokens 2046")
        sys.exit(1)

    # inspeccionar MIDI generado
    try:
        import pretty_midi
        midi_obj = pretty_midi.PrettyMIDI(out_path)
        n_instruments = len(midi_obj.instruments)
        n_notes = sum(len(i.notes) for i in midi_obj.instruments)
        instrs = [i.program for i in midi_obj.instruments]
        duration = midi_obj.get_end_time()
        print(f"[midi]   {n_instruments} pistas | {n_notes} notas | {duration:.1f}s | instrumentos: {instrs}")
    except Exception as e:
        print(f"[midi]   no se pudo inspeccionar: {e}")

    # resumen para RESEARCH.md
    print("\n--- RESULTADOS (copiar en RESEARCH.md) ---")
    print(f"  prompt:          {prompt}")
    print(f"  device:          {device}")
    print(f"  carga (s):       {t_load:.1f}")
    print(f"  inferencia (s):  {t_infer:.1f}")
    print(f"  RAM delta (MB):  {mem_load:.0f}")
    print(f"  output:          {os.path.abspath(out_path)}")


def main():
    parser = argparse.ArgumentParser(description="PoC MIDI-LLM en Mac")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", default="out_mllm.mid")
    parser.add_argument("--temperature", type=float, default=0.95,
                        help="Temperatura de muestreo (default 0.95; probar 0.8-1.0)")
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Tokens MIDI a generar (default 1024; max 2046)")
    parser.add_argument("--device", default=None,
                        help="Forzar dispositivo: cpu, mps, cuda (auto si se omite)")
    args = parser.parse_args()

    generate(
        args.prompt,
        args.out,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
