"""
PoC para evaluar Text2midi (AMAAI-Lab) en Mac Apple Silicon.

Uso:
    uv run research_text2midi.py --prompt "upbeat pop melody in C major, 120 BPM" --out out_t2m.mid
    uv run research_text2midi.py  # prompt por defecto

Documenta: tiempo de carga, tiempo de inferencia, RAM usada.

Dependencias del repo text2midi clonadas localmente en text2midi/ (se clonan
automáticamente si no están presentes).

Referencia: https://github.com/AMAAI-Lab/Text2midi
Licencia del modelo: MIT
"""

import argparse
import os
import pickle
import subprocess
import sys
import time

REPO_DIR = os.path.join(os.path.dirname(__file__), "text2midi")
REPO_URL = "https://github.com/AMAAI-Lab/Text2midi.git"

DEFAULT_PROMPT = (
    "upbeat pop melody in C major, 120 BPM, piano and light strings, "
    "happy mood, 30 seconds"
)


def _clone_repo_if_needed():
    sentinel = os.path.join(REPO_DIR, ".deps_installed")
    if not os.path.isdir(os.path.join(REPO_DIR, "model")):
        print(f"[setup] Clonando {REPO_URL} → {REPO_DIR} ...")
        subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, REPO_DIR])
    if not os.path.isfile(sentinel):
        req = os.path.join(REPO_DIR, "requirements-mac.txt")
        if not os.path.isfile(req):
            req = os.path.join(REPO_DIR, "requirements.txt")
        if os.path.isfile(req):
            print(f"[setup] Instalando dependencias de {os.path.basename(req)} ...")
            subprocess.check_call(["uv", "pip", "install", "-r", req, "-q"])
            open(sentinel, "w").close()


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


def generate(prompt: str, out_path: str, temperature: float = 1.0, max_len: int = 512,
             device_override: str | None = None, half_precision: bool = True):
    _clone_repo_if_needed()

    # el model/ del repo text2midi debe estar en el path
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import T5Tokenizer

    # importar desde el repo clonado
    from model.transformer_model import Transformer  # noqa: E402

    device = device_override if device_override else _device()
    print(f"[info] device={device}  half_precision={half_precision}")

    # --- carga del modelo ---
    t0 = time.time()
    mem_before = _mem_mb()

    repo_id = "amaai-lab/text2midi"
    model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")

    with open(tokenizer_path, "rb") as f:
        r_tokenizer = pickle.load(f)

    vocab_size = len(r_tokenizer)
    # Cargar siempre en CPU primero para evitar picos de memoria en MPS
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device="cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    if half_precision:
        model = model.half()
    model = model.to(device)
    model.eval()

    text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    t_load = time.time() - t0
    mem_load = _mem_mb() - mem_before
    print(f"[timing] carga modelo: {t_load:.1f}s | RAM delta: {mem_load:.0f} MB")

    # --- inferencia ---
    t1 = time.time()

    inputs = text_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    # input_ids son enteros — no convertir a half; sólo los pesos del modelo van en fp16

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask,
            max_len=max_len,
            temperature=temperature,
        )

    generated_midi = r_tokenizer.decode(output[0].tolist())
    generated_midi.dump_midi(out_path)

    t_infer = time.time() - t1
    print(f"[timing] inferencia: {t_infer:.1f}s")
    print(f"[output] MIDI guardado en: {os.path.abspath(out_path)}")

    # resumen para RESEARCH.md
    print("\n--- RESULTADOS (copiar en RESEARCH.md) ---")
    print(f"  prompt:          {prompt}")
    print(f"  device:          {device}")
    print(f"  carga (s):       {t_load:.1f}")
    print(f"  inferencia (s):  {t_infer:.1f}")
    print(f"  RAM delta (MB):  {mem_load:.0f}")
    print(f"  output:          {os.path.abspath(out_path)}")


def main():
    parser = argparse.ArgumentParser(description="PoC Text2midi en Mac")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", default="out_t2m.mid")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512,
                        help="Tokens a generar (default 512; 2000 para piezas largas)")
    parser.add_argument("--device", default=None,
                        help="Forzar dispositivo: cpu, mps, cuda (auto si se omite)")
    parser.add_argument("--no-half", action="store_true",
                        help="Desactivar float16 (más RAM, más preciso)")
    args = parser.parse_args()

    generate(args.prompt, args.out, args.temperature, args.max_len,
             device_override=args.device, half_precision=not args.no_half)


if __name__ == "__main__":
    main()
