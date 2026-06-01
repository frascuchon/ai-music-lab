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
    if os.path.isdir(os.path.join(REPO_DIR, "model")):
        return
    print(f"[setup] Clonando {REPO_URL} → {REPO_DIR} ...")
    subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, REPO_DIR])
    # instalar dependencias específicas del repo (además de las de pyproject.toml)
    req = os.path.join(REPO_DIR, "requirements-mac.txt")
    if not os.path.isfile(req):
        req = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.isfile(req):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req, "-q"])


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


def generate(prompt: str, out_path: str, temperature: float = 1.0, max_len: int = 2000):
    _clone_repo_if_needed()

    # el model/ del repo text2midi debe estar en el path
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import T5Tokenizer

    # importar desde el repo clonado
    from model.transformer_model import Transformer  # noqa: E402

    device = _device()
    print(f"[info] device={device}")

    # --- carga del modelo ---
    t0 = time.time()
    mem_before = _mem_mb()

    repo_id = "amaai-lab/text2midi"
    model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")

    with open(tokenizer_path, "rb") as f:
        r_tokenizer = pickle.load(f)

    vocab_size = len(r_tokenizer)
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    parser.add_argument("--max_len", type=int, default=2000)
    args = parser.parse_args()

    generate(args.prompt, args.out, args.temperature, args.max_len)


if __name__ == "__main__":
    main()
