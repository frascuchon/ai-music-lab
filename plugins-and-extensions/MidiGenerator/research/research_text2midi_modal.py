"""
Modal.com inference app para text2midi (AMAAI-Lab, AAAI 2025).

Ejecuta el modelo en GPU CUDA (A10G) sin half-precision, exactamente como el
HuggingFace Space oficial (temperature=0.9, float32, CUDA).

Modelo: amaai-lab/text2midi (autoregressive transformer decoder, ~900 MB)
Repo:   https://github.com/AMAAI-Lab/Text2midi
Demo:   https://amaai-lab.github.io/Text2midi/
Paper:  https://arxiv.org/abs/2412.16526

Setup (pre-descarga de pesos al Volume, ejecutar una vez):
    modal run research_text2midi_modal.py::setup

Inferencia (un prompt, N outputs):
    modal run research_text2midi_modal.py::main \\
        --prompt "A cheerful rock song with bright electric guitars" \\
        --out-dir ../evaluation/text2midi/test4 \\
        --n-outputs 2

Inferencia para todos los tests del eval:
    modal run research_text2midi_modal.py::eval_all \\
        --eval-dir ../evaluation/text2midi \\
        --force

GPUs disponibles (--gpu en eval_all, por defecto A10G):
    A10G     24 GB, ~$1.10/hr  (default) — suficiente para el modelo 900 MB
    L4       24 GB, ~$0.80/hr  — alternativa más barata
    A100-40GB 40 GB, ~$2.10/hr — máxima velocidad

Coste estimado (A10G):
    - 1 output con max_len=2000: ~60-120s de inferencia → ~$0.02-0.04/output
    - 7 tests × 2 outputs = ~$0.28-0.56 total

Parámetros oficiales (HuggingFace Space app.py):
    temperature = 0.9   (slider 0.8–1.1)
    max_len     = 2000  (slider 500–2000; default 500 en la app, pero usamos 2000)
    precision   = float32 (sin half; la app no usa half)
"""

import os
import sys
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Volume — caché HuggingFace persistente
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("text2midi-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"
HF_CACHE = f"{WEIGHTS_MOUNT}/hf_cache"

MODEL_ID = "amaai-lab/text2midi"
REPO_URL = "https://github.com/AMAAI-Lab/Text2midi.git"
REPO_DIR = "/opt/text2midi"

DEFAULT_GPU = os.environ.get("TEXT2MIDI_GPU", "A10G")

# ---------------------------------------------------------------------------
# Container image — CUDA 12 + PyTorch + dependencias text2midi
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "ffmpeg"])
    .pip_install(
        # Core ML
        "torch==2.4.0",
        "transformers==4.44.2",
        "huggingface_hub>=0.24",
        # text2midi deps (from requirements.txt)
        "miditok>=3.0.3",
        "miditoolkit>=0.1.16",
        "pretty_midi>=0.2.10",
        "mido>=1.3",
        "accelerate>=0.34",
        "jsonlines>=4.0",
        "psutil>=6.0",
        "tqdm>=4.66",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # MoE layer used by the decoder (latest available is 0.1.x)
        "st-moe-pytorch",
        "einops>=0.8",
    )
    .run_commands(
        f"git clone --depth=1 {REPO_URL} {REPO_DIR}",
        f"pip install -r {REPO_DIR}/requirements.txt -q 2>/dev/null || true",
    )
)

app = modal.App("text2midi-inference", image=image)


# ---------------------------------------------------------------------------
# Pre-descarga de pesos al Volume
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=600,
    gpu=DEFAULT_GPU,
)
def setup():
    """Descarga los pesos del modelo a un Volume persistente. Ejecutar una vez."""
    from huggingface_hub import hf_hub_download

    os.environ["HF_HOME"] = HF_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)

    print(f"[setup] Descargando {MODEL_ID} → {HF_CACHE} ...")
    for fname in ["pytorch_model.bin", "vocab_remi.pkl"]:
        path = hf_hub_download(repo_id=MODEL_ID, filename=fname, cache_dir=HF_CACHE)
        print(f"[setup]   {fname} → {path}")

    weights_vol.commit()
    print("[setup] Pesos descargados y committed al Volume.")


# ---------------------------------------------------------------------------
# Núcleo de inferencia
# ---------------------------------------------------------------------------
def _load_model():
    """Carga el modelo en CUDA (float32, sin half)."""
    import pickle
    import torch
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, REPO_DIR)
    from model.transformer_model import Transformer
    from transformers import T5Tokenizer

    os.environ["HF_HOME"] = HF_CACHE
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = hf_hub_download(repo_id=MODEL_ID, filename="pytorch_model.bin", cache_dir=HF_CACHE)
    vocab_path = hf_hub_download(repo_id=MODEL_ID, filename="vocab_remi.pkl", cache_dir=HF_CACHE)

    with open(vocab_path, "rb") as f:
        r_tokenizer = pickle.load(f)

    vocab_size = len(r_tokenizer)
    # float32 — sin half; matches the official HuggingFace Space (no explicit precision cast)
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device="cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model = model.to(device)
    model.eval()

    text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    return model, text_tokenizer, r_tokenizer, device


def _infer_one(model, text_tokenizer, r_tokenizer, device, prompt: str,
               max_len: int, temperature: float) -> bytes:
    """Genera un MIDI y devuelve los bytes del fichero."""
    import io
    import torch
    import torch.nn as nn

    inputs = text_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = nn.utils.rnn.pad_sequence(
        inputs.input_ids, batch_first=True, padding_value=0
    ).to(device)
    attention_mask = nn.utils.rnn.pad_sequence(
        inputs.attention_mask, batch_first=True, padding_value=0
    ).to(device)

    with torch.no_grad():
        output = model.generate(input_ids, attention_mask,
                                max_len=max_len, temperature=temperature)

    generated_midi = r_tokenizer.decode(output[0].tolist())

    buf = io.BytesIO()
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    tmp.close()
    generated_midi.dump_midi(tmp.name)
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data


# ---------------------------------------------------------------------------
# Función Modal principal — genera N variantes para M prompts
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=3600,
    gpu=DEFAULT_GPU,
)
def generate(
    prompts: list[str],
    n_outputs: int = 2,
    temperature: float = 0.9,
    max_len: int = 2000,
) -> list[list[bytes]]:
    """
    Genera n_outputs variantes de MIDI para cada prompt.

    Returns: list[prompt] → list[output] → bytes (MIDI)
    """
    import time

    model, text_tokenizer, r_tokenizer, device = _load_model()
    print(f"[modal] device={device}  temp={temperature}  max_len={max_len}  n_outputs={n_outputs}")

    results = []
    for i, prompt in enumerate(prompts):
        prompt_results = []
        for v in range(n_outputs):
            t0 = time.time()
            midi_bytes = _infer_one(model, text_tokenizer, r_tokenizer, device,
                                    prompt, max_len, temperature)
            elapsed = time.time() - t0
            print(f"[modal] [{i+1}/{len(prompts)}] v{v}: {len(midi_bytes)} bytes  {elapsed:.0f}s")
            prompt_results.append(midi_bytes)
        results.append(prompt_results)

    return results


# ---------------------------------------------------------------------------
# Entrypoint: un prompt → out-dir
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = "",
    out_dir: str = ".",
    n_outputs: int = 2,
    temperature: float = 0.9,
    max_len: int = 2000,
    force: bool = False,
):
    """
    Genera MIDI para un prompt y guarda los ficheros en out-dir.

    Ejemplo:
        modal run research_text2midi_modal.py::main \\
            --prompt "A sad pop song with a strong piano presence." \\
            --out-dir ../evaluation/text2midi/test1
    """
    if not prompt:
        print("ERROR: --prompt es obligatorio")
        sys.exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Saltar si ya existen outputs (a menos que --force)
    existing = sorted(out_path.glob("generated_cuda_v*.mid"))
    if existing and not force:
        print(f"[skip] Ya existen {len(existing)} ficheros en {out_dir}. Usa --force para sobreescribir.")
        return

    print(f"[main] Prompt: {prompt[:100]}...")
    print(f"[main] n_outputs={n_outputs}  temperature={temperature}  max_len={max_len}")

    results = generate.remote([prompt], n_outputs=n_outputs,
                              temperature=temperature, max_len=max_len)
    midi_list = results[0]

    for v, midi_bytes in enumerate(midi_list):
        out_file = out_path / f"generated_cuda_v{v}.mid"
        out_file.write_bytes(midi_bytes)
        print(f"[main] → {out_file}  ({len(midi_bytes)} bytes)")

    print(f"[main] Generados {len(midi_list)} ficheros en {out_dir}")


# ---------------------------------------------------------------------------
# Entrypoint: todos los tests del directorio de evaluación
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/text2midi",
    n_outputs: int = 2,
    temperature: float = 0.9,
    max_len: int = 2000,
    force: bool = False,
    only: str = "",
):
    """
    Genera MIDI para todos los tests del directorio de evaluación.

    Ejemplo:
        modal run research_text2midi_modal.py::eval_all \\
            --eval-dir ../evaluation/text2midi \\
            --force

        # Solo tests específicos:
        ONLY=1,4,7 modal run research_text2midi_modal.py::eval_all ...
    """
    import re

    eval_path = Path(eval_dir)

    # Detectar carpetas test*
    test_dirs = sorted(eval_path.glob("test*"), key=lambda p: int(re.sub(r"\D", "", p.name) or "0"))

    if only:
        only_nums = {int(x) for x in only.split(",")}
        test_dirs = [d for d in test_dirs if int(re.sub(r"\D", "", d.name) or "0") in only_nums]

    if not test_dirs:
        print(f"ERROR: No se encontraron carpetas test* en {eval_dir}")
        sys.exit(1)

    # Leer prompts
    prompts = []
    valid_dirs = []
    for td in test_dirs:
        prompt_file = td / "prompt.txt"
        if not prompt_file.exists():
            print(f"[warn] Sin prompt.txt en {td.name}, saltando")
            continue

        # Extraer la línea Prompt: "..."
        text = prompt_file.read_text()
        m = re.search(r'^Prompt:\s*"(.+)"', text, re.MULTILINE)
        if not m:
            print(f"[warn] No se pudo parsear Prompt: en {td.name}, saltando")
            continue

        prompt = m.group(1)

        # Saltar si ya existen outputs
        existing = sorted(td.glob("generated_cuda_v*.mid"))
        if existing and not force:
            print(f"[skip] {td.name}: ya tiene {len(existing)} outputs (usa --force)")
            continue

        prompts.append(prompt)
        valid_dirs.append(td)

    if not prompts:
        print("[eval_all] Nada que generar.")
        return

    print(f"[eval_all] {len(prompts)} tests a generar: {[d.name for d in valid_dirs]}")
    print(f"[eval_all] temperature={temperature}  max_len={max_len}  n_outputs={n_outputs}")

    # Llamada remota — batch único para amortizar carga del modelo
    results = generate.remote(prompts, n_outputs=n_outputs,
                              temperature=temperature, max_len=max_len)

    for td, midi_list in zip(valid_dirs, results):
        for v, midi_bytes in enumerate(midi_list):
            out_file = td / f"generated_cuda_v{v}.mid"
            out_file.write_bytes(midi_bytes)
            print(f"[eval_all] → {td.name}/generated_cuda_v{v}.mid  ({len(midi_bytes)} bytes)")

    print(f"\n[eval_all] Completado — {len(valid_dirs)} tests × {n_outputs} outputs generados.")
