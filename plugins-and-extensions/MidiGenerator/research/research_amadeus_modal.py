"""
Modal.com inference app para Amadeus (lingyu123-su/Amadeus, arXiv 2508.20665).

Amadeus combina un decoder transformer autoregresivo para secuencias de notas con
un decoder de difusión bidireccional para atributos intra-nota (NB encoding).
Entrenado sobre LakhALLFined (subconjunto filtrado del Lakh MIDI Dataset).

Pipeline completo:
    [Text prompt]
        ↓
    T5 text encoder (google/flan-t5-base, max 128 tokens)
        ↓
    AmadeusModel.generate() — NB encoding, top_p sampling
        ↓
    reverse_shift_and_pad_for_tensor() — post-proceso NB específico
        ↓
    MidiDecoder4NB (tokens → MIDI)
        ↓
    generated_cuda_v0.mid

Modelo: longyu1315/Amadeus-S (~2.5 GB checkpoint + vocab JSON 14 kB)
Repo:   https://github.com/lingyu123-su/Amadeus
Paper:  https://arxiv.org/abs/2508.20665

Parámetros de generación (verbatim de demo/Amadeus_app_EN.py):
    threshold=0.99, temperature=1.25, generation_length=1024
    sampling_method='top_p', text_encoder='google/flan-t5-base'

Setup (descarga pesos al Volume, ejecutar una vez):
    modal run research/research_amadeus_modal.py::setup

Inferencia libre:
    modal run research/research_amadeus_modal.py::main \\
        --prompt "A melodic electronic ambient song..." \\
        --out-dir evaluation/amadeus/smoke

Benchmark completo:
    modal run research/research_amadeus_modal.py::eval_all \\
        --eval-dir evaluation/amadeus \\
        --n-outputs 2

GPUs disponibles (--gpu, por defecto A10G):
    A10G     24 GB, ~$1.10/hr  (default) — modelo ~2.5 GB, margen amplio
    T4       16 GB, ~$0.60/hr  — probablemente suficiente

Coste estimado (A10G):
    - 1 output: ~30-90s → ~$0.01-0.03/output
    - 8 tests × 2 outputs = ~$0.16-0.48 total
"""

import os
import re
import sys
import tempfile
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Volume — caché HuggingFace persistente
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("amadeus-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"
HF_CACHE = f"{WEIGHTS_MOUNT}/hf_cache"

MODEL_ID = "longyu1315/Amadeus-S"
AMADEUS_DIR = "/amadeus"
TEXT_ENCODER = "google/flan-t5-base"

DEFAULT_GPU = os.environ.get("AMADEUS_GPU", "A10G")

# ---------------------------------------------------------------------------
# Container image — CUDA 12 + PyTorch + dependencias Amadeus + repo clonado
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "fluidsynth", "fluid-soundfont-gm", "ffmpeg"])
    .run_commands(
        # Clonar el repo de Amadeus (código del modelo y utils)
        f"git clone https://github.com/lingyu123-su/Amadeus {AMADEUS_DIR}",
        # Parchear la ruta hardcodeada del soundfont (apunta al servidor del autor)
        # Usamos el soundfont de sistema instalado via apt (fluid-soundfont-gm)
        "sed -i \"s|DEFAULT_SOUND_FONT = '.*'|"
        "DEFAULT_SOUND_FONT = '/usr/share/sounds/sf2/FluidR3_GM.sf2'|\" "
        f"{AMADEUS_DIR}/Amadeus/symbolic_encoding/midi2audio.py",
    )
    .pip_install(
        # Framework ML
        "torch==2.4.0",
        "transformers>=4.44",
        "huggingface_hub>=0.24",
        "accelerate>=0.34",
        "safetensors>=0.4",
        "sentencepiece>=0.2",
        # Arquitectura del modelo
        "omegaconf>=2.3",
        "x-transformers>=1.30",
        # Encoding simbólico / MIDI I/O
        "music21>=9.1",
        "mido>=1.3",
        "pretty_midi>=0.2.10",
        "matplotlib>=3.8",
        # Evaluación (métricas)
        "muspy>=0.5",
        # Utilidades
        "psutil>=6.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

app = modal.App("amadeus-inference", image=image)


# ---------------------------------------------------------------------------
# Pre-descarga de pesos al Volume
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=1800,
    gpu=DEFAULT_GPU,
)
def setup():
    """Descarga el modelo al Volume persistente. Ejecutar una vez."""
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = HF_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)

    # Excluir soundfont HF (1.69 GB) — usamos el del sistema via apt
    print(f"[setup] Descargando {MODEL_ID} (~2.5 GB checkpoint + vocab) → {HF_CACHE}")
    path = snapshot_download(
        MODEL_ID,
        cache_dir=HF_CACHE,
        ignore_patterns=["*.sf2"],
    )
    print(f"[setup] Descargado en {path}")

    # Pre-cachear el T5 encoder en el Volume también
    print(f"[setup] Descargando {TEXT_ENCODER} ...")
    from transformers import T5Tokenizer, T5EncoderModel
    T5Tokenizer.from_pretrained(TEXT_ENCODER, cache_dir=HF_CACHE)
    T5EncoderModel.from_pretrained(TEXT_ENCODER, cache_dir=HF_CACHE)
    print(f"[setup] T5 cacheado.")

    weights_vol.commit()
    print("[setup] Volume committed.")


# ---------------------------------------------------------------------------
# Núcleo de inferencia
# ---------------------------------------------------------------------------
def _load_model():
    """
    Carga AmadeusModel (Amadeus-S) y T5 encoder desde el Volume.

    Replica verbatim la lógica de load_resources() de demo/Amadeus_app_EN.py.
    """
    import torch
    from omegaconf import OmegaConf
    from huggingface_hub import snapshot_download
    from transformers import T5Tokenizer, T5EncoderModel

    sys.path.insert(0, AMADEUS_DIR)

    from Amadeus.evaluation_utils import (
        wandb_style_config_to_omega_config,
        prepare_model_and_dataset_from_config,
    )

    os.environ["HF_HOME"] = HF_CACHE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] device={device}")

    # Localizar la carpeta del modelo (snapshot_download devuelve la ruta cached)
    model_root = Path(snapshot_download(
        MODEL_ID, cache_dir=HF_CACHE, ignore_patterns=["*.sf2"]
    ))
    # Estructura en HF: <root>/Amadeus-S/files/{config.yaml,checkpoints/}
    wandb_exp_dir = model_root / "Amadeus-S"
    ckpt_dir = wandb_exp_dir / "files" / "checkpoints"
    config_path = wandb_exp_dir / "files" / "config.yaml"
    vocab_path = next(ckpt_dir.glob("vocab*.json"))

    # Checkpoint más reciente (el único: iter103662_loss-0.2098.pt)
    pt_files = sorted(
        ckpt_dir.glob("*.pt"),
        key=lambda fn: int(fn.stem.split("_")[0].replace("iter", "")),
    )
    ckpt_path = pt_files[-1]
    print(f"[load] checkpoint: {ckpt_path.name}  vocab: {vocab_path.name}")

    # Config (formato wandb → OmegaConf)
    config = OmegaConf.load(config_path)
    config = wandb_style_config_to_omega_config(config)

    # Modelo + vocabulario (verbatim del demo)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, vocab = prepare_model_and_dataset_from_config(config, vocab_path)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    # torch.compile(model)  # opcional: acelera pero añade tiempo de compilación

    print(f"[load] modelo listo — {sum(p.numel() for p in model.parameters()):,} params")

    # T5 encoder (verbatim del demo)
    print(f"[load] cargando {TEXT_ENCODER} ...")
    tokenizer = T5Tokenizer.from_pretrained(TEXT_ENCODER, cache_dir=HF_CACHE)
    encoder = T5EncoderModel.from_pretrained(TEXT_ENCODER, cache_dir=HF_CACHE).to(device)
    encoder.eval()
    print("[load] T5 listo.")

    return config, model, vocab, tokenizer, encoder, device


def _infer_one(
    config, model, vocab, tokenizer, encoder, device,
    prompt: str,
    seed: int = 0,
    sampling_method: str = "top_p",
    threshold: float = 0.99,
    temperature: float = 1.25,
    generation_length: int = 1024,
) -> bytes:
    """
    Genera un MIDI para un prompt.

    Replica verbatim generate_with_text_prompt() de demo/Amadeus_app_EN.py.
    """
    import torch

    sys.path.insert(0, AMADEUS_DIR)

    from Amadeus.symbolic_encoding import decoding_utils
    from Amadeus.symbolic_encoding.compile_utils import reverse_shift_and_pad_for_tensor

    encoding_scheme = config.nn_params.encoding_scheme  # 'nb' para Amadeus-S

    # --- Codificar texto con T5 (verbatim del demo) ---
    context_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    ).to(device)
    with torch.no_grad():
        context = encoder(**context_inputs).last_hidden_state

    # --- Decodificador MIDI (verbatim del demo) ---
    in_beat_resolution_dict = {"Pop1k7": 4, "Pop909": 4, "SOD": 12, "LakhClean": 4}
    in_beat_resolution = in_beat_resolution_dict.get(config.dataset, 4)  # LakhALLFined → 4

    midi_decoder_dict = {
        "remi": "MidiDecoder4REMI",
        "cp": "MidiDecoder4CP",
        "nb": "MidiDecoder4NB",
    }
    decoder_name = midi_decoder_dict[encoding_scheme]
    decoder = getattr(decoding_utils, decoder_name)(
        vocab=vocab,
        in_beat_resolution=in_beat_resolution,
        dataset_name=config.dataset,
    )

    # --- Generar secuencia de tokens (verbatim del demo) ---
    with torch.no_grad():
        generated_sample = model.generate(
            seed,
            generation_length,
            condition=None,
            num_target_measures=None,
            sampling_method=sampling_method,
            threshold=threshold,
            temperature=temperature,
            context=context,
        )

    # Post-proceso NB (verbatim del demo)
    if encoding_scheme == "nb":
        generated_sample = reverse_shift_and_pad_for_tensor(
            generated_sample, config.data_params.first_pred_feature
        )

    # --- Decodificar a MIDI ---
    with tempfile.TemporaryDirectory() as tmpdir:
        mid_path = Path(tmpdir) / "generated.mid"
        # El decoder también intenta generar audio via FluidSynth (puede fallar sin soundfont)
        try:
            decoder(generated_sample, output_path=str(mid_path))
        except Exception as e:
            print(f"[warn] decoder error (audio?): {e}")

        if mid_path.exists():
            return mid_path.read_bytes()

    return b""


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
    temperature: float = 1.25,
    threshold: float = 0.99,
    generation_length: int = 1024,
) -> list[list[bytes]]:
    """
    Genera n_outputs variantes MIDI para cada prompt.

    temperature: 1.25 (default demo), rango 0.5-3.0
    threshold: 0.99 (default demo), rango 0.5-1.0 para top_p
    generation_length: 1024 (default demo), rango 256-3072 tokens

    Returns: list[prompt] → list[output] → midi_bytes
    """
    config, model, vocab, tokenizer, encoder, device = _load_model()

    print(f"[modal] n_prompts={len(prompts)}  n_outputs={n_outputs}  "
          f"temperature={temperature}  threshold={threshold}  length={generation_length}")

    results = []
    for i, prompt in enumerate(prompts):
        prompt_results = []
        for v in range(n_outputs):
            t0 = time.time()
            try:
                midi_bytes = _infer_one(
                    config, model, vocab, tokenizer, encoder, device,
                    prompt=prompt,
                    seed=v,
                    temperature=temperature,
                    threshold=threshold,
                    generation_length=generation_length,
                )
                elapsed = time.time() - t0
                print(
                    f"[modal] [{i+1}/{len(prompts)}] v{v}: "
                    f"{len(midi_bytes)} bytes MIDI  {elapsed:.0f}s"
                )
            except Exception as e:
                elapsed = time.time() - t0
                print(f"[modal] [{i+1}/{len(prompts)}] v{v}: ERROR {e}  {elapsed:.0f}s")
                midi_bytes = b""
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
    temperature: float = 1.25,
    threshold: float = 0.99,
    generation_length: int = 1024,
    force: bool = False,
):
    """
    Genera MIDI para un prompt y guarda en out-dir.

    Ejemplos:
        modal run research/research_amadeus_modal.py::main \\
            --prompt "A melodic electronic ambient song with a touch of darkness, \\
set in the key of E major and a 4/4 time signature." \\
            --out-dir evaluation/amadeus/smoke

        modal run research/research_amadeus_modal.py::main \\
            --prompt "A soothing pop song featuring piano and violin." \\
            --temperature 1.5 \\
            --generation-length 2048 \\
            --out-dir /tmp/amadeus_test
    """
    if not prompt:
        print("ERROR: --prompt es obligatorio")
        sys.exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_path.glob("generated_cuda_v*.mid"))
    if existing and not force:
        print(
            f"[skip] Ya existen {len(existing)} ficheros en {out_dir}. "
            "Usa --force para sobreescribir."
        )
        return

    print(f"[main] prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"[main] n_outputs={n_outputs}  temperature={temperature}  "
          f"threshold={threshold}  generation_length={generation_length}")

    results = generate.remote(
        [prompt],
        n_outputs=n_outputs,
        temperature=temperature,
        threshold=threshold,
        generation_length=generation_length,
    )
    output_list = results[0]

    for v, midi_bytes in enumerate(output_list):
        if not midi_bytes:
            print(f"[main] v{v}: sin salida (error en inferencia)")
            continue
        mid_file = out_path / f"generated_cuda_v{v}.mid"
        mid_file.write_bytes(midi_bytes)
        print(f"[main] → {mid_file}  ({len(midi_bytes)} bytes)")

    # Guardar prompt para referencia
    prompt_file = out_path / "prompt.txt"
    if not prompt_file.exists() or force:
        prompt_file.write_text(
            f'Model: Amadeus (longyu1315/Amadeus-S)\n'
            f'Prompt: "{prompt}"\n'
            f'Parameters: temperature={temperature}, threshold={threshold}, '
            f'generation_length={generation_length}, sampling_method=top_p\n'
        )

    print(f"[main] Completado — {len(output_list)} outputs en {out_dir}")


# ---------------------------------------------------------------------------
# Entrypoint: todos los tests del directorio de evaluación
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/amadeus",
    n_outputs: int = 2,
    temperature: float = 1.25,
    threshold: float = 0.99,
    generation_length: int = 1024,
    force: bool = False,
    only: str = "",
):
    """
    Genera MIDI para todos los tests del directorio de evaluación.

    Cada carpeta test*/ debe contener:
      - prompt.txt con línea  Prompt: "..."

    Ejemplo:
        modal run research/research_amadeus_modal.py::eval_all \\
            --eval-dir evaluation/amadeus \\
            --n-outputs 2

        # Solo tests específicos:
        modal run research/research_amadeus_modal.py::eval_all \\
            --eval-dir evaluation/amadeus \\
            --only 1,3,5
    """
    eval_path = Path(eval_dir)

    test_dirs = sorted(
        eval_path.glob("test*"),
        key=lambda p: int(re.sub(r"\D", "", p.name) or "0"),
    )

    if only:
        only_nums = {int(x) for x in only.split(",")}
        test_dirs = [
            d for d in test_dirs
            if int(re.sub(r"\D", "", d.name) or "0") in only_nums
        ]

    if not test_dirs:
        print(f"ERROR: No se encontraron carpetas test* en {eval_dir}")
        sys.exit(1)

    # Recopilar prompts e instrucciones
    prompts = []
    valid_dirs = []
    for td in test_dirs:
        prompt_file = td / "prompt.txt"
        if not prompt_file.exists():
            print(f"[warn] Sin prompt.txt en {td.name}, saltando")
            continue

        text = prompt_file.read_text()
        m = re.search(r'^Prompt:\s*"(.+)"', text, re.MULTILINE)
        if not m:
            print(f"[warn] Sin línea 'Prompt: \"...\"' en {td.name}/prompt.txt, saltando")
            continue

        # Comprobar si ya hay outputs
        existing = sorted(td.glob("generated_cuda_v*.mid"))
        if existing and not force:
            print(f"[skip] {td.name}: ya tiene {len(existing)} outputs. Usa --force para regenerar.")
            continue

        prompts.append(m.group(1))
        valid_dirs.append(td)

    if not prompts:
        print("[eval_all] Nada que generar (todos los tests ya tienen outputs).")
        return

    print(f"[eval_all] Enviando {len(prompts)} prompts × {n_outputs} outputs a Modal...")

    all_results = generate.remote(
        prompts,
        n_outputs=n_outputs,
        temperature=temperature,
        threshold=threshold,
        generation_length=generation_length,
    )

    for td, output_list in zip(valid_dirs, all_results):
        for v, midi_bytes in enumerate(output_list):
            if not midi_bytes:
                print(f"[eval_all] {td.name} v{v}: sin salida")
                continue
            mid_file = td / f"generated_cuda_v{v}.mid"
            mid_file.write_bytes(midi_bytes)
            print(f"[eval_all] {td.name} → {mid_file.name}  ({len(midi_bytes)} bytes)")

    print(f"\n[eval_all] Completado — {len(valid_dirs)} tests generados.")
