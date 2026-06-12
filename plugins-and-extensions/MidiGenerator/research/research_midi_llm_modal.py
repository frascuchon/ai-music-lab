"""
Modal.com inference app para MIDI-LLM (slSeanWU, NeurIPS AI4Music 2025).

Usa vLLM en GPU CUDA (A10G), exactamente como el backend del demo oficial
https://midi-llm-demo.vercel.app — mismos parámetros, mismo pipeline.

Modelo: slseanwu/MIDI-LLM_Llama-3.2-1B (1.4B params, BF16)

Setup (pre-descarga de pesos al Volume, recomendado antes de la primera inferencia):
    modal run research_midi_llm_modal.py::setup

Inferencia (un prompt, 4 outputs):
    modal run research_midi_llm_modal.py::main \
        --prompt "A cheerful rock song with bright electric guitars" \
        --out-dir ../evaluation/midi_llm/comparison_1

Inferencia para todos los comparisons del eval:
    modal run research_midi_llm_modal.py::eval_all \
        --eval-dir ../evaluation/midi_llm \
        --force

GPUs disponibles (--gpu):
    A10G     24 GB, ~$1.10/hr  (default) — suficiente para BF16 1.4B
    L4       24 GB, ~$0.80/hr  — alternativa más barata
    A100-40GB 40 GB, ~$2.10/hr — máxima velocidad

Coste estimado (A10G, vLLM):
    - 4 outputs por prompt: ~15-20s de inferencia → ~$0.006/prompt
    - 12 comparisons × 4 outputs = ~$0.07 total

Referencias:
    Repo:  https://github.com/slSeanWU/MIDI-LLM
    Paper: https://arxiv.org/abs/2511.03942
    Demo:  https://midi-llm-demo.vercel.app
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Volume — caché HuggingFace persistente
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("midi-llm-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"
HF_CACHE = f"{WEIGHTS_MOUNT}/hf_cache"

MODEL_ID = "slseanwu/MIDI-LLM_Llama-3.2-1B"
DEFAULT_GPU = os.environ.get("MIDI_LLM_GPU", "A10G")

SYSTEM_PROMPT = (
    "You are a world-class composer. Please compose some music "
    "according to the following description: "
)

# ---------------------------------------------------------------------------
# Container image — CUDA 12.1 + vLLM + anticipation
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "ffmpeg"])
    .pip_install(
        # Exact versions from the official MIDI-LLM requirements.txt
        "torch==2.8.0",
        "vllm==0.11.0",
        "transformers==4.57.1",
        "tokenizers==0.22.1",
        "huggingface_hub>=0.36",
        "mido>=1.3",
        "pretty_midi>=0.2.10",
        "numpy",
        "safetensors",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .run_commands(
        "pip install 'git+https://github.com/jthickstun/anticipation.git@af37397922665a0fb8d474d7988b0f3755a38d45'",
    )
)

app = modal.App("midi-llm-inference", image=image)


# ---------------------------------------------------------------------------
# Pre-descarga de pesos al Volume
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=600,
    gpu=DEFAULT_GPU,
)
def setup(model_id: str = MODEL_ID):
    """Descarga el modelo a un Volume persistente. Ejecutar una vez."""
    import os
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = HF_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)

    print(f"[setup] Descargando {model_id} → {HF_CACHE} ...")
    snapshot_download(model_id, cache_dir=HF_CACHE)
    weights_vol.commit()
    print("[setup] Pesos descargados y committed al Volume.")


# ---------------------------------------------------------------------------
# Núcleo de inferencia vLLM — exactamente como el demo oficial
# ---------------------------------------------------------------------------
def _generate_vllm(
    prompts: list[str],
    n_outputs: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.98,
    max_tokens: int = 2046,
) -> list[list[list[int]]]:
    """
    Genera n_outputs secuencias de tokens MIDI por prompt usando vLLM.

    Returns: lista de prompts → lista de outputs → lista de token IDs (MIDI vocab, shifted)
    """
    import os
    import time

    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer

    os.environ["HF_HOME"] = HF_CACHE

    # Constantes del modelo
    AMT_GPT2_BOS_ID = 55026
    LLAMA_VOCAB_SIZE = 128256
    ALLOWED_TOKEN_IDS = list(range(LLAMA_VOCAB_SIZE, LLAMA_VOCAB_SIZE + AMT_GPT2_BOS_ID))

    print(f"[vllm] Cargando modelo {MODEL_ID} ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, pad_token="<|eot_id|>", cache_dir=HF_CACHE
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=n_outputs,
        max_tokens=max_tokens,
        allowed_token_ids=ALLOWED_TOKEN_IDS,
    )

    llm = LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        download_dir=HF_CACHE,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    print(f"[vllm] Modelo cargado en {time.time()-t0:.1f}s")

    results = []
    for idx, prompt in enumerate(prompts):
        full_prompt = SYSTEM_PROMPT + prompt + " "
        input_ids = tokenizer(full_prompt, padding=False)["input_ids"]
        input_ids.append(AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE)

        t1 = time.time()
        vllm_input = [TokensPrompt(prompt_token_ids=input_ids)]
        outputs = llm.generate(vllm_input, sampling_params)
        elapsed = time.time() - t1
        print(f"[vllm] [{idx+1}/{len(prompts)}] inferencia: {elapsed:.1f}s")

        prompt_results = []
        for out in outputs[0].outputs:
            # Shift de vuelta al rango MIDI (0..55025)
            midi_tokens = [t - LLAMA_VOCAB_SIZE for t in out.token_ids]
            prompt_results.append(midi_tokens)

        results.append(prompt_results)

    return results


# ---------------------------------------------------------------------------
# Función Modal principal
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=1800,
    gpu=DEFAULT_GPU,
)
def generate(
    prompts: list[str],
    n_outputs: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.98,
    max_tokens: int = 2046,
) -> list[list[bytes]]:
    """
    Genera MIDIs en CUDA vía vLLM. Retorna lista de prompts → lista de bytes MIDI.

    Cada MIDI que falla validación (token inválido o notas simultáneas excesivas)
    es reemplazado por b'' en el output.
    """
    import sys
    import tempfile

    # anticipation está instalada en la imagen
    from anticipation.convert import events_to_midi

    # Ejecutar generación vLLM
    all_tokens = _generate_vllm(
        prompts=prompts,
        n_outputs=n_outputs,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    def tokens_to_midi_bytes(midi_tokens: list[int]) -> bytes:
        """Convierte token IDs a bytes MIDI, o b'' si la conversión falla."""
        try:
            midi_obj = events_to_midi(midi_tokens)
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                midi_obj.save(f.name)
                f.flush()
                with open(f.name, "rb") as mf:
                    return mf.read()
        except Exception as e:
            print(f"  [warn] conversión MIDI falló: {e}")
            return b""

    results = []
    for prompt_idx, prompt_tokens in enumerate(all_tokens):
        prompt_midis = []
        ok = 0
        for out_idx, midi_tokens in enumerate(prompt_tokens):
            midi_bytes = tokens_to_midi_bytes(midi_tokens)
            prompt_midis.append(midi_bytes)
            if midi_bytes:
                ok += 1
        print(f"[output] prompt {prompt_idx+1}: {ok}/{n_outputs} MIDIs válidos")
        results.append(prompt_midis)

    return results


# ---------------------------------------------------------------------------
# Entrypoint: inferencia de un único prompt
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = "A cheerful rock song with bright electric guitars",
    out_dir: str = "./midi_llm_output",
    n_outputs: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.98,
    max_tokens: int = 2046,
    gpu: str = DEFAULT_GPU,
):
    """
    Genera n_outputs MIDIs para un prompt.
    Los guarda como generated_cuda_v0.mid .. generated_cuda_v{n-1}.mid en out_dir.
    """
    import time

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[main] Prompt: {prompt[:80]}...")
    print(f"[main] n_outputs={n_outputs}  temperature={temperature}  top_p={top_p}")
    print(f"[main] Output dir: {out_path.absolute()}")

    t0 = time.time()
    results = generate.with_options(gpu=gpu).remote(
        prompts=[prompt],
        n_outputs=n_outputs,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0

    midi_list = results[0]  # first (only) prompt
    saved = 0
    for i, midi_bytes in enumerate(midi_list):
        if midi_bytes:
            out_file = out_path / f"generated_cuda_v{i}.mid"
            out_file.write_bytes(midi_bytes)
            print(f"  ✓ {out_file}")
            saved += 1
        else:
            print(f"  ✗ output {i}: falló conversión MIDI")

    print(f"\n[done] {saved}/{n_outputs} MIDIs guardados en {elapsed:.1f}s total")


# ---------------------------------------------------------------------------
# Entrypoint: regenerar todos los comparisons del directorio de evaluación
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/midi_llm",
    n_outputs: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.98,
    max_tokens: int = 2046,
    gpu: str = DEFAULT_GPU,
    force: bool = False,
):
    """
    Regenera generated_cuda_v{0..n-1}.mid para todos los comparison_N en eval_dir.

    Lee el prompt de prompt_demo.txt en cada carpeta.
    Salta carpetas donde generated_cuda_v0.mid ya existe (a menos que --force).
    """
    import time

    eval_path = Path(eval_dir)
    comparisons = sorted(eval_path.glob("comparison_*"), key=lambda p: int(p.name.split("_")[1]))

    if not comparisons:
        print(f"[eval_all] No se encontraron carpetas comparison_* en {eval_dir}")
        return

    # Filtrar los que necesitan regenerarse
    to_run = []
    for comp in comparisons:
        prompt_file = comp / "prompt_demo.txt"
        out_v0 = comp / "generated_cuda_v0.mid"
        if not prompt_file.exists():
            print(f"  SKIP {comp.name}: sin prompt_demo.txt")
            continue
        if out_v0.exists() and not force:
            print(f"  SKIP {comp.name}: ya existe generated_cuda_v0.mid (--force para sobreescribir)")
            continue
        to_run.append(comp)

    if not to_run:
        print("[eval_all] Nada que generar.")
        return

    print(f"\n[eval_all] Generando {len(to_run)} comparisons × {n_outputs} outputs ...")
    print(f"           GPU={gpu}  T={temperature}  top_p={top_p}  max_tokens={max_tokens}\n")

    # Leer prompts
    prompts = []
    for comp in to_run:
        raw = (comp / "prompt_demo.txt").read_text(encoding="utf-8").strip()
        prompt_oneline = " ".join(raw.split())
        prompts.append(prompt_oneline)

    # Lanzar inferencia en batch
    t0 = time.time()
    results = generate.with_options(gpu=gpu).remote(
        prompts=prompts,
        n_outputs=n_outputs,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0

    # Guardar resultados
    total_saved = 0
    for comp, midi_list in zip(to_run, results):
        saved = 0
        for i, midi_bytes in enumerate(midi_list):
            if midi_bytes:
                out_file = comp / f"generated_cuda_v{i}.mid"
                out_file.write_bytes(midi_bytes)
                saved += 1
        total_saved += saved
        print(f"  {comp.name}: {saved}/{n_outputs} MIDIs guardados")

    print(f"\n[done] {total_saved} MIDIs totales en {elapsed:.1f}s")
    print("Renderiza MP3 con: bash evaluation/midi_llm/render_cuda_mp3.sh")
