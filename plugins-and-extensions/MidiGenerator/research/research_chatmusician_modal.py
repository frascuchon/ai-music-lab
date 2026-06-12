"""
Modal.com inference app para ChatMusician (m-a-p/ChatMusician, ISMIR 2024).

ChatMusician es un LLaMA 2 7B continually pretrained + SFT sobre ABC notation.
Genera ABC notation (representación textual de partitura), que se convierte a
MIDI mediante abc2midi (herramienta oficial del web demo).

Pipeline completo (bidireccional MIDI ↔ ABC integrado):

    [MIDI input]  ─→ midi2abc ─→ ABC text ─┐
    [ABC input]   ─────────────────────────┤
    [sin input]   ─────────────────────────┤
                                           ↓
                    "Human: {prompt}\\n{abc} </s> Assistant: "
                                           ↓
                               ChatMusician LLM (CUDA A10G)
                                           ↓
                    regex r'(X:\\d+\\n(?:[^\\n]*\\n)+)'
                                           ↓
                               abc2midi  (en container)
                                           ↓
                    generated_cuda_v0.mid + generated_cuda_v0.abc

La conversión MIDI→ABC usa tools/midi_abc.py (localmente, antes de enviar a Modal).
La conversión ABC→MIDI usa abc2midi dentro del container Modal (apt install abcmidi).

Modelo: m-a-p/ChatMusician (~13 GB safetensors, fp16 en CUDA A10G 24GB)
Repo:   https://github.com/hf-lin/ChatMusician
Paper:  https://arxiv.org/abs/2402.16153  (ISMIR 2024)
Demo:   https://ezmonyi.github.io/ChatMusician/

Template de prompt (verbatim de model/infer/predict.py):
    "Human: {instruction} </s> Assistant: "

GenerationConfig (verbatim del model card y chatmusician_web_demo.py):
    temperature=0.2, top_k=40, top_p=0.9, do_sample=True,
    num_beams=1, repetition_penalty=1.1, min_new_tokens=10, max_new_tokens=1536

Post-procesado (verbatim de chatmusician_web_demo.py):
    abc_pattern = r'(X:\\d+\\n(?:[^\\n]*\\n)+)'
    → subprocess.run(["abc2midi", abc_file, "-o", midi_file])

Setup (pre-descarga de pesos al Volume, ejecutar una vez):
    modal run research/research_chatmusician_modal.py::setup

Inferencia libre:
    modal run research/research_chatmusician_modal.py::main \\
        --prompt "Develop a tune influenced by Bach's compositions." \\
        --out-dir evaluation/chatmusician/smoke

Inferencia con condicionante — auto-detecta .abc o .mid:
    modal run research/research_chatmusician_modal.py::main \\
        --prompt "Construct smooth-flowing chord progressions for the supplied music." \\
        --input-file evaluation/chatmusician/test10/input_abc.txt \\
        --out-dir evaluation/chatmusician/test10

    modal run research/research_chatmusician_modal.py::main \\
        --prompt "Formulate chord combinations to increase the harmonic complexity." \\
        --input-file evaluation/text2midi/test1/reference_official.mid \\
        --out-dir /tmp/cm_harmonize

Benchmark completo:
    modal run research/research_chatmusician_modal.py::eval_all \\
        --eval-dir evaluation/chatmusician \\
        --n-outputs 2

Prerequisitos locales (para conversión MIDI→ABC):
    brew install abcmidi   # provee midi2abc y abc2midi

GPUs disponibles (--gpu, por defecto A10G):
    A10G     24 GB, ~$1.10/hr  (default) — margen cómodo para LLaMA2 7B fp16 (~14 GB)
    A100-40GB 40 GB, ~$2.10/hr — más rápido, innecesario para este modelo

Coste estimado (A10G, n_outputs=2):
    - 1 output con max_new_tokens=1536: ~30-90s → ~$0.01-0.03/output
    - 12 tests × 2 outputs = ~$0.24-0.72 total
"""

import os
import re
import sys
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Volume — caché HuggingFace persistente
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("chatmusician-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"
HF_CACHE = f"{WEIGHTS_MOUNT}/hf_cache"

MODEL_ID = "m-a-p/ChatMusician"

DEFAULT_GPU = os.environ.get("CHATMUSICIAN_GPU", "A10G")

# ---------------------------------------------------------------------------
# Container image — CUDA 12 + PyTorch + abcmidi + dependencias HF
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["abcmidi", "ffmpeg"])
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "huggingface_hub>=0.24",
        "accelerate>=0.34",
        "safetensors>=0.4",
        "sentencepiece>=0.2",
        "mido>=1.3",
        "pretty_midi>=0.2.10",
        "psutil>=6.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

app = modal.App("chatmusician-inference", image=image)


# ---------------------------------------------------------------------------
# Pre-descarga de pesos al Volume
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=1800,
    gpu=DEFAULT_GPU,
)
def setup():
    """Descarga los pesos del modelo a un Volume persistente. Ejecutar una vez."""
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = HF_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)

    print(f"[setup] Descargando {MODEL_ID} (~13 GB) → {HF_CACHE} ...")
    path = snapshot_download(MODEL_ID, cache_dir=HF_CACHE)
    print(f"[setup] Descargado en {path}")

    weights_vol.commit()
    print("[setup] Pesos committed al Volume.")


# ---------------------------------------------------------------------------
# Núcleo de inferencia
# ---------------------------------------------------------------------------
def _load_model():
    """Carga ChatMusician (LLaMA2 7B fp16) en CUDA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = HF_CACHE

    print("[load] Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE
    )

    print("[load] Cargando modelo (fp16, device_map=cuda)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        resume_download=True,
        cache_dir=HF_CACHE,
    ).eval()

    print("[load] Modelo listo.")
    return model, tokenizer


def _abc_to_midi_bytes(abc_text: str) -> bytes:
    """
    Convierte ABC notation a bytes MIDI usando abc2midi.

    Corre dentro del container Modal (abcmidi instalado via apt).
    Para uso local importa tools.midi_abc.abc_to_midi_bytes.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        abc_path = Path(tmpdir) / "score.abc"
        midi_path = Path(tmpdir) / "score.mid"
        abc_path.write_text(abc_text, encoding="utf-8")

        result = subprocess.run(
            ["abc2midi", str(abc_path), "-o", str(midi_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"abc2midi failed:\n{result.stderr}")
        if not midi_path.exists():
            raise RuntimeError("abc2midi no produjo fichero MIDI")

        return midi_path.read_bytes()


def _extract_abc(response: str) -> str:
    """
    Extrae la primera sección ABC de la respuesta del modelo.

    Estrategia en dos pasos:
    1. Regex oficial del web demo (X:\\d+\\n...): captura la mayoría de casos.
    2. Fallback: el modelo a veces omite el encabezado X: pero genera los
       demás campos (M:, L:, K:). Se captura el bloque y se antepone X:1.
       El bloque debe contener al menos M: o K: para ser ABC válido.
    """
    # 1. Regex oficial (verbatim de chatmusician_web_demo.py)
    matches = re.findall(r"(X:\d+\n(?:[^\n]*\n)+)", response + "\n")
    if matches:
        return matches[0]

    # 2. Fallback: bloque con headers ABC reconocidos pero sin X:
    #    Busca la primera línea con patrón "Letra: contenido" (campo ABC)
    m = re.search(
        r"^([TMKLQRBCGSPFHZONU]:[^\n]*(?:\n[^\n]*)*)",
        response + "\n",
        re.MULTILINE,
    )
    if m:
        candidate = m.group(0) + "\n"
        if re.search(r"^[MK]:", candidate, re.MULTILINE):
            return "X:1\n" + candidate

    raise ValueError(f"No ABC notation found in response:\n{response[:500]}")


def _infer_one(model, tokenizer, instruction: str, temperature: float = 0.2) -> tuple[bytes, str]:
    """
    Genera MIDI para una instrucción y devuelve (midi_bytes, abc_text).

    Prompt template (verbatim de predict.py):
        "Human: {instruction} </s> Assistant: "

    GenerationConfig (verbatim del model card):
        temperature=0.2 (default), top_k=40, top_p=0.9, repetition_penalty=1.1

    Para prompts que producen respuestas en modo ensayo (teoría musical en lugar de ABC),
    incrementar temperature a 0.5-0.7 mejora la tasa de éxito.
    """
    import torch
    from transformers import GenerationConfig

    prompt = f"Human: {instruction} </s> Assistant: "

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        min_new_tokens=10,
        max_new_tokens=1536,
    )

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=generation_config,
        )

    # Eliminar tokens del input (verbatim de predict.py)
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    abc_text = _extract_abc(response)
    midi_bytes = _abc_to_midi_bytes(abc_text)
    return midi_bytes, abc_text


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
    temperature: float = 0.2,
) -> list[list[tuple[bytes, str]]]:
    """
    Genera n_outputs variantes para cada prompt.

    temperature: 0.2 (default, verbatim model card) — subir a 0.5-0.7 para
    prompts abstractos que producen respuestas en modo ensayo.

    Returns: list[prompt] → list[output] → (midi_bytes, abc_text)
    """
    import time

    model, tokenizer = _load_model()
    print(f"[modal] n_prompts={len(prompts)}  n_outputs={n_outputs}  temperature={temperature}")

    results = []
    for i, prompt in enumerate(prompts):
        prompt_results = []
        for v in range(n_outputs):
            t0 = time.time()
            try:
                midi_bytes, abc_text = _infer_one(model, tokenizer, prompt, temperature)
                elapsed = time.time() - t0
                print(
                    f"[modal] [{i+1}/{len(prompts)}] v{v}: {len(midi_bytes)} bytes MIDI  "
                    f"{len(abc_text)} chars ABC  {elapsed:.0f}s"
                )
            except Exception as e:
                elapsed = time.time() - t0
                print(f"[modal] [{i+1}/{len(prompts)}] v{v}: ERROR {e}  {elapsed:.0f}s")
                midi_bytes, abc_text = b"", ""
            prompt_results.append((midi_bytes, abc_text))
        results.append(prompt_results)

    return results


# ---------------------------------------------------------------------------
# Entrypoint: un prompt → out-dir
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = "",
    input_file: str = "",
    out_dir: str = ".",
    n_outputs: int = 2,
    temperature: float = 0.2,
    force: bool = False,
):
    """
    Genera MIDI para un prompt y guarda .mid y .abc en out-dir.

    --input-file acepta tanto .abc (ABC notation) como .mid (MIDI auto-convertido a ABC).

    Ejemplos:
        modal run research/research_chatmusician_modal.py::main \\
            --prompt "Develop a tune influenced by Bach's compositions." \\
            --out-dir /tmp/cm_smoke

        modal run research/research_chatmusician_modal.py::main \\
            --prompt "Construct smooth-flowing chord progressions." \\
            --input-file evaluation/chatmusician/test10/input_abc.txt \\
            --out-dir evaluation/chatmusician/test10

        modal run research/research_chatmusician_modal.py::main \\
            --prompt "Formulate chord combinations to increase the harmonic complexity." \\
            --input-file evaluation/text2midi/test1/reference_official.mid \\
            --out-dir /tmp/cm_harmonize
    """
    if not prompt:
        print("ERROR: --prompt es obligatorio")
        sys.exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_path.glob("generated_cuda_v*.mid"))
    if existing and not force:
        print(f"[skip] Ya existen {len(existing)} ficheros en {out_dir}. Usa --force para sobreescribir.")
        return

    instruction = _build_instruction(prompt, input_file)

    print(f"[main] n_outputs={n_outputs}")
    print(f"[main] Instruction (primeros 200 chars): {instruction[:200]}")

    results = generate.remote([instruction], n_outputs=n_outputs, temperature=temperature)
    output_list = results[0]

    for v, (midi_bytes, abc_text) in enumerate(output_list):
        if not midi_bytes:
            print(f"[main] v{v}: sin salida (error en inferencia)")
            continue
        mid_file = out_path / f"generated_cuda_v{v}.mid"
        abc_file = out_path / f"generated_cuda_v{v}.abc"
        mid_file.write_bytes(midi_bytes)
        abc_file.write_text(abc_text, encoding="utf-8")
        print(f"[main] → {mid_file}  ({len(midi_bytes)} bytes)")
        print(f"[main] → {abc_file}  ({len(abc_text)} chars)")

    print(f"[main] Completado — {len(output_list)} outputs en {out_dir}")


# ---------------------------------------------------------------------------
# Entrypoint: todos los tests del directorio de evaluación
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/chatmusician",
    n_outputs: int = 2,
    temperature: float = 0.2,
    force: bool = False,
    only: str = "",
):
    """
    Genera MIDI para todos los tests del directorio de evaluación.

    Cada carpeta test*/ debe contener:
      - prompt.txt   con línea  Prompt: "..."
      - input_abc.txt (opcional) — ABC notation concatenada al prompt
      - input_midi.mid (opcional) — MIDI convertido a ABC automáticamente

    Ejemplo:
        modal run research/research_chatmusician_modal.py::eval_all \\
            --eval-dir evaluation/chatmusician \\
            --n-outputs 2

        # Solo tests específicos:
        modal run research/research_chatmusician_modal.py::eval_all \\
            --eval-dir evaluation/chatmusician \\
            --only 1,3,7
    """
    eval_path = Path(eval_dir)

    test_dirs = sorted(
        eval_path.glob("test*"),
        key=lambda p: int(re.sub(r"\D", "", p.name) or "0"),
    )

    if only:
        only_nums = {int(x) for x in only.split(",")}
        test_dirs = [d for d in test_dirs if int(re.sub(r"\D", "", d.name) or "0") in only_nums]

    if not test_dirs:
        print(f"ERROR: No se encontraron carpetas test* en {eval_dir}")
        sys.exit(1)

    instructions = []
    valid_dirs = []
    for td in test_dirs:
        prompt_file = td / "prompt.txt"
        if not prompt_file.exists():
            print(f"[warn] Sin prompt.txt en {td.name}, saltando")
            continue

        text = prompt_file.read_text()
        m = re.search(r'^Prompt:\s*"(.+)"', text, re.MULTILINE)
        if not m:
            print(f"[warn] No se pudo parsear Prompt: en {td.name}, saltando")
            continue

        prompt = m.group(1)

        existing = sorted(td.glob("generated_cuda_v*.mid"))
        if existing and not force:
            print(f"[skip] {td.name}: ya tiene {len(existing)} outputs (usa --force)")
            continue

        # Detectar input: MIDI tiene prioridad sobre ABC si ambos existen
        if (td / "input_midi.mid").exists():
            input_file = str(td / "input_midi.mid")
        elif (td / "input_abc.txt").exists():
            input_file = str(td / "input_abc.txt")
        else:
            input_file = ""
        instruction = _build_instruction(prompt, input_file)

        instructions.append(instruction)
        valid_dirs.append(td)

    if not instructions:
        print("[eval_all] Nada que generar.")
        return

    print(f"[eval_all] {len(instructions)} tests a generar: {[d.name for d in valid_dirs]}")
    print(f"[eval_all] n_outputs={n_outputs}  temperature={temperature}")

    results = generate.remote(instructions, n_outputs=n_outputs, temperature=temperature)

    for td, output_list in zip(valid_dirs, results):
        for v, (midi_bytes, abc_text) in enumerate(output_list):
            if not midi_bytes:
                print(f"[eval_all] {td.name} v{v}: sin salida (error en inferencia)")
                continue
            mid_file = td / f"generated_cuda_v{v}.mid"
            abc_file = td / f"generated_cuda_v{v}.abc"
            mid_file.write_bytes(midi_bytes)
            abc_file.write_text(abc_text, encoding="utf-8")
            print(f"[eval_all] → {td.name}/generated_cuda_v{v}.mid  ({len(midi_bytes)} bytes)")

    print(f"\n[eval_all] Completado — {len(valid_dirs)} tests × {n_outputs} outputs.")


# ---------------------------------------------------------------------------
# Helper local — construye la instrucción completa (prompt + condicionante)
# ---------------------------------------------------------------------------
def _build_instruction(prompt: str, input_file: str = "") -> str:
    """
    Construye la instrucción que va dentro de "Human: {instruction} </s> Assistant: ".

    input_file puede ser:
      - vacío   → devuelve el prompt tal cual
      - .abc    → lee el ABC y lo concatena al prompt
      - .mid    → convierte MIDI→ABC (via tools.midi_abc.midi_to_abc_text) y concatena

    La detección es automática por extensión del fichero.
    """
    if not input_file:
        return prompt

    in_p = Path(input_file)
    if not in_p.exists():
        print(f"[warn] input_file no existe: {input_file}")
        return prompt

    suffix = in_p.suffix.lower()

    if suffix in (".abc", ".txt"):
        abc_text = in_p.read_text(encoding="utf-8").strip()

    elif suffix in (".mid", ".midi"):
        # Importación lazy — sólo en entrypoints locales, no dentro del container
        sys.path.insert(0, str(Path(__file__).parent))
        from tools.midi_abc import midi_to_abc_text
        print(f"[build_instruction] Convirtiendo MIDI→ABC: {in_p.name}")
        try:
            abc_text = midi_to_abc_text(str(in_p)).strip()
        except RuntimeError as e:
            print(f"[warn] {e}\n  Falling back: usando prompt sin condicionante")
            return prompt

    else:
        print(f"[warn] Extensión no reconocida en input_file: {suffix}. Ignorando.")
        return prompt

    return f"{prompt}\n{abc_text}"
