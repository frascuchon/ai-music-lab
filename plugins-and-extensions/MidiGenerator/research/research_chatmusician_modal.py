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
# Prompt adapter — small, SELF-HOSTED LLM that rewrites the user's free-text
# prompt into ChatMusician's validated harmonization-instruction style
# before a seeded request is sent to the model (see _adapt_prompt_for_seed /
# adapt_prompt below). Runs entirely inside its own Modal container — no
# external API, no secret to configure — sharing the same weights Volume as
# ChatMusician for its (much smaller) model cache.
# ---------------------------------------------------------------------------
ADAPTER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # Apache-2.0, ~3GB fp16, fast

adapt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "huggingface_hub>=0.24",
        "accelerate>=0.34",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)


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


@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=600,
)
def setup_adapter():
    """Descarga los pesos del prompt-adapter (~3 GB) al Volume persistente.
    Opcional — sin esto, la primera llamada a adapt_prompt() descarga los
    pesos on-demand (más lenta pero funciona igual)."""
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = HF_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)

    print(f"[setup_adapter] Descargando {ADAPTER_MODEL_ID} (~3 GB) → {HF_CACHE} ...")
    path = snapshot_download(ADAPTER_MODEL_ID, cache_dir=HF_CACHE)
    print(f"[setup_adapter] Descargado en {path}")

    weights_vol.commit()
    print("[setup_adapter] Pesos committed al Volume.")


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


# All valid ABC 2.1 information-field letters (abcnotation.com/wiki/abc:standard:v2.1
# section 3): A B C D F G H I K L M N O P Q R S T U V W X Z (E/J/Y are not fields).
# The fallback below used to check only a subset (TMKLQRBCGSPFHZONU) — when a
# harmonization response starts with one of the missing letters (seen in practice:
# a model that echoes the seed's own "V:1" voice header, or answers with an "A:"
# field) the whole block was invisible to both regexes and _extract_abc raised
# "No ABC notation found" even though a usable field block followed a line later.
_ABC_FIELD_LETTERS = "ABCDFGHIKLMNOPQRSTUVWXZ"


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
        rf"^([{_ABC_FIELD_LETTERS}]:[^\n]*(?:\n[^\n]*)*)",
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
    temperature: float = 0.5,
) -> list[list[tuple[bytes, str]]]:
    """
    Genera n_outputs variantes para cada prompt.

    temperature: 0.5 default — NOT the bare model-card value (0.2). Our own
    RESEARCH.md ChatMusician evaluation (harmonization/chord/form/motif
    tests) found 0.2 too conservative for abstract/seeded prompts (model
    answers in academic prose instead of ABC — MusicPile is full of music
    theory text) and settled on 0.5 as the "GenerationConfig recomendado
    (ajustado tras evaluación)". See main()'s automatic retry for the case a
    caller explicitly passes an even lower temperature.

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
                # Collapse embedded newlines (e.g. abc2midi's stderr, or the
                # first 500 chars of a malformed response in _extract_abc's
                # ValueError) onto one line so the whole cause survives as a
                # single line for the orchestrator (midigen.py) to capture
                # and surface to the user — a bare "ERROR" keyword buried in
                # a "[modal] ..." line used to be invisible to it, and this
                # per-candidate failure alone didn't fail the overall run,
                # so the only thing reaching the user was the unrelated,
                # generic "No .mid files found" from the output-collection
                # stage several steps later.
                err_str = str(e).replace("\n", " | ")
                print(f"ERROR: [{i+1}/{len(prompts)}] v{v} generation failed: "
                      f"{err_str} ({elapsed:.0f}s)")
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
    temperature: float = 0.5,
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

    try:
        instruction = _build_instruction(prompt, input_file)
    except RuntimeError as e:
        # Seed conversion failed (e.g. midi2abc missing/failing on the
        # exported take) — fail fast with a one-line, greppable cause
        # instead of silently running unseeded and wasting a GPU call.
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"[main] n_outputs={n_outputs}")
    print(f"[main] Instruction (primeros 200 chars): {instruction[:200]}")

    output_list = generate.remote([instruction], n_outputs=n_outputs, temperature=temperature)[0]

    # A conservative temperature occasionally makes the model answer in
    # music-theory prose instead of ABC notation — MusicPile (its training
    # data) is full of theory text, and RESEARCH.md's own ChatMusician
    # evaluation documents this exact failure mode plus the fix ("subir a
    # 0.5-0.7 mejora la tasa de éxito para prompts abstractos"; test06 only
    # passed once raised to 0.5). Rather than surface a first failure at
    # <=0.5 straight away, retry once at 0.7 (the documented upper end of
    # that range — higher still tends toward the "chaotic"/noise failure
    # mode instead) before giving up. One extra ~30-90s GPU call is cheap
    # next to forcing the user to manually retune and rerun from REAPER.
    if not any(midi for midi, _ in output_list) and temperature <= 0.5:
        retry_temp = 0.7
        print(f"[main] All {n_outputs} candidate(s) failed at temperature={temperature:.2f} — "
              f"retrying once at temperature={retry_temp:.2f} (mid-range temperature often "
              "rescues responses that come back as prose instead of ABC notation).")
        output_list = generate.remote([instruction], n_outputs=n_outputs, temperature=retry_temp)[0]

    n_saved = 0
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
        n_saved += 1

    if n_saved == 0:
        # Every candidate failed inside generate() (each already printed its
        # own "ERROR: ..." line above with the actual cause — ABC extraction,
        # abc2midi, etc.). Deliberately NOT printing a new "ERROR:" line
        # here: midigen.py surfaces the LAST such line it saw, and the most
        # specific per-candidate cause above is more useful to the user than
        # a generic wrapper would be. Exiting non-zero — instead of silently
        # returning 0 as before — is what makes midigen.py look at that
        # captured line at all, instead of falling through to its own
        # generic, unrelated "No .mid files found in <out_dir>" message.
        print(f"[main] 0/{len(output_list)} candidates produced a valid MIDI "
              "output — see the ERROR line(s) above for the per-candidate cause.")
        sys.exit(1)

    print(f"[main] Completado — {n_saved}/{len(output_list)} outputs en {out_dir}")


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
# Character budget for the seed ABC block. LLaMA2 7B (ChatMusician's base) has a
# 4096-token context and generation reserves max_new_tokens=1536 of it, leaving
# roughly 2500 tokens for "Human: {prompt}\n{seed abc}". Our own evaluation notes
# (evaluation/chatmusician/test12) already diagnosed a real harmonization failure
# ("0 MIDIs ... posiblemente el input más largo excede la ventana de atención
# efectiva") on a 16-bar seed — and panel.lua lets a user attach an entire track
# with no length warning (unlike the Anticipatory seed, which does warn). ~2400
# chars is a generous margin (~3-4 chars/ABC token) that keeps a full short
# melody intact while capping pathological whole-song seeds before they push the
# model past its effective attention window and break extraction for every output.
_MAX_SEED_ABC_CHARS = 2400


def _truncate_seed_abc(abc_text: str, max_chars: int = _MAX_SEED_ABC_CHARS) -> str:
    """Cap the seed ABC block to max_chars, cutting at the last full line so the
    result stays syntactically valid ABC (no half-written note/bar)."""
    if len(abc_text) <= max_chars:
        return abc_text
    truncated = abc_text[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > 0:
        truncated = truncated[:last_nl]
    print(f"[build_instruction] Seed ABC truncated: {len(abc_text)} -> {len(truncated)} chars "
          "(long seeds can exceed ChatMusician's effective attention window and "
          "break extraction for every output — see test12 in evaluation/chatmusician)")
    return truncated


_SEED_BRIDGE = "Here is the musical excerpt in ABC notation:"

# System prompt for the adapter: teaches it ChatMusician's known-good
# instruction shape, verified empirically (2026-09-14) against a real Modal
# run — the official web demo's harmonization examples (which DO work) all
# share the same shape: imperative mood, one sentence, ending with an
# explicit reference to the attached excerpt. A short/casual user prompt
# like "simplify the verses" that skips that reference makes the model
# respond with clarifying questions instead of ABC notation, regardless of
# temperature.
_ADAPT_SYSTEM_PROMPT = """You rewrite short, casual music-editing requests into the precise instruction style that the ChatMusician model was fine-tuned on for melody harmonization/elaboration. ChatMusician only reliably follows instructions shaped like these official, verified-working examples:
- "Formulate chord combinations to increase the harmonic complexity of the specified musical excerpt."
- "Design a fitting succession of chords that blend well with the provided musical score."
- "Develop a series of chord pairings that amplify the harmonious elements in the given music piece."

Rewrite the user's request so it:
1. Preserves their actual musical intent (harmonize, simplify, add chords, change rhythm/texture, etc.) — do not silently change what they asked for.
2. Ends with an explicit reference to the attached excerpt (e.g. "...of the following musical excerpt.").
3. Is one sentence, imperative mood, no preamble, no explanation, no quotes.

Reply with ONLY the rewritten instruction, nothing else."""


@app.function(
    image=adapt_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=300,
    gpu="T4",
)
def adapt_prompt(user_prompt: str) -> str:
    """Rewrite user_prompt into ChatMusician's validated instruction style
    using a small, self-hosted instruction-tuned LLM (Qwen2.5-1.5B-Instruct)
    — runs entirely inside this Modal container on a cheap T4 GPU. No
    external API, no secret to configure; weights are cached in the same
    Volume as ChatMusician's (see setup_adapter() to pre-warm)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = HF_CACHE

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_ID, cache_dir=HF_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        ADAPTER_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        cache_dir=HF_CACHE,
    ).eval()

    messages = [
        {"role": "system", "content": _ADAPT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=False,  # deterministic — this is a rewrite task, not a creative one
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(
        output[0][input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    # Small instruct models sometimes wrap the answer in quotes despite the
    # system prompt saying not to — strip them rather than pass them through.
    return text.strip("\"' ")


def _adapt_prompt_for_seed(prompt: str) -> str:
    """Local-side wrapper around adapt_prompt.remote() with a graceful
    fallback: prompt adaptation is a reliability BOOSTER, not a hard
    requirement, so any failure (adapter container error, empty reply,
    cold-start timeout) must fall back to the user's original prompt rather
    than break the run — a seeded generation that used to at least attempt
    to run unassisted must keep doing so even if the adapter is unavailable.
    """
    try:
        adapted = adapt_prompt.remote(prompt).strip()
    except Exception as e:
        print(f"[warn] Prompt adaptation skipped ({e}). Using the original prompt.")
        return prompt
    if not adapted:
        print("[warn] Prompt adaptation returned empty text — using original prompt.")
        return prompt
    print(f"[build_instruction] Prompt adapted for ChatMusician: {adapted[:150]}")
    return adapted


def _build_instruction(prompt: str, input_file: str = "") -> str:
    """
    Construye la instrucción que va dentro de "Human: {instruction} </s> Assistant: ".

    input_file puede ser:
      - vacío   → devuelve el prompt tal cual
      - .abc    → lee el ABC y lo concatena al prompt
      - .mid    → convierte MIDI→ABC (via tools.midi_abc.midi_to_abc_text) y concatena

    La detección es automática por extensión del fichero. En ambos casos el ABC
    resultante se trunca a _MAX_SEED_ABC_CHARS (ver _truncate_seed_abc) y se
    antepone con _SEED_BRIDGE.

    Por qué el bridge: la web demo oficial (evaluation/chatmusician/test10-12)
    solo tiene éxito cuando el PROMPT DEL USUARIO ya termina refiriéndose
    explícitamente al material adjunto ("...of the provided musical score",
    "...of the specified musical excerpt"). Verificado empíricamente
    (2026-09-14): con un prompt corto que no hace esa referencia (p.ej.
    "simplify the verses") y el seed simplemente concatenado con un salto de
    línea (sin bridge), el modelo no entiende que el bloque ABC es el
    material a transformar y responde con preguntas aclaratorias ("What is
    the last note of this music? Is there a specific key signature...")
    en vez de generar — cero ABC en la respuesta, para cualquier prompt que
    el usuario escriba. Insertar una frase-puente fija después del prompt
    ata el bloque ABC a la instrucción sin depender de que el usuario
    redacte su prompt con esa referencia explícita.

    El bridge por sí solo no basta si el prompt del usuario está muy lejos
    del estilo de instrucción con el que ChatMusician fue afinado (p.ej.
    "simplify the verses" verificado empíricamente: sigue fallando incluso
    con el bridge). Por eso, cuando hay seed, el prompt pasa primero por
    _adapt_prompt_for_seed — reescribe la petición del usuario al estilo
    validado usando un LLM pequeño self-hosted (Qwen2.5-1.5B-Instruct, ver
    adapt_prompt) que corre dentro de su propio contenedor Modal — sin API
    externa ni secret que configurar. Si el contenedor del adaptador falla
    o tarda demasiado en arrancar (cold-start), se usa el prompt original
    sin más — es un refuerzo de fiabilidad, no un requisito duro.
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
        # No silent fallback here: a seed the user explicitly picked that
        # fails to convert (midi2abc missing/failing) must abort the run
        # loudly, not proceed as an unseeded prompt that then fails ABC
        # extraction 3 stages later with no clue why.
        abc_text = midi_to_abc_text(str(in_p)).strip()

    else:
        print(f"[warn] Extensión no reconocida en input_file: {suffix}. Ignorando.")
        return prompt

    abc_text = _truncate_seed_abc(abc_text)
    prompt = _adapt_prompt_for_seed(prompt)
    return f"{prompt}\n{_SEED_BRIDGE}\n{abc_text}"
