"""
Modal.com inference app for SAM-Audio source separation.

Runs facebook/sam-audio-large (or other sizes) on a GPU cloud instance.
The HuggingFace model cache is stored in a persistent Modal Volume to avoid
re-downloading the 14 GB large model on every run (~downloaded once, cached forever).

Usage
-----
# Separate trumpet from a local audio file (large model, default):
modal run tools/modal_sam_audio.py --input song.mp3 --prompt "jazz trumpet"

# Confidence gate: suppress chunks where the separated output doesn't match the
# prompt (e.g. piano intro when no trumpet is playing). Use --confidence-threshold
# to set the minimum CLAP text-audio cosine similarity; chunks below it are
# replaced with silence. Start around 0.20 and tune by looking at printed scores.
modal run tools/modal_sam_audio.py --input song.mp3 --prompt "trumpet" \\
    --confidence-threshold 0.20

# Audio query: generate N candidates per chunk, pick the one most similar to a
# reference recording using CLAP audio-vs-audio similarity:
modal run tools/modal_sam_audio.py --input song.mp3 --prompt "jazz trumpet" \\
    --audio-query /datasets/Trompeta/ref_trumpet.wav --reranking-candidates 4

# Custom model / GPU (--gpu must appear before Typer-processed args):
modal run tools/modal_sam_audio.py --gpu A100 --input song.mp3 --prompt "jazz trumpet" \\
    --model facebook/sam-audio-large

# Override chunk / ODE params:
modal run tools/modal_sam_audio.py --input song.mp3 --prompt "jazz trumpet" \\
    --chunk-dur 15 --overlap 2 --ode-steps 64

Cost estimate (Modal on-demand):
  A10G  24 GB VRAM  ~$1.10/hr   large model ~5 min/track  → ~$0.09/track
  A100  40 GB VRAM  ~$2.78/hr   large model ~3 min/track  → ~$0.14/track
"""

import base64
import os
from pathlib import Path

import modal

hf_secret = modal.Secret.from_name("huggingface-secret")

# ---------------------------------------------------------------------------
# GPU selection — default for @app.function(gpu=...), overridable at run
# time via the main entrypoint's --gpu flag (which uses .options() under
# the hood).  Options: A10G (24GB), A100 (40GB), A100-80GB (80GB), H100 (80GB)
# ---------------------------------------------------------------------------
_gpu_type = os.environ.get("SAM_AUDIO_GPU", "A100")

# ---------------------------------------------------------------------------
# Container image
# On Linux x86_64 + CUDA all sam_audio deps install via pip (no conda needed).
# The base.py patch adds defaults for proxies/resume_download removed in newer
# huggingface_hub versions (>=1.x stopped passing them to _from_pretrained).
# ---------------------------------------------------------------------------
_patch_base_py = r"""
import importlib.util, os, re

spec = importlib.util.find_spec("sam_audio")
base_path = os.path.join(os.path.dirname(spec.origin), "model", "base.py")

with open(base_path) as f:
    src = f.read()

orig = src
src = re.sub(r'proxies: Optional\[Dict\](?!\s*=)', 'proxies: Optional[Dict] = None', src)
src = re.sub(r'resume_download: bool(?!\s*=)', 'resume_download: bool = False', src)

if src != orig:
    with open(base_path, "w") as f:
        f.write(src)
    print("base.py patch: fixed proxies/resume_download defaults")
else:
    print("base.py patch: no changes needed")
"""

DEFAULT_MODEL = "facebook/sam-audio-large"


def _download_default_model() -> None:
    """Pre-download the default SAM-Audio model AND all its dependencies at build time.

    Runs during `modal.Image.run_function()` so it has access to the HF secret.
    SAMAudio.from_pretrained() triggers downloads of:
      - SAM-Audio weights (~14 GB) → ~/.cache/huggingface
      - ImageBind weights (4.47 GB) → .checkpoints/imagebind_huge.pth
      - RoBERTa / other backbone configs

    All of these land on disk during build and are baked into the image,
    so the runtime from_pretrained() call hits the local cache.
    """
    import gc
    import inspect
    import os

    import torch

    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
    from sam_audio import SAMAudio, SAMAudioProcessor
    from sam_audio.model.base import BaseModel as _SamBase

    # Same runtime patch as in separate() — newer huggingface_hub no longer
    # passes proxies/resume_download but sam_audio's _from_pretrained requires them.
    _orig = _SamBase._from_pretrained.__func__
    _params = inspect.signature(_orig).parameters
    if any(
        p in _params and _params[p].default is inspect.Parameter.empty
        for p in ("proxies", "resume_download")
    ):
        def _patched(cls, *, proxies=None, resume_download=False, **kw):
            return _orig(cls, proxies=proxies, resume_download=resume_download, **kw)
        _SamBase._from_pretrained = classmethod(_patched)
        print("Applied runtime patch: added defaults for proxies/resume_download")

    print(f"Pre-downloading default model: {DEFAULT_MODEL}")
    model = SAMAudio.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.float16,
        skip_rankers=True,
    )
    SAMAudioProcessor.from_pretrained(DEFAULT_MODEL)
    # Free memory; the model is only needed to trigger file caching
    del model
    gc.collect()
    print(f"Default model cached in image")


def _clap_embed(clap_model, laion_data, audio_48k: "torch.Tensor") -> "torch.Tensor":
    """Compute a CLAP audio embedding for a (channels, time) tensor at 48 kHz.

    Averages channels to mono, quantizes for CLAP's expected format, and returns
    a (1, dim) normalized embedding tensor on the same device as clap_model.
    """
    import torch
    mono = audio_48k.mean(0)  # (time,)
    quantized = laion_data.int16_to_float32_torch(
        laion_data.float32_to_int16_torch(mono.unsqueeze(0))
    ).float()[0]
    d = laion_data.get_audio_features(
        {},
        quantized,
        480000,
        data_truncating="rand_trunc",
        data_filling="repeatpad",
        audio_cfg=clap_model.model_cfg["audio_cfg"],
        require_grad=False,
    )
    emb = clap_model.model.get_audio_embedding([d])  # (1, dim)
    return torch.nn.functional.normalize(emb, dim=-1)


def _clap_text_embed(clap_model, text: str) -> "torch.Tensor":
    """Compute a normalized CLAP text embedding for a prompt string.

    Returns a (1, dim) tensor on the same device as clap_model.
    """
    import torch
    emb = clap_model.get_text_embedding([text], use_tensor=True)  # (1, dim)
    return torch.nn.functional.normalize(emb.float(), dim=-1)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "ffmpeg", "libsndfile1"])
    .pip_install(
        "soundfile",
        "tqdm",
        "huggingface_hub",
        "ftfy",
        "decord",
    )
    .pip_install(
        "torch",
        "torchaudio",
        "xformers",
        "git+https://github.com/facebookresearch/sam-audio.git",
    )
    .run_commands(
        "python3 -c 'import base64; exec(base64.b64decode("
        f'"{base64.b64encode(_patch_base_py.encode()).decode()}"'
        ").decode())'"
    )
    # Pre-download the default model (facebook/sam-audio-large, ~14 GB) at build
    # time so it is baked into the image — no download at runtime for the default.
    .run_function(
        _download_default_model,
        secrets=[hf_secret],
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # Modal 1.x client code (mounted into the container at /pkg/modal/) uses
    # api_pb2.EnvironmentRole.ValueType which is only present in protobuf>=5.
    # This layer is placed last so the expensive run_function layer stays cached.
    .pip_install("protobuf>=5")
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("sam-audio-hf-cache", create_if_missing=True)

app = modal.App("sam-audio-inference", image=image)


# ---------------------------------------------------------------------------
# Remote separation function
# ---------------------------------------------------------------------------
@app.function(
    gpu=_gpu_type,
    timeout=3600,
    volumes={"/hf_cache": hf_cache_vol},
    memory=32768,
    secrets=[hf_secret],
)
def separate(
    audio_bytes: bytes,
    filename: str,
    prompt: str = "jazz trumpet",
    model_name: str = "facebook/sam-audio-large",
    chunk_dur: float = 15.0,
    overlap: float = 2.0,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    audio_query_bytes: bytes = b"",
    reranking_candidates: int = 1,
    confidence_threshold: float = 0.0,
    predict_spans: bool = True,
    internal_candidates: int = 2,
) -> tuple[bytes, bytes]:
    """
    Run SAM-Audio separation on the remote GPU.
    Returns (target_wav_bytes, residual_wav_bytes).

    When audio_query_bytes is provided and reranking_candidates > 1, generates
    N candidates per chunk and selects the one most similar to the reference audio
    using CLAP audio-vs-audio cosine similarity (audio query reranking).

    When confidence_threshold > 0, loads CLAP and computes text-audio cosine
    similarity for each separated chunk. Chunks below the threshold are replaced
    with silence (target → zeros, residual → original chunk audio), preventing the
    model from hallucinating wrong instruments when the target is absent.

    predict_spans controls whether the model uses span prediction (True) or
    direct regression (False) for the separation masks.
    internal_candidates sets the number of candidates SAM-Audio internally
    generates per chunk before combining them (independent of the audio-query
    reranking, which generates multiple independent runs).
    """
    import io
    import math
    import os
    import tempfile
    import time

    import soundfile as sf
    import torch
    import torchaudio

    # Use persistent volume cache only for non-default models.
    # The default model is pre-downloaded in the image during build.
    cache_in_image = model_name == DEFAULT_MODEL
    if not cache_in_image:
        os.environ["HF_HOME"] = "/hf_cache"

    SR = 48_000

    # ── Write input audio to a temp file so torchaudio can load it ───────────
    suffix = Path(filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    audio, sr = torchaudio.load(tmp_path)
    os.unlink(tmp_path)
    duration = audio.shape[-1] / sr
    print(f"Loaded: {filename}  {audio.shape}, {sr} Hz, {duration:.1f}s")

    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    print(f"Resampled to {SR} Hz: {audio.shape}")

    # ── CLAP: load for audio query and/or confidence gate ────────────────────
    use_audio_query = bool(audio_query_bytes) and reranking_candidates > 1
    use_conf_gate   = confidence_threshold > 0.0
    clap_model = ref_embed = text_embed = laion_data = None
    needs_rankers = use_audio_query or use_conf_gate

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {model_name}")
    import inspect
    from sam_audio import SAMAudio, SAMAudioProcessor
    from sam_audio.model.base import BaseModel as _SamBase

    # Newer huggingface_hub (>=0.21) no longer passes proxies/resume_download to
    # _from_pretrained, but the GitHub tip of sam_audio still requires them.
    _orig = _SamBase._from_pretrained.__func__
    _params = inspect.signature(_orig).parameters
    if any(
        p in _params and _params[p].default is inspect.Parameter.empty
        for p in ("proxies", "resume_download")
    ):
        def _patched(cls, *, proxies=None, resume_download=False, **kw):
            return _orig(cls, proxies=proxies, resume_download=resume_download, **kw)
        _SamBase._from_pretrained = classmethod(_patched)
        print("Applied runtime patch: added defaults for proxies/resume_download")

    t0 = time.time()
    model = SAMAudio.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        skip_rankers=not needs_rankers,
    )
    processor = SAMAudioProcessor.from_pretrained(model_name)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    device = "cuda"
    model = model.to(device)
    torch.cuda.empty_cache()
    print(f"  float16 on {device}")
    model.eval()

    # Check if xFormers CUDA extensions are active
    _xformers_ok = False
    try:
        from xformers.ops import memory_efficient_attention  # noqa: F401
        _xformers_ok = True
    except (ImportError, RuntimeError):
        pass

    if needs_rankers:
        from sam_audio.ranking.clap import get_model as get_clap
        for _tried in (
            "laion_clap.training.data",
            "sam_audio.laion_clap.training.data",
        ):
            try:
                _laion_data = __import__(_tried, fromlist=[""])
                break
            except ImportError:
                continue
        else:
            raise ImportError(
                "Could not import laion_clap.training.data — tried both "
                "top-level and sam_audio.vendored paths"
            )

        laion_data = _laion_data
        print("\nLoading CLAP…")
        clap_model = get_clap(device="cuda")

    if use_audio_query:
        ref_audio, ref_sr = torchaudio.load(io.BytesIO(audio_query_bytes))
        if ref_sr != SR:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, SR)
        ref_embed = _clap_embed(clap_model, laion_data, ref_audio.mean(0, keepdim=True))
        print(f"  Audio query embedded (shape: {ref_embed.shape})")
        print(f"  Generating {reranking_candidates} candidates per chunk")

    if use_conf_gate:
        text_embed = _clap_text_embed(clap_model, prompt)
        print(f"  Confidence gate enabled — threshold={confidence_threshold:.3f}"
              f"  (prompt: '{prompt}')")

    # ── ODE options ───────────────────────────────────────────────────────────
    nfe_per_step = {"midpoint": 2, "euler": 1, "rk4": 4}[ode_method]
    step_size = 1.0 / (ode_steps / nfe_per_step)
    ode_opt = {"method": ode_method, "options": {"step_size": step_size}}

    print(f"\nConfiguration:")
    print(f"  Prompt:              '{prompt}'")
    print(f"  Model:               {model_name}")
    print(f"  Device:              {device}")
    print(f"  Precision:           float16")
    print(f"  xFormers:            {'yes' if _xformers_ok else 'no (fallback to PyTorch attention)'}")
    if not needs_rankers:
        print(f"  skip_rankers:        yes")
    print(f"  ODE:                 {ode_method}, {ode_steps} NFE (step_size={step_size:.4f})")
    print(f"  Chunk duration:      {chunk_dur}s")
    print(f"  Overlap:             {overlap}s")
    print(f"  predict_spans:       {predict_spans}")
    print(f"  internal_candidates: {internal_candidates}")
    print(f"  audio_query:         {'yes' if use_audio_query else 'no'}")
    print(f"  confidence_threshold: {confidence_threshold}")

    # ── Separation ────────────────────────────────────────────────────────────
    def cosine_crossfade(a, b, ov):
        if ov <= 0:
            return torch.cat([a, b], dim=-1)
        ov = min(ov, a.shape[-1], b.shape[-1])
        t = torch.linspace(0, math.pi / 2, ov, device=a.device)
        fade_out = torch.cos(t).unsqueeze(0)
        fade_in  = torch.sin(t).unsqueeze(0)
        blended = a[..., -ov:] * fade_out + b[..., :ov] * fade_in
        return torch.cat([a[..., :-ov], blended, b[..., ov:]], dim=-1)

    chunk_samples   = int(chunk_dur * SR)
    overlap_samples = int(overlap * SR)
    step            = chunk_samples - overlap_samples
    total           = audio.shape[-1]

    # Handle disabled chunking or invalid parameters
    if chunk_samples <= 0 or step <= 0:
        # Process entire audio as single chunk (no chunking)
        starts = [0]
        chunk_samples = total
    else:
        starts = list(range(0, total, step))
        # Ensure coverage: if the last start + chunk_samples doesn't reach the end,
        # add one more start so the last chunk covers the tail.
        if starts[-1] + chunk_samples < total:
            starts.append(total - chunk_samples)

    # If the last chunk would be shorter than chunk_samples, merge it into the
    # previous chunk by removing its start. The previous chunk
    # will extend to `total` in the loop below (since it becomes the last).
    if len(starts) > 1 and total - starts[-1] < chunk_samples:
        starts.pop()

    target_out = residual_out = None
    t0 = time.time()

    for i, start in enumerate(starts):
        is_last = i == len(starts) - 1
        end     = total if is_last else min(start + chunk_samples, total)
        chunk = audio[..., start:end]
        print(f"  Chunk {i+1}/{len(starts)}: {start/SR:.1f}s – {end/SR:.1f}s", flush=True)

        batch = processor(descriptions=[prompt], audios=[chunk])
        batch = batch.to(device)

        if use_audio_query:
            # Generate N candidates with different noise seeds, pick best by CLAP
            cand_targets, cand_residuals = [], []
            with torch.inference_mode():
                for _ in range(reranking_candidates):
                    res = model.separate(batch, predict_spans=predict_spans,
                                         reranking_candidates=internal_candidates, ode_opt=ode_opt)
                    t = res.target[0].float().cpu()
                    r = res.residual[0].float().cpu()
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                        r = r.unsqueeze(0)
                    cand_targets.append(t)
                    cand_residuals.append(r)

            cand_embeds = torch.cat([
                _clap_embed(clap_model, laion_data, c.mean(0, keepdim=True))
                for c in cand_targets
            ], dim=0)  # (N, dim)
            scores = (cand_embeds @ ref_embed.T).squeeze(-1)  # (N,)
            best = int(scores.argmax())
            print(f"    → candidate {best+1}/{reranking_candidates} selected "
                  f"(scores: {[f'{s:.3f}' for s in scores.tolist()]})", flush=True)
            t_chunk = cand_targets[best]
            r_chunk = cand_residuals[best]
        else:
            with torch.inference_mode():
                result = model.separate(batch, predict_spans=predict_spans,
                                        reranking_candidates=internal_candidates, ode_opt=ode_opt)
            t_chunk = result.target[0].float().cpu()
            r_chunk = result.residual[0].float().cpu()
            if t_chunk.dim() == 1:
                t_chunk = t_chunk.unsqueeze(0)
                r_chunk = r_chunk.unsqueeze(0)

        # ── Confidence gate ───────────────────────────────────────────────────
        if use_conf_gate:
            chunk_embed = _clap_embed(clap_model, laion_data, t_chunk.mean(0, keepdim=True))
            sim = float((chunk_embed @ text_embed.T).squeeze())
            if sim < confidence_threshold:
                print(f"    → GATED (sim={sim:.3f} < {confidence_threshold:.3f})"
                      f" — replacing with silence", flush=True)
                # Target is silenced; residual gets the original chunk so no audio is lost
                r_chunk = r_chunk + t_chunk
                t_chunk = torch.zeros_like(t_chunk)
            else:
                print(f"    → OK (sim={sim:.3f})", flush=True)

        if target_out is None:
            target_out   = t_chunk
            residual_out = r_chunk
        else:
            target_out   = cosine_crossfade(target_out,   t_chunk,   overlap_samples)
            residual_out = cosine_crossfade(residual_out, r_chunk,   overlap_samples)

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/duration:.2f}× realtime)")

    # ── Encode outputs to WAV bytes ───────────────────────────────────────────
    def to_wav_bytes(tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, tensor.permute(1, 0).numpy(), SR, format="WAV")
        buf.seek(0)
        return buf.read()

    if not cache_in_image:
        hf_cache_vol.commit()   # persist newly cached model files from non-default models
    return to_wav_bytes(target_out), to_wav_bytes(residual_out)


# ---------------------------------------------------------------------------
# Pre-download helper — warms up the HF cache volume so the first real run
# doesn't pay the ~14 GB download + ImageBind download cost.
# ---------------------------------------------------------------------------
@app.function(
    gpu=_gpu_type,
    timeout=1800,
    volumes={"/hf_cache": hf_cache_vol},
    secrets=[hf_secret],
)
def download_model(model_name: str = "facebook/sam-audio-large") -> None:
    """Download model weights to the persistent volume (no separation)."""
    import inspect
    import os
    import time

    import torch

    os.environ["HF_HOME"] = "/hf_cache"

    from sam_audio import SAMAudio, SAMAudioProcessor
    from sam_audio.model.base import BaseModel as _SamBase

    # Same proxies/resume_download patch as in separate()
    _orig = _SamBase._from_pretrained.__func__
    _params = inspect.signature(_orig).parameters
    if any(
        p in _params and _params[p].default is inspect.Parameter.empty
        for p in ("proxies", "resume_download")
    ):
        def _patched(cls, *, proxies=None, resume_download=False, **kw):
            return _orig(cls, proxies=proxies, resume_download=resume_download, **kw)
        _SamBase._from_pretrained = classmethod(_patched)

    t0 = time.time()
    print(f"Downloading model: {model_name}")
    SAMAudio.from_pretrained(model_name, torch_dtype=torch.float16, skip_rankers=True)
    SAMAudioProcessor.from_pretrained(model_name)
    print(f"Downloaded in {time.time() - t0:.1f}s")

    hf_cache_vol.commit()
    print("HF cache volume committed.")


@app.local_entrypoint()
def run_download(model: str = "facebook/sam-audio-large") -> None:
    """Pre-download the model to the Modal volume."""
    print(f"Pre-downloading {model} to cache volume…")
    download_model.remote(model_name=model)


# ---------------------------------------------------------------------------
# Local entrypoint — handles file I/O and CLI parsing
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input: str,
    output_dir: str = "separated/modal_sam_audio",
    prompt: str = "jazz trumpet",
    model: str = "facebook/sam-audio-large",
    chunk_dur: float = 15.0,
    overlap: float = 2.0,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    audio_query: str = "",
    reranking_candidates: int = 1,
    confidence_threshold: float = 0.0,
    predict_spans: bool = True,
    internal_candidates: int = 2,
):
    input_path = Path(input)
    audio_bytes = input_path.read_bytes()

    audio_query_bytes = b""
    if audio_query:
        audio_query_path = Path(audio_query)
        audio_query_bytes = audio_query_path.read_bytes()
        print(f"Audio query: {audio_query_path.name} "
              f"({len(audio_query_bytes)/1e6:.1f} MB), "
              f"{reranking_candidates} candidates/chunk")

    print(f"Uploading {input_path.name} ({len(audio_bytes)/1e6:.1f} MB) → Modal [{_gpu_type}]…")

    target_bytes, residual_bytes = separate.remote(
        audio_bytes=audio_bytes,
        filename=input_path.name,
        prompt=prompt,
        model_name=model,
        chunk_dur=chunk_dur,
        overlap=overlap,
        ode_steps=ode_steps,
        ode_method=ode_method,
        audio_query_bytes=audio_query_bytes,
        reranking_candidates=reranking_candidates,
        confidence_threshold=confidence_threshold,
        predict_spans=predict_spans,
        internal_candidates=internal_candidates,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prompt_tag = prompt.replace(" ", "_")
    ode_tag    = f"{ode_method}{ode_steps}"
    stem       = input_path.stem
    aq_tag     = f"_aq{reranking_candidates}" if audio_query else ""
    cg_tag     = f"_cg{confidence_threshold:.2f}".replace(".", "") if confidence_threshold > 0 else ""
    ps_tag     = "" if predict_spans else "_nops"
    ic_tag     = f"_ic{internal_candidates}"
    ch_tag     = f"_ch{int(chunk_dur)}"

    target_path   = out / f"{stem}_sam_{ode_tag}{ps_tag}{ic_tag}{ch_tag}{aq_tag}{cg_tag}_{prompt_tag}_target.wav"
    residual_path = out / f"{stem}_sam_{ode_tag}{ps_tag}{ic_tag}{ch_tag}{aq_tag}{cg_tag}_{prompt_tag}_residual.wav"

    target_path.write_bytes(target_bytes)
    residual_path.write_bytes(residual_bytes)

    print(f"\nSaved:")
    print(f"  Target:   {target_path}")
    print(f"  Residual: {residual_path}")
