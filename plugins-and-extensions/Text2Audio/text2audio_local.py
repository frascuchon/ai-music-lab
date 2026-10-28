#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local (non-Modal) CLI backend for the 5 Text2Audio models the feasibility
registry (shared/model_execution.py TEXT2AUDIO) marks as local_supported:
sao, sao_edit, musicgen_gen, audiogen.

Called by text2audio.py's own ``--execution local`` dispatch (not by the Lua
panel directly) as a plain subprocess, mirroring how text2audio.py already
drives ``modal run research_*_modal.py::main`` for the remote path: this
script has no --progress/write_progress of its own — text2audio.py remains
the single progress-file writer for both execution modes (it already parses
"step N/M"-style lines from stdout and tails raw output on failure). This
avoids two processes racing to write the same progress file.

Contract:
    python3 text2audio_local.py
        --model   <sao|sao_edit|musicgen_gen|audiogen>
        --mode    <generate|edit>
        --prompt  <text>
        --out-dir <dir>
        [--seconds <float>]       # generate mode (all)
        [--input  <path>]         # edit mode: source audio
        [--intensity <subtle|moderate|strong>]   # edit mode (sao_edit)
        [--seed <int>]
        [--device <auto|mps|cpu>]

Output: <out-dir>/output.wav — same filename text2audio.py already expects
from the remote path, so its existing rename-to-descriptive-name logic needs
no changes.

Device selection NEVER hardcodes "cuda" — this script only ever runs on the
user's own machine (Apple Silicon MPS or CPU). See _resolve_device().

IMPORTANT — scope of verification: this file was written by inspecting the
real APIs used by the sibling research_*_modal.py scripts (diffusers
StableAudioPipeline, stable-audio-tools generate_diffusion_cond, transformers
Musicgen*ForConditionalGeneration, audiocraft AudioGen) and adapting
their device handling from hardcoded "cuda" to the local device. It has NOT
been run end-to-end against real model weights in this environment (no
GPU/MPS hardware or multi-GB downloads available here) — only the argument
parsing, model-support validation, device resolution and command-building
logic are covered by tests (test_text2audio_local.py).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Models this script implements, and their dependency family.
# ---------------------------------------------------------------------------
_SAO_FAMILY_GENERATE = {
    "sao": "stabilityai/stable-audio-open-1.0",
}
_SAO_EDIT_REPO = "stabilityai/stable-audio-open-1.0"

# transformers MusicgenForConditionalGeneration — same library the Modal
# script uses for this model (NOT audiocraft, which needs xformers/CUDA headers).
_MUSICGEN_TRANSFORMERS = {
    "musicgen_gen": "facebook/musicgen-medium",
}

# audiocraft get_pretrained() — used as-is by the Modal script (just with
# device="cuda" hardcoded there; we pass the local device instead).
_AUDIOCRAFT_MODELS = {
    "audiogen": "facebook/audiogen-medium",
}

LOCAL_SUPPORTED_MODELS = (
    set(_SAO_FAMILY_GENERATE) | {"sao_edit"}
    | set(_MUSICGEN_TRANSFORMERS) | set(_AUDIOCRAFT_MODELS)
)

# Same mapping text2audio.py's INTENSITY_MAP uses for sao_edit, duplicated
# here (not imported) because this script must stay importable/runnable
# without a working text2audio.py import path.
_SAO_EDIT_INTENSITY = {"subtle": 0.4, "moderate": 1.0, "strong": 4.0}


def _resolve_device(requested: str, mps_available: bool, cuda_available: bool) -> str:
    """Pure function (no torch import) so tests can exercise every branch
    without real hardware. "auto" prefers MPS (this script's whole reason
    for existing) over CUDA over CPU; an explicit request is honored as-is
    except falling back to CPU if it names unavailable hardware."""
    if requested == "auto":
        if mps_available:
            return "mps"
        if cuda_available:
            return "cuda"
        return "cpu"
    if requested == "mps" and not mps_available:
        return "cpu"
    if requested == "cuda" and not cuda_available:
        return "cpu"
    return requested


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())[:n].strip("_").lower()


# ---------------------------------------------------------------------------
# SAO family (sao) — diffusers StableAudioPipeline, generate mode
# ---------------------------------------------------------------------------

def _run_sao_generate(model: str, prompt: str, seconds: float, seed: int,
                       device: str, out_wav: Path) -> None:
    import torch
    from diffusers import StableAudioPipeline

    repo_id = _SAO_FAMILY_GENERATE[model]
    print(f"[text2audio_local] Loading {repo_id} on {device} (float32)...", flush=True)
    pipe = StableAudioPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device if device != "mps" else "cpu")
    # torch.Generator does not support device="mps" directly as of current
    # PyTorch releases; seeding on CPU and letting the pipeline run on MPS
    # still produces a seeded, reproducible run (the pipeline's internal
    # random ops consume this generator for initial noise).
    generator.manual_seed(seed)

    num_steps = 100
    print(f"[text2audio_local] step 0/{num_steps}", flush=True)
    output = pipe(
        prompt,
        audio_end_in_s=seconds,
        num_inference_steps=num_steps,
        generator=generator,
        callback=lambda i, t, latents: print(f"[text2audio_local] step {i+1}/{num_steps}", flush=True),
        callback_steps=max(1, num_steps // 20),
    )
    print(f"[text2audio_local] step {num_steps}/{num_steps}", flush=True)

    audio = output.audios[0].float().cpu().numpy().T  # (samples, channels)
    import soundfile as sf
    sf.write(str(out_wav), audio, samplerate=44100, subtype="PCM_16")


# ---------------------------------------------------------------------------
# SAO edit (sao_edit) — stable-audio-tools generate_diffusion_cond (SDEdit)
# ---------------------------------------------------------------------------

def _run_sao_edit(prompt: str, input_path: Path, intensity: str, seconds: float | None,
                   seed: int, device: str, out_wav: Path) -> None:
    import torch
    import torchaudio
    import soundfile as sf
    from stable_audio_tools.models.pretrained import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    print(f"[text2audio_local] Loading {_SAO_EDIT_REPO} on {device} (float32)...", flush=True)
    model, model_config = get_pretrained_model(_SAO_EDIT_REPO)
    model = model.to(device).eval().requires_grad_(False)

    wav, sr = torchaudio.load(str(input_path))
    if seconds is None:
        seconds = wav.shape[-1] / sr

    sample_rate = model_config.get("sample_rate", 44100)
    sample_size = model_config["sample_size"]
    init_noise_level = _SAO_EDIT_INTENSITY.get(intensity, 1.0)

    conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": seconds}]

    print("[text2audio_local] step 0/100", flush=True)
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        init_audio=(sr, wav),
        init_noise_level=init_noise_level,
        seed=seed,
        device=device,
    )
    print("[text2audio_local] step 100/100", flush=True)

    audio = output[0].to(torch.float32)
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val
    n_samples = int(seconds * sample_rate)
    audio = audio[..., :n_samples].cpu().numpy().T
    sf.write(str(out_wav), audio, samplerate=sample_rate, subtype="PCM_16")


# ---------------------------------------------------------------------------
# MusicGen family (musicgen_gen, musicgen) — transformers, not audiocraft
# ---------------------------------------------------------------------------

def _run_musicgen_gen(prompt: str, seconds: float, device: str, out_wav: Path) -> None:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    variant = _MUSICGEN_TRANSFORMERS["musicgen_gen"]
    print(f"[text2audio_local] Loading {variant} on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(variant)
    model = MusicgenForConditionalGeneration.from_pretrained(variant).to(device)

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    max_new_tokens = int(seconds * 50)
    print(f"[text2audio_local] step 0/{max_new_tokens}", flush=True)
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3,
                                   max_new_tokens=max_new_tokens)
    print(f"[text2audio_local] step {max_new_tokens}/{max_new_tokens}", flush=True)

    sr_out = model.config.audio_encoder.sampling_rate
    audio = audio_values[0].cpu().float().numpy().T
    import soundfile as sf
    sf.write(str(out_wav), audio, samplerate=sr_out, subtype="PCM_16")


# ---------------------------------------------------------------------------
# audiocraft family (audiogen)
# ---------------------------------------------------------------------------

def _run_audiocraft_generate(model: str, prompt: str, seconds: float, device: str,
                             out_wav: Path) -> None:
    variant = _AUDIOCRAFT_MODELS[model]
    print(f"[text2audio_local] Loading {variant} on {device}...", flush=True)

    from audiocraft.models import AudioGen
    mdl = AudioGen.get_pretrained(variant, device=device)

    mdl.set_generation_params(duration=seconds)
    print("[text2audio_local] step 0/1", flush=True)
    wav = mdl.generate([prompt])
    print("[text2audio_local] step 1/1", flush=True)

    audio = wav[0].cpu().float().numpy().T
    import soundfile as sf
    sf.write(str(out_wav), audio, samplerate=mdl.sample_rate, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Local (MPS/CPU) Text2Audio backend.")
    p.add_argument("--model", required=True, choices=sorted(LOCAL_SUPPORTED_MODELS))
    p.add_argument("--mode", required=True, choices=["generate", "edit"])
    p.add_argument("--prompt", default="")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--input", default="")
    p.add_argument("--intensity", default="moderate", choices=["subtle", "moderate", "strong"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    p.add_argument("--out-dir", required=True, dest="out_dir")
    args = p.parse_args()

    # Fail fast on an unsupported model BEFORE importing torch/diffusers/
    # audiocraft (expensive / possibly-not-installed) — belt-and-suspenders
    # in case the Lua panel's local/remote gating is ever bypassed.
    if args.model not in LOCAL_SUPPORTED_MODELS:
        print(f"ERROR: {args.model!r} has no local backend.", flush=True)
        return 1
    if not args.prompt.strip():
        print("ERROR: Empty prompt.", flush=True)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "output.wav"

    input_path = Path(args.input) if args.input else None
    if args.mode == "edit":
        if input_path is None or not input_path.exists():
            print(f"ERROR: Source audio not found: {args.input!r}", flush=True)
            return 1
    elif args.seconds is None:
        # generate mode always needs a concrete duration; text2audio.py
        # always passes --seconds explicitly, this is just a safe default
        # for direct/manual invocation.
        args.seconds = 8.0

    try:
        import torch
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda_available = torch.cuda.is_available()
        device = _resolve_device(args.device, mps_available, cuda_available)
        print(f"[text2audio_local] device={device}", flush=True)

        if args.model in _SAO_FAMILY_GENERATE:
            if args.mode != "generate":
                print(f"ERROR: {args.model} only supports generate mode locally.", flush=True)
                return 1
            _run_sao_generate(args.model, args.prompt, args.seconds, args.seed, device, out_wav)
        elif args.model == "sao_edit":
            if args.mode != "edit":
                print("ERROR: sao_edit only supports edit mode.", flush=True)
                return 1
            _run_sao_edit(args.prompt, input_path, args.intensity,
                         args.seconds, args.seed, device, out_wav)
        elif args.model == "musicgen_gen":
            if args.mode != "generate":
                print("ERROR: musicgen_gen only supports generate mode.", flush=True)
                return 1
            _run_musicgen_gen(args.prompt, args.seconds, device, out_wav)
        elif args.model in _AUDIOCRAFT_MODELS:
            if args.mode != "generate":
                print(f"ERROR: {args.model} only supports generate mode.", flush=True)
                return 1
            _run_audiocraft_generate(args.model, args.prompt, args.seconds, device, out_wav)
        else:
            print(f"ERROR: Unhandled model {args.model!r}.", flush=True)
            return 1
    except Exception as exc:  # noqa: BLE001 — surface any failure to the caller's log tail
        print(f"ERROR: {exc}", flush=True)
        return 1

    if not out_wav.exists():
        print(f"ERROR: {out_wav} was not written.", flush=True)
        return 1

    print(f"[text2audio_local] Saved: {out_wav}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
