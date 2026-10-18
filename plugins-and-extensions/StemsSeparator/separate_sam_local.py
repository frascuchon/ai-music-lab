#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local (non-Modal) CLI backend for SAM Audio — ``sam_large``/``sam_base``.

Called by separate_sam.py's own ``--execution local`` dispatch (not by the
Lua panel directly) as a plain subprocess, mirroring how modal_sam_audio.py
itself behaves from separate_sam.py's point of view: this script has no
``--progress``/``write_progress`` of its own — separate_sam.py remains the
single progress-file writer for both execution modes (it already tails
stdout for "Chunk N/M" lines and discovers output .wav files in --outdir by
diffing the directory listing before/after the subprocess runs, exactly like
it does for the Modal path). This script just prints plain status lines
("Chunk i/n: ...", "ERROR: ...") and writes target.wav/residual.wav into
--outdir — same contract modal_sam_audio.py's own stdout already satisfies.

Why this is feasible at all (SAM Audio was assumed CUDA-only until this was
actually checked on 2026-10-01): reading the real sam_audio package source
(git+https://github.com/facebookresearch/sam-audio.git) plus its dacvae and
ImageBind git dependencies found ZERO hardcoded ``.cuda()`` calls anywhere.
Attention uses ``torch.nn.functional.scaled_dot_product_attention`` (native
MPS support), not a CUDA-only kernel. ``xformers`` — the only CUDA-specific
package in the picture — is never imported here; it was only ever an
optional opportunistic speedup in the existing Modal script, guarded by
``try/except``. So this is NOT architecturally blocked, just heavy:
SAM-Audio weights + ImageBind (~4.47 GB, downloaded unconditionally by
``SAMAudio.from_pretrained()`` even with ``skip_rankers=True``) + a
Perception-Encoder vision backbone + T5 text encoder add up to a large local
footprint (large variant likely 15-20+ GB; base meaningfully smaller).

Scope cut: this script always loads with ``skip_rankers=True``. The
audio-query (CLAP/ImageBind reranking) and confidence-gate features from
modal_sam_audio.py are NOT implemented here — separate_sam.py's dispatch
rejects ``--execution local`` up front if the user has those enabled
(reranking_candidates > 1 or confidence_threshold > 0), rather than silently
ignoring them.

Contract:
    python3 separate_sam_local.py
        --input   <audio file>
        --model   <sam_large|sam_base>
        --prompt  <text, e.g. "jazz trumpet">
        --outdir  <dir>
        [--chunk <seconds>] [--overlap <seconds>]
        [--steps <int>] [--ode-method <midpoint|euler|rk4>]
        [--predict-spans / --no-predict-spans] [--internal-candidates <int>]
        [--start <float>] [--duration <float>]   # section of --input
        [--device <auto|mps|cpu>]

Output: <outdir>/target.wav, <outdir>/residual.wav.

Device selection NEVER hardcodes "cuda" — this script only ever runs on the
user's own machine (Apple Silicon MPS or CPU). See _resolve_device().

IMPORTANT — scope of verification: this file was written by adapting
modal_sam_audio.py's own ``separate()`` function (device/dtype swapped from
hardcoded "cuda"/float16 to the resolved local device/float32) — it has NOT
been run end-to-end against real model weights in this environment (no
GPU/MPS hardware or the ~15-20 GB weight download attempted here). Only
argument parsing, model-support validation, and device-resolution logic are
covered by tests (test_separate_sam_local.py).
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Short registry key -> actual HF repo id (mirrors panel.lua's
# SS_SAM_MODEL_NAME table).
# ---------------------------------------------------------------------------
MODEL_HF_REPO: dict[str, str] = {
    "sam_large": "facebook/sam-audio-large",
    "sam_base": "facebook/sam-audio-base",
}

LOCAL_SUPPORTED_MODELS = set(MODEL_HF_REPO)

SR = 48_000


def extract_section(src_path: Path, start: float, duration: float, dst_path: Path):
    """Write a time slice of src_path to dst_path using soundfile. Same
    helper as separate_demucs.py/separate_sam.py."""
    import soundfile as sf
    with sf.SoundFile(str(src_path)) as f:
        sr = f.samplerate
        start_frame = int(start * sr)
        num_frames = int(duration * sr)
        f.seek(start_frame)
        data = f.read(num_frames, dtype="float32", always_2d=True)
    sf.write(str(dst_path), data, sr)


def _resolve_device(requested: str, mps_available: bool, cuda_available: bool) -> str:
    """Pure function (no torch import) so tests can exercise every branch
    without real hardware. "auto" prefers MPS over CUDA over CPU; an
    explicit request is honored as-is except falling back to CPU if it
    names unavailable hardware. Mirrors Text2Audio/text2audio_local.py's
    own _resolve_device for consistency."""
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


def _cosine_crossfade(a, b, ov):
    """Same crossfade helper as modal_sam_audio.py's separate(), operating
    on local torch tensors."""
    import torch
    if ov <= 0:
        return torch.cat([a, b], dim=-1)
    ov = min(ov, a.shape[-1], b.shape[-1])
    t = torch.linspace(0, math.pi / 2, ov, device=a.device)
    fade_out = torch.cos(t).unsqueeze(0)
    fade_in = torch.sin(t).unsqueeze(0)
    blended = a[..., -ov:] * fade_out + b[..., :ov] * fade_in
    return torch.cat([a[..., :-ov], blended, b[..., ov:]], dim=-1)


def _run_separate(input_path: Path, model_key: str, prompt: str, chunk_dur: float,
                  overlap: float, ode_steps: int, ode_method: str,
                  predict_spans: bool, internal_candidates: int, device: str,
                  out_dir: Path) -> tuple[Path, Path]:
    """Adapted from modal_sam_audio.py's separate(): same chunking/crossfade
    algorithm, run in-process on `device` (mps/cpu) instead of a Modal GPU
    container. Always skip_rankers=True (no audio-query/confidence-gate)."""
    import torch
    import torchaudio
    import soundfile as sf

    model_name = MODEL_HF_REPO[model_key]

    print(f"Loading model: {model_name}", flush=True)
    audio, sr = torchaudio.load(str(input_path))
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    print(f"Loaded: {input_path.name}  {audio.shape}, resampled to {SR} Hz", flush=True)

    from sam_audio import SAMAudio, SAMAudioProcessor

    # float32 (not float16 like the Modal/CUDA path): safer default for
    # CPU/MPS correctness — float16 matmul is slow/unsupported in places on
    # both backends, and this is already a slow, non-realtime local path.
    model = SAMAudio.from_pretrained(model_name, torch_dtype=torch.float32, skip_rankers=True)
    processor = SAMAudioProcessor.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device} (float32)", flush=True)

    nfe_per_step = {"midpoint": 2, "euler": 1, "rk4": 4}[ode_method]
    step_size = 1.0 / (ode_steps / nfe_per_step)
    ode_opt = {"method": ode_method, "options": {"step_size": step_size}}

    chunk_samples = int(chunk_dur * SR)
    overlap_samples = int(overlap * SR)
    step = chunk_samples - overlap_samples
    total = audio.shape[-1]

    if chunk_samples <= 0 or step <= 0:
        starts = [0]
        chunk_samples = total
    else:
        starts = list(range(0, total, step))
        if len(starts) > 1 and total - starts[-1] < overlap_samples:
            starts.pop()

    target_out = residual_out = None

    for i, start in enumerate(starts):
        end = min(start + chunk_samples, total)
        chunk = audio[..., start:end]
        print(f"  Chunk {i+1}/{len(starts)}: {start/SR:.1f}s - {end/SR:.1f}s", flush=True)

        batch = processor(descriptions=[prompt], audios=[chunk])
        batch = batch.to(device)

        with torch.inference_mode():
            result = model.separate(batch, predict_spans=predict_spans,
                                    reranking_candidates=internal_candidates, ode_opt=ode_opt)
        t_chunk = result.target[0].float().cpu()
        r_chunk = result.residual[0].float().cpu()
        if t_chunk.dim() == 1:
            t_chunk = t_chunk.unsqueeze(0)
            r_chunk = r_chunk.unsqueeze(0)

        if target_out is None:
            target_out = t_chunk
            residual_out = r_chunk
        else:
            target_out = _cosine_crossfade(target_out, t_chunk, overlap_samples)
            residual_out = _cosine_crossfade(residual_out, r_chunk, overlap_samples)

        batch = chunk = t_chunk = r_chunk = result = None

    target_path = out_dir / "target.wav"
    residual_path = out_dir / "residual.wav"
    sf.write(str(target_path), target_out.permute(1, 0).numpy(), SR, format="WAV")
    sf.write(str(residual_path), residual_out.permute(1, 0).numpy(), SR, format="WAV")
    print(f"Saved: {target_path}", flush=True)
    print(f"Saved: {residual_path}", flush=True)
    return target_path, residual_path


def main() -> int:
    p = argparse.ArgumentParser(description="Local (MPS/CPU) SAM Audio backend.")
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True, choices=sorted(LOCAL_SUPPORTED_MODELS))
    p.add_argument("--prompt", default="vocals")
    p.add_argument("--chunk", type=float, default=15.0, dest="chunk_dur")
    p.add_argument("--overlap", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=64, dest="ode_steps")
    p.add_argument("--ode-method", default="midpoint", dest="ode_method")
    p.add_argument("--predict-spans", default=True, dest="predict_spans",
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--internal-candidates", type=int, default=1, dest="internal_candidates")
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    # Fail fast on an unsupported model BEFORE importing torch/sam_audio
    # (expensive / possibly-not-installed) — belt-and-suspenders in case the
    # Lua panel's / separate_sam.py's local/remote gating is ever bypassed.
    if args.model not in LOCAL_SUPPORTED_MODELS:
        print(f"ERROR: {args.model!r} has no local backend.", flush=True)
        return 1

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: Input file does not exist: {input_path}", flush=True)
        return 1

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = None
    process_path = input_path
    if args.start is not None and args.duration is not None:
        print(f"Extracting section {args.start:.2f}s -> "
              f"{args.start + args.duration:.2f}s...", flush=True)
        try:
            temp_dir = tempfile.mkdtemp(prefix="stemsep_sam_local_")
            section_file = Path(temp_dir) / input_path.name
            extract_section(input_path, args.start, args.duration, section_file)
            process_path = section_file
        except ModuleNotFoundError as exc:
            print(f"ERROR: Missing Python package '{exc.name}'. Open the "
                  f"Setup tab and click Install next to demucs to "
                  f"reinstall it (this pulls in {exc.name}).", flush=True)
            return 1
        except Exception as exc:
            print(f"ERROR: Error extracting section: {exc}", flush=True)
            return 1

    try:
        import torch
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda_available = torch.cuda.is_available()
        device = _resolve_device(args.device, mps_available, cuda_available)
        print(f"Device: {device}", flush=True)

        _run_separate(
            process_path, args.model, args.prompt, args.chunk_dur, args.overlap,
            args.ode_steps, args.ode_method, args.predict_spans,
            args.internal_candidates, device, out_dir,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced via separate_sam.py's stdout tail
        print(f"ERROR: {exc}", flush=True)
        return 1
    finally:
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
