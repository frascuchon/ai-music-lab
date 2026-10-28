#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local/remote execution feasibility registry — Python mirror of
``lib/model_execution.lua``, for use by midigen.py / text2audio.py's CLI
backends when validating ``--execution local`` requests.

Keep this in sync with ``lib/model_execution.lua`` — same model keys, same
``local_supported``/``reason`` values. See that file's header for sources.
"""

from __future__ import annotations

MIDIGEN: dict[str, dict] = {
    "anticipatory": {
        "local_supported": True,
        "reason": "CPU-only PyTorch, no GPU required (research_anticipatory.py).",
    },
    "midigpt": {
        "local_supported": False,
        "reason": ("Runs on Modal only: gated Hugging Face weights (non-commercial "
                   "license) and a CUDA-sized model; no local backend implemented."),
    },
    "amadeus": {
        "local_supported": False,
        "reason": ("No local backend implemented; research pipeline assumes "
                   "Modal CUDA container (fluidsynth soundfont path, GPU-only checkpoint load)."),
    },
}

TEXT2AUDIO: dict[str, dict] = {
    "sao": {
        "local_supported": True,
        "reason": "diffusers StableAudioPipeline runs on MPS/CPU (slow).",
    },
    "sao_edit": {
        "local_supported": True,
        "reason": "Same SAO pipeline via SDEdit (init_audio/init_noise_level), MPS/CPU capable.",
    },
    "foundation1": {
        "local_supported": True,
        "reason": "Same SAO pipeline/weights family, MPS/CPU capable.",
    },
    "musicgen_gen": {
        "local_supported": True,
        "reason": "audiocraft MusicGen runs on MPS/CPU (slow, autoregressive).",
    },
    "magnet": {
        "local_supported": True,
        "reason": "audiocraft MAGNeT runs on MPS/CPU.",
    },
    "audiogen": {
        "local_supported": True,
        "reason": "audiocraft AudioGen runs on MPS/CPU.",
    },
    "acestep_gen": {
        "local_supported": False,
        "reason": "No MPS/CPU support confirmed in research; CUDA-only.",
    },
    "acestep": {
        "local_supported": False,
        "reason": "No MPS/CPU support confirmed in research; CUDA-only (edit mode).",
    },
    "mustango": {
        "local_supported": False,
        "reason": "Research confirms no MPS/CPU support.",
    },
    "inspiremusic_gen": {
        "local_supported": False,
        "reason": "Research confirms no MPS/CPU support.",
    },
}

STEMSEPARATOR: dict[str, dict] = {
    # Demucs variants: always local, no remote backend exists at all — these
    # entries exist only for registry completeness/consistency, not because
    # the UI needs to toggle them.
    "htdemucs": {
        "local_supported": True,
        "reason": "Local-only Demucs model; no remote backend exists.",
    },
    "htdemucs_ft": {
        "local_supported": True,
        "reason": "Local-only Demucs model; no remote backend exists.",
    },
    "mdx_extra": {
        "local_supported": True,
        "reason": "Local-only Demucs model; no remote backend exists.",
    },
    "htdemucs_6s": {
        "local_supported": True,
        "reason": "Local-only Demucs model; no remote backend exists.",
    },
    # SAM Audio: investigated 2026-10-01 by reading the actual sam_audio
    # package source (git+https://github.com/facebookresearch/sam-audio.git)
    # plus its dacvae/ImageBind git dependencies. No hardcoded .cuda() calls
    # anywhere; attention uses torch.nn.functional.scaled_dot_product_attention
    # (native MPS support), not a CUDA-only kernel or xformers (xformers is
    # only an optional opportunistic speedup in the existing Modal script,
    # guarded by try/except). Not architecturally blocked — but heavy:
    # SAM-Audio weights + ImageBind (~4.47 GB, downloaded unconditionally by
    # SAMAudio.from_pretrained() even with skip_rankers=True) + a Perception
    # Encoder vision backbone + T5 text encoder add up to a large local
    # footprint (large variant likely 15-20+ GB; base meaningfully smaller).
    # Local backend (separate_sam_local.py) always runs with skip_rankers=True
    # — the audio-query/confidence-gate ranking features are remote-only.
    "sam_large": {
        "local_supported": True,
        "reason": ("MPS/CPU capable (scaled_dot_product_attention, no CUDA-only "
                   "kernels) but heavy: ~15-20 GB combined download/RAM "
                   "(SAM-Audio-large + ImageBind + vision/text encoders)."),
    },
    "sam_base": {
        "local_supported": True,
        "reason": ("MPS/CPU capable (scaled_dot_product_attention, no CUDA-only "
                   "kernels); smaller than sam_large but still several GB "
                   "(ImageBind alone is ~4.47 GB, downloaded unconditionally)."),
    },
}

_DOMAINS = {"midigen": MIDIGEN, "text2audio": TEXT2AUDIO, "stemseparator": STEMSEPARATOR}


def is_local_supported(domain: str, model_key: str) -> tuple[bool, str]:
    """Return (supported, reason) for model_key within domain.

    Unknown domain/model_key => (False, explanatory reason), never raises —
    callers treat "unknown" the same as "unsupported" defensively.
    """
    table = _DOMAINS.get(domain)
    if table is None:
        return False, f"Unknown domain: {domain!r}"
    entry = table.get(model_key)
    if entry is None:
        return False, f"Unknown model: {model_key!r}"
    return bool(entry["local_supported"]), str(entry["reason"])
