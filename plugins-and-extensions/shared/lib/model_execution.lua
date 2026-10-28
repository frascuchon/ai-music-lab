-- Local/remote execution feasibility registry, shared by MidiGenerator and
-- Text2Audio panels (and any future plugin that gains the same toggle).
--
-- Each entry records whether a model's *current* implementation can run
-- locally (Mac Apple Silicon MPS, or CPU) at all — not whether local is fast
-- or production-quality. "false" means the UI must disable the Local option
-- for that model, with `reason` shown to the user.
--
-- Sources: MidiGenerator/RESEARCH.md and Text2Audio/RESEARCH.md (Mac MPS/CPU
-- feasibility columns per model).

local M = {}

M.midigen = {
  anticipatory = {
    local_supported = true,
    reason = "CPU-only PyTorch, no GPU required (research_anticipatory.py).",
  },
  midigpt = {
    local_supported = false,
    reason = "Runs on Modal only: gated Hugging Face weights (non-commercial "
      .. "license) and a CUDA-sized model; no local backend implemented.",
  },
  amadeus = {
    local_supported = false,
    reason = "No local backend implemented; research pipeline assumes "
      .. "Modal CUDA container (fluidsynth soundfont path, GPU-only checkpoint load).",
  },
}

M.text2audio = {
  sao = {
    local_supported = true,
    reason = "diffusers StableAudioPipeline runs on MPS/CPU (slow).",
  },
  sao_edit = {
    local_supported = true,
    reason = "Same SAO pipeline via SDEdit (init_audio/init_noise_level), MPS/CPU capable.",
  },
  musicgen_gen = {
    local_supported = true,
    reason = "audiocraft MusicGen runs on MPS/CPU (slow, autoregressive).",
  },
  audiogen = {
    local_supported = true,
    reason = "audiocraft AudioGen runs on MPS/CPU.",
  },
  acestep_gen = {
    local_supported = false,
    reason = "No MPS/CPU support confirmed in research; CUDA-only.",
  },
  acestep = {
    local_supported = false,
    reason = "No MPS/CPU support confirmed in research; CUDA-only (edit mode).",
  },
}

M.stemseparator = {
  -- Demucs variants: always local, no remote backend exists at all — these
  -- entries exist only for registry completeness/consistency, not because
  -- the UI needs to toggle them.
  htdemucs = {
    local_supported = true,
    reason = "Local-only Demucs model; no remote backend exists.",
  },
  htdemucs_ft = {
    local_supported = true,
    reason = "Local-only Demucs model; no remote backend exists.",
  },
  mdx_extra = {
    local_supported = true,
    reason = "Local-only Demucs model; no remote backend exists.",
  },
  htdemucs_6s = {
    local_supported = true,
    reason = "Local-only Demucs model; no remote backend exists.",
  },
  -- SAM Audio: investigated 2026-10-01 by reading the actual sam_audio
  -- package source (git+https://github.com/facebookresearch/sam-audio.git)
  -- plus its dacvae/ImageBind git dependencies. No hardcoded .cuda() calls
  -- anywhere; attention uses torch.nn.functional.scaled_dot_product_attention
  -- (native MPS support), not a CUDA-only kernel or xformers (xformers is
  -- only an optional opportunistic speedup in the existing Modal script,
  -- guarded by try/except). Not architecturally blocked — but heavy:
  -- SAM-Audio weights + ImageBind (~4.47 GB, downloaded unconditionally by
  -- SAMAudio.from_pretrained() even with skip_rankers=True) + a Perception
  -- Encoder vision backbone + T5 text encoder add up to a large local
  -- footprint (large variant likely 15-20+ GB; base meaningfully smaller).
  -- Local backend (separate_sam_local.py) always runs with skip_rankers=true
  -- — the audio-query/confidence-gate ranking features are remote-only.
  sam_large = {
    local_supported = true,
    reason = "MPS/CPU capable (scaled_dot_product_attention, no CUDA-only "
      .. "kernels) but heavy: ~15-20 GB combined download/RAM "
      .. "(SAM-Audio-large + ImageBind + vision/text encoders).",
  },
  sam_base = {
    local_supported = true,
    reason = "MPS/CPU capable (scaled_dot_product_attention, no CUDA-only "
      .. "kernels); smaller than sam_large but still several GB "
      .. "(ImageBind alone is ~4.47 GB, downloaded unconditionally).",
  },
}

-- M.is_local_supported(domain, model_key) -> boolean, reason
-- domain: "midigen" | "text2audio" | "stemseparator"
function M.is_local_supported(domain, model_key)
  local table_ = M[domain]
  if not table_ then return false, "Unknown domain: " .. tostring(domain) end
  local entry = table_[model_key]
  if not entry then return false, "Unknown model: " .. tostring(model_key) end
  return entry.local_supported, entry.reason
end

return M
