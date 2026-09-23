-- AI Music Lab - Text2Audio panel (text → audio generation and editing)
-- Panel module: no gfx.init / reaper.defer of its own. Loaded by the
-- unified ai-music-lab.lua container, which owns the window/main loop
-- and calls M.init() once, M.poll() every frame, M.draw() when active.

-- ── PATHS + LIB ──────────────────────────────────────────────────
local _info      = debug.getinfo(1, "S")
local SCRIPT_DIR = _info.source:match("@?(.*[/\\])") or ""

local SHARED_DIR = SCRIPT_DIR .. "../shared/"
package.path = SHARED_DIR .. "lib/?.lua;" .. package.path

local common  = require("common")
local theme   = require("theme")
local gui     = require("gui")
local widgets = require("widgets_extra")

local HOME   = common.HOME
local TMPDIR = common.TMPDIR
local PYTHON, PYTHON_ERR = common.detect_reaper_python()

-- ── CONSTANTS ────────────────────────────────────────────────────
-- Generation models (text → audio, no source)
local GEN_MODELS  = { "sao", "foundation1", "acestep_gen", "inspiremusic_gen",
                      "mustango", "audiogen", "musicgen_gen", "magnet" }
local GEN_LABELS  = {
  "Stable Audio Open 1.0  (A10G, 44.1 kHz stereo)",
  "Foundation-1  (A10G, electronic, TAG format)",
  "ACE-Step 1.5  (A10G, full-song, Apache 2.0)",
  "InspireMusic 1.5B  (A10G, 48 kHz, Apache 2.0)",
  "Mustango  (A10G, ~10 s fixed, MuBERT features)",
  "AudioGen-medium  (A10G, 16 kHz, effects/sound)",
  "MusicGen-medium  (A10G, 32 kHz, CC-BY-NC)",
  "MAGNeT-medium  (A10G, 32 kHz, non-AR, CC-BY-NC)",
}
local GEN_SCRIPTS = {
  sao            = SCRIPT_DIR .. "research/research_stable_audio_open_modal.py",
  foundation1    = SCRIPT_DIR .. "research/research_foundation1_modal.py",
  acestep_gen    = SCRIPT_DIR .. "research/research_acestep_gen_modal.py",
  inspiremusic_gen = SCRIPT_DIR .. "research/research_inspiremusic_gen_modal.py",
  mustango       = SCRIPT_DIR .. "research/research_mustango_modal.py",
  audiogen       = SCRIPT_DIR .. "research/research_audiogen_modal.py",
  musicgen_gen   = SCRIPT_DIR .. "research/research_musicgen_gen_modal.py",
  magnet         = SCRIPT_DIR .. "research/research_magnet_modal.py",
}
local GEN_MAX_SEC = {
  sao = 47.0, foundation1 = 47.0,
  acestep_gen = 180.0, inspiremusic_gen = 240.0,
  mustango = 10.0, audiogen = 30.0, musicgen_gen = 30.0, magnet = 30.0,
}

-- Edit models (source audio + prompt → transformed audio)
local EDIT_MODELS  = { "sao_edit", "acestep", "musicgen",
                       "melodyflow", "zeta", "inspiremusic" }
local EDIT_LABELS  = {
  "SAO Style Transfer  (A10G, SDEdit init_audio)",
  "ACE-Step 1.5  (A10G, cover/re-style, Apache 2.0)",
  "MusicGen-melody  (A10G, melodic conditioning, CC-BY-NC)",
  "MelodyFlow  (A10G, ≤30 s, flow matching, MIT/CC-BY-NC)",
  "ZETA/AudioLDM2  (A10G, ≤10 s, zero-shot, Apache/CC-BY-SA)",
  "InspireMusic continuation  (A10G, ≤30 s, Apache 2.0)",
}
local EDIT_SCRIPTS = {
  sao_edit    = SCRIPT_DIR .. "research/research_sao_edit_modal.py",
  acestep     = SCRIPT_DIR .. "research/research_acestep_edit_modal.py",
  musicgen    = SCRIPT_DIR .. "research/research_musicgen_melody_modal.py",
  melodyflow  = SCRIPT_DIR .. "research/research_melodyflow_modal.py",
  zeta        = SCRIPT_DIR .. "research/research_zeta_edit_modal.py",
  inspiremusic= SCRIPT_DIR .. "research/research_inspiremusic_modal.py",
}
local EDIT_NEEDS_SECONDS = {
  sao_edit = false, acestep = false, musicgen = true,
  melodyflow = false, zeta = false, inspiremusic = false,
}

local GPUS        = { "A10G", "A100", "T4" }
local INTENSITIES = { "subtle", "moderate", "strong" }
local INTENSITY_LABELS = { "Subtle", "Moderate", "Strong" }

-- ACE-Step 1.5 (acestep_gen) advanced inference parameters — ranges mirror
-- the official ACE-Step Gradio UI per checkpoint type
-- (acestep/ui/gradio/events/generation/model_config.py::get_ui_control_config,
-- is_turbo branch vs. else; and generation_advanced_dit_controls.py's slider
-- bounds). Sliders below already keep values inside these ranges; clamp()
-- is still applied when building the command as a defensive second layer,
-- and the Modal script (research_acestep_gen_modal.py) clamps again
-- server-side.
--
-- CFG (guidance_scale/use_adg/cfg_interval_start-end) only has an effect on
-- the base checkpoint — turbo is CFG-distilled and generate_music() itself
-- hard-overrides guidance_scale to 1.0 for it (verified empirically: a real
-- Modal run with guidance_scale=8.0 on turbo logged "overriding
-- guidance_scale 8.0 -> 1.0"). Those controls are hidden entirely for
-- turbo below rather than just grayed out, since setting them there is a
-- no-op.
local ACESTEP_VARIANTS        = { "turbo", "base" }
local ACESTEP_VARIANT_LABELS  = {
  "Turbo  (fast, 8-20 steps, no CFG)",
  "Base  (32-200 steps, CFG-capable — larger download)",
}
local ACESTEP_STEPS_RANGE_BY_VARIANT = {
  turbo = { 1, 20  },
  base  = { 1, 200 },
}
local ACESTEP_STEPS_DEFAULT_BY_VARIANT = { turbo = 8, base = 32 }
local ACESTEP_GUIDANCE_RANGE   = { 1.0, 15.0 }  -- base only
local ACESTEP_SHIFT_RANGE      = { 1.0, 5.0  }
local ACESTEP_CFG_RANGE        = { 0.0, 1.0  }  -- base only
local ACESTEP_LM_TEMP_RANGE    = { 0.0, 2.0  }
local ACESTEP_LORA_SCALE_RANGE = { 0.0, 1.0  }

local function clamp(v, lo, hi) return math.max(lo, math.min(hi, v)) end

local TEXT2AUDIO_PY = SCRIPT_DIR .. "text2audio.py"
local PROGRESS_F    = TMPDIR .. "t2a_progress.txt"
local LOG_F         = TMPDIR .. "t2a.log"

-- ── STATE ────────────────────────────────────────────────────────
local S = {
  -- Mode: 1=Generate, 2=Edit
  mode              = 1,
  -- Generate mode
  prompt            = "",
  duration          = 8.0,
  gen_model_idx     = 1,
  -- ACE-Step 1.5 advanced inference parameters (acestep_gen only)
  acestep_variant_idx = 1,   -- 1=turbo, 2=base
  acestep_steps       = 8,
  acestep_guidance    = 7.0,
  acestep_shift       = 3.0,
  acestep_seed        = -1,     -- -1 = random
  acestep_use_adg     = false,
  acestep_cfg_start   = 0.0,
  acestep_cfg_end     = 1.0,
  acestep_thinking    = false,
  acestep_lm_temp     = 0.85,
  acestep_lora_path   = "",
  acestep_lora_scale  = 1.0,
  -- Edit mode
  src               = "",
  src_track_name    = "",
  src_track_idx     = -1,
  src_start_offs    = 0,
  src_section_dur   = 0,
  src_item_pos      = nil,
  src_is_section    = false,
  edit_prompt       = "",
  edit_duration     = 10.0,
  intensity_idx     = 2,   -- "moderate"
  edit_model_idx    = 1,
  -- Common
  gpu_idx           = 1,
  -- Runtime
  running           = false,
  done              = false,
  progress          = 0.0,
  status            = "Ready.",
  log               = {},
  out_files         = {},
  log_scroll_to_bottom = false,
}

-- ── CORE HELPERS ─────────────────────────────────────────────────
local function add_log(s)
  table.insert(S.log, tostring(s):sub(1, 200))
  if #S.log > 200 then table.remove(S.log, 1) end
  S.log_scroll_to_bottom = true
end

local function q(s) return common.q(s) end

local _run_id = 0
local function make_run_dir()
  _run_id = _run_id + 1
  local d = TMPDIR .. "t2a_run" .. _run_id .. "/"
  os.execute("mkdir -p " .. q(d))
  return d
end

-- forward declaration (defined further below, referenced earlier)
local import_audio

-- ── PROGRESS ─────────────────────────────────────────────────────
local function read_progress()
  local r = common.read_progress_file(PROGRESS_F)
  if not r then return end

  S.progress = r.pct or S.progress
  if r.msg ~= S.status then
    S.status = r.msg
    add_log(r.msg)
  end

  if r.state == "done" and not S.done then
    S.running   = false
    S.done      = true
    S.out_files = {}
    for _, line in ipairs(r.extra) do
      local p = line:match("^%s*(.-)%s*$")
      if p ~= "" then
        table.insert(S.out_files, p)
      end
    end
    if #S.out_files > 0 then
      add_log("Audio ready: " .. (S.out_files[1]:match("[^/\\]+$") or S.out_files[1]))
      import_audio()
    end

  elseif r.state == "error" and not S.done then
    S.running = false
    S.done    = true
    add_log("ERROR: " .. (r.msg or "?"))
  end
end

-- ── REAPER INTEGRATION ───────────────────────────────────────────
local function detect_section(item, take, src)
  local item_pos    = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
  local item_len    = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
  local start_offs  = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
  local play_rate   = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
  if play_rate == 0 then play_rate = 1.0 end
  local src_len     = reaper.GetMediaSourceLength(src)
  local section_dur = item_len * play_rate
  local is_section  = (start_offs > 0.001) or (section_dur < src_len - 0.001)

  S.src_item_pos    = item_pos
  S.src_start_offs  = start_offs
  S.src_section_dur = section_dur
  S.src_is_section  = is_section
  return is_section, start_offs, section_dur
end

local function _set_src_from_item(item, context_label)
  local take = reaper.GetActiveTake(item)
  if not take then
    reaper.MB("Item has no active take.", "Text2Audio", 0); return false
  end
  local src   = reaper.GetMediaItemTake_Source(take)
  local fname = reaper.GetMediaSourceFileName(src, "")
  if not fname or fname == "" then return false end

  local tr = reaper.GetMediaItemTrack(item)
  if tr then
    local _, tname = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
    S.src_track_name = tname ~= "" and tname
      or ("Track " .. (reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER") or "?"))
    S.src_track_idx  = reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER")
  end

  S.src = fname
  local is_sec, offs, dur = detect_section(item, take, src)
  local kind = is_sec and "split" or context_label
  if is_sec then
    add_log(string.format("Source (%s): %s [%.2fs → %.2fs]",
      kind, fname:match("[^/\\]+$") or fname, offs, offs + dur))
  else
    add_log(string.format("Source (%s): %s", kind, fname:match("[^/\\]+$") or fname))
  end
  return true
end

local function grab_from_reaper()
  local n_items = reaper.CountSelectedMediaItems(0)
  if n_items > 0 then
    local item = reaper.GetSelectedMediaItem(0, 0)
    _set_src_from_item(item, "item")
    return
  end

  local tcnt = reaper.CountSelectedTracks(0)
  if tcnt > 0 then
    local tr   = reaper.GetSelectedTrack(0, 0)
    local icnt = reaper.CountTrackMediaItems(tr)
    for i = 0, icnt - 1 do
      local item = reaper.GetTrackMediaItem(tr, i)
      if _set_src_from_item(item, "track") then return end
    end
    reaper.MB("Selected track has no audio items.", "Text2Audio", 0)
    return
  end

  reaper.MB("No item or track selected in REAPER.", "Text2Audio", 0)
end

-- ── IMPORT AUDIO ─────────────────────────────────────────────────
import_audio = function()
  if #S.out_files == 0 then return end
  local wav_path = S.out_files[1]
  local f = io.open(wav_path, "rb")
  if not f then
    add_log("Error: cannot read " .. wav_path); return
  end
  f:close()

  reaper.Undo_BeginBlock()
  local cursor = S.src_item_pos or reaper.GetCursorPosition()

  -- Base name for the new track
  local model_key
  if S.mode == 1 then
    model_key = GEN_MODELS[S.gen_model_idx] or "sao"
  else
    model_key = EDIT_MODELS[S.edit_model_idx] or "sao_edit"
  end

  local base_name
  if S.mode == 2 and S.src_track_name ~= "" then
    base_name = S.src_track_name
  else
    base_name = "Audio"
  end
  local track_name = base_name .. " [" .. model_key .. "]"

  local tcnt_before = reaper.CountTracks(0)
  reaper.InsertTrackAtIndex(tcnt_before, true)
  local new_track = reaper.GetTrack(0, tcnt_before)
  reaper.GetSetMediaTrackInfo_String(new_track, "P_NAME", track_name, true)
  reaper.SetOnlyTrackSelected(new_track)
  reaper.SetEditCurPos(cursor, false, false)

  reaper.InsertMedia(wav_path, 0)
  reaper.UpdateArrange()
  add_log("Imported: " .. track_name)
  reaper.Undo_EndBlock("Text2Audio: import WAV", -1)
end

-- ── LAUNCH GENERATION/EDITING ────────────────────────────────────
local function clear_run(label)
  local f = io.open(PROGRESS_F, "w")
  if f then f:write("running|0.00|" .. label); f:close() end
  local lf = io.open(LOG_F, "w"); if lf then lf:close() end
  S.running   = true
  S.done      = false
  S.progress  = 0
  S.out_files = {}
  S.log       = {}
  S.status    = label
  S.log_scroll_to_bottom = false
end

local function launch_t2a()
  local run_dir = make_run_dir()

  -- ── GENERATE mode ──
  if S.mode == 1 then
    local prompt = S.prompt:match("^%s*(.-)%s*$")
    if prompt == "" then
      reaper.MB("Write a text prompt before generating.", "Text2Audio", 0)
      return
    end
    local model_key = GEN_MODELS[S.gen_model_idx]
    local script    = GEN_SCRIPTS[model_key]
    local f = io.open(script, "r")
    if not f then
      reaper.MB("Script not found:\n" .. tostring(script), "Text2Audio", 0)
      return
    end
    f:close()

    local label = "Starting " .. (GEN_LABELS[S.gen_model_idx] or model_key) .. "..."
    clear_run(label)
    add_log("Mode: Generate")
    add_log("Model: " .. (GEN_LABELS[S.gen_model_idx] or model_key))
    add_log("GPU: " .. GPUS[S.gpu_idx])
    add_log(string.format("Duration: %.1fs", S.duration))
    add_log("Prompt: " .. prompt:sub(1, 80))

    local extra = ""
    if model_key == "acestep_gen" then
      local variant = ACESTEP_VARIANTS[S.acestep_variant_idx] or "turbo"
      local steps_range = ACESTEP_STEPS_RANGE_BY_VARIANT[variant]
      local steps    = clamp(math.floor(S.acestep_steps + 0.5), steps_range[1], steps_range[2])
      local shift    = clamp(S.acestep_shift, ACESTEP_SHIFT_RANGE[1], ACESTEP_SHIFT_RANGE[2])
      local lm_temp  = clamp(S.acestep_lm_temp, ACESTEP_LM_TEMP_RANGE[1], ACESTEP_LM_TEMP_RANGE[2])

      extra = extra .. " --dit-variant " .. q(variant)
      extra = extra .. string.format(" --steps %d --shift %.2f", steps, shift)
      extra = extra .. string.format(" --seed %d", math.floor(S.acestep_seed + 0.5))

      -- CFG params only meaningful for "base" — omit entirely for turbo
      -- rather than send values the model would silently ignore.
      if variant == "base" then
        local guidance = clamp(S.acestep_guidance, ACESTEP_GUIDANCE_RANGE[1], ACESTEP_GUIDANCE_RANGE[2])
        local cfg_s    = clamp(S.acestep_cfg_start, ACESTEP_CFG_RANGE[1], ACESTEP_CFG_RANGE[2])
        local cfg_e    = clamp(S.acestep_cfg_end, ACESTEP_CFG_RANGE[1], ACESTEP_CFG_RANGE[2])
        if cfg_s > cfg_e then cfg_s, cfg_e = cfg_e, cfg_s end
        extra = extra .. string.format(" --guidance-scale %.2f", guidance)
        if S.acestep_use_adg then extra = extra .. " --use-adg" end
        extra = extra .. string.format(" --cfg-start %.2f --cfg-end %.2f", cfg_s, cfg_e)
      end

      if S.acestep_thinking then
        extra = extra .. " --thinking"
        extra = extra .. string.format(" --lm-temperature %.2f", lm_temp)
      end
      if S.acestep_lora_path:match("%S") then
        local lora_scale = clamp(S.acestep_lora_scale, ACESTEP_LORA_SCALE_RANGE[1], ACESTEP_LORA_SCALE_RANGE[2])
        extra = extra .. " --lora-path " .. q(S.acestep_lora_path)
        extra = extra .. string.format(" --lora-scale %.2f", lora_scale)
        add_log("LoRA adapter: " .. (S.acestep_lora_path:match("[^/\\]+$") or S.acestep_lora_path)
          .. string.format(" (scale %.2f)", lora_scale))
      end
      add_log(string.format(
        "ACE-Step params: variant=%s steps=%d shift=%.1f seed=%d thinking=%s",
        variant, steps, shift, math.floor(S.acestep_seed + 0.5), tostring(S.acestep_thinking)))
    end

    local cmd = string.format(
      '%s %s --shared-dir %s --script %s --model %s --mode generate'
      .. ' --prompt %s --seconds %.2f --gpu %s'
      .. ' --out-dir %s%s --progress %s >>%s 2>&1 &',
      q(PYTHON), q(TEXT2AUDIO_PY),
      q(SHARED_DIR), q(script),
      q(model_key),
      q(prompt), S.duration, q(GPUS[S.gpu_idx]),
      q(run_dir), extra, q(PROGRESS_F), q(LOG_F))

    add_log("Launching Modal process...")
    os.execute(cmd)

  -- ── EDIT mode ──
  else
    if S.src == "" then
      reaper.MB("Select an audio track, item or section first.\n"
        .. "Use the R button to capture the REAPER selection.", "Text2Audio", 0)
      return
    end
    local prompt = S.edit_prompt:match("^%s*(.-)%s*$")
    if prompt == "" then
      reaper.MB("Write the change intent (e.g. 'jazz style with piano').", "Text2Audio", 0)
      return
    end
    local model_key = EDIT_MODELS[S.edit_model_idx]
    local script    = EDIT_SCRIPTS[model_key]
    local f = io.open(script, "r")
    if not f then
      reaper.MB("Script not found:\n" .. tostring(script), "Text2Audio", 0)
      return
    end
    f:close()

    local label = "Starting " .. (EDIT_LABELS[S.edit_model_idx] or model_key) .. "..."
    clear_run(label)
    add_log("Mode: Edit")
    add_log("Model: " .. (EDIT_LABELS[S.edit_model_idx] or model_key))
    add_log("GPU: " .. GPUS[S.gpu_idx])
    add_log("Intensity: " .. INTENSITIES[S.intensity_idx])
    add_log("Source: " .. (S.src:match("[^/\\]+$") or S.src))
    add_log("Prompt: " .. prompt:sub(1, 80))

    local section_args = ""
    if S.src_is_section then
      section_args = string.format(" --start %.6f --duration %.6f",
        S.src_start_offs, S.src_section_dur)
      add_log(string.format("Section: %.2fs → %.2fs",
        S.src_start_offs, S.src_start_offs + S.src_section_dur))
    end

    -- MusicGen needs --seconds
    local seconds_arg = ""
    if EDIT_NEEDS_SECONDS[model_key] then
      seconds_arg = string.format(" --seconds %.2f", S.edit_duration)
    end

    local cmd = string.format(
      '%s %s --shared-dir %s --script %s --model %s --mode edit'
      .. ' --input %s --prompt %s --intensity %s --gpu %s'
      .. ' --out-dir %s%s%s --progress %s >>%s 2>&1 &',
      q(PYTHON), q(TEXT2AUDIO_PY),
      q(SHARED_DIR), q(script),
      q(model_key),
      q(S.src), q(prompt), q(INTENSITIES[S.intensity_idx]), q(GPUS[S.gpu_idx]),
      q(run_dir), section_args, seconds_arg,
      q(PROGRESS_F), q(LOG_F))

    add_log("Launching Modal process...")
    os.execute(cmd)
  end
end

-- ── MODULE ───────────────────────────────────────────────────────
local M = { title = "Text2Audio" }

function M.init()
  if PYTHON_ERR then
    reaper.ShowConsoleMsg("Text2Audio - WARNING: " .. PYTHON_ERR .. "\n")
  end
  add_log("Text2Audio ready.")
  add_log("Python: " .. PYTHON)
  add_log("Backend: " .. TEXT2AUDIO_PY)
end

function M.poll()
  if S.running then read_progress() end
end

function M.draw()
  local g = gui
  local t = theme

  -- ── Mode tabs ──
  local half_w = math.floor((gfx.w - 2 * t.PAD_X - t.SPACING_X) / 2)
  local c_gen_act  = { norm = {0x14/255, 0x5A/255, 0x9C/255},
                       hover= {0x1A/255, 0x72/255, 0xC5/255},
                       active={0x20/255, 0x88/255, 0xE8/255} }
  local c_gen_dim  = { norm = {0x1E/255, 0x1E/255, 0x28/255},
                       hover= {0x28/255, 0x28/255, 0x38/255},
                       active={0x30/255, 0x30/255, 0x44/255} }
  local c_edit_act = { norm = {0x5C/255, 0x2A/255, 0x9C/255},
                       hover= {0x73/255, 0x35/255, 0xC5/255},
                       active={0x8A/255, 0x40/255, 0xE8/255} }
  local c_edit_dim = c_gen_dim

  g.next_width(half_w)
  if g.button("⊕ Generate", half_w, t.sc(30),
      { solid = S.mode == 1 and c_gen_act or c_gen_dim }) then
    S.mode = 1
  end
  g.same_line()
  if g.button("✏ Edit", half_w, t.sc(30),
      { solid = S.mode == 2 and c_edit_act or c_edit_dim }) then
    S.mode = 2
  end
  g.spacing()
  g.separator()

  -- ── SCROLL REGION: entire page (prompt/model/params/button/log) ──
  -- Single, non-nested scroll_region for everything below the mode tabs.
  -- widgets_extra.lua's scroll_region does not support nesting (its clip/
  -- scroll math doesn't compound an outer scroll offset into an inner
  -- one), so the log below prints its lines directly into THIS region
  -- instead of opening its own nested scroll_region — new lines call
  -- widgets.scroll_to_bottom("##t2a_page") to bring the log into view.
  local scroll_h = math.max(t.sc(60), gfx.h - gui.ctx.y - t.PAD_Y)
  widgets.scroll_region("##t2a_page", 0, scroll_h, function()

  g.spacing()

  -- ════════════════════════════════════════════
  if S.mode == 1 then
  -- ── GENERATE MODE ───────────────────────────

    -- Prompt
    g.push_font(t.F.H1)
    g.text("Prompt")
    g.pop_font()
    g.text_disabled("Describe the audio: instrument, BPM, genre, duration, key...")
    g.spacing()

    local changed_p, new_p = widgets.input_textarea("##gen_prompt", S.prompt, 4)
    if changed_p then S.prompt = new_p end

    -- Foundation-1 hint
    if GEN_MODELS[S.gen_model_idx] == "foundation1" then
      g.text_colored(
        "Foundation-1: use TAG format → Instrument, FX, Genre, N Bars, BPM, Key",
        "YELLOW")
    end
    g.spacing()

    -- Duration
    local max_sec = GEN_MAX_SEC[GEN_MODELS[S.gen_model_idx]] or 47.0
    g.row_label("Duration:", t.sc(70))
    g.next_width(-1)
    local ch_dur, new_dur = g.slider_float("##gen_dur", S.duration, 1.0, max_sec, "%.1f s")
    if ch_dur then S.duration = new_dur end

    -- Model
    g.row_label("Model:", t.sc(70))
    g.next_width(-1)
    S.gen_model_idx = widgets.combo("##gen_model", S.gen_model_idx, GEN_LABELS)

    -- GPU
    g.row_label("GPU:", t.sc(70))
    g.next_width(t.sc(90))
    S.gpu_idx = widgets.combo("##gen_gpu", S.gpu_idx, GPUS)
    g.spacing()

    -- ACE-Step 1.5 advanced inference parameters + LoRA adapter
    if GEN_MODELS[S.gen_model_idx] == "acestep_gen" then
      if widgets.collapsing_header("ACE-Step advanced parameters", false) then
        local lw = t.sc(90)
        local rv, nv

        -- Checkpoint variant — decides which params below are shown/active.
        -- Switching resets Steps to that variant's own default (matches the
        -- official Gradio UI's own model-type-change behavior) since 8
        -- steps (turbo's default) is a poor default for base, and vice versa.
        g.row_label("Checkpoint:", lw)
        g.next_width(-1)
        local old_variant_idx = S.acestep_variant_idx
        S.acestep_variant_idx = widgets.combo("##ace_variant", S.acestep_variant_idx, ACESTEP_VARIANT_LABELS)
        if S.acestep_variant_idx ~= old_variant_idx then
          local variant = ACESTEP_VARIANTS[S.acestep_variant_idx] or "turbo"
          S.acestep_steps = ACESTEP_STEPS_DEFAULT_BY_VARIANT[variant]
        end
        local variant = ACESTEP_VARIANTS[S.acestep_variant_idx] or "turbo"
        local steps_range = ACESTEP_STEPS_RANGE_BY_VARIANT[variant]
        if variant == "base" then
          g.text_disabled("Base checkpoint downloads separately on first use (larger than turbo).")
        end
        g.spacing()

        g.row_label("Steps:", lw)
        g.next_width(t.sc(120))
        rv, nv = g.slider_int("##ace_steps", S.acestep_steps,
          steps_range[1], steps_range[2])
        if rv then S.acestep_steps = nv end
        g.same_line(t.sc(14)); g.inline_text("Shift:")
        g.same_line(t.sc(6)); g.next_width(t.sc(120))
        rv, nv = g.slider_float("##ace_shift", S.acestep_shift,
          ACESTEP_SHIFT_RANGE[1], ACESTEP_SHIFT_RANGE[2], "%.2f")
        if rv then S.acestep_shift = nv end

        g.row_label("Seed:", lw)
        g.next_width(t.sc(120))
        local seed_changed, seed_new = widgets.input_text("##ace_seed", tostring(S.acestep_seed))
        if seed_changed then
          local n = tonumber(seed_new)
          if n then S.acestep_seed = math.floor(n) end
        end
        g.same_line(t.sc(10)); g.text_disabled("(-1 = random)")

        -- CFG controls — base checkpoint only; turbo is CFG-distilled and
        -- generate_music() hard-overrides guidance_scale to 1.0 for it, so
        -- these are hidden entirely for turbo rather than shown-but-inert.
        if variant == "base" then
          g.spacing()
          g.row_label("Guidance:", lw)
          g.next_width(t.sc(120))
          rv, nv = g.slider_float("##ace_guidance", S.acestep_guidance,
            ACESTEP_GUIDANCE_RANGE[1], ACESTEP_GUIDANCE_RANGE[2], "%.2f")
          if rv then S.acestep_guidance = nv end

          local chg, nv2 = g.checkbox("Use ADG##ace_adg", S.acestep_use_adg)
          if chg then S.acestep_use_adg = nv2 end

          g.row_label("CFG start:", lw)
          g.next_width(t.sc(120))
          rv, nv = g.slider_float("##ace_cfg_s", S.acestep_cfg_start,
            ACESTEP_CFG_RANGE[1], ACESTEP_CFG_RANGE[2], "%.2f")
          if rv then S.acestep_cfg_start = nv end
          g.same_line(t.sc(14)); g.inline_text("CFG end:")
          g.same_line(t.sc(6)); g.next_width(t.sc(120))
          rv, nv = g.slider_float("##ace_cfg_e", S.acestep_cfg_end,
            ACESTEP_CFG_RANGE[1], ACESTEP_CFG_RANGE[2], "%.2f")
          if rv then S.acestep_cfg_end = nv end
        end

        g.spacing()
        local tchg, tnv = g.checkbox("Thinking (5Hz LM planner)##ace_think", S.acestep_thinking)
        if tchg then S.acestep_thinking = tnv end
        if S.acestep_thinking then
          g.row_label("LM temp.:", lw)
          g.next_width(t.sc(120))
          rv, nv = g.slider_float("##ace_lmtemp", S.acestep_lm_temp,
            ACESTEP_LM_TEMP_RANGE[1], ACESTEP_LM_TEMP_RANGE[2], "%.2f")
          if rv then S.acestep_lm_temp = nv end
        end

        g.spacing()
        g.text("LoRA / LoKr adapter (optional):")
        g.row_label("Folder:", lw)
        g.next_width(-(2 * t.SPACING_X + t.sc(44)))
        local pchg, pnv = widgets.input_text("##ace_lora_path", S.acestep_lora_path)
        if pchg then S.acestep_lora_path = pnv end
        g.same_line()
        if g.button("...", t.sc(44), t.ITEM_H) then
          if reaper.APIExists("JS_Dialog_BrowseForFolder") then
            local ok, folder = reaper.JS_Dialog_BrowseForFolder("LoRA/LoKr adapter folder", "")
            if ok and folder and folder ~= "" then S.acestep_lora_path = folder end
          else
            reaper.MB(
              "Folder browsing requires the js_ReaScriptAPI extension "
              .. "(install via ReaPack).\nYou can also paste the adapter "
              .. "folder path directly into the text field.",
              "ACE-Step LoRA adapter", 0)
          end
        end
        if S.acestep_lora_path:match("%S") then
          g.row_label("Scale:", lw)
          g.next_width(t.sc(120))
          rv, nv = g.slider_float("##ace_lora_scale", S.acestep_lora_scale,
            ACESTEP_LORA_SCALE_RANGE[1], ACESTEP_LORA_SCALE_RANGE[2], "%.2f")
          if rv then S.acestep_lora_scale = nv end
          g.text_disabled("Folder must contain adapter_config.json + adapter_model.safetensors, "
            .. "or a lokr_weights.safetensors (Side-Step / PEFT output).")
        end
      end
      g.spacing()
    end

    -- Mustango fixed duration hint
    if GEN_MODELS[S.gen_model_idx] == "mustango" then
      g.text_colored("Mustango: fixed duration ~10 s (slider is ignored)", "YELLOW")
    end
    -- Cost hint
    g.text_disabled("A10G: ~$0.05/min  |  SAO/Foundation-1 ~20-40 s  |  ACE-Step/MusicGen ~30-60 s")

  -- ════════════════════════════════════════════
  else
  -- ── EDIT MODE ───────────────────────────────

    -- Source
    g.push_font(t.F.H1)
    g.text("Source audio")
    g.pop_font()

    g.row_label("Source:", t.sc(54))
    local display_src = (S.src_track_name ~= "")
      and (S.src_track_name .. "  (" .. (S.src:match("[^/\\]+$") or "") .. ")")
      or S.src
    g.next_width(-(2 * t.SPACING_X + 2 * t.sc(44)))
    widgets.input_text("##src_disp", display_src, { readonly = true })
    g.same_line()
    if g.button("...", t.sc(44), t.ITEM_H) then
      local ok, fn = reaper.GetUserFileNameForRead("", "Open audio", "wav")
      if ok then
        S.src = fn; S.src_track_name = ""; S.src_track_idx = -1
        S.src_is_section = false; S.src_item_pos = nil
      end
    end
    g.same_line()
    if g.button("R", t.sc(44), t.ITEM_H) then grab_from_reaper() end

    if S.src_track_name ~= "" then
      g.text_disabled("Track selected  |  click R to update")
    else
      g.text_disabled("Click R to use the active REAPER track/item/split")
    end
    if S.src_is_section then
      g.text_colored(string.format("Section: %.2fs → %.2fs  (%.2fs)",
        S.src_start_offs, S.src_start_offs + S.src_section_dur, S.src_section_dur),
        "YELLOW")
    end
    g.spacing()

    -- Change intent prompt
    g.push_font(t.F.H1)
    g.text("Change intent")
    g.pop_font()
    g.text_disabled("Describe how to transform the audio (e.g. 'jazz version with piano')")
    g.spacing()

    local changed_ep, new_ep = widgets.input_textarea("##edit_prompt", S.edit_prompt, 3)
    if changed_ep then S.edit_prompt = new_ep end
    g.spacing()

    -- Edit model
    g.row_label("Model:", t.sc(80))
    g.next_width(-1)
    S.edit_model_idx = widgets.combo("##edit_model", S.edit_model_idx, EDIT_LABELS)
    g.spacing()

    -- Intensity (not applicable to MusicGen or InspireMusic continuation)
    local cur_edit_model = EDIT_MODELS[S.edit_model_idx]
    local EDIT_NO_INTENSITY = { musicgen = true, inspiremusic = true }
    if not EDIT_NO_INTENSITY[cur_edit_model] then
      g.row_label("Intensity:", t.sc(80))
      g.next_width(t.sc(170))
      S.intensity_idx = widgets.combo("##intensity", S.intensity_idx, INTENSITY_LABELS)
      g.same_line(t.sc(12))
      local hints = {
        subtle   = "Preserves original structure",
        moderate = "Balance between transformation/fidelity",
        strong   = "Deep transformation",
      }
      g.text_disabled(hints[INTENSITIES[S.intensity_idx]] or "")
      g.spacing()
    end

    -- Duration (MusicGen only)
    if EDIT_NEEDS_SECONDS[cur_edit_model] then
      g.row_label("Duration:", t.sc(80))
      g.next_width(-(t.sc(50) + t.SPACING_X))
      local ch_ed, new_ed = g.slider_float("##edit_dur", S.edit_duration, 1.0, 30.0, "%.1f s")
      if ch_ed then S.edit_duration = new_ed end
      g.same_line()
      g.text(string.format("%.1fs", S.edit_duration))
      g.spacing()
    end

    -- GPU
    g.row_label("GPU:", t.sc(80))
    g.next_width(t.sc(90))
    S.gpu_idx = widgets.combo("##edit_gpu", S.gpu_idx, GPUS)
    g.spacing()

    -- Hints per model
    local model_hints = {
      sao_edit    = "SAO SDEdit: reuses weights already downloaded from SAO 1.0 (no extra setup cost)",
      acestep     = "ACE-Step: requires initial setup (~5 GB). Apache 2.0, commercial use allowed.",
      musicgen    = "MusicGen-melody: melodic conditioning. CC-BY-NC, non-commercial use only.",
      melodyflow  = "MelodyFlow: flow matching with latent inversion. High fidelity. ≤30 s.",
      zeta        = "ZETA/AudioLDM2: DDIM inversion zero-shot. Output 16 kHz mono ≤10 s.",
      inspiremusic= "InspireMusic: continues the audio with the style from the prompt. ≤30 s.",
    }
    g.text_disabled(model_hints[cur_edit_model] or "")
  end

  -- ── Common section ───────────────────────────────────────────
  g.spacing()
  g.separator()
  g.spacing()

  -- Main button
  local btn_colors = {
    norm   = S.mode == 1 and {0x14/255, 0x6A/255, 0x3C/255} or {0x3C/255, 0x14/255, 0x6A/255},
    hover  = S.mode == 1 and {0x1A/255, 0x88/255, 0x4D/255} or {0x4D/255, 0x1A/255, 0x88/255},
    active = S.mode == 1 and {0x22/255, 0xA5/255, 0x5E/255} or {0x5E/255, 0x22/255, 0xA5/255},
  }
  local btn_lbl = S.running
    and (S.mode == 1 and "[ Generating... ]" or "[ Editing... ]")
    or  (S.mode == 1 and "GENERATE AUDIO"    or "EDIT AUDIO")

  g.begin_disabled(S.running)
  g.next_width(-1)
  if g.button(btn_lbl, nil, t.sc(36), { solid = btn_colors }) then
    launch_t2a()
  end
  g.end_disabled()
  g.spacing()

  -- Progress bar
  local pct_str = string.format("%d%%", math.floor(S.progress * 100))
  g.progress_bar(S.progress, nil, t.sc(16), pct_str)

  local status_color = S.running and "YELLOW"
    or (S.done and #S.out_files > 0 and "GREEN")
    or (S.done and "RED")
    or "FG_DIM"
  g.text_colored(S.status:sub(1, 90), status_color)
  g.spacing()

  -- Logs
  if widgets.collapsing_header("Logs", true) then
    if g.button("Copy log", t.sc(90), t.ITEM_H) then
      local ok, _ = pcall(function()
        reaper.CF_SetClipboard(table.concat(S.log, "\n"))
      end)
      if not ok then
        reaper.ShowConsoleMsg(table.concat(S.log, "\n") .. "\n")
      end
    end
    g.same_line()
    if g.button("Clear", t.sc(70), t.ITEM_H) then S.log = {} end
    g.spacing()

    -- No auto-scroll-to-bottom here: the log now shares the page-level
    -- scroll_region with everything else, so forcing it to the bottom on
    -- every new line would yank the whole page out from under the user
    -- while they're scrolled up looking at something else.
    S.log_scroll_to_bottom = false

    g.push_font(t.F.MONO)
    for i = 1, #S.log do
      local ln = S.log[i]
      if ln:find("^ERROR") then
        g.text_colored(ln, "RED")
      else
        g.text_colored(ln, "LOG_FG")
      end
    end
    g.pop_font()
  end

  end)  -- end scroll_region ##t2a_page
end

return M
