-- @description AI Music Lab - Text2Audio - AI audio generation and editing
-- @version 1.1
-- @author AI Music Lab
-- @about Generates audio from text or edits an existing audio using cloud AI
--        models (Modal). Two modes:
--          · Generate: text prompt → stereo WAV
--                     (SAO, Foundation-1, ACE-Step, InspireMusic,
--                      Mustango, AudioGen, MusicGen, MAGNeT)
--          · Edit:    selected item/track + intent → transformed WAV
--                     (SAO style transfer, ACE-Step 1.5, MusicGen-melody,
--                      MelodyFlow, ZETA/AudioLDM2, InspireMusic continuation)
--        Edit input: selected REAPER track, item or split (section).
--        Output: new audio track with the generated WAV, at the source position.
--        Native gfx UI: no external REAPER extension dependencies.

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
if PYTHON_ERR then
  reaper.ShowConsoleMsg("Text2Audio - WARNING: " .. PYTHON_ERR .. "\n")
end

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

-- ── SETUP CHECK ──────────────────────────────────────────────────
local SETUP_CHECK_F = TMPDIR .. "reaperai_setup_check.txt"
local SETUP_HELPER  = SHARED_DIR .. "setup_helpers.py"
local setup_missing = {}
local setup_checked = false

local function launch_setup_check()
  local f = io.open(SETUP_CHECK_F, "w")
  if f then f:write("running|0.00|..."); f:close() end
  local cmd = string.format('%s %s check --progress %s >>%s 2>&1 &',
    q(PYTHON), q(SETUP_HELPER), q(SETUP_CHECK_F),
    q(TMPDIR .. "reaperai_setup.log"))
  os.execute(cmd)
end

local function poll_setup_check()
  if setup_checked then return end
  local r = common.read_progress_file(SETUP_CHECK_F)
  if not r or r.state ~= "done" then return end
  setup_checked = true
  local CORE_LABELS = {
    python         = "Python REAPER",
    uv             = "uv",
    ["modal-cli"]  = "Modal CLI",
    ["modal-auth"] = "Modal not authenticated",
  }
  for _, line in ipairs(r.extra) do
    local name, status = line:match("^CHECK|([^|]+)|([^|]+)|")
    if name and status == "missing" and CORE_LABELS[name] then
      table.insert(setup_missing, CORE_LABELS[name])
    end
  end
end

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
function import_audio()
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

    local cmd = string.format(
      '%s %s --shared-dir %s --script %s --model %s --mode generate'
      .. ' --prompt %s --seconds %.2f --gpu %s'
      .. ' --out-dir %s --progress %s >>%s 2>&1 &',
      q(PYTHON), q(TEXT2AUDIO_PY),
      q(SHARED_DIR), q(script),
      q(model_key),
      q(prompt), S.duration, q(GPUS[S.gpu_idx]),
      q(run_dir), q(PROGRESS_F), q(LOG_F))

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

-- ── GFX INIT ─────────────────────────────────────────────────────
if gfx.w > 0 then gfx.quit() end
local LOGICAL_W = 560
gfx.init("Text2Audio", LOGICAL_W, 700)
gfx.ext_retina = 1
theme.init_fonts()

-- ── MAIN LOOP ────────────────────────────────────────────────────
local _scale_init = false

local function loop()
  if not _scale_init then
    _scale_init = true
    local s = math.floor(gfx.w / LOGICAL_W + 0.5)
    if s > 1 then
      theme.apply_scale(s)
      theme.init_fonts(s)
    end
  end

  gui.frame_begin()
  if gui.ctx.should_close then gfx.quit(); return end

  local g = gui
  local t = theme

  -- Setup banner
  poll_setup_check()
  if setup_checked and #setup_missing > 0 then
    g.text_wrapped("⚠  Incomplete setup: " .. table.concat(setup_missing, " · "))
    g.text_disabled("Load shared/Setup.lua in Actions > Load ReaScript to configure.")
    g.spacing()
  end

  -- Header
  g.push_font(t.F.H1)
  g.text("Text → Audio")
  g.pop_font()
  g.same_line(10)
  g.text_colored("● REAPER OK", "GREEN")
  g.separator()
  g.spacing()

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

    if S.log_scroll_to_bottom then
      widgets.scroll_to_bottom("##logscroll")
      S.log_scroll_to_bottom = false
    end

    g.push_font(t.F.MONO)
    local log_h = math.max(t.sc(60), gfx.h - gui.ctx.y - t.PAD_Y - t.sc(10))
    widgets.scroll_region("##logscroll", 0, log_h, function()
      for i = 1, #S.log do
        local ln = S.log[i]
        if ln:find("^ERROR") then
          g.text_colored(ln, "RED")
        else
          g.text_colored(ln, "LOG_FG")
        end
      end
    end, { hscroll = true })
    g.pop_font()
  end

  gui.frame_end()

  if S.running then read_progress() end
  reaper.defer(loop)
end

-- ── STARTUP ──────────────────────────────────────────────────────
add_log("Text2Audio ready.")
add_log("Python: " .. PYTHON)
add_log("Backend: " .. TEXT2AUDIO_PY)
launch_setup_check()
reaper.defer(loop)
