-- @description AI Music Lab - Audio2Midi — Audio to MIDI transcription
-- @version 1.0
-- @author AI Music Lab
-- @about Transcribes a track, item or audio section to MIDI using cloud AI
--        models (Modal). Supports MIROS and YourMT3+.
--        Input: selected REAPER track, item or split (section).
--        Output: a new MIDI track (mono-instrument) or a folder of tracks
--        (multi-instrument), inserted at the source audio position.
--        Native gfx UI: no external REAPER extension dependencies.

-- ── PATHS + LIB ──────────────────────────────────────────────────
local _info      = debug.getinfo(1, "S")
local SCRIPT_DIR = _info.source:match("@?(.*[/\\])") or ""

-- shared/ is sibling of Audio2Midi/
local SHARED_DIR = SCRIPT_DIR .. "../shared/"
package.path = SHARED_DIR .. "lib/?.lua;" .. package.path

local common  = require("common")
local theme   = require("theme")
local gui     = require("gui")
local widgets = require("widgets_extra")

local HOME      = common.HOME
local TMPDIR    = common.TMPDIR
local PYTHON, PYTHON_ERR = common.detect_reaper_python()
if PYTHON_ERR then
  reaper.ShowConsoleMsg("Audio2Midi - WARNING: " .. PYTHON_ERR .. "\n")
end

-- ── CONSTANTS ────────────────────────────────────────────────────
local A2M_MODELS  = { "miros", "yourmt3" }
local A2M_LABELS  = {
  "MIROS  (multi-instr., A10G, internal use)",
  "YourMT3+  (multi-instr., Apache 2.0)",
}
local A2M_SCRIPTS = {
  miros   = SCRIPT_DIR .. "research/research_miros_modal.py",
  yourmt3 = SCRIPT_DIR .. "research/research_yourmt3_modal.py",
}
-- Minimum recommended GPU per model (A10G for MIROS; T4 sufficient for YourMT3+)
local A2M_GPU_DEFAULT = { miros = 1, yourmt3 = 3 }  -- index in A2M_GPUS
local A2M_GPUS   = { "A10G", "A100", "T4" }

local TRANSCRIBE_PY = SCRIPT_DIR .. "transcribe.py"
local PROGRESS_F    = TMPDIR .. "a2m_progress.txt"
local LOG_F         = TMPDIR .. "a2m.log"

-- ── STATE ────────────────────────────────────────────────────────
local S = {
  -- source
  src             = "",
  src_track_name  = "",
  src_track_idx   = -1,
  src_start_offs  = 0,
  src_section_dur = 0,
  src_item_pos    = nil,
  src_is_section  = false,
  -- model
  model_idx       = 1,
  gpu_idx         = A2M_GPU_DEFAULT[A2M_MODELS[1]],
  beat_tracking   = true,
  -- runtime
  running         = false,
  done            = false,
  progress        = 0.0,
  status          = "Ready.",
  log             = {},
  out_files       = {},
  n_instruments   = -1,
  log_scroll_to_bottom = false,
}

-- ── CORE HELPERS ─────────────────────────────────────────────────
local function add_log(s)
  table.insert(S.log, tostring(s):sub(1, 200))
  if #S.log > 200 then table.remove(S.log, 1) end
  S.log_scroll_to_bottom = true
end

local function q(s) return common.q(s) end

-- Generate a unique temp directory per run (avoids Modal skipping when
-- transcribed_cuda.mid already exists).
local _run_id = 0
local function make_run_dir()
  _run_id = _run_id + 1
  local d = TMPDIR .. "a2m_run" .. _run_id .. "/"
  os.execute("mkdir -p " .. q(d))
  return d
end

-- ── SETUP CHECK (async at startup) ───────────────────────────────
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
    python        = "Python REAPER",
    uv            = "uv",
    ["modal-cli"] = "Modal CLI",
    ["modal-auth"]= "Modal not authenticated",
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
    S.n_instruments = -1
    for _, line in ipairs(r.extra) do
      local p = line:match("^%s*(.-)%s*$")
      if p ~= "" then
        local n = p:match("^INSTRUMENTS|(.+)$")
        if n then
          S.n_instruments = tonumber(n) or -1
        else
          table.insert(S.out_files, p)
        end
      end
    end
    if #S.out_files > 0 then
      add_log(string.format("MIDI ready (%s instrument%s)",
        S.n_instruments >= 0 and tostring(S.n_instruments) or "?",
        S.n_instruments ~= 1 and "s" or ""))
      import_midi()
    end

  elseif r.state == "error" and not S.done then
    S.running = false
    S.done    = true
    add_log("ERROR: " .. (r.msg or "?"))
  end
end

-- ── REAPER INTEGRATION ───────────────────────────────────────────
local function detect_section(item, take, src)
  local item_pos   = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
  local item_len   = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
  local start_offs = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
  local play_rate  = reaper.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE")
  if play_rate == 0 then play_rate = 1.0 end
  local src_len    = reaper.GetMediaSourceLength(src)
  local section_dur = item_len * play_rate
  local is_section = (start_offs > 0.001) or (section_dur < src_len - 0.001)

  S.src_item_pos    = item_pos
  S.src_start_offs  = start_offs
  S.src_section_dur = section_dur
  S.src_is_section  = is_section
  return is_section, start_offs, section_dur
end

local function _set_src_from_item(item, context_label)
  local take = reaper.GetActiveTake(item)
  if not take then
    reaper.MB("Item has no active take.", "Audio2Midi", 0); return false
  end
  local src   = reaper.GetMediaItemTake_Source(take)
  local fname = reaper.GetMediaSourceFileName(src, "")
  if not fname or fname == "" then return false end

  -- Record parent track for naming the output
  local tr = reaper.GetMediaItemTrack(item)
  if tr then
    local _, tname = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
    S.src_track_name = tname ~= "" and tname
      or ("Track " .. (reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER") or "?"))
    S.src_track_idx = reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER")
  end

  S.src = fname
  local is_sec, offs, dur = detect_section(item, take, src)
  local kind = is_sec and "split" or context_label
  if is_sec then
    add_log(string.format("Source (%s): %s [%.2fs → %.2fs]",
      kind, fname:match("[^/\\]+$") or fname, offs, offs + dur))
  else
    add_log(string.format("Source (%s): %s", kind,
      fname:match("[^/\\]+$") or fname))
  end
  return true
end

local function grab_from_reaper()
  -- Level 1/2: selected item or split (priority over track, because in REAPER
  -- selecting an item also selects its track — without this priority we would
  -- always pick the first item on the track, not the selected one).
  local n_items = reaper.CountSelectedMediaItems(0)
  if n_items > 0 then
    local item = reaper.GetSelectedMediaItem(0, 0)
    _set_src_from_item(item, "item")
    return
  end

  -- Level 3: selected track with no active item → use first track item
  local tcnt = reaper.CountSelectedTracks(0)
  if tcnt > 0 then
    local tr = reaper.GetSelectedTrack(0, 0)
    local icnt = reaper.CountTrackMediaItems(tr)
    for i = 0, icnt - 1 do
      local item = reaper.GetTrackMediaItem(tr, i)
      if _set_src_from_item(item, "track") then return end
    end
    reaper.MB("Selected track has no active audio items.", "Audio2Midi", 0)
    return
  end

  reaper.MB("No item or track selected in REAPER.", "Audio2Midi", 0)
end

-- ── IMPORT MIDI ──────────────────────────────────────────────────
function import_midi()
  if #S.out_files == 0 then return end
  local mid_path = S.out_files[1]
  local f = io.open(mid_path, "rb")
  if not f then
    add_log("Error: cannot read " .. mid_path); return
  end
  f:close()

  reaper.Undo_BeginBlock()
  local cursor = S.src_item_pos or reaper.GetCursorPosition()

  -- Base name for the track(s)
  local base_name = S.src_track_name ~= "" and S.src_track_name
    or (S.src:match("([^/\\]+)%.%w+$") or "audio")
  local model_tag = A2M_MODELS[S.model_idx] or "midi"

  -- Record track count before import
  local tcnt_before = reaper.CountTracks(0)

  -- Insert a new track and position the cursor
  reaper.InsertTrackAtIndex(tcnt_before, true)
  local new_track = reaper.GetTrack(0, tcnt_before)
  reaper.SetOnlyTrackSelected(new_track)
  reaper.SetEditCurPos(cursor, false, false)

  -- InsertMedia on the selected track
  reaper.InsertMedia(mid_path, 0)

  -- Detect how many tracks REAPER created (multi-track MIDI can open several)
  local tcnt_after = reaper.CountTracks(0)
  local delta = tcnt_after - tcnt_before

  if delta <= 0 then
    -- No tracks created — InsertMedia failed or went to an existing track
    add_log("Warning: InsertMedia did not add new tracks.")
    reaper.DeleteTrack(new_track)
    reaper.Undo_EndBlock("Audio2Midi: import MIDI", -1)
    return
  end

  if delta == 1 then
    -- Mono-instrument: name the track cleanly
    local track_name = base_name .. " [MIDI " .. model_tag .. "]"
    reaper.GetSetMediaTrackInfo_String(new_track, "P_NAME", track_name, true)
    add_log("Imported: " .. track_name)
  else
    -- Multi-instrument: wrap in a folder with an EMPTY parent.
    local folder_name = base_name .. " [MIDI " .. model_tag .. "]"

    -- Insert an empty folder track BEFORE the imported block.
    -- InsertMedia already placed the first instrument in new_track (tcnt_before);
    -- inserting here at tcnt_before shifts that track to tcnt_before+1,
    -- making it the first child, not the folder.
    reaper.InsertTrackAtIndex(tcnt_before, true)
    local folder_tr = reaper.GetTrack(0, tcnt_before)
    reaper.GetSetMediaTrackInfo_String(folder_tr, "P_NAME", folder_name, true)
    reaper.SetMediaTrackInfo_Value(folder_tr, "I_FOLDERDEPTH", 1)

    -- Name the instrument tracks (now at tcnt_before+1 .. tcnt_before+delta).
    -- Preserve the name InsertMedia may have taken from the SMF if it exists.
    for i = 1, delta do
      local tr = reaper.GetTrack(0, tcnt_before + i)
      if tr then
        local _, existing = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
        if existing == "" then
          reaper.GetSetMediaTrackInfo_String(tr, "P_NAME",
            folder_name .. " " .. i, true)
        end
      end
    end

    -- Close the folder on the last child track.
    local last_tr = reaper.GetTrack(0, tcnt_before + delta)
    if last_tr then
      reaper.SetMediaTrackInfo_Value(last_tr, "I_FOLDERDEPTH", -1)
    end

    add_log(string.format("Imported into folder '%s' (%d tracks)", folder_name, delta))
  end

  reaper.UpdateArrange()
  reaper.Undo_EndBlock("Audio2Midi: import MIDI", -1)
end

-- ── LAUNCH TRANSCRIPTION ─────────────────────────────────────────
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

local function launch_transcribe()
  if S.src == "" then
    reaper.MB("Select an audio track, item or section first.\n"
      .. "Use the R button to capture the REAPER selection.", "Audio2Midi", 0)
    return
  end

  local model_key = A2M_MODELS[S.model_idx]
  local script    = A2M_SCRIPTS[model_key]
  if not script then
    reaper.MB("Model script not configured: " .. tostring(model_key), "Audio2Midi", 0)
    return
  end
  local f = io.open(script, "r")
  if not f then
    reaper.MB("Model script not found:\n" .. script ..
      "\n\nCheck that Audio2Midi/research/ is in place.",
      "Audio2Midi", 0)
    return
  end
  f:close()

  local label = "Starting " .. (A2M_LABELS[S.model_idx] or model_key) .. "..."
  clear_run(label)
  add_log("Model: " .. (A2M_LABELS[S.model_idx] or model_key))
  add_log("GPU: " .. A2M_GPUS[S.gpu_idx])
  add_log("Beat tracking: " .. (S.beat_tracking and "yes" or "no"))

  local section_args = ""
  if S.src_is_section then
    section_args = string.format(" --start %.6f --duration %.6f",
      S.src_start_offs, S.src_section_dur)
    add_log(string.format("Section: %.2fs → %.2fs",
      S.src_start_offs, S.src_start_offs + S.src_section_dur))
  end

  local beat_arg = S.beat_tracking and "" or " --no-beat-tracking"
  local run_dir  = make_run_dir()

  local cmd = string.format(
    '%s %s --shared-dir %s --script %s --input %s --out-dir %s'
    .. ' --model %s --gpu %s%s%s --progress %s >>%s 2>&1 &',
    q(PYTHON), q(TRANSCRIBE_PY),
    q(SHARED_DIR), q(script),
    q(S.src), q(run_dir),
    q(model_key), q(A2M_GPUS[S.gpu_idx]),
    section_args, beat_arg,
    q(PROGRESS_F), q(LOG_F))

  add_log("Launching Modal process...")
  os.execute(cmd)
end

-- ── GFX INIT ─────────────────────────────────────────────────────
if gfx.w > 0 then gfx.quit() end
local LOGICAL_W = 540
gfx.init("Audio2Midi", LOGICAL_W, 600)
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
  g.text("Audio → MIDI")
  g.pop_font()
  g.same_line(10)
  g.text_colored("● REAPER OK", "GREEN")
  g.separator()
  g.spacing()

  -- Source
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
  g.separator()
  g.spacing()

  -- Model
  g.row_label("Model:", t.sc(68))
  g.next_width(-1)
  local old_idx = S.model_idx
  S.model_idx = widgets.combo("##a2m_model", S.model_idx, A2M_LABELS)
  if S.model_idx ~= old_idx then
    -- Update default GPU when model changes
    S.gpu_idx = A2M_GPU_DEFAULT[A2M_MODELS[S.model_idx]] or 1
  end

  -- GPU
  g.row_label("GPU:", t.sc(68))
  g.next_width(t.sc(100))
  S.gpu_idx = widgets.combo("##a2m_gpu", S.gpu_idx, A2M_GPUS)

  -- Beat tracking (only shows a note for YourMT3+ which doesn't use it)
  g.same_line(t.sc(18))
  local bt_changed, bt_new = g.checkbox("Beat tracking##bt", S.beat_tracking)
  if bt_changed then S.beat_tracking = bt_new end
  if S.model_idx == 2 then  -- YourMT3+ has no flag
    g.same_line(t.sc(8))
    g.text_disabled("(N/A for YourMT3+)")
  end

  g.spacing()

  -- Cost hint
  local hints = {
    "A10G: ~$0.05/min  |  MIROS requires A10G minimum (flash-attn Ampere+)",
    "A10G: ~$0.05/min  |  YourMT3+ works on T4 and above",
  }
  g.text_disabled(hints[S.model_idx] or "")

  g.spacing()
  g.separator()
  g.spacing()

  -- GENERATE MIDI button
  local btn_color = {
    norm   = { 0x1A/255, 0x7A/255, 0x3C/255 },
    hover  = { 0x22/255, 0x99/255, 0x4D/255 },
    active = { 0x2A/255, 0xB5/255, 0x5C/255 },
  }
  local btn_lbl = S.running and "[ Transcribing... ]" or "GENERATE MIDI"
  g.begin_disabled(S.running)
  g.next_width(-1)
  if g.button(btn_lbl, nil, t.sc(36), { solid = btn_color }) then
    launch_transcribe()
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

  -- Log
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
add_log("Audio2Midi ready.")
add_log("Python: " .. PYTHON)
add_log("MIROS:  " .. A2M_SCRIPTS.miros)
add_log("YourMT3+: " .. A2M_SCRIPTS.yourmt3)
launch_setup_check()
reaper.defer(loop)
