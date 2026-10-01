-- AI Music Lab - Stem Separator panel (Demucs + SAM Audio)
-- Panel module: no gfx.init / reaper.defer of its own. Loaded by the
-- unified ai-music-lab.lua container, which owns the window/main loop
-- and calls M.init() once, M.poll() every frame, M.draw() when active.

-- ── PATHS + LIB ──────────────────────────────────────────────────
local _info      = debug.getinfo(1, "S")
local SCRIPT_DIR = _info.source:match("@?(.*[/\\])") or ""

-- shared/ is sibling of StemsSeparator/
local SHARED_DIR = SCRIPT_DIR .. "../shared/"
package.path = SHARED_DIR .. "lib/?.lua;" .. package.path

local common  = require("common")
local theme   = require("theme")
local gui     = require("gui")
local widgets = require("widgets_extra")
local track_placement = require("track_placement")

local HOME       = common.HOME
local TMPDIR     = common.TMPDIR
local SAM_DIR    = SCRIPT_DIR
local SAM_SCRIPT = "modal_sam_audio.py"
local DEMUCS_PY  = SCRIPT_DIR .. "separate_demucs.py"
local SAM_PY     = SCRIPT_DIR .. "separate_sam.py"
local PROGRESS_F = TMPDIR .. "stemsep_progress.txt"
local LOG_F      = TMPDIR .. "stemsep.log"

local PYTHON, PYTHON_ERR = common.detect_reaper_python()

-- ── CONSTANTS ────────────────────────────────────────────────────
-- Single unified Model list (replaces the old DEMUCS/SAM AUDIO sub-tabs):
-- picking a model drives which parameter section is shown below, the same
-- pattern MidiGenerator/AudioGenerator use for their own Model dropdowns.
-- Ordered by real-world popularity (community adoption / HF downloads)
local SS_MODELS = { "htdemucs", "htdemucs_ft", "mdx_extra", "htdemucs_6s",
                    "sam_large", "sam_base" }
local SS_LABELS = {
  "htdemucs  (4 stems, local Demucs)",
  "htdemucs_ft  (4 stems, fine-tuned, local Demucs)",
  "mdx_extra  (4 stems, MDX-Net, local Demucs)",
  "htdemucs_6s  (6 stems, local Demucs)",
  "SAM Audio large  (cloud, prompt-based)",
  "SAM Audio base  (cloud, prompt-based)",
}
local SS_IS_SAM = { htdemucs=false, htdemucs_ft=false, htdemucs_6s=false,
                    mdx_extra=false, sam_large=true, sam_base=true }
local SS_SAM_MODEL_NAME = {
  sam_large = "facebook/sam-audio-large",
  sam_base  = "facebook/sam-audio-base",
}
local STEM_KEYS  = { "vocals", "drums", "bass", "other", "guitar", "piano" }
local STEM_NAMES = { vocals="Vocals", drums="Drums", bass="Bass",
                     other="Other", guitar="Guitar*", piano="Piano*" }
local SAM_GPUS   = { "A100-80GB", "A100", "H100", "A10G" }
local ODE_METHODS= { "midpoint", "euler", "rk4" }

-- SAM Audio takes a short CLAP-style instrument/sound descriptor, not a
-- full sentence (see modal_sam_audio.py's own examples: "jazz trumpet",
-- "vocals", "saxophone").
local SAM_EXAMPLE_PROMPTS = {
  { label = "vocals", text = "vocals" },
  { label = "jazz trumpet", text = "jazz trumpet" },
  { label = "saxophone", text = "saxophone" },
}

-- ── STATE ────────────────────────────────────────────────────────
local S = {
  model_idx      = 1,
  src            = "",
  src_track_name = "",
  src_track_idx  = -1,
  src_start_offs  = 0,    -- start offset into source file (seconds)
  src_section_dur = 0,    -- section duration in source time (seconds)
  src_item_pos    = nil,  -- item position in project timeline
  src_is_section  = false,
  outdir         = HOME .. "/stems",
  -- Demucs
  dm_stems       = { vocals=true, drums=true, bass=true, other=true,
                     guitar=false, piano=false },
  -- SAM
  sam_prompt     = "jazz trumpet",
  sam_prompt_example = { idx = 1 },
  sam_gidx       = 1,
  sam_oidx       = 1,
  sam_steps      = 64,
  sam_chunk      = 15.0,
  sam_overlap    = 2.0,
  sam_conf       = 0.0,
  sam_cands      = 1,
  -- runtime
  running        = false,
  pid            = nil,
  done           = false,
  progress       = 0.0,
  status         = "Ready.",
  log            = {},
  out_files      = {},
  log_scroll_to_bottom = false,
}

-- ── CORE HELPERS ─────────────────────────────────────────────────
local function add_log(s)
  table.insert(S.log, tostring(s):sub(1, 200))
  if #S.log > 200 then table.remove(S.log, 1) end
  S.log_scroll_to_bottom = true
end

local function q(s) return common.q(s) end

-- forward declarations (defined further below, referenced earlier)
local import_stems

-- ── PROGRESS ─────────────────────────────────────────────────────
local function read_progress()
  local r = common.read_progress_file(PROGRESS_F)
  if not r then return end

  local pct = r.pct or S.progress
  S.progress = pct
  if r.msg ~= S.status then
    S.status = r.msg
    add_log(r.msg)
  end

  if r.state == "done" and not S.done then
    S.running   = false
    S.pid       = nil
    S.done      = true
    S.out_files = {}
    for _, line in ipairs(r.extra) do
      local p = line:match("^%s*(.-)%s*$")
      if p ~= "" then table.insert(S.out_files, p) end
    end
    if #S.out_files > 0 then
      add_log("Files ready: " .. #S.out_files)
      import_stems()
    end
  elseif r.state == "error" and not S.done then
    S.running = false
    S.pid     = nil
    S.done    = true
    add_log("ERROR: " .. (r.msg or "?"))
    for _, line in ipairs(r.extra or {}) do
      local p = line:match("^%s*(.-)%s*$")
      if p ~= "" then add_log("  " .. p) end
    end
    add_log('See "Full log" button for the complete traceback.')
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
    reaper.MB("Item has no active take.", "Stem Separator", 0); return false
  end
  local src   = reaper.GetMediaItemTake_Source(take)
  local fname = reaper.GetMediaSourceFileName(src, "")
  if not fname or fname == "" then return false end

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
  -- Item/split takes priority: in REAPER selecting an item also selects
  -- its track, so we must check items before tracks.
  local n_items = reaper.CountSelectedMediaItems(0)
  if n_items > 0 then
    local item = reaper.GetSelectedMediaItem(0, 0)
    _set_src_from_item(item, "item")
    return
  end

  local tcnt = reaper.CountSelectedTracks(0)
  if tcnt > 0 then
    local tr = reaper.GetSelectedTrack(0, 0)
    local icnt = reaper.CountTrackMediaItems(tr)
    for i = 0, icnt - 1 do
      local item = reaper.GetTrackMediaItem(tr, i)
      if _set_src_from_item(item, "track") then return end
    end
    reaper.MB("Selected track has no active audio items.", "Stem Separator", 0)
    return
  end

  reaper.MB("No item or track selected in REAPER.", "Stem Separator", 0)
end

import_stems = function()
  if #S.out_files == 0 then return end
  reaper.Undo_BeginBlock()
  local cursor   = S.src_item_pos or reaper.GetCursorPosition()
  local imported = 0

  local folder_name = S.src_track_name
  if folder_name == "" then
    local base = S.src:match("([^/\\]+)%.%w+$")
    folder_name = (base or "stems") .. " [stems]"
  end

  -- Land the new folder right below the source track/clip instead of
  -- always at the very end of the project's track list.
  local tcnt = track_placement.insert_index(S.src_track_idx, reaper.CountTracks(0))
  reaper.InsertTrackAtIndex(tcnt, true)
  local folder_tr = reaper.GetTrack(0, tcnt)
  reaper.GetSetMediaTrackInfo_String(folder_tr, "P_NAME", folder_name, true)
  reaper.SetMediaTrackInfo_Value(folder_tr, "I_FOLDERDEPTH", 1)

  for _, fp in ipairs(S.out_files) do
    local f = io.open(fp, "rb")
    if f then
      f:close()
      -- Insert right after the folder (or the last stem added so far), NOT
      -- at reaper.CountTracks(0) — that would append at the end of the
      -- whole project instead of keeping every stem inside the folder.
      local tidx = tcnt + 1 + imported
      reaper.InsertTrackAtIndex(tidx, true)
      local track      = reaper.GetTrack(0, tidx)
      local stem_name  = fp:match("([^/\\]+)%.wav$") or fp:match("([^/\\]+)$")
      local track_name = folder_name .. " - " .. (stem_name or "stem")
      reaper.GetSetMediaTrackInfo_String(track, "P_NAME", track_name, true)
      reaper.SetOnlyTrackSelected(track)
      reaper.SetEditCurPos(cursor, false, false)
      reaper.InsertMedia(fp, 0)
      imported = imported + 1
      add_log("Imported: " .. track_name)
    else
      add_log("Not found: " .. fp)
    end
  end

  if imported > 0 then
    -- Last stem track is tcnt+imported (folder at tcnt, stems at
    -- tcnt+1..tcnt+imported) — NOT reaper.CountTracks(0)-1, which would be
    -- wrong whenever the folder was inserted mid-project rather than at
    -- the very end.
    local last_tr = reaper.GetTrack(0, tcnt + imported)
    reaper.SetMediaTrackInfo_Value(last_tr, "I_FOLDERDEPTH", -1)
    for i = 0, imported - 1 do
      local tr   = reaper.GetTrack(0, tcnt + 1 + i)
      local icnt = reaper.CountTrackMediaItems(tr)
      for j = 0, icnt - 1 do
        local item = reaper.GetTrackMediaItem(tr, j)
        local take = reaper.GetActiveTake(item)
        if take then
          local _, sname = reaper.GetSetMediaItemTakeInfo_String(take, "P_NAME", "", false)
          if sname == "" then
            local _, tname = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
            reaper.GetSetMediaItemTakeInfo_String(take, "P_NAME", tname, true)
          end
        end
      end
    end
  end

  if imported == 0 then reaper.DeleteTrack(folder_tr) end
  reaper.UpdateArrange()
  reaper.Undo_EndBlock("Stem Separator: import stems to folder", -1)
  add_log("Imported " .. imported .. " stems into '" .. folder_name .. "'")
end

-- ── LAUNCH PROCESSES ─────────────────────────────────────────────
local function clear_run(label)
  local f = io.open(PROGRESS_F, "w")
  if f then f:write("running|0.00|" .. label); f:close() end
  local lf = io.open(LOG_F, "w"); if lf then lf:close() end
  S.running   = true
  S.pid       = nil
  S.done      = false
  S.progress  = 0
  S.out_files = {}
  S.log       = {}
  S.status    = label
  S.log_scroll_to_bottom = false
end

-- Stops the in-flight separation (best-effort: signals the local process;
-- a remote Modal job already dispatched by SAM Audio may keep running).
local function stop_run()
  local signaled = common.stop_process(S.pid)
  add_log(signaled and "Stopped by user."
    or "Stopped by user (process already finished).")
  S.running = false
  S.pid     = nil
  S.done    = true
end

local function launch_demucs()
  if S.src == "" then
    reaper.MB("Select an audio file first.", "Stem Separator", 0); return
  end
  local stems = {}
  for _, k in ipairs(STEM_KEYS) do
    if S.dm_stems[k] then table.insert(stems, k) end
  end
  if #stems == 0 then
    reaper.MB("Select at least one stem.", "Stem Separator", 0); return
  end
  local model = SS_MODELS[S.model_idx]
  clear_run("Starting Demucs (" .. model .. ")...")
  add_log("Model: " .. model .. " | Stems: " .. table.concat(stems, ", "))
  local section_args = ""
  if S.src_is_section then
    section_args = string.format(" --start %.6f --duration %.6f",
      S.src_start_offs, S.src_section_dur)
    add_log(string.format("Section: %.2fs → %.2fs",
      S.src_start_offs, S.src_start_offs + S.src_section_dur))
  end
  local cmd = string.format(
    '%s %s --input %s --model %s --stems %s --outdir %s --python %s%s --progress %s >>%s 2>&1 &',
    q(PYTHON), q(DEMUCS_PY),
    q(S.src), q(model), table.concat(stems, ","),
    q(S.outdir), q(PYTHON), section_args, q(PROGRESS_F), q(LOG_F))
  add_log("Launching process...")
  S.pid = common.launch_tracked(cmd)
end

local function launch_sam()
  if S.src == "" then
    reaper.MB("Select an audio file first.", "Stem Separator", 0); return
  end
  if S.sam_prompt == "" then
    reaper.MB("Write a prompt for SAM Audio.", "Stem Separator", 0); return
  end
  local sam_script_path = SAM_DIR .. "/" .. SAM_SCRIPT
  local f = io.open(sam_script_path, "r")
  if not f then
    reaper.MB("Not found: " .. sam_script_path ..
      "\n\nThe plugin must contain modal_sam_audio.py and pyproject.toml.\n" ..
      "Check the StemsSeparator plugin installation.",
      "Stem Separator", 0)
    return
  end
  f:close()
  local sam_model_name = SS_SAM_MODEL_NAME[SS_MODELS[S.model_idx]]
  clear_run("Starting SAM Audio via Modal...")
  add_log("Prompt: " .. S.sam_prompt)
  add_log("Model: " .. sam_model_name .. " | GPU: " .. SAM_GPUS[S.sam_gidx])
  local section_args = ""
  if S.src_is_section then
    section_args = string.format(" --start %.6f --duration %.6f",
      S.src_start_offs, S.src_section_dur)
    add_log(string.format("Section: %.2fs → %.2fs",
      S.src_start_offs, S.src_start_offs + S.src_section_dur))
  end
  local cmd = string.format(
    '%s %s --sam-dir %s --shared-dir %s --input %s --prompt %s --model %s --gpu %s' ..
    ' --steps %d --ode-method %s --chunk %.1f --overlap %.1f' ..
    ' --confidence %.2f --candidates %d%s --outdir %s --progress %s >>%s 2>&1 &',
    q(PYTHON), q(SAM_PY), q(SAM_DIR), q(SHARED_DIR),
    q(S.src), q(S.sam_prompt),
    q(sam_model_name), q(SAM_GPUS[S.sam_gidx]),
    S.sam_steps, ODE_METHODS[S.sam_oidx],
    S.sam_chunk, S.sam_overlap, S.sam_conf, S.sam_cands,
    section_args, q(S.outdir), q(PROGRESS_F), q(LOG_F))
  add_log("Launching Modal process...")
  S.pid = common.launch_tracked(cmd)
end

-- ── DEMUCS PARAMETERS (shown when a local Demucs model is selected) ──
local function draw_demucs_params()
  local g = gui
  local t = theme

  g.text("Stems:")
  g.spacing()

  local is6s = SS_MODELS[S.model_idx] == "htdemucs_6s"
  local row1 = { "vocals", "drums", "bass", "other" }
  for i, k in ipairs(row1) do
    local cl, nv = g.checkbox(STEM_NAMES[k] .. "##" .. k, S.dm_stems[k])
    if cl then S.dm_stems[k] = nv end
    if i < #row1 then g.same_line(t.sc(22)) end
  end

  g.begin_disabled(not is6s)
  local cl, nv = g.checkbox(STEM_NAMES.guitar .. "##guitar", S.dm_stems.guitar)
  if cl then S.dm_stems.guitar = nv end
  g.same_line(t.sc(22))
  cl, nv = g.checkbox(STEM_NAMES.piano .. "##piano", S.dm_stems.piano)
  if cl then S.dm_stems.piano = nv end
  g.end_disabled()
  if not is6s then
    g.same_line(t.sc(8))
    g.text_disabled("(htdemucs_6s only)")
  end

  g.spacing()
  if g.button("All", t.sc(70), t.ITEM_H) then
    for _, k in ipairs(STEM_KEYS) do S.dm_stems[k] = true end
  end
  g.same_line()
  if g.button("None", t.sc(70), t.ITEM_H) then
    for _, k in ipairs(STEM_KEYS) do S.dm_stems[k] = false end
  end
end

-- ── SAM AUDIO PARAMETERS (shown when a SAM Audio model is selected) ──
local function draw_sam_params()
  local g = gui
  local t = theme
  local lw = t.sc(78)  -- label column width

  -- Prompt
  g.row_label("Prompt:", lw)
  local rv, nv = widgets.input_text("##prompt", S.sam_prompt)
  if rv then S.sam_prompt = nv end

  g.row_label("Examples:", lw)
  g.next_width(-1)
  local picked = widgets.example_prompt_picker("##sam_prompt_ex", S.sam_prompt_example, SAM_EXAMPLE_PROMPTS)
  if picked then S.sam_prompt = picked end

  -- GPU + ODE method
  g.row_label("GPU:", lw)
  g.next_width(t.sc(90))
  S.sam_gidx = widgets.combo("##sam_gpu", S.sam_gidx, SAM_GPUS)
  g.same_line(t.sc(14))
  g.inline_text("ODE:")
  g.same_line(t.sc(6))
  g.next_width(-1)
  S.sam_oidx = widgets.combo("##sam_ode", S.sam_oidx, ODE_METHODS)

  -- ODE steps + Confidence
  g.row_label("ODE steps:", lw)
  g.next_width(t.sc(120))
  rv, nv = g.slider_int("##steps", S.sam_steps, 1, 128)
  if rv then S.sam_steps = nv end
  g.same_line(t.sc(14))
  g.inline_text("Conf.:")
  g.same_line(t.sc(6))
  g.next_width(-1)
  rv, nv = g.slider_float("##conf", S.sam_conf, 0.0, 1.0, "%.2f")
  if rv then S.sam_conf = nv end

  -- Chunk + Overlap + Candidates
  g.row_label("Chunk s:", lw)
  g.next_width(t.sc(90))
  rv, nv = g.slider_float("##chunk", S.sam_chunk, 1.0, 30.0, "%.1f")
  if rv then S.sam_chunk = nv end
  g.same_line(t.sc(14))
  g.inline_text("Overlap:")
  g.same_line(t.sc(6))
  g.next_width(t.sc(80))
  rv, nv = g.slider_float("##overlap", S.sam_overlap, 0.0, 10.0, "%.1f")
  if rv then S.sam_overlap = nv end
  g.same_line(t.sc(14))
  g.inline_text("Cand.:")
  g.same_line(t.sc(6))
  g.next_width(-1)
  rv, nv = g.slider_int("##cands", S.sam_cands, 1, 8)
  if rv then S.sam_cands = nv end

  g.spacing()
  g.text_disabled("A100: ~$0.14/track  |  A10G: ~$0.09/track  |  Requires Modal.com account")
end

-- ── MODULE ───────────────────────────────────────────────────────
local M = { title = "Stems Separator" }

function M.init()
  if PYTHON_ERR then
    reaper.ShowConsoleMsg("Stem Separator - WARNING: " .. PYTHON_ERR .. "\n")
  end
  add_log("Stem Separator ready.")
  add_log("Python: " .. PYTHON)
  add_log("SAM dir: " .. SAM_DIR)
end

function M.poll()
  if S.running then read_progress() end
end

function M.draw()
  local g = gui
  local t = theme
  local is_sam = SS_IS_SAM[SS_MODELS[S.model_idx]]

  -- ── MODEL (fixed, above the scrollable content — same position across
  -- every tab in the app). Replaces the old DEMUCS/SAM AUDIO sub-tabs:
  -- the selected model now drives which parameter section is shown below. ──
  g.row_label("Model:", t.sc(68))
  g.next_width(-1)
  S.model_idx = widgets.combo("##ss_model", S.model_idx, SS_LABELS)
  g.spacing()

  -- ── SOURCE (fixed, directly below Model — same position across every
  -- tab that has a source file to pick) ────────────────────────────
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
    g.text_disabled("Click R to use the active Reaper track/item")
  end
  if S.src_is_section then
    g.text_colored(string.format("Section: %.2fs → %.2fs  (%.2fs)",
      S.src_start_offs, S.src_start_offs + S.src_section_dur, S.src_section_dur),
      "YELLOW")
  end
  g.spacing()
  g.separator(); g.spacing()

  -- ── SCROLL REGION: rest of the page (params/button/log) ───────────
  -- Single, non-nested scroll_region for everything below Model/Source.
  -- widgets_extra.lua's scroll_region does not support nesting (its clip/
  -- scroll math doesn't compound an outer scroll offset into an inner one),
  -- so the log below prints its lines directly into THIS region instead of
  -- opening its own nested scroll_region — new lines call
  -- widgets.scroll_to_bottom("##ss_page") to bring the log into view.
  local scroll_h = math.max(t.sc(60), gfx.h - gui.ctx.y - t.PAD_Y)
  widgets.scroll_region("##ss_page", 0, scroll_h, function()

  if is_sam then draw_sam_params()
  else           draw_demucs_params() end

  g.spacing()
  g.separator()
  g.spacing()

  -- SEPARATE button — colors change per model type; doubles as STOP while running
  local sep_colors
  if is_sam then
    sep_colors = {
      norm   = { 0x4D/255, 0x19/255, 0xC4/255 },
      hover  = { 0x66/255, 0x26/255, 0xE0/255 },
      active = { 0x80/255, 0x33/255, 0xD1/255 },
    }
  else
    sep_colors = {
      norm   = { 0x29/255, 0x66/255, 0xB0/255 },
      hover  = { 0x3D/255, 0x80/255, 0xD8/255 },
      active = { 0x47/255, 0x99/255, 0xFF/255 },
    }
  end
  local stop_colors = {
    norm   = t.C.RED,
    hover  = { 0xE8/255, 0x5A/255, 0x50/255 },
    active = { 0xF2/255, 0x70/255, 0x66/255 },
  }
  local sep_lbl = S.running and "STOP  (Processing...)"
    or (is_sam and "SEPARATE  (SAM Audio)" or "SEPARATE  (Demucs)")
  g.next_width(-1)
  if g.button(sep_lbl, nil, t.sc(36), { solid = S.running and stop_colors or sep_colors }) then
    if S.running then
      stop_run()
    elseif is_sam then launch_sam() else launch_demucs() end
  end
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

  -- Log area
  if widgets.collapsing_header("Logs", true) then
    if g.button("Copy log", t.sc(90), t.ITEM_H) then
      local ok, set_cb = pcall(function()
        reaper.CF_SetClipboard(table.concat(S.log, "\n"))
      end)
      if not ok then
        -- SWS/CF_ not available; print to console instead
        reaper.ShowConsoleMsg(table.concat(S.log, "\n") .. "\n")
      end
    end
    g.same_line()
    if g.button("Full log", t.sc(90), t.ITEM_H) then
      local f = io.open(LOG_F, "r")
      if f then
        local content = f:read("*a")
        f:close()
        local ok = pcall(function()
          reaper.CF_SetClipboard(content)
        end)
        reaper.ShowConsoleMsg("---- Stem Separator full log (" .. LOG_F .. ") ----\n")
        reaper.ShowConsoleMsg(content .. "\n")
        if ok then reaper.ShowConsoleMsg("(also copied to clipboard)\n") end
      else
        reaper.ShowConsoleMsg("Stem Separator: log file not found: " .. LOG_F .. "\n")
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

  end)  -- end scroll_region ##ss_page
end

return M
