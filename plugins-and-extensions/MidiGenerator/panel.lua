-- AI Music Lab - MidiGenerator panel (text/seed → MIDI generation)
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
local track_placement = require("track_placement")
local midi_import = require("midi_import")
local model_switch = require("model_switch")
local model_execution = require("model_execution")
local smf     = dofile(SCRIPT_DIR .. "smf_writer.lua")

local HOME   = common.HOME
local TMPDIR = common.TMPDIR
local PYTHON, PYTHON_ERR = common.detect_reaper_python()

-- ── CONSTANTS: MODELS ────────────────────────────────────────────
-- Only the models that held up in evaluation are exposed (see RESEARCH.md);
-- text2midi, MuseCoco, MIDI-LLM and ChatMusician were retired from the UI.
-- Anticipatory and MIDI-GPT take a MELODY (not a text prompt) and write the accompaniment.
local MG_MODELS = { "amadeus", "anticipatory", "midigpt" }
local MG_LABELS = {
  "Amadeus  (multi-track from a text description, A10G)",
  "Anticipatory  (accompaniment of your melody, A10G)",
  "MIDI-GPT  (accompaniment of your melody, A10G, non-commercial)",
}
local function takes_melody(mk) return mk == "anticipatory" or mk == "midigpt" end
local MG_SCRIPTS = {}
for _, k in ipairs(MG_MODELS) do
  MG_SCRIPTS[k] = SCRIPT_DIR .. "research/research_" .. k .. "_modal.py"
end

-- Plain (non-Modal) local scripts — only for the models that support
-- --execution local (see shared/lib/model_execution.lua's M.midigen table).
local MG_LOCAL_SCRIPTS = {
  anticipatory = SCRIPT_DIR .. "research/research_anticipatory.py",
}

local MG_GPUS        = { "A10G", "A100-40GB", "T4", "L4" }
local MG_GPU_DEFAULT = { amadeus="A10G", anticipatory="A10G", midigpt="A10G" }
-- Models where GPU is passed to the script (others ignore it)
local MG_GPU_RELEVANT = { anticipatory=true }

local MG_TEMP_DEFAULT = { amadeus=1.0 }
local MG_TEMP_RANGE_DEFAULT = { 0.5, 2.0 }

-- Style of the automatic bass + chords prompt Anticipatory continues from
-- (only used when no accompaniment tracks are added)
local AMT_STYLES = { "pop", "pad", "arp" }
local AMT_STYLE_LABELS = { "Pop (bass + chord stabs)", "Pad (bass + sustained chords)", "Arpeggio (bass + arpeggiated chords)" }

-- Accompaniment controls shared by Anticipatory and MIDI-GPT
local ACC_PARTS = { "bass", "keys", "strings", "guitar", "drums" }
local ACC_PART_LABELS = { bass = "Bass", keys = "Keys", strings = "Strings", guitar = "Guitar", drums = "Drums" }
local ACC_DENSITY_LABELS = { "Sparse", "Normal", "Busy" }   -- 0, 1, 2
local ACC_BASS_PROGRAMS = { 33, 32, 38, 35 }
local ACC_BASS_LABELS = { "Electric bass", "Acoustic bass", "Synth bass", "Fretless bass" }
local ACC_KEYS_PROGRAMS = { 4, 0, 16, 11 }
local ACC_KEYS_LABELS = { "Electric piano", "Piano", "Organ", "Vibraphone" }

-- Example prompts per text model, selectable from a dropdown next to the
-- prompt box. Written in MidiCaps style, the dialect Amadeus expects (see
-- evaluation/amadeus/README.md and research_amadeus_modal.py).
local MG_EXAMPLE_PROMPTS = {
  amadeus = {
    { label = "Electronic ambient, E major, tubular bells...",
      text = "A melodic electronic ambient song with a touch of darkness, set in the key of E major and a 4/4 time signature. Tubular bells, electric guitar, synth effects, synth pad, and oboe weave together to create an epic, space-like atmosphere. The tempo is a steady Andante, and the chord progression of A, B, and E forms the harmonic backbone of this captivating piece." },
    { label = "Jazz fusion, F minor, 120 BPM, piano/bass",
      text = "An upbeat jazz fusion track in the key of F minor at 120 BPM, featuring piano, upright bass, and brushed drums. The chord progression Fm-Db-Ab-Eb repeats throughout, building a relaxed but energetic groove." },
    { label = "Cinematic orchestral, D minor, 90 BPM",
      text = "A cinematic orchestral piece in D minor, 90 BPM, 3/4 time signature. Strings, French horn, and timpani build a tense, dramatic atmosphere over a Dm-Bb-F-C chord progression." },
  },
}

local MIDIGEN_PY = SCRIPT_DIR .. "midigen.py"
local PROGRESS_F = TMPDIR .. "midigen_progress.txt"
local LOG_F      = TMPDIR .. "midigen.log"

-- ── STATE ────────────────────────────────────────────────────────
local S = {
  -- Model
  model_idx = 1,
  gpu       = "A10G",
  -- Prompt (text models)
  prompt    = "",
  prompt_example = { idx = 1 },
  -- Optional MidiCaps fields (amadeus)
  field_key         = "",
  field_instruments = "",
  field_chords      = "",
  -- AMT parameters
  amt_style_idx    = 1,   -- index in AMT_STYLES
  amt_prompt_len   = 5,   -- seconds of accompaniment the model continues from
  acc_parts        = { bass = true, keys = true, strings = false, guitar = false, drums = false },
  acc_density_idx  = 2,   -- 1 sparse, 2 normal, 3 busy
  acc_bass_idx     = 1,
  acc_keys_idx     = 1,
  acc_creativity   = 0.5,
  acc_chords       = "",  -- optional own progression (Anticipatory)
  acc_beats        = 4,   -- beats per chord when acc_chords is given
  acc_stay_in_key  = false,  -- MIDI-GPT: only the melody's pitch classes
  -- AMT: melody track
  amt_melody_item  = nil,   -- REAPER item (for position/length on import)
  amt_melody_take  = nil,
  amt_melody_label = "",
  -- AMT: accompaniment seed tracks
  amt_seed_takes   = {},  -- list of {take=take, label="track name"}
  -- Execution mode: "remote" (Modal) or "local" (this machine). Forced back
  -- to "remote" whenever the current model doesn't support local (see
  -- shared/lib/model_execution.lua's M.midigen registry).
  execution = "remote",
  -- Common parameters
  n_outputs   = 2,
  temperature = MG_TEMP_DEFAULT[MG_MODELS[1]] or 1.0,
  -- Runtime
  running    = false,
  pid        = nil,
  done       = false,
  progress   = 0.0,
  status     = "Ready.",
  log        = {},
  out_files  = {},
  n_instruments = -1,
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
  local d = TMPDIR .. "midigen_run" .. _run_id .. "/"
  os.execute("mkdir -p " .. q(d))
  -- Clean up previous runs (but not the current one) so they don't pollute _collect_outputs
  os.execute(string.format(
    "find %s -maxdepth 1 -type d -name 'midigen_run*' ! -name 'midigen_run%d' -exec rm -rf {} + 2>/dev/null || true",
    q(TMPDIR), _run_id))
  return d
end

local function model_key() return MG_MODELS[S.model_idx] end

-- ── SMF WRITER (export in-project MIDI takes) ────────────────────
-- See smf_writer.lua for write_combined_midi.
local write_combined_midi  = smf.write_combined_midi

-- forward declaration (defined further below, referenced earlier)
local import_midi_all

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
    S.running = false; S.pid = nil; S.done = true
    S.out_files = {}; S.n_instruments = -1
    for _, line in ipairs(r.extra) do
      local p = line:match("^%s*(.-)%s*$")
      if p ~= "" then
        local n = p:match("^INSTRUMENTS|(.+)$")
        if n then S.n_instruments = tonumber(n) or -1
        else table.insert(S.out_files, p) end
      end
    end
    if #S.out_files > 0 then
      add_log(string.format("MIDI ready (%s instrument%s, %d candidate%s)",
        S.n_instruments >= 0 and tostring(S.n_instruments) or "?",
        S.n_instruments ~= 1 and "s" or "",
        #S.out_files, #S.out_files ~= 1 and "s" or ""))
      import_midi_all()
    end
  elseif r.state == "error" and not S.done then
    S.running = false; S.pid = nil; S.done = true
    add_log("ERROR: " .. (r.msg or "?"))
    -- Trailing raw output from the Modal/Python subprocess (see midigen.py's
    -- tail_lines) — the single summarized message above is a best-effort
    -- guess at the cause; these extra lines give the actual context (model
    -- response preview, per-candidate outcomes) so the user isn't forced to
    -- go dig through midigen.log for detail.
    if r.extra and #r.extra > 0 then
      add_log("--- last output lines ---")
      for _, line in ipairs(r.extra) do
        local l = line:match("^%s*(.-)%s*$")
        if l ~= "" then add_log("  | " .. l) end
      end
    end
  end
end

-- ── IMPORT MIDI (multi-candidate) ────────────────────────────────
-- Each candidate (.mid) is imported into its own track folder.
-- Reuses Audio2Midi's I_FOLDERDEPTH folder logic.

-- Imports one candidate .mid into its own folder, anchored at `insert_at`
-- (a 0-based track index — see track_placement.lua) instead of always at
-- reaper.CountTracks(0). Returns the insert_at index the NEXT candidate
-- should use, so multiple candidates stack below each other right under
-- the source instead of all landing at the end of the project.
--
-- Tracks and clips are created explicitly (lib/midi_import.lua) rather than
-- through reaper.InsertMedia, so existing tracks and MIDI clips are never
-- wrapped in the new folder, moved or overwritten.
local function _import_one(mid_path, folder_name, insert_at)
  local f = io.open(mid_path, "rb")
  if not f then add_log("Error: cannot read " .. mid_path); return insert_at end
  f:close()

  local res, err = midi_import.import_file(mid_path, {
    insert_at = insert_at, position = reaper.GetCursorPosition(),
    name = folder_name, always_folder = true,
  })
  if not res then
    add_log("Warning: could not import " .. mid_path:match("[^/\\]+$") .. ": " .. tostring(err))
    return insert_at
  end
  add_log(string.format("Imported into folder '%s' (%d track%s)",
    folder_name, res.instruments, res.instruments == 1 and "" or "s"))
  return res.next_insert_at
end

-- Reference track number (1-based IP_TRACKNUMBER) for AMT generations that
-- are based on other tracks (melody item + seed takes) — nil when there is
-- no such basis (e.g. text-only models like amadeus), in which
-- case track_placement.insert_index falls back to appending at the end.
local function _amt_ref_track_number()
  local nums = {}
  if S.amt_melody_item then
    local tr = reaper.GetMediaItemTrack(S.amt_melody_item)
    if tr then table.insert(nums, reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER")) end
  end
  for _, seed in ipairs(S.amt_seed_takes) do
    local item = seed.take and reaper.GetMediaItemTake_Item(seed.take)
    local tr   = item and reaper.GetMediaItemTrack(item)
    if tr then table.insert(nums, reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER")) end
  end
  return track_placement.max_track_number(nums)
end

import_midi_all = function()
  if #S.out_files == 0 then return end
  local mk    = model_key()
  local label = "[MIDI " .. mk .. "]"

  -- The accompaniment is timed relative to the start of the melody item, so
  -- import it at that position (not wherever the edit cursor happens to be).
  if takes_melody(mk) and S.amt_melody_item then
    local pos = reaper.GetMediaItemInfo_Value(S.amt_melody_item, "D_POSITION")
    reaper.SetEditCurPos(pos, false, false)
    add_log(string.format("Cursor → %.2fs (start of the melody item)", pos))
  end

  -- Land the generated folder(s) right below the melody/seed tracks this
  -- generation is based on (AMT), instead of always at the very end of the
  -- project's track list. Each extra candidate then stacks right below the
  -- previous candidate's folder.
  local ref_track = takes_melody(mk) and _amt_ref_track_number() or nil
  local insert_at = track_placement.insert_index(ref_track, reaper.CountTracks(0))

  reaper.Undo_BeginBlock()
  for i, path in ipairs(S.out_files) do
    local suffix = #S.out_files > 1 and (" — candidate " .. i) or ""
    insert_at = _import_one(path, label .. suffix, insert_at)
  end
  reaper.UpdateArrange()
  reaper.Undo_EndBlock("MidiGenerator: import MIDI", -1)
end

local function _get_track_label(item)
  local tr = reaper.GetMediaItemTrack(item)
  if not tr then return "?" end
  local _, tname = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
  local tnum = reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER")
  return (tname ~= "" and tname) or ("Track " .. math.floor(tnum))
end

local function grab_melody_item()
  local item = nil
  local n = reaper.CountSelectedMediaItems(0)
  if n > 0 then
    item = reaper.GetSelectedMediaItem(0, 0)
  else
    local tcnt = reaper.CountSelectedTracks(0)
    if tcnt > 0 then
      local tr = reaper.GetSelectedTrack(0, 0)
      for i = 0, reaper.CountTrackMediaItems(tr) - 1 do
        local it = reaper.GetTrackMediaItem(tr, i)
        if it then item = it; break end
      end
    end
  end
  if not item then
    reaper.MB("No MIDI item selected in REAPER.", "MidiGenerator", 0)
    return
  end
  local take = reaper.GetActiveTake(item)
  if not take or not reaper.TakeIsMIDI(take) then
    reaper.MB("Selected item does not contain MIDI.", "MidiGenerator", 0)
    return
  end
  S.amt_melody_item  = item
  S.amt_melody_take  = take
  S.amt_melody_label = _get_track_label(item)
  add_log("Melody captured: " .. S.amt_melody_label)
end

local function add_seed_items_from_reaper()
  local added = 0
  for i = 0, reaper.CountSelectedMediaItems(0) - 1 do
    local item = reaper.GetSelectedMediaItem(0, i)
    local take = reaper.GetActiveTake(item)
    if take and reaper.TakeIsMIDI(take) then
      local label = _get_track_label(item)
      table.insert(S.amt_seed_takes, { take=take, label=label })
      added = added + 1
    end
  end
  if added == 0 then
    reaper.MB("No MIDI items selected in REAPER.", "MidiGenerator", 0)
  else
    add_log("Added " .. added .. " seed track(s).")
  end
end

-- ── LAUNCH GENERATION ────────────────────────────────────────────
local function clear_run(label)
  local f = io.open(PROGRESS_F, "w")
  if f then f:write("running|0.00|" .. label); f:close() end
  local lf = io.open(LOG_F, "w"); if lf then lf:close() end
  S.running = true; S.pid = nil; S.done = false
  S.progress = 0; S.out_files = {}
  S.log = {}; S.status = label
  S.log_scroll_to_bottom = false
end

-- Stops the in-flight generation (best-effort: signals the local Modal
-- client process; a remote Modal job already dispatched may keep running).
local function stop_generate()
  local signaled = common.stop_process(S.pid)
  add_log(signaled and "Stopped by user."
    or "Stopped by user (process already finished).")
  S.running = false
  S.pid     = nil
  S.done    = true
end

local function launch_generate()
  local mk       = model_key()
  local script   = (S.execution == "local") and MG_LOCAL_SCRIPTS[mk] or MG_SCRIPTS[mk]
  local is_amt   = takes_melody(mk)
  local is_text  = not is_amt
  local has_fields = (mk == "amadeus")

  -- Validations
  if is_text and S.prompt:match("^%s*$") then
    reaper.MB("Write a text prompt before generating.", "MidiGenerator", 0)
    return
  end
  if is_amt and not S.amt_melody_take then
    reaper.MB(
      "This model needs your melody.\n"
      .. "Select a MIDI item in REAPER and click R next to 'Melody'.",
      "MidiGenerator", 0)
    return
  end
  local chosen_parts = {}
  for _, part in ipairs(ACC_PARTS) do
    if S.acc_parts[part] then table.insert(chosen_parts, part) end
  end
  if is_amt and #chosen_parts == 0 then
    reaper.MB("Choose at least one part (bass, keys, strings, guitar or drums).", "MidiGenerator", 0)
    return
  end

  local f = io.open(script, "r")
  if not f then
    reaper.MB("Script not found:\n" .. script, "MidiGenerator", 0); return
  end
  f:close()

  local label = "Starting " .. (MG_LABELS[S.model_idx] or mk) .. "..."
  clear_run(label)
  add_log("Model: " .. (MG_LABELS[S.model_idx] or mk))
  add_log("Execution: " .. (S.execution == "local" and "Local (this machine)" or "Remote (Modal)"))
  if S.execution ~= "local" then add_log("GPU: " .. S.gpu) end

  local run_dir = make_run_dir()

  -- Build base command (not yet backgrounded). --gpu is omitted for local
  -- execution: the plain research scripts don't accept it and auto-detect
  -- their own device (MPS/CPU).
  local base = string.format(
    '%s %s --shared-dir %s --script %s --model %s'
    .. ' --out-dir %s --execution %s --n-outputs %d --progress %s',
    q(PYTHON), q(MIDIGEN_PY),
    q(SHARED_DIR), q(script), q(mk),
    q(run_dir), q(S.execution), S.n_outputs,
    q(PROGRESS_F))
  if S.execution ~= "local" then
    base = base .. " --gpu " .. q(S.gpu)
  end

  local extra = ""
  if is_text then
    add_log("Prompt: " .. S.prompt:sub(1,80))
    extra = extra .. " --prompt " .. q(S.prompt)
    extra = extra .. string.format(" --temperature %.2f", S.temperature)
    -- BPM always from project for amadeus (not user-editable)
    if has_fields then
      extra = extra .. " --field-bpm " .. q(string.format("%.0f", reaper.Master_GetTempo()))
    end
    if S.field_key         ~= "" then extra = extra .. " --field-key "         .. q(S.field_key)         end
    if S.field_instruments ~= "" then extra = extra .. " --field-instruments " .. q(S.field_instruments) end
    if S.field_chords      ~= "" then extra = extra .. " --field-chords "      .. q(S.field_chords)      end
  else
    -- AMT: build one MIDI with the melody on channel 0 and the optional
    -- accompaniment tracks on channels 1..N. Written to TMPDIR (outside
    -- run_dir) so _collect_outputs doesn't pick it up.
    local melody_path = TMPDIR .. "combined_melody.mid"
    local ok_s, err_s = write_combined_midi(S.amt_melody_take, S.amt_seed_takes, melody_path)
    if not ok_s then
      reaper.MB("Error building the melody MIDI: " .. (err_s or "?"), "MidiGenerator", 0)
      S.running = false; S.pid = nil
      return
    end
    add_log("Melody: " .. S.amt_melody_label)
    add_log(string.format("Parts: %s | density: %s | creativity: %.2f",
      table.concat(chosen_parts, ", "), ACC_DENSITY_LABELS[S.acc_density_idx], S.acc_creativity))
    if mk == "anticipatory" then
      if #S.amt_seed_takes > 0 then
        add_log("Accompaniment tracks given: " .. #S.amt_seed_takes)
      else
        add_log("No accompaniment given: building a " .. AMT_STYLES[S.amt_style_idx] .. " prompt from the melody")
      end
    end
    extra = extra .. " --melody "            .. q(melody_path)
    extra = extra .. " --style "             .. q(AMT_STYLES[S.amt_style_idx])
    extra = extra .. " --parts "             .. q(table.concat(chosen_parts, ","))
    extra = extra .. " --density "           .. tostring(S.acc_density_idx - 1)
    extra = extra .. " --bass-program "      .. tostring(ACC_BASS_PROGRAMS[S.acc_bass_idx])
    extra = extra .. " --keys-program "      .. tostring(ACC_KEYS_PROGRAMS[S.acc_keys_idx])
    extra = extra .. string.format(" --creativity %.3f", S.acc_creativity)
    extra = extra .. " --clip-length 0"     -- 0 = length of the melody
    extra = extra .. " --melody-instrument 0"
    if mk == "anticipatory" then
      extra = extra .. " --prompt-length " .. tostring(S.amt_prompt_len)
      if S.acc_chords:match("%S") then
        extra = extra .. " --chords " .. q(S.acc_chords)
                      .. " --beats-per-chord " .. tostring(S.acc_beats)
      end
    elseif S.acc_stay_in_key then
      extra = extra .. " --stay-in-key"
    end
    -- --n-outputs is already in base; midigen.py maps it internally to --multiplicity
  end

  local cmd = base .. extra .. " >>" .. q(LOG_F) .. " 2>&1 &"
  add_log("Launching Modal...")
  S.pid = common.launch_tracked(cmd)
end

-- ── MODULE ───────────────────────────────────────────────────────
local M = { title = "MidiGenerator" }

function M.init()
  if PYTHON_ERR then
    reaper.ShowConsoleMsg("MidiGenerator - WARNING: " .. PYTHON_ERR .. "\n")
  end
  add_log("MidiGenerator ready.")
  add_log("Python: " .. PYTHON)
  add_log("midigen.py: " .. MIDIGEN_PY)
end

function M.poll()
  if S.running then read_progress() end
end

function M.draw()
  local g = gui
  local t = theme
  local mk = model_key()
  local is_amt  = takes_melody(mk)
  local is_text = not is_amt
  local has_fields = (mk == "amadeus")

  -- ── MODEL (fixed, above the scrollable content — same position across
  -- every tab in the app) ──────────────────────────────────────────
  g.row_label("Model:", t.sc(70))
  g.next_width(-1)
  local old_idx = S.model_idx
  S.model_idx = widgets.combo("##mg_model", S.model_idx, MG_LABELS)
  if model_switch.changed(old_idx, S.model_idx) then
    local new_mk = MG_MODELS[S.model_idx]
    S.gpu = model_switch.default_for(MG_GPU_DEFAULT, new_mk, "A10G")
    S.temperature = model_switch.default_for(MG_TEMP_DEFAULT, new_mk, 1.0)
    -- Don't leave a stale "local" selection on a model that doesn't support it.
    local supported_now, _ = model_execution.is_local_supported("midigen", new_mk)
    if not supported_now then S.execution = "remote" end
  end
  -- The prompt (and its model-specific phrasing advice) doesn't carry over
  -- meaningfully to a different model — e.g. amadeus's descriptive-sentence
  -- style doesn't apply to Anticipatory, which takes no text prompt.
  S.prompt = model_switch.next_prompt(old_idx, S.model_idx, S.prompt)

  -- ── EXECUTION: LOCAL vs REMOTE ────────────────────────────────────
  local exec_supported, exec_reason = model_execution.is_local_supported("midigen", mk)
  if not exec_supported then S.execution = "remote" end
  g.row_label("Run on:", t.sc(70))
  g.next_width(-1)
  if not exec_supported then g.begin_disabled(true) end
  local EXEC_OPTIONS = { "Remote (Modal)", "Local (this machine)" }
  local exec_idx = (S.execution == "local") and 2 or 1
  local new_exec_idx = widgets.combo("##mg_execution", exec_idx, EXEC_OPTIONS)
  if exec_supported then
    S.execution = (new_exec_idx == 2) and "local" or "remote"
  end
  if not exec_supported then
    g.end_disabled()
    g.text_disabled("  Local unavailable: " .. (exec_reason or "not supported"):sub(1, 110))
  end

  g.spacing()
  g.separator(); g.spacing()

  -- ── SCROLL REGION: rest of the page (prompt/params/button/log) ───
  -- Single, non-nested scroll_region for everything below Model.
  -- widgets_extra.lua's scroll_region does not support nesting (its clip/
  -- scroll math doesn't compound an outer scroll offset into an inner one),
  -- so the log below prints its lines directly into THIS region instead of
  -- opening its own nested scroll_region — new lines call
  -- widgets.scroll_to_bottom("##mg_page") to bring the log into view.
  local scroll_h = math.max(t.sc(60), gfx.h - gui.ctx.y - t.PAD_Y)
  widgets.scroll_region("##mg_page", 0, scroll_h, function()

  -- ── INPUT: PROMPT OR SEED ───────────────────────────────────────
  if is_text then
    -- ── Prompt ────────────────────────────────────────────────────
    g.text("Prompt:")
    g.next_width(-1)
    local rv, nv = widgets.input_textarea("##mg_prompt", S.prompt, 3,
      { placeholder = "Describe the musical style, instruments, mood..." })
    if rv then S.prompt = nv end

    local mg_examples = MG_EXAMPLE_PROMPTS[mk]
    if mg_examples then
      g.next_width(-1)
      local picked = widgets.example_prompt_picker("##mg_prompt_ex", S.prompt_example, mg_examples)
      if picked then S.prompt = picked end
    end

    -- Informational note per model
    if mk == "amadeus" then
      g.text_disabled("  Fill in the optional fields for more predictable results.")
    end

    -- ── Optional MidiCaps fields ───────────────────────────────────
    if has_fields then
      g.spacing()
      if widgets.collapsing_header("Optional fields (key / BPM / instruments / chords)", false) then
        local lw = t.sc(90)
        g.row_label("Key:", lw)
        g.next_width(t.sc(120))
        local r1, v1 = widgets.input_text("##mg_key", S.field_key)
        if r1 then S.field_key = v1 end
        g.same_line(t.sc(14)); g.inline_text("BPM:")
        g.same_line(t.sc(6))
        g.text_colored(string.format("%.1f", reaper.Master_GetTempo()), "GREEN")
        g.same_line(t.sc(4)); g.text_disabled("(project)")

        g.row_label("Instruments:", lw)
        g.next_width(-1)
        local r3, v3 = widgets.input_text("##mg_instr", S.field_instruments)
        if r3 then S.field_instruments = v3 end

        g.row_label("Chords:", lw)
        g.next_width(-1)
        local r4, v4 = widgets.input_text("##mg_chords", S.field_chords)
        if r4 then S.field_chords = v4 end

        g.text_disabled("  Example: F minor | 120 | piano and bass | Fm-Db-Ab-Eb")
      end
    end

  else
    -- ── AMT: melody + accompaniment seed ──────────────────────────
    local lw_amt = t.sc(90)

    -- Melody track
    g.row_label("Melody:", lw_amt)
    g.next_width(-(t.SPACING_X + t.sc(44)))
    widgets.input_text("##amt_mel_disp",
      S.amt_melody_label ~= "" and S.amt_melody_label or "(none — select a MIDI item)",
      { readonly=true })
    g.same_line()
    if g.button("R", t.sc(44), t.ITEM_H) then grab_melody_item() end
    g.text_disabled("  Select the melody item in REAPER and click R.")

    -- Optional accompaniment tracks (Anticipatory only: they steer the style)
    if mk == "anticipatory" then
      g.spacing()
      g.row_label("Accomp.:", lw_amt)
      if g.button("+", t.sc(44), t.ITEM_H) then add_seed_items_from_reaper() end
      g.same_line(t.sc(8))
      if g.button("Clear", t.sc(70), t.ITEM_H) then
        S.amt_seed_takes = {}
        add_log("Accompaniment tracks cleared.")
      end
      if #S.amt_seed_takes == 0 then
        g.text_disabled("  (optional) Select tracks in REAPER and click '+' to steer the style.")
      else
        for _, entry in ipairs(S.amt_seed_takes) do
          g.text_disabled("  • " .. entry.label)
        end
      end
    end

    -- ── ARRANGEMENT: what is written, how much, how it sounds ─────────
    g.spacing()
    g.row_label("Parts:", lw_amt)
    for i, part in ipairs(ACC_PARTS) do
      if i > 1 then g.same_line(t.sc(6)) end
      local chg, nv = g.checkbox(ACC_PART_LABELS[part] .. "##acc_" .. part, S.acc_parts[part])
      if chg then S.acc_parts[part] = nv end
    end

    local has_own_tracks = (mk == "anticipatory" and #S.amt_seed_takes > 0)
    g.row_label("Style:", lw_amt)
    g.next_width(t.sc(250))
    if has_own_tracks then g.begin_disabled(true) end
    S.amt_style_idx = widgets.combo("##amt_style", S.amt_style_idx, AMT_STYLE_LABELS)
    if has_own_tracks then g.end_disabled() end
    g.same_line(t.sc(10)); g.inline_text("Density:")
    g.same_line(t.sc(6))
    g.next_width(t.sc(110))
    S.acc_density_idx = widgets.combo("##acc_density", S.acc_density_idx, ACC_DENSITY_LABELS)

    g.row_label("Bass:", lw_amt)
    g.next_width(t.sc(150))
    S.acc_bass_idx = widgets.combo("##acc_bass", S.acc_bass_idx, ACC_BASS_LABELS)
    g.same_line(t.sc(10)); g.inline_text("Keys:")
    g.same_line(t.sc(6))
    g.next_width(t.sc(150))
    S.acc_keys_idx = widgets.combo("##acc_keys", S.acc_keys_idx, ACC_KEYS_LABELS)

    g.row_label("Creativity:", lw_amt)
    g.next_width(-1)
    local cr, cv = g.slider_float("##acc_creativity", S.acc_creativity, 0.0, 1.0, "%.2f")
    if cr then S.acc_creativity = cv end
    if mk == "anticipatory" then
      g.text_disabled("  Higher: the model adds instruments and variations of its own.")

      g.spacing()
      g.row_label("Chords:", lw_amt)
      g.next_width(-1)
      local r4, v4 = widgets.input_text("##acc_chords", S.acc_chords)
      if r4 then S.acc_chords = v4 end
      g.text_disabled("  Optional, e.g.  C#m G# A# F   (empty = detected from your melody)")
      if S.acc_chords:match("%S") then
        g.row_label("Beats/chord:", lw_amt)
        g.next_width(t.sc(120))
        local rb, vb = g.slider_int("##acc_beats", S.acc_beats, 1, 8)
        if rb then S.acc_beats = vb end
      end

      g.row_label("Prompt (s):", lw_amt)
      g.next_width(t.sc(120))
      local rv, nv = g.slider_int("##amt_plen", S.amt_prompt_len, 2, 15)
      if rv then S.amt_prompt_len = nv end
      g.same_line(t.sc(14))
      g.text_disabled("Length follows the melody.")
    else
      g.text_disabled("  Higher: more varied notes.")
      local sk, sv = g.checkbox("Stay in the melody's notes##acc_key", S.acc_stay_in_key)
      if sk then S.acc_stay_in_key = sv end
      g.text_colored("  Non-commercial weights: accept the license at huggingface.co/Metacreation/MIDI-GPT", "YELLOW")
    end
  end

  g.spacing(); g.separator(); g.spacing()

  -- ── COMMON PARAMETERS ───────────────────────────────────────────
  local lw2 = t.sc(90)
  g.row_label("Candidates:", lw2)
  g.next_width(t.sc(90))
  local rv, nv = g.slider_int("##mg_nout", S.n_outputs, 1, 4)
  if rv then S.n_outputs = nv end

  if is_text then
    g.same_line(t.sc(18)); g.inline_text("Temperature:")
    g.same_line(t.sc(6))
    g.next_width(-1)
    local temp_min, temp_max = table.unpack(MG_TEMP_RANGE_DEFAULT)
    rv, nv = g.slider_float("##mg_temp", S.temperature, temp_min, temp_max, "%.2f")
    if rv then S.temperature = nv end
  end

  -- GPU (always visible, but noted if not applicable)
  g.row_label("GPU:", lw2)
  local gpu_relevant = MG_GPU_RELEVANT[mk]
  if not gpu_relevant then g.begin_disabled(true) end
  g.next_width(t.sc(90))
  local gpu_list = MG_GPUS
  -- Find current index
  local gpu_idx = 1
  for i, v in ipairs(gpu_list) do if v == S.gpu then gpu_idx = i; break end end
  local new_gpu_idx = widgets.combo("##mg_gpu", gpu_idx, gpu_list)
  if new_gpu_idx ~= gpu_idx then S.gpu = gpu_list[new_gpu_idx] end
  if not gpu_relevant then
    g.end_disabled()
    g.same_line(t.sc(10))
    g.text_disabled("(fixed by model)")
  end

  -- Cost/warning hint
  g.spacing()
  local hints = {
    amadeus      = "A10G: ~$0.05/min  |  multi-track, variable duration",
    anticipatory = "A10G: ~$0.05/min (remote)  |  Local runs on CPU: slower but free",
    midigpt      = "A10G: ~$0.05/min  |  needs the Hugging Face token in Setup (gated weights)",
  }
  g.text_disabled(hints[mk] or "")

  g.spacing(); g.separator(); g.spacing()

  -- ── GENERATE BUTTON ─────────────────────────────────────────────
  -- (doubles as STOP while a generation is running)
  local btn_color = {
    norm   = { 0x1A/255, 0x7A/255, 0x3C/255 },
    hover  = { 0x22/255, 0x99/255, 0x4D/255 },
    active = { 0x2A/255, 0xB5/255, 0x5C/255 },
  }
  local stop_color = {
    norm   = t.C.RED,
    hover  = { 0xE8/255, 0x5A/255, 0x50/255 },
    active = { 0xF2/255, 0x70/255, 0x66/255 },
  }
  local btn_lbl = S.running and "STOP  (Generating...)" or "GENERATE MIDI"
  g.next_width(-1)
  if g.button(btn_lbl, nil, t.sc(36), { solid = S.running and stop_color or btn_color }) then
    if S.running then stop_generate() else launch_generate() end
  end
  g.spacing()

  -- ── PROGRESS ────────────────────────────────────────────────────
  local pct_str = string.format("%d%%", math.floor(S.progress * 100))
  g.progress_bar(S.progress, nil, t.sc(16), pct_str)

  local status_color = S.running and "YELLOW"
    or (S.done and #S.out_files > 0 and "GREEN")
    or (S.done and "RED")
    or "FG_DIM"
  g.text_colored(S.status:sub(1, 90), status_color)
  g.spacing()

  -- ── LOG ─────────────────────────────────────────────────────────
  if widgets.collapsing_header("Logs", true) then
    if g.button("Copy log", t.sc(90), t.ITEM_H) then
      local ok2, _ = pcall(function()
        reaper.CF_SetClipboard(table.concat(S.log, "\n"))
      end)
      if not ok2 then reaper.ShowConsoleMsg(table.concat(S.log, "\n") .. "\n") end
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

  end)  -- end scroll_region ##mg_page
end

return M
