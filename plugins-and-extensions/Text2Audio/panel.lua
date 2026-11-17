-- AI Music Lab - AudioGenerator panel (text → audio generation and editing)
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
local prepared = require("prepared")
local track_placement = require("track_placement")
local model_switch = require("model_switch")
local model_execution = require("model_execution")

local HOME   = common.HOME
local TMPDIR = common.TMPDIR
local PYTHON, PYTHON_ERR = common.detect_reaper_python()

-- ── CONSTANTS ────────────────────────────────────────────────────
-- Generation models (text → audio, no source)
local GEN_MODELS  = { "acestep_gen", "sao", "musicgen_gen", "audiogen" }
local GEN_LABELS  = {
  "ACE-Step 1.5  (A10G, full-song, Apache 2.0)",
  "Stable Audio Open 1.0  (A10G, 44.1 kHz stereo)",
  "MusicGen-medium  (A10G, 32 kHz, CC-BY-NC)",
  "AudioGen-medium  (A10G, 16 kHz, effects/sound)",
}
local GEN_SCRIPTS = {
  acestep_gen    = SCRIPT_DIR .. "research/research_acestep_gen_modal.py",
  sao            = SCRIPT_DIR .. "research/research_stable_audio_open_modal.py",
  musicgen_gen   = SCRIPT_DIR .. "research/research_musicgen_gen_modal.py",
  audiogen       = SCRIPT_DIR .. "research/research_audiogen_modal.py",
}
local GEN_MAX_SEC = {
  acestep_gen = 180.0, sao = 47.0, musicgen_gen = 30.0, audiogen = 30.0,
}

-- Edit models (source audio + prompt → transformed audio)
-- Ordered by real-world popularity of the underlying base model (verified via GitHub
-- stars / official announcements); AudioLDM2's own repo has only ~6★
local EDIT_MODELS  = { "sao_edit", "acestep" }
local EDIT_LABELS  = {
  "SAO Style Transfer  (A10G, SDEdit init_audio)",
  "ACE-Step 1.5  (A10G, cover/re-style, Apache 2.0)",
}
local EDIT_SCRIPTS = {
  sao_edit    = SCRIPT_DIR .. "research/research_sao_edit_modal.py",
  acestep     = SCRIPT_DIR .. "research/research_acestep_edit_modal.py",
}

local GPUS        = { "A10G", "A100", "A100-80GB", "T4" }
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
local ACESTEP_VARIANTS        = { "turbo", "sft", "base", "xl_turbo", "xl_sft", "xl_base" }
local ACESTEP_VARIANT_LABELS  = {
  "Turbo  (fast, 8-20 steps, no CFG)",
  "SFT  (best quality, 100 steps, CFG — slower, larger download)",
  "Base  (32-200 steps, CFG — requires a LoRA adapter)",
  "XL Turbo  (4B DiT, 8 steps — fast; runs on A10G)",
  "XL SFT  (4B DiT, 50 steps — a bit more quality by ear; runs on A10G)",
  "XL Base  (4B DiT — requires a LoRA adapter)",
}
-- XL checkpoints are CFG-distilled (turbo) or CFG-capable (sft/base) exactly like
-- their 2B counterparts; the helpers below keep that logic in one place.
local function acestep_is_turbo(v) return v == "turbo" or v == "xl_turbo" end
local function acestep_is_base(v)  return v == "base" or v == "xl_base" end
local function acestep_is_xl(v)    return v == "xl_turbo" or v == "xl_sft" or v == "xl_base" end
local ACESTEP_STEPS_RANGE_BY_VARIANT = {
  turbo = { 1, 20  },
  sft   = { 1, 200 },
  base  = { 1, 200 },
  xl_turbo = { 1, 20  },
  xl_sft   = { 1, 200 },
  xl_base  = { 1, 200 },
}
-- Mirrors research_acestep_gen_modal.py *_DEFAULT_BY_VARIANT. Modal sweep (8 samples,
-- CLAP): sft 50/g7 0.546 > turbo 0.518 > base 32/g7 0.479; base at 64 steps /
-- guidance 8 (+/- ADG) is worst and sometimes near-noise, so ADG stays off.
-- ACE-Step's docs also rate base "Medium" quality: it is meant for LoRA / fine-tunes.
local ACESTEP_STEPS_DEFAULT_BY_VARIANT    = { turbo = 8,   sft = 100, base = 32, xl_turbo = 8, xl_sft = 50, xl_base = 50 }
local ACESTEP_GUIDANCE_DEFAULT_BY_VARIANT = { turbo = 7.0, sft = 7.0, base = 7.0, xl_turbo = 7.0, xl_sft = 7.0, xl_base = 7.0 }
local ACESTEP_ADG_DEFAULT_BY_VARIANT      = { turbo = false, sft = false, base = false, xl_turbo = false, xl_sft = false, xl_base = false }
local ACESTEP_GUIDANCE_RANGE   = { 1.0, 15.0 }  -- base only
local ACESTEP_SHIFT_RANGE      = { 1.0, 5.0  }
local ACESTEP_CFG_RANGE        = { 0.0, 1.0  }  -- base only
local ACESTEP_LM_TEMP_RANGE    = { 0.0, 2.0  }
local ACESTEP_LORA_SCALE_RANGE = { 0.0, 1.0  }

-- ACE-Step song controls (docs: Tutorial.md "Input Fields", deapi prompting guide).
-- Put style/mood/instruments in the caption; tempo, key and meter go in their own
-- fields (anchors, not exact commands); lyrics + [structure] tags in the lyrics box.
local ACESTEP_KEYS = { "Auto" }
for _, n in ipairs({ "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B" }) do
  table.insert(ACESTEP_KEYS, n .. " major")
  table.insert(ACESTEP_KEYS, n .. " minor")
end
local ACESTEP_TS_LABELS = { "Auto", "4/4", "3/4", "6/8", "2/4" }
local ACESTEP_TS_VALUES = { "",     "4",   "3",   "6",   "2"   }
local ACESTEP_LANG_LABELS = { "Auto", "English", "Spanish", "Chinese", "Japanese", "Korean",
                              "German", "French", "Portuguese", "Italian", "Russian" }
local ACESTEP_LANG_VALUES = { "", "en", "es", "zh", "ja", "ko", "de", "fr", "pt", "it", "ru" }
local ACESTEP_BPM_RANGE   = { 0, 300 }  -- 0 = auto
-- 5Hz LM planner size (ACE-Step docs): 0.6B fast / <8GB VRAM, 1.7B default
-- (8-16GB), 4B for >16GB VRAM and complex or long-tail prompts.
local ACESTEP_LM_SIZES       = { "0.6B", "1.7B", "4B" }
local ACESTEP_LM_SIZE_LABELS = {
  "0.6B  (fastest, lowest VRAM)",
  "1.7B  (default)",
  "4B  (best planning, slower — use A100 with XL)",
}
local ACESTEP_STRUCT_TAGS = { "Insert structure tag...", "[intro]", "[verse]", "[pre-chorus]",
  "[chorus]", "[bridge]", "[inst]", "[build-up]", "[drop]", "[breakdown]", "[outro]" }
-- Edit tasks (ACE-Step "task_type"): cover keeps the song's melody/structure and
-- restyles it; repaint regenerates one region; lego ("Add layer") writes a new
-- instrument that fits the source and needs the base checkpoint.
local ACESTEP_EDIT_TASKS = { "cover", "repaint", "lego" }
local ACESTEP_EDIT_TASK_LABELS = {
  "Cover  (restyle the whole clip)",
  "Repaint  (regenerate a region)",
  "Add layer  (new instrument, base / XL base checkpoint)",
}
local ACESTEP_EDIT_VARIANTS = { "turbo", "sft", "base", "xl_turbo", "xl_sft", "xl_base" }
local ACESTEP_EDIT_VARIANT_LABELS = {
  "Turbo  (fast, 8 steps)",
  "SFT  (more detail, 50 steps, slower)",
  "Base  (required for Add layer)",
  "XL Turbo  (4B DiT, 8 steps)",
  "XL SFT  (4B DiT, highest quality)",
  "XL Base  (4B DiT)",
}
local ACESTEP_TRACKS = { "vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard",
                         "percussion", "strings", "synth", "brass", "woodwinds", "fx" }
local ACESTEP_REPAINT_MAX = 90  -- ACE-Step: repaint operates on 3-90 s regions

local function clamp(v, lo, hi) return math.max(lo, math.min(hi, v)) end

-- Example prompts per model, shown as a dropdown next to each prompt box.
-- Generation models each expect their own prompt "dialect"; edit models take a
-- transformation instruction applied to the source audio rather than a
-- from-scratch description. Sourced from each model's card / demo and from
-- evaluation/prompts_official.json and evaluation/edit/README.md, which
-- already collect the prompts used to validate these models.
local GEN_EXAMPLE_PROMPTS = {
  sao = {
    { label = "128 BPM tech house drum loop",
      text = "128 BPM tech house drum loop" },
    { label = "Hammer hitting a wooden surface (SFX)",
      text = "The sound of a hammer hitting a wooden surface." },
    { label = "Lo-fi slow BPM electro chill",
      text = "Lo-fi slow BPM electro chill with organic samples" },
  },
  -- The five full-prompt examples of the ACE-Step prompting guide (deapi.ai),
  -- BPM kept inside the tags as the guide recommends (also set it in Song
  -- controls: matching values tighten the rhythm).
  acestep_gen = {
    -- Examples may carry `lyrics` (+ `bpm`, `lang_idx`): picking one fills the
    -- Song controls too, so users see how tags and lyrics fit together.
    { label = "Lo-fi hip-hop, boom bap, 88 BPM (with lyrics)",
      text = "lo-fi hip-hop, boom bap, dusty drums, vinyl crackle, jazz sample, upright bass, rhodes piano, muted trumpet, male vocals, rap vocals, laid-back, warm, 90s, 88 bpm",
      bpm = 88, lang_idx = 2,
      lyrics = [[[intro]

[verse]
Rain on the window, vinyl spinning slow
Coffee going cold but I'm in the flow
Dusty drums knocking like a heartbeat
Walking through the city on tired feet

[chorus]
Lo-fi nights, lo-fi nights
Dim the lights, let it ride
Lo-fi nights, lo-fi nights
Hold the groove till the morning light

[verse]
Trumpet in the distance, smoke in the air
Old jazz samples floating everywhere
Pocket full of dreams and a notebook worn
Writing down the rhythm till the break of dawn

[chorus]
Lo-fi nights, lo-fi nights
Dim the lights, let it ride
Lo-fi nights, lo-fi nights
Hold the groove till the morning light

[inst]

[outro]
Easy now, let it fade
Vinyl crackle, slow decay]] },
    { label = "Dark progressive techno (instrumental), 134 BPM",
      text = "progressive techno, dark, hypnotic, driving, industrial, analog modular synth, 909 drums, acid bass, tom rolls, atmospheric pad, no vocals, hi-fi, wide stereo, 134 bpm" },
    { label = "Chamber folk ballad, 68 BPM",
      text = "chamber folk, acoustic ballad, intimate, melancholic, hopeful, fingerpicked classical guitar, cello, solo violin, upright bass, soft female vocals, subtle harmonies, warm analog, 2010s, 68 bpm" },
    { label = "Hybrid orchestral trailer, 90 BPM",
      text = "cinematic, epic hybrid trailer, dark, tense, grandiose, taiko drums, sub drops, low brass swell, staccato strings, choir ooh, piano ostinato, synth stab, no vocals, wide stereo, cinematic score, 2020s, 90 bpm" },
    { label = "Dark ambient, long-form, 55 BPM",
      text = "dark ambient, cinematic, atmospheric, ethereal, haunting, sustained analog drone, granular texture, reverb-heavy piano, field recording wind, distant choir, sub rumble, no vocals, slow evolving, hi-fi, wide stereo, 55 bpm" },
  },
  audiogen = {
    { label = "Footsteps on wooden floor, indoors",
      text = "Footsteps walking on a wooden floor, indoors, quiet environment" },
    { label = "Heavy rain on window, distant thunder",
      text = "Heavy rain falling on a window, with distant thunder rumbling" },
    { label = "Crowd cheering in a large stadium",
      text = "A crowd of people cheering and applauding in a large stadium" },
  },
  musicgen_gen = {
    { label = "80s pop, heavy drums, synth pads",
      text = "An 80s driving pop song with heavy drums and synth pads in the background" },
    { label = "Cheerful country, acoustic guitars, banjo",
      text = "A cheerful country song with acoustic guitars, banjo, and light drums" },
    { label = "Melancholic jazz, piano/bass/brushed drums",
      text = "Jazz music with piano, acoustic bass, and brushed drums. Melancholic mood." },
  },
}
local EDIT_EXAMPLE_PROMPTS = {
  sao_edit = {
    { label = "Jazz piano version, swing feel",
      text = "jazz piano version, swing feel, relaxed tempo" },
    { label = "Lo-fi hip hop remix, dusty vinyl texture",
      text = "lo-fi hip hop remix, chill, dusty vinyl texture, slow tempo" },
    { label = "Orchestral symphonic arrangement",
      text = "orchestral symphonic arrangement" },
  },
  acestep = {
    { label = "Cheerful country, acoustic guitars",
      text = "A cheerful country song with acoustic guitars" },
    { label = "8-bit chiptune arcade soundtrack version",
      text = "8-bit chiptune arcade video game soundtrack version" },
    { label = "Dark ambient, heavy reverb, ominous mood",
      text = "dark ambient version, heavy reverb, ominous and haunting mood" },
  },
}

local TEXT2AUDIO_PY = SCRIPT_DIR .. "text2audio.py"
local PROGRESS_F    = TMPDIR .. "t2a_progress.txt"
local LOG_F         = TMPDIR .. "t2a.log"

-- ── STATE ────────────────────────────────────────────────────────
local S = {
  -- Mode: 1=Generate, 2=Edit
  mode              = 1,
  -- Generate mode
  prompt            = "",
  prompt_example    = { idx = 1 },
  duration          = 20.0,
  gen_model_idx     = 1,
  -- ACE-Step 1.5 advanced inference parameters (acestep_gen only)
  acestep_variant_idx = 5,   -- 1=turbo 2=sft 3=base 4=xl_turbo 5=xl_sft (default: sounds best by ear) 6=xl_base
  acestep_steps       = 50,      -- ACESTEP_STEPS_DEFAULT_BY_VARIANT.xl_sft
  acestep_guidance    = 7.0,
  acestep_shift       = 3.0,
  acestep_seed        = -1,     -- -1 = random
  acestep_use_adg     = false,
  acestep_cfg_start   = 0.0,
  acestep_cfg_end     = 1.0,
  acestep_thinking    = true,    -- 5Hz LM planner: CLAP 0.518 -> 0.565 on turbo, floor 0.50 -> 0.52 on sft
  acestep_lm_temp     = 0.85,
  acestep_lm_idx      = 2,       -- 1.7B
  acestep_lora_path   = "",
  acestep_lora_scale  = 1.0,
  -- ACE-Step song controls (generate)
  acestep_instrumental = true,   -- true = lyrics "[Instrumental]" (no invented vocals)
  acestep_lyrics      = "",
  acestep_bpm         = 0,       -- 0 = auto
  acestep_key_idx     = 1,
  acestep_ts_idx      = 1,
  acestep_lang_idx    = 1,
  acestep_tag_idx     = 1,
  -- ACE-Step edit
  ace_task_idx        = 1,
  ace_edit_variant_idx = 1,
  ace_lego_xl         = false,   -- Add layer on the XL base checkpoint
  ace_strength        = 0.7,
  ace_rp_start        = 0.0,
  ace_rp_end          = 0.0,     -- 0 = until the end of the clip
  ace_track_idx       = 3,
  ace_edit_lyrics     = "",
  ace_edit_tag_idx    = 1,
  -- Edit mode
  src               = "",
  src_track_name    = "",
  src_track_idx     = -1,
  src_start_offs    = 0,
  src_section_dur   = 0,
  src_item_pos      = nil,
  src_is_section    = false,
  edit_prompt       = "",
  edit_prompt_example = { idx = 1 },
  intensity_idx     = 2,   -- "moderate"
  edit_model_idx    = 1,
  -- Execution: "local" (this machine, MPS/CPU) or "remote" (Modal cloud)
  execution         = "remote",
  -- Common
  gpu_idx           = 1,
  -- Runtime
  running           = false,
  pid               = nil,
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
    if S.prep then prepared.mark(S.prep[1], S.prep[2], S.prep[3]) end
    S.running   = false
    S.pid       = nil
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
    S.pid     = nil
    S.done    = true
    add_log("ERROR: " .. (r.msg or "?"))
    for _, line in ipairs(r.extra or {}) do
      local p = line:match("^%s*(.-)%s*$")
      if p ~= "" then add_log("  " .. p) end
    end
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
    reaper.MB("Item has no active take.", "AudioGenerator", 0); return false
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
    reaper.MB("Selected track has no audio items.", "AudioGenerator", 0)
    return
  end

  reaper.MB("No item or track selected in REAPER.", "AudioGenerator", 0)
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

  -- Land the new track right below the source track/clip (when there is
  -- one) instead of always at the very end of the project's track list.
  local tcnt_before = track_placement.insert_index(S.src_track_idx, reaper.CountTracks(0))
  reaper.InsertTrackAtIndex(tcnt_before, true)
  local new_track = reaper.GetTrack(0, tcnt_before)
  reaper.GetSetMediaTrackInfo_String(new_track, "P_NAME", track_name, true)
  reaper.SetOnlyTrackSelected(new_track)
  reaper.SetEditCurPos(cursor, false, false)

  -- Render happened in the temp folder: move a copy into the project's media
  -- folder first so the project never points outside itself.
  local dest, why = common.copy_into_project_media(wav_path)
  if not dest then
    add_log("Warning: audio stays outside the project (" .. tostring(why)
      .. "). Save the project and generate again to keep it inside.")
    dest = wav_path
  end
  reaper.InsertMedia(dest, 0)
  reaper.UpdateArrange()
  add_log("Imported: " .. track_name)
  reaper.Undo_EndBlock("AudioGenerator: import WAV", -1)
end

-- ── LAUNCH GENERATION/EDITING ────────────────────────────────────
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

-- Stops the in-flight generation/edit (best-effort: signals the local Modal
-- client process; a remote Modal job already dispatched may keep running).
local function stop_t2a()
  local signaled = common.stop_process(S.pid)
  add_log(signaled and "Stopped by user."
    or "Stopped by user (process already finished).")
  S.running = false
  S.pid     = nil
  S.done    = true
end

-- ACE-Step base is meant to run with a LoRA/LoKr adapter (alone it scores far
-- below SFT/turbo and some samples come out as near-noise), so generating with
-- the base checkpoint and no adapter folder is blocked.
-- ── ACE-STEP SONG ARGS ───────────────────────────────────────────
-- Lyrics travel through a UTF-8 file (multi-line text can't be shell-quoted).
local LYRICS_F = TMPDIR .. "t2a_lyrics.txt"
local function write_lyrics_file(text)
  local f = io.open(LYRICS_F, "wb")
  if not f then return false end
  f:write(text); f:close()
  return true
end

-- Builds the shared CLI flags: lyrics (if any) + BPM/key/meter/language.
local function acestep_song_args(lyrics, bpm, key_idx, ts_idx, lang_idx)
  local a = ""
  if lyrics and lyrics:match("%S") and write_lyrics_file(lyrics) then
    a = a .. " --lyrics-file " .. q(LYRICS_F)
    add_log("Lyrics: " .. #lyrics .. " chars")
  end
  bpm = clamp(math.floor((bpm or 0) + 0.5), ACESTEP_BPM_RANGE[1], ACESTEP_BPM_RANGE[2])
  if bpm > 0 then a = a .. string.format(" --bpm %d", bpm) end
  if key_idx and key_idx > 1 then a = a .. " --keyscale " .. q(ACESTEP_KEYS[key_idx]) end
  local ts = ACESTEP_TS_VALUES[ts_idx or 1] or ""
  if ts ~= "" then a = a .. " --timesignature " .. q(ts) end
  local lang = ACESTEP_LANG_VALUES[lang_idx or 1] or ""
  if lang ~= "" then a = a .. " --vocal-language " .. q(lang) end
  return a
end

-- XL DiT ~9 GB (docs: min 12 GB, 20 GB+ recommended); the 4B LM adds ~8 GB.
-- T4 has 16 GB, A10G 24 GB. Returns a warning string or nil.
local function acestep_gpu_warning(variant, thinking, lm_idx, gpu)
  local xl  = acestep_is_xl(variant)
  local lm4 = thinking and ACESTEP_LM_SIZES[lm_idx] == "4B"
  if gpu == "T4" and (xl or lm4) then
    return "T4 (16 GB) is too small for the XL DiT / 4B planner. Choose A100 or A100-80GB."
  end
  if gpu == "A10G" and xl and lm4 then
    return "XL DiT + 4B planner will likely run out of memory on an A10G (24 GB). Choose A100 or A100-80GB."
  end
  return nil
end

local function acestep_base_without_lora()
  return S.mode == 1
    and GEN_MODELS[S.gen_model_idx] == "acestep_gen"
    and acestep_is_base(ACESTEP_VARIANTS[S.acestep_variant_idx])
    and not S.acestep_lora_path:match("%S")
end

local function launch_t2a()
  if acestep_base_without_lora() then
    reaper.MB("ACE-Step base needs a LoRA/LoKr adapter folder.\n"
      .. "Pick an adapter, or switch the checkpoint to SFT.", "AudioGenerator", 0)
    return
  end
  local run_dir = make_run_dir()

  -- ── GENERATE mode ──
  if S.mode == 1 then
    local prompt = S.prompt:match("^%s*(.-)%s*$")
    if prompt == "" then
      reaper.MB("Write a text prompt before generating.", "AudioGenerator", 0)
      return
    end
    local model_key = GEN_MODELS[S.gen_model_idx]
    local script    = GEN_SCRIPTS[model_key]
    local f = io.open(script, "r")
    if not f then
      reaper.MB("Script not found:\n" .. tostring(script), "AudioGenerator", 0)
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

      -- CFG params only meaningful for base/sft — omit entirely for turbo
      -- rather than send values the model would silently ignore.
      if not acestep_is_turbo(variant) then
        local guidance = clamp(S.acestep_guidance, ACESTEP_GUIDANCE_RANGE[1], ACESTEP_GUIDANCE_RANGE[2])
        local cfg_s    = clamp(S.acestep_cfg_start, ACESTEP_CFG_RANGE[1], ACESTEP_CFG_RANGE[2])
        local cfg_e    = clamp(S.acestep_cfg_end, ACESTEP_CFG_RANGE[1], ACESTEP_CFG_RANGE[2])
        if cfg_s > cfg_e then cfg_s, cfg_e = cfg_e, cfg_s end
        extra = extra .. string.format(" --guidance-scale %.2f", guidance)
        if S.acestep_use_adg then extra = extra .. " --use-adg" end
        extra = extra .. string.format(" --cfg-start %.2f --cfg-end %.2f", cfg_s, cfg_e)
      end

      -- Keep the caption and the BPM field consistent (ACE-Step guide: both
      -- should carry the tempo). Only added when the caption doesn't state one.
      if S.acestep_bpm > 0 and not prompt:lower():find("%d+%s*bpm") then
        prompt = prompt .. string.format(", %d bpm", math.floor(S.acestep_bpm + 0.5))
        add_log("Caption + BPM: " .. prompt:sub(-24))
      end

      -- Instrumental (default) sends no lyrics: the backend then forces
      -- "[Instrumental]" so the DiT doesn't invent sung/mumbled vocals.
      local lyr = S.acestep_instrumental and "" or S.acestep_lyrics
      if not S.acestep_instrumental and not lyr:match("%S") then
        add_log("Warning: Instrumental is off but the lyrics are empty; generating instrumental.")
      end
      extra = extra .. acestep_song_args(lyr, S.acestep_bpm, S.acestep_key_idx,
                                         S.acestep_ts_idx, S.acestep_lang_idx)

      if S.acestep_thinking then
        extra = extra .. " --thinking"
        extra = extra .. " --lm-size " .. q(ACESTEP_LM_SIZES[S.acestep_lm_idx] or "1.7B")
        extra = extra .. string.format(" --lm-temperature %.2f", lm_temp)
      end
      if acestep_is_base(variant) and S.acestep_lora_path:match("%S") then
        local lora_scale = clamp(S.acestep_lora_scale, ACESTEP_LORA_SCALE_RANGE[1], ACESTEP_LORA_SCALE_RANGE[2])
        extra = extra .. " --lora-path " .. q(S.acestep_lora_path)
        extra = extra .. string.format(" --lora-scale %.2f", lora_scale)
        add_log("LoRA adapter: " .. (S.acestep_lora_path:match("[^/\\]+$") or S.acestep_lora_path)
          .. string.format(" (scale %.2f)", lora_scale))
      end
      local gpu_warn = acestep_gpu_warning(variant, S.acestep_thinking, S.acestep_lm_idx, GPUS[S.gpu_idx])
      if gpu_warn then add_log("Warning: " .. gpu_warn) end
      add_log(string.format(
        "ACE-Step params: variant=%s steps=%d shift=%.1f seed=%d thinking=%s",
        variant, steps, shift, math.floor(S.acestep_seed + 0.5), tostring(S.acestep_thinking)))
    end

    add_log("Execution: " .. S.execution)
    local cmd = string.format(
      '%s %s --shared-dir %s --script %s --model %s --mode generate'
      .. ' --prompt %s --seconds %.2f --gpu %s --execution %s'
      .. ' --out-dir %s%s --progress %s >>%s 2>&1 &',
      q(PYTHON), q(TEXT2AUDIO_PY),
      q(SHARED_DIR), q(script),
      q(model_key),
      q(prompt), S.duration, q(GPUS[S.gpu_idx]), q(S.execution),
      q(run_dir), extra, q(PROGRESS_F), q(LOG_F))

    add_log("Launching " .. (S.execution == "local" and "local" or "Modal") .. " process...")
    S.pid = common.launch_tracked(cmd)
    S.prep = { "text2audio", model_key, S.execution }

  -- ── EDIT mode ──
  else
    if S.src == "" then
      reaper.MB("Select an audio track, item or section first.\n"
        .. "Use the R button to capture the REAPER selection.", "AudioGenerator", 0)
      return
    end
    local prompt = S.edit_prompt:match("^%s*(.-)%s*$")
    if prompt == "" then
      reaper.MB("Write the change intent (e.g. 'jazz style with piano').", "AudioGenerator", 0)
      return
    end
    local model_key = EDIT_MODELS[S.edit_model_idx]
    local script    = EDIT_SCRIPTS[model_key]
    local f = io.open(script, "r")
    if not f then
      reaper.MB("Script not found:\n" .. tostring(script), "AudioGenerator", 0)
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

    local edit_extra = ""
    if model_key == "acestep" then
      local task    = ACESTEP_EDIT_TASKS[S.ace_task_idx] or "cover"
      local variant = ACESTEP_EDIT_VARIANTS[S.ace_edit_variant_idx] or "turbo"
      if task == "lego" then variant = S.ace_lego_xl and "xl_base" or "base" end
      local gpu_warn = acestep_gpu_warning(variant, false, 2, GPUS[S.gpu_idx])
      if gpu_warn then add_log("Warning: " .. gpu_warn) end
      edit_extra = edit_extra .. " --ace-task " .. task .. " --dit-variant " .. variant
      if task == "cover" then
        edit_extra = edit_extra .. string.format(" --strength %.2f", clamp(S.ace_strength, 0, 1))
      else
        local src_len = S.src_section_dur > 0 and S.src_section_dur or 600
        local rs = clamp(S.ace_rp_start, 0, src_len)
        local re = S.ace_rp_end > 0 and clamp(S.ace_rp_end, 0, src_len) or -1
        if re ~= -1 and re <= rs then
          reaper.MB("Repaint end must be after the start (or 0 for 'until the end').",
                    "AudioGenerator", 0)
          S.running = false
          S.status  = "Ready."
          return
        end
        edit_extra = edit_extra .. string.format(" --repaint-start %.3f --repaint-end %.3f", rs, re)
        add_log(string.format("Region: %.2fs → %s", rs, re == -1 and "end" or string.format("%.2fs", re)))
      end
      if task == "lego" then
        edit_extra = edit_extra .. " --track-name " .. q(ACESTEP_TRACKS[S.ace_track_idx] or "drums")
      end
      edit_extra = edit_extra .. acestep_song_args(S.ace_edit_lyrics, S.acestep_bpm,
        S.acestep_key_idx, S.acestep_ts_idx, S.acestep_lang_idx)
      add_log("ACE-Step edit: " .. task .. " · " .. variant)
    end

    add_log("Execution: " .. S.execution)
    local cmd = string.format(
      '%s %s --shared-dir %s --script %s --model %s --mode edit'
      .. ' --input %s --prompt %s --intensity %s --gpu %s --execution %s'
      .. ' --out-dir %s%s --progress %s >>%s 2>&1 &',
      q(PYTHON), q(TEXT2AUDIO_PY),
      q(SHARED_DIR), q(script),
      q(model_key),
      q(S.src), q(prompt), q(INTENSITIES[S.intensity_idx]), q(GPUS[S.gpu_idx]), q(S.execution),
      q(run_dir), section_args .. edit_extra,
      q(PROGRESS_F), q(LOG_F))

    add_log("Launching " .. (S.execution == "local" and "local" or "Modal") .. " process...")
    S.pid = common.launch_tracked(cmd)
    S.prep = { "text2audio", model_key, S.execution }
  end
end

-- ── MODULE ───────────────────────────────────────────────────────
local M = { title = "AudioGenerator" }

function M.init()
  if PYTHON_ERR then
    reaper.ShowConsoleMsg("AudioGenerator - WARNING: " .. PYTHON_ERR .. "\n")
  end
  add_log("AudioGenerator ready.")
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
  g.separator(); g.spacing()

  -- ── MODEL (fixed, above the scrollable content — same position across
  -- every tab in the app). Which combo is shown depends on Generate/Edit
  -- mode, but it's always the first thing under the mode tabs. ──────
  if S.mode == 1 then
    g.row_label("Model:", t.sc(70))
    g.next_width(-1)
    local old_gen_model_idx = S.gen_model_idx
    S.gen_model_idx = widgets.combo("##gen_model", S.gen_model_idx, GEN_LABELS)
    -- A prompt written for one model's phrasing conventions (e.g.
    -- a tag list vs a sentence) rarely still makes sense for another, so
    -- clear it instead of silently carrying it over.
    S.prompt = model_switch.next_prompt(old_gen_model_idx, S.gen_model_idx, S.prompt)
  else
    g.row_label("Model:", t.sc(70))
    g.next_width(-1)
    local old_edit_model_idx = S.edit_model_idx
    S.edit_model_idx = widgets.combo("##edit_model", S.edit_model_idx, EDIT_LABELS)
    -- The change-intent prompt's phrasing advice is model-specific, so
    -- clear it instead of silently carrying it over to a different model.
    S.edit_prompt = model_switch.next_prompt(old_edit_model_idx, S.edit_model_idx, S.edit_prompt)
  end

  -- ── EXECUTION (Local / Remote) ──────────────────────────────────
  local cur_model_key = (S.mode == 1) and GEN_MODELS[S.gen_model_idx] or EDIT_MODELS[S.edit_model_idx]
  local local_ok, local_reason = model_execution.is_local_supported("text2audio", cur_model_key)
  if not local_ok and S.execution == "local" then
    S.execution = "remote"  -- model switched into one with no local backend
  end
  g.row_label("Execution:", t.sc(70))
  if not local_ok then g.begin_disabled(true) end
  g.next_width(t.sc(140))
  local EXEC_OPTS = { "remote", "local" }
  local EXEC_LABELS = { "Remote (Modal)", "Local (MPS/CPU)" }
  local exec_idx = (S.execution == "local") and 2 or 1
  local new_exec_idx = widgets.combo("##t2a_exec", exec_idx, EXEC_LABELS)
  S.execution = EXEC_OPTS[new_exec_idx]
  if not local_ok then
    g.end_disabled()
    g.same_line(t.sc(10))
    g.text_disabled("Local unavailable: " .. local_reason)
  end
  g.spacing()
  prepared.notice(g, "text2audio", cur_model_key, S.execution)

  -- ── SOURCE (fixed, directly below Model — same position across every
  -- tab that has a source file to pick). Edit mode only. ────────────
  if S.mode == 2 then
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
  end

  g.separator(); g.spacing()

  -- ── SCROLL REGION: rest of the page (prompt/params/button/log) ───
  -- Single, non-nested scroll_region for everything below Model/Source.
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

    -- GPU
    g.row_label("GPU:", t.sc(70))
    g.next_width(t.sc(90))
    S.gpu_idx = widgets.combo("##gen_gpu", S.gpu_idx, GPUS)
    g.spacing()
    g.separator(); g.spacing()

    -- Prompt
    g.push_font(t.F.H1)
    g.text("Prompt")
    g.pop_font()
    if GEN_MODELS[S.gen_model_idx] == "acestep_gen" then
      g.text_disabled("Tags: genre, mood, 2-3 specific instruments, vocal type, production, then BPM")
      g.text_disabled("(e.g. '..., hi-fi, wide stereo, 90 bpm'). See the examples below.")
    else
      g.text_disabled("Describe the audio: instrument, BPM, genre, duration, key...")
    end
    g.spacing()

    local changed_p, new_p = widgets.input_textarea("##gen_prompt", S.prompt, 4)
    if changed_p then S.prompt = new_p end

    local gen_examples = GEN_EXAMPLE_PROMPTS[GEN_MODELS[S.gen_model_idx]]
    if gen_examples then
      g.next_width(-1)
      local picked = widgets.example_prompt_picker("##gen_prompt_ex", S.prompt_example, gen_examples)
      if picked then
        S.prompt = picked
        for _, ex in ipairs(gen_examples) do
          if ex.text == picked and ex.lyrics then
            S.acestep_lyrics      = ex.lyrics
            S.acestep_instrumental = false
            if ex.bpm then S.acestep_bpm = ex.bpm end
            if ex.lang_idx then S.acestep_lang_idx = ex.lang_idx end
          end
        end
      end
    end

    g.spacing()

    -- Duration
    local max_sec = GEN_MAX_SEC[GEN_MODELS[S.gen_model_idx]] or 47.0
    g.row_label("Duration:", t.sc(70))
    g.next_width(-1)
    local ch_dur, new_dur = g.slider_float("##gen_dur", S.duration, 1.0, max_sec, "%.1f s")
    if ch_dur then S.duration = new_dur end
    g.spacing()

    -- ACE-Step song controls: lyrics + structure + musical metadata
    if GEN_MODELS[S.gen_model_idx] == "acestep_gen" then
      if widgets.collapsing_header("Song controls (lyrics, tempo, key)", true) then
        local lw = t.sc(90)
        local c_i, v_i = g.checkbox("Instrumental (no vocals)##ace_instr", S.acestep_instrumental)
        if c_i then S.acestep_instrumental = v_i end
        if not S.acestep_instrumental then
          g.text_disabled("Lyrics with [verse] / [chorus] tags. 4-8 syllables per line; no empty [chorus].")
          local c_l, v_l = widgets.input_textarea("##ace_lyrics", S.acestep_lyrics, 8)
          if c_l then S.acestep_lyrics = v_l end
          g.next_width(t.sc(200))
          local old_tag = S.acestep_tag_idx
          S.acestep_tag_idx = widgets.combo("##ace_tag", S.acestep_tag_idx, ACESTEP_STRUCT_TAGS)
          if S.acestep_tag_idx ~= old_tag and S.acestep_tag_idx > 1 then
            local sep = (S.acestep_lyrics == "" or S.acestep_lyrics:sub(-1) == "\n") and "" or "\n"
            S.acestep_lyrics = S.acestep_lyrics .. sep .. ACESTEP_STRUCT_TAGS[S.acestep_tag_idx] .. "\n"
            S.acestep_tag_idx = 1
          end
          g.row_label("Language:", lw)
          g.next_width(t.sc(140))
          S.acestep_lang_idx = widgets.combo("##ace_lang", S.acestep_lang_idx, ACESTEP_LANG_LABELS)
        end
        g.spacing()
        g.row_label("BPM:", lw)
        g.next_width(t.sc(150))
        local rb, nb = g.slider_int("##ace_bpm", S.acestep_bpm, ACESTEP_BPM_RANGE[1], ACESTEP_BPM_RANGE[2])
        if rb then S.acestep_bpm = nb end
        g.same_line(t.sc(10)); g.inline_text(S.acestep_bpm == 0 and "(0 = auto)" or ""); g.end_row()
        g.row_label("Key:", lw)
        g.next_width(t.sc(140))
        S.acestep_key_idx = widgets.combo("##ace_key", S.acestep_key_idx, ACESTEP_KEYS)
        g.same_line(t.sc(14)); g.inline_text("Meter:")
        g.same_line(t.sc(6)); g.next_width(t.sc(90))
        S.acestep_ts_idx = widgets.combo("##ace_ts", S.acestep_ts_idx, ACESTEP_TS_LABELS)
        g.text_disabled("Optional anchors, not exact commands. Leave on Auto unless you need a specific tempo/key.")
      end
      g.spacing()
    end

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
          S.acestep_steps    = ACESTEP_STEPS_DEFAULT_BY_VARIANT[variant]
          S.acestep_guidance = ACESTEP_GUIDANCE_DEFAULT_BY_VARIANT[variant]
          S.acestep_use_adg  = ACESTEP_ADG_DEFAULT_BY_VARIANT[variant]
        end
        local variant = ACESTEP_VARIANTS[S.acestep_variant_idx] or "turbo"
        local steps_range = ACESTEP_STEPS_RANGE_BY_VARIANT[variant]
        if not acestep_is_turbo(variant) or acestep_is_xl(variant) then
          g.text_disabled("This checkpoint downloads separately on first use (larger than turbo).")
        end
        local gw = acestep_gpu_warning(variant, S.acestep_thinking, S.acestep_lm_idx, GPUS[S.gpu_idx])
        if gw then g.text_colored(gw, "YELLOW") end
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
        g.same_line(t.sc(10)); g.inline_text("(-1 = random)"); g.end_row()

        -- CFG controls — base checkpoint only; turbo is CFG-distilled and
        -- generate_music() hard-overrides guidance_scale to 1.0 for it, so
        -- these are hidden entirely for turbo rather than shown-but-inert.
        if not acestep_is_turbo(variant) then
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
          g.row_label("Planner:", lw)
          g.next_width(-1)
          S.acestep_lm_idx = widgets.combo("##ace_lm_size", S.acestep_lm_idx, ACESTEP_LM_SIZE_LABELS)
          g.row_label("LM temp.:", lw)
          g.next_width(t.sc(120))
          rv, nv = g.slider_float("##ace_lmtemp", S.acestep_lm_temp,
            ACESTEP_LM_TEMP_RANGE[1], ACESTEP_LM_TEMP_RANGE[2], "%.2f")
          if rv then S.acestep_lm_temp = nv end
        end

        if acestep_is_base(variant) then
        g.spacing()
        g.text("LoRA / LoKr adapter (required for base):")
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
        end -- acestep_is_base(variant)
      end
      g.spacing()
    end

    -- Cost hint
    g.text_disabled("A10G: ~$0.05/min  |  SAO ~20-40 s  |  ACE-Step/MusicGen ~30-60 s")

  -- ════════════════════════════════════════════
  else
  -- ── EDIT MODE ───────────────────────────────

    -- Change intent prompt
    g.push_font(t.F.H1)
    g.text("Change intent")
    g.pop_font()
    g.text_disabled("Describe how to transform the audio (e.g. 'jazz version with piano')")
    g.spacing()

    local changed_ep, new_ep = widgets.input_textarea("##edit_prompt", S.edit_prompt, 3)
    if changed_ep then S.edit_prompt = new_ep end

    local edit_examples = EDIT_EXAMPLE_PROMPTS[EDIT_MODELS[S.edit_model_idx]]
    if edit_examples then
      g.next_width(-1)
      local picked = widgets.example_prompt_picker("##edit_prompt_ex", S.edit_prompt_example, edit_examples)
      if picked then S.edit_prompt = picked end
    end
    g.spacing()

    -- Intensity (SAO edit) / ACE-Step task controls
    local cur_edit_model = EDIT_MODELS[S.edit_model_idx]
    if cur_edit_model == "acestep" then
      local lw = t.sc(90)
      g.row_label("Task:", lw)
      g.next_width(-1)
      local old_task = S.ace_task_idx
      S.ace_task_idx = widgets.combo("##ace_task", S.ace_task_idx, ACESTEP_EDIT_TASK_LABELS)
      -- Add layer only exists on the base checkpoints: base, or XL base.
      local task = ACESTEP_EDIT_TASKS[S.ace_task_idx]
      g.row_label("Checkpoint:", lw)
      g.next_width(-1)
      if task == "lego" then
        local cx, vx = g.checkbox("Use XL base (4B, needs A100 class GPU)##ace_lego_xl", S.ace_lego_xl)
        if cx then S.ace_lego_xl = vx end
        g.text_disabled(S.ace_lego_xl and "Checkpoint: XL Base" or "Checkpoint: Base")
      else
        S.ace_edit_variant_idx = widgets.combo("##ace_edit_variant", S.ace_edit_variant_idx,
                                               ACESTEP_EDIT_VARIANT_LABELS)
      end
      g.spacing()
      if task == "cover" then
        g.row_label("Fidelity:", lw)
        g.next_width(t.sc(200))
        local rs, ns = g.slider_float("##ace_strength", S.ace_strength, 0.0, 1.0, "%.2f")
        if rs then S.ace_strength = ns end
        g.text_disabled("Higher = closer to the original melody/structure; lower = freer restyle.")
      else
        local src_len = S.src_section_dur > 0 and S.src_section_dur or 60
        g.row_label("From (s):", lw)
        g.next_width(t.sc(150))
        local r1, n1 = g.slider_float("##ace_rp_s", S.ace_rp_start, 0.0, src_len, "%.2f")
        if r1 then S.ace_rp_start = n1 end
        g.row_label("To (s):", lw)
        g.next_width(t.sc(150))
        local r2, n2 = g.slider_float("##ace_rp_e", S.ace_rp_end, 0.0, src_len, "%.2f")
        if r2 then S.ace_rp_end = n2 end
        g.text_disabled(string.format("0 = until the end. Region is relative to the captured clip; ACE-Step works on 3-%d s.",
          ACESTEP_REPAINT_MAX))
        if task == "lego" then
          g.row_label("Add:", lw)
          g.next_width(t.sc(170))
          S.ace_track_idx = widgets.combo("##ace_track", S.ace_track_idx, ACESTEP_TRACKS)
          g.text_disabled("Returns ONLY the new layer (verified), imported on its own track next to the source. Base checkpoint, larger download.")
        end
      end
      g.spacing()
      if widgets.collapsing_header("Song controls (lyrics, tempo, key)", false) then
        g.text_disabled("Optional. Lyrics of the source (or new ones) help covers with vocals.")
        local c_l, v_l = widgets.input_textarea("##ace_edit_lyrics", S.ace_edit_lyrics, 6)
        if c_l then S.ace_edit_lyrics = v_l end
        g.row_label("BPM:", lw)
        g.next_width(t.sc(150))
        local rb, nb = g.slider_int("##ace_edit_bpm", S.acestep_bpm, ACESTEP_BPM_RANGE[1], ACESTEP_BPM_RANGE[2])
        if rb then S.acestep_bpm = nb end
        g.same_line(t.sc(10)); g.inline_text(S.acestep_bpm == 0 and "(0 = auto)" or ""); g.end_row()
        g.row_label("Key:", lw)
        g.next_width(t.sc(140))
        S.acestep_key_idx = widgets.combo("##ace_edit_key", S.acestep_key_idx, ACESTEP_KEYS)
        g.same_line(t.sc(14)); g.inline_text("Meter:")
        g.same_line(t.sc(6)); g.next_width(t.sc(90))
        S.acestep_ts_idx = widgets.combo("##ace_edit_ts", S.acestep_ts_idx, ACESTEP_TS_LABELS)
      end
      g.spacing()
    else
    do
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
    }
    g.text_disabled(model_hints[cur_edit_model] or "")
  end

  -- ── Common section ───────────────────────────────────────────
  g.spacing()
  g.separator()
  g.spacing()

  -- Main button (doubles as STOP while running)
  local btn_colors = {
    norm   = S.mode == 1 and {0x14/255, 0x6A/255, 0x3C/255} or {0x3C/255, 0x14/255, 0x6A/255},
    hover  = S.mode == 1 and {0x1A/255, 0x88/255, 0x4D/255} or {0x4D/255, 0x1A/255, 0x88/255},
    active = S.mode == 1 and {0x22/255, 0xA5/255, 0x5E/255} or {0x5E/255, 0x22/255, 0xA5/255},
  }
  local stop_colors = {
    norm   = t.C.RED,
    hover  = { 0xE8/255, 0x5A/255, 0x50/255 },
    active = { 0xF2/255, 0x70/255, 0x66/255 },
  }
  local btn_lbl = S.running
    and (S.mode == 1 and "STOP  (Generating...)" or "STOP  (Editing...)")
    or  (S.mode == 1 and "GENERATE AUDIO"        or "EDIT AUDIO")

  local blocked = (not S.running) and acestep_base_without_lora()
  if blocked then g.begin_disabled(true) end
  g.next_width(-1)
  if g.button(btn_lbl, nil, t.sc(36), { solid = S.running and stop_colors or btn_colors }) then
    if S.running then stop_t2a() else launch_t2a() end
  end
  if blocked then
    g.end_disabled()
    g.text_colored("Base needs a LoRA adapter: choose a folder above, or switch to SFT.", "YELLOW")
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
