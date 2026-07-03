-- @description Text2Audio — Generación y edición de audio con IA
-- @version 1.1
-- @author IAClaude
-- @about Genera audio desde texto o edita un audio existente usando modelos de IA
--        en la nube (Modal). Dos modos:
--          · Generar: prompt de texto → WAV estéreo
--                     (SAO, Foundation-1, ACE-Step, InspireMusic,
--                      Mustango, AudioGen, MusicGen, MAGNeT)
--          · Editar:  item/pista seleccionado + intención → WAV transformado
--                     (SAO style transfer, ACE-Step 1.5, MusicGen-melody,
--                      MelodyFlow, ZETA/AudioLDM2, InspireMusic continuation)
--        Input edición: pista, item o split (sección) seleccionado en REAPER.
--        Output: pista de audio nueva con el WAV generado, en la posición del source.
--        UI nativa gfx: sin dependencias externas de extensiones REAPER.

-- ── RUTAS + LIB ──────────────────────────────────────────────────
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

-- ── CONSTANTES ───────────────────────────────────────────────────
-- Modelos de generación (text → audio, sin source)
local GEN_MODELS  = { "sao", "foundation1", "acestep_gen", "inspiremusic_gen",
                      "mustango", "audiogen", "musicgen_gen", "magnet" }
local GEN_LABELS  = {
  "Stable Audio Open 1.0  (A10G, 44.1 kHz estéreo)",
  "Foundation-1  (A10G, electrónica, formato TAG)",
  "ACE-Step 1.5  (A10G, full-song, Apache 2.0)",
  "InspireMusic 1.5B  (A10G, 48 kHz, Apache 2.0)",
  "Mustango  (A10G, ~10 s fijo, MuBERT features)",
  "AudioGen-medium  (A10G, 16 kHz, efectos/sonido)",
  "MusicGen-medium  (A10G, 32 kHz, CC-BY-NC)",
  "MAGNeT-medium  (A10G, 32 kHz, sin-AR, CC-BY-NC)",
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

-- Modelos de edición (source audio + prompt → audio transformado)
local EDIT_MODELS  = { "sao_edit", "acestep", "musicgen",
                       "melodyflow", "zeta", "inspiremusic" }
local EDIT_LABELS  = {
  "SAO Style Transfer  (A10G, SDEdit init_audio)",
  "ACE-Step 1.5  (A10G, cover/re-estilo, Apache 2.0)",
  "MusicGen-melody  (A10G, condicionamiento melódico, CC-BY-NC)",
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
local INTENSITY_LABELS = { "Suave (subtle)", "Moderado (moderate)", "Fuerte (strong)" }

local TEXT2AUDIO_PY = SCRIPT_DIR .. "text2audio.py"
local PROGRESS_F    = TMPDIR .. "t2a_progress.txt"
local LOG_F         = TMPDIR .. "t2a.log"

-- ── ESTADO ───────────────────────────────────────────────────────
local S = {
  -- Modo: 1=Generar, 2=Editar
  mode              = 1,
  -- Modo Generar
  prompt            = "",
  duration          = 8.0,
  gen_model_idx     = 1,
  -- Modo Editar
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
  -- Común
  gpu_idx           = 1,
  -- Runtime
  running           = false,
  done              = false,
  progress          = 0.0,
  status            = "Listo.",
  log               = {},
  out_files         = {},
  log_scroll_to_bottom = false,
}

-- ── HELPERS CORE ─────────────────────────────────────────────────
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
    ["modal-auth"] = "Modal sin auth",
  }
  for _, line in ipairs(r.extra) do
    local name, status = line:match("^CHECK|([^|]+)|([^|]+)|")
    if name and status == "missing" and CORE_LABELS[name] then
      table.insert(setup_missing, CORE_LABELS[name])
    end
  end
end

-- ── PROGRESO ─────────────────────────────────────────────────────
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
      add_log("Audio listo: " .. (S.out_files[1]:match("[^/\\]+$") or S.out_files[1]))
      import_audio()
    end

  elseif r.state == "error" and not S.done then
    S.running = false
    S.done    = true
    add_log("ERROR: " .. (r.msg or "?"))
  end
end

-- ── INTEGRACIÓN REAPER ───────────────────────────────────────────
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
    reaper.MB("El item no tiene take activo.", "Text2Audio", 0); return false
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
    add_log(string.format("Fuente (%s): %s [%.2fs → %.2fs]",
      kind, fname:match("[^/\\]+$") or fname, offs, offs + dur))
  else
    add_log(string.format("Fuente (%s): %s", kind, fname:match("[^/\\]+$") or fname))
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
      if _set_src_from_item(item, "pista") then return end
    end
    reaper.MB("La pista seleccionada no tiene items de audio.", "Text2Audio", 0)
    return
  end

  reaper.MB("No hay ningún item ni pista seleccionada en REAPER.", "Text2Audio", 0)
end

-- ── IMPORTAR AUDIO ───────────────────────────────────────────────
function import_audio()
  if #S.out_files == 0 then return end
  local wav_path = S.out_files[1]
  local f = io.open(wav_path, "rb")
  if not f then
    add_log("Error: no se puede leer " .. wav_path); return
  end
  f:close()

  reaper.Undo_BeginBlock()
  local cursor = S.src_item_pos or reaper.GetCursorPosition()

  -- Nombre base para la nueva pista
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
  add_log("Importado: " .. track_name)
  reaper.Undo_EndBlock("Text2Audio: import WAV", -1)
end

-- ── LANZAR GENERACIÓN/EDICIÓN ────────────────────────────────────
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

  -- ── Modo GENERAR ──
  if S.mode == 1 then
    local prompt = S.prompt:match("^%s*(.-)%s*$")
    if prompt == "" then
      reaper.MB("Escribe un prompt de texto antes de generar.", "Text2Audio", 0)
      return
    end
    local model_key = GEN_MODELS[S.gen_model_idx]
    local script    = GEN_SCRIPTS[model_key]
    local f = io.open(script, "r")
    if not f then
      reaper.MB("Script no encontrado:\n" .. tostring(script), "Text2Audio", 0)
      return
    end
    f:close()

    local label = "Iniciando " .. (GEN_LABELS[S.gen_model_idx] or model_key) .. "..."
    clear_run(label)
    add_log("Modo: Generar")
    add_log("Modelo: " .. (GEN_LABELS[S.gen_model_idx] or model_key))
    add_log("GPU: " .. GPUS[S.gpu_idx])
    add_log(string.format("Duración: %.1fs", S.duration))
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

    add_log("Lanzando proceso Modal...")
    os.execute(cmd)

  -- ── Modo EDITAR ──
  else
    if S.src == "" then
      reaper.MB("Selecciona una pista, item o sección de audio primero.\n"
        .. "Usa el botón R para capturar la selección de REAPER.", "Text2Audio", 0)
      return
    end
    local prompt = S.edit_prompt:match("^%s*(.-)%s*$")
    if prompt == "" then
      reaper.MB("Escribe la intención del cambio (ej. 'estilo jazz con piano').", "Text2Audio", 0)
      return
    end
    local model_key = EDIT_MODELS[S.edit_model_idx]
    local script    = EDIT_SCRIPTS[model_key]
    local f = io.open(script, "r")
    if not f then
      reaper.MB("Script no encontrado:\n" .. tostring(script), "Text2Audio", 0)
      return
    end
    f:close()

    local label = "Iniciando " .. (EDIT_LABELS[S.edit_model_idx] or model_key) .. "..."
    clear_run(label)
    add_log("Modo: Editar")
    add_log("Modelo: " .. (EDIT_LABELS[S.edit_model_idx] or model_key))
    add_log("GPU: " .. GPUS[S.gpu_idx])
    add_log("Intensidad: " .. INTENSITIES[S.intensity_idx])
    add_log("Fuente: " .. (S.src:match("[^/\\]+$") or S.src))
    add_log("Prompt: " .. prompt:sub(1, 80))

    local section_args = ""
    if S.src_is_section then
      section_args = string.format(" --start %.6f --duration %.6f",
        S.src_start_offs, S.src_section_dur)
      add_log(string.format("Sección: %.2fs → %.2fs",
        S.src_start_offs, S.src_start_offs + S.src_section_dur))
    end

    -- MusicGen necesita --seconds
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

    add_log("Lanzando proceso Modal...")
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
    g.text_wrapped("⚠  Config. incompleta: " .. table.concat(setup_missing, " · "))
    g.text_disabled("Carga shared/Setup.lua en Actions > Load ReaScript para configurar.")
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

  -- ── Tabs de modo ──
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
  if g.button("⊕ Generar", half_w, t.sc(30),
      { solid = S.mode == 1 and c_gen_act or c_gen_dim }) then
    S.mode = 1
  end
  g.same_line()
  if g.button("✏ Editar", half_w, t.sc(30),
      { solid = S.mode == 2 and c_edit_act or c_edit_dim }) then
    S.mode = 2
  end
  g.spacing()
  g.separator()
  g.spacing()

  -- ════════════════════════════════════════════
  if S.mode == 1 then
  -- ── MODO GENERAR ────────────────────────────

    -- Prompt
    g.push_font(t.F.H1)
    g.text("Prompt")
    g.pop_font()
    g.text_disabled("Describe el audio: instrumento, BPM, género, duración, tonalidad...")
    g.spacing()

    local changed_p, new_p = widgets.input_textarea("##gen_prompt", S.prompt, 4)
    if changed_p then S.prompt = new_p end

    -- Hint Foundation-1
    if GEN_MODELS[S.gen_model_idx] == "foundation1" then
      g.text_colored(
        "Foundation-1: usa formato TAG → Instrumento, FX, Genre, N Bars, BPM, Key",
        "YELLOW")
    end
    g.spacing()

    -- Duración
    local max_sec = GEN_MAX_SEC[GEN_MODELS[S.gen_model_idx]] or 47.0
    g.row_label("Duración:", t.sc(70))
    g.next_width(-1)
    local ch_dur, new_dur = g.slider_float("##gen_dur", S.duration, 1.0, max_sec, "%.1f s")
    if ch_dur then S.duration = new_dur end

    -- Modelo
    g.row_label("Modelo:", t.sc(70))
    g.next_width(-1)
    S.gen_model_idx = widgets.combo("##gen_model", S.gen_model_idx, GEN_LABELS)

    -- GPU
    g.row_label("GPU:", t.sc(70))
    g.next_width(t.sc(90))
    S.gpu_idx = widgets.combo("##gen_gpu", S.gpu_idx, GPUS)
    g.spacing()

    -- Hint Mustango (duración fija)
    if GEN_MODELS[S.gen_model_idx] == "mustango" then
      g.text_colored("Mustango: duración fija ~10 s (el slider se ignora)", "YELLOW")
    end
    -- Hint coste
    g.text_disabled("A10G: ~$0.05/min  |  SAO/Foundation-1 ~20-40 s  |  ACE-Step/MusicGen ~30-60 s")

  -- ════════════════════════════════════════════
  else
  -- ── MODO EDITAR ─────────────────────────────

    -- Fuente
    g.push_font(t.F.H1)
    g.text("Audio fuente")
    g.pop_font()

    g.row_label("Fuente:", t.sc(54))
    local display_src = (S.src_track_name ~= "")
      and (S.src_track_name .. "  (" .. (S.src:match("[^/\\]+$") or "") .. ")")
      or S.src
    g.next_width(-(2 * t.SPACING_X + 2 * t.sc(44)))
    widgets.input_text("##src_disp", display_src, { readonly = true })
    g.same_line()
    if g.button("...", t.sc(44), t.ITEM_H) then
      local ok, fn = reaper.GetUserFileNameForRead("", "Abrir audio", "wav")
      if ok then
        S.src = fn; S.src_track_name = ""; S.src_track_idx = -1
        S.src_is_section = false; S.src_item_pos = nil
      end
    end
    g.same_line()
    if g.button("R", t.sc(44), t.ITEM_H) then grab_from_reaper() end

    if S.src_track_name ~= "" then
      g.text_disabled("Pista seleccionada  |  clic en R para actualizar")
    else
      g.text_disabled("Clic en R para usar la pista/item/split activo de REAPER")
    end
    if S.src_is_section then
      g.text_colored(string.format("Sección: %.2fs → %.2fs  (%.2fs)",
        S.src_start_offs, S.src_start_offs + S.src_section_dur, S.src_section_dur),
        "YELLOW")
    end
    g.spacing()

    -- Prompt intención
    g.push_font(t.F.H1)
    g.text("Intención del cambio")
    g.pop_font()
    g.text_disabled("Describe cómo transformar el audio (ej. 'versión jazz con piano')")
    g.spacing()

    local changed_ep, new_ep = widgets.input_textarea("##edit_prompt", S.edit_prompt, 3)
    if changed_ep then S.edit_prompt = new_ep end
    g.spacing()

    -- Modelo edición
    g.row_label("Modelo:", t.sc(80))
    g.next_width(-1)
    S.edit_model_idx = widgets.combo("##edit_model", S.edit_model_idx, EDIT_LABELS)
    g.spacing()

    -- Intensidad (no aplica a MusicGen ni InspireMusic continuation)
    local cur_edit_model = EDIT_MODELS[S.edit_model_idx]
    local EDIT_NO_INTENSITY = { musicgen = true, inspiremusic = true }
    if not EDIT_NO_INTENSITY[cur_edit_model] then
      g.row_label("Intensidad:", t.sc(80))
      g.next_width(t.sc(170))
      S.intensity_idx = widgets.combo("##intensity", S.intensity_idx, INTENSITY_LABELS)
      g.same_line(t.sc(12))
      local hints = {
        subtle   = "Conserva estructura original",
        moderate = "Equilibrio transformación/fidelidad",
        strong   = "Transformación profunda",
      }
      g.text_disabled(hints[INTENSITIES[S.intensity_idx]] or "")
      g.spacing()
    end

    -- Duración (solo MusicGen)
    if EDIT_NEEDS_SECONDS[cur_edit_model] then
      g.row_label("Duración:", t.sc(80))
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

    -- Hints por modelo
    local model_hints = {
      sao_edit    = "SAO SDEdit: reutiliza pesos ya descargados de SAO 1.0 (sin coste extra de setup)",
      acestep     = "ACE-Step: requiere setup inicial (~5 GB). Apache 2.0, uso comercial libre.",
      musicgen    = "MusicGen-melody: melody conditioning. CC-BY-NC, solo uso no comercial.",
      melodyflow  = "MelodyFlow: flow matching con inversión latente. Alta fidelidad. ≤30 s.",
      zeta        = "ZETA/AudioLDM2: DDIM inversion zero-shot. Output 16 kHz mono ≤10 s.",
      inspiremusic= "InspireMusic: continúa el audio con el estilo indicado en el prompt. ≤30 s.",
    }
    g.text_disabled(model_hints[cur_edit_model] or "")
  end

  -- ── Sección común ────────────────────────────────────────────
  g.spacing()
  g.separator()
  g.spacing()

  -- Botón principal
  local btn_colors = {
    norm   = S.mode == 1 and {0x14/255, 0x6A/255, 0x3C/255} or {0x3C/255, 0x14/255, 0x6A/255},
    hover  = S.mode == 1 and {0x1A/255, 0x88/255, 0x4D/255} or {0x4D/255, 0x1A/255, 0x88/255},
    active = S.mode == 1 and {0x22/255, 0xA5/255, 0x5E/255} or {0x5E/255, 0x22/255, 0xA5/255},
  }
  local btn_lbl = S.running
    and (S.mode == 1 and "[ Generando... ]" or "[ Editando... ]")
    or  (S.mode == 1 and "GENERAR AUDIO"    or "EDITAR AUDIO")

  g.begin_disabled(S.running)
  g.next_width(-1)
  if g.button(btn_lbl, nil, t.sc(36), { solid = btn_colors }) then
    launch_t2a()
  end
  g.end_disabled()
  g.spacing()

  -- Barra de progreso
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
    if g.button("Copiar log", t.sc(90), t.ITEM_H) then
      local ok, _ = pcall(function()
        reaper.CF_SetClipboard(table.concat(S.log, "\n"))
      end)
      if not ok then
        reaper.ShowConsoleMsg(table.concat(S.log, "\n") .. "\n")
      end
    end
    g.same_line()
    if g.button("Limpiar", t.sc(70), t.ITEM_H) then S.log = {} end
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

-- ── INICIO ───────────────────────────────────────────────────────
add_log("Text2Audio listo.")
add_log("Python: " .. PYTHON)
add_log("Backend: " .. TEXT2AUDIO_PY)
launch_setup_check()
reaper.defer(loop)
