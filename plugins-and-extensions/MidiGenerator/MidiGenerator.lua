-- @description MidiGenerator — Generación de MIDI con IA
-- @version 1.0
-- @author IAClaude
-- @about Genera MIDI desde texto o seed MIDI usando 6 modelos evaluados en la nube
--        (Modal CUDA). Soporta: Amadeus, MIDI-LLM, text2midi, ChatMusician,
--        MuseCoco, y Anticipatory (acompañamiento/continuación).
--        UI nativa gfx; sin dependencias de extensiones REAPER.

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
  reaper.ShowConsoleMsg("MidiGenerator - WARNING: " .. PYTHON_ERR .. "\n")
end

-- ── CONSTANTES: MODELOS ──────────────────────────────────────────
local MG_MODELS = {
  "amadeus", "midi_llm", "text2midi",
  "chatmusician", "musecoco", "anticipatory",
}
local MG_LABELS = {
  "Amadeus  (multi-track, MidiCaps, A10G)",
  "MIDI-LLM  (multi-track, libre, A10G) [solo CUDA]",
  "text2midi  (baseline, multi-track) [calidad baja]",
  "ChatMusician  (multi-voz, ABC notation) [limitado]",
  "MuseCoco  (multi-track, atributos, A100) [~11 min]",
  "Anticipatory  (acompañamiento/cover, seed MIDI, A10G)",
}
local MG_SCRIPTS = {}
for _, k in ipairs(MG_MODELS) do
  MG_SCRIPTS[k] = SCRIPT_DIR .. "research/research_" .. k .. "_modal.py"
end
-- MuseCoco usa nombre diferente en disco
MG_SCRIPTS["musecoco"] = SCRIPT_DIR .. "research/research_musecoco_modal.py"

local MG_GPUS        = { "A10G", "A100", "T4", "L4" }
local MG_GPU_DEFAULT = {
  amadeus="A10G", midi_llm="A10G", text2midi="A10G",
  chatmusician="A10G", musecoco="A100", anticipatory="A10G",
}
-- Modelos donde GPU es pasada al script (los demás la ignoran)
local MG_GPU_RELEVANT = { midi_llm=true, anticipatory=true }

local AMT_MODES = { "accompaniment", "continuation" }
local AMT_MODE_LABELS = { "Acompañamiento", "Continuación" }

local MIDIGEN_PY = SCRIPT_DIR .. "midigen.py"
local PROGRESS_F = TMPDIR .. "midigen_progress.txt"
local LOG_F      = TMPDIR .. "midigen.log"

-- ── ESTADO ───────────────────────────────────────────────────────
local S = {
  -- Modelo
  model_idx = 1,
  gpu       = "A10G",
  -- Prompt (modelos de texto)
  prompt    = "",
  -- Campos opcionales MidiCaps (amadeus / text2midi)
  field_key         = "",
  field_bpm         = "",
  field_instruments = "",
  field_chords      = "",
  -- ChatMusician: seed opcional para armonización
  cm_use_seed = false,
  -- Seed MIDI (ChatMusician armonización)
  seed_path       = "",
  seed_label      = "",
  -- Parámetros AMT
  amt_mode_idx     = 1,   -- 1=accompaniment, 2=continuation
  amt_duration     = 20,  -- clip_length (accompaniment) o duration (continuation)
  amt_prompt_len   = 5,   -- prompt_length (solo accompaniment)
  -- AMT: pista de melodía
  amt_melody_item  = nil,   -- item REAPER (para posición/longitud al importar)
  amt_melody_take  = nil,
  amt_melody_label = "",
  -- AMT: pistas de seed de acompañamiento
  amt_seed_takes   = {},  -- lista de {take=take, label="nombre pista"}
  -- Parámetros comunes
  n_outputs   = 2,
  temperature = 1.0,
  -- Runtime
  running    = false,
  done       = false,
  progress   = 0.0,
  status     = "Listo.",
  log        = {},
  out_files  = {},
  n_instruments = -1,
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
  local d = TMPDIR .. "midigen_run" .. _run_id .. "/"
  os.execute("mkdir -p " .. q(d))
  return d
end

local function model_key() return MG_MODELS[S.model_idx] end

-- ── ESCRITOR SMF (exportar toma MIDI in-project) ─────────────────
-- Escribe un SMF Type-1 mínimo a partir de una take MIDI de REAPER.
-- Soporta note on/off. Usado cuando el source no es un .mid en disco.

local function _vlq(n)
  if n < 0x80 then return string.char(n) end
  local bytes = {}
  while n > 0 do
    table.insert(bytes, 1, n % 128)
    n = math.floor(n / 128)
  end
  for i = 1, #bytes - 1 do bytes[i] = bytes[i] + 128 end
  return string.char(table.unpack(bytes))
end

local function _u32be(v)
  v = math.floor(v) % (2^32)
  return string.char(
    math.floor(v/16777216)%256,
    math.floor(v/65536)%256,
    math.floor(v/256)%256,
    v%256)
end

local function _u16be(v)
  v = math.floor(v) % 65536
  return string.char(math.floor(v/256)%256, v%256)
end

local function write_midi_from_take(take, filepath)
  local PPQ = 480
  local bpm = reaper.Master_GetTempo()
  local tempo_uspb = math.floor(60000000 / bpm)

  -- Tiempo de proyecto en tick 0 del take (posición del item + offset de fuente).
  -- Restarlo normaliza todos los eventos a t=0 relativo al inicio del item,
  -- que es lo que esperan las librerías AMT (anticipation usa clip(events, 0, N)).
  local t_item_start = reaper.MIDI_GetProjTimeFromPPQPos(take, 0)

  -- Recoger notas: {ppq_start, ppq_end, chan, pitch, vel}
  -- MIDI_CountEvts devuelve (retval, notes, ccs, sysex)
  local _, noteCount = reaper.MIDI_CountEvts(take)
  noteCount = noteCount or 0
  local events = {}
  for i = 0, noteCount - 1 do
    local _, _, _, startppq, endppq, chan, pitch, vel =
      reaper.MIDI_GetNote(take, i)
    -- Convertir a tiempo relativo (s desde inicio del item) y luego a PPQ-480
    local t_start = reaper.MIDI_GetProjTimeFromPPQPos(take, startppq) - t_item_start
    local t_end   = reaper.MIDI_GetProjTimeFromPPQPos(take, endppq)   - t_item_start
    local p_start = math.floor(t_start * (bpm/60) * PPQ + 0.5)
    local p_end   = math.floor(t_end   * (bpm/60) * PPQ + 0.5)
    if p_end > p_start then
      table.insert(events, { tick=p_start, status=0x90+(chan%16), d1=pitch, d2=vel })
      table.insert(events, { tick=p_end,   status=0x80+(chan%16), d1=pitch, d2=0   })
    end
  end
  table.sort(events, function(a,b) return a.tick < b.tick end)

  -- Construir track de notas
  local trk = ""
  local prev_tick = 0
  for _, ev in ipairs(events) do
    local delta = math.max(0, ev.tick - prev_tick)
    trk = trk .. _vlq(delta) .. string.char(ev.status, ev.d1, ev.d2)
    prev_tick = ev.tick
  end
  trk = trk .. "\000\255\047\000"  -- delta=0, meta End of Track

  -- Construir track de tempo
  local tmp = "\000\255\081\003"  -- delta=0, meta Tempo, len=3
    .. string.char(
        math.floor(tempo_uspb/65536)%256,
        math.floor(tempo_uspb/256)%256,
        tempo_uspb%256)
    .. "\000\255\047\000"

  -- Ensamblar SMF
  local f = io.open(filepath, "wb")
  if not f then return false, "No se pudo crear " .. filepath end
  -- MThd
  f:write("MThd" .. _u32be(6) .. _u16be(1) .. _u16be(2) .. _u16be(PPQ))
  -- Tempo track
  f:write("MTrk" .. _u32be(#tmp) .. tmp)
  -- Notas track
  f:write("MTrk" .. _u32be(#trk) .. trk)
  f:close()
  return true, nil
end

-- ── COMBINAR MÚLTIPLES TOMAS EN UN SOLO MIDI (para AMT) ─────────
-- melodía → canal 0, cada seed take → canal 1,2,3...
local function write_combined_midi(melody_take, seed_takes, filepath)
  local PPQ = 480
  local bpm = reaper.Master_GetTempo()
  local tempo_uspb = math.floor(60000000 / bpm)
  local events = {}

  local function collect_take(take, force_chan)
    local t0 = reaper.MIDI_GetProjTimeFromPPQPos(take, 0)
    local _, nc = reaper.MIDI_CountEvts(take)
    for i = 0, (nc or 0) - 1 do
      local _, _, _, sp, ep, _, pitch, vel = reaper.MIDI_GetNote(take, i)
      local ts = reaper.MIDI_GetProjTimeFromPPQPos(take, sp) - t0
      local te = reaper.MIDI_GetProjTimeFromPPQPos(take, ep) - t0
      local ps = math.floor(ts * (bpm/60) * PPQ + 0.5)
      local pe = math.floor(te * (bpm/60) * PPQ + 0.5)
      if pe > ps then
        table.insert(events, { tick=ps, status=0x90+force_chan, d1=pitch, d2=vel })
        table.insert(events, { tick=pe, status=0x80+force_chan, d1=pitch, d2=0   })
      end
    end
  end

  collect_take(melody_take, 0)
  for i, entry in ipairs(seed_takes) do
    collect_take(entry.take, math.min(i, 15))
  end
  table.sort(events, function(a,b) return a.tick < b.tick end)

  local trk = ""
  local prev = 0
  for _, ev in ipairs(events) do
    local d = math.max(0, ev.tick - prev)
    trk = trk .. _vlq(d) .. string.char(ev.status, ev.d1, ev.d2)
    prev = ev.tick
  end
  trk = trk .. "\000\255\047\000"

  local tmp = "\000\255\081\003"
    .. string.char(
        math.floor(tempo_uspb/65536)%256,
        math.floor(tempo_uspb/256)%256,
        tempo_uspb%256)
    .. "\000\255\047\000"

  local f = io.open(filepath, "wb")
  if not f then return false, "No se pudo crear " .. filepath end
  f:write("MThd" .. _u32be(6) .. _u16be(1) .. _u16be(2) .. _u16be(PPQ))
  f:write("MTrk" .. _u32be(#tmp) .. tmp)
  f:write("MTrk" .. _u32be(#trk) .. trk)
  f:close()
  return true, nil
end

-- ── SETUP CHECK ──────────────────────────────────────────────────
local SETUP_CHECK_F = TMPDIR .. "reaperai_setup_check.txt"
local SETUP_HELPER  = SHARED_DIR .. "setup_helpers.py"
local setup_missing  = {}
local setup_checked  = false

local function launch_setup_check()
  local f = io.open(SETUP_CHECK_F, "w")
  if f then f:write("running|0.00|..."); f:close() end
  os.execute(string.format('%s %s check --progress %s >>%s 2>&1 &',
    q(PYTHON), q(SETUP_HELPER), q(SETUP_CHECK_F),
    q(TMPDIR .. "reaperai_setup.log")))
end

local function poll_setup_check()
  if setup_checked then return end
  local r = common.read_progress_file(SETUP_CHECK_F)
  if not r or r.state ~= "done" then return end
  setup_checked = true
  local CORE = { python="Python REAPER", uv="uv",
                 ["modal-cli"]="Modal CLI", ["modal-auth"]="Modal sin auth" }
  for _, line in ipairs(r.extra) do
    local name, status = line:match("^CHECK|([^|]+)|([^|]+)|")
    if name and status == "missing" and CORE[name] then
      table.insert(setup_missing, CORE[name])
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
    S.running = false; S.done = true
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
      add_log(string.format("MIDI listo (%s instrumento%s, %d candidato%s)",
        S.n_instruments >= 0 and tostring(S.n_instruments) or "?",
        S.n_instruments ~= 1 and "s" or "",
        #S.out_files, #S.out_files ~= 1 and "s" or ""))
      import_midi_all()
    end
  elseif r.state == "error" and not S.done then
    S.running = false; S.done = true
    add_log("ERROR: " .. (r.msg or "?"))
  end
end

-- ── IMPORTAR MIDI (multi-candidato) ──────────────────────────────
-- Cada candidato (.mid) se importa en su propia carpeta de pistas.
-- Reutiliza la lógica de Audio2Midi con I_FOLDERDEPTH para la carpeta.

local function _import_one(mid_path, folder_name)
  local f = io.open(mid_path, "rb")
  if not f then add_log("Error: no se puede leer " .. mid_path); return end
  f:close()

  local cursor      = reaper.GetCursorPosition()
  local tcnt_before = reaper.CountTracks(0)

  -- Snapshot de markers existentes: InsertMedia puede crear markers desde
  -- meta-events MIDI (tipo 0x06/0x07), que crean ruido visual en el timeline.
  local marker_snap = {}
  for i = 0, reaper.CountProjectMarkers(0) - 1 do
    local _, _, _, _, _, idx = reaper.EnumProjectMarkers(i)
    marker_snap[idx] = true
  end

  -- Deseleccionar todas las pistas antes de InsertMedia: si hay pistas seleccionadas
  -- (p.ej. del candidato anterior), InsertMedia añadiría items a esas pistas
  -- en lugar de crear pistas nuevas.
  for i = 0, reaper.CountTracks(0) - 1 do
    reaper.SetTrackSelected(reaper.GetTrack(0, i), false)
  end

  reaper.SetEditCurPos(cursor, false, false)
  reaper.InsertMedia(mid_path, 0)

  -- Eliminar markers añadidos por la importación
  for i = reaper.CountProjectMarkers(0) - 1, 0, -1 do
    local _, isrgn, _, _, _, idx = reaper.EnumProjectMarkers(i)
    if not isrgn and not marker_snap[idx] then
      reaper.DeleteProjectMarker(0, idx, false)
    end
  end

  local delta = reaper.CountTracks(0) - tcnt_before
  if delta <= 0 then
    add_log("Aviso: InsertMedia no añadió pistas para " .. mid_path:match("[^/\\]+$"))
    return
  end

  -- Corregir posición: InsertMedia para MIDI multi-track puede ignorar
  -- SetEditCurPos e insertar los items en posición 0. Detectar y offsetear.
  local min_pos = math.huge
  for i = tcnt_before, tcnt_before + delta - 1 do
    local tr = reaper.GetTrack(0, i)
    if tr then
      for j = 0, reaper.CountTrackMediaItems(tr) - 1 do
        local it = reaper.GetTrackMediaItem(tr, j)
        if it then
          local p = reaper.GetMediaItemInfo_Value(it, "D_POSITION")
          if p < min_pos then min_pos = p end
        end
      end
    end
  end
  if min_pos ~= math.huge and math.abs(min_pos - cursor) > 0.001 then
    local offset = cursor - min_pos
    for i = tcnt_before, tcnt_before + delta - 1 do
      local tr = reaper.GetTrack(0, i)
      if tr then
        for j = 0, reaper.CountTrackMediaItems(tr) - 1 do
          local it = reaper.GetTrackMediaItem(tr, j)
          if it then
            local p = reaper.GetMediaItemInfo_Value(it, "D_POSITION")
            reaper.SetMediaItemInfo_Value(it, "D_POSITION", p + offset)
          end
        end
      end
    end
  end

  -- Siempre crear estructura de carpeta (independientemente de cuántas pistas)
  reaper.InsertTrackAtIndex(tcnt_before, true)
  local folder_tr = reaper.GetTrack(0, tcnt_before)
  reaper.GetSetMediaTrackInfo_String(folder_tr, "P_NAME", folder_name, true)
  reaper.SetMediaTrackInfo_Value(folder_tr, "I_FOLDERDEPTH", 1)
  for i = 1, delta do
    local tr = reaper.GetTrack(0, tcnt_before + i)
    if tr then
      local _, existing = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
      if existing == "" then
        reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", folder_name .. " " .. i, true)
      end
    end
  end
  local last_tr = reaper.GetTrack(0, tcnt_before + delta)
  if last_tr then reaper.SetMediaTrackInfo_Value(last_tr, "I_FOLDERDEPTH", -1) end
  add_log(string.format("Importado en carpeta '%s' (%d pista%s)",
    folder_name, delta, delta == 1 and "" or "s"))
end

function import_midi_all()
  if #S.out_files == 0 then return end
  local mk    = model_key()
  local label = "[MIDI " .. mk .. "]"

  -- Para AMT continuación: posicionar cursor al final del item de melodía
  -- para que la continuación se inserte justo después (no solapando el seed).
  if mk == "anticipatory"
      and AMT_MODES[S.amt_mode_idx] == "continuation"
      and S.amt_melody_item then
    local pos = reaper.GetMediaItemInfo_Value(S.amt_melody_item, "D_POSITION")
    local len = reaper.GetMediaItemInfo_Value(S.amt_melody_item, "D_LENGTH")
    reaper.SetEditCurPos(pos + len, false, false)
    add_log(string.format("Cursor → %.2fs (fin del item de melodía)", pos + len))
  end

  reaper.Undo_BeginBlock()
  for i, path in ipairs(S.out_files) do
    local suffix = #S.out_files > 1 and (" — candidato " .. i) or ""
    _import_one(path, label .. suffix)
  end
  reaper.UpdateArrange()
  reaper.Undo_EndBlock("MidiGenerator: import MIDI", -1)
end

-- ── CAPTURA DE SEED MIDI ─────────────────────────────────────────
local function _try_get_source_path(take)
  local src = reaper.GetMediaItemTake_Source(take)
  if not src then return nil end
  local fname = reaper.GetMediaSourceFileName(src, "")
  if not fname or fname == "" then return nil end
  -- Verificar que es un .mid existente en disco
  local ext = fname:match("%.([^%.]+)$")
  if not ext then return nil end
  ext = ext:lower()
  if ext == "mid" or ext == "midi" then
    local f = io.open(fname, "rb")
    if f then f:close(); return fname end
  end
  return nil
end

local function _export_take_to_tmp(take)
  local tmp_path = TMPDIR .. "midigen_seed_" .. _run_id .. ".mid"
  local ok, err = write_midi_from_take(take, tmp_path)
  if ok then return tmp_path end
  add_log("Aviso al exportar MIDI: " .. (err or "error desconocido"))
  return nil
end

local function grab_midi_from_reaper()
  local n_items = reaper.CountSelectedMediaItems(0)
  local item = nil
  if n_items > 0 then
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
    reaper.MB("No hay ningún item MIDI seleccionado en REAPER.", "MidiGenerator", 0)
    return
  end

  local take = reaper.GetActiveTake(item)
  if not take or not reaper.TakeIsMIDI(take) then
    reaper.MB("El item seleccionado no contiene MIDI.", "MidiGenerator", 0)
    return
  end

  -- Intentar ruta directa de fichero .mid
  local path = _try_get_source_path(take)
  if path then
    S.seed_path  = path
    S.seed_label = path:match("[^/\\]+$") or path
    add_log("Seed MIDI (fichero): " .. S.seed_label)
    return
  end

  -- Exportar take in-project a fichero temporal
  add_log("MIDI in-project detectado; exportando a fichero temporal...")
  path = _export_take_to_tmp(take)
  if path then
    S.seed_path  = path
    S.seed_label = "(take in-project → exportado)"
    add_log("Seed exportado: " .. path)
  else
    reaper.MB(
      "No se pudo exportar el MIDI in-project.\n"
      .. "Exporta la pista MIDI a un fichero .mid (File → Export → MIDI file)"
      .. " y selecciónalo con el botón '...'.",
      "MidiGenerator", 0)
  end
end

local function _get_track_label(item)
  local tr = reaper.GetMediaItemTrack(item)
  if not tr then return "?" end
  local _, tname = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
  local tnum = reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER")
  return (tname ~= "" and tname) or ("Pista " .. math.floor(tnum))
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
    reaper.MB("No hay ningún item MIDI seleccionado en REAPER.", "MidiGenerator", 0)
    return
  end
  local take = reaper.GetActiveTake(item)
  if not take or not reaper.TakeIsMIDI(take) then
    reaper.MB("El item seleccionado no contiene MIDI.", "MidiGenerator", 0)
    return
  end
  S.amt_melody_item  = item
  S.amt_melody_take  = take
  S.amt_melody_label = _get_track_label(item)
  add_log("Melodía capturada: " .. S.amt_melody_label)
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
    reaper.MB("No hay items MIDI seleccionados en REAPER.", "MidiGenerator", 0)
  else
    add_log("Añadidas " .. added .. " pista(s) de seed.")
  end
end

-- ── LANZAR GENERACIÓN ────────────────────────────────────────────
local function clear_run(label)
  local f = io.open(PROGRESS_F, "w")
  if f then f:write("running|0.00|" .. label); f:close() end
  local lf = io.open(LOG_F, "w"); if lf then lf:close() end
  S.running = true; S.done = false
  S.progress = 0; S.out_files = {}
  S.log = {}; S.status = label
  S.log_scroll_to_bottom = false
end

local function launch_generate()
  local mk       = model_key()
  local script   = MG_SCRIPTS[mk]
  local is_amt   = (mk == "anticipatory")
  local is_text  = not is_amt
  local has_fields = (mk == "amadeus" or mk == "text2midi")

  -- Validaciones
  if is_text and S.prompt:match("^%s*$") then
    reaper.MB("Escribe un prompt de texto antes de generar.", "MidiGenerator", 0)
    return
  end
  if is_amt and not S.amt_melody_take then
    reaper.MB(
      "Anticipatory necesita una pista de melodía.\n"
      .. "Selecciona un item MIDI en REAPER y clic R junto a 'Melodía'.",
      "MidiGenerator", 0)
    return
  end
  local amt_mode_is_acc = is_amt and (AMT_MODES[S.amt_mode_idx] == "accompaniment")
  if amt_mode_is_acc and #S.amt_seed_takes == 0 then
    reaper.MB(
      "El modo acompañamiento necesita al menos una pista de seed.\n"
      .. "Selecciona items MIDI en REAPER y clic '+' junto a 'Seed acc.'.",
      "MidiGenerator", 0)
    return
  end
  if mk == "chatmusician" and S.cm_use_seed and S.seed_path == "" then
    reaper.MB("Activaste 'Armonizar seed' pero no has cargado ningún MIDI.", "MidiGenerator", 0)
    return
  end

  local f = io.open(script, "r")
  if not f then
    reaper.MB("Script no encontrado:\n" .. script, "MidiGenerator", 0); return
  end
  f:close()

  local label = "Iniciando " .. (MG_LABELS[S.model_idx] or mk) .. "..."
  clear_run(label)
  add_log("Modelo: " .. (MG_LABELS[S.model_idx] or mk))
  add_log("GPU: " .. S.gpu)

  local run_dir = make_run_dir()

  -- Construir comando base (sin backgroundear todavía)
  local base = string.format(
    '%s %s --shared-dir %s --script %s --model %s'
    .. ' --out-dir %s --gpu %s --n-outputs %d --progress %s',
    q(PYTHON), q(MIDIGEN_PY),
    q(SHARED_DIR), q(script), q(mk),
    q(run_dir), q(S.gpu), S.n_outputs,
    q(PROGRESS_F))

  local extra = ""
  if is_text then
    add_log("Prompt: " .. S.prompt:sub(1,80))
    extra = extra .. " --prompt " .. q(S.prompt)
    extra = extra .. string.format(" --temperature %.2f", S.temperature)
    -- BPM siempre desde el proyecto para amadeus/text2midi (el usuario no lo edita)
    if has_fields then
      extra = extra .. " --field-bpm " .. q(string.format("%.0f", reaper.Master_GetTempo()))
    end
    if S.field_key         ~= "" then extra = extra .. " --field-key "         .. q(S.field_key)         end
    if S.field_instruments ~= "" then extra = extra .. " --field-instruments " .. q(S.field_instruments) end
    if S.field_chords      ~= "" then extra = extra .. " --field-chords "      .. q(S.field_chords)      end
    if mk == "chatmusician" and S.cm_use_seed and S.seed_path ~= "" then
      extra = extra .. " --seed-file " .. q(S.seed_path)
    end
  else
    -- AMT: construir MIDI combinado (melodía canal 0 + seed canales 1..N)
    -- Escribir a TMPDIR (fuera de run_dir) para que _collect_outputs no lo recoja
    local seed_path = TMPDIR .. "combined_seed.mid"
    local ok_s, err_s = write_combined_midi(S.amt_melody_take, S.amt_seed_takes, seed_path)
    if not ok_s then
      reaper.MB("Error al construir seed MIDI: " .. (err_s or "?"), "MidiGenerator", 0)
      S.running = false
      return
    end
    local mode = AMT_MODES[S.amt_mode_idx]
    add_log("Modo AMT: " .. mode)
    add_log("Melodía: " .. S.amt_melody_label)
    add_log("Seed tracks: " .. #S.amt_seed_takes)
    add_log("Seed MIDI combinado: " .. seed_path)
    extra = extra .. " --seed "              .. q(seed_path)
    extra = extra .. " --mode "              .. q(mode)
    extra = extra .. " --prompt-length "     .. tostring(S.amt_prompt_len)
    extra = extra .. " --clip-length "       .. tostring(S.amt_duration)
    extra = extra .. " --melody-instrument 0"
    -- --n-outputs ya está en base; midigen.py lo mapea internamente a --multiplicity
  end

  local cmd = base .. extra .. " >>" .. q(LOG_F) .. " 2>&1 &"
  add_log("Lanzando Modal...")
  os.execute(cmd)
end

-- ── GFX INIT ─────────────────────────────────────────────────────
if gfx.w > 0 then gfx.quit() end
local LOGICAL_W = 560
gfx.init("MidiGenerator", LOGICAL_W, 680)
gfx.ext_retina = 1
theme.init_fonts()

-- ── MAIN LOOP ────────────────────────────────────────────────────
local _scale_init = false

local function loop()
  if not _scale_init then
    _scale_init = true
    local s = math.floor(gfx.w / LOGICAL_W + 0.5)
    if s > 1 then theme.apply_scale(s); theme.init_fonts(s) end
  end

  gui.frame_begin()
  if gui.ctx.should_close then gfx.quit(); return end

  local g = gui
  local t = theme
  local mk = model_key()
  local is_amt  = (mk == "anticipatory")
  local is_text = not is_amt
  local has_fields = (mk == "amadeus" or mk == "text2midi")
  local has_opt_seed = (mk == "chatmusician")

  -- Setup banner
  poll_setup_check()
  if setup_checked and #setup_missing > 0 then
    g.text_wrapped("⚠  Config. incompleta: " .. table.concat(setup_missing, " · "))
    g.text_disabled("Carga shared/Setup.lua en Actions > Load ReaScript.")
    g.spacing()
  end

  -- Header
  g.push_font(t.F.H1)
  g.text("MIDI Generator")
  g.pop_font()
  g.same_line(10)
  g.text_colored("● REAPER OK", "GREEN")
  g.separator(); g.spacing()

  -- ── MODELO ─────────────────────────────────────────────────────
  g.row_label("Modelo:", t.sc(70))
  g.next_width(-1)
  local old_idx = S.model_idx
  S.model_idx = widgets.combo("##mg_model", S.model_idx, MG_LABELS)
  if S.model_idx ~= old_idx then
    S.gpu = MG_GPU_DEFAULT[MG_MODELS[S.model_idx]] or "A10G"
  end
  g.spacing()
  g.separator(); g.spacing()

  -- ── INPUT: PROMPT O SEED ────────────────────────────────────────
  if is_text then
    -- ── Prompt ────────────────────────────────────────────────────
    g.text("Prompt:")
    g.next_width(-1)
    local rv, nv = widgets.input_textarea("##mg_prompt", S.prompt, 3,
      { placeholder = "Describe el estilo musical, instrumentos, mood..." })
    if rv then S.prompt = nv end

    -- Nota informativa por modelo
    if mk == "midi_llm" then
      g.text_disabled("  El modelo antepone su system-prompt internamente.")
    elseif mk == "chatmusician" then
      g.text_disabled("  Usa frases como: 'chord progression Am-F-C-G'.")
    elseif mk == "musecoco" then
      g.text_disabled("  Incluye instrumentos en el texto para activar el override de clases.")
    elseif mk == "amadeus" or mk == "text2midi" then
      g.text_disabled("  Rellena los campos opcionales para resultados más predecibles.")
    end

    -- ── Seed opcional ChatMusician ─────────────────────────────────
    if has_opt_seed then
      g.spacing()
      local chg, nv2 = g.checkbox("Armonizar seed MIDI##cm_seed", S.cm_use_seed)
      if chg then S.cm_use_seed = nv2 end
      if S.cm_use_seed then
        g.row_label("Seed:", t.sc(70))
        g.next_width(-(2 * t.SPACING_X + 2 * t.sc(44)))
        widgets.input_text("##mg_seed_disp", S.seed_label ~= "" and S.seed_label or "(ninguno)", { readonly=true })
        g.same_line()
        if g.button("...", t.sc(44), t.ITEM_H) then
          local ok, fn = reaper.GetUserFileNameForRead("", "Seed MIDI", "mid")
          if ok then S.seed_path = fn; S.seed_label = fn:match("[^/\\]+$") or fn end
        end
        g.same_line()
        if g.button("R", t.sc(44), t.ITEM_H) then grab_midi_from_reaper() end
      end
    end

    -- ── Campos opcionales MidiCaps ─────────────────────────────────
    if has_fields then
      g.spacing()
      if widgets.collapsing_header("Campos opcionales (key / BPM / instrumentos / acordes)", false) then
        local lw = t.sc(90)
        g.row_label("Tonalidad:", lw)
        g.next_width(t.sc(120))
        local r1, v1 = widgets.input_text("##mg_key", S.field_key)
        if r1 then S.field_key = v1 end
        g.same_line(t.sc(14)); g.inline_text("BPM:")
        g.same_line(t.sc(6))
        g.text_colored(string.format("%.1f", reaper.Master_GetTempo()), "GREEN")
        g.same_line(t.sc(4)); g.text_disabled("(proyecto)")

        g.row_label("Instrumentos:", lw)
        g.next_width(-1)
        local r3, v3 = widgets.input_text("##mg_instr", S.field_instruments)
        if r3 then S.field_instruments = v3 end

        g.row_label("Acordes:", lw)
        g.next_width(-1)
        local r4, v4 = widgets.input_text("##mg_chords", S.field_chords)
        if r4 then S.field_chords = v4 end

        g.text_disabled("  Ejemplo: F minor | 120 | piano and bass | Fm-Db-Ab-Eb")
      end
    end

  else
    -- ── AMT: melodía + seed de acompañamiento ─────────────────────
    local lw_amt = t.sc(90)

    -- Pista de melodía
    g.row_label("Melodía:", lw_amt)
    g.next_width(-(t.SPACING_X + t.sc(44)))
    widgets.input_text("##amt_mel_disp",
      S.amt_melody_label ~= "" and S.amt_melody_label or "(ninguna — selecciona un item MIDI)",
      { readonly=true })
    g.same_line()
    if g.button("R", t.sc(44), t.ITEM_H) then grab_melody_item() end
    g.text_disabled("  Selecciona el item de melodía en REAPER y clic R.")

    g.spacing()
    g.row_label("Modo:", lw_amt)
    g.next_width(t.sc(160))
    S.amt_mode_idx = widgets.combo("##amt_mode", S.amt_mode_idx, AMT_MODE_LABELS)
    local amt_mode_key = AMT_MODES[S.amt_mode_idx]

    -- Pistas de seed de acompañamiento (solo en modo accompaniment)
    if amt_mode_key == "accompaniment" then
      g.spacing()
      g.row_label("Seed acc.:", lw_amt)
      if g.button("+", t.sc(44), t.ITEM_H) then add_seed_items_from_reaper() end
      g.same_line(t.sc(8))
      if g.button("Limpiar", t.sc(70), t.ITEM_H) then
        S.amt_seed_takes = {}
        add_log("Seed de acompañamiento limpiado.")
      end
      if #S.amt_seed_takes == 0 then
        g.text_disabled("  (ninguna pista añadida)")
      else
        for _, entry in ipairs(S.amt_seed_takes) do
          g.text_disabled("  • " .. entry.label)
        end
      end
      g.text_disabled("  Selecciona pistas de acompañamiento en REAPER y clic '+'.")
    end

    g.spacing()
    local lw = t.sc(110)
    g.row_label("Duración (s):", lw)
    g.next_width(t.sc(120))
    local rv, nv = g.slider_int("##amt_dur", S.amt_duration, 5, 120)
    if rv then S.amt_duration = nv end

    if amt_mode_key == "accompaniment" then
      g.same_line(t.sc(14)); g.inline_text("Hist. (s):")
      g.same_line(t.sc(6))
      g.next_width(t.sc(90))
      rv, nv = g.slider_int("##amt_plen", S.amt_prompt_len, 1, 15)
      if rv then S.amt_prompt_len = nv end

      g.spacing()
      g.text_colored(
        "⚠  El seed de acompañamiento debe contener ≥" .. S.amt_prompt_len ..
        "s de historia para resultados densos.", "YELLOW")
    end
  end

  g.spacing(); g.separator(); g.spacing()

  -- ── PARÁMETROS COMUNES ──────────────────────────────────────────
  local lw2 = t.sc(90)
  g.row_label("Candidatos:", lw2)
  g.next_width(t.sc(90))
  local rv, nv = g.slider_int("##mg_nout", S.n_outputs, 1, 4)
  if rv then S.n_outputs = nv end

  if is_text then
    g.same_line(t.sc(18)); g.inline_text("Temperatura:")
    g.same_line(t.sc(6))
    g.next_width(-1)
    rv, nv = g.slider_float("##mg_temp", S.temperature, 0.5, 2.0, "%.2f")
    if rv then S.temperature = nv end
  end

  -- GPU (siempre visible, pero nota si no aplica)
  g.row_label("GPU:", lw2)
  local gpu_relevant = MG_GPU_RELEVANT[mk]
  if not gpu_relevant then g.begin_disabled(true) end
  g.next_width(t.sc(90))
  local gpu_list = MG_GPUS
  -- Buscar índice actual
  local gpu_idx = 1
  for i, v in ipairs(gpu_list) do if v == S.gpu then gpu_idx = i; break end end
  local new_gpu_idx = widgets.combo("##mg_gpu", gpu_idx, gpu_list)
  if new_gpu_idx ~= gpu_idx then S.gpu = gpu_list[new_gpu_idx] end
  if not gpu_relevant then
    g.end_disabled()
    g.same_line(t.sc(10))
    g.text_disabled("(fijada por el modelo)")
  end

  -- Hint de coste/aviso
  g.spacing()
  local hints = {
    amadeus      = "A10G: ~$0.05/min  |  multi-track, duración variable",
    midi_llm     = "A10G: ~$0.05/min  |  requiere CUDA BF16 (no MPS)",
    text2midi    = "A10G: ~$0.05/min  |  baseline académico, calidad baja",
    chatmusician = "A10G: ~$0.05/min  |  multi-voz ABC; a veces genera texto en lugar de ABC",
    musecoco     = "A100: ~$0.14/min  |  stage-2 tarda ~11 min, usa spawn+poll",
    anticipatory = "A10G: ~$0.05/min  |  ~15-45 min en CPU → Modal necesario",
  }
  g.text_disabled(hints[mk] or "")

  g.spacing(); g.separator(); g.spacing()

  -- ── BOTÓN GENERAR ──────────────────────────────────────────────
  local btn_color = {
    norm   = { 0x1A/255, 0x7A/255, 0x3C/255 },
    hover  = { 0x22/255, 0x99/255, 0x4D/255 },
    active = { 0x2A/255, 0xB5/255, 0x5C/255 },
  }
  local btn_lbl = S.running and "[ Generando... ]" or "GENERAR MIDI"
  g.begin_disabled(S.running)
  g.next_width(-1)
  if g.button(btn_lbl, nil, t.sc(36), { solid = btn_color }) then
    launch_generate()
  end
  g.end_disabled()
  g.spacing()

  -- ── PROGRESO ───────────────────────────────────────────────────
  local pct_str = string.format("%d%%", math.floor(S.progress * 100))
  g.progress_bar(S.progress, nil, t.sc(16), pct_str)

  local status_color = S.running and "YELLOW"
    or (S.done and #S.out_files > 0 and "GREEN")
    or (S.done and "RED")
    or "FG_DIM"
  g.text_colored(S.status:sub(1, 90), status_color)
  g.spacing()

  -- ── LOG ────────────────────────────────────────────────────────
  if widgets.collapsing_header("Logs", true) then
    if g.button("Copiar log", t.sc(90), t.ITEM_H) then
      local ok2, _ = pcall(function()
        reaper.CF_SetClipboard(table.concat(S.log, "\n"))
      end)
      if not ok2 then reaper.ShowConsoleMsg(table.concat(S.log, "\n") .. "\n") end
    end
    g.same_line()
    if g.button("Limpiar", t.sc(70), t.ITEM_H) then S.log = {} end
    g.spacing()

    if S.log_scroll_to_bottom then
      widgets.scroll_to_bottom("##mg_logscroll")
      S.log_scroll_to_bottom = false
    end

    g.push_font(t.F.MONO)
    local log_h = math.max(t.sc(60), gfx.h - gui.ctx.y - t.PAD_Y - t.sc(10))
    widgets.scroll_region("##mg_logscroll", 0, log_h, function()
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
add_log("MidiGenerator listo.")
add_log("Python: " .. PYTHON)
add_log("midigen.py: " .. MIDIGEN_PY)
launch_setup_check()
reaper.defer(loop)
