-- @description Stem Separator - Demucs + SAM Audio
-- @version 2.0
-- @author IAClaude
-- @about Separacion de stems con Demucs (local) y SAM Audio (Modal cloud).
--        Requiere la extension ReaImGui (instalar via ReaPack).

-- ── CHECK EXTENSIÓN ────────────────────────────────────────────
if not reaper.ImGui_GetVersion then
  local _info   = debug.getinfo(1, "S")
  local _dir    = _info.source:match("@?(.*[/\\])") or ""
  local _helper = _dir .. "setup_helpers.py"
  local _tmpdir = os.getenv("TMPDIR") or "/tmp/"
  local _pf     = _tmpdir .. "stemsep_reaimgui_install.txt"

  if reaper.APIExists("ReaPack_BrowsePackages") then
    -- ReaPack disponible: abre el browser ya filtrado (2 clics para instalar)
    reaper.ReaPack_BrowsePackages("ReaImGui")
    reaper.MB(
      "ReaImGui no está instalado.\n\n" ..
      "Hemos abierto ReaPack filtrado por 'ReaImGui'.\n\n" ..
      "  1. Selecciona 'ReaImGui' de cfillion\n" ..
      "  2. Clic derecho → Install latest version\n" ..
      "  3. Apply\n" ..
      "  4. Reinicia REAPER y vuelve a abrir el script.",
      "Stem Separator — Instalar ReaImGui", 0)
  else
    -- Sin ReaPack: descarga directa desde GitHub (síncrono, ~5-10 s)
    reaper.MB("ReaImGui no encontrado. Descargando desde GitHub, espera unos segundos...",
      "Stem Separator — ReaImGui", 0)
    os.execute(string.format('python3 "%s" install-reaimgui --progress "%s"', _helper, _pf))
    local _msg = "Revisa ~/Library/Application Support/REAPER/UserPlugins/"
    local _f = io.open(_pf, "r")
    if _f then
      local _raw = _f:read("*a"); _f:close()
      _msg = _raw:match("|[^|]*|(.+)$") or _msg
    end
    reaper.MB(_msg .. "\n\nReinicia REAPER para activar ReaImGui.",
      "Stem Separator — ReaImGui", 0)
  end
  return
end

-- ── RUTAS ──────────────────────────────────────────────────────
local info       = debug.getinfo(1, "S")
local SCRIPT_DIR = info.source:match("@?(.*[/\\])") or ""

local HOME       = os.getenv("HOME") or ""
local TMPDIR     = os.getenv("TMPDIR") or "/tmp/"
local SAM_DIR    = SCRIPT_DIR
local SAM_SCRIPT = "modal_sam_audio.py"
local DEMUCS_PY  = SCRIPT_DIR .. "separate_demucs.py"
local SAM_PY     = SCRIPT_DIR .. "separate_sam.py"
local PROGRESS_F = TMPDIR .. "stemsep_progress.txt"
local LOG_F      = TMPDIR .. "stemsep.log"
local REAPER_INI = reaper.GetResourcePath() .. "/reaper.ini"

-- ── PYTHON DETECTION ───────────────────────────────────────────
local function detect_reaper_python()
  local f = io.open(REAPER_INI, "r")
  if not f then return nil, "Cannot open " .. REAPER_INI end
  local libpath
  for line in f:lines() do
    local key, val = line:match("^(pythonlibpath64)=(.*)$")
    if not key then key, val = line:match("^(pythonlibpath32)=(.*)$") end
    if key and val ~= "" then libpath = val; break end
  end
  f:close()
  if not libpath then return nil, "pythonlibpath not found in reaper.ini" end
  local parent = libpath:match("^(.*)/lib$")
  if not parent then return nil, "Unexpected libpath format: " .. libpath end
  local exe = parent .. "/bin/python3"
  if not io.open(exe, "r") then return nil, "Detected python not found: " .. exe end
  return exe, nil
end

local PYTHON, PYTHON_ERR = detect_reaper_python()
if not PYTHON then
  reaper.ShowConsoleMsg("Stem Separator - ERROR: " .. PYTHON_ERR .. "\n")
  PYTHON = "python3"
end

-- ── CONSTANTES ─────────────────────────────────────────────────
local DM_MODELS = { "htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra" }
local DM_LABELS = {
  "htdemucs  (4 stems)",
  "htdemucs_ft  (4 stems, fine-tuned)",
  "htdemucs_6s  (6 stems)",
  "mdx_extra  (4 stems, MDX-Net)",
}
local STEM_KEYS  = { "vocals", "drums", "bass", "other", "guitar", "piano" }
local STEM_NAMES = { vocals="Vocales", drums="Bateria", bass="Bajo",
                     other="Otros", guitar="Guitarra*", piano="Piano*" }
local SAM_MODELS = { "facebook/sam-audio-large", "facebook/sam-audio-base" }
local SAM_GPUS   = { "A100", "A10G", "T4" }
local ODE_METHODS= { "midpoint", "euler", "rk4" }

-- ── ESTADO ─────────────────────────────────────────────────────
local S = {
  tab            = 1,
  src            = "",
  src_track_name = "",
  src_track_idx  = -1,
  outdir         = HOME .. "/stems",
  -- Demucs
  dm_idx         = 1,
  dm_stems       = { vocals=true, drums=true, bass=true, other=true,
                     guitar=false, piano=false },
  -- SAM
  sam_prompt     = "jazz trumpet",
  sam_midx       = 1,
  sam_gidx       = 1,
  sam_oidx       = 1,
  sam_steps      = 64,
  sam_chunk      = 15.0,
  sam_overlap    = 2.0,
  sam_conf       = 0.0,
  sam_cands      = 1,
  -- runtime
  running        = false,
  done           = false,
  progress       = 0.0,
  status         = "Listo.",
  log            = {},
  out_files      = {},
  log_scroll_to_bottom = false,
}

-- ── HELPERS CORE ───────────────────────────────────────────────
local function add_log(s)
  table.insert(S.log, tostring(s):sub(1, 200))
  if #S.log > 200 then table.remove(S.log, 1) end
  S.log_scroll_to_bottom = true
end

local function q(s)  -- shell quoting
  return '"' .. s:gsub('"', '\\"') .. '"'
end

-- ── SETUP CHECK (asíncrono, solo al inicio) ───────────────────
local SETUP_CHECK_F = TMPDIR .. "stemsep_setup_check.txt"
local SETUP_HELPER  = SCRIPT_DIR .. "setup_helpers.py"
local setup_missing = {}
local setup_checked = false

local function launch_setup_check()
  local f = io.open(SETUP_CHECK_F, "w")
  if f then f:write("running|0.00|..."); f:close() end
  local cmd = string.format('%s %s check --progress %s >> %s 2>&1 &',
    q(PYTHON), q(SETUP_HELPER), q(SETUP_CHECK_F),
    q(TMPDIR .. "stemsep_setup.log"))
  os.execute(cmd)
end

local function poll_setup_check()
  if setup_checked then return end
  local f = io.open(SETUP_CHECK_F, "r")
  if not f then return end
  local raw = f:read("*a"); f:close()
  if raw:match("^done|") == nil then return end
  setup_checked = true
  local LABELS = {
    python         = "Python REAPER",
    uv             = "uv",
    demucs         = "demucs",
    ["modal-cli"]  = "Modal CLI",
    ["modal-auth"] = "Modal sin auth",
    ["hf-secret"]  = "HF secret faltante",
  }
  for line in (raw .. "\n"):gmatch("([^\n]*)\n") do
    local name, status = line:match("^CHECK|([^|]+)|([^|]+)|")
    if name and status == "missing" then
      table.insert(setup_missing, LABELS[name] or name)
    end
  end
end

-- ── LECTURA DE PROGRESO ────────────────────────────────────────
local function read_progress()
  local f = io.open(PROGRESS_F, "r")
  if not f then return end
  local content = f:read("*a"); f:close()
  if content == "" then return end

  local lines = {}
  for line in (content .. "\n"):gmatch("([^\n]*)\n") do
    if line ~= "" then table.insert(lines, line) end
  end
  if #lines == 0 then return end

  local state_s, pct_s, msg = lines[1]:match("^([^|]+)|([^|]+)|(.+)$")
  if not state_s then return end

  local pct = tonumber(pct_s) or S.progress
  S.progress = pct
  if msg ~= S.status then
    S.status = msg
    add_log(msg)
  end

  if state_s == "done" and not S.done then
    S.running   = false
    S.done      = true
    S.out_files = {}
    for i = 2, #lines do
      local p = lines[i]:match("^%s*(.-)%s*$")
      if p ~= "" then table.insert(S.out_files, p) end
    end
    if #S.out_files > 0 then
      add_log("Archivos listos: " .. #S.out_files)
      import_stems()
    end
  elseif state_s == "error" and not S.done then
    S.running = false
    S.done    = true
    add_log("ERROR: " .. (msg or "?"))
  end
end

-- ── INTEGRACIÓN CON REAPER ─────────────────────────────────────
local function grab_from_reaper()
  local tcnt = reaper.CountSelectedTracks(0)
  if tcnt > 0 then
    local tr = reaper.GetSelectedTrack(0, 0)
    local _, tname = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
    S.src_track_name = tname ~= "" and tname
      or ("Track " .. (reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER") or "?"))
    S.src_track_idx = reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER")
    local icnt = reaper.CountTrackMediaItems(tr)
    for i = 0, icnt - 1 do
      local item = reaper.GetTrackMediaItem(tr, i)
      local take = reaper.GetActiveTake(item)
      if take then
        local src   = reaper.GetMediaItemTake_Source(take)
        local fname = reaper.GetMediaSourceFileName(src, "")
        if fname and fname ~= "" then
          S.src = fname
          add_log("Fuente (pista): " .. S.src_track_name .. " | " .. fname)
          return
        end
      end
    end
    reaper.MB("La pista seleccionada no tiene items de audio activos.", "Stem Separator", 0)
    return
  end

  local n = reaper.CountSelectedMediaItems(0)
  if n == 0 then
    reaper.MB("No hay ningun item ni pista seleccionada en Reaper.", "Stem Separator", 0); return
  end
  local item = reaper.GetSelectedMediaItem(0, 0)
  local take  = reaper.GetActiveTake(item)
  if not take then
    reaper.MB("El item no tiene take activo.", "Stem Separator", 0); return
  end
  local src   = reaper.GetMediaItemTake_Source(take)
  local fname = reaper.GetMediaSourceFileName(src, "")
  if fname and fname ~= "" then
    S.src = fname; S.src_track_name = ""; S.src_track_idx = -1
    add_log("Fuente: " .. fname)
  end
end

function import_stems()
  if #S.out_files == 0 then return end
  reaper.Undo_BeginBlock()
  local cursor  = reaper.GetCursorPosition()
  local imported = 0

  local folder_name = S.src_track_name
  if folder_name == "" then
    local base = S.src:match("([^/\\]+)%.%w+$")
    folder_name = (base or "stems") .. " [stems]"
  end

  local tcnt = reaper.CountTracks(0)
  reaper.InsertTrackAtIndex(tcnt, true)
  local folder_tr = reaper.GetTrack(0, tcnt)
  reaper.GetSetMediaTrackInfo_String(folder_tr, "P_NAME", folder_name, true)
  reaper.SetMediaTrackInfo_Value(folder_tr, "I_FOLDERDEPTH", 1)

  for _, fp in ipairs(S.out_files) do
    local f = io.open(fp, "rb")
    if f then
      f:close()
      local tidx = reaper.CountTracks(0)
      reaper.InsertTrackAtIndex(tidx, true)
      local track      = reaper.GetTrack(0, tidx)
      local stem_name  = fp:match("([^/\\]+)%.wav$") or fp:match("([^/\\]+)$")
      local track_name = folder_name .. " - " .. (stem_name or "stem")
      reaper.GetSetMediaTrackInfo_String(track, "P_NAME", track_name, true)
      reaper.SetOnlyTrackSelected(track)
      reaper.SetEditCurPos(cursor, false, false)
      reaper.InsertMedia(fp, 0)
      imported = imported + 1
      add_log("Importado: " .. track_name)
    else
      add_log("No encontrado: " .. fp)
    end
  end

  if imported > 0 then
    local last_tr = reaper.GetTrack(0, reaper.CountTracks(0) - 1)
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
  add_log("Importados " .. imported .. " stems en carpeta '" .. folder_name .. "'")
end

-- ── LANZAR PROCESOS ────────────────────────────────────────────
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

local function launch_demucs()
  if S.src == "" then
    reaper.MB("Selecciona un archivo de audio primero.", "Stem Separator", 0); return
  end
  local stems = {}
  for _, k in ipairs(STEM_KEYS) do
    if S.dm_stems[k] then table.insert(stems, k) end
  end
  if #stems == 0 then
    reaper.MB("Selecciona al menos un stem.", "Stem Separator", 0); return
  end
  local model = DM_MODELS[S.dm_idx]
  clear_run("Iniciando Demucs (" .. model .. ")...")
  add_log("Modelo: " .. model .. " | Stems: " .. table.concat(stems, ", "))
  local cmd = string.format(
    '%s %s --input %s --model %s --stems %s --outdir %s --python %s --progress %s >> %s 2>&1 &',
    q(PYTHON), q(DEMUCS_PY),
    q(S.src), q(model), table.concat(stems, ","),
    q(S.outdir), q(PYTHON), q(PROGRESS_F), q(LOG_F))
  add_log("Lanzando proceso...")
  os.execute(cmd)
end

local function launch_sam()
  if S.src == "" then
    reaper.MB("Selecciona un archivo de audio primero.", "Stem Separator", 0); return
  end
  if S.sam_prompt == "" then
    reaper.MB("Escribe un prompt para SAM Audio.", "Stem Separator", 0); return
  end
  local sam_script_path = SAM_DIR .. "/" .. SAM_SCRIPT
  local f = io.open(sam_script_path, "r")
  if not f then
    reaper.MB("No encontrado: " .. sam_script_path ..
      "\n\nEl plugin debe contener modal_sam_audio.py y pyproject.toml.\n" ..
      "Revisa la instalacion del plugin StemsSeparator.",
      "Stem Separator", 0)
    return
  end
  f:close()
  clear_run("Iniciando SAM Audio via Modal...")
  add_log("Prompt: " .. S.sam_prompt)
  add_log("Modelo: " .. SAM_MODELS[S.sam_midx] .. " | GPU: " .. SAM_GPUS[S.sam_gidx])
  local cmd = string.format(
    '%s %s --sam-dir %s --input %s --prompt %s --model %s --gpu %s' ..
    ' --steps %d --ode-method %s --chunk %.1f --overlap %.1f' ..
    ' --confidence %.2f --candidates %d --outdir %s --progress %s >> %s 2>&1 &',
    q(PYTHON), q(SAM_PY), q(SAM_DIR),
    q(S.src), q(S.sam_prompt),
    q(SAM_MODELS[S.sam_midx]), q(SAM_GPUS[S.sam_gidx]),
    S.sam_steps, ODE_METHODS[S.sam_oidx],
    S.sam_chunk, S.sam_overlap, S.sam_conf, S.sam_cands,
    q(S.outdir), q(PROGRESS_F), q(LOG_F))
  add_log("Lanzando proceso Modal...")
  os.execute(cmd)
end

-- ── IMGUI: CONTEXTO Y FUENTES ──────────────────────────────────
local ctx       = reaper.ImGui_CreateContext('Stem Separator')
local font_ui   = reaper.ImGui_CreateFont('sans-serif', 14)
local font_h1   = reaper.ImGui_CreateFont('sans-serif', 18)
local font_mono = reaper.ImGui_CreateFont('monospace', 13)
reaper.ImGui_Attach(ctx, font_ui)
reaper.ImGui_Attach(ctx, font_h1)
reaper.ImGui_Attach(ctx, font_mono)

-- ── COLORES (RGBA 0xRRGGBBAA) ──────────────────────────────────
local C_GREEN  = 0x40B261FF
local C_RED    = 0xD94238FF
local C_YELLOW = 0xF2B81AFF
local C_FG2    = 0x848491FF
local C_LOG    = 0xB4B4BEFF

-- ── COMBO WRAPPER (estado 1-indexed → ImGui 0-indexed) ─────────
local function combo1(label, idx, tbl)
  local items = table.concat(tbl, "\0") .. "\0"
  local rv, new0 = reaper.ImGui_Combo(ctx, label, idx - 1, items)
  if rv then return new0 + 1 end
  return idx
end

-- ── TAB DEMUCS ─────────────────────────────────────────────────
local function draw_demucs_tab()
  reaper.ImGui_AlignTextToFramePadding(ctx)
  reaper.ImGui_Text(ctx, 'Modelo:')
  reaper.ImGui_SameLine(ctx)
  reaper.ImGui_SetNextItemWidth(ctx, -1)
  S.dm_idx = combo1('##dm_model', S.dm_idx, DM_LABELS)

  reaper.ImGui_Spacing(ctx)
  reaper.ImGui_Text(ctx, 'Stems:')
  reaper.ImGui_Spacing(ctx)

  local is6s = DM_MODELS[S.dm_idx] == 'htdemucs_6s'

  for i, k in ipairs({'vocals', 'drums', 'bass', 'other'}) do
    if i > 1 then reaper.ImGui_SameLine(ctx, 0, 22) end
    local rv, nv = reaper.ImGui_Checkbox(ctx, STEM_NAMES[k] .. '##' .. k, S.dm_stems[k])
    if rv then S.dm_stems[k] = nv end
  end

  reaper.ImGui_BeginDisabled(ctx, not is6s)
  local rv, nv = reaper.ImGui_Checkbox(ctx, STEM_NAMES.guitar .. '##guitar', S.dm_stems.guitar)
  if rv then S.dm_stems.guitar = nv end
  reaper.ImGui_SameLine(ctx, 0, 22)
  rv, nv = reaper.ImGui_Checkbox(ctx, STEM_NAMES.piano .. '##piano', S.dm_stems.piano)
  if rv then S.dm_stems.piano = nv end
  reaper.ImGui_EndDisabled(ctx)
  if not is6s then
    reaper.ImGui_SameLine(ctx)
    reaper.ImGui_TextDisabled(ctx, '  (solo htdemucs_6s)')
  end

  reaper.ImGui_Spacing(ctx)
  if reaper.ImGui_Button(ctx, 'Todos') then
    for _, k in ipairs(STEM_KEYS) do S.dm_stems[k] = true end
  end
  reaper.ImGui_SameLine(ctx)
  if reaper.ImGui_Button(ctx, 'Ninguno') then
    for _, k in ipairs(STEM_KEYS) do S.dm_stems[k] = false end
  end
end

-- ── TAB SAM AUDIO ──────────────────────────────────────────────
local function draw_sam_tab()
  local tbl_flags = 0
  if reaper.ImGui_BeginTable(ctx, '##samtbl', 2, tbl_flags, 0, 0) then
    reaper.ImGui_TableSetupColumn(ctx, '##lbl',
      reaper.ImGui_TableColumnFlags_WidthFixed(), 90)
    reaper.ImGui_TableSetupColumn(ctx, '##wgt',
      reaper.ImGui_TableColumnFlags_WidthStretch())

    -- Prompt ─────────────────────────────────────────────────────
    reaper.ImGui_TableNextRow(ctx)
    reaper.ImGui_TableSetColumnIndex(ctx, 0)
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'Prompt:')
    reaper.ImGui_TableSetColumnIndex(ctx, 1)
    reaper.ImGui_SetNextItemWidth(ctx, -1)
    local rv, new_val = reaper.ImGui_InputText(ctx, '##prompt', S.sam_prompt,
      reaper.ImGui_InputTextFlags_AutoSelectAll())
    if rv then S.sam_prompt = new_val end

    -- Modelo ─────────────────────────────────────────────────────
    reaper.ImGui_TableNextRow(ctx)
    reaper.ImGui_TableSetColumnIndex(ctx, 0)
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'Modelo:')
    reaper.ImGui_TableSetColumnIndex(ctx, 1)
    reaper.ImGui_SetNextItemWidth(ctx, -1)
    S.sam_midx = combo1('##sam_model', S.sam_midx, SAM_MODELS)

    -- GPU + ODE method (mismo row) ───────────────────────────────
    reaper.ImGui_TableNextRow(ctx)
    reaper.ImGui_TableSetColumnIndex(ctx, 0)
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'GPU:')
    reaper.ImGui_TableSetColumnIndex(ctx, 1)
    reaper.ImGui_SetNextItemWidth(ctx, 90)
    S.sam_gidx = combo1('##sam_gpu', S.sam_gidx, SAM_GPUS)
    reaper.ImGui_SameLine(ctx, 0, 14)
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'ODE:')
    reaper.ImGui_SameLine(ctx, 0, 6)
    reaper.ImGui_SetNextItemWidth(ctx, -1)
    S.sam_oidx = combo1('##sam_ode', S.sam_oidx, ODE_METHODS)

    -- ODE steps + Confianza (mismo row) ─────────────────────────
    reaper.ImGui_TableNextRow(ctx)
    reaper.ImGui_TableSetColumnIndex(ctx, 0)
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'ODE steps:')
    reaper.ImGui_TableSetColumnIndex(ctx, 1)
    reaper.ImGui_SetNextItemWidth(ctx, 120)
    rv, new_val = reaper.ImGui_SliderInt(ctx, '##steps', S.sam_steps, 1, 128)
    if rv then S.sam_steps = new_val end
    reaper.ImGui_SameLine(ctx, 0, 14)
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'Conf.:')
    reaper.ImGui_SameLine(ctx, 0, 6)
    reaper.ImGui_SetNextItemWidth(ctx, -1)
    rv, new_val = reaper.ImGui_SliderDouble(ctx, '##conf', S.sam_conf, 0.0, 1.0, '%.2f')
    if rv then S.sam_conf = new_val end

    -- Chunk + Overlap + Candidatos (mismo row) ───────────────────
    reaper.ImGui_TableNextRow(ctx)
    reaper.ImGui_TableSetColumnIndex(ctx, 0)
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'Chunk s:')
    reaper.ImGui_TableSetColumnIndex(ctx, 1)
    reaper.ImGui_SetNextItemWidth(ctx, 90)
    rv, new_val = reaper.ImGui_SliderDouble(ctx, '##chunk', S.sam_chunk, 1.0, 30.0, '%.1f')
    if rv then S.sam_chunk = new_val end
    reaper.ImGui_SameLine(ctx, 0, 14)
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'Overlap:')
    reaper.ImGui_SameLine(ctx, 0, 6)
    reaper.ImGui_SetNextItemWidth(ctx, 80)
    rv, new_val = reaper.ImGui_SliderDouble(ctx, '##overlap', S.sam_overlap, 0.0, 10.0, '%.1f')
    if rv then S.sam_overlap = new_val end
    reaper.ImGui_SameLine(ctx, 0, 14)
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'Cand.:')
    reaper.ImGui_SameLine(ctx, 0, 6)
    reaper.ImGui_SetNextItemWidth(ctx, -1)
    rv, new_val = reaper.ImGui_SliderInt(ctx, '##cands', S.sam_cands, 1, 8)
    if rv then S.sam_cands = new_val end

    reaper.ImGui_EndTable(ctx)
  end

  reaper.ImGui_Spacing(ctx)
  reaper.ImGui_TextDisabled(ctx,
    'A100: ~$0.14/pista  |  A10G: ~$0.09/pista  |  Requiere cuenta Modal.com')
end

-- ── LOOP PRINCIPAL ─────────────────────────────────────────────
local function loop()
  -- Colores del tema oscuro
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_WindowBg(),         0x1A1A1CFF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_TitleBg(),          0x232327FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_TitleBgActive(),    0x2E2E35FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_FrameBg(),          0x2E2E38FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_FrameBgHovered(),   0x3C3C48FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_FrameBgActive(),    0x4A4A58FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_CheckMark(),        0x4799FFFF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_SliderGrab(),       0x4799FFFF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_SliderGrabActive(), 0x6AADFFFF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_Button(),           0x4799FF30)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_ButtonHovered(),    0x4799FF70)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_ButtonActive(),     0x4799FFAA)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_PopupBg(),          0x252528FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_Separator(),        0x3A3A44FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_Text(),             0xEBEBEBFF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_TextDisabled(),     0x848491FF)
  local N_COL = 16

  -- Bordes redondeados
  reaper.ImGui_PushStyleVar(ctx, reaper.ImGui_StyleVar_FrameRounding(),  4.0)
  reaper.ImGui_PushStyleVar(ctx, reaper.ImGui_StyleVar_GrabRounding(),   4.0)
  reaper.ImGui_PushStyleVar(ctx, reaper.ImGui_StyleVar_WindowRounding(), 6.0)
  local N_VAR = 3

  reaper.ImGui_PushFont(ctx, font_ui, 14)
  reaper.ImGui_SetNextWindowSize(ctx, 540, 720, reaper.ImGui_Cond_FirstUseEver())

  local visible, open = reaper.ImGui_Begin(ctx, 'Stem Separator##ss', true)

  if visible then

    -- SETUP BANNER (se rellena en segundo plano; desaparece si todo está OK)
    poll_setup_check()
    if setup_checked and #setup_missing > 0 then
      reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_Text(), C_YELLOW)
      reaper.ImGui_TextWrapped(ctx, '⚠  Config. incompleta: ' .. table.concat(setup_missing, ' · '))
      reaper.ImGui_PopStyleColor(ctx)
      reaper.ImGui_TextDisabled(ctx, 'Carga Setup.lua en Actions > Load ReaScript para configurar.')
      reaper.ImGui_Spacing(ctx)
    end

    -- HEADER ──────────────────────────────────────────────────────
    reaper.ImGui_PushFont(ctx, font_h1, 18)
    reaper.ImGui_Text(ctx, 'Stem Separator')
    reaper.ImGui_PopFont(ctx)
    reaper.ImGui_SameLine(ctx, 0, 10)
    local dot_c = (reaper.CountTracks(0) >= 0) and C_GREEN or C_RED
    reaper.ImGui_TextColored(ctx, dot_c, '● Reaper OK')
    reaper.ImGui_Separator(ctx)
    reaper.ImGui_Spacing(ctx)

    -- FUENTE ──────────────────────────────────────────────────────
    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'Fuente:')
    reaper.ImGui_SameLine(ctx)
    local display_src = (S.src_track_name ~= "")
      and (S.src_track_name .. '  (' .. (S.src:match('[^/\\]+$') or '') .. ')')
      or S.src
    reaper.ImGui_SetNextItemWidth(ctx, -95)
    reaper.ImGui_InputText(ctx, '##src_disp', display_src,
      reaper.ImGui_InputTextFlags_ReadOnly())
    reaper.ImGui_SameLine(ctx)
    if reaper.ImGui_Button(ctx, '...', 42, 0) then
      local ok, fn = reaper.GetUserFileNameForRead("", "Abrir audio", "wav")
      if ok then S.src = fn; S.src_track_name = ""; S.src_track_idx = -1 end
    end
    reaper.ImGui_SameLine(ctx)
    if reaper.ImGui_Button(ctx, 'R', 42, 0) then grab_from_reaper() end

    if S.src_track_name ~= "" then
      reaper.ImGui_TextDisabled(ctx, 'Pista seleccionada  |  clic en R para actualizar')
    else
      reaper.ImGui_TextDisabled(ctx, 'Clic en R para usar pista/item activo de Reaper')
    end
    reaper.ImGui_Spacing(ctx)

    -- TABS ────────────────────────────────────────────────────────
    if reaper.ImGui_BeginTabBar(ctx, '##maintabs') then
      if reaper.ImGui_BeginTabItem(ctx, 'DEMUCS  (local)') then
        S.tab = 1
        reaper.ImGui_Spacing(ctx)
        draw_demucs_tab()
        reaper.ImGui_EndTabItem(ctx)
      end
      if reaper.ImGui_BeginTabItem(ctx, 'SAM AUDIO  (cloud)') then
        S.tab = 2
        reaper.ImGui_Spacing(ctx)
        draw_sam_tab()
        reaper.ImGui_EndTabItem(ctx)
      end
      reaper.ImGui_EndTabBar(ctx)
    end

    reaper.ImGui_Spacing(ctx)
    reaper.ImGui_Separator(ctx)
    reaper.ImGui_Spacing(ctx)

    -- BOTÓN SEPARAR ───────────────────────────────────────────────
    local btn_c = S.tab == 1 and 0x2966B0CC or 0x4D19C4CC
    local btn_h = S.tab == 1 and 0x3D80D8CC or 0x6626E0CC
    local btn_a = S.tab == 1 and 0x4799FFCC or 0x8033D1CC
    reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_Button(),        btn_c)
    reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_ButtonHovered(), btn_h)
    reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_ButtonActive(),  btn_a)
    local sep_lbl = S.running and '[ Procesando... ]'
      or (S.tab == 1 and 'SEPARAR  (Demucs)' or 'SEPARAR  (SAM Audio)')
    reaper.ImGui_BeginDisabled(ctx, S.running)
    if reaper.ImGui_Button(ctx, sep_lbl, -1, 36) then
      if S.tab == 1 then launch_demucs() else launch_sam() end
    end
    reaper.ImGui_EndDisabled(ctx)
    reaper.ImGui_PopStyleColor(ctx, 3)

    reaper.ImGui_Spacing(ctx)

    -- BARRA DE PROGRESO ───────────────────────────────────────────
    local pct_str = string.format('%d%%', math.floor(S.progress * 100))
    reaper.ImGui_ProgressBar(ctx, S.progress, -1, 16, pct_str)

    local msg_c
    if     S.running                       then msg_c = C_YELLOW
    elseif S.done and #S.out_files > 0     then msg_c = C_GREEN
    elseif S.done                          then msg_c = C_RED
    else                                        msg_c = C_FG2  end
    reaper.ImGui_TextColored(ctx, msg_c, S.status:sub(1, 90))

    reaper.ImGui_Spacing(ctx)

    -- ÁREA DE LOGS (colapsable) ───────────────────────────────────
    if reaper.ImGui_CollapsingHeader(ctx, '  Logs##loghdr',
        reaper.ImGui_TreeNodeFlags_DefaultOpen()) then

      if reaper.ImGui_Button(ctx, 'Copiar log') then
        reaper.ImGui_SetClipboardText(ctx, table.concat(S.log, '\n'))
      end
      reaper.ImGui_SameLine(ctx)
      if reaper.ImGui_Button(ctx, 'Limpiar') then
        S.log = {}
      end

      reaper.ImGui_PushFont(ctx, font_mono, 13)
      reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_ChildBg(), 0x111116FF)
      local _, avail_h = reaper.ImGui_GetContentRegionAvail(ctx)
      local log_h = math.max(60, avail_h - 4)
      reaper.ImGui_BeginChild(ctx, '##logscroll', 0, log_h,
        reaper.ImGui_ChildFlags_Borders())
      reaper.ImGui_PushStyleVar(ctx, reaper.ImGui_StyleVar_ItemSpacing(), 0, 2)
      for i = 1, #S.log do
        local ln = S.log[i]
        if ln:find('^ERROR') then
          reaper.ImGui_TextColored(ctx, C_RED, ln)
        else
          reaper.ImGui_TextColored(ctx, C_LOG, ln)
        end
      end
      reaper.ImGui_PopStyleVar(ctx)
      if S.log_scroll_to_bottom then
        reaper.ImGui_SetScrollHereY(ctx, 1.0)
        S.log_scroll_to_bottom = false
      end
      reaper.ImGui_EndChild(ctx)
      reaper.ImGui_PopStyleColor(ctx)
      reaper.ImGui_PopFont(ctx)
    end

  reaper.ImGui_End(ctx)

  end -- if visible

  reaper.ImGui_PopStyleVar(ctx,   N_VAR)
  reaper.ImGui_PopStyleColor(ctx, N_COL)
  reaper.ImGui_PopFont(ctx)

  if S.running then read_progress() end

  if open then reaper.defer(loop) end
end

-- ── INICIO ─────────────────────────────────────────────────────
add_log("Stem Separator listo.")
add_log("Demucs: " .. PYTHON)
add_log("SAM: " .. SAM_DIR)
launch_setup_check()
reaper.defer(loop)
