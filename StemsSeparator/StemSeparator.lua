-- @description Stem Separator - Demucs + SAM Audio
-- @version 1.0
-- @author IAClaude
-- @about Separacion de stems con Demucs (local) y SAM Audio (Modal cloud).

-- ── RUTAS ────────────────────────────────────────────────────
local info = debug.getinfo(1, "S")
local SCRIPT_DIR = info.source:match("@?(.*[/\\])") or ""

local HOME        = os.getenv("HOME") or ""
local TMPDIR      = os.getenv("TMPDIR") or "/tmp/"
local SAM_DIR     = SCRIPT_DIR
local SAM_SCRIPT  = "modal_sam_audio.py"
--         ^-- SAM_DIR es el directorio del plugin (StemsSeparator/),
--             que contiene pyproject.toml, modal_sam_audio.py, etc.
local DEMUCS_PY   = SCRIPT_DIR .. "separate_demucs.py"
local SAM_PY      = SCRIPT_DIR .. "separate_sam.py"
local PROGRESS_F  = TMPDIR .. "stemsep_progress.txt"
local LOG_F       = TMPDIR .. "stemsep.log"
local REAPER_INI  = reaper.GetResourcePath() .. "/reaper.ini"

-- ── PYTHON DETECTION ──────────────────────────────────────────
local function detect_reaper_python()
  -- Read pythonlibpath from reaper.ini to find the Python REAPER uses.
  local f = io.open(REAPER_INI, "r")
  if not f then return nil, "Cannot open " .. REAPER_INI end
  local libpath
  for line in f:lines() do
    local key, val = line:match("^(pythonlibpath64)=(.*)$")
    if not key then
      key, val = line:match("^(pythonlibpath32)=(.*)$")
    end
    if key and val ~= "" then
      libpath = val
      break
    end
  end
  f:close()
  if not libpath then
    return nil, "pythonlibpath not found in reaper.ini"
  end
  -- libpath is the lib/ directory, the executable is at ../bin/python3
  local parent = libpath:match("^(.*)/lib$")
  if not parent then
    return nil, "Unexpected libpath format: " .. libpath
  end
  local exe = parent .. "/bin/python3"
  if not io.open(exe, "r") then
    return nil, "Detected python not found: " .. exe
  end
  return exe, nil
end

local PYTHON, PYTHON_ERR = detect_reaper_python()
if not PYTHON then
  reaper.ShowConsoleMsg("Stem Separator - ERROR: " .. PYTHON_ERR .. "\n")
  -- fallback: try python3 on PATH
  PYTHON = "python3"
end

-- ── CONSTANTES ────────────────────────────────────────────────
local DM_MODELS = { "htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra" }
local DM_LABELS = {
  "htdemucs  (4 stems)",
  "htdemucs_ft  (4 stems, fine-tuned)",
  "htdemucs_6s  (6 stems)",
  "mdx_extra  (4 stems, MDX-Net)",
}
local STEM_KEYS   = { "vocals", "drums", "bass", "other", "guitar", "piano" }
local STEM_NAMES  = { vocals="Vocales", drums="Bateria", bass="Bajo",
                      other="Otros", guitar="Guitarra*", piano="Piano*" }
local SAM_MODELS  = { "facebook/sam-audio-large", "facebook/sam-audio-base" }
local SAM_GPUS    = { "A100", "A10G", "T4" }
local ODE_METHODS = { "midpoint", "euler", "rk4" }

-- ── ESTADO ────────────────────────────────────────────────────
local S = {
  tab           = 1,
  src           = "",
  src_track_name = "",
  src_track_idx = -1,
  outdir        = HOME .. "/stems",
  dm_idx        = 1,
  dm_stems   = { vocals=true, drums=true, bass=true, other=true, guitar=false, piano=false },
  sam_prompt = "jazz trumpet",
  sam_midx   = 1,
  sam_gidx   = 1,
  sam_oidx   = 1,
  sam_steps  = 64,
  sam_chunk  = 15,
  sam_overlap= 2,
  sam_conf   = 0.0,
  sam_cands  = 1,
  -- inline editing
  edit_field = nil,
  edit_buf   = "",
  edit_cursor_show = true,
  edit_blink_tick  = 0,
  -- runtime
  running    = false,
  done       = false,
  progress   = 0.0,
  status     = "Listo.",
  log        = {},
  out_files  = {},
}

-- ── MOUSE ─────────────────────────────────────────────────────
local prev_lb     = 0
local clicked     = false   -- true only on frame where LMB pressed
local clicked_r   = false   -- right click

-- ── COLORES (r,g,b en 0-1) ───────────────────────────────────
local BG    = { 0.10, 0.10, 0.11 }
local BG2   = { 0.16, 0.16, 0.18 }
local BG3   = { 0.22, 0.22, 0.25 }
local FG    = { 0.92, 0.92, 0.92 }
local FG2   = { 0.52, 0.52, 0.57 }
local ACC   = { 0.28, 0.60, 1.00 }
local ACC2  = { 0.16, 0.42, 0.76 }
local GRN   = { 0.25, 0.70, 0.38 }
local GRN2  = { 0.16, 0.46, 0.24 }
local RED   = { 0.85, 0.26, 0.22 }
local YLW   = { 0.95, 0.72, 0.10 }
local PRP   = { 0.50, 0.20, 0.82 }
local PRP2  = { 0.30, 0.10, 0.52 }

local function col(c, a) gfx.set(c[1], c[2], c[3], a or 1) end

-- ── FUENTES ──────────────────────────────────────────────────
local function setup_fonts()
  gfx.setfont(1, "Helvetica Neue", 14)                 -- normal
  gfx.setfont(2, "Helvetica Neue", 12)                 -- small
  gfx.setfont(3, "Helvetica Neue", 17, string.byte("b")) -- bold title
  gfx.setfont(4, "Menlo", 11)                          -- monospace log
end

-- ── HELPERS ───────────────────────────────────────────────────
local function in_rect(x, y, w, h)
  return gfx.mouse_x >= x and gfx.mouse_x < x + w
     and gfx.mouse_y >= y and gfx.mouse_y < y + h
end

local function add_log(s)
  table.insert(S.log, tostring(s):sub(1, 100))
  if #S.log > 120 then table.remove(S.log, 1) end
end

local function q(s)   -- quote for shell
  return '"' .. s:gsub('"', '\\"') .. '"'
end

local function ctext(x, y, w, h, s, c, fi)
  gfx.setfont(fi or 1)
  local sw, sh = gfx.measurestr(s)
  col(c or FG)
  gfx.x = x + math.floor((w - sw) / 2)
  gfx.y = y + math.floor((h - sh) / 2)
  gfx.drawstr(s)
end

local function ltext(x, y, s, c, fi)
  gfx.setfont(fi or 1)
  col(c or FG)
  gfx.x, gfx.y = x, y
  gfx.drawstr(s)
end

local function trunc(s, max_w, fi)
  gfx.setfont(fi or 1)
  if gfx.measurestr(s) <= max_w then return s end
  while #s > 3 and gfx.measurestr(s .. "...") > max_w do
    s = s:sub(1, -2)
  end
  return s .. "..."
end

-- ── WIDGETS ───────────────────────────────────────────────────

-- Boton basico — devuelve true si se clico
local function btn(x, y, w, h, label, bg, lc, fi)
  local hover = in_rect(x, y, w, h)
  local bc = bg or BG3
  if hover then
    col({ bc[1] * 1.5, bc[2] * 1.5, bc[3] * 1.5 })
  else
    col(bc)
  end
  gfx.rect(x, y, w, h, 1)
  if hover then col(ACC); gfx.rect(x, y, w, h, 0) end
  ctext(x, y, w, h, label, lc or FG, fi or 1)
  return hover and clicked
end

-- Dropdown con gfx.showmenu — devuelve nuevo indice
local function dropdown(x, y, w, h, labels, idx)
  local hover = in_rect(x, y, w, h)
  col(hover and BG3 or BG2)
  gfx.rect(x, y, w, h, 1)
  if hover then col(ACC); gfx.rect(x, y, w, h, 0) end
  local label = (labels[idx] or "?"):sub(1, 38)
  gfx.setfont(1)
  col(FG)
  gfx.x, gfx.y = x + 6, y + math.floor((h - 13) / 2)
  gfx.drawstr(label .. " v")
  if hover and clicked then
    local menu = ""
    for i, lbl in ipairs(labels) do
      if i == idx then menu = menu .. "!" end
      menu = menu .. lbl
      if i < #labels then menu = menu .. "|" end
    end
    gfx.x, gfx.y = x, y + h
    local r = gfx.showmenu(menu)
    if r > 0 then return r end
  end
  return idx
end

-- Checkbox — devuelve nuevo valor
local function checkbox(x, y, label, val, disabled, w)
  w = w or 130
  local hover = (not disabled) and in_rect(x, y, w, 20)
  -- caja
  col(disabled and BG2 or BG3)
  gfx.rect(x, y + 3, 13, 13, 1)
  if val and not disabled then col(ACC); gfx.rect(x + 2, y + 5, 9, 9, 1) end
  if val and disabled     then col(FG2); gfx.rect(x + 2, y + 5, 9, 9, 1) end
  -- etiqueta
  gfx.setfont(2)
  col(disabled and FG2 or (hover and FG or FG2))
  gfx.x, gfx.y = x + 18, y + 4
  gfx.drawstr(label)
  if hover and clicked then return not val end
  return val
end

-- Barra de progreso
local function progbar(x, y, w, h, pct)
  col(BG3); gfx.rect(x, y, w, h, 1)
  local fw = math.floor(w * math.max(0, math.min(1, pct)))
  if fw > 0 then col(ACC); gfx.rect(x, y, fw, h, 1) end
  local ps = math.floor(pct * 100) .. "%"
  gfx.setfont(2)
  local sw = gfx.measurestr(ps)
  col(FG)
  gfx.x = x + w - sw - 4
  gfx.y = y + math.floor((h - 10) / 2)
  gfx.drawstr(ps)
end

-- Campo de texto editable inline (click para editar, Enter confirma, Escape cancela)
local function field(x, y, w, h, value, hint, id)
  local hover = in_rect(x, y, w, h)
  local is_editing = S.edit_field == id

  if is_editing then
    col({ BG3[1] * 1.3, BG3[2] * 1.3, BG3[3] * 1.3 })
  else
    col(BG3)
  end
  gfx.rect(x, y, w, h, 1)

  if is_editing then
    col(ACC); gfx.rect(x, y, w, h, 0)
  elseif hover then
    col(ACC2); gfx.rect(x, y, w, h, 0)
  end

  gfx.setfont(1)
  if is_editing then
    local display = S.edit_buf
    if S.edit_cursor_show then
      display = display .. "|"
    end
    col(FG)
    gfx.x, gfx.y = x + 5, y + math.floor((h - 13) / 2)
    gfx.drawstr(trunc(display, w - 8, 1))
  else
    local display = value ~= "" and trunc(value, w - 8, 1) or hint
    col(value ~= "" and FG or FG2)
    gfx.x, gfx.y = x + 5, y + math.floor((h - 13) / 2)
    gfx.drawstr(display)
  end

  if hover and clicked then
    if not is_editing then
      if S.edit_field then commit_edit() end
      S.edit_field = id
      S.edit_buf = value
      S.edit_blink_tick = 0
      S.edit_cursor_show = true
    end
  end

  return value
end

local function commit_edit()
  if not S.edit_field then return end
  local id = S.edit_field
  local val = S.edit_buf
  S.edit_field = nil

  if id == "prompt" then
    S.sam_prompt = val ~= "" and val or S.sam_prompt
  elseif id == "steps" then
    local nv = tonumber(val)
    if nv then S.sam_steps = math.floor(math.max(1, nv)) end
  elseif id == "conf" then
    local nv = tonumber(val)
    if nv then S.sam_conf = math.max(0, nv) end
  elseif id == "chunk" then
    local nv = tonumber(val)
    if nv then S.sam_chunk = math.max(1, nv) end
  elseif id == "overlap" then
    local nv = tonumber(val)
    if nv then S.sam_overlap = math.max(0, nv) end
  elseif id == "cands" then
    local nv = tonumber(val)
    if nv then S.sam_cands = math.floor(math.max(1, nv)) end
  end
end

-- ── LECTURA DE PROGRESO ──────────────────────────────────────
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
    S.running  = false
    S.done     = true
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

-- ── INTEGRACION CON REAPER ────────────────────────────────────
local function grab_from_reaper()
  -- try track first
  local tcnt = reaper.CountSelectedTracks(0)
  if tcnt > 0 then
    local tr = reaper.GetSelectedTrack(0, 0)
    local _, tname = reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
    S.src_track_name = tname ~= "" and tname or ("Track " .. (reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER") or "?"))
    S.src_track_idx  = reaper.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER")
    -- find first selected item on this track to use as audio source
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

  -- fallback: selected media item
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
  local cursor = reaper.GetCursorPosition()
  local imported = 0

  -- folder track name: use source track name or default
  local folder_name = S.src_track_name
  if folder_name == "" then
    local base = S.src:match("([^/\\]+)%.%w+$")
    folder_name = (base or "stems") .. " [stems]"
  end

  -- insert folder track at end
  local tcnt = reaper.CountTracks(0)
  reaper.InsertTrackAtIndex(tcnt, true)
  local folder_tr = reaper.GetTrack(0, tcnt)
  reaper.GetSetMediaTrackInfo_String(folder_tr, "P_NAME", folder_name, true)
  reaper.SetMediaTrackInfo_Value(folder_tr, "I_FOLDERDEPTH", 1)  -- open folder

  for _, fp in ipairs(S.out_files) do
    local f = io.open(fp, "rb")
    if f then
      f:close()
      local tidx = reaper.CountTracks(0)
      reaper.InsertTrackAtIndex(tidx, true)
      local track = reaper.GetTrack(0, tidx)
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

  -- close folder on last imported track
  if imported > 0 then
    local last_idx = reaper.CountTracks(0) - 1
    local last_tr  = reaper.GetTrack(0, last_idx)
    reaper.SetMediaTrackInfo_Value(last_tr, "I_FOLDERDEPTH", -1)

    -- rename items to include the stem name
    for i = 0, imported - 1 do
      local tr = reaper.GetTrack(0, tcnt + 1 + i)
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

  -- else: no stems imported -> undo folder track insertion
  if imported == 0 then
    reaper.DeleteTrack(folder_tr)
  end

  reaper.UpdateArrange()
  reaper.Undo_EndBlock("Stem Separator: import stems to folder", -1)
  add_log("Importados " .. imported .. " stems en carpeta '" .. folder_name .. "'")
end

-- ── LANZAR PROCESOS ───────────────────────────────────────────
local function clear_run(label)
  local f = io.open(PROGRESS_F, "w")
  if f then f:write("running|0.00|" .. label); f:close() end
  local lf = io.open(LOG_F, "w")
  if lf then lf:close() end
  S.running  = true
  S.done     = false
  S.progress = 0
  S.out_files = {}
  S.log = {}
  S.status = label
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
    q(S.outdir), q(PYTHON), q(PROGRESS_F), q(LOG_F)
  )
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

  -- Verify modal_sam_audio.py exists before launching
  local sam_script_path = SAM_DIR .. "/" .. SAM_SCRIPT
  local f = io.open(sam_script_path, "r")
  if not f then
    reaper.MB(
      "No encontrado: " .. sam_script_path .. "\n\n"
      .. "El plugin debe contener modal_sam_audio.py y pyproject.toml.\n"
      .. "Revisa la instalacion del plugin StemsSeparator.",
      "Stem Separator", 0
    )
    return
  end
  f:close()

  clear_run("Iniciando SAM Audio via Modal...")
  add_log("Prompt: " .. S.sam_prompt)
  add_log("Modelo: " .. SAM_MODELS[S.sam_midx] .. " | GPU: " .. SAM_GPUS[S.sam_gidx])

  -- separate_sam.py internally uses uv run --project SAM_DIR so uv
  -- manages the venv, deps, and PATH — no modal binary resolution needed.
  local cmd = string.format(
    '%s %s --sam-dir %s --input %s --prompt %s --model %s --gpu %s --steps %d --ode-method %s --chunk %.1f --overlap %.1f --confidence %.2f --candidates %d --outdir %s --progress %s >> %s 2>&1 &',
    q(PYTHON), q(SAM_PY),
    q(SAM_DIR),
    q(S.src), q(S.sam_prompt),
    q(SAM_MODELS[S.sam_midx]), q(SAM_GPUS[S.sam_gidx]),
    S.sam_steps, ODE_METHODS[S.sam_oidx],
    S.sam_chunk, S.sam_overlap, S.sam_conf, S.sam_cands,
    q(S.outdir), q(PROGRESS_F), q(LOG_F)
  )
  add_log("Lanzando proceso Modal...")
  os.execute(cmd)
end

-- ── LAYOUT ───────────────────────────────────────────────────
local W          = 520
local H          = 680
local SRC_Y      = 48
local TAB_Y      = 136
local CONT_Y     = 178
local ACT_Y      = 450
local PROG_Y     = 495
local LOG_Y      = 525

-- ── DRAW ─────────────────────────────────────────────────────

local function draw_header()
  gfx.setfont(3); col(FG)
  gfx.x, gfx.y = 12, 10
  gfx.drawstr("Stem Separator")

  -- reaper dot
  local rok = reaper.CountTracks(0) >= 0
  col(rok and GRN or RED)
  gfx.circle(W - 18, 22, 6, 1, 1)
  gfx.setfont(2); col(FG2)
  gfx.x, gfx.y = W - 90, 16
  gfx.drawstr(rok and "Reaper OK" or "Reaper --")

  col(BG3); gfx.rect(0, 40, W, 1, 1)
end

local function draw_source()
  -- fila fuente
  ltext(12, SRC_Y + 4, "Fuente:", FG2, 2)
  local display_src = S.src
  if S.src_track_name ~= "" then
    display_src = S.src_track_name .. "  (" .. (S.src:match("[^/\\]+$") or "") .. ")"
  end
  -- use a read-only display for src (editing via "..." or "R" button)
  local x0 = 68
  col(BG3); gfx.rect(x0, SRC_Y, W - 148, 26, 1)
  gfx.setfont(1)
  col(FG)
  gfx.x, gfx.y = x0 + 5, SRC_Y + math.floor((26 - 13) / 2)
  gfx.drawstr(trunc(display_src, W - 158, 1))

  if btn(W - 76, SRC_Y, 30, 26, "...", BG3) then
    local ok, fn = reaper.GetUserFileNameForRead("", "Abrir audio", "wav")
    if ok then S.src = fn; S.src_track_name = ""; S.src_track_idx = -1 end
  end
  if btn(W - 40, SRC_Y, 30, 26, "R", ACC2, FG) then
    grab_from_reaper()
  end

  -- hint
  gfx.setfont(2); col(FG2)
  gfx.x, gfx.y = 12, SRC_Y + 30
  gfx.drawstr(S.src_track_name ~= "" and "Pista seleccionada  |  clic en 'R' para usar pista activa" or "Clic en 'R' para usar pista/item activo de Reaper")

  end

local function draw_tabs()
  local tw = math.floor((W - 24) / 2)
  -- tab Demucs
  local d_active = S.tab == 1
  col(d_active and BG3 or BG2); gfx.rect(12, TAB_Y, tw, 36, 1)
  if d_active then col(ACC); gfx.rect(12, TAB_Y, tw, 3, 1) end
  ctext(12, TAB_Y + 3, tw, 33, "DEMUCS  (local)", d_active and FG or FG2,
        d_active and 3 or 1)
  if in_rect(12, TAB_Y, tw, 36) and clicked then S.tab = 1 end

  -- tab SAM Audio
  local s_active = S.tab == 2
  col(s_active and BG3 or BG2); gfx.rect(12 + tw + 2, TAB_Y, tw, 36, 1)
  if s_active then col(PRP); gfx.rect(12 + tw + 2, TAB_Y, tw, 3, 1) end
  ctext(12 + tw + 2, TAB_Y + 3, tw, 33, "SAM AUDIO  (cloud)", s_active and FG or FG2,
        s_active and 3 or 1)
  if in_rect(12 + tw + 2, TAB_Y, tw, 36) and clicked then S.tab = 2 end

  -- fondo del contenido
  col(BG2); gfx.rect(12, TAB_Y + 36, W - 24, ACT_Y - TAB_Y - 38, 1)
end

local function draw_demucs()
  local x0, y0 = 20, CONT_Y

  ltext(x0, y0 + 4, "Modelo:", FG2, 2)
  S.dm_idx = dropdown(x0 + 70, y0, W - 100, 26, DM_LABELS, S.dm_idx)

  ltext(x0, y0 + 38, "Stems:", FG2, 2)

  local is6s = DM_MODELS[S.dm_idx] == "htdemucs_6s"
  local cols4 = { "vocals", "drums", "bass", "other" }
  local col_w  = math.floor((W - 50) / 4)

  for i, k in ipairs(cols4) do
    S.dm_stems[k] = checkbox(x0 + (i - 1) * col_w, y0 + 56, STEM_NAMES[k], S.dm_stems[k], false, col_w)
  end

  S.dm_stems.guitar = checkbox(x0,              y0 + 80, STEM_NAMES.guitar, S.dm_stems.guitar, not is6s, col_w)
  S.dm_stems.piano  = checkbox(x0 + col_w * 2,  y0 + 80, STEM_NAMES.piano,  S.dm_stems.piano,  not is6s, col_w)
  if not is6s then
    ltext(x0 + col_w, y0 + 84, "(solo htdemucs_6s)", FG2, 2)
  end

  if btn(x0, y0 + 108, 64, 22, "Todos",   BG3, ACC, 2) then
    for _, k in ipairs(STEM_KEYS) do S.dm_stems[k] = true end
  end
  if btn(x0 + 70, y0 + 108, 64, 22, "Ninguno", BG3, FG2, 2) then
    for _, k in ipairs(STEM_KEYS) do S.dm_stems[k] = false end
  end
end

local function draw_sam()
  local x0, y0 = 20, CONT_Y
  local rw = W - 100

  ltext(x0, y0 + 4,  "Prompt:", FG2, 2)
  field(x0 + 68, y0, rw, 26, S.sam_prompt, "<instrumento>", "prompt")

  ltext(x0, y0 + 36, "Modelo:", FG2, 2)
  S.sam_midx = dropdown(x0 + 68, y0 + 32, rw, 26, SAM_MODELS, S.sam_midx)

  ltext(x0, y0 + 68, "GPU:", FG2, 2)
  S.sam_gidx = dropdown(x0 + 68, y0 + 64, 90, 26, SAM_GPUS, S.sam_gidx)

  ltext(x0 + 170, y0 + 68, "ODE:", FG2, 2)
  S.sam_oidx = dropdown(x0 + 210, y0 + 64, 110, 26, ODE_METHODS, S.sam_oidx)

  -- row: steps, confidence
  ltext(x0, y0 + 100, "ODE steps:", FG2, 2)
  field(x0 + 80, y0 + 96, 58, 24, tostring(S.sam_steps), "64", "steps")

  ltext(x0 + 152, y0 + 100, "Conf.:", FG2, 2)
  field(x0 + 196, y0 + 96, 58, 24, string.format("%.2f", S.sam_conf), "0.0", "conf")

  -- row: chunk, overlap, candidates
  ltext(x0, y0 + 130, "Chunk s:", FG2, 2)
  field(x0 + 68, y0 + 126, 50, 24, tostring(S.sam_chunk), "15", "chunk")

  ltext(x0 + 130, y0 + 130, "Overlap:", FG2, 2)
  field(x0 + 196, y0 + 126, 50, 24, tostring(S.sam_overlap), "2", "overlap")

  ltext(x0 + 260, y0 + 130, "Cand.:", FG2, 2)
  field(x0 + 304, y0 + 126, 40, 24, tostring(S.sam_cands), "1", "cands")

  -- nota de coste
  ltext(x0, y0 + 162, "A100: ~$0.14/pista  |  A10G: ~$0.09/pista  |  Requiere cuenta Modal.com", FG2, 2)
end

local function draw_actions()
  local y = ACT_Y
  -- Boton separar
  local running = S.running
  local sep_lbl, sep_bg
  if running then
    sep_lbl, sep_bg = "[ Procesando... ]", BG2
  elseif S.tab == 1 then
    sep_lbl, sep_bg = "SEPARAR  (Demucs)", ACC2
  else
    sep_lbl, sep_bg = "SEPARAR  (SAM Audio)", PRP2
  end
  if btn(12, y, 180, 36, sep_lbl, sep_bg) and not running then
    if S.tab == 1 then launch_demucs() else launch_sam() end
  end
end

local function draw_progress()
  -- barra
  progbar(12, PROG_Y, W - 24, 14, S.progress)

  -- estado
  local sc
  if S.running then
    sc = YLW
  elseif S.done and #S.out_files > 0 then
    sc = GRN
  elseif S.done then
    sc = RED
  else
    sc = FG2
  end
  gfx.setfont(2)
  col(sc)
  gfx.x, gfx.y = 12, PROG_Y + 18
  gfx.drawstr(S.status:sub(1, 80))

  -- area de log
  col(BG2); gfx.rect(12, LOG_Y, W - 24, H - LOG_Y - 6, 1)
  gfx.setfont(4)
  local line_h = 14
  local max_lines = math.floor((H - LOG_Y - 8) / line_h)
  local start_i   = math.max(1, #S.log - max_lines + 1)
  for i = start_i, #S.log do
    local entry = S.log[i]
    col(entry:find("^ERROR") and RED or FG2)
    gfx.x = 16
    gfx.y = LOG_Y + 4 + (i - start_i) * line_h
    gfx.drawstr(entry)
  end
end

local function draw_all()
  col(BG); gfx.rect(0, 0, gfx.w, gfx.h, 1)
  draw_header()
  draw_source()
  draw_tabs()
  if S.tab == 1 then draw_demucs() else draw_sam() end
  draw_actions()
  draw_progress()
end

-- ── LOOP PRINCIPAL ────────────────────────────────────────────
local poll_tick = 0

local function handle_keyboard(char)
  if not S.edit_field then return end
  if char == 8 then -- backspace
    S.edit_buf = S.edit_buf:sub(1, -2)
    S.edit_blink_tick = 0; S.edit_cursor_show = true
  elseif char == 13 or char == 10 then -- enter
    commit_edit()
  elseif char == 27 then -- escape
    S.edit_field = nil
  elseif char >= 32 and char <= 126 then -- printable ASCII
    S.edit_buf = S.edit_buf .. string.char(char)
    S.edit_blink_tick = 0; S.edit_cursor_show = true
  end
end

local function loop()
  local char = gfx.getchar()
  if char < 0 then return end

  -- commit editing on any click before redraw; field() will re-enter if the click lands on a field
  local lb = gfx.mouse_cap & 1
  clicked   = lb == 1 and prev_lb == 0
  clicked_r = (gfx.mouse_cap & 2) == 2
  prev_lb   = lb
  if clicked and S.edit_field then
    commit_edit()
  end

  handle_keyboard(char)

  draw_all()
  gfx.update()

  poll_tick = poll_tick + 1
  if poll_tick % 16 == 0 and S.edit_field then
    S.edit_cursor_show = not S.edit_cursor_show
  end
  if poll_tick >= 8 then
    poll_tick = 0
    if S.running then read_progress() end
  end

  reaper.defer(loop)
end

-- ── INICIO ────────────────────────────────────────────────────
gfx.init("Stem Separator", W, H, 0, 200, 150)
setup_fonts()
add_log("Stem Separator listo.")
add_log("Demucs: " .. PYTHON)
add_log("SAM: " .. SAM_DIR)
loop()
