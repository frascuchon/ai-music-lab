-- @description Stem Separator - Demucs + SAM Audio
-- @version 3.0
-- @author IAClaude
-- @about Separacion de stems con Demucs (local) y SAM Audio (Modal cloud).
--        UI nativa gfx: sin dependencias externas de extensiones REAPER.

-- ── RUTAS + LIB ──────────────────────────────────────────────────
local _info      = debug.getinfo(1, "S")
local SCRIPT_DIR = _info.source:match("@?(.*[/\\])") or ""

-- shared/ es hermano de StemsSeparator/
local SHARED_DIR = SCRIPT_DIR .. "../shared/"
package.path = SHARED_DIR .. "lib/?.lua;" .. package.path

local common  = require("common")
local theme   = require("theme")
local gui     = require("gui")
local widgets = require("widgets_extra")

local HOME       = common.HOME
local TMPDIR     = common.TMPDIR
local SAM_DIR    = SCRIPT_DIR
local SAM_SCRIPT = "modal_sam_audio.py"
local DEMUCS_PY  = SCRIPT_DIR .. "separate_demucs.py"
local SAM_PY     = SCRIPT_DIR .. "separate_sam.py"
local PROGRESS_F = TMPDIR .. "stemsep_progress.txt"
local LOG_F      = TMPDIR .. "stemsep.log"

local PYTHON, PYTHON_ERR = common.detect_reaper_python()
if PYTHON_ERR then
  reaper.ShowConsoleMsg("Stem Separator - WARNING: " .. PYTHON_ERR .. "\n")
end

-- ── CONSTANTES ───────────────────────────────────────────────────
local DM_MODELS = { "htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra" }
local DM_LABELS = {
  "htdemucs  (4 stems)",
  "htdemucs_ft  (4 stems, fine-tuned)",
  "htdemucs_6s  (6 stems)",
  "mdx_extra  (4 stems, MDX-Net)",
}
local STEM_KEYS  = { "vocals", "drums", "bass", "other", "guitar", "piano" }
local STEM_NAMES = { vocals="Vocales", drums="Batería", bass="Bajo",
                     other="Otros", guitar="Guitarra*", piano="Piano*" }
local SAM_MODELS = { "facebook/sam-audio-large", "facebook/sam-audio-base" }
local SAM_GPUS   = { "A100", "A10G", "T4" }
local ODE_METHODS= { "midpoint", "euler", "rk4" }

-- ── ESTADO ───────────────────────────────────────────────────────
local S = {
  tab            = 1,
  src            = "",
  src_track_name = "",
  src_track_idx  = -1,
  src_start_offs  = 0,    -- start offset into source file (seconds)
  src_section_dur = 0,    -- section duration in source time (seconds)
  src_item_pos    = nil,  -- item position in project timeline
  src_is_section  = false,
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

-- ── HELPERS CORE ─────────────────────────────────────────────────
local function add_log(s)
  table.insert(S.log, tostring(s):sub(1, 200))
  if #S.log > 200 then table.remove(S.log, 1) end
  S.log_scroll_to_bottom = true
end

local function q(s) return common.q(s) end

-- ── SETUP CHECK (asíncrono al inicio) ────────────────────────────
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
  local LABELS = {
    python         = "Python REAPER",
    uv             = "uv",
    demucs         = "demucs",
    ["modal-cli"]  = "Modal CLI",
    ["modal-auth"] = "Modal sin auth",
    ["hf-secret"]  = "HF secret faltante",
  }
  for _, line in ipairs(r.extra) do
    local name, status = line:match("^CHECK|([^|]+)|([^|]+)|")
    if name and status == "missing" then
      table.insert(setup_missing, LABELS[name] or name)
    end
  end
end

-- ── PROGRESO ─────────────────────────────────────────────────────
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
    S.done      = true
    S.out_files = {}
    for _, line in ipairs(r.extra) do
      local p = line:match("^%s*(.-)%s*$")
      if p ~= "" then table.insert(S.out_files, p) end
    end
    if #S.out_files > 0 then
      add_log("Archivos listos: " .. #S.out_files)
      import_stems()
    end
  elseif r.state == "error" and not S.done then
    S.running = false
    S.done    = true
    add_log("ERROR: " .. (r.msg or "?"))
  end
end

-- ── INTEGRACIÓN REAPER ───────────────────────────────────────────
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
          local is_sec, offs, dur = detect_section(item, take, src)
          if is_sec then
            add_log(string.format("Fuente (pista): %s | %s [sec %.2fs → %.2fs]",
              S.src_track_name, fname:match("[^/\\]+$") or fname, offs, offs + dur))
          else
            add_log("Fuente (pista): " .. S.src_track_name .. " | " .. fname)
          end
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
    local is_sec, offs, dur = detect_section(item, take, src)
    if is_sec then
      add_log(string.format("Fuente: %s [sec %.2fs → %.2fs]",
        fname:match("[^/\\]+$") or fname, offs, offs + dur))
    else
      add_log("Fuente: " .. fname)
    end
  end
end

function import_stems()
  if #S.out_files == 0 then return end
  reaper.Undo_BeginBlock()
  local cursor   = S.src_item_pos or reaper.GetCursorPosition()
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
  add_log("Importados " .. imported .. " stems en '" .. folder_name .. "'")
end

-- ── LANZAR PROCESOS ──────────────────────────────────────────────
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
  local section_args = ""
  if S.src_is_section then
    section_args = string.format(" --start %.6f --duration %.6f",
      S.src_start_offs, S.src_section_dur)
    add_log(string.format("Sección: %.2fs → %.2fs",
      S.src_start_offs, S.src_start_offs + S.src_section_dur))
  end
  local cmd = string.format(
    '%s %s --input %s --model %s --stems %s --outdir %s --python %s%s --progress %s >>%s 2>&1 &',
    q(PYTHON), q(DEMUCS_PY),
    q(S.src), q(model), table.concat(stems, ","),
    q(S.outdir), q(PYTHON), section_args, q(PROGRESS_F), q(LOG_F))
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
  local section_args = ""
  if S.src_is_section then
    section_args = string.format(" --start %.6f --duration %.6f",
      S.src_start_offs, S.src_section_dur)
    add_log(string.format("Sección: %.2fs → %.2fs",
      S.src_start_offs, S.src_start_offs + S.src_section_dur))
  end
  local cmd = string.format(
    '%s %s --sam-dir %s --shared-dir %s --input %s --prompt %s --model %s --gpu %s' ..
    ' --steps %d --ode-method %s --chunk %.1f --overlap %.1f' ..
    ' --confidence %.2f --candidates %d%s --outdir %s --progress %s >>%s 2>&1 &',
    q(PYTHON), q(SAM_PY), q(SAM_DIR), q(SHARED_DIR),
    q(S.src), q(S.sam_prompt),
    q(SAM_MODELS[S.sam_midx]), q(SAM_GPUS[S.sam_gidx]),
    S.sam_steps, ODE_METHODS[S.sam_oidx],
    S.sam_chunk, S.sam_overlap, S.sam_conf, S.sam_cands,
    section_args, q(S.outdir), q(PROGRESS_F), q(LOG_F))
  add_log("Lanzando proceso Modal...")
  os.execute(cmd)
end

-- ── TAB DEMUCS ───────────────────────────────────────────────────
local function draw_demucs_tab()
  local g = gui
  local t = theme

  g.row_label("Modelo:", t.sc(68))
  g.next_width(-1)
  S.dm_idx = widgets.combo("##dm_model", S.dm_idx, DM_LABELS)
  g.spacing()

  g.text("Stems:")
  g.spacing()

  local is6s = DM_MODELS[S.dm_idx] == "htdemucs_6s"
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
    g.text_disabled("(solo htdemucs_6s)")
  end

  g.spacing()
  if g.button("Todos", t.sc(70), t.ITEM_H) then
    for _, k in ipairs(STEM_KEYS) do S.dm_stems[k] = true end
  end
  g.same_line()
  if g.button("Ninguno", t.sc(70), t.ITEM_H) then
    for _, k in ipairs(STEM_KEYS) do S.dm_stems[k] = false end
  end
end

-- ── TAB SAM AUDIO ────────────────────────────────────────────────
local function draw_sam_tab()
  local g = gui
  local t = theme
  local lw = t.sc(78)  -- label column width

  -- Prompt
  g.row_label("Prompt:", lw)
  local rv, nv = widgets.input_text("##prompt", S.sam_prompt)
  if rv then S.sam_prompt = nv end

  -- Modelo
  g.row_label("Modelo:", lw)
  g.next_width(-1)
  S.sam_midx = widgets.combo("##sam_model", S.sam_midx, SAM_MODELS)

  -- GPU + ODE method
  g.row_label("GPU:", lw)
  g.next_width(t.sc(90))
  S.sam_gidx = widgets.combo("##sam_gpu", S.sam_gidx, SAM_GPUS)
  g.same_line(t.sc(14))
  g.inline_text("ODE:")
  g.same_line(t.sc(6))
  g.next_width(-1)
  S.sam_oidx = widgets.combo("##sam_ode", S.sam_oidx, ODE_METHODS)

  -- ODE steps + Confianza
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

  -- Chunk + Overlap + Candidatos
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
  g.text_disabled("A100: ~$0.14/pista  |  A10G: ~$0.09/pista  |  Requiere cuenta Modal.com")
end

-- ── GFX INIT ─────────────────────────────────────────────────────
if gfx.w > 0 then gfx.quit() end
local LOGICAL_W = 560
gfx.init("Stem Separator", LOGICAL_W, 740)
gfx.ext_retina = 1
theme.init_fonts()

-- ── MAIN LOOP ────────────────────────────────────────────────────
local _scale_init = false

local function loop()
  -- On first frame, detect Retina scale from physical vs logical width.
  -- gfx.ext_retina=1 makes gfx.w reflect physical pixels (2x on Retina).
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

  -- Setup banner (async background check)
  poll_setup_check()
  if setup_checked and #setup_missing > 0 then
    g.text_wrapped("⚠  Config. incompleta: " .. table.concat(setup_missing, " · "))
    g.text_disabled("Carga Setup.lua en Actions > Load ReaScript para configurar.")
    g.spacing()
  end

  -- Header
  g.push_font(t.F.H1)
  g.text("Stem Separator")
  g.pop_font()
  g.same_line(10)
  g.text_colored("● Reaper OK", "GREEN")
  g.separator()
  g.spacing()

  -- Source file row
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
    g.text_disabled("Clic en R para usar pista/item activo de Reaper")
  end
  if S.src_is_section then
    g.text_colored(string.format("Sección: %.2fs → %.2fs  (%.2fs)",
      S.src_start_offs, S.src_start_offs + S.src_section_dur, S.src_section_dur),
      "YELLOW")
  end
  g.spacing()

  -- Tab bar
  S.tab = widgets.tab_bar("##maintabs", S.tab, {"DEMUCS  (local)", "SAM AUDIO  (cloud)"})
  g.spacing()

  if S.tab == 1 then draw_demucs_tab()
  else               draw_sam_tab() end

  g.spacing()
  g.separator()
  g.spacing()

  -- SEPARAR button — colors change per tab
  local sep_colors
  if S.tab == 1 then
    sep_colors = {
      norm   = { 0x29/255, 0x66/255, 0xB0/255 },
      hover  = { 0x3D/255, 0x80/255, 0xD8/255 },
      active = { 0x47/255, 0x99/255, 0xFF/255 },
    }
  else
    sep_colors = {
      norm   = { 0x4D/255, 0x19/255, 0xC4/255 },
      hover  = { 0x66/255, 0x26/255, 0xE0/255 },
      active = { 0x80/255, 0x33/255, 0xD1/255 },
    }
  end
  local sep_lbl = S.running and "[ Procesando... ]"
    or (S.tab == 1 and "SEPARAR  (Demucs)" or "SEPARAR  (SAM Audio)")
  g.begin_disabled(S.running)
  g.next_width(-1)
  if g.button(sep_lbl, nil, t.sc(36), { solid = sep_colors }) then
    if S.tab == 1 then launch_demucs() else launch_sam() end
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

  -- Log area
  if widgets.collapsing_header("Logs", true) then
    if g.button("Copiar log", t.sc(90), t.ITEM_H) then
      local ok, set_cb = pcall(function()
        reaper.CF_SetClipboard(table.concat(S.log, "\n"))
      end)
      if not ok then
        -- SWS/CF_ not available; print to console instead
        reaper.ShowConsoleMsg(table.concat(S.log, "\n") .. "\n")
      end
    end
    g.same_line()
    if g.button("Limpiar", t.sc(70), t.ITEM_H) then S.log = {} end
    g.spacing()

    -- Scroll to bottom if new lines arrived
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
add_log("Stem Separator listo.")
add_log("Python: " .. PYTHON)
add_log("SAM dir: " .. SAM_DIR)
launch_setup_check()
reaper.defer(loop)
