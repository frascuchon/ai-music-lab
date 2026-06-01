-- @description Stem Separator — Configuración y Setup
-- @version 2.0
-- @author IAClaude
-- @about Wizard de primera vez: detecta el entorno, guía el login en Modal,
--        crea el secret de Hugging Face y descarga el modelo SAM Audio.
--        UI nativa gfx (sin dependencias externas).

-- ── RUTAS + LIB ──────────────────────────────────────────────────
local _info      = debug.getinfo(1, "S")
local SCRIPT_DIR = _info.source:match("@?(.*[/\\])") or ""

package.path = SCRIPT_DIR .. "lib/?.lua;" .. package.path

local common  = require("common")
local theme   = require("theme")
local gui     = require("gui")
local widgets = require("widgets_extra")

local TMPDIR     = common.TMPDIR
local HELPER     = SCRIPT_DIR .. "setup_helpers.py"
local REAPER_INI = reaper.GetResourcePath() .. "/reaper.ini"

local PYTHON = common.detect_reaper_python()

-- ── HELPERS ──────────────────────────────────────────────────────
local function q(s) return common.q(s) end

local PF = {
  check    = TMPDIR .. "stemsep_setup_check.txt",
  uv       = TMPDIR .. "stemsep_setup_uv.txt",
  sync     = TMPDIR .. "stemsep_setup_sync.txt",
  demucs   = TMPDIR .. "stemsep_setup_demucs.txt",
  login    = TMPDIR .. "stemsep_setup_login.txt",
  hfsecret = TMPDIR .. "stemsep_setup_hfsecret.txt",
  prewarm  = TMPDIR .. "stemsep_setup_prewarm.txt",
}

local function launch(key, subcmd_args)
  local pf  = PF[key]
  local log = TMPDIR .. "stemsep_setup.log"
  local f = io.open(pf, "w")
  if f then f:write("running|0.00|Iniciando..."); f:close() end
  local cmd = string.format('%s %s %s --progress %s >>%s 2>&1 &',
    q(PYTHON), q(HELPER), subcmd_args, q(pf), q(log))
  os.execute(cmd)
end

local function read_pf(path)
  return common.read_progress_file(path)
end

-- ── ESTADO ───────────────────────────────────────────────────────
local CHECKS = {
  { name="python",     label="Python (REAPER)" },
  { name="uv",         label="uv" },
  { name="demucs",     label="demucs (tab Demucs)" },
  { name="modal-cli",  label="Modal CLI" },
  { name="modal-auth", label="Modal autenticado" },
  { name="hf-secret",  label="HF secret 'huggingface-secret'" },
}
local CHECK_IDX = {}
for i, c in ipairs(CHECKS) do
  c.status = "?"; c.detail = ""
  CHECK_IDX[c.name] = i
end

local ST = {
  check_state   = "idle",
  uv_state      = "idle", uv_msg      = "",
  sync_state    = "idle", sync_msg    = "",
  demucs_state  = "idle", demucs_msg  = "",
  login_state   = "idle", login_msg   = "",
  hf_state      = "idle", hf_msg      = "",
  hf_token      = "",
  prewarm_state = "idle", prewarm_msg = "", prewarm_pct = 0.0,
}

-- ── POLLING ──────────────────────────────────────────────────────
local function poll_check()
  if ST.check_state ~= "running" then return end
  local r = read_pf(PF.check)
  if not r then return end
  if r.state == "done" then
    ST.check_state = "done"
    for _, line in ipairs(r.extra) do
      local name, status, detail = line:match("^CHECK|([^|]+)|([^|]+)|(.-)$")
      if name and CHECK_IDX[name] then
        local c = CHECKS[CHECK_IDX[name]]
        c.status = status or "?"; c.detail = detail or ""
      end
    end
  elseif r.state == "error" then
    ST.check_state = "error"
  end
end

local function poll_simple(state_key, msg_key, pf_key)
  if ST[state_key] ~= "running" then return end
  local r = read_pf(PF[pf_key])
  if not r then return end
  if msg_key then ST[msg_key] = r.msg or "" end
  if r.state == "done" or r.state == "error" then
    ST[state_key] = r.state
  end
end

local function poll_prewarm()
  if ST.prewarm_state ~= "running" then return end
  local r = read_pf(PF.prewarm)
  if not r then return end
  ST.prewarm_msg = r.msg or ""
  ST.prewarm_pct = r.pct or 0
  if r.state == "done" or r.state == "error" then
    ST.prewarm_state = r.state
  end
end

local function run_check()
  ST.check_state = "running"
  for _, c in ipairs(CHECKS) do c.status = "?"; c.detail = "" end
  local f = io.open(PF.check, "w")
  if f then f:write("running|0.00|..."); f:close() end
  local log = TMPDIR .. "stemsep_setup.log"
  local cmd = string.format('%s %s check --progress %s >>%s 2>&1 &',
    q(PYTHON), q(HELPER), q(PF.check), q(log))
  os.execute(cmd)
end

-- ── STATUS HELPERS ───────────────────────────────────────────────
local function status_color(s)
  if s == "ok"      then return "GREEN"
  elseif s == "missing" then return "RED"
  elseif s == "?"   then return "FG_DIM"
  else                   return "YELLOW" end
end

local function status_icon(s)
  if s == "ok"      then return "OK"
  elseif s == "missing" then return "X"
  else                   return "..." end
end

-- ── GFX INIT ─────────────────────────────────────────────────────
if gfx.w > 0 then gfx.quit() end
local LOGICAL_W = 500
gfx.init("Stem Separator — Configuración", LOGICAL_W, 560)
gfx.ext_retina = 1
theme.init_fonts()

local first_frame = true
local _scale_init = false
local pending_recheck = false

-- ── DRAW CHECKS LIST ─────────────────────────────────────────────
local function draw_checks()
  local g = gui
  local t = theme
  local checking = ST.check_state == "running"
  local uv_ok = CHECKS[CHECK_IDX["uv"]] and CHECKS[CHECK_IDX["uv"]].status == "ok"

  for _, c in ipairs(CHECKS) do
    local icon  = checking and "..." or status_icon(c.status)
    local color = checking and "YELLOW" or status_color(c.status)

    g.text_colored("[" .. icon .. "]", color)
    g.same_line(6)
    g.text(c.label)

    if c.detail ~= "" then
      g.same_line(10)
      g.text_disabled(c.detail:sub(1, 52))
    end

    -- Inline action buttons for fixable items
    if not checking then
      if c.name == "uv" and c.status == "missing" then
        g.same_line()
        g.begin_disabled(ST.uv_state == "running")
        if g.button("Instalar uv", 90, t.ITEM_H) then
          ST.uv_state = "running"; ST.uv_msg = ""
          launch("uv", "install-uv")
        end
        g.end_disabled()
        if ST.uv_state == "running" then
          g.same_line(6)
          g.text_colored(ST.uv_msg ~= "" and ST.uv_msg:sub(1,45) or "Instalando...", "YELLOW")
        elseif ST.uv_state == "error" then
          g.same_line(6)
          g.text_colored("Error", "RED")
        end

      elseif c.name == "modal-cli" and c.status == "missing" and uv_ok then
        g.same_line()
        g.begin_disabled(ST.sync_state == "running")
        if g.button("Instalar deps", 102, t.ITEM_H) then
          ST.sync_state = "running"; ST.sync_msg = ""
          launch("sync", "sync-deps")
        end
        g.end_disabled()
        if ST.sync_state == "running" then
          g.same_line(6)
          g.text_colored(ST.sync_msg ~= "" and ST.sync_msg:sub(1,40) or "Instalando...", "YELLOW")
        elseif ST.sync_state == "error" then
          g.same_line(6)
          g.text_colored("Error", "RED")
        end

      elseif c.name == "demucs" and c.status == "missing" then
        g.same_line()
        g.begin_disabled(ST.demucs_state == "running")
        if g.button("Instalar", 70, t.ITEM_H) then
          ST.demucs_state = "running"; ST.demucs_msg = ""
          launch("demucs", "install-demucs --python " .. q(PYTHON))
        end
        g.end_disabled()
        if ST.demucs_state == "running" then
          g.same_line(6); g.text_colored("Instalando...", "YELLOW")
        elseif ST.demucs_state == "error" then
          g.same_line(6); g.text_colored("Error — revisa el log", "RED")
        end

      elseif c.name == "modal-auth" and c.status == "missing" then
        g.same_line()
        g.begin_disabled(ST.login_state == "running")
        if g.button("Login", 50, t.ITEM_H) then
          ST.login_state = "running"; ST.login_msg = ""
          launch("login", "modal-login")
        end
        g.end_disabled()
        if ST.login_state == "running" then
          local msg = ST.login_msg ~= "" and ST.login_msg:sub(1,45) or "Abriendo navegador..."
          g.same_line(6); g.text_colored(msg, "YELLOW")
        elseif ST.login_state == "error" then
          g.same_line(6); g.text_colored("Error — reintenta", "RED")
        end
      end
    end
  end
end

-- ── MAIN LOOP ────────────────────────────────────────────────────
local function loop()
  -- On first frame, detect Retina scale from physical vs logical width.
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

  -- Auto-check on first frame
  if first_frame then first_frame = false; run_check() end

  -- Poll async operations
  poll_check()
  poll_simple("uv_state",     "uv_msg",     "uv")
  poll_simple("sync_state",   "sync_msg",   "sync")
  poll_simple("demucs_state", "demucs_msg", "demucs")
  poll_simple("login_state",  "login_msg",  "login")
  poll_simple("hf_state",     "hf_msg",     "hfsecret")
  poll_prewarm()

  -- Trigger recheck after an action finishes
  local function mark_done(key)
    if ST[key] == "done" then ST[key] = "idle"; pending_recheck = true end
  end
  mark_done("uv_state"); mark_done("sync_state")
  mark_done("demucs_state"); mark_done("login_state"); mark_done("hf_state")
  if pending_recheck and ST.check_state ~= "running" then
    pending_recheck = false; run_check()
  end

  -- ── HEADER ─────────────────────────────────────────────────────
  g.push_font(t.F.H1)
  g.text("Stem Separator — Configuración")
  g.pop_font()
  g.spacing()

  g.begin_disabled(ST.check_state == "running")
  g.next_width(-1)
  if g.button("Comprobar todo de nuevo") then run_check() end
  g.end_disabled()
  g.spacing()
  g.separator()
  g.spacing()

  -- ── CHECKS ─────────────────────────────────────────────────────
  draw_checks()

  g.spacing()
  g.separator()
  g.spacing()

  -- ── HUGGING FACE TOKEN ─────────────────────────────────────────
  g.text("Hugging Face — secret para SAM Audio")
  g.spacing()
  if g.button("Crear token en huggingface.co/settings/tokens", -1, t.BTN_H) then
    os.execute('open "https://huggingface.co/settings/tokens" &')
  end
  g.spacing()

  g.row_label("Token HF:", 72)
  local hf_changed, hf_new = widgets.input_text("##hftoken", ST.hf_token,
    { password = true })
  if hf_changed then ST.hf_token = hf_new end

  local can_save = ST.hf_token:sub(1, 3) == "hf_" and ST.hf_state ~= "running"
  g.begin_disabled(not can_save)
  g.next_width(-1)
  if g.button('Guardar como Modal secret "huggingface-secret"') then
    ST.hf_state = "running"; ST.hf_msg = ""
    launch("hfsecret", "modal-secret-create --token " .. q(ST.hf_token))
    ST.hf_token = ""
  end
  g.end_disabled()

  if ST.hf_state == "running" then
    g.text_colored("Guardando secret en Modal...", "YELLOW")
  elseif ST.hf_state == "done" then
    g.text_colored("OK  " .. (ST.hf_msg ~= "" and ST.hf_msg:sub(1,60) or "Guardado"), "GREEN")
  elseif ST.hf_state == "error" then
    g.text_colored("Error: " .. ST.hf_msg:sub(1,60), "RED")
  end

  g.spacing()
  g.separator()
  g.spacing()

  -- ── PRE-WARM ───────────────────────────────────────────────────
  g.text("Pre-cargar modelo SAM Audio (opcional)")
  g.text_disabled("~14 GB  |  primera vez 10-20 min  |  queda cacheado en Modal")
  g.spacing()

  g.begin_disabled(ST.prewarm_state == "running")
  g.next_width(-1)
  if g.button("Descargar facebook/sam-audio-large") then
    ST.prewarm_state = "running"; ST.prewarm_pct = 0; ST.prewarm_msg = ""
    launch("prewarm", "prewarm --model facebook/sam-audio-large")
  end
  g.end_disabled()

  if ST.prewarm_state == "running" then
    g.progress_bar(ST.prewarm_pct, nil, 14,
      string.format('%d%%', math.floor(ST.prewarm_pct * 100)))
    if ST.prewarm_msg ~= "" then
      g.text_colored(ST.prewarm_msg:sub(1, 70), "YELLOW")
    end
  elseif ST.prewarm_state == "done" then
    g.text_colored("OK  Descarga completada", "GREEN")
  elseif ST.prewarm_state == "error" then
    g.text_colored("Error: " .. ST.prewarm_msg:sub(1,60), "RED")
  end

  gui.frame_end()
  reaper.defer(loop)
end

reaper.defer(loop)
