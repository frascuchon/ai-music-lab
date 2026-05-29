-- @description Stem Separator — Configuración y Setup
-- @version 1.0
-- @author IAClaude
-- @about Wizard de primera vez: detecta el entorno, guía el login en Modal,
--        crea el secret de Hugging Face y descarga el modelo SAM Audio.

-- ── CHECK REAIMGUI ────────────────────────────────────────────────
if not reaper.ImGui_GetVersion then
  local _info   = debug.getinfo(1, "S")
  local _dir    = _info.source:match("@?(.*[/\\])") or ""
  local _helper = _dir .. "setup_helpers.py"
  local _tmpdir = os.getenv("TMPDIR") or "/tmp/"
  local _pf     = _tmpdir .. "stemsep_reaimgui_install.txt"

  if reaper.APIExists("ReaPack_BrowsePackages") then
    reaper.ReaPack_BrowsePackages("ReaImGui")
    reaper.MB(
      "ReaImGui no está instalado.\n\n" ..
      "Hemos abierto ReaPack filtrado por 'ReaImGui'.\n\n" ..
      "  1. Selecciona 'ReaImGui' de cfillion\n" ..
      "  2. Clic derecho → Install latest version\n" ..
      "  3. Apply\n" ..
      "  4. Reinicia REAPER y vuelve a abrir el script.",
      "Setup — Instalar ReaImGui", 0)
  else
    reaper.MB("ReaImGui no encontrado. Descargando desde GitHub, espera unos segundos...",
      "Setup — ReaImGui", 0)
    os.execute(string.format('python3 "%s" install-reaimgui --progress "%s"', _helper, _pf))
    local _msg = "Revisa ~/Library/Application Support/REAPER/UserPlugins/"
    local _f = io.open(_pf, "r")
    if _f then
      local _raw = _f:read("*a"); _f:close()
      _msg = _raw:match("|[^|]*|(.+)$") or _msg
    end
    reaper.MB(_msg .. "\n\nReinicia REAPER para activar ReaImGui.",
      "Setup — ReaImGui", 0)
  end
  return
end

-- ── RUTAS ────────────────────────────────────────────────────────
local info       = debug.getinfo(1, "S")
local SCRIPT_DIR = info.source:match("@?(.*[/\\])") or ""
local TMPDIR     = os.getenv("TMPDIR") or "/tmp/"
local HELPER     = SCRIPT_DIR .. "setup_helpers.py"
local REAPER_INI = reaper.GetResourcePath() .. "/reaper.ini"

-- ── PYTHON DETECTION ─────────────────────────────────────────────
local function detect_python()
  local f = io.open(REAPER_INI, "r")
  if not f then return "python3" end
  local libpath
  for line in f:lines() do
    local k, v = line:match("^(pythonlibpath64)=(.*)$")
    if not k then k, v = line:match("^(pythonlibpath32)=(.*)$") end
    if k and v ~= "" then libpath = v; break end
  end
  f:close()
  if not libpath then return "python3" end
  local parent = libpath:match("^(.+)/lib[^/]*$") or libpath
  local exe    = parent .. "/bin/python3"
  local tf = io.open(exe, "r")
  if tf then tf:close(); return exe end
  return "python3"
end

local PYTHON = detect_python()

-- ── HELPERS ──────────────────────────────────────────────────────
local function q(s) return '"' .. s:gsub('"', '\\"') .. '"' end

local PF = {
  check    = TMPDIR .. "stemsep_setup_check.txt",
  uv       = TMPDIR .. "stemsep_setup_uv.txt",
  sync     = TMPDIR .. "stemsep_setup_sync.txt",
  demucs   = TMPDIR .. "stemsep_setup_demucs.txt",
  login    = TMPDIR .. "stemsep_setup_login.txt",
  hfsecret = TMPDIR .. "stemsep_setup_hfsecret.txt",
  prewarm  = TMPDIR .. "stemsep_setup_prewarm.txt",
}

local function read_pf(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local raw = f:read("*a"); f:close()
  if raw == "" then return nil end
  local r = { extra = {} }
  local first = true
  for line in (raw .. "\n"):gmatch("([^\n]*)\n") do
    if line ~= "" then
      if first then
        r.state, r.pct, r.msg = line:match("^([^|]+)|([^|]+)|(.+)$")
        r.pct = tonumber(r.pct) or 0
        first = false
      else
        table.insert(r.extra, line)
      end
    end
  end
  return r
end

local function launch(key, subcmd_args)
  local pf = PF[key]
  local f = io.open(pf, "w")
  if f then f:write("running|0.00|Iniciando..."); f:close() end
  local log = TMPDIR .. "stemsep_setup.log"
  local cmd = string.format('%s %s %s --progress %s >> %s 2>&1 &',
    q(PYTHON), q(HELPER), subcmd_args, q(pf), q(log))
  os.execute(cmd)
end

-- ── ESTADO ───────────────────────────────────────────────────────
local CHECKS = {
  { name="python",     label="Python (REAPER)" },
  { name="uv",         label="uv" },
  { name="demucs",     label="demucs (Demucs tab)" },
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
  launch("check", "check")
end

-- ── IMGUI ────────────────────────────────────────────────────────
local ctx     = reaper.ImGui_CreateContext('SS Setup')
local font_ui = reaper.ImGui_CreateFont('sans-serif', 14)
reaper.ImGui_Attach(ctx, font_ui)

local C_GREEN  = 0x40B261FF
local C_RED    = 0xD94238FF
local C_YELLOW = 0xF2B81AFF
local C_GRAY   = 0x848491FF

local function status_color(s)
  if s == "ok"      then return C_GREEN
  elseif s == "missing" then return C_RED
  elseif s == "?"   then return C_GRAY
  else                   return C_YELLOW end
end
local function status_icon(s)
  if s == "ok"      then return "✓"
  elseif s == "missing" then return "✗"
  else                   return "⋯" end
end

local first_frame = true
local pending_recheck = false  -- trigger re-check after an action completes

local function loop()
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_WindowBg(),         0x1A1A1CFF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_TitleBg(),          0x232327FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_TitleBgActive(),    0x2E2E35FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_FrameBg(),          0x2E2E38FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_FrameBgHovered(),   0x3C3C48FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_FrameBgActive(),    0x4A4A58FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_Button(),           0x4799FF30)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_ButtonHovered(),    0x4799FF70)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_ButtonActive(),     0x4799FFAA)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_Text(),             0xEBEBEBFF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_TextDisabled(),     0x848491FF)
  reaper.ImGui_PushStyleColor(ctx, reaper.ImGui_Col_Separator(),        0x3A3A44FF)
  local N_COL = 12

  reaper.ImGui_PushStyleVar(ctx, reaper.ImGui_StyleVar_FrameRounding(),  4.0)
  reaper.ImGui_PushStyleVar(ctx, reaper.ImGui_StyleVar_WindowRounding(), 6.0)
  local N_VAR = 2

  reaper.ImGui_PushFont(ctx, font_ui, 14)
  reaper.ImGui_SetNextWindowSize(ctx, 500, 540, reaper.ImGui_Cond_FirstUseEver())

  local visible, open = reaper.ImGui_Begin(ctx, 'Stem Separator — Configuración##setup', true)

  if visible then

    -- Auto-check on first frame
    if first_frame then first_frame = false; run_check() end

    -- Poll all async operations
    poll_check()
    poll_simple("uv_state",     "uv_msg",     "uv")
    poll_simple("sync_state",   "sync_msg",   "sync")
    poll_simple("demucs_state", "demucs_msg", "demucs")
    poll_simple("login_state",  "login_msg",  "login")
    poll_simple("hf_state",     "hf_msg",     "hfsecret")
    poll_prewarm()

    -- Re-check after an action finishes
    if pending_recheck and ST.check_state ~= "running" then
      pending_recheck = false
      run_check()
    end
    local function mark_done(key)
      if ST[key] == "done" then ST[key] = "idle"; pending_recheck = true end
    end
    mark_done("uv_state"); mark_done("sync_state")
    mark_done("demucs_state"); mark_done("login_state"); mark_done("hf_state")

    -- ── TÍTULO + BOTÓN RECHECK ──────────────────────────────────
    reaper.ImGui_Text(ctx, 'Stem Separator — Configuración')
    reaper.ImGui_Spacing(ctx)
    reaper.ImGui_BeginDisabled(ctx, ST.check_state == "running")
    if reaper.ImGui_Button(ctx, 'Comprobar todo de nuevo##recheck', -1, 0) then
      run_check()
    end
    reaper.ImGui_EndDisabled(ctx)
    reaper.ImGui_Spacing(ctx)
    reaper.ImGui_Separator(ctx)
    reaper.ImGui_Spacing(ctx)

    -- ── LISTA DE CHECKS ─────────────────────────────────────────
    local checking = ST.check_state == "running"
    for _, c in ipairs(CHECKS) do
      local icon  = checking and "⋯" or status_icon(c.status)
      local color = checking and C_YELLOW or status_color(c.status)

      reaper.ImGui_TextColored(ctx, color, icon)
      reaper.ImGui_SameLine(ctx, 0, 6)
      reaper.ImGui_Text(ctx, c.label)
      if c.detail ~= "" then
        reaper.ImGui_SameLine(ctx, 0, 10)
        reaper.ImGui_TextDisabled(ctx, c.detail:sub(1, 52))
      end

      -- Inline actions for fixable items
      if not checking then
        -- uv missing: install button
        if c.name == "uv" and c.status == "missing" then
          reaper.ImGui_SameLine(ctx)
          reaper.ImGui_BeginDisabled(ctx, ST.uv_state == "running")
          if reaper.ImGui_Button(ctx, 'Instalar uv##uvinstall', 88, 0) then
            ST.uv_state = "running"; ST.uv_msg = ""
            launch("uv", "install-uv")
          end
          reaper.ImGui_EndDisabled(ctx)
          if ST.uv_state == "running" then
            reaper.ImGui_SameLine(ctx)
            reaper.ImGui_TextColored(ctx, C_YELLOW, ST.uv_msg ~= "" and ST.uv_msg:sub(1,45) or "Instalando...")
          elseif ST.uv_state == "error" then
            reaper.ImGui_SameLine(ctx)
            reaper.ImGui_TextColored(ctx, C_RED, "Error — " .. ST.uv_msg:sub(1,40))
          end

        -- modal-cli missing (uv ok): sync project deps
        elseif c.name == "modal-cli" and c.status == "missing" then
          local uv_ok = CHECKS[CHECK_IDX["uv"]] and CHECKS[CHECK_IDX["uv"]].status == "ok"
          if uv_ok then
            reaper.ImGui_SameLine(ctx)
            reaper.ImGui_BeginDisabled(ctx, ST.sync_state == "running")
            if reaper.ImGui_Button(ctx, 'Instalar deps##syncdeps', 102, 0) then
              ST.sync_state = "running"; ST.sync_msg = ""
              launch("sync", "sync-deps")
            end
            reaper.ImGui_EndDisabled(ctx)
            if ST.sync_state == "running" then
              reaper.ImGui_SameLine(ctx)
              reaper.ImGui_TextColored(ctx, C_YELLOW, ST.sync_msg ~= "" and ST.sync_msg:sub(1,40) or "Instalando...")
            elseif ST.sync_state == "error" then
              reaper.ImGui_SameLine(ctx)
              reaper.ImGui_TextColored(ctx, C_RED, "Error")
            end
          end

        -- demucs missing: install button
        elseif c.name == "demucs" and c.status == "missing" then
          reaper.ImGui_SameLine(ctx)
          reaper.ImGui_BeginDisabled(ctx, ST.demucs_state == "running")
          if reaper.ImGui_Button(ctx, 'Instalar##dem', 68, 0) then
            ST.demucs_state = "running"; ST.demucs_msg = ""
            launch("demucs", "install-demucs --python " .. q(PYTHON))
          end
          reaper.ImGui_EndDisabled(ctx)
          if ST.demucs_state == "running" then
            reaper.ImGui_SameLine(ctx)
            reaper.ImGui_TextColored(ctx, C_YELLOW, "Instalando...")
          elseif ST.demucs_state == "error" then
            reaper.ImGui_SameLine(ctx)
            reaper.ImGui_TextColored(ctx, C_RED, "Error — revisa el log")
          end

        -- modal-auth missing: login button
        elseif c.name == "modal-auth" and c.status == "missing" then
          reaper.ImGui_SameLine(ctx)
          reaper.ImGui_BeginDisabled(ctx, ST.login_state == "running")
          if reaper.ImGui_Button(ctx, 'Login##modallogin', 48, 0) then
            ST.login_state = "running"; ST.login_msg = ""
            launch("login", "modal-login")
          end
          reaper.ImGui_EndDisabled(ctx)
          if ST.login_state == "running" then
            local msg = ST.login_msg ~= "" and ST.login_msg:sub(1, 45) or "Abriendo navegador..."
            reaper.ImGui_SameLine(ctx)
            reaper.ImGui_TextColored(ctx, C_YELLOW, msg)
          elseif ST.login_state == "error" then
            reaper.ImGui_SameLine(ctx)
            reaper.ImGui_TextColored(ctx, C_RED, "Error — reintenta")
          end
        end
      end
    end

    reaper.ImGui_Spacing(ctx)
    reaper.ImGui_Separator(ctx)
    reaper.ImGui_Spacing(ctx)

    -- ── HUGGING FACE SECRET ─────────────────────────────────────
    reaper.ImGui_Text(ctx, 'Hugging Face — secret para SAM Audio')
    if reaper.ImGui_Button(ctx, 'Crear token en huggingface.co/settings/tokens##hfurl', -1, 0) then
      os.execute('open "https://huggingface.co/settings/tokens" &')
    end
    reaper.ImGui_Spacing(ctx)

    reaper.ImGui_AlignTextToFramePadding(ctx)
    reaper.ImGui_Text(ctx, 'Token HF:')
    reaper.ImGui_SameLine(ctx)
    reaper.ImGui_SetNextItemWidth(ctx, -1)
    local rv, new_tok = reaper.ImGui_InputText(ctx, '##hftoken', ST.hf_token,
      reaper.ImGui_InputTextFlags_Password())
    if rv then ST.hf_token = new_tok end

    local can_save = ST.hf_token:sub(1, 3) == "hf_" and ST.hf_state ~= "running"
    reaper.ImGui_BeginDisabled(ctx, not can_save)
    if reaper.ImGui_Button(ctx, 'Guardar como Modal secret "huggingface-secret"##hfsave', -1, 0) then
      ST.hf_state = "running"; ST.hf_msg = ""
      launch("hfsecret", "modal-secret-create --token " .. q(ST.hf_token))
      ST.hf_token = ""
    end
    reaper.ImGui_EndDisabled(ctx)

    if ST.hf_state == "running" then
      reaper.ImGui_TextColored(ctx, C_YELLOW, "Guardando secret en Modal...")
    elseif ST.hf_state == "done" then
      reaper.ImGui_TextColored(ctx, C_GREEN,  "✓ " .. ST.hf_msg:sub(1, 60))
    elseif ST.hf_state == "error" then
      reaper.ImGui_TextColored(ctx, C_RED,    "✗ " .. ST.hf_msg:sub(1, 60))
    end

    reaper.ImGui_Spacing(ctx)
    reaper.ImGui_Separator(ctx)
    reaper.ImGui_Spacing(ctx)

    -- ── PRE-WARM MODELO ─────────────────────────────────────────
    reaper.ImGui_Text(ctx, 'Pre-cargar modelo SAM Audio (opcional)')
    reaper.ImGui_TextDisabled(ctx, '~14 GB · primera vez 10-20 min · queda cacheado en Modal')
    reaper.ImGui_Spacing(ctx)
    reaper.ImGui_BeginDisabled(ctx, ST.prewarm_state == "running")
    if reaper.ImGui_Button(ctx, 'Descargar facebook/sam-audio-large##pw', -1, 0) then
      ST.prewarm_state = "running"; ST.prewarm_pct = 0; ST.prewarm_msg = ""
      launch("prewarm", "prewarm --model facebook/sam-audio-large")
    end
    reaper.ImGui_EndDisabled(ctx)

    if ST.prewarm_state == "running" then
      reaper.ImGui_ProgressBar(ctx, ST.prewarm_pct, -1, 14,
        string.format('%d%%', math.floor(ST.prewarm_pct * 100)))
      if ST.prewarm_msg ~= "" then
        reaper.ImGui_TextColored(ctx, C_YELLOW, ST.prewarm_msg:sub(1, 70))
      end
    elseif ST.prewarm_state == "done" then
      reaper.ImGui_TextColored(ctx, C_GREEN, "✓ Descarga completada")
    elseif ST.prewarm_state == "error" then
      reaper.ImGui_TextColored(ctx, C_RED,   "✗ " .. ST.prewarm_msg:sub(1, 60))
    end

    reaper.ImGui_End(ctx)
  end -- if visible

  reaper.ImGui_PopFont(ctx)
  reaper.ImGui_PopStyleVar(ctx, N_VAR)
  reaper.ImGui_PopStyleColor(ctx, N_COL)

  if open then reaper.defer(loop) end
end

reaper.defer(loop)
