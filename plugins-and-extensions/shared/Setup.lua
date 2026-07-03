-- @description AI Music Lab - Setup — Configuración global
-- @version 1.0
-- @author AI Music Lab
-- @about Wizard de configuración global para todos los plugins REAPER AI.
--        Gestiona el entorno común (uv, Modal) y los extras de cada plugin
--        (StemsSeparator: demucs, HF secret, SAM prewarm;
--         Audio2Midi: prewarm MIROS/YourMT3+).
--        UI nativa gfx (sin dependencias externas de extensiones REAPER).

-- ── RUTAS + LIB ──────────────────────────────────────────────────
local _info      = debug.getinfo(1, "S")
local SCRIPT_DIR = _info.source:match("@?(.*[/\\])") or ""  -- shared/

-- Derivar directorio de plugins (hermano de shared/)
local PLUGINS_DIR = SCRIPT_DIR:match("^(.*[/\\])shared[/\\]$") or (SCRIPT_DIR .. "../")

package.path = SCRIPT_DIR .. "lib/?.lua;" .. package.path

local common  = require("common")
local theme   = require("theme")
local gui     = require("gui")
local widgets = require("widgets_extra")

local TMPDIR  = common.TMPDIR
local HELPER  = SCRIPT_DIR .. "setup_helpers.py"
local PYTHON  = common.detect_reaper_python()

-- ── HELPERS ──────────────────────────────────────────────────────
local function q(s) return common.q(s) end

local function plugin_exists(name)
  local p = io.open(PLUGINS_DIR .. name, "r")
  if p then p:close(); return true end
  -- try as directory by checking a marker file
  p = io.open(PLUGINS_DIR .. name .. "/", "r")
  if p then p:close(); return true end
  -- best-effort: check if any file exists in the dir
  local test = io.open(PLUGINS_DIR .. name .. "/README.md", "r")
  if test then test:close(); return true end
  test = io.open(PLUGINS_DIR .. name .. "/StemSeparator.lua", "r")
  if test then test:close(); return true end
  test = io.open(PLUGINS_DIR .. name .. "/Audio2Midi.lua", "r")
  if test then test:close(); return true end
  test = io.open(PLUGINS_DIR .. name .. "/MidiGenerator.lua", "r")
  if test then test:close(); return true end
  return false
end

local HAS_STEMS = plugin_exists("StemsSeparator")
local HAS_A2M   = plugin_exists("Audio2Midi")
local HAS_MG    = plugin_exists("MidiGenerator")

local PF = {
  check        = TMPDIR .. "reaperai_setup_check.txt",
  uv           = TMPDIR .. "reaperai_setup_uv.txt",
  sync         = TMPDIR .. "reaperai_setup_sync.txt",
  login        = TMPDIR .. "reaperai_setup_login.txt",
  demucs       = TMPDIR .. "reaperai_setup_demucs.txt",
  hfsecret     = TMPDIR .. "reaperai_setup_hfsecret.txt",
  prewarm_sam  = TMPDIR .. "reaperai_setup_prewarm_sam.txt",
  prewarm_mir  = TMPDIR .. "reaperai_setup_prewarm_miros.txt",
  prewarm_yt3  = TMPDIR .. "reaperai_setup_prewarm_yourmt3.txt",
  -- MidiGenerator
  prw_mg_amadeus      = TMPDIR .. "reaperai_setup_prw_mg_amadeus.txt",
  prw_mg_midi_llm     = TMPDIR .. "reaperai_setup_prw_mg_midi_llm.txt",
  prw_mg_text2midi    = TMPDIR .. "reaperai_setup_prw_mg_text2midi.txt",
  prw_mg_chatmusician = TMPDIR .. "reaperai_setup_prw_mg_chatmusician.txt",
  prw_mg_musecoco     = TMPDIR .. "reaperai_setup_prw_mg_musecoco.txt",
  prw_mg_anticipatory = TMPDIR .. "reaperai_setup_prw_mg_anticipatory.txt",
}

local function launch(key, subcmd_args)
  local pf  = PF[key]
  local log = TMPDIR .. "reaperai_setup.log"
  local f = io.open(pf, "w")
  if f then f:write("running|0.00|Iniciando..."); f:close() end
  local cmd = string.format('%s %s %s --progress %s >>%s 2>&1 &',
    q(PYTHON), q(HELPER), subcmd_args, q(pf), q(log))
  os.execute(cmd)
end

-- ── ESTADO ───────────────────────────────────────────────────────
-- Core checks always present; plugin extras detected at runtime
local CORE_NAMES = { "python", "uv", "modal-cli", "modal-auth" }
local STEMS_NAMES = { "demucs", "hf-secret" }

local CHECKS = {}  -- filled dynamically from CHECK| lines
local CHECK_IDX = {}

local function init_checks()
  CHECKS = {}; CHECK_IDX = {}
  local names = { "python", "uv", "modal-cli", "modal-auth" }
  if HAS_STEMS then
    table.insert(names, "demucs"); table.insert(names, "hf-secret")
  end
  local labels = {
    python        = "Python (REAPER)",
    uv            = "uv",
    ["modal-cli"] = "Modal CLI",
    ["modal-auth"]= "Modal autenticado",
    demucs        = "demucs  (StemsSeparator)",
    ["hf-secret"] = "HF secret 'huggingface-secret'  (StemsSeparator)",
  }
  for _, name in ipairs(names) do
    table.insert(CHECKS, { name=name, label=labels[name] or name,
                            status="?", detail="" })
    CHECK_IDX[name] = #CHECKS
  end
end

init_checks()

local ST = {
  check_state     = "idle",
  uv_state        = "idle", uv_msg        = "",
  sync_state      = "idle", sync_msg      = "",
  login_state     = "idle", login_msg     = "",
  demucs_state    = "idle", demucs_msg    = "",
  hf_state        = "idle", hf_msg        = "",
  hf_token        = "",
  prw_sam_state   = "idle", prw_sam_msg   = "", prw_sam_pct   = 0.0,
  prw_mir_state   = "idle", prw_mir_msg   = "", prw_mir_pct   = 0.0,
  prw_yt3_state   = "idle", prw_yt3_msg   = "", prw_yt3_pct   = 0.0,
  -- MidiGenerator
  prw_mg_amadeus_state      = "idle", prw_mg_amadeus_msg      = "", prw_mg_amadeus_pct      = 0.0,
  prw_mg_midi_llm_state     = "idle", prw_mg_midi_llm_msg     = "", prw_mg_midi_llm_pct     = 0.0,
  prw_mg_text2midi_state    = "idle", prw_mg_text2midi_msg    = "", prw_mg_text2midi_pct    = 0.0,
  prw_mg_chatmusician_state = "idle", prw_mg_chatmusician_msg = "", prw_mg_chatmusician_pct = 0.0,
  prw_mg_musecoco_state     = "idle", prw_mg_musecoco_msg     = "", prw_mg_musecoco_pct     = 0.0,
  prw_mg_anticipatory_state = "idle", prw_mg_anticipatory_msg = "", prw_mg_anticipatory_pct = 0.0,
}

-- ── POLLING ──────────────────────────────────────────────────────
local function poll_check()
  if ST.check_state ~= "running" then return end
  local r = common.read_progress_file(PF.check)
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
  local r = common.read_progress_file(PF[pf_key])
  if not r then return end
  if msg_key then ST[msg_key] = r.msg or "" end
  if r.state == "done" or r.state == "error" then
    ST[state_key] = r.state
  end
end

local function poll_prewarm(state_key, msg_key, pct_key, pf_key)
  if ST[state_key] ~= "running" then return end
  local r = common.read_progress_file(PF[pf_key])
  if not r then return end
  ST[msg_key] = r.msg or ""; ST[pct_key] = r.pct or 0
  if r.state == "done" or r.state == "error" then
    ST[state_key] = r.state
  end
end

local function run_check()
  ST.check_state = "running"
  for _, c in ipairs(CHECKS) do c.status = "?"; c.detail = "" end
  local f = io.open(PF.check, "w")
  if f then f:write("running|0.00|..."); f:close() end
  local log = TMPDIR .. "reaperai_setup.log"
  local cmd = string.format('%s %s check --progress %s >>%s 2>&1 &',
    q(PYTHON), q(HELPER), q(PF.check), q(log))
  os.execute(cmd)
end

-- ── STATUS HELPERS ───────────────────────────────────────────────
local function status_color(s)
  if s == "ok"       then return "GREEN"
  elseif s == "missing" then return "RED"
  elseif s == "?"    then return "FG_DIM"
  else                    return "YELLOW" end
end

local function status_icon(s)
  if s == "ok"       then return "OK "
  elseif s == "missing" then return "X  "
  else                    return "..." end
end

-- ── GFX INIT ─────────────────────────────────────────────────────
if gfx.w > 0 then gfx.quit() end
local LOGICAL_W = 520
gfx.init("REAPER AI Plugins — Configuración", LOGICAL_W, 680)
gfx.ext_retina = 1
theme.init_fonts()

local _scale_init   = false
local first_frame   = true
local pending_recheck = false

-- ── DRAW HELPERS ─────────────────────────────────────────────────
local function draw_prewarm_row(label, cost_hint, state_key, msg_key, pct_key, subcmd)
  local g, t = gui, theme
  g.text_disabled(cost_hint)
  g.begin_disabled(ST[state_key] == "running")
  g.next_width(-1)
  if g.button(label) then
    ST[state_key] = "running"; ST[msg_key] = ""; ST[pct_key] = 0
    local pf_key = subcmd:gsub("-", "_"):gsub("prewarm_", "prw_"):gsub("_$","")
    -- map subcmd → PF key
    local pf_map = {
      ["prewarm-sam"]     = "prw_sam_pf",
      ["prewarm-miros"]   = "prw_mir_pf",
      ["prewarm-yourmt3"] = "prw_yt3_pf",
    }
    -- launch with the right PF
    local pf_keys = {
      ["prewarm-sam"]     = "prewarm_sam",
      ["prewarm-miros"]   = "prewarm_mir",
      ["prewarm-yourmt3"] = "prewarm_yt3",
    }
    launch(pf_keys[subcmd] or "prewarm_sam", subcmd)
  end
  g.end_disabled()
  if ST[state_key] == "running" then
    g.progress_bar(ST[pct_key], nil, t.sc(14),
      string.format("%d%%", math.floor(ST[pct_key] * 100)))
    if ST[msg_key] ~= "" then
      g.text_colored(ST[msg_key]:sub(1,70), "YELLOW")
    end
  elseif ST[state_key] == "done" then
    g.text_colored("OK  Descarga completada", "GREEN")
  elseif ST[state_key] == "error" then
    g.text_colored("Error: " .. ST[msg_key]:sub(1,60), "RED")
  end
end

-- ── MAIN LOOP ────────────────────────────────────────────────────
local function loop()
  if not _scale_init then
    _scale_init = true
    local s = math.floor(gfx.w / LOGICAL_W + 0.5)
    if s > 1 then theme.apply_scale(s); theme.init_fonts(s) end
  end

  gui.frame_begin()
  if gui.ctx.should_close then gfx.quit(); return end

  local g, t = gui, theme

  if first_frame then first_frame = false; run_check() end

  -- Poll all async ops
  poll_check()
  poll_simple("uv_state",     "uv_msg",     "uv")
  poll_simple("sync_state",   "sync_msg",   "sync")
  poll_simple("login_state",  "login_msg",  "login")
  poll_simple("demucs_state", "demucs_msg", "demucs")
  poll_simple("hf_state",     "hf_msg",     "hfsecret")
  poll_prewarm("prw_sam_state", "prw_sam_msg", "prw_sam_pct", "prewarm_sam")
  poll_prewarm("prw_mir_state", "prw_mir_msg", "prw_mir_pct", "prewarm_mir")
  poll_prewarm("prw_yt3_state", "prw_yt3_msg", "prw_yt3_pct", "prewarm_yt3")
  -- MidiGenerator prewarms
  if HAS_MG then
    poll_prewarm("prw_mg_amadeus_state",      "prw_mg_amadeus_msg",      "prw_mg_amadeus_pct",      "prw_mg_amadeus")
    poll_prewarm("prw_mg_midi_llm_state",     "prw_mg_midi_llm_msg",     "prw_mg_midi_llm_pct",     "prw_mg_midi_llm")
    poll_prewarm("prw_mg_text2midi_state",    "prw_mg_text2midi_msg",    "prw_mg_text2midi_pct",    "prw_mg_text2midi")
    poll_prewarm("prw_mg_chatmusician_state", "prw_mg_chatmusician_msg", "prw_mg_chatmusician_pct", "prw_mg_chatmusician")
    poll_prewarm("prw_mg_musecoco_state",     "prw_mg_musecoco_msg",     "prw_mg_musecoco_pct",     "prw_mg_musecoco")
    poll_prewarm("prw_mg_anticipatory_state", "prw_mg_anticipatory_msg", "prw_mg_anticipatory_pct", "prw_mg_anticipatory")
  end

  -- Recheck after action finishes
  local function maybe_recheck(key)
    if ST[key] == "done" then ST[key] = "idle"; pending_recheck = true end
  end
  maybe_recheck("uv_state"); maybe_recheck("sync_state")
  maybe_recheck("demucs_state"); maybe_recheck("login_state"); maybe_recheck("hf_state")
  if pending_recheck and ST.check_state ~= "running" then
    pending_recheck = false; run_check()
  end

  -- ── HEADER ─────────────────────────────────────────────────────
  g.push_font(t.F.H1)
  g.text("REAPER AI — Configuración global")
  g.pop_font()
  g.spacing()

  g.begin_disabled(ST.check_state == "running")
  g.next_width(-1)
  if g.button("Comprobar todo de nuevo") then run_check() end
  g.end_disabled()
  g.spacing()
  g.separator()

  -- ── SCROLL REGION: todo el contenido de secciones ──────────────
  local scroll_h = math.max(t.sc(60), gfx.h - gui.ctx.y - t.PAD_Y)
  widgets.scroll_region("##setup_main", 0, scroll_h, function()

  g.spacing()

  -- ── SECCIÓN: ENTORNO COMÚN ─────────────────────────────────────
  g.push_font(t.F.H1)
  g.text("Entorno común")
  g.pop_font()
  g.spacing()

  local checking = ST.check_state == "running"
  local uv_ok = CHECKS[CHECK_IDX["uv"]] and CHECKS[CHECK_IDX["uv"]].status == "ok"

  local core_names_set = {}
  for _, n in ipairs(CORE_NAMES) do core_names_set[n] = true end

  for _, c in ipairs(CHECKS) do
    if core_names_set[c.name] then
      local icon  = checking and "..." or status_icon(c.status)
      local color = checking and "YELLOW" or status_color(c.status)
      g.text_colored("[" .. icon .. "]", color)
      g.same_line(6)
      g.text(c.label)
      if c.detail ~= "" then
        g.same_line(10); g.text_disabled(c.detail:sub(1, 50))
      end

      if not checking then
        if c.name == "uv" and c.status == "missing" then
          g.same_line()
          g.begin_disabled(ST.uv_state == "running")
          if g.button("Instalar uv", t.sc(90), t.ITEM_H) then
            ST.uv_state = "running"; ST.uv_msg = ""; launch("uv", "install-uv")
          end
          g.end_disabled()
          if ST.uv_state == "running" then
            g.same_line(6)
            g.text_colored(ST.uv_msg ~= "" and ST.uv_msg:sub(1,40) or "Instalando...", "YELLOW")
          elseif ST.uv_state == "error" then
            g.same_line(6); g.text_colored("Error", "RED")
          end

        elseif c.name == "modal-cli" and c.status == "missing" and uv_ok then
          g.same_line()
          g.begin_disabled(ST.sync_state == "running")
          if g.button("Instalar deps", t.sc(102), t.ITEM_H) then
            ST.sync_state = "running"; ST.sync_msg = ""; launch("sync", "sync-deps")
          end
          g.end_disabled()
          if ST.sync_state == "running" then
            g.same_line(6)
            g.text_colored(ST.sync_msg ~= "" and ST.sync_msg:sub(1,38) or "Instalando...", "YELLOW")
          elseif ST.sync_state == "error" then
            g.same_line(6); g.text_colored("Error", "RED")
          end

        elseif c.name == "modal-auth" and c.status == "missing" then
          g.same_line()
          g.begin_disabled(ST.login_state == "running")
          if g.button("Login", t.sc(50), t.ITEM_H) then
            ST.login_state = "running"; ST.login_msg = ""; launch("login", "modal-login")
          end
          g.end_disabled()
          if ST.login_state == "running" then
            local msg = ST.login_msg ~= "" and ST.login_msg:sub(1,40) or "Abriendo navegador..."
            g.same_line(6); g.text_colored(msg, "YELLOW")
          elseif ST.login_state == "error" then
            g.same_line(6); g.text_colored("Error — reintenta", "RED")
          end
        end
      end
    end
  end

  -- ── SECCIÓN: STEMS SEPARATOR ───────────────────────────────────
  if HAS_STEMS then
    g.spacing(); g.separator(); g.spacing()
    g.push_font(t.F.H1)
    g.text("StemsSeparator")
    g.pop_font()
    g.spacing()

    -- Checks específicos de StemsSeparator
    for _, c in ipairs(CHECKS) do
      if c.name == "demucs" or c.name == "hf-secret" then
        local icon  = checking and "..." or status_icon(c.status)
        local color = checking and "YELLOW" or status_color(c.status)
        g.text_colored("[" .. icon .. "]", color)
        g.same_line(6); g.text(c.label)
        if c.detail ~= "" then
          g.same_line(10); g.text_disabled(c.detail:sub(1, 48))
        end
        if not checking and c.name == "demucs" and c.status == "missing" then
          g.same_line()
          g.begin_disabled(ST.demucs_state == "running")
          if g.button("Instalar", t.sc(70), t.ITEM_H) then
            ST.demucs_state = "running"; ST.demucs_msg = ""
            launch("demucs", "install-demucs --python " .. q(PYTHON))
          end
          g.end_disabled()
          if ST.demucs_state == "running" then
            g.same_line(6); g.text_colored("Instalando...", "YELLOW")
          elseif ST.demucs_state == "error" then
            g.same_line(6); g.text_colored("Error — revisa el log", "RED")
          end
        end
      end
    end

    g.spacing()
    -- HF Token
    g.text("Hugging Face — secret para SAM Audio")
    if g.button("Crear token en huggingface.co/settings/tokens", -1, t.BTN_H) then
      os.execute('open "https://huggingface.co/settings/tokens" &')
    end
    g.spacing()
    g.row_label("Token HF:", t.sc(72))
    local hf_changed, hf_new = widgets.input_text("##hftoken", ST.hf_token,
      { password = true })
    if hf_changed then ST.hf_token = hf_new end
    local can_save = ST.hf_token:sub(1,3) == "hf_" and ST.hf_state ~= "running"
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
      g.text_colored("OK  " .. (ST.hf_msg ~= "" and ST.hf_msg:sub(1,55) or "Guardado"), "GREEN")
    elseif ST.hf_state == "error" then
      g.text_colored("Error: " .. ST.hf_msg:sub(1,55), "RED")
    end

    g.spacing()
    g.text("Pre-cargar SAM Audio (opcional)")
    g.text_disabled("~14 GB  |  primera vez 10-20 min  |  queda cacheado en Modal")
    g.begin_disabled(ST.prw_sam_state == "running")
    g.next_width(-1)
    if g.button("Descargar facebook/sam-audio-large") then
      ST.prw_sam_state = "running"; ST.prw_sam_pct = 0; ST.prw_sam_msg = ""
      launch("prewarm_sam", "prewarm-sam --model facebook/sam-audio-large")
    end
    g.end_disabled()
    if ST.prw_sam_state == "running" then
      g.progress_bar(ST.prw_sam_pct, nil, t.sc(14),
        string.format("%d%%", math.floor(ST.prw_sam_pct * 100)))
      if ST.prw_sam_msg ~= "" then
        g.text_colored(ST.prw_sam_msg:sub(1,68), "YELLOW")
      end
    elseif ST.prw_sam_state == "done" then
      g.text_colored("OK  Descarga SAM completada", "GREEN")
    elseif ST.prw_sam_state == "error" then
      g.text_colored("Error: " .. ST.prw_sam_msg:sub(1,55), "RED")
    end
  end

  -- ── SECCIÓN: AUDIO2MIDI ─────────────────────────────────────────
  if HAS_A2M then
    g.spacing(); g.separator(); g.spacing()
    g.push_font(t.F.H1)
    g.text("Audio2Midi")
    g.pop_font()
    g.spacing()
    g.text_colored("[OK]", "GREEN")
    g.same_line(6)
    g.text("Sin dependencias locales — pesos en Modal Volumes")
    g.spacing()

    -- Prewarm MIROS
    g.text("Pre-cargar MIROS (opcional, recomendado)")
    g.text_disabled("~8 GB  |  primera vez 10-15 min  |  requiere A10G en Modal")
    g.begin_disabled(ST.prw_mir_state == "running")
    g.next_width(-1)
    if g.button("Descargar pesos MIROS") then
      ST.prw_mir_state = "running"; ST.prw_mir_pct = 0; ST.prw_mir_msg = ""
      launch("prewarm_mir", "prewarm-miros")
    end
    g.end_disabled()
    if ST.prw_mir_state == "running" then
      g.progress_bar(ST.prw_mir_pct, nil, t.sc(14),
        string.format("%d%%", math.floor(ST.prw_mir_pct * 100)))
      if ST.prw_mir_msg ~= "" then
        g.text_colored(ST.prw_mir_msg:sub(1,68), "YELLOW")
      end
    elseif ST.prw_mir_state == "done" then
      g.text_colored("OK  MIROS cacheado en Modal Volume", "GREEN")
    elseif ST.prw_mir_state == "error" then
      g.text_colored("Error MIROS: " .. ST.prw_mir_msg:sub(1,50), "RED")
    end

    g.spacing()
    -- Prewarm YourMT3+
    g.text("Pre-cargar YourMT3+ (opcional)")
    g.text_disabled("~2 GB  |  primera vez 5-10 min")
    g.begin_disabled(ST.prw_yt3_state == "running")
    g.next_width(-1)
    if g.button("Descargar pesos YourMT3+") then
      ST.prw_yt3_state = "running"; ST.prw_yt3_pct = 0; ST.prw_yt3_msg = ""
      launch("prewarm_yt3", "prewarm-yourmt3")
    end
    g.end_disabled()
    if ST.prw_yt3_state == "running" then
      g.progress_bar(ST.prw_yt3_pct, nil, t.sc(14),
        string.format("%d%%", math.floor(ST.prw_yt3_pct * 100)))
      if ST.prw_yt3_msg ~= "" then
        g.text_colored(ST.prw_yt3_msg:sub(1,68), "YELLOW")
      end
    elseif ST.prw_yt3_state == "done" then
      g.text_colored("OK  YourMT3+ cacheado en Modal Volume", "GREEN")
    elseif ST.prw_yt3_state == "error" then
      g.text_colored("Error YourMT3+: " .. ST.prw_yt3_msg:sub(1,48), "RED")
    end
  end

  -- ── SECCIÓN: MIDI GENERATOR ────────────────────────────────────
  if HAS_MG then
    g.spacing(); g.separator(); g.spacing()
    g.push_font(t.F.H1)
    g.text("MidiGenerator")
    g.pop_font()
    g.spacing()
    g.text_colored("[OK]", "GREEN")
    g.same_line(6)
    g.text("Sin dependencias locales — pesos en Modal Volumes")
    g.spacing()
    g.text("Pre-cargar pesos en Modal (opcional — primera vez puede tardar 10-20 min)")
    g.text_disabled("Recomendado: Amadeus + MIDI-LLM + Anticipatory. MuseCoco = ~16 GB.")
    g.spacing()

    local MG_PREWARM = {
      { key="amadeus",      label="Descargar Amadeus (~2.5 GB)",      cost="A10G",  state="prw_mg_amadeus_state",      msg="prw_mg_amadeus_msg",      pct="prw_mg_amadeus_pct",      subcmd="prewarm-midigen-amadeus" },
      { key="midi_llm",     label="Descargar MIDI-LLM (~3.5 GB)",     cost="A10G",  state="prw_mg_midi_llm_state",     msg="prw_mg_midi_llm_msg",     pct="prw_mg_midi_llm_pct",     subcmd="prewarm-midigen-midi-llm" },
      { key="text2midi",    label="Descargar text2midi (~900 MB)",     cost="A10G",  state="prw_mg_text2midi_state",    msg="prw_mg_text2midi_msg",    pct="prw_mg_text2midi_pct",    subcmd="prewarm-midigen-text2midi" },
      { key="chatmusician", label="Descargar ChatMusician (~13 GB)",   cost="A10G",  state="prw_mg_chatmusician_state", msg="prw_mg_chatmusician_msg", pct="prw_mg_chatmusician_pct", subcmd="prewarm-midigen-chatmusician" },
      { key="musecoco",     label="Descargar MuseCoco (~16 GB)",       cost="A100",  state="prw_mg_musecoco_state",     msg="prw_mg_musecoco_msg",     pct="prw_mg_musecoco_pct",     subcmd="prewarm-midigen-musecoco" },
      { key="anticipatory", label="Descargar Anticipatory (~1.6 GB)",  cost="A10G",  state="prw_mg_anticipatory_state", msg="prw_mg_anticipatory_msg", pct="prw_mg_anticipatory_pct", subcmd="prewarm-midigen-anticipatory" },
    }

    for _, m in ipairs(MG_PREWARM) do
      local pf_key = m.state:gsub("_state$", ""):gsub("prw_", "prw_")
      g.begin_disabled(ST[m.state] == "running")
      g.next_width(-1)
      if g.button(m.label .. "  [" .. m.cost .. "]") then
        ST[m.state] = "running"; ST[m.msg] = ""; ST[m.pct] = 0
        -- launch usando la PF key correcta
        local pf_map_key = m.state:gsub("_state$","")
        local f = io.open(PF[pf_map_key], "w")
        if f then f:write("running|0.00|Iniciando..."); f:close() end
        local log = TMPDIR .. "reaperai_setup.log"
        local cmd = string.format('%s %s %s --progress %s >>%s 2>&1 &',
          q(PYTHON), q(HELPER), m.subcmd, q(PF[pf_map_key]), q(log))
        os.execute(cmd)
      end
      g.end_disabled()
      if ST[m.state] == "running" then
        g.progress_bar(ST[m.pct], nil, t.sc(12),
          string.format("%d%%", math.floor(ST[m.pct] * 100)))
        if ST[m.msg] ~= "" then
          g.text_colored(ST[m.msg]:sub(1,68), "YELLOW")
        end
      elseif ST[m.state] == "done" then
        g.text_colored("OK  cacheado", "GREEN")
      elseif ST[m.state] == "error" then
        g.text_colored("Error: " .. ST[m.msg]:sub(1,60), "RED")
      end
    end
  end

  g.spacing(); g.separator(); g.spacing()
  g.text_disabled("shared/ — Python: " .. PYTHON)
  g.spacing()

  end)  -- fin scroll_region

  gui.frame_end()
  reaper.defer(loop)
end

reaper.defer(loop)
