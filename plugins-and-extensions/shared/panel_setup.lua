-- AI Music Lab - Setup panel (global configuration wizard)
-- Panel module: no gfx.init / reaper.defer of its own. Loaded by the
-- unified ai-music-lab.lua container, which owns the window/main loop
-- and calls M.init() once, M.poll() every frame, M.draw() when active.
-- Manages the common environment (uv, Modal) and plugin-specific extras
-- (StemsSeparator: demucs, HF secret).
-- "Prepare models": the first inference with a model is slow (Modal image
-- build + weights download, or local environment + weights). Each target
-- (model x execution) can be warmed up here with the smallest real job via
-- `setup_helpers.py prepare`; the result is recorded in
-- ~/.ai-musiclab/prepared.txt (lib/prepared.lua), which every plugin panel reads
-- to warn about a first slow run.

-- ── PATHS + LIB ──────────────────────────────────────────────────
local _info      = debug.getinfo(1, "S")
local SCRIPT_DIR = _info.source:match("@?(.*[/\\])") or ""  -- shared/

-- Derive plugins directory (sibling of shared/)
local PLUGINS_DIR = SCRIPT_DIR:match("^(.*[/\\])shared[/\\]$") or (SCRIPT_DIR .. "../")

package.path = SCRIPT_DIR .. "lib/?.lua;" .. package.path

local common  = require("common")
local theme   = require("theme")
local gui     = require("gui")
local widgets = require("widgets_extra")
local prepared = require("prepared")
local pq       = require("prepare_queue")

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
  test = io.open(PLUGINS_DIR .. name .. "/panel.lua", "r")
  if test then test:close(); return true end
  return false
end

local HAS_STEMS = plugin_exists("StemsSeparator")

local PF = {
  check        = TMPDIR .. "reaperai_setup_check.txt",
  uv           = TMPDIR .. "reaperai_setup_uv.txt",
  sync         = TMPDIR .. "reaperai_setup_sync.txt",
  login        = TMPDIR .. "reaperai_setup_login.txt",
  demucs       = TMPDIR .. "reaperai_setup_demucs.txt",
  hfsecret     = TMPDIR .. "reaperai_setup_hfsecret.txt",
}

-- Background command for one helper subcommand. If the interpreter dies
-- before the helper can report anything (ImportError, SyntaxError, a Python
-- too old for the script...), the progress file would stay on "running"
-- forever and the user would only ever see a bare "Error"/spinner. The
-- fallback turns that into an "error" line carrying the last log line.
local function build_cmd(subcmd_args, pf)
  local log = TMPDIR .. "reaperai_setup.log"
  return string.format(
    "( %s %s %s --progress %s >>%s 2>&1 || { head -n1 %s | grep -q '^running' && " ..
    "printf 'error|0.00|%%s\\n' \"$(tail -n1 %s | cut -c1-160)\" > %s; } ) &",
    q(PYTHON), q(HELPER), subcmd_args, q(pf), q(log), q(pf), q(log), q(pf))
end

local function launch(key, subcmd_args)
  local pf  = PF[key]
  local f = io.open(pf, "w")
  if f then f:write("running|0.00|Starting..."); f:close() end
  os.execute(build_cmd(subcmd_args, pf))
end

-- ── STATE ────────────────────────────────────────────────────────
-- Core checks always present; plugin extras detected at runtime
local CORE_NAMES = { "python", "uv", "modal-cli", "modal-auth" }

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
    ["modal-auth"]= "Modal authenticated",
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

local dismiss_failures  -- defined with the prepare scheduler below
local PREPS = {}  -- filled from PREP| lines: { domain, model, execution, label, status, hint }
local PREP_DOMAINS = {
  { "text2audio",    "Text2Audio" },
  { "midigen",       "MidiGenerator" },
  { "audio2midi",    "Audio2Midi" },
  { "stemseparator", "StemsSeparator (SAM Audio)" },
}

local ST = {
  prep_queue = {},   -- targets waiting for a free slot (see lib/prepare_queue.lua)
  prep_jobs  = {},   -- { target, pf, state, msg, report } in flight or failed
  check_state     = "idle", check_msg     = "",
  uv_state        = "idle", uv_msg        = "",
  sync_state      = "idle", sync_msg      = "",
  login_state     = "idle", login_msg     = "",
  demucs_state    = "idle", demucs_msg    = "",
  hf_state        = "idle", hf_msg        = "",
  hf_token        = "",
}

-- ── POLLING ──────────────────────────────────────────────────────
local function poll_check()
  if ST.check_state ~= "running" then return end
  local r = common.read_progress_file(PF.check)
  if not r then return end
  if r.state == "done" then
    ST.check_state = "done"
    PREPS = {}
    prepared.invalidate()
    for _, line in ipairs(r.extra) do
      local pd, pm, pe, pl, ps, pw, pg, ph = line:match("^PREP|([^|]*)|([^|]*)|([^|]*)|([^|]*)|([^|]*)|([^|]*)|([^|]*)|(.-)$")
      if pd then
        if ps == "ready" then dismiss_failures(pd .. ":" .. pm .. ":" .. pe) end
        PREPS[#PREPS + 1] = { domain = pd, model = pm, execution = pe,
                              label = pl, status = ps, heavy = pw == "heavy", group = pg, hint = ph }
      end
      local name, status, detail = line:match("^CHECK|([^|]+)|([^|]+)|(.-)$")
      if name and CHECK_IDX[name] then
        local c = CHECKS[CHECK_IDX[name]]
        c.status = status or "?"; c.detail = detail or ""
      end
    end
  elseif r.state == "error" then
    ST.check_state = "error"
    ST.check_msg = r.msg or ""
  end
end

local function poll_simple(state_key, msg_key, pf_key)
  if ST[state_key] ~= "running" then return end
  local r = common.read_progress_file(PF[pf_key])
  if not r then return end
  if msg_key then ST[msg_key] = r.msg or "" end
  if r.state == "done" or r.state == "error" then
    ST[state_key] = r.state
    ST[state_key .. "_report"] = nil
    for _, line in ipairs(r.extra) do
      local path = line:match("^REPORT|(.+)$")
      if path then ST[state_key .. "_report"] = path end
    end
  end
end

local pending_recheck = false
local prep_seq = 0

local function prep_key(p) return pq.key(p) end

-- Job of a target; failed ones are only returned with include_failed (a failed
-- target must stay retryable, so it does not count as "in progress").
local function prep_job(p, include_failed)
  for _, j in ipairs(ST.prep_jobs) do
    if prep_key(j.target) == prep_key(p) and (include_failed or j.state == "running") then
      return j
    end
  end
end

dismiss_failures = function(only_key)
  for i = #ST.prep_jobs, 1, -1 do
    local j = ST.prep_jobs[i]
    if j.state == "error" and (not only_key or prep_key(j.target) == only_key) then
      table.remove(ST.prep_jobs, i)
    end
  end
end

local function prep_queued(p)
  for _, q_ in ipairs(ST.prep_queue) do
    if prep_key(q_) == prep_key(p) then return true end
  end
  return false
end

-- Queue targets (skipping ones already queued/running); the scheduler in
-- M.poll starts them as slots free up.
local function enqueue_prepare(list)
  -- A new batch replaces previous failures on screen.
  dismiss_failures()
  for _, p in ipairs(list) do
    if not prep_job(p) and not prep_queued(p) then
      table.insert(ST.prep_queue, p)
    end
  end
end

local function start_job(p)
  prep_seq = prep_seq + 1
  local job = { target = p, state = "running", msg = "",
                pf = TMPDIR .. "reaperai_setup_prepare_" .. prep_seq .. ".txt" }
  table.insert(ST.prep_jobs, job)
  local f = io.open(job.pf, "w")
  if f then f:write("running|0.00|Starting..."); f:close() end
  os.execute(build_cmd("prepare --target " .. q(prep_key(p)), job.pf))
end

local function running_targets()
  local list = {}
  for _, j in ipairs(ST.prep_jobs) do
    if j.state == "running" then list[#list + 1] = j.target end
  end
  return list
end

local function prep_active(execution)
  for _, j in ipairs(ST.prep_jobs) do
    if j.state == "running" and j.target.execution == execution then return true end
  end
  for _, p in ipairs(ST.prep_queue) do
    if p.execution == execution then return true end
  end
  return false
end

local function poll_prepare()
  local finished = false
  for i = #ST.prep_jobs, 1, -1 do
    local j = ST.prep_jobs[i]
    if j.state == "running" then
      local r = common.read_progress_file(j.pf)
      if r then
        j.msg = r.msg or ""
        if r.state == "done" then
          table.remove(ST.prep_jobs, i); finished = true
        elseif r.state == "error" then
          j.state = "error"; finished = true
          for _, line in ipairs(r.extra) do
            local path = line:match("^REPORT|(.+)$")
            if path then j.report = path end
          end
          ST.prep_queue = {}  -- likely network/auth: do not start more
        end
      end
    end
  end
  if finished then prepared.invalidate() end

  local picked = pq.pick(ST.prep_queue, running_targets())
  local starting = {}
  for _, idx in ipairs(picked) do starting[#starting + 1] = ST.prep_queue[idx] end
  for n = #picked, 1, -1 do table.remove(ST.prep_queue, picked[n]) end
  for _, p in ipairs(starting) do start_job(p) end

  if finished and #ST.prep_queue == 0 and #running_targets() == 0 then
    pending_recheck = true
  end
end

local function run_check()
  ST.check_state = "running"
  for _, c in ipairs(CHECKS) do c.status = "?"; c.detail = "" end
  local f = io.open(PF.check, "w")
  if f then f:write("running|0.00|..."); f:close() end
  os.execute(build_cmd("check", PF.check))
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


-- Error line + path of the diagnostic report written by setup_helpers.py,
-- with a button to open it (so the user can debug it or paste it to an assistant).
local function error_details(state_key, msg)
  local g, t = gui, theme
  g.text_wrapped(msg ~= "" and msg or "Failed — see reaperai_setup.log")
  local report = ST[state_key .. "_report"]
  if report then
    g.text_disabled("Full report: " .. report)
    if g.button("Open report##" .. state_key, t.sc(110), t.ITEM_H) then
      os.execute('open ' .. q(report) .. ' &')
    end
  end
end

-- ── MODULE ───────────────────────────────────────────────────────
local M = { title = "Setup" }

function M.init()
  run_check()
end

function M.poll()
  poll_check()
  poll_simple("uv_state",     "uv_msg",     "uv")
  poll_simple("sync_state",   "sync_msg",   "sync")
  poll_simple("login_state",  "login_msg",  "login")
  poll_simple("demucs_state", "demucs_msg", "demucs")
  poll_simple("hf_state",     "hf_msg",     "hfsecret")
  poll_prepare()

  -- Recheck after action finishes
  local function maybe_recheck(key)
    if ST[key] == "done" then ST[key] = "idle"; pending_recheck = true end
  end
  maybe_recheck("uv_state"); maybe_recheck("sync_state")
  maybe_recheck("demucs_state"); maybe_recheck("login_state"); maybe_recheck("hf_state")
  if pending_recheck and ST.check_state ~= "running" then
    pending_recheck = false; run_check()
  end
end

-- Returns a list of missing core check labels — used by the container
-- for the "incomplete setup" banner shown on the other tabs.
function M.get_missing()
  local missing = {}
  local LABELS = {
    python         = "Python REAPER",
    uv             = "uv",
    ["modal-cli"]  = "Modal CLI",
    ["modal-auth"] = "Modal not authenticated",
  }
  if ST.check_state ~= "done" then return missing end
  for _, c in ipairs(CHECKS) do
    if LABELS[c.name] and c.status == "missing" then
      table.insert(missing, LABELS[c.name])
    end
  end
  return missing
end

function M.draw()
  local g, t = gui, theme

  g.begin_disabled(ST.check_state == "running")
  g.next_width(-1)
  if g.button("Re-check all") then run_check() end
  g.end_disabled()
  g.spacing()
  g.separator()

  -- ── SCROLL REGION: all section content ─────────────────────────
  local scroll_h = math.max(t.sc(60), gfx.h - gui.ctx.y - t.PAD_Y)
  widgets.scroll_region("##setup_main", 0, scroll_h, function()

  g.spacing()

  -- ── SECTION: COMMON ENVIRONMENT ────────────────────────────────
  g.push_font(t.F.H1)
  g.text("Common environment")
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
          g.same_line_or_wrap(t.sc(90))
          g.begin_disabled(ST.uv_state == "running")
          if g.button("Install uv", t.sc(90), t.ITEM_H) then
            ST.uv_state = "running"; ST.uv_msg = ""; launch("uv", "install-uv")
          end
          g.end_disabled()
          if ST.uv_state == "running" then
            g.same_line_or_wrap(t.sc(80), 6)
            g.text_colored(ST.uv_msg ~= "" and ST.uv_msg:sub(1,40) or "Installing...", "YELLOW")
          elseif ST.uv_state == "error" then
            g.same_line_or_wrap(t.sc(40), 6); g.text_colored("Error", "RED")
            error_details("uv_state", ST.uv_msg)
          end

        elseif c.name == "modal-cli" and c.status == "missing" and uv_ok then
          g.same_line_or_wrap(t.sc(102))
          g.begin_disabled(ST.sync_state == "running")
          if g.button("Install deps", t.sc(102), t.ITEM_H) then
            ST.sync_state = "running"; ST.sync_msg = ""; launch("sync", "sync-deps")
          end
          g.end_disabled()
          if ST.sync_state == "running" then
            g.same_line_or_wrap(t.sc(80), 6)
            g.text_colored(ST.sync_msg ~= "" and ST.sync_msg:sub(1,38) or "Installing...", "YELLOW")
          elseif ST.sync_state == "error" then
            g.same_line_or_wrap(t.sc(40), 6); g.text_colored("Error", "RED")
            error_details("sync_state", ST.sync_msg)
          end

        elseif c.name == "modal-auth" and c.status == "missing" then
          g.same_line_or_wrap(t.sc(50))
          g.begin_disabled(ST.login_state == "running")
          if g.button("Login", t.sc(50), t.ITEM_H) then
            ST.login_state = "running"; ST.login_msg = ""; launch("login", "modal-login")
          end
          g.end_disabled()
          if ST.login_state == "running" then
            local msg = ST.login_msg ~= "" and ST.login_msg:sub(1,40) or "Opening browser..."
            g.same_line_or_wrap(t.sc(90), 6); g.text_colored(msg, "YELLOW")
          elseif ST.login_state == "error" then
            g.same_line_or_wrap(t.sc(80), 6); g.text_colored("Error — retry", "RED")
            error_details("login_state", ST.login_msg)
          end
        end
      end
    end
  end

  -- ── SECTION: PREPARE MODELS ─────────────────────────────────────
  if #PREPS > 0 then
    g.spacing(); g.separator(); g.spacing()
    g.push_font(t.F.H1)
    g.text("Prepare models")
    g.pop_font()
    g.text_wrapped("The first run of each model is slow: Modal builds its container "
      .. "image and downloads the weights (or, locally, the environment and weights "
      .. "are downloaded). Prepare the ones you plan to use now, so your first "
      .. "generation is fast.")
    g.spacing()

    local modal_ok = CHECKS[CHECK_IDX["modal-auth"]]
      and CHECKS[CHECK_IDX["modal-auth"]].status == "ok"
    -- Targets still to prepare for one execution. Ready ones are skipped;
    -- heavy ones (tens of GB, e.g. local SAM) only get their own button.
    local function pending_for(execution)
      local list = {}
      for _, p in ipairs(PREPS) do
        if p.execution == execution and p.status ~= "ready" and not p.heavy
            and not prep_job(p) and not prep_queued(p) then
          list[#list + 1] = p
        end
      end
      return list
    end
    local function prepare_all_button(execution, label, enabled)
      local pending = pending_for(execution)
      g.begin_disabled(checking or not enabled or #pending == 0)
      g.next_width(-1)
      if g.button(string.format("%s  (%s)", label,
          #pending == 0 and (prep_active(execution) and "in progress" or "all prepared")
            or (#pending .. " pending"))) then
        enqueue_prepare(pending)
      end
      g.end_disabled()
    end
    prepare_all_button("remote", "Prepare all Modal models", modal_ok)
    if not modal_ok then g.text_disabled("Log in to Modal first (Common environment).") end
    prepare_all_button("local", "Prepare all local models", true)
    g.text_disabled(string.format("Up to %d Modal models are prepared at once (one local at a time). "
      .. "Already prepared models are skipped; large local models (SAM Audio, tens of GB) "
      .. "are only prepared with their own button.", pq.MAX.remote))

    for _, j in ipairs(ST.prep_jobs) do
      local name = prep_key(j.target)
      if j.state == "running" then
        g.text_colored("Preparing " .. name, "YELLOW")
        g.same_line(8)
        g.text_disabled((j.msg ~= "" and j.msg or "Starting... a first image build can take 5-15 minutes."):sub(1, 70))
      else
        g.text_colored("Failed: " .. name, "RED")
        g.same_line(8)
        if g.button("Dismiss##dis_" .. name, t.sc(70), t.ITEM_H) then
          dismiss_failures(name)
        end
        g.text_wrapped(j.msg ~= "" and j.msg or "Failed — see reaperai_setup.log")
        if j.report then
          g.text_disabled("Full report: " .. j.report)
          if g.button("Open report##" .. name, t.sc(110), t.ITEM_H) then
            os.execute('open ' .. q(j.report) .. ' &')
          end
        end
      end
    end
    if #ST.prep_queue > 0 then
      g.text_disabled(#ST.prep_queue .. " queued")
    end
    g.spacing()

    for _, dom in ipairs(PREP_DOMAINS) do
      local first = true
      for _, p in ipairs(PREPS) do
        if p.domain == dom[1] then
          if first then
            g.text_disabled(dom[2]); first = false
          end
          local ready = p.status == "ready"
          local ctx = g.ctx
          local row_y = ctx.y
          -- The action column is right-aligned so every Prepare button lines up
          -- whatever the label length.
          local btn_w = t.sc(80)
          local btn_x = ctx.row_x0 + ctx.content_w - btn_w
          g.text_colored(ready and "[OK ]" or "[ ! ]", ready and "GREEN" or "YELLOW")
          g.same_line(6)
          g.text(p.label .. "  ·  " .. (p.execution == "local" and "local" or "Modal"))
          if not ready and p.hint ~= "" then
            -- Only when it still fits left of the button column.
            local next_x = ctx.last_x + ctx.last_w + t.sc(10)
            if next_x + gfx.measurestr(p.hint) + t.sc(10) <= btn_x then
              g.same_line(10); g.text_disabled(p.hint)
            end
          end
          if not ready then
            local job = prep_job(p)
            ctx.x, ctx.y = btn_x, row_y
            if job and job.state == "running" then
              g.text_colored("preparing...", "YELLOW")
            elseif prep_queued(p) then
              g.text_disabled("queued")
            else
              local can = not checking and (p.execution == "local" or modal_ok)
              g.begin_disabled(not can)
              if g.button("Prepare##prep_" .. p.domain .. p.model .. p.execution,
                          btn_w, t.ITEM_H) then
                enqueue_prepare({ p })
              end
              g.end_disabled()
            end
          end
          g.end_row()
        end
      end
    end
  end

  -- ── SECTION: STEMS SEPARATOR ────────────────────────────────────
  if HAS_STEMS then
    g.spacing(); g.separator(); g.spacing()
    g.push_font(t.F.H1)
    g.text("StemsSeparator")
    g.pop_font()
    g.spacing()

    -- StemsSeparator-specific checks
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
          g.same_line_or_wrap(t.sc(70))
          g.begin_disabled(ST.demucs_state == "running")
          if g.button("Install", t.sc(70), t.ITEM_H) then
            ST.demucs_state = "running"; ST.demucs_msg = ""
            launch("demucs", "install-demucs --python " .. q(PYTHON))
          end
          g.end_disabled()
          if ST.demucs_state == "running" then
            g.same_line_or_wrap(t.sc(80), 6); g.text_colored("Installing...", "YELLOW")
          elseif ST.demucs_state == "error" then
            g.same_line_or_wrap(t.sc(130), 6); g.text_colored("Error — check the log", "RED")
            error_details("demucs_state", ST.demucs_msg)
          end
        end
      end
    end

    g.spacing()
    -- HF Token
    g.text("Hugging Face — token for gated models")
    g.text_wrapped("Some models (SAM Audio, Stable Audio Open, MIDI-GPT) ask you to accept "
      .. "their license once. Do it with the same Hugging Face account that owns the token.")
    g.spacing()
    g.text_disabled("1. Accept the license of the models you plan to use:")
    for _, lic in ipairs({
      { "SAM Audio (large)",  "https://huggingface.co/facebook/sam-audio-large" },
      { "SAM Audio (base)",   "https://huggingface.co/facebook/sam-audio-base" },
      { "Stable Audio Open",  "https://huggingface.co/stabilityai/stable-audio-open-1.0" },
      { "MIDI-GPT",           "https://huggingface.co/Metacreation/MIDI-GPT" },
    }) do
      if g.button(lic[1] .. "##lic_" .. lic[1], -1, t.BTN_H, { icon = "link" }) then
        os.execute('open "' .. lic[2] .. '" &')
      end
    end
    g.spacing()
    g.text_disabled("2. Create a token and save it here:")
    if g.button("Create token at huggingface.co/settings/tokens", -1, t.BTN_H, { icon = "link" }) then
      os.execute('open "https://huggingface.co/settings/tokens" &')
    end
    g.spacing()
    g.row_label("HF Token:", t.sc(72))
    local hf_changed, hf_new = widgets.input_text("##hftoken", ST.hf_token,
      { password = true })
    if hf_changed then ST.hf_token = hf_new end
    local can_save = ST.hf_token:sub(1,3) == "hf_" and ST.hf_state ~= "running"
    g.begin_disabled(not can_save)
    g.next_width(-1)
    if g.button('Save as Modal secret "huggingface-secret"') then
      ST.hf_state = "running"; ST.hf_msg = ""
      launch("hfsecret", "modal-secret-create --token " .. q(ST.hf_token))
      ST.hf_token = ""
    end
    g.end_disabled()
    if ST.hf_state == "running" then
      g.text_colored("Saving secret to Modal...", "YELLOW")
    elseif ST.hf_state == "done" then
      g.text_colored("OK  " .. (ST.hf_msg ~= "" and ST.hf_msg:sub(1,55) or "Saved"), "GREEN")
    elseif ST.hf_state == "error" then
      g.text_colored("Error: " .. ST.hf_msg:sub(1,55), "RED")
    end
  end

  g.spacing(); g.separator(); g.spacing()
  g.text_disabled("shared/ — Python: " .. PYTHON)
  g.spacing()

  end)  -- end scroll_region
end

return M
