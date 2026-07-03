-- lib/common.lua  Non-UI helpers shared between StemSeparator.lua and Setup.lua.

local M = {}

M.TMPDIR = os.getenv("TMPDIR") or "/tmp/"
M.HOME   = os.getenv("HOME")   or ""

function M.q(s)
  return '"' .. s:gsub('"', '\\"') .. '"'
end

-- Returns (python_path, err_or_nil). Falls back to "python3" on failure.
function M.detect_reaper_python()
  local ini = reaper.GetResourcePath() .. "/reaper.ini"
  local f = io.open(ini, "r")
  if not f then return "python3", "Cannot open " .. ini end
  local libpath
  for line in f:lines() do
    local k, v = line:match("^(pythonlibpath64)=(.*)$")
    if not k then k, v = line:match("^(pythonlibpath32)=(.*)$") end
    if k and v ~= "" then libpath = v; break end
  end
  f:close()
  if not libpath then return "python3", "pythonlibpath not in reaper.ini" end
  local parent = libpath:match("^(.*)/lib$")
              or libpath:match("^(.+)/lib[^/]*$")
              or libpath
  local exe = parent .. "/bin/python3"
  local tf = io.open(exe, "r")
  if tf then tf:close(); return exe, nil end
  return "python3", "Not found: " .. exe
end

-- Returns nil if file missing/empty/unparseable, else:
--   { state, pct (0..1), msg, extra = {lines} }
function M.read_progress_file(path)
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
  if not r.state then return nil end
  return r
end

-- Writes "running|0.00|Starting..." to progress_path then launches:
--   python script_path [extra_args] --progress progress_path >> log_path 2>&1 &
function M.launch_async(python, script_path, extra_args, progress_path, log_path)
  local f = io.open(progress_path, "w")
  if f then f:write("running|0.00|Starting..."); f:close() end
  local q = M.q
  local cmd
  if reaper.GetOS():find("^Win") then
    cmd = string.format('start /B "" %s %s %s --progress %s 1>>%s 2>&1',
      q(python), q(script_path), extra_args, q(progress_path), q(log_path))
  else
    cmd = string.format('%s %s %s --progress %s >>%s 2>&1 &',
      q(python), q(script_path), extra_args, q(progress_path), q(log_path))
  end
  os.execute(cmd)
end

return M
