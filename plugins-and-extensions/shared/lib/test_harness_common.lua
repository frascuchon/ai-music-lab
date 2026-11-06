-- Lua-side harness for test_common.lua (shared/lib/common.lua).
--
-- Covers detect_reaper_python() / _find_versioned_python3() — the logic
-- that picks which Python binary every panel (StemsSeparator, Text2Audio,
-- Audio2Midi, MidiGenerator) hands to its backend scripts. A real user hit
-- "No module named 'soundfile'" plus a "WARNING: Not found: .../lib" line
-- from every plugin: their REAPER-configured Python (a Homebrew
-- python@3.10 Framework build) ships bin/python3.10 but no bare
-- bin/python3, so the literal "bin/python3" lookup always missed and
-- detect_reaper_python fell back to whatever bare "python3" happened to
-- resolve to on PATH (a completely different interpreter — miniconda's
-- python3.12 here) instead of the one Setup had actually installed
-- packages into.
--
-- common.lua requires process_control.lua and calls reaper.GetResourcePath/
-- reaper.GetOS — stub a minimal `reaper` global so this runs under the
-- plain `lua` interpreter (same approach as test_harness_process_control.lua
-- and test_harness_collapse_state.lua).

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path

reaper = {
  GetOS = function() return "macOS" end,
}

local common = require("common")

local failures = 0

local function case_(name, fn)
  local ok, err = pcall(fn)
  if ok then
    print("PASS " .. name)
  else
    failures = failures + 1
    print("FAIL " .. name .. ": " .. tostring(err))
  end
end

-- ---------------------------------------------------------------------
-- Temp fixture dirs on real disk (io.popen("ls ...") needs real files).
-- ---------------------------------------------------------------------
local function mktemp_dir(suffix)
  local pipe = io.popen("mktemp -d 2>/dev/null")
  local base = pipe:read("*l")
  pipe:close()
  if not base then error("mktemp -d failed") end
  return base
end

local function touch_exe(path)
  local f = io.open(path, "w")
  f:write("#!/bin/sh\n")
  f:close()
  os.execute('chmod +x "' .. path .. '"')
end

local function write_ini(dir, libpath)
  local f = io.open(dir .. "/reaper.ini", "w")
  f:write("[REAPER]\n")
  if libpath then
    f:write("pythonlibpath64=" .. libpath .. "\n")
  end
  f:close()
end

-- ---------------------------------------------------------------------
-- _find_versioned_python3
-- ---------------------------------------------------------------------
case_("_find_versioned_python3 finds a single versioned binary", function()
  local bindir = mktemp_dir()
  touch_exe(bindir .. "/python3.10")
  local found = common._find_versioned_python3(bindir)
  assert(found == bindir .. "/python3.10", tostring(found))
end)

case_("_find_versioned_python3 picks the highest version when several exist", function()
  local bindir = mktemp_dir()
  touch_exe(bindir .. "/python3.9")
  touch_exe(bindir .. "/python3.11")
  touch_exe(bindir .. "/python3.10")
  local found = common._find_versioned_python3(bindir)
  assert(found == bindir .. "/python3.11", tostring(found))
end)

case_("_find_versioned_python3 ignores non-python3.N names", function()
  local bindir = mktemp_dir()
  touch_exe(bindir .. "/python3.10-config")
  touch_exe(bindir .. "/pip3.10")
  local found = common._find_versioned_python3(bindir)
  assert(found == nil, tostring(found))
end)

case_("_find_versioned_python3 returns nil for a missing directory", function()
  local found = common._find_versioned_python3("/no/such/dir/xyz123")
  assert(found == nil, tostring(found))
end)

-- ---------------------------------------------------------------------
-- detect_reaper_python (full flow, via a stubbed GetResourcePath)
-- ---------------------------------------------------------------------
case_("detect_reaper_python resolves a versioned-only Framework bin/ (the reported bug)", function()
  local resourcedir = mktemp_dir()
  local fw = mktemp_dir()
  os.execute('mkdir -p "' .. fw .. '/Versions/3.10/bin"')
  os.execute('mkdir -p "' .. fw .. '/Versions/3.10/lib"')
  touch_exe(fw .. "/Versions/3.10/bin/python3.10")
  write_ini(resourcedir, fw .. "/Versions/3.10/lib")

  reaper.GetResourcePath = function() return resourcedir end
  local path, err = common.detect_reaper_python()
  assert(err == nil, tostring(err))
  assert(path == fw .. "/Versions/3.10/bin/python3.10", tostring(path))
end)

case_("detect_reaper_python prefers a bare bin/python3 when present", function()
  local resourcedir = mktemp_dir()
  local fw = mktemp_dir()
  os.execute('mkdir -p "' .. fw .. '/Versions/3.11/bin"')
  os.execute('mkdir -p "' .. fw .. '/Versions/3.11/lib"')
  touch_exe(fw .. "/Versions/3.11/bin/python3")
  touch_exe(fw .. "/Versions/3.11/bin/python3.11")
  write_ini(resourcedir, fw .. "/Versions/3.11/lib")

  reaper.GetResourcePath = function() return resourcedir end
  local path, err = common.detect_reaper_python()
  assert(err == nil, tostring(err))
  assert(path == fw .. "/Versions/3.11/bin/python3", tostring(path))
end)

case_("detect_reaper_python falls back to 'python3' when nothing is found", function()
  local resourcedir = mktemp_dir()
  write_ini(resourcedir, "/no/such/lib/dir")

  reaper.GetResourcePath = function() return resourcedir end
  -- find_stable_python scans the real /opt/homebrew/bin, so on a machine with
  -- Homebrew Python this case would find one; stub it to keep the test hermetic.
  local real_find = common.find_stable_python
  common.find_stable_python = function() return nil end
  local path, err = common.detect_reaper_python()
  common.find_stable_python = real_find
  assert(path == "python3", tostring(path))
  assert(err ~= nil)
end)

case_("detect_reaper_python falls back to 'python3' when reaper.ini is missing", function()
  local resourcedir = mktemp_dir()
  reaper.GetResourcePath = function() return resourcedir end
  local path, err = common.detect_reaper_python()
  assert(path == "python3", tostring(path))
  assert(err ~= nil)
end)

if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
os.exit(0)
