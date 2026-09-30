-- Lua-side harness for test_process_control.py.
--
-- process_control.lua has no reaper/gfx dependency, so this harness just
-- requires it directly and runs plain assertions — no mocking needed
-- (same approach as test_harness_collapse_state.lua). Each case prints
-- "PASS <name>" or "FAIL <name>: <msg>"; the process exits non-zero if any
-- case fails, so test_process_control.py can just check the return code.

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path

local process_control = require("process_control")

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

case_("wrap_for_pid_capture appends an echo of $! on a new statement", function()
  local wrapped = process_control.wrap_for_pid_capture("python3 foo.py >>log 2>&1 &")
  assert(wrapped == "python3 foo.py >>log 2>&1 &\necho $!", wrapped)
end)

case_("parse_pid reads a plain numeric line", function()
  assert(process_control.parse_pid("12345\n") == 12345)
end)

case_("parse_pid tolerates surrounding whitespace", function()
  assert(process_control.parse_pid("  42  ") == 42)
end)

case_("parse_pid returns nil for empty output", function()
  assert(process_control.parse_pid("") == nil)
end)

case_("parse_pid returns nil for nil input (io.popen failure)", function()
  assert(process_control.parse_pid(nil) == nil)
end)

case_("parse_pid returns nil for non-numeric garbage", function()
  assert(process_control.parse_pid("not a pid") == nil)
end)

case_("is_valid_pid accepts a positive integer", function()
  assert(process_control.is_valid_pid(4242) == true)
end)

case_("is_valid_pid rejects nil", function()
  assert(process_control.is_valid_pid(nil) == false)
end)

case_("is_valid_pid rejects zero", function()
  assert(process_control.is_valid_pid(0) == false)
end)

case_("is_valid_pid rejects negative numbers", function()
  assert(process_control.is_valid_pid(-1) == false)
end)

case_("is_valid_pid rejects non-integers", function()
  assert(process_control.is_valid_pid(1.5) == false)
end)

case_("is_valid_pid rejects non-numbers", function()
  assert(process_control.is_valid_pid("4242") == false)
end)

case_("build_kill_command targets children then the pid itself", function()
  local cmd = process_control.build_kill_command(777)
  assert(cmd:find("pkill %-TERM %-P 777") ~= nil, cmd)
  assert(cmd:find("kill %-TERM 777") ~= nil, cmd)
end)

case_("build_kill_command returns nil for an invalid pid", function()
  assert(process_control.build_kill_command(nil) == nil)
  assert(process_control.build_kill_command(0) == nil)
  assert(process_control.build_kill_command(-5) == nil)
end)

if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
os.exit(0)
