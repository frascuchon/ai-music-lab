-- lib/process_control.lua  Pure string/logic helpers behind the "Stop" button
-- feature: capturing the PID of a backgrounded shell command and building a
-- best-effort kill command for it. No reaper/gfx dependency (see
-- collapse_state.lua for the precedent) so it can be unit tested directly
-- with a plain `lua` interpreter.

local M = {}

-- `cmd` is a Unix shell command that ends by backgrounding a process with a
-- trailing "&" (the convention used by every panel's launch_xxx()). Appends
-- a second statement that prints the PID of that backgrounded job ($!) so
-- the caller (via io.popen) can capture it from stdout.
function M.wrap_for_pid_capture(cmd)
  return cmd .. "\necho $!"
end

-- Parses the single line of output produced by running wrap_for_pid_capture's
-- result through io.popen. Returns a numeric pid, or nil if capture failed
-- (empty output, non-numeric, etc).
function M.parse_pid(output)
  if not output then return nil end
  local digits = output:match("^%s*(%d+)%s*$")
  if not digits then return nil end
  return tonumber(digits)
end

-- Guards against issuing a kill against an unset/garbage pid (0 and negative
-- numbers address whole process groups or init on Unix).
function M.is_valid_pid(pid)
  return type(pid) == "number" and pid > 0 and pid == math.floor(pid)
end

-- Best-effort termination: SIGTERM any direct children first (so a python
-- wrapper's own subprocess doesn't survive it), then the tracked pid itself.
-- Returns nil when pid isn't valid (nothing to run).
function M.build_kill_command(pid)
  if not M.is_valid_pid(pid) then return nil end
  return string.format(
    "pkill -TERM -P %d >/dev/null 2>&1; kill -TERM %d >/dev/null 2>&1",
    pid, pid)
end

return M
