-- Lua-side harness for test_prepare.py (TestPrepareQueueLua).
local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path
local pq = require("prepare_queue")

local failures = 0
local function case_(name, fn)
  local ok, err = pcall(fn)
  if ok then print("PASS " .. name) else failures = failures + 1; print("FAIL " .. name .. ": " .. tostring(err)) end
end
local function T(model, ex, group) return { domain = "d", model = model, execution = ex, group = group or "" } end
local function eq(a, b) assert(table.concat(a, ",") == table.concat(b, ","), table.concat(a, ",") .. " ~= " .. table.concat(b, ",")) end

case_("remote starts up to four at once", function()
  local q = {}
  for i = 1, 7 do q[i] = T("m" .. i, "remote") end
  eq(pq.pick(q, {}), { 1, 2, 3, 4 })
end)
case_("running jobs count against the limit", function()
  local q = { T("a", "remote"), T("b", "remote"), T("c", "remote") }
  eq(pq.pick(q, { T("x", "remote"), T("y", "remote"), T("z", "remote") }), { 1 })
  eq(pq.pick(q, { T("x", "remote"), T("y", "remote"), T("z", "remote"), T("w", "remote") }), {})
end)
case_("local runs one at a time, alongside remote", function()
  local q = { T("a", "local"), T("b", "local"), T("c", "remote") }
  eq(pq.pick(q, {}), { 1, 3 })
  eq(pq.pick({ T("b", "local") }, { T("a", "local") }), {})
end)
case_("targets sharing an image never run together", function()
  local q = { T("gen", "remote", "ace"), T("edit", "remote", "ace"), T("sao", "remote") }
  eq(pq.pick(q, {}), { 1, 3 })
  eq(pq.pick({ T("edit", "remote", "ace") }, { T("gen", "remote", "ace") }), {})
end)
case_("blocked items do not block later ones", function()
  local q = { T("edit", "remote", "ace"), T("sao", "remote") }
  eq(pq.pick(q, { T("gen", "remote", "ace") }), { 2 })
end)
os.exit(failures == 0 and 0 or 1)
