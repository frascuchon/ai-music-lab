-- Lua-side harness for test_model_switch.py.
--
-- model_switch.lua has no reaper dependency, so this harness just requires
-- it directly and runs plain assertions — no mocking needed. Each case
-- prints "PASS <name>" or "FAIL <name>: <msg>"; the process exits non-zero
-- if any case fails, so test_model_switch.py can just check the return
-- code (and print stdout on failure for detail).

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path

local ms = require("model_switch")

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

-- ── changed ──────────────────────────────────────────────────────

case_("changed: same index is not a change", function()
  assert(ms.changed(2, 2) == false)
end)

case_("changed: different index is a change", function()
  assert(ms.changed(2, 5) == true)
end)

-- ── next_prompt ──────────────────────────────────────────────────

case_("next_prompt: unchanged model keeps the current prompt", function()
  assert(ms.next_prompt(1, 1, "chord progression Am-F-C-G") == "chord progression Am-F-C-G")
end)

case_("next_prompt: changed model clears the prompt", function()
  assert(ms.next_prompt(1, 2, "chord progression Am-F-C-G") == "")
end)

case_("next_prompt: changed model clears even an already-empty prompt", function()
  assert(ms.next_prompt(1, 2, "") == "")
end)

case_("next_prompt: clears regardless of direction (moving back to a previous model)", function()
  assert(ms.next_prompt(3, 1, "some text") == "")
end)

-- ── default_for ──────────────────────────────────────────────────

case_("default_for: returns the model's own default when present", function()
  local t = { amadeus = "A10G", midi_llm = "H100" }
  assert(ms.default_for(t, "midi_llm", "T4") == "H100")
end)

case_("default_for: falls back when the model has no entry", function()
  local t = { amadeus = "A10G" }
  assert(ms.default_for(t, "unknown_model", "T4") == "T4")
end)

case_("default_for: falls back when the table itself is nil", function()
  assert(ms.default_for(nil, "amadeus", "T4") == "T4")
end)

case_("default_for: falls back when model_key is nil", function()
  local t = { amadeus = "A10G" }
  assert(ms.default_for(t, nil, "T4") == "T4")
end)

if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
os.exit(0)
