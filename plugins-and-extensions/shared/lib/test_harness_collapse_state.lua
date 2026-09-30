-- Lua-side harness for test_collapse_state.py.
--
-- collapse_state.lua has no reaper/gfx dependency, so this harness just
-- requires it directly and runs plain assertions — no mocking needed
-- (unlike gen_test_smf.lua, which mocks the `reaper` global for
-- smf_writer.lua). Each case prints "PASS <name>" or "FAIL <name>: <msg>";
-- the process exits non-zero if any case fails, so test_collapse_state.py
-- can just check the return code (and print stdout on failure for detail).

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path

local collapse_state = require("collapse_state")

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

case_("toggle flips false to true", function()
  assert(collapse_state.toggle(false) == true)
end)

case_("toggle flips true to false", function()
  assert(collapse_state.toggle(true) == false)
end)

case_("target_height returns header_h when collapsed", function()
  assert(collapse_state.target_height(true, 780, 40) == 40)
end)

case_("target_height returns expanded_h when not collapsed", function()
  assert(collapse_state.target_height(false, 780, 40) == 780)
end)

case_("track_expanded_h adopts current_h while expanded", function()
  assert(collapse_state.track_expanded_h(false, 900, 780) == 900)
end)

case_("track_expanded_h keeps prior expanded_h while collapsed", function()
  -- current_h here would be the header-only height (e.g. 40) — must NOT
  -- overwrite the remembered pre-collapse height.
  assert(collapse_state.track_expanded_h(true, 40, 780) == 780)
end)

case_("anchor_top_y keeps the top edge fixed when shrinking (collapse)", function()
  -- Window at y=200 (bottom-up coord), height 780 -> top edge at 200+780=980.
  -- Collapsing to height 40 must keep that top edge at 980.
  local new_y = collapse_state.anchor_top_y(200, 780, 40)
  assert(new_y + 40 == 200 + 780, "top edge moved: " .. tostring(new_y + 40))
end)

case_("anchor_top_y keeps the top edge fixed when growing (expand)", function()
  local new_y = collapse_state.anchor_top_y(940, 40, 780)
  assert(new_y + 780 == 940 + 40, "top edge moved: " .. tostring(new_y + 780))
end)

case_("collapse then expand round-trips through target_height", function()
  local header_h, expanded_h = 40, 900
  local collapsed = false
  collapsed = collapse_state.toggle(collapsed)
  assert(collapse_state.target_height(collapsed, expanded_h, header_h) == header_h)
  collapsed = collapse_state.toggle(collapsed)
  assert(collapse_state.target_height(collapsed, expanded_h, header_h) == expanded_h)
end)

if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
os.exit(0)
