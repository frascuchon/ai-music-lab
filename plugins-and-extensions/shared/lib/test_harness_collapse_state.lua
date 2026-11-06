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

case_("to_logical is identity at scale 1", function()
  assert(collapse_state.to_logical(620, 1) == 620)
end)

case_("to_logical halves Retina pixels", function()
  assert(collapse_state.to_logical(1240, 2) == 620)
end)

case_("to_logical guards against nil/invalid scale", function()
  assert(collapse_state.to_logical(620, nil) == 620)
  assert(collapse_state.to_logical(620, 0) == 620)
end)

case_("repeated toggles on Retina keep the width stable", function()
  -- Simulates resize_window: width read in px, converted, window recreated
  -- at the logical size, which then reports px = logical * scale again.
  local scale, logical_w = 2, 620
  local px_w = logical_w * scale
  for _ = 1, 6 do
    local new_logical = collapse_state.to_logical(px_w, scale)
    assert(new_logical == logical_w, "width drifted to " .. new_logical)
    px_w = new_logical * scale
  end
end)

case_("expanded height round-trips on Retina without growing", function()
  local scale, expanded_logical = 2, 780
  local px_h = expanded_logical * scale
  local stored = collapse_state.track_expanded_h(false, collapse_state.to_logical(px_h, scale), 0)
  assert(stored == 780)
  -- collapse then expand: target (px) -> logical must equal stored again
  local target_px = collapse_state.target_height(false, stored * scale, 80)
  assert(collapse_state.to_logical(target_px, scale) == 780)
end)

case_("scrolling-independent: collapsed keeps remembered height", function()
  assert(collapse_state.track_expanded_h(true, 40, 780) == 780)
end)

-- Fake gfx that behaves like Retina REAPER: init() takes logical units,
-- w/h read back in physical pixels (logical * scale).
local function fake_gfx(scale, lw, lh)
  local g = { scale = scale, w = lw * scale, h = lh * scale, inits = {}, y = 200 }
  g.dock = function() return 0, 100, g.y end
  g.quit = function() g.w, g.h = 0, 0 end
  g.init = function(title, w, h, dock, x, y)
    g.inits[#g.inits + 1] = { w = w, h = h, y = y }
    g.w, g.h, g.y = w * scale, h * scale, y
  end
  return g
end

case_("recreate_window keeps width across repeated collapse/expand on Retina", function()
  local g = fake_gfx(2, 620, 780)
  local header_px, expanded_px = 80, 780 * 2
  for i = 1, 5 do
    collapse_state.recreate_window(g, "t", header_px, 2, true)
    assert(g.w == 620 * 2, "collapse #" .. i .. " width " .. g.w)
    assert(g.h == 80, "collapsed height " .. g.h)
    collapse_state.recreate_window(g, "t", expanded_px, 2, true)
    assert(g.w == 620 * 2, "expand #" .. i .. " width " .. g.w)
    assert(g.h == 780 * 2, "expanded height " .. g.h)
  end
end)

case_("recreate_window keeps the top edge anchored on macOS", function()
  local g = fake_gfx(2, 620, 780)
  local top = g.y + 780
  collapse_state.recreate_window(g, "t", 80, 2, true)
  assert(g.y + 40 == top, "top moved")
end)

case_("recreate_window does not touch y off macOS and works at scale 1", function()
  local g = fake_gfx(1, 620, 780)
  collapse_state.recreate_window(g, "t", 40, 1, false)
  assert(g.inits[1].y == 200 and g.w == 620 and g.h == 40)
end)

case_("recreate_window re-runs the on_ready hook (fonts)", function()
  local g, called = fake_gfx(2, 620, 780), 0
  collapse_state.recreate_window(g, "t", 80, 2, false, function() called = called + 1 end)
  assert(called == 1)
end)

if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
os.exit(0)
