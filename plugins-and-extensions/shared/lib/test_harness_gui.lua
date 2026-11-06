-- Lua-side harness for test_gui.lua (shared/lib/gui.lua).
--
-- Covers two layout bugs reported from the Setup tab: the "Installing..."
-- status text rendering outside the visible Setup panel area after the
-- demucs row's icon+label+detail+button pushed the draw cursor (ctx.x)
-- past the panel's right edge.
--
--   1. drawstr_clipped's vertical-scroll-only branch clipped text against
--      `x + ctx.content_w` instead of `ctx.row_x0 + ctx.content_w` — the
--      further right x had already drifted via same_line() calls, the
--      further right (and more useless) the clip boundary became,
--      effectively disabling clipping for exactly the crowded rows that
--      needed it.
--   2. same_line() has no concept of "won't fit" — it always advances,
--      so a button/status placed after a long label+detail string keeps
--      sliding right past the panel edge instead of wrapping. The new
--      same_line_or_wrap() fixes that.
--
-- gui.lua only touches `gfx` inside drawing functions (never at module
-- load), so a minimal recording stub covers both tests below. Same
-- approach as test_harness_process_control.lua / test_harness_common.lua.

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path

-- Minimal recording gfx stub: measurestr returns 6px per char (fake but
-- deterministic), drawstr/rect/etc are no-ops that just record calls.
local drawstr_calls = {}
gfx = {
  x = 0, y = 0,
  mouse_x = -1, mouse_y = -1, mouse_cap = 0, mouse_wheel = 0, mouse_hwheel = 0,
  measurestr = function(s) return #s * 6, 10 end,
  setfont = function() end,
  set = function() end,
  rect = function() end,
  line = function() end,
  drawstr = function(s, flags, right, bottom)
    table.insert(drawstr_calls, { s = s, flags = flags, right = right, bottom = bottom })
  end,
}

local gui = require("gui")
local ctx = gui.ctx

local failures = 0

local function case_(name, fn)
  drawstr_calls = {}
  local ok, err = pcall(fn)
  if ok then
    print("PASS " .. name)
  else
    failures = failures + 1
    print("FAIL " .. name .. ": " .. tostring(err))
  end
end

-- Resets ctx to a simulated scroll_region's inner coordinate system:
-- row_x0=20, content_w=300 → the visible right edge is at screen x=320.
local function setup_region()
  ctx.row_x0     = 20
  ctx.x          = 20
  ctx.y          = 0
  ctx.content_w  = 300
  ctx.clip_y1    = 0
  ctx.clip_y2    = 1000
  ctx.clip_y_off = 0
  ctx.clip_x1    = nil  -- vertical-scroll-only: the buggy branch
  ctx.clip_x2    = nil
  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = ctx.row_x0, 0, 0, 0
end

-- ---------------------------------------------------------------------
-- drawstr_clipped right-clip boundary (via M.text_colored)
-- ---------------------------------------------------------------------
case_("text drawn near the region's left edge clips at the region's right edge", function()
  setup_region()
  gui.text_colored("hello", "FG")
  assert(#drawstr_calls == 1)
  local call = drawstr_calls[1]
  -- region right edge = row_x0(20) + content_w(300) = 320
  assert(call.right == 320, tostring(call.right))
end)

case_("text drawn after the cursor has drifted right (same_line chain) still clips at the SAME region right edge", function()
  setup_region()
  -- Simulate a crowded row: advance x way to the right via same_line(),
  -- as the demucs row does (icon -> label -> detail -> button -> status).
  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = 250, 0, 40, 10
  gui.same_line(6)
  assert(ctx.x == 296, tostring(ctx.x))  -- 250 + 40 + 6
  gui.text_colored("Installing...", "YELLOW")
  local call = drawstr_calls[1]
  -- Must still be 320 (row_x0+content_w), NOT x(296) + content_w(300) = 596
  -- — the old bug let the clip boundary run away past the panel entirely.
  assert(call.right == 320, tostring(call.right))
end)

-- ---------------------------------------------------------------------
-- same_line_or_wrap
-- ---------------------------------------------------------------------
case_("same_line_or_wrap stays on the same line when the widget still fits", function()
  setup_region()
  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = 20, 50, 40, 22
  gui.same_line_or_wrap(70, 6)  -- next_x = 20+40+6=66; 66+70=136 <= 320
  assert(ctx.x == 66, tostring(ctx.x))
  assert(ctx.y == 50, tostring(ctx.y))
end)

case_("same_line_or_wrap wraps to a new line when the widget would overflow the region", function()
  setup_region()
  -- last widget already sits near the right edge of a 300px-wide region
  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = 280, 50, 30, 22
  gui.same_line_or_wrap(70, 6)  -- next_x = 280+30+6=316; 316+70=386 > 320
  assert(ctx.x == ctx.row_x0, tostring(ctx.x))
  assert(ctx.y == 50 + 22 + gui.theme.SPACING_Y, tostring(ctx.y))
end)

case_("same_line_or_wrap exactly at the boundary still counts as fitting", function()
  setup_region()
  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = 20, 0, 0, 22
  -- next_x = 20+0+0 = 20; 20+300 = 320 == row_x0+content_w(320) -> fits
  gui.same_line_or_wrap(300, 0)
  assert(ctx.x == 20, tostring(ctx.x))
  assert(ctx.y == 0, tostring(ctx.y))
end)

-- ---------------------------------------------------------------------
-- button: "##id" suffix is never drawn; optional small icon

case_("button label drops the ##id suffix", function()
  setup_region()
  gui.button("MIDI-GPT##lic_MIDI-GPT", 200, 28)
  assert(#drawstr_calls == 1, tostring(#drawstr_calls))
  assert(drawstr_calls[1].s == "MIDI-GPT", drawstr_calls[1].s)
end)

case_("button icon is drawn with lines, stays inside the button and keeps the label clean", function()
  setup_region()
  local lines = 0
  local orig = gfx.line
  gfx.line = function() lines = lines + 1 end
  gui.button("Stable Audio Open##lic_x", 200, 28, { icon = "link" })
  gfx.line = orig
  assert(lines == 3, tostring(lines))
  assert(#drawstr_calls == 1 and drawstr_calls[1].s == "Stable Audio Open", drawstr_calls[1].s)
  -- the label starts to the right of the icon, still inside the button (x 20..220)
  assert(gfx.x > 20 and gfx.x < 220, tostring(gfx.x))
end)

case_("button without icon draws no lines", function()
  setup_region()
  local lines = 0
  local orig = gfx.line
  gfx.line = function() lines = lines + 1 end
  gui.button("Plain", 200, 28)
  gfx.line = orig
  assert(lines == 0, tostring(lines))
end)

if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
os.exit(0)
