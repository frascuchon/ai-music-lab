-- Regression harness for widgets_extra.scroll_region (scroll must keep
-- working alongside the window collapse/resize fix). Runs the real
-- scroll_region with a recording gfx stub.

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path

gfx = {
  x = 0, y = 0, w = 620, h = 780,
  mouse_x = -1, mouse_y = -1, mouse_cap = 0, mouse_wheel = 0, mouse_hwheel = 0,
  measurestr = function(s) return #s * 6, 10 end,
  setfont = function() end, set = function() end, rect = function() end,
  line = function() end, drawstr = function() end,
}

local gui = require("gui")
local wx = require("widgets_extra")
local theme = require("theme")
local ctx = gui.ctx

local failures = 0
local function case_(name, fn)
  local ok, err = pcall(fn)
  if ok then print("PASS " .. name) else
    failures = failures + 1; print("FAIL " .. name .. ": " .. tostring(err)) end
end

local function reset()
  ctx.state = {}
  ctx.x, ctx.y, ctx.row_x0, ctx.content_w = 12, 10, 12, 596
  ctx.mx, ctx.my, ctx.mw, ctx.mhw, ctx.mb, ctx.mb_prev = 0, 0, 0, 0, 0, 0
end

local function content() for _ = 1, 40 do gui.spacing(20) end end
-- Each frame starts with the cursor back at the region origin, like a
-- real redraw (scroll_region advances ctx.y past itself).
local function frame(id, h)
  ctx.x, ctx.y = 12, 10
  wx.scroll_region(id, 0, h, content)
end

case_("content taller than region is measured", function()
  reset(); frame("r", 200)
  assert(ctx.state.r.content_h > 200, "content_h=" .. ctx.state.r.content_h)
end)

case_("wheel over the region scrolls down, clamped to content", function()
  reset(); frame("r", 200)
  ctx.mx, ctx.my, ctx.mw = 50, 50, -120
  frame("r", 200)
  local sy = ctx.state.r.scroll_y
  assert(sy > 0, "did not scroll: " .. sy)
  assert(sy <= ctx.state.r.content_h - (200 - 0), "overscrolled")
end)

case_("wheel outside the region does not scroll", function()
  reset(); frame("r", 200)
  ctx.mx, ctx.my, ctx.mw = 50, 700, -120
  frame("r", 200)
  assert(ctx.state.r.scroll_y == 0)
end)

case_("scroll never goes negative", function()
  reset(); frame("r", 200)
  ctx.mx, ctx.my, ctx.mw = 50, 50, 120
  frame("r", 200)
  assert(ctx.state.r.scroll_y == 0)
end)

case_("scroll position survives a taller (expanded) region and re-clamps", function()
  reset(); frame("r", 200)
  ctx.mx, ctx.my, ctx.mw = 50, 50, -120
  frame("r", 200)
  ctx.mw = 0
  -- window expanded: region becomes tall enough to show all content
  frame("r", 5000); frame("r", 5000)
  assert(ctx.state.r.scroll_y == 0, "scroll_y=" .. ctx.state.r.scroll_y)
end)

case_("scroll_to_bottom reaches the end", function()
  reset(); frame("r", 200)
  wx.scroll_to_bottom("r"); frame("r", 200)
  local s = ctx.state.r
  local max_y = s.content_h - 200
  assert(s.scroll_y > 1000, "did not jump to bottom: " .. s.scroll_y)
  assert(s.scroll_y <= max_y, "past the end: " .. s.scroll_y)
end)

case_("layout context is restored after the region", function()
  reset(); frame("r", 200)
  assert(ctx.row_x0 == 12 and ctx.content_w == 596 and ctx.clip_y1 == nil)
end)

if failures > 0 then print(failures .. " failure(s)"); os.exit(1) end
os.exit(0)
