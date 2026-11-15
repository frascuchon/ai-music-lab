-- Regression harness for widgets_extra.input_textarea: with long multi-line
-- text and an active scroll, lines must never be painted on top of each other
-- or outside the box. gfx.drawstr cannot clip at the TOP edge, so a line that
-- is only partly scrolled out has to be skipped (or the scroll snapped to whole
-- lines) rather than pinned to the top edge, where it overprinted the next line.

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path

local LH = 10
local draws = {}
gfx = {
  x = 0, y = 0, w = 620, h = 780,
  mouse_x = -1, mouse_y = -1, mouse_cap = 0, mouse_wheel = 0, mouse_hwheel = 0,
  measurestr = function(s) return #s * 6, LH end,
  setfont = function() end, set = function() end, rect = function() end,
  line = function() end,
  drawstr = function(s) draws[#draws + 1] = { s = s, y = gfx.y } end,
}

local gui = require("gui")
local wx = require("widgets_extra")
local ctx = gui.ctx

local PROMPT = [[[Intro]
(Atmospheric synth pads, subtle 808 roll building up)
Yeah, yeah... EsDeeKid vibe, you know how it goes.
Dark nights, neon lights.

[Verse 1]
Caminando solo por la fría ciudad,
las luces parpadean, no hay vuelta atrás.
El bajo suena fuerte retumbando en el pecho,
buscando la salida sin mirar a los techos.
Gotas en el vidrio, frío en la piel,
escribiendo mi historia con un pincel de hiel.

[Chorus]
Y ya no hay tiempo para mirar atrás,
somos sombra y fuego que no van a apagar.
El bajo retumba, la noche es eterna,
mientras mi alma en el frío gobierna y gobierna.
(¡Trap!)

[Verse 2]
Cuentas pendientes, miradas de reojo,
el humo en el aire me nubla los ojos.
Seguimos subiendo sin pedir perdón,
marcando el compás con el corazón.

[Chorus]
Y ya no hay tiempo para mirar atrás,
somos sombra y fuego que no van a apagar.
El bajo retumba, la noche es eterna,
mientras mi alma en el frío gobierna y gobierna.

[Outro]
(Synth pads fading out, 808 drops softly)
Yeah... fin del juego.]]

local failures = 0
local function case_(name, fn)
  local ok, err = pcall(fn)
  if ok then print("PASS " .. name) else
    failures = failures + 1; print("FAIL " .. name .. ": " .. tostring(err)) end
end

local LINES_VISIBLE = 8
local TOP = 40
local function draw_frame(scroll_y)
  ctx.state = ctx.state or {}
  ctx.x, ctx.y, ctx.row_x0, ctx.content_w = 12, TOP, 12, 596
  ctx.mx, ctx.my, ctx.mw, ctx.mhw, ctx.mb, ctx.mb_prev = 0, 0, 0, 0, 0, 0
  ctx.char_queue, ctx.key_queue = {}, {}
  draws = {}
  if scroll_y then ctx.state.ta = ctx.state.ta or {}; ctx.state.ta.scroll_y = scroll_y end
  wx.input_textarea("ta", PROMPT, LINES_VISIBLE)
  return draws
end

local function assert_no_overlap(d, label)
  table.sort(d, function(a, b) return a.y < b.y end)
  for i = 2, #d do
    assert(d[i].y - d[i - 1].y >= LH,
      string.format("%s: lines overlap: %q@%.1f vs %q@%.1f",
        label, d[i - 1].s, d[i - 1].y, d[i].s, d[i].y))
  end
end

-- box geometry as the widget computes it (ls = LH+3, pad = 5)
local LS, PAD = LH + 3, 5
local BOX_H = LINES_VISIBLE * LS + PAD * 2

case_("unscrolled text draws without overlap", function()
  draw_frame(); assert_no_overlap(draw_frame(0), "scroll 0")
end)

case_("no overlap at any fractional scroll position", function()
  draw_frame()
  for sy = 0, 200, 1.5 do
    assert_no_overlap(draw_frame(sy), "scroll " .. sy)
  end
end)

case_("every drawn line stays inside the box", function()
  draw_frame()
  for sy = 0, 200, 2.5 do
    for _, d in ipairs(draw_frame(sy)) do
      assert(d.y >= TOP, string.format("scroll %.1f: %q above box (y=%.1f)", sy, d.s, d.y))
      -- a line cut by the bottom edge is fine: drawstr clips it there
      assert(d.y < TOP + BOX_H,
        string.format("scroll %.1f: %q below box (y=%.1f)", sy, d.s, d.y))
    end
  end
end)

case_("scrolled to the end shows the last line", function()
  draw_frame()
  local d = draw_frame(100000)
  local found = false
  for _, x in ipairs(d) do if x.s:find("fin del juego", 1, true) then found = true end end
  assert(found, "last line not visible at max scroll")
end)

case_("gentle wheel notches still move the box, by whole lines", function()
  draw_frame(0)
  local mx = 12 + 20
  for _, mw in ipairs({ -10, -30, -120 }) do
    draw_frame(0)
    ctx.x, ctx.y = 12, TOP
    ctx.mx, ctx.my, ctx.mw = mx, TOP + 20, mw
    draws = {}
    wx.input_textarea("ta", PROMPT, LINES_VISIBLE)
    local sy = ctx.state.ta.scroll_y
    assert(sy > 0, "wheel " .. mw .. " did not scroll")
    assert(sy % LS == 0, "wheel " .. mw .. " left a partial line: " .. sy)
    assert_no_overlap(draws, "after wheel " .. mw)
  end
end)

case_("wheel up returns to the top and never goes negative", function()
  draw_frame(0)
  ctx.x, ctx.y = 12, TOP
  ctx.mx, ctx.my, ctx.mw = 32, TOP + 20, 120
  wx.input_textarea("ta", PROMPT, LINES_VISIBLE)
  assert(ctx.state.ta.scroll_y == 0)
end)

case_("caret at the end of a long text keeps the last line visible", function()
  draw_frame(0)
  ctx.state.ta.caret = #PROMPT
  ctx.focused_id = "ta"
  local d = draw_frame()
  ctx.focused_id = nil
  local found = false
  for _, x in ipairs(d) do if x.s:find("fin del juego", 1, true) then found = true end end
  assert(found, "caret line not scrolled into view")
  assert_no_overlap(d, "caret at end")
end)

case_("typing a newline at the end of a long text does not overlap lines", function()
  draw_frame(0)
  ctx.state.ta.caret = #PROMPT
  ctx.focused_id = "ta"
  ctx.x, ctx.y, ctx.row_x0, ctx.content_w = 12, TOP, 12, 596
  ctx.char_queue, ctx.key_queue = {}, { 13 }   -- Enter
  draws = {}
  local _, nt = wx.input_textarea("ta", PROMPT, LINES_VISIBLE)
  ctx.focused_id = nil
  assert(nt == PROMPT .. "\n", "newline not inserted")
  assert_no_overlap(draws, "after Enter")
end)

if failures > 0 then print(failures .. " failure(s)"); os.exit(1) end
os.exit(0)
