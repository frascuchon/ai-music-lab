-- lib/theme.lua  Color palette, font constants, layout metrics.

local M = {}

local function rgb(h)
  return { ((h>>16)&0xFF)/255, ((h>>8)&0xFF)/255, (h&0xFF)/255 }
end

M.C = {
  BG         = rgb(0x1A1A1C),
  FRAME      = rgb(0x2E2E38),
  FRAME_HOV  = rgb(0x3C3C48),
  FRAME_ACT  = rgb(0x4A4A58),
  ACCENT     = rgb(0x4799FF),
  ACCENT_HOV = rgb(0x6AADFF),
  SEP        = rgb(0x3A3A44),
  FG         = rgb(0xEBEBEB),
  FG_DIM     = rgb(0x848491),
  LOG_BG     = rgb(0x111116),
  LOG_FG     = rgb(0xB4B4BE),
  GREEN      = rgb(0x40B261),
  RED        = rgb(0xD94238),
  YELLOW     = rgb(0xF2B81A),
  POPUP_BG   = rgb(0x252528),
  SCROLLBAR  = rgb(0x3A3A44),
}

-- gfx font slot indices
M.F = { UI = 1, H1 = 2, MONO = 3 }

-- Current display scale (1.0 = normal, 2.0 = Retina 2x). Set by apply_scale().
M.SCALE = 1

-- Layout metrics (defaults for 1x display)
M.PAD_X         = 12
M.PAD_Y         = 10
M.ITEM_H        = 22   -- standard widget row height
M.BTN_H         = 28   -- button height
M.SPACING_X     = 8
M.SPACING_Y     = 6
M.ROUND         = 4    -- corner radius for buttons/frames
M.SCROLL_W      = 8    -- scrollbar width
M.CHECK_BOX_S   = 14   -- checkbox square size
M.SLIDER_GRAB_W = 12   -- slider grab handle width
M.SLIDER_GRAB_H = 18   -- slider grab handle height

-- Button alpha levels (for translucent accent fill)
M.A_BTN_NORM = 0.19
M.A_BTN_HOV  = 0.44
M.A_BTN_ACT  = 0.67

function M.set(name, alpha)
  local c = M.C[name]
  gfx.set(c[1], c[2], c[3], alpha or 1.0)
end

-- Scale a 1x pixel value to the current display scale.
function M.sc(v)
  return math.floor(v * M.SCALE + 0.5)
end

local function sc(v, s) return math.floor(v * s + 0.5) end

-- Scale all layout metrics by s (call once after detecting display DPI).
function M.apply_scale(s)
  s = s or 1
  M.SCALE         = s
  M.PAD_X         = sc(12, s)
  M.PAD_Y         = sc(10, s)
  M.ITEM_H        = sc(22, s)
  M.BTN_H         = sc(28, s)
  M.SPACING_X     = sc( 8, s)
  M.SPACING_Y     = sc( 6, s)
  M.ROUND         = sc( 4, s)
  M.SCROLL_W      = sc( 8, s)
  M.CHECK_BOX_S   = sc(14, s)
  M.SLIDER_GRAB_W = sc(12, s)
  M.SLIDER_GRAB_H = sc(18, s)
end

function M.init_fonts(s)
  s = s or 1
  local mono = reaper.GetOS():find("^Win") and "Consolas" or "Menlo"
  gfx.setfont(M.F.UI,   "Arial", sc(14, s))
  gfx.setfont(M.F.H1,   "Arial", sc(18, s), string.byte("b"))
  gfx.setfont(M.F.MONO, mono,    sc(13, s))
end

return M
