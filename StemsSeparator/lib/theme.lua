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

-- Layout metrics
M.PAD_X     = 12
M.PAD_Y     = 10
M.ITEM_H    = 22   -- standard widget row height
M.BTN_H     = 28   -- button height
M.SPACING_X = 8
M.SPACING_Y = 6
M.ROUND     = 4    -- corner radius for buttons/frames
M.SCROLL_W  = 8    -- scrollbar width

-- Button alpha levels (for translucent accent fill)
M.A_BTN_NORM = 0.19
M.A_BTN_HOV  = 0.44
M.A_BTN_ACT  = 0.67

function M.set(name, alpha)
  local c = M.C[name]
  gfx.set(c[1], c[2], c[3], alpha or 1.0)
end

function M.init_fonts()
  local mono = reaper.GetOS():find("^Win") and "Consolas" or "Menlo"
  gfx.setfont(M.F.UI,   "Arial", 14)
  gfx.setfont(M.F.H1,   "Arial", 18, string.byte("b"))
  gfx.setfont(M.F.MONO, mono,    13)
end

return M
