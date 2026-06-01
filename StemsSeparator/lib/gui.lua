-- lib/gui.lua  Immediate-mode widget toolkit built on REAPER gfx.
-- No external dependencies: only REAPER built-in gfx API.

local theme = require("theme")

local M = {}
M.theme = theme

-- ── CONTEXT SINGLETON ─────────────────────────────────────────────
local ctx = {
  -- Layout cursor
  x = 0, y = 0, content_w = 0,
  last_x = 0, last_y = 0, last_w = 0, last_h = 0,
  -- Clip region (used inside scroll_region)
  clip_y1 = nil, clip_y2 = nil,
  clip_y_off = 0,   -- screen_y = logical_y + clip_y_off
  -- Mouse
  mx = 0, my = 0,
  mb = 0, mb_prev = 0,
  mw = 0,
  -- Keyboard queues drained each frame
  char_queue = {}, key_queue = {},
  -- Widget interaction
  active_id = nil,      -- slider being dragged
  focused_id = nil,     -- text input with focus
  disabled_depth = 0,
  font_stack = {},
  next_width = nil,     -- SetNextItemWidth override
  -- Retained state keyed by string id (combos, headers, text inputs, scrollers)
  state = {},
  -- Frame bookkeeping
  should_close = false,
  popup = nil,           -- open combo popup descriptor
  popup_just_opened = false,
  popup_closed_id = nil, -- id of popup closed this frame (prevent immediate reopen)
  deferred = {},         -- fns drawn on top in frame_end
}
M.ctx = ctx

-- ── DRAWING HELPERS ───────────────────────────────────────────────

-- Filled rounded rectangle using horizontal strips (one rect per corner row).
-- Avoids gfx.circle which overlaps with the fill rects, causing double alpha
-- blending and visible corner artifacts when using semi-transparent colors.
local function filled_roundrect(x, y, w, h, r)
  r = math.min(r, math.floor(math.min(w, h) / 2))
  if r <= 0 then gfx.rect(x, y, w, h, 1); return end
  -- Corner rows: each strip widens as we move away from the corner tip
  for i = 0, r - 1 do
    local dx   = math.floor(math.sqrt(r * r - (r - i) * (r - i)) + 0.5)
    local left = r - dx
    gfx.rect(x + left, y + i,         w - 2 * left, 1, 1)
    gfx.rect(x + left, y + h - 1 - i, w - 2 * left, 1, 1)
  end
  -- Middle section: full width
  gfx.rect(x, y + r, w, h - 2 * r, 1)
end

-- ── INTERNAL HELPERS ──────────────────────────────────────────────

local function cur_font()
  return ctx.font_stack[#ctx.font_stack] or theme.F.UI
end

local function af()
  return ctx.disabled_depth > 0 and 0.38 or 1.0
end

local function in_rect(x, y, w, h)
  return ctx.mx >= x and ctx.mx < x+w and ctx.my >= y and ctx.my < y+h
end

local function just_clicked()
  return ctx.mb == 1 and ctx.mb_prev == 0
end

-- Resolve width: nil→remaining, >=0→literal, <0→remaining+w
local function resolve_w(w)
  local rem = ctx.content_w - (ctx.x - theme.PAD_X)
  if w == nil then return math.max(1, rem) end
  if w >= 0   then return w end
  return math.max(1, rem + w)
end

local function advance(x, y, w, h)
  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, w, h
  ctx.x = theme.PAD_X
  ctx.y = y + h + theme.SPACING_Y
end

-- screen_y: convert logical Y (inside scroll_region) to screen Y
local function sy(logical_y)
  return logical_y + ctx.clip_y_off
end

-- Returns true if a widget at logical Y with height h is outside the clip region
local function clipped(logical_y, h)
  if not ctx.clip_y1 then return false end
  local s = sy(logical_y)
  return s < ctx.clip_y1 or s + h > ctx.clip_y2
end

-- ── FRAME ─────────────────────────────────────────────────────────

function M.frame_begin()
  -- Read mouse
  ctx.mx = gfx.mouse_x
  ctx.my = gfx.mouse_y
  ctx.mb = gfx.mouse_cap & 1
  ctx.mw = gfx.mouse_wheel
  gfx.mouse_wheel = 0

  -- Drain keyboard buffer
  ctx.char_queue = {}
  ctx.key_queue  = {}
  while true do
    local c = gfx.getchar()
    if     c == -1 then ctx.should_close = true; break
    elseif c ==  0 then break
    elseif c == 27 then ctx.should_close = true   -- Escape
    elseif c >= 32 and c < 127 then
      table.insert(ctx.char_queue, string.char(c))
    else
      table.insert(ctx.key_queue, c)
    end
  end

  -- Handle popup close (click outside popup rect)
  ctx.popup_closed_id = nil
  if ctx.popup and not ctx.popup_just_opened then
    if just_clicked() then
      local p = ctx.popup
      if not in_rect(p.x, p.y, p.w, p.h) then
        ctx.popup_closed_id = p.id
        ctx.popup = nil
      end
    end
  end
  ctx.popup_just_opened = false

  -- Clear background
  local bg = theme.C.BG
  gfx.set(bg[1], bg[2], bg[3], 1)
  gfx.rect(0, 0, gfx.w, gfx.h, 1)

  -- Reset layout
  ctx.x          = theme.PAD_X
  ctx.y          = theme.PAD_Y
  ctx.content_w  = gfx.w - 2 * theme.PAD_X
  ctx.clip_y1    = nil
  ctx.clip_y2    = nil
  ctx.clip_y_off = 0
  ctx.disabled_depth = 0
  ctx.font_stack = {}
  ctx.next_width = nil
  ctx.deferred   = {}

  gfx.setfont(theme.F.UI)
end

function M.frame_end()
  -- Draw popups and other deferred elements on top
  for _, fn in ipairs(ctx.deferred) do fn() end
  ctx.mb_prev = ctx.mb
  gfx.update()
end

-- ── LAYOUT ────────────────────────────────────────────────────────

function M.same_line(spacing)
  spacing = spacing ~= nil and spacing or theme.SPACING_X
  ctx.x = ctx.last_x + ctx.last_w + spacing
  ctx.y = ctx.last_y
end

function M.spacing(n)
  ctx.y = ctx.y + (n or 1) * theme.SPACING_Y
end

function M.separator()
  local t = theme
  local y = ctx.y + 2
  if not clipped(y, 1) then
    local c = t.C.SEP
    gfx.set(c[1], c[2], c[3], 1)
    gfx.line(t.PAD_X, sy(y), t.PAD_X + ctx.content_w, sy(y))
  end
  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = t.PAD_X, ctx.y, ctx.content_w, 5
  ctx.x = t.PAD_X
  ctx.y = y + 3 + t.SPACING_Y
end

function M.next_width(w)
  ctx.next_width = w
end

function M.push_font(f)
  table.insert(ctx.font_stack, f)
  gfx.setfont(f)
end

function M.pop_font()
  table.remove(ctx.font_stack)
  gfx.setfont(cur_font())
end

function M.begin_disabled(cond)
  if cond == nil or cond then ctx.disabled_depth = ctx.disabled_depth + 1 end
end

function M.end_disabled()
  if ctx.disabled_depth > 0 then ctx.disabled_depth = ctx.disabled_depth - 1 end
end

-- ── TEXT ──────────────────────────────────────────────────────────

function M.text(s)
  gfx.setfont(cur_font())
  local tw, th = gfx.measurestr(s)
  local x, y = ctx.x, ctx.y
  if not clipped(y, th) then
    local c = theme.C.FG
    gfx.set(c[1], c[2], c[3], af())
    gfx.x = x; gfx.y = sy(y)
    gfx.drawstr(s)
  end
  advance(x, y, tw, th)
end

function M.text_colored(s, color_name, alpha_override)
  gfx.setfont(cur_font())
  local tw, th = gfx.measurestr(s)
  local x, y = ctx.x, ctx.y
  if not clipped(y, th) then
    local c = theme.C[color_name] or theme.C.FG
    gfx.set(c[1], c[2], c[3], (alpha_override or 1.0) * af())
    gfx.x = x; gfx.y = sy(y)
    gfx.drawstr(s)
  end
  advance(x, y, tw, th)
end

function M.text_disabled(s)
  gfx.setfont(cur_font())
  local tw, th = gfx.measurestr(s)
  local x, y = ctx.x, ctx.y
  if not clipped(y, th) then
    local c = theme.C.FG_DIM
    gfx.set(c[1], c[2], c[3], af())
    gfx.x = x; gfx.y = sy(y)
    gfx.drawstr(s)
  end
  advance(x, y, tw, th)
end

function M.text_wrapped(s)
  local t = theme
  local max_w = ctx.content_w - (ctx.x - t.PAD_X)
  gfx.setfont(cur_font())
  local lh = select(2, gfx.measurestr("Ay"))
  local start_x, start_y = ctx.x, ctx.y
  local line = ""
  local row = 0

  local function flush_line(l)
    if l == "" then return end
    local y_off = row * (lh + 2)
    if not clipped(start_y + y_off, lh) then
      local c = t.C.FG
      gfx.set(c[1], c[2], c[3], af())
      gfx.x = start_x
      gfx.y = sy(start_y + y_off)
      gfx.drawstr(l)
    end
  end

  for w in s:gmatch("%S+") do
    local test = line == "" and w or (line .. " " .. w)
    if gfx.measurestr(test) > max_w and line ~= "" then
      flush_line(line)
      row = row + 1
      line = w
    else
      line = test
    end
  end
  flush_line(line)
  local total_h = (row + 1) * (lh + 2)
  advance(start_x, start_y, max_w, total_h)
end

-- Draws a label then advances X only (Y stays — next widget shares this row)
function M.row_label(s, label_w)
  label_w = label_w or 90
  gfx.setfont(cur_font())
  local _, th = gfx.measurestr(s)
  local x, y = ctx.x, ctx.y
  if not clipped(y, theme.ITEM_H) then
    local c = theme.C.FG
    gfx.set(c[1], c[2], c[3], af())
    gfx.x = x
    gfx.y = sy(y) + math.floor((theme.ITEM_H - th) / 2)
    gfx.drawstr(s)
  end
  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, label_w, theme.ITEM_H
  ctx.x = x + label_w
  -- y stays — next widget is on the same row
end

-- Inline text: draws text, advances X only (same row as previous same_line)
function M.inline_text(s, color_name)
  gfx.setfont(cur_font())
  local tw, th = gfx.measurestr(s)
  local x, y = ctx.x, ctx.y
  if not clipped(y, theme.ITEM_H) then
    local c = color_name and theme.C[color_name] or theme.C.FG_DIM
    gfx.set(c[1], c[2], c[3], af())
    gfx.x = x
    gfx.y = sy(y) + math.floor((theme.ITEM_H - th) / 2)
    gfx.drawstr(s)
  end
  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, tw, theme.ITEM_H
  ctx.x = x + tw
end

-- ── BUTTON ────────────────────────────────────────────────────────

-- opts.solid: {norm={r,g,b}, hover={r,g,b}, active={r,g,b}} for solid-fill buttons
function M.button(label, w, h, opts)
  opts = opts or {}
  local t = theme
  h = h or t.BTN_H
  local nw = ctx.next_width; ctx.next_width = nil
  w = resolve_w(nw ~= nil and nw or w)

  local x, y = ctx.x, ctx.y
  local a = af()
  local hover   = in_rect(x, y, w, h) and ctx.disabled_depth == 0 and not ctx.popup
  local pressed = hover and ctx.mb == 1
  local clicked = hover and just_clicked()

  if not clipped(y, h) then
    local gy = sy(y)
    -- Background
    if opts.solid then
      local sc = pressed and opts.solid.active or (hover and opts.solid.hover or opts.solid.norm)
      if sc then gfx.set(sc[1], sc[2], sc[3], a) end
    else
      local ba = pressed and t.A_BTN_ACT or (hover and t.A_BTN_HOV or t.A_BTN_NORM)
      local ac = t.C.ACCENT
      gfx.set(ac[1], ac[2], ac[3], ba * a)
    end
    filled_roundrect(x, gy, w, h, t.ROUND)

    -- Label
    gfx.setfont(cur_font())
    local tw, th = gfx.measurestr(label)
    local fc = t.C.FG
    gfx.set(fc[1], fc[2], fc[3], a)
    gfx.x = x + math.floor((w - tw) / 2)
    gfx.y = gy + math.floor((h - th) / 2)
    gfx.drawstr(label)
  end

  advance(x, y, w, h)
  return clicked
end

-- ── CHECKBOX ──────────────────────────────────────────────────────

-- Returns: clicked (bool), new_value (bool)
function M.checkbox(label, value)
  local t = theme
  local h = t.ITEM_H
  local bs = t.CHECK_BOX_S
  local x, y = ctx.x, ctx.y
  local disp = label:match("^(.-)##") or label   -- strip ##id suffix
  gfx.setfont(cur_font())
  local lw = gfx.measurestr(disp)
  local w = bs + 6 + lw

  local a = af()
  local hover   = in_rect(x, y, w, h) and ctx.disabled_depth == 0 and not ctx.popup
  local clicked = hover and just_clicked()

  if not clipped(y, h) then
    local gy = sy(y)
    local bx = x
    local by = gy + math.floor((h - bs) / 2)

    local bg = hover and t.C.FRAME_HOV or t.C.FRAME
    gfx.set(bg[1], bg[2], bg[3], a)
    gfx.rect(bx, by, bs, bs, 1)

    if value then
      local cc = t.C.ACCENT
      gfx.set(cc[1], cc[2], cc[3], a)
      local f = bs / 14
      local p2 = math.floor(2*f+0.5); local p5 = math.floor(5*f+0.5)
      local p6 = math.floor(6*f+0.5); local p3 = math.floor(3*f+0.5)
      gfx.line(bx+p2,    by+p6,      bx+p5,    by+bs-p3, 1)
      gfx.line(bx+p5,    by+bs-p3,   bx+bs-p2, by+p2,    1)
    end

    local _, th = gfx.measurestr(disp)
    local fc = t.C.FG
    gfx.set(fc[1], fc[2], fc[3], a)
    gfx.x = x + bs + 6
    gfx.y = gy + math.floor((h - th) / 2)
    gfx.drawstr(disp)
  end

  -- NOTE: don't use `clicked and (not value) or value` — that pattern breaks
  -- when `not value` is false (i.e. value=true), returning true instead of false.
  local new_val = value
  if clicked then new_val = not value end
  advance(x, y, w, h)
  return clicked, new_val
end

-- ── PROGRESS BAR ──────────────────────────────────────────────────

function M.progress_bar(val, w, h, label)
  local t = theme
  h = h or 16
  local nw = ctx.next_width; ctx.next_width = nil
  w = resolve_w(nw ~= nil and nw or w)
  local x, y = ctx.x, ctx.y
  val = math.max(0, math.min(1, val or 0))

  if not clipped(y, h) then
    local gy = sy(y)
    local fc = t.C.FRAME
    gfx.set(fc[1], fc[2], fc[3], 1)
    gfx.rect(x, gy, w, h, 1)

    if val > 0 then
      local fw = math.max(1, math.floor(w * val))
      local ac = t.C.ACCENT
      gfx.set(ac[1], ac[2], ac[3], 1)
      gfx.rect(x, gy, fw, h, 1)
    end

    if label and label ~= "" then
      gfx.setfont(cur_font())
      local tw, th = gfx.measurestr(label)
      local lc = t.C.FG
      gfx.set(lc[1], lc[2], lc[3], 1)
      gfx.x = x + math.floor((w - tw) / 2)
      gfx.y = gy + math.floor((h - th) / 2)
      gfx.drawstr(label)
    end
  end

  advance(x, y, w, h)
end

-- ── SLIDERS ───────────────────────────────────────────────────────

local function draw_slider(id, val, vmin, vmax, fmt, is_int)
  local t = theme
  local h = t.ITEM_H
  local nw = ctx.next_width; ctx.next_width = nil
  local w = resolve_w(nw)
  local x, y = ctx.x, ctx.y
  local grab_w, grab_h = t.SLIDER_GRAB_W, t.SLIDER_GRAB_H

  local a = af()
  local hover = in_rect(x, y, w, h) and ctx.disabled_depth == 0 and not ctx.popup

  if hover and ctx.mb == 1 then ctx.active_id = id end
  local is_active = (ctx.active_id == id) and ctx.mb == 1
  if ctx.mb == 0 and ctx.active_id == id then ctx.active_id = nil end

  local changed = false
  if is_active and ctx.disabled_depth == 0 then
    local rel = (ctx.mx - x - grab_w/2) / math.max(1, w - grab_w)
    local new_v = vmin + math.max(0, math.min(1, rel)) * (vmax - vmin)
    if is_int then new_v = math.floor(new_v + 0.5) end
    if new_v ~= val then changed = true end
    val = new_v
  end
  val = math.max(vmin, math.min(vmax, val))

  if not clipped(y, h) then
    local gy = sy(y)
    local track_y = gy + math.floor((h - 6) / 2)

    local fc = (hover or is_active) and t.C.FRAME_HOV or t.C.FRAME
    gfx.set(fc[1], fc[2], fc[3], a)
    gfx.rect(x, track_y, w, 6, 1)

    local norm = vmax > vmin and (val - vmin) / (vmax - vmin) or 0
    local gx = x + math.floor(norm * (w - grab_w))
    local gy2 = gy + math.floor((h - grab_h) / 2)
    local gc = is_active and t.C.ACCENT_HOV or t.C.ACCENT
    gfx.set(gc[1], gc[2], gc[3], a)
    gfx.rect(gx, gy2, grab_w, grab_h, 1)

    local display = string.format(fmt, val)
    gfx.setfont(cur_font())
    local tw, th = gfx.measurestr(display)
    local tc = t.C.FG
    gfx.set(tc[1], tc[2], tc[3], a)
    gfx.x = x + math.floor((w - tw) / 2)
    gfx.y = gy + math.floor((h - th) / 2)
    gfx.drawstr(display)
  end

  advance(x, y, w, h)
  return changed, val
end

function M.slider_int(id, val, vmin, vmax, fmt)
  return draw_slider(id, val, vmin, vmax, fmt or "%d", true)
end

function M.slider_float(id, val, vmin, vmax, fmt)
  return draw_slider(id, val, vmin, vmax, fmt or "%.2f", false)
end

return M
