-- lib/widgets_extra.lua  Complex retained-state widgets (tab bar, combo, input_text,
--                         collapsing_header, scroll_region).

local gui   = require("gui")
local theme = require("theme")

local M = {}
local ctx = gui.ctx

-- just_clicked: true on the frame a mouse button goes from 0→1
local function just_clicked()
  return ctx.mb == 1 and ctx.mb_prev == 0
end

-- Strip ImGui-style ##id suffix from display labels ("Label##id" → "Label")
local function strip_id(s) return s:match("^(.-)##") or s end

-- ── TAB BAR ───────────────────────────────────────────────────────

-- Returns new active index (1-indexed). Draws all tabs at current cursor Y.
function M.tab_bar(id, active_idx, tabs)
  local t = theme
  local tab_h = t.ITEM_H + 6
  local x, y = ctx.x, ctx.y

  gfx.setfont(t.F.UI)
  local tab_widths = {}
  for i, name in ipairs(tabs) do
    tab_widths[i] = math.max(80, gfx.measurestr(name) + 24)
  end

  local new_idx = active_idx
  local off_x = 0
  for i, name in ipairs(tabs) do
    local tx = x + off_x
    local tw = tab_widths[i]
    local is_active = (i == active_idx)
    local hover = ctx.mx >= tx and ctx.mx < tx + tw
               and ctx.my >= y and ctx.my < y + tab_h
               and ctx.disabled_depth == 0 and not ctx.popup

    if hover and ctx.mb == 1 and ctx.mb_prev == 0 then
      new_idx = i
    end

    if is_active then
      local c = t.C.FRAME_ACT; gfx.set(c[1], c[2], c[3], 1)
    elseif hover then
      local c = t.C.FRAME_HOV; gfx.set(c[1], c[2], c[3], 1)
    else
      local c = t.C.FRAME; gfx.set(c[1], c[2], c[3], 0.6)
    end
    gfx.rect(tx, y, tw, tab_h, 1)

    if is_active then
      local ac = t.C.ACCENT; gfx.set(ac[1], ac[2], ac[3], 1)
      gfx.rect(tx, y + tab_h - 2, tw, 2, 1)
    end

    local tlw, tlh = gfx.measurestr(name)
    local fc = is_active and t.C.FG or t.C.FG_DIM
    gfx.set(fc[1], fc[2], fc[3], 1)
    gfx.x = tx + math.floor((tw - tlw) / 2)
    gfx.y = y  + math.floor((tab_h - tlh) / 2)
    gfx.drawstr(name)

    off_x = off_x + tw + 1
  end

  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, ctx.content_w, tab_h
  ctx.x = theme.PAD_X
  ctx.y = y + tab_h + theme.SPACING_Y
  return new_idx
end

-- ── COLLAPSING HEADER ─────────────────────────────────────────────

-- Returns true when content should be drawn.
function M.collapsing_header(label, default_open)
  local t = theme
  local h = t.ITEM_H + 4
  local x, y = ctx.x, ctx.y
  local w = ctx.content_w

  local sid = "hdr_" .. label
  if ctx.state[sid] == nil then
    ctx.state[sid] = (default_open ~= false)
  end
  local is_open = ctx.state[sid]

  local hover = ctx.mx >= x and ctx.mx < x + w
             and ctx.my >= y and ctx.my < y + h
             and ctx.disabled_depth == 0

  if hover and ctx.mb == 1 and ctx.mb_prev == 0 then
    is_open = not is_open
    ctx.state[sid] = is_open
  end

  local bg = hover and t.C.FRAME_HOV or t.C.FRAME
  gfx.set(bg[1], bg[2], bg[3], 0.7)
  gfx.rect(x, y, w, h, 1)

  -- Triangle arrow
  local tri_x = x + 8
  local tri_cy = y + math.floor(h / 2)
  local dc = t.C.FG_DIM; gfx.set(dc[1], dc[2], dc[3], 1)
  if is_open then
    gfx.line(tri_x,     tri_cy-3, tri_x+6,  tri_cy-3)
    gfx.line(tri_x,     tri_cy-3, tri_x+3,  tri_cy+2)
    gfx.line(tri_x+6,   tri_cy-3, tri_x+3,  tri_cy+2)
  else
    gfx.line(tri_x,     tri_cy-5, tri_x,    tri_cy+5)
    gfx.line(tri_x,     tri_cy-5, tri_x+5,  tri_cy)
    gfx.line(tri_x,     tri_cy+5, tri_x+5,  tri_cy)
  end

  gfx.setfont(t.F.UI)
  local _, lh = gfx.measurestr(label)
  local fc = t.C.FG; gfx.set(fc[1], fc[2], fc[3], 1)
  gfx.x = x + 22
  gfx.y = y + math.floor((h - lh) / 2)
  gfx.drawstr(label)

  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, w, h
  ctx.x = theme.PAD_X
  ctx.y = y + h + (is_open and theme.SPACING_Y or 0)
  return is_open
end

-- ── COMBO (dropdown) ──────────────────────────────────────────────

-- Returns new_idx (1-indexed; same as input if no change).
function M.combo(id, idx, items)
  local t = theme
  local h = t.ITEM_H
  local nw = ctx.next_width; ctx.next_width = nil
  local rem = ctx.content_w - (ctx.x - t.PAD_X)
  local w = nw and (nw >= 0 and nw or math.max(1, rem + nw)) or rem
  local x, y = ctx.x, ctx.y

  -- Apply pending selection from popup click
  local sel_key = "csel_" .. id
  if ctx.state[sel_key] then
    idx = ctx.state[sel_key]
    ctx.state[sel_key] = nil
  end

  local a = ctx.disabled_depth > 0 and 0.38 or 1.0
  local popup_mine = ctx.popup and ctx.popup.id == id
  local hover = ctx.mx >= x and ctx.mx < x+w
             and ctx.my >= y and ctx.my < y+h
             and ctx.disabled_depth == 0

  -- Background
  local bg = (popup_mine or hover) and t.C.FRAME_HOV or t.C.FRAME
  gfx.set(bg[1], bg[2], bg[3], a)
  gfx.rect(x, y, w, h, 1)

  -- Selected label
  gfx.setfont(t.F.UI)
  local label = items[idx] or ""
  local _, lh = gfx.measurestr(label)
  local fc = t.C.FG; gfx.set(fc[1], fc[2], fc[3], a)
  gfx.x = x + 6
  gfx.y = y + math.floor((h - lh) / 2)
  gfx.drawstr(label)

  -- Down arrow
  local ax = x + w - 14
  local ay = y + math.floor(h / 2)
  local dc = t.C.FG_DIM; gfx.set(dc[1], dc[2], dc[3], a)
  gfx.line(ax, ay-3, ax+6, ay-3)
  gfx.line(ax, ay-3, ax+3, ay+2)
  gfx.line(ax+6, ay-3, ax+3, ay+2)

  -- Open popup on click (only if no OTHER popup open and not just closed this one)
  if hover and just_clicked() and not ctx.popup
  and ctx.popup_closed_id ~= id and ctx.disabled_depth == 0 then
    local item_h = h + 2
    local pad_y  = 4
    local p_h    = #items * item_h + 2 * pad_y
    local p_y    = y + h + 2
    if p_y + p_h > gfx.h - 4 then p_y = y - p_h - 2 end
    ctx.popup = {
      id      = id,
      x       = x, y = p_y, w = w, h = p_h,
      items   = items, item_h = item_h, pad_y = pad_y,
      cur_idx = idx, sel_key = sel_key,
    }
    ctx.popup_just_opened = true
  end

  -- Re-register deferred draw every frame while popup is open for this combo.
  -- ctx.deferred is reset each frame_begin, so without this the popup only
  -- renders on the first frame and ctx.popup stays set blocking other widgets.
  if ctx.popup and ctx.popup.id == id then
    local p_ref = ctx.popup
    table.insert(ctx.deferred, function() M._draw_combo_popup(p_ref) end)
  end

  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, w, h
  ctx.x = t.PAD_X
  ctx.y = y + h + t.SPACING_Y
  return idx
end

function M._draw_combo_popup(p)
  if not p then return end
  local t = theme
  local bg = t.C.POPUP_BG; gfx.set(bg[1], bg[2], bg[3], 1)
  gfx.rect(p.x, p.y, p.w, p.h, 1)

  gfx.setfont(t.F.UI)
  for i, item in ipairs(p.items) do
    local iy = p.y + p.pad_y + (i-1) * p.item_h
    local hover = ctx.mx >= p.x and ctx.mx < p.x + p.w
               and ctx.my >= iy and ctx.my < iy + p.item_h

    if hover then
      local hc = t.C.FRAME_HOV; gfx.set(hc[1], hc[2], hc[3], 1)
      gfx.rect(p.x+2, iy, p.w-4, p.item_h, 1)
    end
    if i == p.cur_idx then
      local ac = t.C.ACCENT; gfx.set(ac[1], ac[2], ac[3], 0.2)
      gfx.rect(p.x+2, iy, p.w-4, p.item_h, 1)
    end

    local _, lh = gfx.measurestr(item)
    local fc = hover and t.C.FG or t.C.FG_DIM
    gfx.set(fc[1], fc[2], fc[3], 1)
    gfx.x = p.x + 8
    gfx.y = iy + math.floor((p.item_h - lh) / 2)
    gfx.drawstr(item)

    if hover and ctx.mb == 1 and ctx.mb_prev == 0 then
      ctx.state[p.sel_key] = i
      ctx.popup = nil
    end
  end

  local bc = t.C.FRAME_HOV; gfx.set(bc[1], bc[2], bc[3], 1)
  gfx.rect(p.x, p.y, p.w, 1, 1)
  gfx.rect(p.x, p.y+p.h-1, p.w, 1, 1)
  gfx.rect(p.x, p.y, 1, p.h, 1)
  gfx.rect(p.x+p.w-1, p.y, 1, p.h, 1)
end

-- ── INPUT TEXT ────────────────────────────────────────────────────

-- opts: { width, readonly, password }
-- Returns: changed (bool), new_text (string)
function M.input_text(id, text, opts)
  opts = opts or {}
  local t = theme
  local h = t.ITEM_H
  local nw = ctx.next_width; ctx.next_width = nil
  local rem = ctx.content_w - (ctx.x - t.PAD_X)
  local w
  if opts.width then
    w = opts.width >= 0 and opts.width or math.max(1, rem + opts.width)
  elseif nw then
    w = nw >= 0 and nw or math.max(1, rem + nw)
  else
    w = rem
  end
  local x, y = ctx.x, ctx.y

  if not ctx.state[id] then
    ctx.state[id] = { caret = #text, blink_t = os.clock() }
  end
  local s = ctx.state[id]
  -- Keep caret in bounds after external text changes
  s.caret = math.max(0, math.min(s.caret, #text))

  -- Click to focus and position caret
  local hover_field = ctx.mx >= x and ctx.mx < x+w
                   and ctx.my >= y and ctx.my < y+h
                   and ctx.disabled_depth == 0 and not opts.readonly
  if hover_field and just_clicked() then
    ctx.focused_id = id
    gfx.setfont(t.F.UI)
    local rel_x = ctx.mx - x - 4
    local display = opts.password and string.rep("*", #text) or text
    s.caret = #display
    for i = 1, #display do
      if gfx.measurestr(display:sub(1, i)) > rel_x then
        s.caret = i - 1; break
      end
    end
    s.blink_t = os.clock()
  end
  -- Unfocus on click elsewhere
  if not hover_field and just_clicked() and ctx.focused_id == id then
    ctx.focused_id = nil
  end

  -- Process input when focused
  local changed = false
  local new_text = text
  if ctx.focused_id == id and not opts.readonly then
    for _, ch in ipairs(ctx.char_queue) do
      new_text = new_text:sub(1, s.caret) .. ch .. new_text:sub(s.caret+1)
      s.caret = s.caret + 1
      changed = true
    end
    ctx.char_queue = {}

    for _, k in ipairs(ctx.key_queue) do
      if k == 8 or k == 127 then           -- backspace
        if s.caret > 0 then
          new_text = new_text:sub(1, s.caret-1) .. new_text:sub(s.caret+1)
          s.caret = s.caret - 1
          changed = true
        end
      elseif k == 1818584692 then           -- left arrow
        s.caret = math.max(0, s.caret-1)
      elseif k == 1919379572 then           -- right arrow
        s.caret = math.min(#new_text, s.caret+1)
      elseif k == 1752132965 then           -- home
        s.caret = 0
      elseif k == 6647396 then              -- end
        s.caret = #new_text
      end
    end
    ctx.key_queue = {}
    if changed then s.blink_t = os.clock() end
  end
  s.caret = math.max(0, math.min(s.caret, #new_text))

  -- Draw
  local focused = (ctx.focused_id == id)
  local bg = focused and t.C.FRAME_ACT or (hover_field and t.C.FRAME_HOV or t.C.FRAME)
  gfx.set(bg[1], bg[2], bg[3], 1)
  gfx.rect(x, y, w, h, 1)

  if focused then
    local ac = t.C.ACCENT; gfx.set(ac[1], ac[2], ac[3], 0.5)
    gfx.rect(x, y, w, 1, 1)
    gfx.rect(x, y+h-1, w, 1, 1)
    gfx.rect(x, y, 1, h, 1)
    gfx.rect(x+w-1, y, 1, h, 1)
  end

  gfx.setfont(t.F.UI)
  local display = opts.password and string.rep("*", #new_text) or new_text
  local max_tw = w - 8
  local _, th = gfx.measurestr("Ay")
  local fc = opts.readonly and t.C.FG_DIM or t.C.FG
  gfx.set(fc[1], fc[2], fc[3], 1)
  -- Scroll display so caret is visible
  local caret_disp = opts.password and string.rep("*", s.caret) or new_text:sub(1, s.caret)
  local caret_px = gfx.measurestr(caret_disp)
  local draw_disp = display
  if caret_px > max_tw then
    -- Show the tail of the string starting from where caret would be in view
    local trim = 0
    for i = 1, #display do
      if gfx.measurestr(display:sub(i)) <= max_tw then trim = i-1; break end
    end
    draw_disp = display:sub(trim+1)
    caret_px = gfx.measurestr(draw_disp:sub(1, s.caret - trim))
  end
  gfx.x = x + 4
  gfx.y = y + math.floor((h - th) / 2)
  gfx.drawstr(draw_disp)

  -- Blinking caret
  if focused then
    local blink_on = math.floor((os.clock() - s.blink_t) * 2) % 2 == 0
    if blink_on then
      local cx = x + 4 + caret_px
      local cc = t.C.FG; gfx.set(cc[1], cc[2], cc[3], 1)
      gfx.line(cx, y+3, cx, y+h-3)
    end
  end

  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, w, h
  ctx.x = t.PAD_X
  ctx.y = y + h + t.SPACING_Y
  return changed, new_text
end

-- ── SCROLL REGION ─────────────────────────────────────────────────

-- draw_fn draws content using gui.* calls.
-- Content Y coordinates are logical (start at 0); screen mapping is automatic.
function M.scroll_region(id, w, h, draw_fn)
  local t = theme
  local x, y = ctx.x, ctx.y
  local sw = t.SCROLL_W + 2  -- scrollbar strip

  if not w or w == 0 then
    w = ctx.content_w - (ctx.x - t.PAD_X)
  elseif w < 0 then
    w = math.max(1, (ctx.content_w - (ctx.x - t.PAD_X)) + w)
  end
  local inner_w = w - sw

  if not ctx.state[id] then
    ctx.state[id] = { scroll_y = 0, content_h = h }
  end
  local s = ctx.state[id]

  -- Wheel scroll
  -- gfx.mouse_wheel gives ±120 per step (Windows) or fine deltas (macOS trackpad).
  -- Normalise: treat anything > 60 magnitude as discrete step of 20px.
  if ctx.mw ~= 0 and ctx.mx >= x and ctx.mx < x+w
  and ctx.my >= y and ctx.my < y+h then
    local delta = math.abs(ctx.mw) > 60
      and (ctx.mw > 0 and -20 or 20)
      or  -ctx.mw * 0.25
    s.scroll_y = s.scroll_y + delta
  end
  if s.scroll_to_bottom then
    s.scroll_y = math.max(0, (s.content_h or 0) - h)
    s.scroll_to_bottom = false
  end

  -- Scrollbar thumb drag (uses content_h from previous frame — accurate after first render)
  local max_sc_drag = math.max(0, (s.content_h or h) - h)
  if max_sc_drag > 0 then
    local ratio_d   = h / (s.content_h or h)
    local thumb_h_d = math.max(16, math.floor(h * ratio_d))
    local sb_x_d    = x + inner_w + 2
    local thumb_y_d = y + math.floor(s.scroll_y / max_sc_drag * (h - thumb_h_d))

    local on_thumb = ctx.mx >= sb_x_d and ctx.mx < sb_x_d + t.SCROLL_W
                  and ctx.my >= thumb_y_d and ctx.my < thumb_y_d + thumb_h_d
    if on_thumb and ctx.mb == 1 and ctx.mb_prev == 0 then
      s.drag_active       = true
      s.drag_start_my     = ctx.my
      s.drag_start_scroll = s.scroll_y
    end

    if s.drag_active then
      if ctx.mb == 1 then
        local delta_my     = ctx.my - s.drag_start_my
        local delta_scroll = delta_my * max_sc_drag / (h - thumb_h_d)
        s.scroll_y = math.max(0, math.min(max_sc_drag,
                       s.drag_start_scroll + delta_scroll))
      else
        s.drag_active = false
      end
    end
  else
    s.drag_active = false
  end

  -- Draw background before text (text draws on top within clipped region)
  local bg = t.C.LOG_BG; gfx.set(bg[1], bg[2], bg[3], 1)
  gfx.rect(x, y, inner_w, h, 1)

  -- Save outer layout context
  local sv = {
    x=ctx.x, y=ctx.y, cw=ctx.content_w,
    lx=ctx.last_x, ly=ctx.last_y, lw=ctx.last_w, lh=ctx.last_h,
    cy1=ctx.clip_y1, cy2=ctx.clip_y2, cyo=ctx.clip_y_off,
  }

  -- Set up inner coordinate system:
  -- logical Y=0 maps to screen Y = y - scroll_y  (clip_y_off)
  s.scroll_y = math.max(0, math.min(math.max(0, (s.content_h or 0) - h), s.scroll_y))
  ctx.clip_y1    = y
  ctx.clip_y2    = y + h
  ctx.clip_y_off = y - math.floor(s.scroll_y)
  ctx.x          = x + 4
  ctx.y          = 4          -- logical Y with inner top-padding
  ctx.content_w  = inner_w - 8

  draw_fn()

  -- Compute content height from where the cursor ended up
  s.content_h = ctx.y + math.floor(s.scroll_y)

  -- Restore outer context
  ctx.x         = sv.x;  ctx.y         = sv.y;   ctx.content_w = sv.cw
  ctx.last_x    = sv.lx; ctx.last_y    = sv.ly;  ctx.last_w    = sv.lw; ctx.last_h = sv.lh
  ctx.clip_y1   = sv.cy1; ctx.clip_y2  = sv.cy2; ctx.clip_y_off = sv.cyo

  -- Clamp scroll after content height update
  s.scroll_y = math.max(0, math.min(math.max(0, s.content_h - h), s.scroll_y))

  -- Border
  local bc = t.C.FRAME; gfx.set(bc[1], bc[2], bc[3], 1)
  gfx.line(x,        y,    x+inner_w, y)
  gfx.line(x,        y+h,  x+inner_w, y+h)
  gfx.line(x,        y,    x,         y+h)
  gfx.line(x+inner_w, y,   x+inner_w, y+h)

  -- Scrollbar
  if s.content_h and s.content_h > h then
    local sb_x    = x + inner_w + 2
    local ratio   = h / s.content_h
    local thumb_h = math.max(16, math.floor(h * ratio))
    local max_sc  = s.content_h - h
    local thumb_y = max_sc > 0
      and (y + math.floor(s.scroll_y / max_sc * (h - thumb_h)))
      or  y

    local on_thumb_hover = ctx.mx >= sb_x and ctx.mx < sb_x + t.SCROLL_W
                        and ctx.my >= thumb_y and ctx.my < thumb_y + thumb_h

    local sc = t.C.SCROLLBAR; gfx.set(sc[1], sc[2], sc[3], 0.3)
    gfx.rect(sb_x, y, t.SCROLL_W, h, 1)

    local thumb_alpha = (s.drag_active or on_thumb_hover) and 1.0 or 0.8
    gfx.set(sc[1], sc[2], sc[3], thumb_alpha)
    gfx.rect(sb_x+1, thumb_y, t.SCROLL_W-2, thumb_h, 1)
  end

  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, w, h
  ctx.x = t.PAD_X
  ctx.y = y + h + t.SPACING_Y
end

-- Call to programmatically scroll a scroll_region to the bottom next frame.
function M.scroll_to_bottom(id)
  if ctx.state[id] then ctx.state[id].scroll_to_bottom = true end
end

return M
