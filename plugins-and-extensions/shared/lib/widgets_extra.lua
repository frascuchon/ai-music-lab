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

-- ── CLIPBOARD ─────────────────────────────────────────────────────

local function _clip_get()
  if reaper and reaper.CF_GetClipboard then
    local ok, s = pcall(reaper.CF_GetClipboard)
    if ok and type(s) == "string" then return s end
  end
  return ctx._clipboard or ""
end

local function _clip_set(s)
  if reaper and reaper.CF_SetClipboard then
    pcall(reaper.CF_SetClipboard, s)
  end
  ctx._clipboard = s
end

-- ── TEXT INPUT HELPERS ────────────────────────────────────────────

-- Selection is tracked as (sel_anchor, caret): range is [min, max).
-- Returns (a, b) in order, or nil,nil if no selection.
local function _sel_range(s)
  if s.sel_anchor == nil then return nil, nil end
  local a = math.min(s.sel_anchor, s.caret)
  local b = math.max(s.sel_anchor, s.caret)
  if a == b then return nil, nil end
  return a, b
end

-- Delete selected text. Returns (new_text, deleted).
local function _del_sel(s, txt)
  local a, b = _sel_range(s)
  if not a then return txt, false end
  local r = txt:sub(1, a) .. txt:sub(b + 1)
  s.caret = a; s.sel_anchor = nil
  return r, true
end

-- ── WORD-WRAP HELPER (for input_textarea) ─────────────────────────

-- Returns list of { text=str, start=int } where start is the 0-based caret
-- position at the beginning of each wrapped display line.
-- Handles hard \n and soft wrap at character level.
local function _wrap_lines(text, max_w)
  if #text == 0 then return {{ text = "", start = 0 }} end
  local result = {}
  local i = 1
  local line = ""
  local line_start = 0  -- 0-based caret position at start of current line

  while i <= #text do
    local ch = text:sub(i, i)
    if ch == "\n" then
      table.insert(result, { text = line, start = line_start })
      line = ""
      line_start = i  -- char after \n is at caret=i (1-based i → 0-based offset i)
      i = i + 1
    else
      local test = line .. ch
      if gfx.measurestr(test) > max_w and #line > 0 then
        table.insert(result, { text = line, start = line_start })
        line = ch
        line_start = i - 1  -- 0-based offset of this char
        i = i + 1
      else
        line = test
        i = i + 1
      end
    end
  end
  table.insert(result, { text = line, start = line_start })

  -- If text ends with \n, add trailing empty line
  if text:sub(#text, #text) == "\n" then
    table.insert(result, { text = "", start = #text })
  end

  return result
end

-- Return the 1-based index of the wrapped line that contains caret.
local function _caret_line(wrapped, caret)
  for i = #wrapped, 1, -1 do
    if caret >= wrapped[i].start then return i end
  end
  return 1
end

-- Return the pixel x of caret within wrapped line li.
local function _line_caret_px(wrapped, li, caret)
  local ln = wrapped[li]
  local off = math.max(0, math.min(caret - ln.start, #ln.text))
  return gfx.measurestr(ln.text:sub(1, off))
end

-- Move caret up one wrapped line keeping same column.
local function _move_up(wrapped, caret)
  local li = _caret_line(wrapped, caret)
  if li <= 1 then return 0 end
  local px = _line_caret_px(wrapped, li, caret)
  local prev = wrapped[li - 1]
  for c = #prev.text, 0, -1 do
    if gfx.measurestr(prev.text:sub(1, c)) <= px then
      return prev.start + c
    end
  end
  return prev.start
end

-- Move caret down one wrapped line keeping same column.
local function _move_down(wrapped, caret, text_len)
  local li = _caret_line(wrapped, caret)
  if li >= #wrapped then return text_len end
  local px = _line_caret_px(wrapped, li, caret)
  local nxt = wrapped[li + 1]
  for c = #nxt.text, 0, -1 do
    if gfx.measurestr(nxt.text:sub(1, c)) <= px then
      return nxt.start + c
    end
  end
  return nxt.start
end

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
    local scy_tab = y + ctx.clip_y_off
    local hover = ctx.mx >= tx and ctx.mx < tx + tw
               and ctx.my >= scy_tab and ctx.my < scy_tab + tab_h
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

  local scy_hdr = y + ctx.clip_y_off
  local hover = ctx.mx >= x and ctx.mx < x + w
             and ctx.my >= scy_hdr and ctx.my < scy_hdr + h
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
  local scy = y + ctx.clip_y_off
  local hover = ctx.mx >= x and ctx.mx < x+w
             and ctx.my >= scy and ctx.my < scy+h
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
    local p_y    = scy + h + 2  -- scy is already screen Y
    if p_y + p_h > gfx.h - 4 then p_y = scy - p_h - 2 end
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
-- Keyboard shortcuts: Ctrl+A (select all), Ctrl+C (copy), Ctrl+V (paste),
--   Ctrl+X (cut), Shift+Left/Right (extend selection).
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
  local gy = y + (ctx.clip_y_off or 0)  -- screen Y (y is logical inside scroll_region)

  if not ctx.state[id] then
    ctx.state[id] = { caret = #text, blink_t = os.clock(), sel_anchor = nil }
  end
  local s = ctx.state[id]
  if s.sel_anchor == nil then s.sel_anchor = nil end  -- compat
  s.caret = math.max(0, math.min(s.caret, #text))

  -- Click to focus and position caret
  local hover_field = ctx.mx >= x and ctx.mx < x+w
                   and ctx.my >= gy and ctx.my < gy+h
                   and ctx.disabled_depth == 0 and not opts.readonly
  if hover_field and just_clicked() then
    ctx.focused_id = id
    s.sel_anchor = nil
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
  if not hover_field and just_clicked() and ctx.focused_id == id then
    ctx.focused_id = nil
  end

  -- Process input when focused
  local changed = false
  local new_text = text
  if ctx.focused_id == id and not opts.readonly then
    for _, ch in ipairs(ctx.char_queue) do
      local nt, d = _del_sel(s, new_text); new_text = nt; if d then changed = true end
      new_text = new_text:sub(1, s.caret) .. ch .. new_text:sub(s.caret+1)
      s.caret = s.caret + 1
      changed = true
    end
    ctx.char_queue = {}

    for _, k in ipairs(ctx.key_queue) do
      if k == 1 then                        -- Ctrl+A: select all
        s.sel_anchor = 0; s.caret = #new_text
      elseif k == 3 then                    -- Ctrl+C: copy
        local a, b = _sel_range(s)
        _clip_set(a and new_text:sub(a+1, b) or new_text)
      elseif k == 24 then                   -- Ctrl+X: cut
        local a, b = _sel_range(s)
        if a then
          _clip_set(new_text:sub(a+1, b))
          local nt, _ = _del_sel(s, new_text); new_text = nt; changed = true
        end
      elseif k == 22 then                   -- Ctrl+V: paste
        local nt, _ = _del_sel(s, new_text); new_text = nt
        local clip = _clip_get()
        new_text = new_text:sub(1, s.caret) .. clip .. new_text:sub(s.caret+1)
        s.caret = s.caret + #clip; changed = true
      elseif k == 8 or k == 127 then        -- backspace
        local nt, d = _del_sel(s, new_text); new_text = nt
        if d then changed = true
        elseif s.caret > 0 then
          new_text = new_text:sub(1, s.caret-1) .. new_text:sub(s.caret+1)
          s.caret = s.caret - 1; changed = true
        end
      elseif k == 1818584692 then           -- left arrow
        if ctx.shift then
          if s.sel_anchor == nil then s.sel_anchor = s.caret end
          s.caret = math.max(0, s.caret - 1)
        else
          if s.sel_anchor ~= nil then
            s.caret = math.min(s.sel_anchor, s.caret); s.sel_anchor = nil
          else
            s.caret = math.max(0, s.caret - 1)
          end
        end
      elseif k == 1919379572 then           -- right arrow
        if ctx.shift then
          if s.sel_anchor == nil then s.sel_anchor = s.caret end
          s.caret = math.min(#new_text, s.caret + 1)
        else
          if s.sel_anchor ~= nil then
            s.caret = math.max(s.sel_anchor, s.caret); s.sel_anchor = nil
          else
            s.caret = math.min(#new_text, s.caret + 1)
          end
        end
      elseif k == 1752132965 then           -- home
        s.caret = 0; s.sel_anchor = nil
      elseif k == 6647396 then              -- end
        s.caret = #new_text; s.sel_anchor = nil
      end
    end
    ctx.key_queue = {}
    if changed then s.blink_t = os.clock() end
  end
  s.caret = math.max(0, math.min(s.caret, #new_text))

  -- Draw
  local focused = (ctx.focused_id == id)
  if ctx.clip_y1 and (gy + h < ctx.clip_y1 or gy > ctx.clip_y2) then
    ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, w, h
    ctx.x = t.PAD_X; ctx.y = y + h + t.SPACING_Y
    return changed, new_text
  end
  local bg = focused and t.C.FRAME_ACT or (hover_field and t.C.FRAME_HOV or t.C.FRAME)
  gfx.set(bg[1], bg[2], bg[3], 1)
  gfx.rect(x, gy, w, h, 1)

  if focused then
    local ac = t.C.ACCENT; gfx.set(ac[1], ac[2], ac[3], 0.5)
    gfx.rect(x, gy, w, 1, 1)
    gfx.rect(x, gy+h-1, w, 1, 1)
    gfx.rect(x, gy, 1, h, 1)
    gfx.rect(x+w-1, gy, 1, h, 1)
  end

  gfx.setfont(t.F.UI)
  local display = opts.password and string.rep("*", #new_text) or new_text
  local max_tw = w - 8
  local _, th = gfx.measurestr("Ay")
  -- Scroll display so caret is visible
  local caret_disp = opts.password and string.rep("*", s.caret) or new_text:sub(1, s.caret)
  local caret_px = gfx.measurestr(caret_disp)
  local trim = 0
  local draw_disp = display
  if caret_px > max_tw then
    for i = 1, #display do
      if gfx.measurestr(display:sub(i)) <= max_tw then trim = i-1; break end
    end
    draw_disp = display:sub(trim+1)
    caret_px = gfx.measurestr(draw_disp:sub(1, s.caret - trim))
  end

  -- Selection highlight (before drawing text)
  local sel_a, sel_b = _sel_range(s)
  if sel_a ~= nil then
    local vis_a = math.max(0, sel_a - trim)
    local vis_b = math.max(0, sel_b - trim)
    vis_a = math.min(vis_a, #draw_disp)
    vis_b = math.min(vis_b, #draw_disp)
    if vis_a < vis_b then
      local px_a = gfx.measurestr(draw_disp:sub(1, vis_a))
      local px_b = gfx.measurestr(draw_disp:sub(1, vis_b))
      local ac = t.C.ACCENT; gfx.set(ac[1], ac[2], ac[3], 0.3)
      gfx.rect(x + 4 + px_a, gy + 2, px_b - px_a, h - 4, 1)
    end
  end

  local fc = opts.readonly and t.C.FG_DIM or t.C.FG
  gfx.set(fc[1], fc[2], fc[3], 1)
  gfx.x = x + 4
  gfx.y = gy + math.floor((h - th) / 2)
  gfx.drawstr(draw_disp)

  -- Blinking caret
  if focused then
    local blink_on = math.floor((os.clock() - s.blink_t) * 2) % 2 == 0
    if blink_on then
      local cx = x + 4 + caret_px
      local cc = t.C.FG; gfx.set(cc[1], cc[2], cc[3], 1)
      gfx.line(cx, gy+3, cx, gy+h-3)
    end
  end

  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, w, h
  ctx.x = t.PAD_X
  ctx.y = y + h + t.SPACING_Y
  return changed, new_text
end

-- ── INPUT TEXTAREA ────────────────────────────────────────────────

-- Multi-line text area with word-wrap, selection and clipboard support.
-- lines_visible: number of visible text rows (default 3).
-- opts: { readonly, placeholder }
-- Keyboard: Ctrl+A/C/V/X, Enter (newline), arrow keys (incl. Up/Down across lines).
-- Returns: changed (bool), new_text (string)
function M.input_textarea(id, text, lines_visible, opts)
  opts = opts or {}
  lines_visible = lines_visible or 3
  local t = theme
  local nw = ctx.next_width; ctx.next_width = nil
  local rem = ctx.content_w - (ctx.x - t.PAD_X)
  local w = nw and (nw >= 0 and nw or math.max(1, rem + nw)) or rem
  local x, y = ctx.x, ctx.y
  local gy = y + (ctx.clip_y_off or 0)

  gfx.setfont(t.F.UI)
  local _, lh = gfx.measurestr("Ay")
  local ls = lh + 3           -- line spacing
  local pad = 5
  local sb_w = 6              -- scrollbar strip width
  local h = lines_visible * ls + pad * 2
  local text_w = math.max(10, w - pad * 2 - sb_w)

  if not ctx.state[id] then
    ctx.state[id] = { caret = #text, scroll_y = 0, blink_t = os.clock(), sel_anchor = nil }
  end
  local s = ctx.state[id]
  s.caret = math.max(0, math.min(s.caret, #text))

  -- Compute wrapped lines (every frame; cheap enough for prompt-length text)
  gfx.setfont(t.F.UI)
  local wrapped = _wrap_lines(text, text_w)

  -- Focus / click
  local hover_field = ctx.mx >= x and ctx.mx < x + w
                   and ctx.my >= gy and ctx.my < gy + h
                   and ctx.disabled_depth == 0 and not opts.readonly
  if hover_field and just_clicked() then
    ctx.focused_id = id
    s.sel_anchor = nil
    local rel_y = ctx.my - gy - pad + s.scroll_y
    local cli = math.max(1, math.min(#wrapped, math.floor(rel_y / ls) + 1))
    local ln  = wrapped[cli]
    local rel_x = ctx.mx - x - pad
    local coff = #ln.text
    for ci = 1, #ln.text do
      if gfx.measurestr(ln.text:sub(1, ci)) > rel_x then coff = ci - 1; break end
    end
    s.caret    = ln.start + coff
    s.blink_t  = os.clock()
  end
  if not hover_field and just_clicked() and ctx.focused_id == id then
    ctx.focused_id = nil
  end

  -- Mouse wheel scroll inside widget
  if hover_field and ctx.mw ~= 0 then
    local delta = math.abs(ctx.mw) > 60 and (ctx.mw > 0 and -20 or 20) or -ctx.mw * 0.25
    s.scroll_y = s.scroll_y + delta
  end

  local focused = (ctx.focused_id == id)

  -- Keyboard input
  local changed = false
  local new_text = text
  if focused and not opts.readonly then
    for _, ch in ipairs(ctx.char_queue) do
      local nt, d = _del_sel(s, new_text); new_text = nt; if d then changed = true end
      new_text = new_text:sub(1, s.caret) .. ch .. new_text:sub(s.caret + 1)
      s.caret  = s.caret + 1; changed = true
    end
    ctx.char_queue = {}

    for _, k in ipairs(ctx.key_queue) do
      if k == 1 then                          -- Ctrl+A
        s.sel_anchor = 0; s.caret = #new_text
      elseif k == 3 then                      -- Ctrl+C
        local a, b = _sel_range(s)
        _clip_set(a and new_text:sub(a+1, b) or new_text)
      elseif k == 24 then                     -- Ctrl+X
        local a, b = _sel_range(s)
        if a then
          _clip_set(new_text:sub(a+1, b))
          local nt, _ = _del_sel(s, new_text); new_text = nt; changed = true
        end
      elseif k == 22 then                     -- Ctrl+V
        local nt, _ = _del_sel(s, new_text); new_text = nt
        local clip = _clip_get()
        new_text = new_text:sub(1, s.caret) .. clip .. new_text:sub(s.caret + 1)
        s.caret  = s.caret + #clip; changed = true
      elseif k == 13 then                     -- Enter → newline
        local nt, d = _del_sel(s, new_text); new_text = nt; if d then changed = true end
        new_text = new_text:sub(1, s.caret) .. "\n" .. new_text:sub(s.caret + 1)
        s.caret  = s.caret + 1; changed = true
      elseif k == 8 or k == 127 then          -- backspace
        local nt, d = _del_sel(s, new_text); new_text = nt
        if d then changed = true
        elseif s.caret > 0 then
          new_text = new_text:sub(1, s.caret-1) .. new_text:sub(s.caret+1)
          s.caret  = s.caret - 1; changed = true
        end
      elseif k == 6579564 then                -- Delete (forward)
        local nt, d = _del_sel(s, new_text); new_text = nt
        if d then changed = true
        elseif s.caret < #new_text then
          new_text = new_text:sub(1, s.caret) .. new_text:sub(s.caret + 2); changed = true
        end
      elseif k == 1818584692 then             -- left
        s.sel_anchor = nil; s.caret = math.max(0, s.caret - 1)
      elseif k == 1919379572 then             -- right
        s.sel_anchor = nil; s.caret = math.min(#new_text, s.caret + 1)
      elseif k == 30064 then                  -- up
        s.sel_anchor = nil
        local wr2 = changed and _wrap_lines(new_text, text_w) or wrapped
        s.caret = _move_up(wr2, s.caret)
      elseif k == 1685026670 then             -- down
        s.sel_anchor = nil
        local wr2 = changed and _wrap_lines(new_text, text_w) or wrapped
        s.caret = _move_down(wr2, s.caret, #new_text)
      elseif k == 1752132965 then             -- home
        s.sel_anchor = nil
        local cli = _caret_line(wrapped, s.caret)
        s.caret = wrapped[cli].start
      elseif k == 6647396 then                -- end
        s.sel_anchor = nil
        local cli = _caret_line(wrapped, s.caret)
        local ln = wrapped[cli]
        s.caret = ln.start + #ln.text
      end
    end
    ctx.key_queue = {}
    if changed then s.blink_t = os.clock() end
  end
  s.caret = math.max(0, math.min(s.caret, #new_text))

  -- Recompute wrap if text changed
  if changed then
    gfx.setfont(t.F.UI)
    wrapped = _wrap_lines(new_text, text_w)
  end

  -- Scroll to keep caret visible when focused
  if focused then
    local cli = _caret_line(wrapped, s.caret)
    local caret_top = (cli - 1) * ls
    local vis_h = h - pad * 2
    if caret_top < s.scroll_y then
      s.scroll_y = caret_top
    elseif caret_top + lh > s.scroll_y + vis_h then
      s.scroll_y = caret_top + lh - vis_h
    end
  end

  local content_h = #wrapped * ls
  local max_scroll = math.max(0, content_h - (h - pad * 2))
  s.scroll_y = math.max(0, math.min(max_scroll, s.scroll_y))

  -- Skip draw if clipped by outer scroll_region
  if ctx.clip_y1 and (gy + h < ctx.clip_y1 or gy > ctx.clip_y2) then
    ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, w, h
    ctx.x = t.PAD_X; ctx.y = y + h + t.SPACING_Y
    return changed, new_text
  end

  -- Background
  local bg = focused and t.C.FRAME_ACT or (hover_field and t.C.FRAME_HOV or t.C.FRAME)
  gfx.set(bg[1], bg[2], bg[3], 1)
  gfx.rect(x, gy, w, h, 1)

  if focused then
    local ac = t.C.ACCENT; gfx.set(ac[1], ac[2], ac[3], 0.5)
    gfx.rect(x, gy, w, 1, 1); gfx.rect(x, gy+h-1, w, 1, 1)
    gfx.rect(x, gy, 1, h, 1); gfx.rect(x+w-1, gy, 1, h, 1)
  end

  gfx.setfont(t.F.UI)

  -- Selection range
  local sel_a, sel_b = _sel_range(s)

  -- Draw wrapped lines
  for li, ln in ipairs(wrapped) do
    local line_y = gy + pad + (li - 1) * ls - s.scroll_y
    if line_y + lh >= gy and line_y < gy + h then
      -- Selection highlight on this line
      if sel_a ~= nil then
        local s_a = math.max(sel_a, ln.start)
        local s_b = math.min(sel_b, ln.start + #ln.text)
        if s_a < s_b then
          local px_a = gfx.measurestr(ln.text:sub(1, s_a - ln.start))
          local px_b = gfx.measurestr(ln.text:sub(1, s_b - ln.start))
          local ac = t.C.ACCENT; gfx.set(ac[1], ac[2], ac[3], 0.3)
          local dy = math.max(gy + 1, line_y)
          gfx.rect(x + pad + px_a, dy, math.max(1, px_b - px_a), math.min(lh, gy + h - 1 - dy), 1)
        end
      end
      -- Text
      local fc = opts.readonly and t.C.FG_DIM or t.C.FG
      gfx.set(fc[1], fc[2], fc[3], 1)
      gfx.x = x + pad
      gfx.y = math.max(gy + 1, line_y)
      gfx.drawstr(ln.text, 0, x + w - pad - sb_w, gy + h - 1)
    end
  end

  -- Placeholder
  if #new_text == 0 and opts.placeholder then
    local fc = t.C.FG_DIM; gfx.set(fc[1], fc[2], fc[3], 0.5)
    gfx.x = x + pad; gfx.y = gy + pad
    gfx.drawstr(opts.placeholder, 0, x + w - pad, gy + h - 1)
  end

  -- Caret
  if focused then
    local cli = _caret_line(wrapped, s.caret)
    local ln  = wrapped[cli]
    local coff = math.max(0, math.min(s.caret - ln.start, #ln.text))
    local cpx  = gfx.measurestr(ln.text:sub(1, coff))
    local cy   = gy + pad + (cli - 1) * ls - s.scroll_y
    local blink_on = math.floor((os.clock() - s.blink_t) * 2) % 2 == 0
    if blink_on and cy >= gy and cy + lh <= gy + h then
      local cc = t.C.FG; gfx.set(cc[1], cc[2], cc[3], 1)
      gfx.line(x + pad + cpx, cy + 1, x + pad + cpx, cy + lh - 1)
    end
  end

  -- Scrollbar
  if content_h > h - pad * 2 then
    local sb_x    = x + w - sb_w + 1
    local vis_h   = h
    local ratio   = vis_h / (content_h + pad * 2)
    local thumb_h = math.max(12, math.floor(vis_h * ratio))
    local sb_y    = gy + (max_scroll > 0
      and math.floor(s.scroll_y / max_scroll * (vis_h - thumb_h)) or 0)
    local sc = t.C.SCROLLBAR
    gfx.set(sc[1], sc[2], sc[3], 0.25); gfx.rect(sb_x, gy, sb_w - 2, h, 1)
    gfx.set(sc[1], sc[2], sc[3], 0.8);  gfx.rect(sb_x + 1, sb_y + 1, sb_w - 4, thumb_h - 2, 1)
  end

  ctx.last_x, ctx.last_y, ctx.last_w, ctx.last_h = x, y, w, h
  ctx.x = t.PAD_X
  ctx.y = y + h + t.SPACING_Y
  return changed, new_text
end

-- ── SCROLL REGION ─────────────────────────────────────────────────

-- draw_fn draws content using gui.* calls.
-- opts: { hscroll = true } enables a horizontal scrollbar at the bottom.
function M.scroll_region(id, w, h, draw_fn, opts)
  opts = opts or {}
  local t = theme
  local x, y = ctx.x, ctx.y
  local sw = t.SCROLL_W + 2  -- vertical scrollbar strip width
  local sh = opts.hscroll and (t.SCROLL_W + 2) or 0  -- h-scrollbar strip height

  if not w or w == 0 then
    w = ctx.content_w - (ctx.x - t.PAD_X)
  elseif w < 0 then
    w = math.max(1, (ctx.content_w - (ctx.x - t.PAD_X)) + w)
  end
  local inner_w = w - sw     -- width of content area (excluding v-scrollbar strip)
  local inner_h = h - sh     -- height of content area (excluding h-scrollbar strip)

  if not ctx.state[id] then
    ctx.state[id] = { scroll_y = 0, scroll_x = 0, content_h = h, content_x = inner_w }
  end
  local s = ctx.state[id]
  if s.scroll_x == nil then s.scroll_x = 0; s.content_x = inner_w end  -- compat

  local in_region = ctx.mx >= x and ctx.mx < x+w and ctx.my >= y and ctx.my < y+h

  -- Vertical wheel scroll
  if ctx.mw ~= 0 and in_region then
    local delta = math.abs(ctx.mw) > 60
      and (ctx.mw > 0 and -20 or 20)
      or  -ctx.mw * 0.25
    s.scroll_y = s.scroll_y + delta
  end

  -- Horizontal scroll: prefer gfx.mouse_hwheel; fall back to Shift+wheel
  if opts.hscroll then
    local hdelta = 0
    if ctx.mhw ~= 0 and in_region then
      hdelta = math.abs(ctx.mhw) > 60
        and (ctx.mhw > 0 and -20 or 20)
        or  -ctx.mhw * 0.25
    elseif ctx.mw ~= 0 and in_region and (gfx.mouse_cap & 8) ~= 0 then
      -- Shift+vertical wheel as horizontal scroll
      hdelta = math.abs(ctx.mw) > 60
        and (ctx.mw > 0 and 20 or -20)
        or  ctx.mw * 0.25
    end
    if hdelta ~= 0 then s.scroll_x = s.scroll_x + hdelta end
  end

  if s.scroll_to_bottom then
    s.scroll_y = math.max(0, (s.content_h or 0) - inner_h)
    s.scroll_to_bottom = false
  end

  -- Vertical scrollbar thumb drag
  local max_vy = math.max(0, (s.content_h or inner_h) - inner_h)
  if max_vy > 0 then
    local ratio_d   = inner_h / (s.content_h or inner_h)
    local thumb_h_d = math.max(16, math.floor(inner_h * ratio_d))
    local sb_x_d    = x + inner_w + 2
    local thumb_y_d = y + math.floor(s.scroll_y / max_vy * (inner_h - thumb_h_d))
    local on_thumb  = ctx.mx >= sb_x_d and ctx.mx < sb_x_d + t.SCROLL_W
                   and ctx.my >= thumb_y_d and ctx.my < thumb_y_d + thumb_h_d
    if on_thumb and ctx.mb == 1 and ctx.mb_prev == 0 then
      s.drag_active = true; s.drag_start_my = ctx.my; s.drag_start_scroll = s.scroll_y
    end
    if s.drag_active then
      if ctx.mb == 1 then
        s.scroll_y = math.max(0, math.min(max_vy,
          s.drag_start_scroll + (ctx.my - s.drag_start_my) * max_vy / (inner_h - thumb_h_d)))
      else s.drag_active = false end
    end
  else s.drag_active = false end

  -- Horizontal scrollbar thumb drag
  local max_hx = opts.hscroll and math.max(0, (s.content_x or inner_w) - inner_w) or 0
  if max_hx > 0 then
    local ratio_hd    = inner_w / (s.content_x or inner_w)
    local thumb_w_d   = math.max(16, math.floor(inner_w * ratio_hd))
    local sb_y_d      = y + inner_h + 2
    local thumb_x_d   = x + math.floor(s.scroll_x / max_hx * (inner_w - thumb_w_d))
    local on_thumb_h  = ctx.mx >= thumb_x_d and ctx.mx < thumb_x_d + thumb_w_d
                     and ctx.my >= sb_y_d and ctx.my < sb_y_d + t.SCROLL_W
    if on_thumb_h and ctx.mb == 1 and ctx.mb_prev == 0 then
      s.hdrag_active = true; s.hdrag_start_mx = ctx.mx; s.hdrag_start_scroll = s.scroll_x
    end
    if s.hdrag_active then
      if ctx.mb == 1 then
        s.scroll_x = math.max(0, math.min(max_hx,
          s.hdrag_start_scroll + (ctx.mx - s.hdrag_start_mx) * max_hx / (inner_w - thumb_w_d)))
      else s.hdrag_active = false end
    end
  else s.hdrag_active = false end

  -- Clamp scrolls
  s.scroll_y = math.max(0, math.min(math.max(0, (s.content_h or 0) - inner_h), s.scroll_y))
  s.scroll_x = math.max(0, math.min(math.max(0, (s.content_x or 0) - inner_w), s.scroll_x))

  -- Draw background
  local bg = t.C.LOG_BG; gfx.set(bg[1], bg[2], bg[3], 1)
  gfx.rect(x, y, inner_w, inner_h, 1)

  -- Save outer layout context
  local sv = {
    x=ctx.x, y=ctx.y, cw=ctx.content_w,
    lx=ctx.last_x, ly=ctx.last_y, lw=ctx.last_w, lh=ctx.last_h,
    cy1=ctx.clip_y1, cy2=ctx.clip_y2, cyo=ctx.clip_y_off,
    cx1=ctx.clip_x1, cx2=ctx.clip_x2,
    cxm=ctx.content_x_max, cxo=ctx.content_x_origin,
  }

  -- Set up inner coordinate system.
  -- Vertical: logical Y=0 → screen Y = y - scroll_y.
  -- Horizontal: content starts at screen x = x+4 - scroll_x.
  local sx = math.floor(s.scroll_x)
  s.scroll_y = math.max(0, math.min(math.max(0, (s.content_h or 0) - inner_h), s.scroll_y))
  ctx.clip_y1    = y
  ctx.clip_y2    = y + inner_h
  ctx.clip_y_off = y - math.floor(s.scroll_y)
  ctx.x          = x + 4 - sx
  ctx.y          = 4
  ctx.content_w  = inner_w - 8

  if opts.hscroll then
    ctx.clip_x1          = x
    ctx.clip_x2          = x + inner_w
    ctx.content_x_max    = 0
    ctx.content_x_origin = x + 4 - sx
  end

  draw_fn()

  s.content_h = ctx.y + math.floor(s.scroll_y)
  if opts.hscroll and ctx.content_x_max ~= nil then
    s.content_x = math.max(inner_w, ctx.content_x_max + 8)  -- +8 for right padding
  end

  -- Restore outer context
  ctx.x             = sv.x;  ctx.y         = sv.y;   ctx.content_w = sv.cw
  ctx.last_x        = sv.lx; ctx.last_y    = sv.ly;  ctx.last_w    = sv.lw; ctx.last_h = sv.lh
  ctx.clip_y1       = sv.cy1; ctx.clip_y2  = sv.cy2; ctx.clip_y_off = sv.cyo
  ctx.clip_x1       = sv.cx1; ctx.clip_x2  = sv.cx2
  ctx.content_x_max = sv.cxm; ctx.content_x_origin = sv.cxo

  -- Clamp again after content size update
  s.scroll_y = math.max(0, math.min(math.max(0, (s.content_h or 0) - inner_h), s.scroll_y))
  s.scroll_x = math.max(0, math.min(math.max(0, (s.content_x or 0) - inner_w), s.scroll_x))

  -- Border around content area
  local bc = t.C.FRAME; gfx.set(bc[1], bc[2], bc[3], 1)
  gfx.line(x,        y,         x+inner_w, y)
  gfx.line(x,        y+inner_h, x+inner_w, y+inner_h)
  gfx.line(x,        y,         x,         y+inner_h)
  gfx.line(x+inner_w, y,        x+inner_w, y+inner_h)

  -- Vertical scrollbar
  if s.content_h and s.content_h > inner_h then
    local sb_x    = x + inner_w + 2
    local ratio   = inner_h / s.content_h
    local thumb_h = math.max(16, math.floor(inner_h * ratio))
    local max_sc  = s.content_h - inner_h
    local thumb_y = max_sc > 0
      and (y + math.floor(s.scroll_y / max_sc * (inner_h - thumb_h)))
      or  y
    local on_thumb_hover = ctx.mx >= sb_x and ctx.mx < sb_x + t.SCROLL_W
                        and ctx.my >= thumb_y and ctx.my < thumb_y + thumb_h
    local sc = t.C.SCROLLBAR; gfx.set(sc[1], sc[2], sc[3], 0.3)
    gfx.rect(sb_x, y, t.SCROLL_W, inner_h, 1)
    local alpha = (s.drag_active or on_thumb_hover) and 1.0 or 0.8
    gfx.set(sc[1], sc[2], sc[3], alpha)
    gfx.rect(sb_x+1, thumb_y, t.SCROLL_W-2, thumb_h, 1)
  end

  -- Horizontal scrollbar
  if opts.hscroll and s.content_x and s.content_x > inner_w then
    local sb_y    = y + inner_h + 2
    local ratio   = inner_w / s.content_x
    local thumb_w = math.max(16, math.floor(inner_w * ratio))
    local max_sc  = s.content_x - inner_w
    local thumb_x = max_sc > 0
      and (x + math.floor(s.scroll_x / max_sc * (inner_w - thumb_w)))
      or  x
    local on_thumb_hover = ctx.mx >= thumb_x and ctx.mx < thumb_x + thumb_w
                        and ctx.my >= sb_y and ctx.my < sb_y + t.SCROLL_W
    local sc = t.C.SCROLLBAR; gfx.set(sc[1], sc[2], sc[3], 0.3)
    gfx.rect(x, sb_y, inner_w, t.SCROLL_W, 1)
    local alpha = (s.hdrag_active or on_thumb_hover) and 1.0 or 0.8
    gfx.set(sc[1], sc[2], sc[3], alpha)
    gfx.rect(thumb_x+1, sb_y+1, thumb_w-2, t.SCROLL_W-2, 1)
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
