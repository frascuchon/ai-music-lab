-- lib/collapse_state.lua  Pure decision logic for the global "collapse window
-- to header-only" feature. No gfx/reaper dependency — testable with a plain
-- lua interpreter (see shared/lib/test_harness_collapse_state.lua).

local M = {}

-- Flips the collapsed flag.
function M.toggle(collapsed)
  return not collapsed
end

-- Target window height for the current state.
function M.target_height(collapsed, expanded_h, header_h)
  if collapsed then return header_h end
  return expanded_h
end

-- Tracks the "restore to" height while expanded, so a manual resize by the
-- user is remembered next time the window is collapsed. While collapsed,
-- the previously remembered height is left untouched.
function M.track_expanded_h(collapsed, current_h, expanded_h)
  if collapsed then return expanded_h end
  return current_h
end

-- On platforms where the window's y-position is reported from its BOTTOM
-- edge in a bottom-up coordinate system (macOS, via gfx.dock's ypos), simply
-- keeping y unchanged while changing height anchors the bottom edge instead
-- of the top — the window's top edge visibly moves when resized. Returns
-- the y that keeps the TOP edge anchored across a height change from
-- old_h to new_h.
function M.anchor_top_y(y, old_h, new_h)
  return y + (old_h - new_h)
end

-- On HiDPI/Retina displays (gfx.ext_retina = 1) gfx.w / gfx.h are reported in
-- physical pixels (scale x logical), while gfx.init expects logical units.
-- Feeding the queried size straight back into gfx.init multiplies it by
-- `scale` on every recreate (the window width "shoots up" on each toggle).
-- Converts a size read from gfx.w / gfx.h back to logical units.
function M.to_logical(px, scale)
  if not scale or scale < 1 then scale = 1 end
  return math.floor(px / scale + 0.5)
end

-- Recreates the gfx window at target_h (physical pixels, like gfx.h),
-- preserving width and dock/position. `g` is the gfx table (injected so the
-- real code path is testable with a fake). Returns the logical w, h used.
function M.recreate_window(g, title, target_h, scale, is_mac, on_ready)
  local dock, wx, wy = g.dock(-1, 0, 0, 0, 0)
  local w = M.to_logical(g.w, scale)
  local old_h = M.to_logical(g.h, scale)
  local h = M.to_logical(target_h, scale)
  if is_mac then wy = M.anchor_top_y(wy, old_h, h) end
  g.quit()
  g.init(title, w, h, dock, wx, wy)
  g.ext_retina = 1
  if on_ready then on_ready() end
  return w, h
end

return M
