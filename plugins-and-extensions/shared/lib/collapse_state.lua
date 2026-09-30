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

return M
