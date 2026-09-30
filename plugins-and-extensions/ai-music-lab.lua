-- @description AI Music Lab - unified plugin (Audio2Midi, MidiGenerator, Text2Audio, StemsSeparator, Setup)
-- @version 1.0
-- @author AI Music Lab
-- @about Single REAPER window with a tab per tool: Audio2Midi, MidiGenerator,
--        Text2Audio, StemsSeparator and Setup. Replaces the 4 previous
--        standalone plugin actions plus the standalone Setup wizard.
--        Native gfx UI: no external REAPER extension dependencies.

-- ── PATHS + LIB ──────────────────────────────────────────────────
local _info      = debug.getinfo(1, "S")
local SCRIPT_DIR = _info.source:match("@?(.*[/\\])") or ""
local SHARED_DIR = SCRIPT_DIR .. "shared/"
package.path = SHARED_DIR .. "lib/?.lua;" .. package.path

local common  = require("common")
local theme   = require("theme")
local gui     = require("gui")
local widgets = require("widgets_extra")
local collapse_state = require("collapse_state")

-- ── LOAD PANELS ──────────────────────────────────────────────────
-- Each panel.lua lives next to its own Python backend and computes its
-- own SCRIPT_DIR via debug.getinfo, so a plain dofile() is enough — no
-- extra package.path juggling needed here.
local audio2midi   = dofile(SCRIPT_DIR .. "Audio2Midi/panel.lua")
local midigenerator= dofile(SCRIPT_DIR .. "MidiGenerator/panel.lua")
local text2audio   = dofile(SCRIPT_DIR .. "Text2Audio/panel.lua")
local stemssep     = dofile(SCRIPT_DIR .. "StemsSeparator/panel.lua")
local setup_panel  = dofile(SHARED_DIR .. "panel_setup.lua")

local PANELS = { audio2midi, midigenerator, text2audio, stemssep, setup_panel }
local TAB_LABELS = {}
for i, p in ipairs(PANELS) do TAB_LABELS[i] = p.title end
local SETUP_TAB_IDX = #PANELS

-- ── STATE ────────────────────────────────────────────────────────
-- A deprecated per-plugin stub (see Audio2Midi/ai-music-lab-Audio2Midi.lua
-- etc.) can set this global before dofile-ing this script to land on the
-- right tab instead of always opening on the first one.
local initial_tab = 1
if _G.AI_MUSIC_LAB_INITIAL_TAB then
  for i, label in ipairs(TAB_LABELS) do
    if label == _G.AI_MUSIC_LAB_INITIAL_TAB then initial_tab = i; break end
  end
  _G.AI_MUSIC_LAB_INITIAL_TAB = nil
end
local S = { tab = initial_tab }

-- Global window collapse (all tabs): reduces the window to header-only.
-- State is persisted across REAPER sessions via ExtState so the plugin
-- reopens the way the user left it.
local EXTSTATE_NS = "AI_MUSIC_LAB"
S.collapsed  = reaper.GetExtState(EXTSTATE_NS, "collapsed") == "1"
S.expanded_h = tonumber(reaper.GetExtState(EXTSTATE_NS, "expanded_h")) or 780

for _, p in ipairs(PANELS) do
  if p.init then p.init() end
end

-- ── GFX INIT ─────────────────────────────────────────────────────
if gfx.w > 0 then gfx.quit() end
local LOGICAL_W = 620
local WINDOW_TITLE = "AI Music Lab"

-- Height of the header-only row (title + status + collapse button), based
-- on current (possibly scale-adjusted) theme metrics.
local function header_height()
  return theme.PAD_Y * 2 + theme.BTN_H
end

gfx.init(WINDOW_TITLE, LOGICAL_W, S.collapsed and header_height() or S.expanded_h)
gfx.ext_retina = 1
theme.init_fonts()

-- macOS reports gfx window position (via gfx.dock's ypos) as the BOTTOM-left
-- corner in AppKit's bottom-up screen coordinates, unlike Windows/Linux
-- which use the top-left corner. Keeping ypos unchanged while only shrinking
-- height therefore anchors the BOTTOM edge on macOS — the window visibly
-- shrinks by its top edge dropping down, forcing the user to move the mouse
-- down to find the collapsed header. We want the opposite: the header
-- (top edge) stays put and the body folds up underneath it, on every OS.
local _os_id = reaper.GetOS():lower()
local IS_MAC = _os_id:find("osx") ~= nil or _os_id:find("mac") ~= nil

-- Actually shrinks/grows the OS window (not just what we draw inside it) —
-- this is what lets the user work in the REAPER project underneath without
-- the plugin window in the way. gfx has no in-place "set height" call, so
-- we close and recreate the window at the new size, preserving its current
-- width and dock state (queried via gfx.dock(-1, ...) before tearing it
-- down), and adjusting ypos on macOS so the top edge — not the bottom —
-- stays anchored. This mirrors the quit+init pattern this script already
-- uses once at startup (above) to handle a relaunch.
local function resize_window(h)
  local dock, wx, wy = gfx.dock(-1, 0, 0, 0, 0)
  local w, old_h = gfx.w, gfx.h
  if IS_MAC then
    wy = collapse_state.anchor_top_y(wy, old_h, h)
  end
  gfx.quit()
  gfx.init(WINDOW_TITLE, w, h, dock, wx, wy)
  gfx.ext_retina = 1
  theme.init_fonts(theme.SCALE)
end

-- ── MAIN LOOP ────────────────────────────────────────────────────
local _scale_init = false

local function loop()
  if not _scale_init then
    _scale_init = true
    local s = math.floor(gfx.w / LOGICAL_W + 0.5)
    if s > 1 then
      theme.apply_scale(s)
      theme.init_fonts(s)
    end
  end

  gui.frame_begin()
  if gui.ctx.should_close then gfx.quit(); return end

  local g = gui
  local t = theme

  -- Setup banner (shown on every tab except Setup itself) — hidden while
  -- collapsed, since the collapsed window shows only the header row.
  if not S.collapsed and S.tab ~= SETUP_TAB_IDX then
    local missing = setup_panel.get_missing()
    if #missing > 0 then
      g.text_wrapped("⚠  Incomplete setup: " .. table.concat(missing, " · "))
      g.text_disabled("Open the Setup tab to configure.")
      g.spacing()
    end
  end

  -- Header (always drawn, collapsed or not — this is the global,
  -- tab-independent row that hosts the collapse/expand toggle)
  g.push_font(t.F.H1)
  g.text("AI Music Lab")
  g.pop_font()
  g.same_line(10)
  g.text_colored("● REAPER OK", "GREEN")

  -- Collapse/expand toggle, right-aligned on the header row
  local toggle_w = t.BTN_H
  gui.ctx.x = t.PAD_X + gui.ctx.content_w - toggle_w
  gui.ctx.y = gui.ctx.last_y
  if g.button(S.collapsed and "▸" or "▾", toggle_w, t.BTN_H) then
    S.collapsed = collapse_state.toggle(S.collapsed)
    reaper.SetExtState(EXTSTATE_NS, "collapsed", S.collapsed and "1" or "0", true)
    reaper.SetExtState(EXTSTATE_NS, "expanded_h", tostring(S.expanded_h), true)
    resize_window(collapse_state.target_height(S.collapsed, S.expanded_h, header_height()))
    -- The window was just torn down and recreated: stop drawing into this
    -- frame (its gfx context is gone) and let the next frame draw cleanly
    -- against the freshly (re)created window. Update mb_prev manually
    -- (normally frame_end's job) so next frame doesn't see a stale
    -- mb_prev=0 and misread a still-held mouse button as a brand-new click.
    gui.ctx.mb_prev = gui.ctx.mb
    for _, p in ipairs(PANELS) do
      if p.poll then p.poll() end
    end
    reaper.defer(loop)
    return
  end

  if not S.collapsed then
    g.separator()
    g.spacing()

    -- Tab bar
    S.tab = widgets.tab_bar("##maintabs", S.tab, TAB_LABELS)
    g.spacing()

    PANELS[S.tab].draw()

    -- Remember the current window height so it can be restored on expand,
    -- even if the user resized the window manually while expanded.
    S.expanded_h = collapse_state.track_expanded_h(false, gfx.h, S.expanded_h)
  end

  gui.frame_end()

  -- Poll every panel every frame (not just the active one) so background
  -- jobs keep progressing while the user is on a different tab.
  for _, p in ipairs(PANELS) do
    if p.poll then p.poll() end
  end

  reaper.defer(loop)
end

-- ── STARTUP ──────────────────────────────────────────────────────
reaper.defer(loop)
