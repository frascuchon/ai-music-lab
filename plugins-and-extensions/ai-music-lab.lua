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

for _, p in ipairs(PANELS) do
  if p.init then p.init() end
end

-- ── GFX INIT ─────────────────────────────────────────────────────
if gfx.w > 0 then gfx.quit() end
local LOGICAL_W = 620
gfx.init("AI Music Lab", LOGICAL_W, 780)
gfx.ext_retina = 1
theme.init_fonts()

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

  -- Setup banner (shown on every tab except Setup itself)
  if S.tab ~= SETUP_TAB_IDX then
    local missing = setup_panel.get_missing()
    if #missing > 0 then
      g.text_wrapped("⚠  Incomplete setup: " .. table.concat(missing, " · "))
      g.text_disabled("Open the Setup tab to configure.")
      g.spacing()
    end
  end

  -- Header
  g.push_font(t.F.H1)
  g.text("AI Music Lab")
  g.pop_font()
  g.same_line(10)
  g.text_colored("● REAPER OK", "GREEN")
  g.separator()
  g.spacing()

  -- Tab bar
  S.tab = widgets.tab_bar("##maintabs", S.tab, TAB_LABELS)
  g.spacing()

  PANELS[S.tab].draw()

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
