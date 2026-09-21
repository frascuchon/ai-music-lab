-- @description AI Music Lab - Text2Audio (deprecated, merged into AI Music Lab)
-- @version 2.0
-- @author AI Music Lab
-- @about DEPRECATED: Text2Audio is now the "Text2Audio" tab inside the
--        unified "AI Music Lab" plugin (plugins-and-extensions/ai-music-lab.lua).
--        This stub keeps any existing keyboard shortcut / toolbar button
--        working by opening the unified plugin directly on that tab.

local _info = debug.getinfo(1, "S")
local DIR = _info.source:match("@?(.*[/\\])") or ""
_G.AI_MUSIC_LAB_INITIAL_TAB = "Text2Audio"
dofile(DIR .. "../ai-music-lab.lua")
