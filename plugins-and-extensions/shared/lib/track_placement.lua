-- lib/track_placement.lua  Pure decision logic for where newly generated
-- track(s)/folder(s) should be inserted in the project.
--
-- Every panel used to always append new tracks at reaper.CountTracks(0)
-- (the very end of the project's track list), regardless of where the
-- source clip/track the generation was based on actually sat. That put
-- results far away from their source, especially in projects with many
-- tracks. New tracks generated FROM a source track/clip (stem separation,
-- audio→MIDI transcription, audio generation from a source clip, MIDI
-- accompaniment/continuation from seed tracks) must land immediately below
-- that source instead.
--
-- No gfx/reaper dependency — testable with a plain lua interpreter (see
-- shared/lib/test_harness_track_placement.lua).

local M = {}

-- Computes the 0-based track index at which newly generated track(s) should
-- be inserted via reaper.InsertTrackAtIndex, so they land immediately below
-- a reference track instead of always at the end of the project.
--
-- ref_track_number : the reference (source/seed) track's 1-based REAPER
--                     IP_TRACKNUMBER, or nil/<=0 when there is no reference
--                     (e.g. generation from a text prompt with nothing
--                     selected in REAPER) — falls back to appending at the
--                     end of the project.
-- track_count       : reaper.CountTracks(0) at the time of insertion.
--
-- IP_TRACKNUMBER is 1-based, so a reference track at IP_TRACKNUMBER N sits
-- at 0-based index N-1; inserting AT index N therefore places the new
-- track immediately after it.
function M.insert_index(ref_track_number, track_count)
  track_count = track_count or 0
  if not ref_track_number or ref_track_number <= 0 then
    return track_count
  end
  local idx = math.floor(ref_track_number)
  if idx > track_count then return track_count end
  return idx
end

-- Reference track number for a generation based on SEVERAL source tracks
-- (e.g. MidiGenerator's AMT accompaniment: one melody track + N seed
-- tracks). The new folder must land below ALL of them, so the reference is
-- the highest (bottom-most) track number among them. Returns nil if no
-- valid (>0) number is found, so callers fall back to M.insert_index's
-- "append at end" behavior.
function M.max_track_number(numbers)
  local best = nil
  for _, n in ipairs(numbers) do
    if n and n > 0 and (not best or n > best) then best = n end
  end
  return best
end

return M
