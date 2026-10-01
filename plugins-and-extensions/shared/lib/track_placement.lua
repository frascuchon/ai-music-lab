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

-- Plans the folder-wrapping layout for one imported MIDI candidate, given:
--   tcnt_before : 0-based index where the candidate's instrument track(s)
--                 were inserted (reused from the placeholder/anchor track).
--   delta       : number of instrument tracks InsertMedia actually created
--                 (reaper.CountTracks(0) - tcnt_before right after import,
--                 before the folder header is inserted).
--
-- MidiGenerator/panel.lua's _import_one does, in order: insert `delta`
-- instrument tracks at tcnt_before via InsertMedia, THEN insert one more
-- track at tcnt_before to serve as the folder header — which pushes the
-- instrument tracks down by one. This function makes that two-step index
-- arithmetic a pure, testable unit instead of inline reaper.* calls, so a
-- regression (tracks landing one off, folders overlapping the next
-- candidate, etc.) shows up as a failing assertion here instead of only
-- being visible by eye in a live REAPER project.
--
-- Returns:
--   folder_index      : 0-based index to insert the folder header track at.
--   instrument_start   : 0-based index of the first instrument track
--                        AFTER the folder header shifts them down.
--   instrument_end     : 0-based index of the last instrument track (the
--                        one that must get I_FOLDERDEPTH = -1 to close the
--                        folder).
--   next_insert_at      : 0-based index the NEXT candidate's import should
--                        start at (right below this whole folder).
-- Returns nil if delta <= 0 (InsertMedia added no tracks — nothing to wrap).
function M.plan_candidate_import(tcnt_before, delta)
  if not delta or delta <= 0 then return nil end
  return {
    folder_index    = tcnt_before,
    instrument_start = tcnt_before + 1,
    instrument_end   = tcnt_before + delta,
    next_insert_at   = tcnt_before + 1 + delta,
  }
end

return M
