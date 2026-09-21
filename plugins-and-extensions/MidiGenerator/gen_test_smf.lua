-- Lua-side harness for test_smf_writer.py.
--
-- Mocks the `reaper` global (Master_GetTempo, MIDI_GetProjTimeFromPPQPos,
-- MIDI_CountEvts, MIDI_GetNote) with a synthetic MIDI take, runs
-- smf_writer.lua's write_midi_from_take()/write_combined_midi() exactly as
-- panel.lua does when exporting an in-project MIDI take (ChatMusician
-- "Harmonize seed" and Anticipatory melody/accompaniment seed), and writes
-- the resulting .mid to disk so the Python test can validate it with
-- mido/pretty_midi.
--
-- Usage: lua gen_test_smf.lua <case> <bpm> <out.mid> [item_pos]
--   case: "simple"   → single take, 4 notes at known absolute times
--         "combined" → melody + one seed take on separate channels
--         "empty"    → take with zero MIDI notes (write must fail cleanly)

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"

local case     = arg[1] or "simple"
local bpm      = tonumber(arg[2]) or 120.0
local out_path = arg[3] or "out.mid"
local item_pos = tonumber(arg[4]) or 0.0  -- project position of the take's tick 0

-- Notes as absolute PROJECT time (seconds): {start_s, end_s, pitch, vel, chan}
-- These are the "ground truth" the exported MIDI must reproduce once
-- normalized to item-relative time (start_s - item_pos).
local NOTES = {
  { 0.5, 1.0, 60, 90, 0 },
  { 1.5, 2.25, 64, 100, 0 },
  { 3.0, 3.5, 67, 80, 0 },
}
local SEED_NOTES = {
  { 0.5, 1.5, 48, 70, 0 },
  { 2.0, 3.0, 52, 70, 0 },
}

local function make_take(notes)
  -- A "take" here is just an opaque token; MIDI_GetProjTimeFromPPQPos and
  -- MIDI_GetNote below dispatch on it via a side table, exactly like real
  -- REAPER dispatches on a MediaItem_Take* pointer.
  return { notes = notes }
end

-- PPQ position <-> seconds mapping used by the mock: PPQ is just
-- (seconds * bpm/60 * PPQ_takes), inverted here so MIDI_GetNote can report
-- ppq positions that MIDI_GetProjTimeFromPPQPos converts back losslessly —
-- mirroring how real REAPER takes store notes in PPQ and convert via the
-- project tempo map.
local PPQ_TAKE = 960  -- take's own PPQ resolution (independent of writer's 480)

local function seconds_to_take_ppq(t_sec, bpm_)
  return math.floor(t_sec * (bpm_ / 60) * PPQ_TAKE + 0.5)
end

_G.reaper = {
  Master_GetTempo = function() return bpm end,

  MIDI_GetProjTimeFromPPQPos = function(take, ppq)
    -- Inverse of seconds_to_take_ppq, offset by item_pos (project position).
    return item_pos + (ppq / PPQ_TAKE) * (60 / bpm)
  end,

  MIDI_CountEvts = function(take)
    return true, #take.notes, 0, 0
  end,

  MIDI_GetNote = function(take, i)
    local n = take.notes[i + 1]
    -- n[1]/n[2] are item-relative seconds (take-internal PPQ has no notion
    -- of the item's project position — that only enters via
    -- MIDI_GetProjTimeFromPPQPos's item_pos offset above).
    local start_ppq = seconds_to_take_ppq(n[1], bpm)
    local end_ppq   = seconds_to_take_ppq(n[2], bpm)
    -- retval, selected, muted, startppqpos, endppqpos, chan, pitch, vel
    return true, false, false, start_ppq, end_ppq, n[5], n[3], n[4]
  end,
}

local smf = dofile(SCRIPT_DIR .. "smf_writer.lua")

if case == "simple" then
  local take = make_take(NOTES)
  local ok, err = smf.write_midi_from_take(take, out_path)
  if not ok then
    io.stderr:write("write_midi_from_take failed: " .. tostring(err) .. "\n")
    os.exit(1)
  end
elseif case == "combined" then
  local melody = make_take(NOTES)
  local seed   = make_take(SEED_NOTES)
  local ok, err = smf.write_combined_midi(melody, { { take = seed, label = "seed1" } }, out_path)
  if not ok then
    io.stderr:write("write_combined_midi failed: " .. tostring(err) .. "\n")
    os.exit(1)
  end
elseif case == "empty" then
  -- A zero-note take (e.g. wrong item/track selected) must be REJECTED,
  -- not silently written as a valid-but-content-free .mid — this is the
  -- opposite of the other cases: failure is the expected, successful
  -- outcome here, so it's reported on stdout (not stderr+exit 1).
  local take = make_take({})
  local ok, err = smf.write_midi_from_take(take, out_path)
  if ok then
    io.stderr:write("write_midi_from_take unexpectedly succeeded on an empty take\n")
    os.exit(1)
  end
  print("EMPTY_TAKE_ERROR: " .. tostring(err))
else
  io.stderr:write("Unknown case: " .. case .. "\n")
  os.exit(1)
end
