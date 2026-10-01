-- Lua-side harness for test_track_placement.py.
--
-- track_placement.lua has no reaper dependency, so this harness just
-- requires it directly and runs plain assertions — no mocking needed.
-- Each case prints "PASS <name>" or "FAIL <name>: <msg>"; the process exits
-- non-zero if any case fails, so test_track_placement.py can just check the
-- return code (and print stdout on failure for detail).

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path

local tp = require("track_placement")

local failures = 0

local function case_(name, fn)
  local ok, err = pcall(fn)
  if ok then
    print("PASS " .. name)
  else
    failures = failures + 1
    print("FAIL " .. name .. ": " .. tostring(err))
  end
end

-- ── insert_index ─────────────────────────────────────────────────

case_("insert_index: no reference (nil) appends at end", function()
  assert(tp.insert_index(nil, 10) == 10)
end)

case_("insert_index: no reference (0) appends at end", function()
  assert(tp.insert_index(0, 10) == 10)
end)

case_("insert_index: negative reference appends at end", function()
  assert(tp.insert_index(-1, 10) == 10)
end)

case_("insert_index: reference track 1 of 10 inserts right below it", function()
  -- Track #1 (0-based idx 0) -> insert at 0-based idx 1.
  assert(tp.insert_index(1, 10) == 1)
end)

case_("insert_index: reference track in the middle inserts right below it", function()
  -- Track #4 (0-based idx 3) of 10 -> insert at 0-based idx 4, NOT at the
  -- end (10) — this is the bug being fixed: new tracks used to always land
  -- at the bottom of the whole project regardless of the source track.
  assert(tp.insert_index(4, 10) == 4)
end)

case_("insert_index: reference is the last track inserts at the end anyway", function()
  assert(tp.insert_index(10, 10) == 10)
end)

case_("insert_index: reference beyond track_count clamps to track_count", function()
  -- Defensive: a stale/out-of-range reference must never produce an index
  -- reaper.InsertTrackAtIndex can't handle.
  assert(tp.insert_index(99, 10) == 10)
end)

case_("insert_index: fractional reference is floored", function()
  assert(tp.insert_index(3.7, 10) == 3)
end)

case_("insert_index: empty project with no reference is index 0", function()
  assert(tp.insert_index(nil, 0) == 0)
end)

-- ── max_track_number ─────────────────────────────────────────────

case_("max_track_number: picks the highest of several tracks", function()
  assert(tp.max_track_number({2, 5, 3}) == 5)
end)

case_("max_track_number: single value", function()
  assert(tp.max_track_number({7}) == 7)
end)

case_("max_track_number: ignores 0/negative entries", function()
  assert(tp.max_track_number({0, -1, 4}) == 4)
end)

case_("max_track_number: empty list returns nil", function()
  assert(tp.max_track_number({}) == nil)
end)

case_("max_track_number: all-invalid list returns nil", function()
  assert(tp.max_track_number({0, -3}) == nil)
end)

-- ── Composition: seed-based generation lands right below all seeds ──

case_("composition: accompaniment folder lands below the last of melody+seeds", function()
  -- Melody on track 2, seeds on tracks 5 and 3, in a 10-track project.
  local ref = tp.max_track_number({2, 5, 3})
  assert(ref == 5)
  assert(tp.insert_index(ref, 10) == 5)
end)

-- ── plan_candidate_import ────────────────────────────────────────
-- MidiGenerator/panel.lua's _import_one: wraps the `delta` instrument
-- tracks InsertMedia just created (at tcnt_before) in a folder, which
-- involves inserting ONE more track (the folder header) at tcnt_before —
-- shifting the instrument tracks down by one. Regression target for the
-- "Amadeus candidates mix together / overlap existing tracks" report:
-- these are the index computations that must stay internally consistent
-- across candidates for multi-candidate (multi-"version") imports to stack
-- cleanly instead of colliding.

case_("plan_candidate_import: single instrument track", function()
  local plan = tp.plan_candidate_import(10, 1)
  assert(plan.folder_index == 10)
  assert(plan.instrument_start == 11)
  assert(plan.instrument_end == 11)
  assert(plan.next_insert_at == 12)
end)

case_("plan_candidate_import: multi-instrument MIDI (e.g. Amadeus multi-track output)", function()
  -- 4 instrument tracks (piano/bass/drums/strings) imported at index 10.
  local plan = tp.plan_candidate_import(10, 4)
  assert(plan.folder_index == 10)
  assert(plan.instrument_start == 11)
  assert(plan.instrument_end == 14)
  assert(plan.next_insert_at == 15)
end)

case_("plan_candidate_import: delta <= 0 (InsertMedia added no tracks) returns nil", function()
  assert(tp.plan_candidate_import(10, 0) == nil)
  assert(tp.plan_candidate_import(10, -1) == nil)
  assert(tp.plan_candidate_import(10, nil) == nil)
end)

case_("plan_candidate_import: tcnt_before == 0 (first candidate in an empty project)", function()
  local plan = tp.plan_candidate_import(0, 2)
  assert(plan.folder_index == 0)
  assert(plan.instrument_start == 1)
  assert(plan.instrument_end == 2)
  assert(plan.next_insert_at == 3)
end)

case_("plan_candidate_import: chained candidates stack contiguously without overlap", function()
  -- Candidate 1: 3 instrument tracks at index 5 (folder + 3 instruments = 4 tracks).
  local plan1 = tp.plan_candidate_import(5, 3)
  assert(plan1.next_insert_at == 9)
  -- Candidate 2 starts exactly where candidate 1's folder ended — no gap,
  -- no overlap with candidate 1's tracks [5..8].
  local plan2 = tp.plan_candidate_import(plan1.next_insert_at, 2)
  assert(plan2.folder_index == 9)
  assert(plan2.instrument_start == 10)
  assert(plan2.instrument_end == 11)
  -- The two candidates' track ranges (folder + instruments) must not overlap.
  assert(plan2.folder_index > plan1.instrument_end)
end)

if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
os.exit(0)
