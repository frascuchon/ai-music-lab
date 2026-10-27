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

if failures > 0 then
  print(failures .. " failure(s)")
  os.exit(1)
end
os.exit(0)
