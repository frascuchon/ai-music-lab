-- Lua-side harness for test_midi_import.py.
--
-- Mocks just enough of the `reaper` API (tracks, MIDI items, notes, CCs) to run
-- midi_import.import_file() against an in-memory project that ALREADY contains
-- tracks with MIDI clips, then prints the resulting project so the Python test
-- can assert nothing pre-existing was touched.
--
-- usage: lua test_harness_midi_import.lua <file.mid> <existing_tracks> <insert_at> <always_folder 0|1> <position_s>
-- output lines:
--   R|first|count|instruments|next_insert_at     (or ERR|message)
--   T|index|name|folderdepth|n_items|n_notes|n_ccs|item_pos|item_len|first_note_ppq|first_pitch

local SCRIPT_DIR = (arg[0]):match("^(.*/)") or "./"
package.path = SCRIPT_DIR .. "?.lua;" .. package.path

local path, n_existing, insert_at, always_folder, position =
  arg[1], tonumber(arg[2]), tonumber(arg[3]), arg[4] == "1", tonumber(arg[5])

local tracks = {}
for k = 1, n_existing do
  -- a pre-existing track holding one MIDI clip with a single marker note
  table.insert(tracks, {
    name = "exist " .. k, depth = 0,
    items = { { pos = 0, len = 10, take = { notes = { { ppq = 0, pitch = 1 } }, ccs = {} } } },
  })
end

local PPQ_PER_SEC = 1920  -- 120 BPM, 960 PPQ per quarter note

reaper = {
  CountTracks = function() return #tracks end,
  GetTrack = function(_, i) return tracks[i + 1] end,
  InsertTrackAtIndex = function(i) table.insert(tracks, i + 1, { name = "", depth = 0, items = {} }) end,
  GetSetMediaTrackInfo_String = function(tr, key, val, set)
    if set and key == "P_NAME" then tr.name = val end
    return true, tr.name
  end,
  SetMediaTrackInfo_Value = function(tr, key, val)
    if key == "I_FOLDERDEPTH" then tr.depth = val end
  end,
  CreateNewMIDIItemInProj = function(tr, s, e)
    local it = { pos = s, len = e - s, take = { notes = {}, ccs = {} } }
    table.insert(tr.items, it)
    return it
  end,
  GetActiveTake = function(it) return it.take end,
  MIDI_GetPPQPosFromProjTime = function(_, t) return t * PPQ_PER_SEC end,
  MIDI_InsertNote = function(take, _, _, sp, ep, chan, pitch, vel)
    table.insert(take.notes, { ppq = sp, endppq = ep, chan = chan, pitch = pitch, vel = vel })
    return true
  end,
  MIDI_InsertCC = function(take, _, _, ppq, msg, chan, m2)
    table.insert(take.ccs, { ppq = ppq, msg = msg, chan = chan, value = m2 })
    return true
  end,
  MIDI_Sort = function() end,
}

local mi = require("midi_import")
local res, err = mi.import_file(path, {
  insert_at = insert_at, position = position, name = "NEW", always_folder = always_folder,
})
if not res then
  print("ERR|" .. tostring(err))
else
  print(string.format("R|%d|%d|%d|%d", res.first, res.count, res.instruments, res.next_insert_at))
end

for i, tr in ipairs(tracks) do
  local it = tr.items[1]
  local n = it and #it.take.notes or 0
  local c = it and #it.take.ccs or 0
  local first = it and it.take.notes[1]
  print(string.format("T|%d|%s|%d|%d|%d|%d|%.4f|%.4f|%s|%s",
    i - 1, tr.name, tr.depth, #tr.items, n, c,
    it and it.pos or -1, it and it.len or -1,
    first and string.format("%.3f", first.ppq) or "-", first and first.pitch or "-"))
end
