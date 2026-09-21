-- MidiGenerator/smf_writer.lua — Standard MIDI File (SMF) writer used to
-- export in-project REAPER MIDI takes (ChatMusician seed harmonization,
-- Anticipatory melody+accompaniment seed) to disk as .mid files.
--
-- Extracted from panel.lua into its own module so it can be exercised by
-- test_smf_writer.lua without loading the whole GUI panel (gfx/theme/gui
-- dependencies). Depends only on the global `reaper` table (mockable).

local M = {}

local function _vlq(n)
  if n < 0x80 then return string.char(n) end
  local bytes = {}
  while n > 0 do
    table.insert(bytes, 1, n % 128)
    n = math.floor(n / 128)
  end
  for i = 1, #bytes - 1 do bytes[i] = bytes[i] + 128 end
  return string.char(table.unpack(bytes))
end

local function _u32be(v)
  v = math.floor(v) % (2^32)
  return string.char(
    math.floor(v/16777216)%256,
    math.floor(v/65536)%256,
    math.floor(v/256)%256,
    v%256)
end

local function _u16be(v)
  v = math.floor(v) % 65536
  return string.char(math.floor(v/256)%256, v%256)
end

-- Writes a minimal SMF Type-1 from a REAPER MIDI take.
-- Supports note on/off. Used when the source is not a .mid file on disk.
function M.write_midi_from_take(take, filepath)
  local PPQ = 480
  local bpm = reaper.Master_GetTempo()
  local tempo_uspb = math.floor(60000000 / bpm)

  -- Project time at tick 0 of the take (item position + source offset).
  -- Subtracting it normalizes all events to t=0 relative to the item start,
  -- which is what AMT libraries expect (anticipation uses clip(events, 0, N)).
  local t_item_start = reaper.MIDI_GetProjTimeFromPPQPos(take, 0)

  -- Collect notes: {ppq_start, ppq_end, chan, pitch, vel}
  -- MIDI_CountEvts returns (retval, notes, ccs, sysex)
  local _, noteCount = reaper.MIDI_CountEvts(take)
  noteCount = noteCount or 0
  local events = {}
  for i = 0, noteCount - 1 do
    local _, _, _, startppq, endppq, chan, pitch, vel =
      reaper.MIDI_GetNote(take, i)
    -- Convert to relative time (s from item start) then to PPQ-480
    local t_start = reaper.MIDI_GetProjTimeFromPPQPos(take, startppq) - t_item_start
    local t_end   = reaper.MIDI_GetProjTimeFromPPQPos(take, endppq)   - t_item_start
    local p_start = math.floor(t_start * (bpm/60) * PPQ + 0.5)
    local p_end   = math.floor(t_end   * (bpm/60) * PPQ + 0.5)
    if p_end > p_start then
      table.insert(events, { tick=p_start, status=0x90+(chan%16), d1=pitch, d2=vel })
      table.insert(events, { tick=p_end,   status=0x80+(chan%16), d1=pitch, d2=0   })
    end
  end
  table.sort(events, function(a,b) return a.tick < b.tick end)

  if #events == 0 then
    return false, "Selected item has no MIDI notes"
  end

  -- Build note track
  local trk = ""
  local prev_tick = 0
  for _, ev in ipairs(events) do
    local delta = math.max(0, ev.tick - prev_tick)
    trk = trk .. _vlq(delta) .. string.char(ev.status, ev.d1, ev.d2)
    prev_tick = ev.tick
  end
  trk = trk .. "\000\255\047\000"  -- delta=0, meta End of Track

  -- Build tempo track
  local tmp = "\000\255\081\003"  -- delta=0, meta Tempo, len=3
    .. string.char(
        math.floor(tempo_uspb/65536)%256,
        math.floor(tempo_uspb/256)%256,
        tempo_uspb%256)
    .. "\000\255\047\000"

  -- Assemble SMF
  local f = io.open(filepath, "wb")
  if not f then return false, "Could not create " .. filepath end
  -- MThd
  f:write("MThd" .. _u32be(6) .. _u16be(1) .. _u16be(2) .. _u16be(PPQ))
  -- Tempo track
  f:write("MTrk" .. _u32be(#tmp) .. tmp)
  -- Notes track
  f:write("MTrk" .. _u32be(#trk) .. trk)
  f:close()
  return true, nil
end

-- Combines multiple takes into a single MIDI (for AMT):
-- melody → channel 0, each seed take → channel 1,2,3...
function M.write_combined_midi(melody_take, seed_takes, filepath)
  local PPQ = 480
  local bpm = reaper.Master_GetTempo()
  local tempo_uspb = math.floor(60000000 / bpm)
  local events = {}

  local function collect_take(take, force_chan)
    local t0 = reaper.MIDI_GetProjTimeFromPPQPos(take, 0)
    local _, nc = reaper.MIDI_CountEvts(take)
    for i = 0, (nc or 0) - 1 do
      local _, _, _, sp, ep, _, pitch, vel = reaper.MIDI_GetNote(take, i)
      local ts = reaper.MIDI_GetProjTimeFromPPQPos(take, sp) - t0
      local te = reaper.MIDI_GetProjTimeFromPPQPos(take, ep) - t0
      local ps = math.floor(ts * (bpm/60) * PPQ + 0.5)
      local pe = math.floor(te * (bpm/60) * PPQ + 0.5)
      if pe > ps then
        table.insert(events, { tick=ps, status=0x90+force_chan, d1=pitch, d2=vel })
        table.insert(events, { tick=pe, status=0x80+force_chan, d1=pitch, d2=0   })
      end
    end
  end

  collect_take(melody_take, 0)
  for i, entry in ipairs(seed_takes) do
    collect_take(entry.take, math.min(i, 15))
  end
  table.sort(events, function(a,b) return a.tick < b.tick end)

  local trk = ""
  local prev = 0
  for _, ev in ipairs(events) do
    local d = math.max(0, ev.tick - prev)
    trk = trk .. _vlq(d) .. string.char(ev.status, ev.d1, ev.d2)
    prev = ev.tick
  end
  trk = trk .. "\000\255\047\000"

  local tmp = "\000\255\081\003"
    .. string.char(
        math.floor(tempo_uspb/65536)%256,
        math.floor(tempo_uspb/256)%256,
        tempo_uspb%256)
    .. "\000\255\047\000"

  local f = io.open(filepath, "wb")
  if not f then return false, "Could not create " .. filepath end
  f:write("MThd" .. _u32be(6) .. _u16be(1) .. _u16be(2) .. _u16be(PPQ))
  f:write("MTrk" .. _u32be(#tmp) .. tmp)
  f:write("MTrk" .. _u32be(#trk) .. trk)
  f:close()
  return true, nil
end

return M
