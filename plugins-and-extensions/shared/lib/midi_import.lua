-- lib/midi_import.lua  Import a Standard MIDI File into REAPER as new tracks,
-- without reaper.InsertMedia.
--
-- Why: InsertMedia leaves REAPER to decide where the tracks of a multi-track
-- MIDI go (new tracks vs. existing ones, before/after the selected track,
-- depending on the user's MIDI-import preferences). Callers then had to infer
-- which tracks had been created and wrap them in a folder, so with other MIDI
-- clips already in the project the folder swallowed unrelated tracks and
-- items were moved or overwritten. Here the file is parsed in Lua and every
-- track and item is created explicitly at a known index, so nothing that is
-- already in the project is ever touched.
--
--   parse_smf(path)                 pure Lua, no reaper dependency (testable)
--   import_file(path, opts)         creates tracks/items through the reaper API

local M = {}

-- ── SMF PARSER ───────────────────────────────────────────────────

local function u16(s, i) return s:byte(i) * 256 + s:byte(i + 1) end
local function u32(s, i)
  return ((s:byte(i) * 256 + s:byte(i + 1)) * 256 + s:byte(i + 2)) * 256 + s:byte(i + 3)
end

local function varlen(s, p)
  local v = 0
  while true do
    local b = s:byte(p)
    if not b then return v, p end
    p = p + 1
    v = v * 128 + (b & 0x7F)
    if b < 0x80 then return v, p end
  end
end

-- Parse one MTrk body. Returns { name, tempos = {{tick, us}}, channels = {[ch] = {notes, program}} }
local function parse_track(s, p, stop)
  local tick, running = 0, nil
  local out = { name = "", tempos = {}, channels = {} }
  local open = {}   -- open[ch][pitch] = { {tick, vel}, ... } (FIFO)

  local function chan(ch)
    local c = out.channels[ch]
    if not c then c = { notes = {}, program = nil }; out.channels[ch] = c end
    return c
  end
  local function close_note(ch, pitch, at)
    local stack = open[ch] and open[ch][pitch]
    if stack and #stack > 0 then
      local on = table.remove(stack, 1)
      table.insert(chan(ch).notes, { start_tick = on[1], end_tick = at, pitch = pitch, vel = on[2] })
    end
  end

  while p <= stop do
    local delta
    delta, p = varlen(s, p)
    tick = tick + delta
    local status = s:byte(p)
    if not status then break end
    if status >= 0x80 then
      p = p + 1
      if status < 0xF0 then running = status end
    else
      status = running   -- running status: the byte just read is data
      if not status then break end
    end

    if status == 0xFF then
      local mtype = s:byte(p); p = p + 1
      local len; len, p = varlen(s, p)
      if mtype == 0x2F then
        break
      elseif mtype == 0x03 then
        out.name = s:sub(p, p + len - 1)
      elseif mtype == 0x51 and len == 3 then
        table.insert(out.tempos, { tick = tick, us = (s:byte(p) * 256 + s:byte(p + 1)) * 256 + s:byte(p + 2) })
      end
      p = p + len
    elseif status == 0xF0 or status == 0xF7 then
      local len; len, p = varlen(s, p)
      p = p + len
    else
      local hi, ch = status >> 4, status & 0x0F
      if hi == 0xC or hi == 0xD then
        local d1 = s:byte(p); p = p + 1
        if hi == 0xC then
          local c = chan(ch)
          if c.program == nil then c.program = d1 end
        end
      else
        local d1, d2 = s:byte(p), s:byte(p + 1); p = p + 2
        if hi == 0x9 and d2 and d2 > 0 then
          open[ch] = open[ch] or {}
          open[ch][d1] = open[ch][d1] or {}
          table.insert(open[ch][d1], { tick, d2 })
          chan(ch)
        elseif hi == 0x8 or (hi == 0x9 and d2 == 0) then
          close_note(ch, d1, tick)
        end
      end
    end
  end

  -- notes still sounding at the end of the track
  for ch, pitches in pairs(open) do
    for pitch, stack in pairs(pitches) do
      for _ = 1, #stack do close_note(ch, pitch, tick) end
    end
  end
  return out
end

-- Returns { division, bpm, instruments = { {name, channel, program, is_drum, notes} }, length }
-- with note times in SECONDS (start_s / end_s), or nil, error.
function M.parse_smf(path)
  local f = io.open(path, "rb")
  if not f then return nil, "cannot open " .. tostring(path) end
  local s = f:read("a"); f:close()
  if #s < 14 or s:sub(1, 4) ~= "MThd" then return nil, "not a MIDI file" end

  local division = u16(s, 13)
  if division >= 0x8000 then return nil, "SMPTE time division is not supported" end
  local p = 9 + u32(s, 5)

  local tracks = {}
  while p + 8 <= #s + 1 do
    local id, len = s:sub(p, p + 3), u32(s, p + 4)
    local body = p + 8
    if id == "MTrk" then
      table.insert(tracks, parse_track(s, body, math.min(body + len - 1, #s)))
    end
    p = body + len
  end

  -- global tempo map (tempo events can live on any track)
  local tempos = {}
  for _, t in ipairs(tracks) do
    for _, ev in ipairs(t.tempos) do table.insert(tempos, ev) end
  end
  table.sort(tempos, function(a, b) return a.tick < b.tick end)
  local segs, sec, last_tick, us = {}, 0, 0, 500000
  table.insert(segs, { tick = 0, sec = 0, us = us })
  for _, ev in ipairs(tempos) do
    sec = sec + (ev.tick - last_tick) * us / 1e6 / division
    last_tick, us = ev.tick, ev.us
    table.insert(segs, { tick = ev.tick, sec = sec, us = us })
  end
  local function to_sec(tick)
    local seg = segs[1]
    for i = #segs, 1, -1 do
      if segs[i].tick <= tick then seg = segs[i]; break end
    end
    return seg.sec + (tick - seg.tick) * seg.us / 1e6 / division
  end

  local instruments, length = {}, 0
  for _, t in ipairs(tracks) do
    local chans = {}
    for ch, c in pairs(t.channels) do
      if #c.notes > 0 then table.insert(chans, ch) end
    end
    table.sort(chans)
    for _, ch in ipairs(chans) do
      local c = t.channels[ch]
      table.sort(c.notes, function(a, b)
        if a.start_tick ~= b.start_tick then return a.start_tick < b.start_tick end
        return a.pitch < b.pitch
      end)
      local notes = {}
      for _, n in ipairs(c.notes) do
        local ns, ne = to_sec(n.start_tick), to_sec(n.end_tick)
        if ne <= ns then ne = ns + 0.01 end
        table.insert(notes, { start_s = ns, end_s = ne, pitch = n.pitch, vel = n.vel })
        if ne > length then length = ne end
      end
      local name = t.name
      if name ~= "" and #chans > 1 then name = name .. " (ch " .. (ch + 1) .. ")" end
      table.insert(instruments, {
        name = name, channel = ch, program = c.program, is_drum = (ch == 9), notes = notes,
      })
    end
  end

  return {
    division = division,
    bpm = 60e6 / segs[1].us,
    instruments = instruments,
    length = length,
  }
end

-- ── REAPER IMPORT ────────────────────────────────────────────────

-- opts:
--   insert_at     0-based track index where the first new track goes
--   position      project time (s) where the MIDI starts
--   name          name of the single track, or of the folder
--   always_folder wrap even a single instrument in a folder
--
-- Returns { first = insert_at, count = tracks created (folder included),
--           instruments = n, next_insert_at = index right below the new tracks }
-- or nil, error. Existing tracks and items are never modified.
function M.import_file(path, opts)
  local parsed, err = M.parse_smf(path)
  if not parsed then return nil, err end
  local insts = parsed.instruments
  if #insts == 0 then return nil, "the MIDI file contains no notes" end

  opts = opts or {}
  local insert_at = math.min(math.max(opts.insert_at or reaper.CountTracks(0), 0), reaper.CountTracks(0))
  local position  = opts.position or 0
  local name      = opts.name or "MIDI"
  local as_folder = opts.always_folder or #insts > 1

  local idx = insert_at
  if as_folder then
    reaper.InsertTrackAtIndex(idx, true)
    local folder = reaper.GetTrack(0, idx)
    reaper.GetSetMediaTrackInfo_String(folder, "P_NAME", name, true)
    reaper.SetMediaTrackInfo_Value(folder, "I_FOLDERDEPTH", 1)
    idx = idx + 1
  end

  for k, inst in ipairs(insts) do
    reaper.InsertTrackAtIndex(idx, true)
    local tr = reaper.GetTrack(0, idx)
    local tname = name
    if as_folder then
      tname = (inst.name ~= "" and inst.name) or (name .. " " .. k)
    end
    reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", tname, true)

    -- small tail so a PPQ rounding step never clips the last note
    local item = reaper.CreateNewMIDIItemInProj(tr, position, position + parsed.length + 0.05, false)
    local take = item and reaper.GetActiveTake(item)
    if take then
      if inst.program and not inst.is_drum then
        reaper.MIDI_InsertCC(take, false, false, 0, 0xC0, inst.channel, inst.program, 0)
      end
      for _, n in ipairs(inst.notes) do
        local sp = reaper.MIDI_GetPPQPosFromProjTime(take, position + n.start_s)
        local ep = reaper.MIDI_GetPPQPosFromProjTime(take, position + n.end_s)
        reaper.MIDI_InsertNote(take, false, false, sp, ep, inst.channel, n.pitch, n.vel, true)
      end
      reaper.MIDI_Sort(take)
    end
    idx = idx + 1
  end

  if as_folder then
    reaper.SetMediaTrackInfo_Value(reaper.GetTrack(0, idx - 1), "I_FOLDERDEPTH", -1)
  end

  return {
    first = insert_at,
    count = idx - insert_at,
    instruments = #insts,
    next_insert_at = idx,
    bpm = parsed.bpm,
  }
end

return M
