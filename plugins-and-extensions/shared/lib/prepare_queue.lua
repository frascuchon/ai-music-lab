-- Scheduling for the Setup panel's "Prepare models" jobs (pure logic, no reaper).
--
-- Warm-ups are slow and network-bound, so several run at once:
--   * remote (Modal): up to MAX.remote jobs — each is its own Modal app, and the
--     image builds happen on Modal's side. Capped because Modal limits the GPU
--     containers a plan may run concurrently.
--   * local: one job at a time — downloads would only fight for bandwidth/disk.
-- Targets that share one image (group ~= "") never run together: the second
-- would build the same image again; once the first finishes the second is
-- skipped by `setup_helpers.py prepare` because it is already marked prepared.

local M = {}

M.MAX = { remote = 4, ["local"] = 1 }

function M.key(t)
  return t.domain .. ":" .. t.model .. ":" .. t.execution
end

-- queue: list of targets waiting. running: list of targets in progress.
-- Returns the indices of `queue` that may start now, in queue order.
function M.pick(queue, running)
  local count, groups = { remote = 0, ["local"] = 0 }, {}
  for _, t in ipairs(running) do
    count[t.execution] = (count[t.execution] or 0) + 1
    if t.group and t.group ~= "" then groups[t.group] = true end
  end
  local picked = {}
  for i, t in ipairs(queue) do
    local limit = M.MAX[t.execution] or 1
    local grouped = t.group and t.group ~= ""
    if (count[t.execution] or 0) < limit and not (grouped and groups[t.group]) then
      picked[#picked + 1] = i
      count[t.execution] = (count[t.execution] or 0) + 1
      if grouped then groups[t.group] = true end
    end
  end
  return picked
end

return M
