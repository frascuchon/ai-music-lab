-- First-run preparation markers, shared by the Setup panel and every plugin
-- panel. Format mirrors shared/prepare.py: ~/.ai-musiclab/prepared.txt, one
-- "domain|model|execution" per line (execution = "local" | "remote").
--
-- A model is "prepared" once its first inference has paid the one-off cost
-- (Modal image build + weights download, or local environment + weights): the
-- Setup panel's Prepare button writes the marker, and a panel marks the model
-- itself after any successful real run.

local M = {}

local app_dirs = require("app_dirs")
M.PATH = os.getenv("AI_MUSICLAB_PREPARED_FILE") or (app_dirs.APP_DIR .. "/prepared.txt")

local CACHE_SECS = 2
local cache, cache_at = {}, -math.huge

local function now()
  return reaper and reaper.time_precise and reaper.time_precise() or os.clock()
end

local function key(domain, model, execution)
  return domain .. "|" .. model .. "|" .. execution
end

local function load()
  local set = {}
  local f = io.open(M.PATH, "r")
  if f then
    for line in f:lines() do
      if select(2, line:gsub("|", "")) == 2 then set[line] = true end
    end
    f:close()
  end
  return set
end

function M.invalidate() cache_at = -math.huge end

function M.is_prepared(domain, model, execution)
  if now() - cache_at > CACHE_SECS then cache, cache_at = load(), now() end
  return cache[key(domain, model, execution)] == true
end

function M.mark(domain, model, execution)
  if not (domain and model and execution) then return end
  if load()[key(domain, model, execution)] then return end
  os.execute('mkdir -p "' .. M.PATH:match("^(.*)/[^/]*$") .. '"')
  -- Append-only, like prepare.py: concurrent warm-ups must not drop markers.
  local f = io.open(M.PATH, "a")
  if f then f:write(key(domain, model, execution), "\n"); f:close() end
  M.invalidate()
end

-- Draws the first-run warning when (domain, model, execution) is not prepared.
-- Returns true when it drew something.
function M.notice(g, domain, model, execution)
  if M.is_prepared(domain, model, execution) then return false end
  g.text_colored("First run of this model on " ..
    (execution == "local" and "this machine" or "Modal") .. " is slow", "YELLOW")
  g.text_wrapped("The environment has to be prepared first (" ..
    (execution == "local" and "dependencies and weights download"
      or "container image build and weights download") ..
    "), which can take several minutes. Open the Setup tab → \"Prepare models\" " ..
    "to do it now, or just generate and wait.")
  g.spacing()
  return true
end

return M
