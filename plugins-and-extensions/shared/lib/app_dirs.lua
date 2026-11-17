-- Per-user folder for the app's shared resources (state that outlives a REAPER
-- session and is common to every plugin panel). Python mirror: prepare.APP_DIR_NAME.

local M = {}

M.APP_DIR_NAME = ".ai-musiclab"
M.APP_DIR = (os.getenv("HOME") or "") .. "/" .. M.APP_DIR_NAME

return M
