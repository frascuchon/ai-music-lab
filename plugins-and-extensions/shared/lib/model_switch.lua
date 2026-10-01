-- lib/model_switch.lua  Pure decision logic for what must be reset/derived
-- when the user picks a different model in a combo box (MidiGenerator's
-- "Model" dropdown, AudioGenerator's "Model" dropdown in both Generate and
-- Edit mode).
--
-- Free-text prompts are model-specific advice ("use TAG format...", "chord
-- progression Am-F-C-G...") — carrying the previous model's prompt over to
-- a newly picked model is confusing and often produces a bad generation
-- (e.g. Foundation-1's TAG-format prompt fed to a plain descriptive model).
-- Numeric per-model defaults (GPU tier, temperature, ...) should likewise
-- snap to the new model's own default rather than keep the old model's.
--
-- No gfx/reaper dependency — testable with a plain lua interpreter (see
-- shared/lib/test_harness_model_switch.lua).

local M = {}

-- Whether a combo box's index actually changed this frame (the only case
-- in which dependent state should be reset/re-derived).
function M.changed(old_idx, new_idx)
  return old_idx ~= new_idx
end

-- The prompt text to keep for a text field that must be cleared whenever
-- the model changes. Returns "" (a fresh empty prompt) when the model
-- changed, or `current` unchanged otherwise — so callers can always do
-- `S.prompt = model_switch.next_prompt(old_idx, new_idx, S.prompt)`
-- regardless of whether the index actually changed this frame.
function M.next_prompt(old_idx, new_idx, current)
  if M.changed(old_idx, new_idx) then return "" end
  return current
end

-- Looks up a per-model default from a { model_key = value } table, falling
-- back to `default` when the model has no entry of its own. Used for
-- GPU tier / temperature / any other value that should snap to the newly
-- selected model's own default on change.
function M.default_for(defaults_by_key, model_key, default)
  if defaults_by_key and model_key and defaults_by_key[model_key] ~= nil then
    return defaults_by_key[model_key]
  end
  return default
end

return M
