# REAPER AI Assistant — Skills Series

A collection of Claude Code skills for working with REAPER DAW, without MCP dependencies.

## Skills

| Skill | Focus | Integration | Status |
|---|---|---|---|
| **reaper-session-automation** | Live session control — tracks, FX, routing, MIDI, render | Bridge Lua (file-based queue inside REAPER) | Active |
| **reaper-plugin-development** | VST3/AU/CLAP plugin development with JUCE/iPlug2 | None (artifacts only) | Planned |
| **reaper-audio-tools-integration** | External tools wiring — Demucs, voice cloning, BIAB/iReal Pro | `reapy` (Python sockets↔REAPER) | Planned |

## Architecture

Claude acts on the REAPER session **in real time** using two mechanisms:

- **Bridge Lua** (`reaper-session-automation`) — a ReaScript that runs inside REAPER and listens for commands via a file-based queue (`~/.reaper-ai-bridge/`). Claude writes Lua scripts to an inbox folder; the bridge executes them and writes results back.
- **reapy** (`reaper-audio-tools-integration`) — Python library that communicates with REAPER via sockets. Claude writes Python scripts that combine `reapy.Project()` with external audio tools.

No MCP servers involved. The user only needs to start the bridge once per REAPER session.

## Development

Each skill lives in its own folder with the standard layout:

```
skill-name/
├── SKILL.md           # Skill definition (name, description, instructions)
├── bridge/            # Bridge component (where applicable)
├── snippets/          # Reusable Lua/Python code templates
├── recipes/           # Multi-step workflow documentation
├── references/        # API docs, usage guides, gotchas
└── evals/             # Test cases and evaluations
```

For the full skill-creator workflow (test → review → iterate), see the [skill-creator skill](https://github.com/anthropics/claude-code/tree/main/.claude/skills/skill-creator).

## Installation

Once a skill is stable:

```bash
# Copy to global skills directory
cp -r skill-name/ ~/.claude/skills/skill-name/
```