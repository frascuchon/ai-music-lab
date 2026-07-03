# REAPER AI Plugins

A collection of REAPER plugins that bring AI-powered audio and MIDI tools directly into your DAW — no browser required. All heavy inference runs on [Modal](https://modal.com) GPU cloud; only lightweight Python glue runs locally via [uv](https://docs.astral.sh/uv/).

## Plugins

| Plugin | What it does | Models |
|---|---|---|
| **Audio2Midi** | Transcribe any audio to MIDI (polyphonic, multi-instrument) | MIROS, YourMT3+ |
| **MidiGenerator** | Generate MIDI from a text prompt or a seed MIDI file | Amadeus, MIDI-LLM, text2midi, ChatMusician, MuseCoco, Anticipatory MT |
| **Text2Audio** | Generate or edit audio from a text description | Stable Audio Open, Foundation-1, ACE-Step, MusicGen, and more |
| **StemsSeparator** | Separate any audio into stems (vocals, drums, bass, other…) | Demucs (local), SAM Audio (Modal) |

A shared **Setup** wizard handles the full environment configuration — Python, uv, Modal auth, HF secrets, and optional weight pre-loading — for all plugins from a single UI.

---

## Requirements

- **REAPER** 6.x or later ([reaper.fm](https://www.reaper.fm))
- **ReaPack** extension ([reapack.com](https://reapack.com))
- **Python 3.11+** configured in REAPER preferences (macOS: the system Python or a Homebrew/pyenv install)
- **uv** package manager — installed automatically by the Setup wizard, or manually: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Modal account** (free tier is enough for personal use) — [modal.com](https://modal.com)

---

## Installation

### 1. Install ReaPack

If you don't have ReaPack yet:

1. Download the platform-specific extension from [reapack.com](https://reapack.com).
2. Place it in your REAPER `UserPlugins/` folder (macOS: `~/Library/Application Support/REAPER/UserPlugins/`).
3. Restart REAPER — a new **Extensions → ReaPack** menu should appear.

### 2. Add this repository to ReaPack

1. In REAPER: **Extensions → ReaPack → Manage repositories**
2. Click **Import a repository**
3. Paste the URL:
   ```
   https://raw.githubusercontent.com/frascuchon/ai-music-lab/main/index.xml
   ```
4. Click **OK** and confirm. ReaPack will fetch the package list.

### 3. Install the plugins

1. **Extensions → ReaPack → Browse packages**
2. Filter by repository: **ai-music-lab**
3. Select the plugins you want (you can install all of them):
   - `reaper-ai-plugins-shared` *(required — installed automatically as a dependency)*
   - `Audio2Midi`
   - `MidiGenerator`
   - `Text2Audio`
   - `StemsSeparator`
4. Right-click → **Install** → **Apply**

ReaPack copies the Lua scripts to your REAPER Scripts folder and registers them in the action list.

---

## First-time setup

After installing, run the **Setup wizard** once to configure the environment.

### Open Setup

- **Actions → Show action list** → search for `AI Music Lab - Setup` → **Run**

Or navigate to:
```
~/Library/Application Support/REAPER/Scripts/ai-music-lab/shared/Setup.lua
```
and run it from the ReaScript editor.

### What the Setup wizard does

The wizard checks and configures everything in one place:

| Section | What it checks / installs |
|---|---|
| **Common environment** | REAPER's Python path · uv · Modal CLI · Modal authentication |
| **StemsSeparator** | demucs (local pip) · Hugging Face secret for SAM Audio |
| **Audio2Midi** | Optional pre-load of MIROS and YourMT3+ weights to Modal Volume |
| **MidiGenerator** | Optional pre-load of any of the 6 model weights to Modal Volume |

Work through the sections top to bottom:

1. **Python (REAPER)** — must show `[OK]`. If not, set the Python path in REAPER preferences: **Preferences → Plug-ins → ReaScript** → Python library path.
2. **uv** — click **Install uv** if missing. Restart REAPER afterwards.
3. **Modal CLI** — click **Install deps** once uv is installed.
4. **Modal authenticated** — click **Login** to open the browser auth flow. The token is saved automatically to `~/.modal.toml`.
5. *(StemsSeparator only)* **demucs** — click **Install** to install it locally. **HF Token** — paste a Hugging Face token (starting with `hf_`) and click **Save** to create the Modal secret used by SAM Audio.
6. *(Optional)* Use the pre-load buttons to download model weights to Modal Volumes. First-time Modal cold starts are slower; pre-loading eliminates that wait.

Click **Re-check all** at any time to refresh the status of every check.

---

## Configuring Python in REAPER

The plugins use the Python interpreter that REAPER itself is configured to use.

1. **REAPER → Preferences → Plug-ins → ReaScript**
2. Under **Python library path**, set the path to your Python `lib/` directory.
   - macOS with Homebrew Python 3.11: `/opt/homebrew/Cellar/python@3.11/<version>/Frameworks/Python.framework/Versions/3.11/lib`
   - macOS system Python: `/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib`
3. Restart REAPER.

The Setup wizard's **Python (REAPER)** check will confirm the path is correct and show the detected version.

---

## Usage

Each plugin appears in the REAPER action list after installation:

| Action name | Plugin |
|---|---|
| `AI Music Lab - Audio2Midi` | Transcribe selected audio item to MIDI |
| `AI Music Lab - MidiGenerator` | Generate MIDI from prompt or seed |
| `AI Music Lab - Text2Audio` | Generate or edit audio from text |
| `AI Music Lab - StemsSeparator` | Separate selected audio into stems |

Select an audio or MIDI item in REAPER, open the plugin from the action list, configure the options, and click the generate button. Progress is shown live in the plugin's log panel. Results are imported back into REAPER automatically.

---

## How it works

```
REAPER (Lua plugin)
    │  launches as background process
    ▼
Python backend script (transcribe.py / midigen.py / …)
    │  uv run --project shared/  modal run script::main
    ▼
Modal cloud GPU  (A10G / A100)
    │  returns output file (MIDI / WAV)
    ▼
Python backend  →  writes result path to progress file
    │
    ▼
Lua plugin  →  imports result into REAPER project
```

All cloud inference costs are billed to your Modal account. Typical costs:
- Audio2Midi (MIROS, A10G): ~$0.03–0.08 per transcription
- MidiGenerator (most models, A10G): ~$0.05–0.15 per generation
- Text2Audio (SAO, A10G): ~$0.05–0.10 per generation
- StemsSeparator SAM (A100): ~$0.14 per track

---

## Repository structure

```
.
├── index.xml                       ReaPack package index
└── plugins-and-extensions/
    ├── Audio2Midi/
    │   ├── Audio2Midi.lua          Main plugin UI
    │   ├── transcribe.py           Python backend
    │   └── research/               Modal inference scripts
    ├── MidiGenerator/
    │   ├── MidiGenerator.lua
    │   ├── midigen.py
    │   ├── prompt_adapters.py
    │   └── research/
    ├── Text2Audio/
    │   ├── Text2Audio.lua
    │   ├── text2audio.py
    │   └── research/
    ├── StemsSeparator/
    │   ├── StemSeparator.lua
    │   ├── separate_demucs.py      Local Demucs backend
    │   ├── separate_sam.py         SAM Audio (Modal) backend
    │   └── modal_sam_audio.py      Modal inference app
    └── shared/
        ├── Setup.lua               Global setup wizard
        ├── setup_helpers.py        Setup CLI backend
        ├── pyproject.toml          uv project (modal, protobuf)
        └── lib/
            ├── gui.lua             Immediate-mode gfx widget toolkit
            ├── theme.lua
            ├── widgets_extra.lua
            └── common.lua
```

---

## License

MIT — see individual plugin files for details.
