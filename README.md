# AI Music Lab for REAPER

A single REAPER panel with AI tools for your composition workflow: separate stems, turn audio into MIDI, get MIDI accompaniment for a melody, and generate or transform audio. No browser needed. Heavy models run on a rented cloud GPU ([Modal](https://modal.com)) or, for some of them, on your own machine; only a little Python glue lives locally ([uv](https://docs.astral.sh/uv/)).

> The models are **research models, not products**. Some work very well, others only sometimes. Each tool below says what to expect.

| Tab | What it does | Models |
|---|---|---|
| **Stems Separator** | Split a mix into stems (vocals, drums, bass…) or extract a sound by text prompt | Demucs, SAM Audio |
| **Audio2Midi** | Transcribe audio to MIDI | Basic Pitch, YourMT3+ |
| **MidiGenerator** | Generate MIDI from text, or accompany a melody | Amadeus, Anticipatory Music Transformer, MIDI-GPT |
| **AudioGenerator** (Text2Audio) | Generate audio from text, or transform an existing clip | ACE-Step 1.5, Stable Audio Open, MusicGen, AudioGen; edit: SAO style transfer, ACE-Step |
| **Setup** | One-click environment check and configuration | — |

---

## Quick install (about 10 minutes)

You need **REAPER 6.x+** ([reaper.fm](https://www.reaper.fm)) and a free **[Modal](https://modal.com) account** (only for cloud models; Demucs and Basic Pitch run without it).

1. **Install ReaPack** ([reapack.com](https://reapack.com)): drop the extension in REAPER's `UserPlugins/` folder and restart REAPER (macOS: `~/Library/Application Support/REAPER/UserPlugins/`).
2. **Add this repository**: *Extensions → ReaPack → Import repositories…* and paste
   ```
   https://github.com/frascuchon/ai-music-lab/raw/main/index.xml
   ```
3. **Install the plugin**: *Extensions → ReaPack → Browse packages*, search **AI Music Lab**, right-click → *Install*, then *Apply*.
4. **Point REAPER at Python 3.11+**: *Preferences → Plug-ins → ReaScript → Python library path* (the `lib/` folder itself; see [Configuring Python](#configuring-python-in-reaper)). Restart REAPER.
5. **Open it**: *Actions → Show action list*, search `AI Music Lab`, run it. Assign a shortcut if you like.
6. **Run Setup** (the *Setup* button in the panel) and go top to bottom: *uv* → *Modal CLI* → *Login* (opens the browser) → Hugging Face token if you use SAM Audio or MIDI-GPT. Everything should read `[OK]`.

That's it. Select an item in REAPER, pick the tab and model, press the action button; the result appears as new tracks/clips below the source. Press the same button again to stop a run.

**Updating:** *Extensions → ReaPack → Synchronize packages*.

---

## Models

Reliability notes come from the author's own testing. Weights are downloaded the first time a model runs (cloud models cache them in a Modal Volume; local ones in your Hugging Face cache), so the first run is slower.

### Stems Separator

| Model | Use it for | Runs | Links | Notes |
|---|---|---|---|---|
| **Demucs** (`htdemucs`, `htdemucs_ft`, `mdx_extra`, `htdemucs_6s`) | 4- or 6-stem separation (vocals, drums, bass, other; 6s adds guitar and piano) | Local only | [GitHub](https://github.com/facebookresearch/demucs) | Solid, used daily. Installed from Setup (*demucs → Install*). |
| **SAM Audio** (`large`, `base`) | Pull out any sound described in words ("jazz trumpet", "vocals") | Cloud (default) or local | [GitHub](https://github.com/facebookresearch/sam-audio) · [HF large](https://huggingface.co/facebook/sam-audio-large) · [HF base](https://huggingface.co/facebook/sam-audio-base) | Gated weights: accept the license on Hugging Face and save your `hf_…` token in Setup. Cloud default is an A100-80GB. Local needs ~15–20 GB (large). |

### Audio2Midi

| Model | Use it for | Runs | Links | Notes |
|---|---|---|---|---|
| **Basic Pitch** | One instrument or voice (guitar, piano, vocals) | Local (CPU), no Modal | [GitHub](https://github.com/spotify/basic-pitch) | Default. Ghost-note and tail-noise cleanup, adjustable *Threshold* (default 0.7). Apache 2.0. |
| **YourMT3+** | Multi-instrument mixes | Cloud (T4+) | [GitHub](https://github.com/mimbres/YourMT3) · [HF Space](https://huggingface.co/spaces/mimbres/YourMT3) | Good on mixes; on a solo distorted guitar Basic Pitch was clearly better. Apache 2.0. |

### MidiGenerator

| Model | Use it for | Runs | Links | Notes |
|---|---|---|---|---|
| **Anticipatory Music Transformer** | Accompaniment for a MIDI melody, with arrangement controls (parts, density, creativity) | Cloud or local (CPU) | [GitHub](https://github.com/jthickstun/anticipation) · [HF](https://huggingface.co/stanford-crfm/music-medium-800k) | The most reliable for accompaniment. First results are often "basic"; use the controls. |
| **MIDI-GPT** | Accompaniment with fixed rhythm per part | Cloud only | [HF](https://huggingface.co/Metacreation/MIDI-GPT) | **Non-commercial weights**: accept the license on HF and save your token in Setup. |
| **Amadeus** | Multi-track MIDI from a text prompt | Cloud only | [GitHub](https://github.com/lingyu123-su/Amadeus) | Best instrumentation of the text-to-MIDI models tested; durations are inconsistent. |

### AudioGenerator (Text2Audio)

| Mode | Model | Runs | Links | Notes |
|---|---|---|---|---|
| Generate | **ACE-Step 1.5** | Cloud | [GitHub](https://github.com/ace-step/ACE-Step-1.5) | Full songs, up to ~3 min. Apache 2.0. |
| Generate | **Stable Audio Open 1.0** | Cloud or local (slow) | [HF](https://huggingface.co/stabilityai/stable-audio-open-1.0) · [GitHub](https://github.com/Stability-AI/stable-audio-tools) | 44.1 kHz stereo, up to 47 s. Good for textures and loops. Gated: accept the license on HF. |
| Generate | **MusicGen-medium** | Cloud or local (slow) | [GitHub](https://github.com/facebookresearch/audiocraft) · [HF](https://huggingface.co/facebook/musicgen-medium) | 32 kHz, up to 30 s. **CC-BY-NC** weights. |
| Generate | **AudioGen-medium** | Cloud or local | [GitHub](https://github.com/facebookresearch/audiocraft) · [HF](https://huggingface.co/facebook/audiogen-medium) | Sound effects and ambiences, 16 kHz. |
| Edit | **SAO style transfer** | Cloud or local (slow) | [HF](https://huggingface.co/stabilityai/stable-audio-open-1.0) | Re-styles a clip (SDEdit-style). Fun for textures; it does not reliably keep a melody while changing timbre. |
| Edit | **ACE-Step 1.5** (cover / re-style) | Cloud | [GitHub](https://github.com/ace-step/ACE-Step-1.5) | Same caveat on melody preservation. |

Models evaluated and removed from the UI (text2midi, MuseCoco, MIDI-LLM, ChatMusician, MIROS, MusicGen-melody, MelodyFlow, Mustango, MAGNeT, InspireMusic…) keep their scripts and notes in each plugin's `research/` and `RESEARCH.md`.

---

## Local vs. remote execution

Models that can run on your machine show an **Execution: Local / Remote** switch in their panel.

| | **Remote** (Modal cloud GPU) | **Local** (your Mac/PC, MPS or CPU) |
|---|---|---|
| Speed | Fast (A10G / A100) | Slower, especially diffusion and autoregressive audio models |
| Cost | Billed to your Modal account (see below) | Free |
| Setup | Modal login | Extra Python packages and a model download on first run |
| Privacy | Audio is sent to Modal for the run | Nothing leaves your machine |
| Offline | No | Yes, once weights are downloaded |

Supported locally: Demucs (always local), Basic Pitch (always local), Anticipatory, Stable Audio Open (+ edit), MusicGen, AudioGen, SAM Audio. Cloud only: ACE-Step, MIDI-GPT, Amadeus, YourMT3+. When a model can't run locally, the *Local* option is disabled and the panel shows why. Local SAM Audio and Stable Audio paths are newer and less battle-tested than the cloud ones.

---

## Setup wizard details

The wizard checks and configures everything in one place:

| Section | What it checks / installs |
|---|---|
| **Common environment** | REAPER's Python path · uv · Modal CLI · Modal authentication |
| **Stems Separator** | demucs (local pip) · Hugging Face secret for SAM Audio |
| **Audio2Midi**, **MidiGenerator**, **AudioGenerator** | Nothing extra for cloud models; local models install their own dependencies with uv on first use. MIDI-GPT also needs the Hugging Face secret |

Work through the sections top to bottom:

1. **Python (REAPER)** — must show `[OK]`. If not, set the Python path in REAPER preferences: **Preferences → Plug-ins → ReaScript** → Python library path.
2. **uv** — click **Install uv** if missing. Restart REAPER afterwards.
3. **Modal CLI** — click **Install deps** once uv is installed.
4. **Modal authenticated** — click **Login** to open the browser auth flow. The token is saved automatically to `~/.modal.toml`.
5. *(StemsSeparator only)* **demucs** — click **Install** to install it locally. **HF Token** — paste a Hugging Face token (starting with `hf_`) and click **Save** to create the Modal secret used by SAM Audio.
6. The first run of each model is slower because Modal downloads its weights into a Volume. Later runs reuse them.

Click **Re-check all** at any time to refresh the status of every check.

---

## Configuring Python in REAPER

The plugins use the Python interpreter that REAPER itself is configured to use.

1. **REAPER → Preferences → Plug-ins → ReaScript**
2. Under **Python library path**, set the path to your Python `lib/` directory — the
   folder itself, not a subfolder inside it (e.g. not `lib/python3.11/config-3.11-darwin`).
   - macOS with Homebrew Python 3.11: `/opt/homebrew/Cellar/python@3.11/<version>/Frameworks/Python.framework/Versions/3.11/lib`
   - macOS system Python: `/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib`
   - macOS official python.org installer: `/Library/Frameworks/Python.framework/Versions/3.11/lib`
3. Restart REAPER.

The Setup wizard's **Python (REAPER)** check will confirm the path is correct and show the detected version.

---

## Usage

Everything lives in one window (action `AI Music Lab`), with a tab per tool. Select an audio or MIDI item (or a track for stems), choose the tab and model, set the options and press the action button. Progress shows live in the log panel, and results are imported below the source track/clip without touching what is already in your project. The collapse button shrinks the window to its header.

| Tab | Input | Output |
|---|---|---|
| Stems Separator | Selected audio item | One track per stem (folder) |
| Audio2Midi | Selected audio item | MIDI track(s) |
| MidiGenerator | Text prompt, or a selected MIDI item (melody) | MIDI tracks with the accompaniment |
| AudioGenerator | Text prompt, or a selected audio item (edit mode) | New audio item |

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
- Audio2Midi (YourMT3+, A10G): ~$0.03–0.08 per transcription; Basic Pitch runs locally for free
- MidiGenerator (A10G): ~$0.05–0.15 per generation; Anticipatory can run locally for free
- AudioGenerator (SAO, A10G): ~$0.05–0.10 per generation
- Stems Separator: Demucs is free; SAM Audio on an A100 ≈ $0.14 per track

---

## Repository structure

```
.
├── index.xml                     ReaPack package index
└── plugins-and-extensions/
    ├── ai-music-lab.lua          Unified window (entry point)
    ├── Audio2Midi/               panel.lua, transcribe.py, research/ (Modal + local scripts)
    ├── MidiGenerator/            panel.lua, midigen.py, accompaniment.py, harmony.py, smf_writer.lua, research/
    ├── Text2Audio/               panel.lua, text2audio.py, text2audio_local.py, research/
    ├── StemsSeparator/           panel.lua, separate_demucs.py, separate_sam.py, separate_sam_local.py, modal_sam_audio.py
    └── shared/                   Setup panel, uv project (modal), model_execution registry, lib/ (gui, theme, midi_import…)
```

Per-tool details, evaluations and model comparisons are in each folder's `README.md` and `RESEARCH.md`; `docs/research/` holds the cross-model surveys.

---

## License

The code in this repository is MIT (see [LICENSE](LICENSE)). **The models are not**: each has its own license (e.g. MIDI-GPT and MusicGen weights are non-commercial, Stable Audio Open and SAM Audio are gated). Check the linked model pages before commercial use.
