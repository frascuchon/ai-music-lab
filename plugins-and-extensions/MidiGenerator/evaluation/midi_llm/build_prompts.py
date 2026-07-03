#!/usr/bin/env python3
"""
Generates prompt.txt in each comparison_N/ folder following the canonical
format used by the rest of MidiGenerator/evaluation/.
Run from the midi_llm/ evaluation directory.
"""
import os
import re

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://midi-llm-demo.vercel.app"

TEMPLATE = """\
Model: MIDI-LLM (slseanwu/MIDI-LLM_Llama-3.2-1B)
Source: {base_url}/prompts/comparison_{n}.txt
Prompt: "{prompt_oneline}"
References (from the official Vercel demo):
  - reference_official_bf16.mp3           (MIDI-LLM bf16 — primary reference)
  - reference_official_fp8.mp3            (MIDI-LLM FP8 quantized — sanity check)
  - reference_text2midi_competitor.mp3    (text2midi on same prompt — cross-comparison)
Note: MIDI ground-truth is not publicly downloadable (demo API only, ngrok backend).
Our command: uv run research_midi_llm.py --prompt "<see Prompt above>" --max_tokens 2046 --out generated_local_mps_v0.mid
"""

for n in range(1, 13):
    folder = os.path.join(EVAL_DIR, f"comparison_{n}")
    raw_path = os.path.join(folder, "prompt_demo.txt")
    out_path = os.path.join(folder, "prompt.txt")

    if not os.path.isfile(raw_path):
        print(f"  [comparison_{n}] WARN: prompt_demo.txt not found, skipping")
        continue

    with open(raw_path, encoding="utf-8") as f:
        raw = f.read().strip()

    # Collapse multi-line prompt to a single line (for the Prompt: field)
    oneline = re.sub(r"\s+", " ", raw)

    content = TEMPLATE.format(
        base_url=BASE_URL,
        n=n,
        prompt_oneline=oneline,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  [comparison_{n}] prompt.txt written ({len(oneline)} chars)")

print("\nDone — prompt.txt created in all 12 comparison folders.")
