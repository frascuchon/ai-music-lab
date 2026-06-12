#!/usr/bin/env bash
# Downloads the 12 official reference samples from the MIDI-LLM Vercel demo.
# Outputs per comparison_N/:
#   prompt_demo.txt                    (raw multi-line prompt)
#   reference_official_bf16.mp3        (MIDI-LLM bf16 — primary reference)
#   reference_official_fp8.mp3         (MIDI-LLM FP8 quantized)
#   reference_text2midi_competitor.mp3 (text2midi baseline from same prompt)
set -euo pipefail

BASE_URL="https://midi-llm-demo.vercel.app"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Downloading 12 official references from $BASE_URL ..."
for n in $(seq 1 12); do
  dir="$OUT_DIR/comparison_$n"
  mkdir -p "$dir"
  echo -n "  [comparison_$n] prompt..."
  curl -fsSL "$BASE_URL/prompts/comparison_$n.txt" -o "$dir/prompt_demo.txt"
  echo -n " bf16..."
  curl -fsSL "$BASE_URL/audio/comparison_$n/midi_llm_bf16.mp3" -o "$dir/reference_official_bf16.mp3"
  echo -n " fp8..."
  curl -fsSL "$BASE_URL/audio/comparison_$n/midi_llm_fp8.mp3"  -o "$dir/reference_official_fp8.mp3"
  echo -n " text2midi..."
  curl -fsSL "$BASE_URL/audio/comparison_$n/text2midi.mp3"     -o "$dir/reference_text2midi_competitor.mp3"
  echo " done"
done

echo ""
total="$(find "$OUT_DIR"/comparison_* -type f | wc -l | tr -d ' ')"
echo "Downloaded $total files across 12 comparison folders."
