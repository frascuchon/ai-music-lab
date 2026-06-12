#!/usr/bin/env bash
# Generates MIDI (and optionally MP3) for all 12 comparison prompts using
# our local MPS research script. Skips comparisons that already have output.
#
# Usage:
#   bash regenerate_all.sh               # generate all missing MIDIs
#   bash regenerate_all.sh --render-mp3  # also render MP3 via MuseScore 4
#   bash regenerate_all.sh --force       # regenerate even if output exists
#   ONLY=3,7,10 bash regenerate_all.sh   # only specific comparisons
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RESEARCH_DIR="$ROOT/../../research"
SCRIPT="$RESEARCH_DIR/research_midi_llm.py"

RENDER_MP3=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --render-mp3) RENDER_MP3=1 ;;
    --force)      FORCE=1 ;;
  esac
done

# Optionally restrict to a subset: ONLY=3,7,10 bash regenerate_all.sh
if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -ra COMPARISONS <<< "$ONLY"
else
  COMPARISONS=($(seq 1 12))
fi

echo "=== MIDI-LLM local MPS regeneration ==="
echo "Script: $SCRIPT"
echo "Render MP3: $RENDER_MP3  Force: $FORCE"
echo ""

for n in "${COMPARISONS[@]}"; do
  dir="$ROOT/comparison_$n"
  out_mid="$dir/generated_local_mps_v0.mid"
  out_mp3="$dir/generated_local_mps_v0.mp3"

  if [[ ! -d "$dir" ]]; then
    echo "[$n/12] WARN: comparison_$n not found, skipping"; continue
  fi

  if [[ -f "$out_mid" && "$FORCE" -eq 0 ]]; then
    echo "[$n/12] skip (already generated)"
    continue
  fi

  # Read multi-line prompt and collapse to single line
  prompt="$(tr '\n' ' ' < "$dir/prompt_demo.txt" | sed 's/  */ /g' | sed 's/^ //;s/ $//')"
  echo "[$n/12] Generating..."
  echo "       Prompt: ${prompt:0:80}..."

  start="$(date +%s)"
  (cd "$RESEARCH_DIR" && uv run research_midi_llm.py \
      --prompt "$prompt" \
      --max_tokens 2046 \
      --out "$out_mid")
  elapsed=$(( $(date +%s) - start ))
  echo "[$n/12] Done in ${elapsed}s → $out_mid"

  if [[ "$RENDER_MP3" -eq 1 ]]; then
    if command -v mscore &>/dev/null || command -v mscore4 &>/dev/null; then
      cmd="${MSCORE_BIN:-$(command -v mscore4 2>/dev/null || command -v mscore)}"
      echo "[$n/12] Rendering MP3..."
      "$cmd" -o "$out_mp3" "$out_mid" 2>/dev/null
      echo "[$n/12] MP3 → $out_mp3"
    else
      echo "[$n/12] WARN: MuseScore not found, skipping MP3 render"
    fi
  fi
done

echo ""
echo "Regeneration complete."
