#!/usr/bin/env bash
# Generates MIDI (and optionally MP3) for all 12 comparison prompts using
# our local MPS research script.
#
# Usage:
#   bash regenerate_all.sh               # skip comparisons that already have v0
#   bash regenerate_all.sh --render-mp3  # also render MP3 via MuseScore 4
#   bash regenerate_all.sh --force       # overwrite even if output exists
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

if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -ra COMPARISONS <<< "$ONLY"
else
  COMPARISONS=($(seq 1 12))
fi

MSCORE_BIN="${MSCORE_BIN:-$(command -v mscore 2>/dev/null || echo '')}"

echo "=== MIDI-LLM local MPS regeneration ==="
echo "Script:     $SCRIPT"
echo "Render MP3: $RENDER_MP3  Force: $FORCE"
echo "n_outputs=4  temperature=1.0  top_p=0.98  max_tokens=2046"
echo ""

for n in "${COMPARISONS[@]}"; do
  dir="$ROOT/comparison_$n"
  out_v0="$dir/generated_local_mps_v0.mid"

  if [[ ! -d "$dir" ]]; then
    echo "[$n/12] WARN: comparison_$n not found, skipping"; continue
  fi

  if [[ -f "$out_v0" && "$FORCE" -eq 0 ]]; then
    echo "[$n/12] skip (v0 already exists — use --force to overwrite)"
    continue
  fi

  # Remove stale outputs before regenerating
  rm -f "$dir"/generated_local_mps_v*.mid "$dir"/generated_local_mps_v*.mp3

  prompt="$(tr '\n' ' ' < "$dir/prompt_demo.txt" | sed 's/  */ /g;s/^ //;s/ $//')"
  echo "[$n/12] Generating 4 outputs..."
  echo "       Prompt: ${prompt:0:90}..."

  start="$(date +%s)"
  (cd "$RESEARCH_DIR" && uv run research_midi_llm.py \
      --prompt "$prompt" \
      --out "$out_v0" \
      --n_outputs 4 \
      --temperature 1.0 \
      --top_p 0.98 \
      --max_tokens 2046)
  elapsed=$(( $(date +%s) - start ))
  echo "[$n/12] Done in ${elapsed}s"

  if [[ "$RENDER_MP3" -eq 1 && -n "$MSCORE_BIN" ]]; then
    for mid in "$dir"/generated_local_mps_v*.mid; do
      mp3="${mid%.mid}.mp3"
      echo "  rendering $(basename "$mid") → $(basename "$mp3")..."
      "$MSCORE_BIN" -o "$mp3" "$mid" 2>/dev/null
    done
  elif [[ "$RENDER_MP3" -eq 1 ]]; then
    echo "[$n/12] WARN: MuseScore not found, skipping MP3 render"
  fi
done

echo ""
echo "Regeneration complete."
