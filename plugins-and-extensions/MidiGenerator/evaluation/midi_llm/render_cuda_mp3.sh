#!/usr/bin/env bash
# Renders generated_cuda_v*.mid → generated_cuda_v*.mp3 in each comparison_N folder.
# Requires MuseScore (mscore) in PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MSCORE_BIN="${MSCORE_BIN:-$(command -v mscore 2>/dev/null || echo '')}"

if [[ -z "$MSCORE_BIN" ]]; then
  echo "ERROR: MuseScore (mscore) not found. Install or set MSCORE_BIN=/path/to/mscore"
  exit 1
fi

total=0
for mid in "$ROOT"/comparison_*/generated_cuda_v*.mid; do
  mp3="${mid%.mid}.mp3"
  [[ -f "$mp3" ]] && continue  # already rendered
  echo "  rendering $(basename "$(dirname "$mid")")/$(basename "$mid")..."
  "$MSCORE_BIN" -o "$mp3" "$mid" 2>/dev/null && total=$((total+1))
done
echo "Rendered $total MP3 files."
