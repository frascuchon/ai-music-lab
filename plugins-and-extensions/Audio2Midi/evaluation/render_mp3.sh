#!/usr/bin/env bash
# Renderiza todos los transcribed_cuda.mid → transcribed_cuda.mp3 con MuseScore.
# Idempotente: si el MP3 ya existe lo omite.
#
# Uso:
#   cd plugins-and-extensions/Audio2Midi/evaluation
#   bash render_mp3.sh
#
# Requisito: mscore (MuseScore 4) en PATH o en la ubicación por defecto.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSCORE_BIN="${MSCORE_BIN:-$(command -v mscore 2>/dev/null || echo '')}"

if [[ -z "$MSCORE_BIN" ]]; then
    echo "ERROR: MuseScore (mscore) no encontrado."
    echo "Instalar: brew install --cask musescore"
    echo "O bien:   MSCORE_BIN=/path/to/mscore bash render_mp3.sh"
    exit 1
fi

total=0
skipped=0

for mid in "$ROOT"/{yourmt3,compound,compound_v2,miros}/test*/transcribed_cuda.mid; do
    [[ -f "$mid" ]] || continue
    mp3="${mid%.mid}.mp3"
    if [[ -f "$mp3" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    test_dir="$(basename "$(dirname "$mid")")"
    pipeline="$(basename "$(dirname "$(dirname "$mid")")")"
    echo "  rendering ${pipeline}/${test_dir}/transcribed_cuda.mid …"
    "$MSCORE_BIN" -o "$mp3" "$mid" 2>/dev/null && total=$((total + 1))
done

echo ""
echo "Renderizados: $total  |  Ya existían: $skipped"
echo ""
echo "Siguiente paso:"
echo "  bash fetch_ground_truth.sh   # descarga GT Slakh + MusicNet (~39 MB)"
echo "  cd ../research && uv run python ../evaluation/compute_f1.py"
