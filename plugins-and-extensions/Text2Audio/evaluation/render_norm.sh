#!/usr/bin/env bash
# Convierte todos los output.wav → output.mp3 normalizados para escucha en REAPER.
# Idempotente: si el MP3 ya existe lo omite.
#
# Uso:
#   cd plugins-and-extensions/Text2Audio/evaluation
#   bash render_norm.sh
#
# Requisitos: ffmpeg en PATH (brew install ffmpeg)
#
# Normalización: loudnorm EBU R128 a -14 LUFS (estándar streaming), con
# limiter a -1.0 dBFS para evitar clipping. Compatibible con REAPER y DAWs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG_BIN="${FFMPEG_BIN:-$(command -v ffmpeg 2>/dev/null || echo '')}"

if [[ -z "$FFMPEG_BIN" ]]; then
    echo "ERROR: ffmpeg no encontrado."
    echo "Instalar: brew install ffmpeg"
    echo "O bien:   FFMPEG_BIN=/path/to/ffmpeg bash render_norm.sh"
    exit 1
fi

total=0
skipped=0

# Iterar sobre todos los output.wav en subcarpetas de modelos conocidos (profundidad 1 y 2)
# Profundidad 1: evaluation/<model>/<prompt_id>/output.wav  (eval_all)
# Profundidad 2: evaluation/<model>/smoke/<smoke_id>/output.wav  (smoke test)
while IFS= read -r wav; do
    [[ -f "$wav" ]] || continue
    mp3="${wav%.wav}.mp3"
    if [[ -f "$mp3" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    # Extraer ruta relativa desde ROOT para el log
    rel="${wav#"${ROOT}/"}"
    echo "  rendering ${rel} …"
    "$FFMPEG_BIN" -i "$wav" \
        -af "loudnorm=I=-14:LRA=11:TP=-1.0" \
        -codec:a libmp3lame -qscale:a 2 \
        "$mp3" -y -loglevel error \
        && total=$((total + 1))
done < <(find "$ROOT" -maxdepth 5 \
         \( -path "*/stable_audio_open/*" \
            -o -path "*/stable_audio_open_small/*" \
            -o -path "*/musicgen/*" \
            -o -path "*/magnet/*" \
            -o -path "*/audiogen/*" \
         \) -name "output.wav" 2>/dev/null | sort)

echo ""
echo "Renderizados: $total  |  Ya existían: $skipped"
echo ""
echo "Siguiente paso:"
echo "  bash fetch_reference_set.sh       # descarga set de referencia FAD (~150 MB)"
echo "  cd ../research"
echo "  uv run python ../evaluation/compute_metrics.py"
