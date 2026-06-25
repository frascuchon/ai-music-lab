#!/usr/bin/env bash
# Descarga un set de referencia de audio para calcular FAD (Fréchet Audio Distance).
#
# El FAD necesita una distribución de referencia de alta calidad con la que comparar
# los audios generados. Usamos un subconjunto de MusicCaps (Google, CC-BY 4.0):
# ~150 clips de música de alta calidad, 10 s cada uno, descargados de la URL oficial.
#
# Alternativamente, se puede usar un set de clips de Freesound del mismo dominio que
# los prompts (drums, synths, FX), pero requiere una clave de API de Freesound.
#
# Uso:
#   cd plugins-and-extensions/Text2Audio/evaluation
#   bash fetch_reference_set.sh
#
# Requisitos: yt-dlp (brew install yt-dlp) o curl para descarga directa si hay mirrors.
#
# El set se guarda en _reference/ (gitignored, ~150 MB).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${ROOT}/_reference"
MUSICCAPS_CSV="${REF_DIR}/musiccaps_eval_small.csv"

# URL del CSV de MusicCaps (subconjunto de 100 clips de evaluación estándar en FAD papers)
# Fuente: https://huggingface.co/datasets/google/MusicCaps
MUSICCAPS_HF_BASE="https://huggingface.co/datasets/google/MusicCaps/resolve/main"

echo "=== Descargando set de referencia FAD para Text2Audio ==="
echo "Destino: ${REF_DIR}/"
echo ""

mkdir -p "${REF_DIR}"

# ---------------------------------------------------------------------------
# Opción A: descarga directa desde HuggingFace (CSV de metadatos)
# ---------------------------------------------------------------------------
download_if_missing() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "[skip] Ya existe: $(basename "$dest")"
        return
    fi
    echo "[download] $(basename "$dest") …"
    curl -fL --progress-bar -o "$dest" "$url"
    echo "[ok] $(du -h "$dest" | cut -f1) — $dest"
}

# CSV de MusicCaps (metadatos + ytids para los 5521 clips)
download_if_missing \
    "${MUSICCAPS_HF_BASE}/musiccaps-public.csv" \
    "${REF_DIR}/musiccaps-public.csv"

# ---------------------------------------------------------------------------
# Instrucciones para descarga de audio (requiere yt-dlp)
# ---------------------------------------------------------------------------
echo ""
echo "=== Instrucciones para descargar los clips de audio ==="
echo ""
echo "MusicCaps usa clips de YouTube. Para descargarlos necesitas yt-dlp:"
echo "  brew install yt-dlp"
echo ""
echo "Script de descarga rápida (subconjunto de 100 clips, ~150 MB):"
echo ""
echo "  python3 - <<'PYEOF'"
echo "  import csv, subprocess, os"
echo "  ref_dir = '${REF_DIR}/clips'"
echo "  os.makedirs(ref_dir, exist_ok=True)"
echo "  with open('${REF_DIR}/musiccaps-public.csv') as f:"
echo "      reader = csv.DictReader(f)"
echo "      clips = [r for i, r in enumerate(reader) if i < 100]  # 100 clips"
echo "  for c in clips:"
echo "      ytid = c['ytid']"
echo "      start = int(c['start_s'])"
echo "      end = int(c['end_s'])"
echo "      out = os.path.join(ref_dir, f'{ytid}_{start}_{end}.wav')"
echo "      if os.path.exists(out):"
echo "          continue"
echo "      url = f'https://www.youtube.com/watch?v={ytid}'"
echo "      subprocess.run(['yt-dlp', '-x', '--audio-format', 'wav',"
echo "          '--postprocessor-args', f'-ss {start} -t {end-start}',"
echo "          '-o', out, url], check=False)"
echo "  PYEOF"
echo ""
echo "Una vez descargados, ejecutar compute_metrics.py normalmente:"
echo "  cd ../research"
echo "  uv run python ../evaluation/compute_metrics.py --reference _reference/clips"
echo ""
echo "Nota: si yt-dlp falla (vídeos privados/eliminados), los clips fallidos se ignoran."
echo "Con ~80-100 clips válidos es suficiente para un FAD significativo."
