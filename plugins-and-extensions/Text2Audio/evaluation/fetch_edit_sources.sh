#!/usr/bin/env bash
# Descarga los audios FUENTE compartidos del benchmark de edición/versionado audio2audio.
#
# Los audios provienen de los assets oficiales del repo audiocraft (Meta), que son los
# ficheros usados por los demos oficiales de MusicGen-melody y MelodyFlow:
#   - bach.mp3           → demo oficial de condicionamiento por melodía (MusicGen)
#   - bolero_ravel.mp3   → ejemplo oficial de EDICIÓN del demo de MelodyFlow
#                          (par verbatim: bolero + "A cheerful country song with acoustic guitars")
#   - electronic.mp3     → asset de demos de audiocraft (MusicGen-Style)
#   - CJ_Beatbox_Loop_05_90.wav → loop de beatbox a 90 BPM (asset de demos audiocraft)
#
# Además copia un quinto source generado por nosotros en la sesión S1:
#   - stable_audio_open/prompt10/output.wav (jazz guitar loop, 100 BPM)
#     → caso de uso real del plugin: editar un sample generado por Text2Audio.
#
# Uso:
#   cd plugins-and-extensions/Text2Audio/evaluation
#   bash fetch_edit_sources.sh
#
# Destino: evaluation/edit/source_audio/   (gitignored — evaluation/**/*.{wav,mp3})
# Los ids/rutas están registrados en evaluation/prompts_edit.json → "sources".
#
# Fuente: https://github.com/facebookresearch/audiocraft/tree/main/assets

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${ROOT}/edit/source_audio"
AC_BASE="https://raw.githubusercontent.com/facebookresearch/audiocraft/main/assets"

echo "=== Descargando audios fuente del benchmark de edición ==="
echo "Fuente: ${AC_BASE}/"
echo "Destino: ${SRC_DIR}/"
echo ""

mkdir -p "${SRC_DIR}"

download() {
    local url="$1"
    local dest="$2"
    local filename
    filename="$(basename "$dest")"

    if [[ -f "$dest" ]]; then
        echo "[skip] Ya existe: ${filename}"
        return
    fi

    echo "[download] ${filename} …"
    curl -fL --progress-bar -o "$dest" "$url"
    echo "[ok] $(du -h "$dest" | cut -f1) — ${filename}"
}

# ---------------------------------------------------------------------------
# Assets oficiales de audiocraft (MIT repo; audios de demo)
# ---------------------------------------------------------------------------
download "${AC_BASE}/bach.mp3"                   "${SRC_DIR}/bach.mp3"
download "${AC_BASE}/bolero_ravel.mp3"           "${SRC_DIR}/bolero_ravel.mp3"
download "${AC_BASE}/electronic.mp3"             "${SRC_DIR}/electronic.mp3"
download "${AC_BASE}/CJ_Beatbox_Loop_05_90.wav"  "${SRC_DIR}/beatbox_loop_90bpm.wav"

# ---------------------------------------------------------------------------
# Source propio: jazz guitar loop generado con SAO 1.0 en la sesión S1
# ---------------------------------------------------------------------------
SAO_GUITAR="${ROOT}/stable_audio_open/prompt10/output.wav"
DEST_GUITAR="${SRC_DIR}/sao_guitar_loop.wav"

if [[ -f "$DEST_GUITAR" ]]; then
    echo "[skip] Ya existe: sao_guitar_loop.wav"
elif [[ -f "$SAO_GUITAR" ]]; then
    cp "$SAO_GUITAR" "$DEST_GUITAR"
    echo "[ok] $(du -h "$DEST_GUITAR" | cut -f1) — sao_guitar_loop.wav (copiado de stable_audio_open/prompt10/)"
else
    echo "[warn] No existe ${SAO_GUITAR}."
    echo "       Genera prompt10 con research_stable_audio_open_modal.py::eval_all --only prompt10"
    echo "       y vuelve a ejecutar este script. Los casos src05_* quedarán pendientes."
fi

# ---------------------------------------------------------------------------
# Resumen con ffprobe (si está disponible)
# ---------------------------------------------------------------------------
echo ""
echo "=== Audios fuente disponibles ==="
for f in "${SRC_DIR}"/*; do
    [[ -f "$f" ]] || continue
    if command -v ffprobe >/dev/null 2>&1; then
        info="$(ffprobe -v error -show_entries format=duration -show_entries stream=sample_rate,channels \
                -of default=noprint_wrappers=1 "$f" 2>/dev/null | tr '\n' ' ')"
        echo "  $(basename "$f")  (${info})"
    else
        echo "  $(basename "$f")  ($(du -h "$f" | cut -f1))"
    fi
done

echo ""
echo "Siguiente paso: revisar evaluation/prompts_edit.json (sección \"sources\") y lanzar"
echo "el smoke test de la sesión E1 (ACE-Step):"
echo "  cd ../research"
echo "  modal run research_acestep_edit_modal.py::setup"
echo "  modal run research_acestep_edit_modal.py::smoke"
