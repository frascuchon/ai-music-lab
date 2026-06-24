#!/usr/bin/env bash
# Descarga los 10 audios de demo del HuggingFace Space mimbres/YourMT3
# y los coloca en los directorios de evaluación correctos.
#
# Uso:
#   cd plugins-and-extensions/Audio2Midi/evaluation
#   bash fetch_fixtures.sh
#
# Requisitos: curl (preinstalado en macOS)

set -euo pipefail

SPACE_BASE="https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/examples"
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

download() {
    local url="$1"
    local dest="$2"
    local filename
    filename="$(basename "$dest")"

    if [[ -f "$dest" ]]; then
        echo "[skip] Ya existe: $dest"
        return
    fi

    echo "[download] $filename → $(dirname "$dest")/"
    curl -fL --progress-bar -o "$dest" "$url"
    echo "[ok] $(du -h "$dest" | cut -f1) — $dest"
}

echo "=== Descargando fixtures de evaluación Audio2Midi ==="
echo "Fuente: $SPACE_BASE"
echo ""

# ---------------------------------------------------------------------------
# YourMT3 tests
# ---------------------------------------------------------------------------
echo "--- YourMT3 (10 tests) ---"

download \
    "${SPACE_BASE}/00_Funk1-97-C_comp_mic_pshift-3.wav" \
    "${EVAL_DIR}/yourmt3/test01/input.wav"

download \
    "${SPACE_BASE}/MAPS_MUS-chpn-e01_ENSTDkCl.wav" \
    "${EVAL_DIR}/yourmt3/test02/input.wav"

download \
    "${SPACE_BASE}/MAPS_MUS-scn15_11_ENSTDkAm.wav" \
    "${EVAL_DIR}/yourmt3/test03/input.wav"

download \
    "${SPACE_BASE}/Slakh_test_1884.wav" \
    "${EVAL_DIR}/yourmt3/test04/input.wav"

download \
    "${SPACE_BASE}/Slakh_test_1975.wav" \
    "${EVAL_DIR}/yourmt3/test05/input.wav"

download \
    "${SPACE_BASE}/mirst493.wav" \
    "${EVAL_DIR}/yourmt3/test06/input.wav"

download \
    "${SPACE_BASE}/musicnet2556.wav" \
    "${EVAL_DIR}/yourmt3/test07/input.wav"

download \
    "${SPACE_BASE}/musicnet_2628.wav" \
    "${EVAL_DIR}/yourmt3/test08/input.wav"

download \
    "${SPACE_BASE}/rwc087.wav" \
    "${EVAL_DIR}/yourmt3/test09/input.wav"

download \
    "${SPACE_BASE}/rwc089.mp3" \
    "${EVAL_DIR}/yourmt3/test10/input.mp3"

# ---------------------------------------------------------------------------
# Compound pipeline tests (symlinks a los mismos audios)
# ---------------------------------------------------------------------------
echo ""
echo "--- Pipeline compuesto (5 tests, mismos audios) ---"

link_or_copy() {
    local src="$1"
    local dst="$2"
    if [[ -f "$dst" ]]; then
        echo "[skip] Ya existe: $dst"
        return
    fi
    if [[ -f "$src" ]]; then
        ln -s "$src" "$dst"
        echo "[link] $dst → $src"
    else
        echo "[warn] No existe fuente: $src — skipping"
    fi
}

link_or_copy \
    "${EVAL_DIR}/yourmt3/test01/input.wav" \
    "${EVAL_DIR}/compound/test01/input.wav"

link_or_copy \
    "${EVAL_DIR}/yourmt3/test02/input.wav" \
    "${EVAL_DIR}/compound/test02/input.wav"

link_or_copy \
    "${EVAL_DIR}/yourmt3/test04/input.wav" \
    "${EVAL_DIR}/compound/test03/input.wav"

link_or_copy \
    "${EVAL_DIR}/yourmt3/test09/input.wav" \
    "${EVAL_DIR}/compound/test04/input.wav"

link_or_copy \
    "${EVAL_DIR}/yourmt3/test06/input.wav" \
    "${EVAL_DIR}/compound/test05/input.wav"

# ---------------------------------------------------------------------------
# MIROS tests (symlinks a los mismos audios que YourMT3)
# ---------------------------------------------------------------------------
echo ""
echo "--- MIROS (10 tests, mismos audios que YourMT3) ---"

for i in 01 02 03 04 05 06 07 08 09; do
    link_or_copy \
        "${EVAL_DIR}/yourmt3/test${i}/input.wav" \
        "${EVAL_DIR}/miros/test${i}/input.wav"
done

# test10 es mp3
link_or_copy \
    "${EVAL_DIR}/yourmt3/test10/input.mp3" \
    "${EVAL_DIR}/miros/test10/input.mp3"

# ---------------------------------------------------------------------------
# Compound v2 tests (symlinks a los tests con ground truth de YourMT3)
# ---------------------------------------------------------------------------
echo ""
echo "--- Compound v2 (4 tests con GT, mismos audios que YourMT3) ---"

for src_test in test04 test05 test07 test08; do
    dst="${EVAL_DIR}/compound_v2/${src_test}"
    mkdir -p "$dst"
    link_or_copy \
        "${EVAL_DIR}/yourmt3/${src_test}/input.wav" \
        "${dst}/input.wav"
done

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
echo ""
echo "=== Fixtures listos ==="
echo ""
echo "YourMT3 (10 tests):"
for d in "${EVAL_DIR}/yourmt3"/test*/; do
    wav="${d}input.wav"
    mp3="${d}input.mp3"
    if [[ -f "$wav" ]]; then
        echo "  $(basename "$d"): ✓ input.wav ($(du -h "$wav" | cut -f1))"
    elif [[ -f "$mp3" ]]; then
        echo "  $(basename "$d"): ✓ input.mp3 ($(du -h "$mp3" | cut -f1))"
    else
        echo "  $(basename "$d"): ✗ falta input.wav"
    fi
done

echo ""
echo "Compound pipeline (5 tests):"
for d in "${EVAL_DIR}/compound"/test*/; do
    wav="${d}input.wav"
    if [[ -f "$wav" ]]; then
        echo "  $(basename "$d"): ✓ input.wav"
    else
        echo "  $(basename "$d"): ✗ falta input.wav"
    fi
done

echo ""
echo "MIROS (10 tests):"
for d in "${EVAL_DIR}/miros"/test*/; do
    wav="${d}input.wav"
    mp3="${d}input.mp3"
    if [[ -f "$wav" ]]; then
        echo "  $(basename "$d"): ✓ input.wav (symlink)"
    elif [[ -f "$mp3" ]]; then
        echo "  $(basename "$d"): ✓ input.mp3 (symlink)"
    else
        echo "  $(basename "$d"): ✗ falta input"
    fi
done

echo ""
echo "Compound v2 (4 tests con GT):"
for d in "${EVAL_DIR}/compound_v2"/test*/; do
    wav="${d}input.wav"
    if [[ -f "$wav" ]]; then
        echo "  $(basename "$d"): ✓ input.wav (symlink)"
    else
        echo "  $(basename "$d"): ✗ falta input.wav"
    fi
done

echo ""
echo "Siguiente paso:"
echo "  cd ../research"
echo "  modal run research_yourmt3_modal.py::setup              # descarga pesos (una vez)"
echo "  modal run research_yourmt3_modal.py::eval_all           # transcribe los 10 tests"
echo "  modal run research_compound_pipeline_modal.py::eval_all # pipeline compuesto v1"
echo "  modal run research_miros_modal.py::setup                # descarga pesos MIROS (una vez)"
echo "  modal run research_miros_modal.py::eval_all             # transcribe los 10 tests MIROS"
echo "  modal run research_adtof_modal.py::setup                # verifica ADTOF"
echo "  modal run research_compound_v2_modal.py::eval_all --only 4,5,7,8  # compound v2"
