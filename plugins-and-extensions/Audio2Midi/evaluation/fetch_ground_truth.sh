#!/usr/bin/env bash
# Descarga el ground truth MIDI para los tests con GT disponible:
#   - Slakh2100 redux MIDIs (Track01884 y Track01975) → ~36.7 MB total
#   - MusicNet MIDIs (2556 y 2628) → ~2.6 MB total
#
# Uso:
#   cd plugins-and-extensions/Audio2Midi/evaluation
#   bash fetch_ground_truth.sh
#
# Requisitos: curl, tar (ambos preinstalados en macOS)
# El directorio _ground_truth/ está en .gitignore (cache binaria).

set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GT_DIR="${EVAL_DIR}/_ground_truth"
SLAKH_DIR="${GT_DIR}/slakh"
MN_DIR="${GT_DIR}/musicnet"

mkdir -p "$SLAKH_DIR" "$MN_DIR"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
download_if_missing() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "[skip] Ya existe: $(basename "$dest")"
    else
        echo "[download] $(basename "$dest") …"
        curl -fL --progress-bar -o "$dest" "$url"
        echo "[ok] $(du -h "$dest" | cut -f1) — $dest"
    fi
}

echo "=== Descargando ground truth para evaluación F1 ==="
echo ""

# ---------------------------------------------------------------------------
# Slakh2100 redux MIDIs (36.7 MB — solo MIDIs del split redux)
# Fuente: huggingface.co/datasets/projectlosangeles/Slakh2100
# Cubre: yourmt3/test04 (Track01884), yourmt3/test05 (Track01975),
#         compound/test03 (Track01884, mismo que test04)
# ---------------------------------------------------------------------------
echo "--- Slakh2100 redux MIDIs ---"
SLAKH_TAR="${GT_DIR}/slakh2100_redux-mix_midis_only.tar.gz"
download_if_missing \
    "https://huggingface.co/datasets/projectlosangeles/Slakh2100/resolve/main/slakh2100_redux-mix_midis_only.tar.gz" \
    "$SLAKH_TAR"

# Verificar layout del tarball antes de extraer
echo "[info] Inspeccionando layout del tarball …"
SLAKH_PREFIX="$(tar -tzf "$SLAKH_TAR" 2>/dev/null | head -5 | head -1 | cut -d/ -f1)"
echo "[info] Prefijo detectado: ${SLAKH_PREFIX:-<raíz>}"

# Extraer Track01884 y Track01975 (idempotente)
for track in Track01884 Track01975; do
    if [[ -f "${SLAKH_DIR}/${track}/all_src.mid" ]]; then
        echo "[skip] Ya extraído: ${track}"
        continue
    fi
    echo "[extract] ${track} …"
    mkdir -p "${SLAKH_DIR}/${track}"
    # Intentar con prefijo detectado; si falla, extraer todo y mover
    if tar -tzf "$SLAKH_TAR" 2>/dev/null | grep -q "/${track}/"; then
        tar -xzf "$SLAKH_TAR" -C "$SLAKH_DIR" \
            --strip-components=3 \
            --include="*/${track}/*" 2>/dev/null || \
        tar -xzf "$SLAKH_TAR" -C "${SLAKH_DIR}/${track}" \
            --wildcards "*/${track}/*" \
            --strip-components=4 2>/dev/null || true
    fi
    # Fallback: extraer todo en un temporal y mover
    if [[ ! -f "${SLAKH_DIR}/${track}/all_src.mid" ]]; then
        TMP_DIR="$(mktemp -d)"
        tar -xzf "$SLAKH_TAR" -C "$TMP_DIR" 2>/dev/null || true
        found="$(find "$TMP_DIR" -type d -name "$track" 2>/dev/null | head -1)"
        if [[ -n "$found" ]]; then
            cp -r "$found/." "${SLAKH_DIR}/${track}/"
            echo "[ok] Extraído via fallback: ${track}"
        else
            echo "[warn] No se encontró ${track} en el tarball"
        fi
        rm -rf "$TMP_DIR"
    else
        echo "[ok] Extraído: ${track}"
    fi
done

# ---------------------------------------------------------------------------
# MusicNet MIDIs (2.6 MB — todos los MIDIs del set)
# Fuente: Zenodo record 5120004
# Cubre: yourmt3/test07 (2556), yourmt3/test08 (2628)
# ---------------------------------------------------------------------------
echo ""
echo "--- MusicNet MIDIs ---"
MN_TAR="${GT_DIR}/musicnet_midis.tar.gz"
download_if_missing \
    "https://zenodo.org/records/5120004/files/musicnet_midis.tar.gz?download=1" \
    "$MN_TAR"

for rec_id in 2556 2628; do
    dest="${MN_DIR}/${rec_id}.mid"
    if [[ -f "$dest" ]]; then
        echo "[skip] Ya existe: ${rec_id}.mid"
        continue
    fi
    echo "[extract] MusicNet ${rec_id}.mid …"
    TMP_DIR="$(mktemp -d)"
    tar -xzf "$MN_TAR" -C "$TMP_DIR" 2>/dev/null || true
    found="$(find "$TMP_DIR" -name "${rec_id}*.mid" 2>/dev/null | head -1)"
    if [[ -n "$found" ]]; then
        cp "$found" "$dest"
        echo "[ok] $(du -h "$dest" | cut -f1) — $dest"
    else
        echo "[warn] No se encontró ${rec_id}*.mid en el tarball"
    fi
    rm -rf "$TMP_DIR"
done

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
echo ""
echo "=== Estado del ground truth ==="
echo ""
echo "Slakh2100:"
for track in Track01884 Track01975; do
    mid="${SLAKH_DIR}/${track}/all_src.mid"
    if [[ -f "$mid" ]]; then
        echo "  ${track}: ✓ all_src.mid ($(du -h "$mid" | cut -f1))"
    else
        echo "  ${track}: ✗ no encontrado — revisar fetch_ground_truth.sh"
    fi
done
echo ""
echo "MusicNet:"
for rec_id in 2556 2628; do
    mid="${MN_DIR}/${rec_id}.mid"
    if [[ -f "$mid" ]]; then
        echo "  ${rec_id}.mid: ✓ ($(du -h "$mid" | cut -f1))"
    else
        echo "  ${rec_id}.mid: ✗ no encontrado"
    fi
done

echo ""
echo "Siguiente paso:"
echo "  cd ../research"
echo "  uv run python ../evaluation/compute_f1.py"
