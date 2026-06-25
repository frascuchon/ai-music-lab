#!/usr/bin/env bash
# Descarga los ejemplos oficiales de audio de Foundation-1 (RoyalCities).
#
# Foundation-1 tiene archivos de audio de referencia PÚBLICOS en su repo HuggingFace
# (no gated, sin token necesario). Estos son los únicos demos oficiales descargables
# directamente entre todos los candidatos evaluados.
#
# Uso:
#   cd plugins-and-extensions/Text2Audio/evaluation
#   bash fetch_foundation1_demos.sh
#
# Destino: evaluation/foundation_1/demos/
# Los archivos se usan como benchmark de referencia en el smoke test de la Sesión S2:
# correr research_foundation1_modal.py con los mismos prompts y comparar cualitativamente.
#
# Fuente: https://huggingface.co/RoyalCities/Foundation-1/tree/main/examples

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMOS_DIR="${ROOT}/foundation_1/demos"
HF_BASE="https://huggingface.co/RoyalCities/Foundation-1/resolve/main/examples"

echo "=== Descargando demos oficiales de Foundation-1 ==="
echo "Fuente: ${HF_BASE}/"
echo "Destino: ${DEMOS_DIR}/"
echo ""

mkdir -p "${DEMOS_DIR}"

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
# Pares de comparación: versión A (sin fine-tune SAO 1.0) y B (Foundation-1)
# Permite escuchar directamente la ganancia del fine-tuning.
# ---------------------------------------------------------------------------

echo "--- Pares A/B de comparación ---"

download "${HF_BASE}/compare_example_1_a.mp3" "${DEMOS_DIR}/compare_example_1_a__sao_base.mp3"
download "${HF_BASE}/compare_example_1_b.mp3" "${DEMOS_DIR}/compare_example_1_b__foundation1.mp3"

download "${HF_BASE}/compare_example_2_a.mp3" "${DEMOS_DIR}/compare_example_2_a__sao_base.mp3"
download "${HF_BASE}/compare_example_2_b.mp3" "${DEMOS_DIR}/compare_example_2_b__foundation1.mp3"

download "${HF_BASE}/compare_example_3_a.mp3" "${DEMOS_DIR}/compare_example_3_a__sao_base.mp3"
download "${HF_BASE}/compare_example_3_b.mp3" "${DEMOS_DIR}/compare_example_3_b__foundation1.mp3"

# ---------------------------------------------------------------------------
# Convención de nombres:
#   *_sao_base.mp3      — mismo prompt generado con SAO 1.0 sin fine-tune
#   *_foundation1.mp3   — mismo prompt generado con Foundation-1 (fine-tune)
# Permite escuchar directamente la ganancia del dominio acid/house.
# ---------------------------------------------------------------------------

echo ""
echo "=== Demos descargados ==="
for f in "${DEMOS_DIR}"/*.mp3; do
    [[ -f "$f" ]] || continue
    echo "  $(basename "$f")  ($(du -h "$f" | cut -f1))"
done

echo ""
echo "Cómo usar estos demos como benchmark (Sesión S2):"
echo ""
echo "  1. Escuchar los pares A/B para entender la diferencia SAO base vs Foundation-1."
echo "     Los archivos *_foundation1.mp3 son la calidad que debería alcanzar nuestro script."
echo ""
echo "  2. Ejecutar smoke test con el mismo prompt:"
echo "     cd ../research"
echo "     modal run research_foundation1_modal.py::setup"
echo "     modal run research_foundation1_modal.py::main \\"
echo '         --prompt "Bass, FM Bass, Medium Delay, Medium Reverb, Phaser, Acid, Gritty, Dubstep, 8 Bars, 140 BPM, E minor" \\'
echo "         --seconds 13.71 \\"
echo "         --out-dir ../evaluation/foundation_1/smoke"
echo ""
echo "  3. Comparar evaluation/foundation_1/smoke/output.wav con compare_example_1_b__foundation1.mp3"
echo "     Si la calidad es comparable, el script funciona correctamente."
echo ""
echo "  4. Lanzar eval completo con prompts de formato tag:"
echo "     modal run research_foundation1_modal.py::eval_all \\"
echo "         --prompts-json ../evaluation/prompts_official.json \\"
echo "         --model-section foundation_1"
