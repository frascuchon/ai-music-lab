#!/usr/bin/env bash
# Regenera todos los tests del benchmark Amadeus.
#
# Uso:
#   bash evaluation/amadeus/regenerate_all.sh
#   ONLY=1,3,5 bash evaluation/amadeus/regenerate_all.sh
#   TEMPERATURE=1.5 bash evaluation/amadeus/regenerate_all.sh
#   FORCE=1 bash evaluation/amadeus/regenerate_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESEARCH_DIR="$(cd "$SCRIPT_DIR/../../../research" && pwd)"
EVAL_DIR="$SCRIPT_DIR"

TEMPERATURE="${TEMPERATURE:-1.25}"
THRESHOLD="${THRESHOLD:-0.99}"
GENERATION_LENGTH="${GENERATION_LENGTH:-1024}"
N_OUTPUTS="${N_OUTPUTS:-2}"
FORCE="${FORCE:-}"

cd "$RESEARCH_DIR"

ONLY_ARG=""
if [[ -n "$ONLY" ]]; then
    ONLY_ARG="--only $ONLY"
fi

FORCE_ARG=""
if [[ -n "$FORCE" ]]; then
    FORCE_ARG="--force"
fi

echo "[regenerate_all] eval_dir: $EVAL_DIR"
echo "[regenerate_all] temperature=$TEMPERATURE  threshold=$THRESHOLD  generation_length=$GENERATION_LENGTH"
echo "[regenerate_all] n_outputs=$N_OUTPUTS"
[[ -n "$ONLY" ]] && echo "[regenerate_all] only: $ONLY"
[[ -n "$FORCE" ]] && echo "[regenerate_all] force: sí"
echo ""

modal run research_amadeus_modal.py::eval_all \
    --eval-dir "$EVAL_DIR" \
    --temperature "$TEMPERATURE" \
    --threshold "$THRESHOLD" \
    --generation-length "$GENERATION_LENGTH" \
    --n-outputs "$N_OUTPUTS" \
    $ONLY_ARG \
    $FORCE_ARG

echo ""
echo "[regenerate_all] Completado."
