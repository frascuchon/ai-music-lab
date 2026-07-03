#!/usr/bin/env bash
# Genera MIDI localmente (MPS) para todos los tests text2midi.
#
# Nota: las generaciones MPS pueden diferir numéricamente de CUDA (como en MIDI-LLM).
# Para comparación definitiva contra la referencia oficial, usar research_text2midi_modal.py.
#
# Uso:
#   bash regenerate_all.sh                 # salta tests con v0 ya existente
#   bash regenerate_all.sh --force         # sobreescribe
#   bash regenerate_all.sh --render-mp3    # también renderiza MP3 con MuseScore
#   ONLY=1,4 bash regenerate_all.sh        # solo tests específicos
#
# Parámetros oficiales (HuggingFace Space):
#   temperature=0.9  max_len=2000  float32 (--no-half)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RESEARCH_DIR="$ROOT/../../research"
SCRIPT="$RESEARCH_DIR/research_text2midi.py"

RENDER_MP3=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --render-mp3) RENDER_MP3=1 ;;
    --force)      FORCE=1 ;;
  esac
done

if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -ra TESTS <<< "$ONLY"
else
  TESTS=($(seq 1 7))
fi

MSCORE_BIN="${MSCORE_BIN:-$(command -v mscore 2>/dev/null || echo '')}"

echo "=== text2midi local MPS regeneration ==="
echo "Script:     $SCRIPT"
echo "Render MP3: $RENDER_MP3  Force: $FORCE"
echo "n_outputs=2  temperature=0.9  max_len=2000  --no-half"
echo ""

for n in "${TESTS[@]}"; do
  dir="$ROOT/test$n"
  out_v0="$dir/generated_mps_v0.mid"

  if [[ ! -d "$dir" ]]; then
    echo "[test$n] WARN: test$n no encontrado, saltando"; continue
  fi

  if [[ ! -f "$dir/prompt.txt" ]]; then
    echo "[test$n] WARN: sin prompt.txt, saltando"; continue
  fi

  if [[ -f "$out_v0" && "$FORCE" -eq 0 ]]; then
    echo "[test$n] skip (v0 ya existe — usa --force para sobreescribir)"
    continue
  fi

  rm -f "$dir"/generated_mps_v*.mid "$dir"/generated_mps_v*.mp3

  prompt="$(grep '^Prompt:' "$dir/prompt.txt" | sed 's/^Prompt: *"//;s/"$//')"
  echo "[test$n] Generando 2 outputs..."
  echo "         Prompt: ${prompt:0:100}..."

  start="$(date +%s)"
  (cd "$RESEARCH_DIR" && uv run research_text2midi.py \
      --prompt "$prompt" \
      --out "$out_v0" \
      --n_outputs 2 \
      --temperature 0.9 \
      --max_len 2000 \
      --no-half)
  elapsed=$(( $(date +%s) - start ))
  echo "[test$n] Done in ${elapsed}s"

  if [[ "$RENDER_MP3" -eq 1 && -n "$MSCORE_BIN" ]]; then
    for mid in "$dir"/generated_mps_v*.mid; do
      mp3="${mid%.mid}.mp3"
      echo "  rendering $(basename "$mid") → $(basename "$mp3")..."
      "$MSCORE_BIN" -o "$mp3" "$mid" 2>/dev/null
    done
  elif [[ "$RENDER_MP3" -eq 1 ]]; then
    echo "[test$n] WARN: MuseScore no encontrado, saltando MP3 render"
  fi
done

echo ""
echo "Regeneración completa."
