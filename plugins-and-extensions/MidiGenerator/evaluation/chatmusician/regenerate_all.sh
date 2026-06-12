#!/usr/bin/env bash
# Genera MIDI en Modal (CUDA A10G) para todos los tests ChatMusician.
#
# Cada test produce generated_cuda_v0.mid/.abc + generated_cuda_v1.mid/.abc
# El post-procesado ABC→MIDI ocurre dentro del container Modal (abc2midi).
#
# Uso:
#   bash regenerate_all.sh                 # salta tests con v0 ya existente
#   bash regenerate_all.sh --force         # sobreescribe
#   bash regenerate_all.sh --render-mp3    # también renderiza MP3 con MuseScore
#   ONLY=1,4,10 bash regenerate_all.sh     # solo tests específicos
#
# Parámetros (fijos, verbatim del model card):
#   temperature=0.2  top_k=40  top_p=0.9  repetition_penalty=1.1  max_new_tokens=1536
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RESEARCH_DIR="$ROOT/../../research"
MODAL_SCRIPT="$RESEARCH_DIR/research_chatmusician_modal.py"

RENDER_MP3=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --render-mp3) RENDER_MP3=1 ;;
    --force)      FORCE=1 ;;
  esac
done

FORCE_FLAG=""
[[ "$FORCE" -eq 1 ]] && FORCE_FLAG="--force"

if [[ -n "${ONLY:-}" ]]; then
  IFS=',' read -ra RAW_TESTS <<< "$ONLY"
  TESTS=()
  for n in "${RAW_TESTS[@]}"; do
    TESTS+=("$(printf '%02d' "$n")")
  done
else
  TESTS=(01 02 03 04 05 06 07 08 09 10 11 12)
fi

MSCORE_BIN="${MSCORE_BIN:-$(command -v mscore 2>/dev/null || echo '')}"

echo "=== ChatMusician Modal CUDA regeneration ==="
echo "Script:     $MODAL_SCRIPT"
echo "Render MP3: $RENDER_MP3  Force: $FORCE"
echo "n_outputs=2  model=m-a-p/ChatMusician  gpu=A10G  fp16"
echo ""

for n in "${TESTS[@]}"; do
  dir="$ROOT/test${n}"
  out_v0="$dir/generated_cuda_v0.mid"

  if [[ ! -d "$dir" ]]; then
    echo "[test${n}] WARN: directorio no encontrado, saltando"; continue
  fi

  if [[ ! -f "$dir/prompt.txt" ]]; then
    echo "[test${n}] WARN: sin prompt.txt, saltando"; continue
  fi

  if [[ -f "$out_v0" && "$FORCE" -eq 0 ]]; then
    echo "[test${n}] skip (v0 ya existe — usa --force para sobreescribir)"
    continue
  fi

  rm -f "$dir"/generated_cuda_v*.mid "$dir"/generated_cuda_v*.abc "$dir"/generated_cuda_v*.mp3

  prompt="$(grep '^Prompt:' "$dir/prompt.txt" | sed 's/^Prompt: *"//;s/"$//')"
  echo "[test${n}] Generando 2 outputs..."
  echo "           Prompt: ${prompt:0:120}"

  # Detectar condicionante: MIDI tiene prioridad sobre ABC si ambos existen
  input_flag=""
  if [[ -f "$dir/input_midi.mid" ]]; then
    input_flag="--input-file $dir/input_midi.mid"
    echo "           + input_midi.mid (auto-convert MIDI→ABC)"
  elif [[ -f "$dir/input_abc.txt" ]]; then
    input_flag="--input-file $dir/input_abc.txt"
    echo "           + input_abc.txt"
  fi

  start="$(date +%s)"
  (cd "$RESEARCH_DIR" && modal run research_chatmusician_modal.py::main \
      --prompt "$prompt" \
      $input_flag \
      --out-dir "$dir" \
      --n-outputs 2 \
      $FORCE_FLAG)
  elapsed=$(( $(date +%s) - start ))
  echo "[test${n}] Done in ${elapsed}s"

  if [[ "$RENDER_MP3" -eq 1 && -n "$MSCORE_BIN" ]]; then
    for mid in "$dir"/generated_cuda_v*.mid; do
      mp3="${mid%.mid}.mp3"
      echo "  rendering $(basename "$mid") → $(basename "$mp3")..."
      "$MSCORE_BIN" -o "$mp3" "$mid" 2>/dev/null || echo "  WARN: MuseScore falló en $(basename "$mid")"
    done
  elif [[ "$RENDER_MP3" -eq 1 ]]; then
    echo "[test${n}] WARN: MuseScore no encontrado (MSCORE_BIN vacío), saltando MP3 render"
  fi
done

echo ""
echo "Regeneración completa."
echo ""
echo "Nota: para el benchmark completo en un solo job Modal usa eval_all:"
echo "  cd $RESEARCH_DIR && modal run research_chatmusician_modal.py::eval_all \\"
echo "      --eval-dir ../evaluation/chatmusician --n-outputs 2"
