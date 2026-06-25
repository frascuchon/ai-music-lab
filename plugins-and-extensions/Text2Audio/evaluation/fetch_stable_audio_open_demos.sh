#!/usr/bin/env bash
# Descarga los ejemplos de audio del model card de Stable Audio Open 1.0.
#
# El repo de SAO 1.0 es GATED — requiere HF_TOKEN con acceso read Y haber aceptado
# la licencia en https://huggingface.co/stabilityai/stable-audio-open-1.0
#
# Estos archivos son la referencia oficial para el smoke test (Sesión S1): ejecutar
# los mismos prompts del model card con nuestro script y comparar los outputs.
#
# Prerrequisitos:
#   1. pip install huggingface_hub>=0.24  (incluido en uv sync del entorno research/)
#   2. Aceptar licencia en https://huggingface.co/stabilityai/stable-audio-open-1.0
#   3. Crear token HF (read): https://huggingface.co/settings/tokens
#
# Uso:
#   cd plugins-and-extensions/Text2Audio/evaluation
#   HF_TOKEN=<tu_token> bash fetch_stable_audio_open_demos.sh
#
# Destino: evaluation/stable_audio_open/reference_demos/
#
# Alternativa sin token:
#   Los archivos compare_example_*_a__sao_base.mp3 de fetch_foundation1_demos.sh
#   son outputs de SAO 1.0 base (públicos, sin token):
#     bash fetch_foundation1_demos.sh
#   Notar que usan prompts de Foundation-1 (formato tag), NO los prompts del smoke test.
#   Son útiles para calibrar la calidad general del modelo, no para el smoke test S1.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMOS_DIR="${ROOT}/stable_audio_open/reference_demos"
REPO_ID="stabilityai/stable-audio-open-1.0"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== Descargando demos de referencia de Stable Audio Open 1.0 ==="
echo ""

if [[ -z "$HF_TOKEN" ]]; then
    echo "⚠️  HF_TOKEN no encontrado."
    echo ""
    echo "Stable Audio Open es un modelo GATED. Para descargar las demos oficiales:"
    echo "  1. Aceptar licencia en: https://huggingface.co/stabilityai/stable-audio-open-1.0"
    echo "  2. Crear token HF (read): https://huggingface.co/settings/tokens"
    echo "  3. Ejecutar:"
    echo "       HF_TOKEN=<token> bash fetch_stable_audio_open_demos.sh"
    echo ""
    echo "Alternativa sin token (SAO outputs para prompts de Foundation-1):"
    echo "  bash fetch_foundation1_demos.sh"
    echo "  → descarga compare_example_*_a__sao_base.mp3 (SAO base, sin fine-tune)"
    echo ""
    echo "Referencia manual (sin descarga):"
    echo "  https://huggingface.co/spaces/artificialguybr/Stable-Audio-Open-Zero"
    exit 1
fi

mkdir -p "${DEMOS_DIR}"

echo "Repo: ${REPO_ID}"
echo "Destino: ${DEMOS_DIR}/"
echo ""

# ---------------------------------------------------------------------------
# Descubrir y descargar archivos de audio del repo via huggingface_hub
# ---------------------------------------------------------------------------
DEMOS_DIR="${DEMOS_DIR}" HF_TOKEN="${HF_TOKEN}" python3 - <<'PYEOF'
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub no instalado.")
    print("Ejecutar desde research/: uv run pip install huggingface_hub>=0.24")
    sys.exit(1)

token = os.environ["HF_TOKEN"]
repo_id = "stabilityai/stable-audio-open-1.0"
dest_dir = Path(os.environ["DEMOS_DIR"])

api = HfApi()

# Listar archivos del repo
try:
    all_files = list(api.list_repo_files(repo_id, token=token))
except Exception as e:
    print(f"ERROR al acceder al repo '{repo_id}': {e}")
    print("")
    print("Verificar que:")
    print("  - El HF_TOKEN tiene acceso read en huggingface.co/settings/tokens")
    print("  - Has aceptado la licencia en: https://huggingface.co/stabilityai/stable-audio-open-1.0")
    sys.exit(1)

# Filtrar archivos de audio
audio_ext = {".wav", ".mp3", ".flac", ".ogg"}
audio_files = [f for f in all_files if Path(f).suffix.lower() in audio_ext]

print(f"Archivos en {repo_id}: {len(all_files)} total, {len(audio_files)} de audio")
print("")

if not audio_files:
    print("No se encontraron archivos de audio en el repo.")
    print("Los ejemplos del model card puede que estén embebidos como URLs externas")
    print("y no como archivos en el repo. Usar el Space para referencia manual:")
    print("  https://huggingface.co/spaces/artificialguybr/Stable-Audio-Open-Zero")
    sys.exit(0)

print("Archivos de audio encontrados:")
for f in audio_files:
    print(f"  {f}")
print("")

ok = 0
errors = 0
for audio_path in audio_files:
    fname = Path(audio_path).name
    dest = dest_dir / fname
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"  [skip] Ya existe: {fname}  ({size_mb:.1f} MB)")
        ok += 1
        continue
    print(f"  [download] {audio_path} …")
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=audio_path,
            local_dir=str(dest_dir),
            token=token,
        )
        # hf_hub_download puede crear subdirectorios — mover al nivel raíz de DEMOS_DIR
        downloaded = Path(local)
        if downloaded != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"  [ok]  {fname}  ({size_mb:.1f} MB)")
        ok += 1
    except Exception as e:
        print(f"  [error] {audio_path}: {e}")
        errors += 1

print("")
print(f"Descargados: {ok}  |  Errores: {errors}")
PYEOF

echo ""
echo "=== Archivos de referencia disponibles ==="
if ls "${DEMOS_DIR}"/* &>/dev/null 2>&1; then
    for f in "${DEMOS_DIR}"/*; do
        [[ -f "$f" ]] || continue
        echo "  $(basename "$f")  ($(du -h "$f" | cut -f1))"
    done
else
    echo "  (ninguno — ver mensajes de error arriba)"
fi

echo ""
echo "Cómo usar estos demos como benchmark (Sesión S1):"
echo ""
echo "  1. Escuchar los archivos en ${DEMOS_DIR}/"
echo "     para calibrar la calidad esperada del modelo con los prompts del model card."
echo ""
echo "  2. Ejecutar el smoke test con los mismos prompts:"
echo "     cd ../research"
echo "     modal run research_stable_audio_open_modal.py::setup    # una vez"
echo "     modal run research_stable_audio_open_modal.py::smoke \\"
echo "         --prompts-json ../evaluation/prompts_official.json"
echo ""
echo "  3. Comparar los outputs en:"
echo "     evaluation/stable_audio_open/smoke/sao_smoke_0{1,2,3}/output.wav"
echo "     con los archivos de referencia descargados."
echo ""
echo "     Criterio de éxito:"
echo "       - sao_smoke_01 ('128 BPM tech house drum loop'): drum loop reconocible, 8 s"
echo "       - sao_smoke_02 ('hammer hitting wood'): impacto percutivo claro, ~3 s"
echo "       - sao_smoke_03 ('lo-fi electro chill'): textura lo-fi relajada, ~20 s"
echo "     Si los outputs son comparables en calidad → script OK → continuar con eval_all."
echo ""
echo "  4. Eval completo (12 prompts DAW):"
echo "     modal run research_stable_audio_open_modal.py::eval_all \\"
echo "         --prompts-json ../evaluation/prompts.json \\"
echo "         --model-dir stable_audio_open"
