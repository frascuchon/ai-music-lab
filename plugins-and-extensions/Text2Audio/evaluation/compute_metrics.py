"""
compute_metrics.py — Métricas objetivas para la evaluación Text2Audio.

A diferencia de Audio2Midi (mir_eval F1 contra ground truth 1:1), text2audio
no tiene una respuesta "correcta" por prompt. Las métricas son distributivas:

  1. CLAP score (por prompt, directamente comparable):
     Similitud coseno entre el embedding de texto del prompt y el embedding de
     audio del output.wav generado. Usa LAION-CLAP (music_audioset_epoch_15_esc_90.14).
     Rango: 0.0–1.0, mayor es mejor.

  2. FAD — Fréchet Audio Distance (sobre el conjunto completo de un modelo):
     Distancia entre la distribución de features VGGish/MERT de los audios generados
     y un set de referencia de alta calidad (MusicCaps/Freesound, en _reference/clips/).
     Rango: 0.0–∞, menor es mejor. Comparar entre modelos, no en absoluto.

Estructura esperada:
  evaluation/
    prompts.json
    _reference/clips/   ← WAVs de referencia para FAD (ver fetch_reference_set.sh)
    stable_audio_open/
      prompt01/output.wav
      prompt02/output.wav
      ...
    musicgen/
      prompt01/output.wav
      ...

Uso:
  cd plugins-and-extensions/Text2Audio/research
  uv run python ../evaluation/compute_metrics.py
  uv run python ../evaluation/compute_metrics.py --only stable_audio_open
  uv run python ../evaluation/compute_metrics.py --json /tmp/metrics.json
  uv run python ../evaluation/compute_metrics.py --no-fad   # solo CLAP (sin _reference/)

Nota: CLAP requiere ~1.2 GB de pesos que se descargan automáticamente en
~/.cache/huggingface/ en el primer uso.

Nota: FAD con VGGish requiere tensorflow o torchvggish; con MERT requiere
transformers. fadtk instala sus propias dependencias de embedding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class PromptMetrics:
    prompt_id: str
    model: str
    text: str
    clap_score: Optional[float]
    output_path: str
    error: Optional[str] = None


@dataclass
class ModelMetrics:
    model: str
    fad: Optional[float]
    fad_error: Optional[str]
    prompts: list[PromptMetrics]

    @property
    def mean_clap(self) -> Optional[float]:
        scores = [p.clap_score for p in self.prompts if p.clap_score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def n_generated(self) -> int:
        return sum(1 for p in self.prompts if p.clap_score is not None)


# ---------------------------------------------------------------------------
# CLAP score
# ---------------------------------------------------------------------------

def _load_clap_model():
    """Carga LAION-CLAP music checkpoint. Descarga ~1.2 GB en primer uso."""
    try:
        import laion_clap  # type: ignore
    except ImportError:
        print("ERROR: laion-clap no instalado. Ejecutar: uv add laion-clap>=1.1.4")
        sys.exit(1)

    # model_id=0: 630k-best.pt (HTSAT-tiny, no fusion)
    # model_id=1: 630k-audioset-best.pt (HTSAT-tiny, no fusion) ← mejor opción sin fusion
    # model_id=2: 630k-fusion-best.pt (HTSAT-tiny, fusion) → requiere enable_fusion=True
    # model_id=3: 630k-audioset-fusion-best.pt (HTSAT-tiny, fusion) → requiere enable_fusion=True
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    model.load_ckpt(model_id=1)  # 630k-audioset-best.pt
    model.eval()
    return model


def compute_clap_score(model, text: str, audio_path: Path) -> float:
    """Devuelve similitud coseno texto-audio en [0, 1]."""
    import torch
    import numpy as np

    text_embed = model.get_text_embedding([text], use_tensor=True)
    audio_embed = model.get_audio_embedding_from_filelist([str(audio_path)], use_tensor=True)

    # Normalizar a unitario
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)

    sim = (text_embed * audio_embed).sum(dim=-1).item()
    return float(sim)


# ---------------------------------------------------------------------------
# FAD
# ---------------------------------------------------------------------------

def compute_fad(generated_dir: Path, reference_dir: Path) -> float:
    """
    Calcula FAD entre los WAVs de generated_dir y reference_dir.
    Usa fadtk con embeddings VGGish (estándar en papers de audio generation).

    Nota: requiere >=50 clips en reference_dir para FAD significativo.
    """
    try:
        from fadtk import FrechetAudioDistance  # type: ignore
    except ImportError:
        raise ImportError(
            "fadtk no instalado. Ejecutar: uv add fadtk>=0.1\n"
            "O alternativamente: pip install fadtk"
        )

    # Recopilar todos los WAVs de output en generated_dir
    wav_files = list(generated_dir.rglob("output.wav"))
    ref_files = list(reference_dir.glob("*.wav"))

    if len(wav_files) < 2:
        raise ValueError(f"FAD requiere ≥2 archivos en {generated_dir}, encontrados: {len(wav_files)}")
    if len(ref_files) < 10:
        raise ValueError(
            f"FAD requiere ≥10 clips de referencia en {reference_dir}, encontrados: {len(ref_files)}. "
            "Ejecutar: bash evaluation/fetch_reference_set.sh"
        )

    fad = FrechetAudioDistance(use_pca=False, use_activation=False)
    score = fad.score(str(reference_dir), str(generated_dir))
    return float(score)


# ---------------------------------------------------------------------------
# Descubrimiento de modelos y outputs
# ---------------------------------------------------------------------------

KNOWN_MODELS = [
    "stable_audio_open",
    "stable_audio_open_small",
    "musicgen",
    "magnet",
    "audiogen",
]


def discover_models(eval_dir: Path, only: str = "") -> list[str]:
    """Retorna lista de nombres de modelos con al menos un output.wav."""
    if only:
        return [m.strip() for m in only.split(",") if m.strip()]
    found = []
    for model in KNOWN_MODELS:
        model_dir = eval_dir / model
        if model_dir.is_dir() and any(model_dir.rglob("output.wav")):
            found.append(model)
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calcula FAD y CLAP score para los modelos text2audio evaluados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--eval-dir",
        default="../evaluation",
        help="Ruta al directorio evaluation/ (default: ../evaluation)",
    )
    p.add_argument(
        "--prompts-json",
        default="../evaluation/prompts.json",
        help="Ruta al prompts.json (default: ../evaluation/prompts.json)",
    )
    p.add_argument(
        "--reference",
        default="../evaluation/_reference/clips",
        help="Directorio con clips WAV de referencia para FAD (default: ../evaluation/_reference/clips)",
    )
    p.add_argument(
        "--only",
        default="",
        help="Evaluar solo estos modelos (coma-separado). Default: todos los descubiertos.",
    )
    p.add_argument(
        "--no-fad",
        action="store_true",
        help="Omitir cálculo FAD (útil si _reference/ no está disponible).",
    )
    p.add_argument(
        "--json",
        default="",
        metavar="FILE",
        help="Exportar resultados a JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    prompts_path = Path(args.prompts_json).resolve()
    reference_dir = Path(args.reference).resolve()

    # Cargar prompts
    with open(prompts_path) as f:
        prompts_data = json.load(f)
    prompts = {p["id"]: p for p in prompts_data["prompts"]}

    models = discover_models(eval_dir, args.only)
    if not models:
        print(f"No se encontraron modelos con output.wav en {eval_dir}")
        print("Ejecutar primero: modal run research/research_stable_audio_open_modal.py::eval_all")
        sys.exit(1)

    print(f"Modelos a evaluar: {models}")
    print(f"Prompts disponibles: {len(prompts)}")
    print("")

    # Cargar CLAP una sola vez
    print("Cargando modelo CLAP (descarga ~1.2 GB en primer uso) …")
    clap_model = _load_clap_model()
    print("CLAP listo.")
    print("")

    all_results: list[ModelMetrics] = []

    for model_name in models:
        model_dir = eval_dir / model_name
        print(f"=== {model_name} ===")

        prompt_metrics: list[PromptMetrics] = []

        for prompt_id, prompt in sorted(prompts.items()):
            output_wav = model_dir / prompt_id / "output.wav"
            if not output_wav.exists():
                print(f"  [{prompt_id}] SKIP — output.wav no encontrado")
                continue

            # CLAP
            try:
                clap = compute_clap_score(clap_model, prompt["text"], output_wav)
                print(f"  [{prompt_id}] CLAP={clap:.4f}  ({prompt['category']})")
                pm = PromptMetrics(
                    prompt_id=prompt_id,
                    model=model_name,
                    text=prompt["text"],
                    clap_score=clap,
                    output_path=str(output_wav),
                )
            except Exception as exc:
                print(f"  [{prompt_id}] ERROR CLAP: {exc}")
                pm = PromptMetrics(
                    prompt_id=prompt_id,
                    model=model_name,
                    text=prompt["text"],
                    clap_score=None,
                    output_path=str(output_wav),
                    error=str(exc),
                )
            prompt_metrics.append(pm)

        # FAD
        fad_score: Optional[float] = None
        fad_error: Optional[str] = None

        if not args.no_fad:
            if not reference_dir.is_dir() or not any(reference_dir.glob("*.wav")):
                fad_error = (
                    f"_reference/clips/ no encontrado o vacío. "
                    "Ejecutar: bash evaluation/fetch_reference_set.sh"
                )
                print(f"  FAD: SKIP — {fad_error}")
            else:
                try:
                    fad_score = compute_fad(model_dir, reference_dir)
                    print(f"  FAD={fad_score:.4f}")
                except Exception as exc:
                    fad_error = str(exc)
                    print(f"  FAD: ERROR — {exc}")
        else:
            fad_error = "omitido (--no-fad)"

        model_result = ModelMetrics(
            model=model_name,
            fad=fad_score,
            fad_error=fad_error,
            prompts=prompt_metrics,
        )
        all_results.append(model_result)

        mean_clap = model_result.mean_clap
        clap_str = f"{mean_clap:.4f}" if mean_clap is not None else "N/A"
        fad_str = f"{fad_score:.4f}" if fad_score is not None else str(fad_error)
        print(f"  Resumen: {model_result.n_generated}/{len(prompts)} prompts | "
              f"CLAP medio={clap_str} | FAD={fad_str}")
        print("")

    # Tabla resumen
    print("=" * 70)
    print(f"{'Modelo':<30} {'n':>4}  {'CLAP medio':>11}  {'FAD':>10}")
    print("-" * 70)
    for mr in all_results:
        clap_str = f"{mr.mean_clap:.4f}" if mr.mean_clap is not None else "    N/A"
        fad_str = f"{mr.fad:.4f}" if mr.fad is not None else "       N/A"
        print(f"  {mr.model:<28} {mr.n_generated:>4}  {clap_str:>11}  {fad_str:>10}")
    print("=" * 70)

    # Exportar JSON
    if args.json:
        output_data = {
            "models": [
                {
                    "model": mr.model,
                    "n_generated": mr.n_generated,
                    "mean_clap": mr.mean_clap,
                    "fad": mr.fad,
                    "fad_error": mr.fad_error,
                    "prompts": [asdict(p) for p in mr.prompts],
                }
                for mr in all_results
            ]
        }
        with open(args.json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResultados exportados a: {args.json}")


if __name__ == "__main__":
    main()
