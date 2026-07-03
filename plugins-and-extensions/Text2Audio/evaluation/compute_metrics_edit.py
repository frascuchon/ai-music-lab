"""
compute_metrics_edit.py — Métricas objetivas para el benchmark audio2audio (edición/versionado).

A diferencia del benchmark text2audio (adherencia texto→audio), la edición tiene DOS ejes
complementarios que los papers evalúan conjuntamente:

  1. CLAP text→output  (adherencia a la edición):
     Similitud coseno entre el embedding del target_prompt y el embedding del output.wav.
     Mayor = el output suena más al estilo/instrumento pedido.

  2. CLAP audio↔audio  (preservación de contenido):
     Similitud coseno entre el embedding del source_audio y el embedding del output.wav
     (mismo espacio CLAP, modelo cargado una sola vez).
     Mayor = el output conserva más el "sonido" del original.
     Nota: para casos continuation, se mide sobre el segmento continuado (sin el prompt prefix).

  3. Correlación de croma  (preservación melódico-armónica):
     Media de similitudes coseno frame a frame entre los cromagramas CQT del source y el
     output. Rango [0, 1], mayor = más afinidad harmónica/tonal.
     Interpretación varía según categoría:
       - style_transfer / instrumentation: alta correlación esperada (preserva la melodía).
       - mood_texture: correlación media esperada (transforma el color pero no el motivo).
       - continuation: coherencia tonal source ↔ segmento generado.

Setup CLAP — igual que compute_metrics.py (parámetros FIJOS, no cambiar):
  laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny").load_ckpt(model_id=1)
  # model_id=1 → 630k-audioset-best.pt (HTSAT-tiny, sin fusión)

Casos continuation (InspireMusic):
  El output de InspireMusic incluye el audio prompt al inicio seguido de la continuación.
  Este script recorta los primeros source_duration_s segundos del output antes de medir
  CLAP text y correlación de croma. CLAP audio↔audio se calcula sobre el output completo
  (source + continuación) para medir la coherencia global de la continuación.

Estructura esperada:
  evaluation/
    prompts_edit.json
    edit/
      source_audio/          ← generado por fetch_edit_sources.sh
      acestep15/
        case01_bach_jazz/output.wav
        case02_bolero_country/output.wav
        smoke/ace_smoke_01.../output.wav   ← ignorado por este script
      sao_a2a/
        ...
      musicgen_melody/
        ...
      melodyflow/
        ...
      zeta_audioldm2/
        ...
      inspiremusic/
        case11_guitar_continue_jazz/output.wav
        ...

Uso:
  cd plugins-and-extensions/Text2Audio/research
  uv run python ../evaluation/compute_metrics_edit.py
  uv run python ../evaluation/compute_metrics_edit.py --only musicgen_melody,sao_a2a
  uv run python ../evaluation/compute_metrics_edit.py --no-chroma   # solo CLAP (sin librosa)
  uv run python ../evaluation/compute_metrics_edit.py --json /tmp/metrics_edit.json

Dependencias:
  uv add laion-clap>=1.1.4 librosa soundfile numpy
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KNOWN_EDIT_MODELS = [
    "acestep15",
    "sao_a2a",
    "musicgen_melody",
    "melodyflow",
    "zeta_audioldm2",
    "inspiremusic",
]

CHROMA_SR = 22050  # rate de re-muestreo para cromagramas (independiente del source SR)


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class CaseMetrics:
    case_id: str
    model: str
    category: str
    target_prompt: str
    source_id: str
    clap_text: Optional[float]    # adherencia a la edición
    clap_audio: Optional[float]   # preservación de contenido
    chroma_corr: Optional[float]  # afinidad harmónica/tonal
    output_path: str
    trimmed: bool = False         # True si se recortó el prefix (continuation)
    error: Optional[str] = None


@dataclass
class ModelSummary:
    model: str
    n_cases: int
    mean_clap_text: Optional[float]
    mean_clap_audio: Optional[float]
    mean_chroma: Optional[float]
    cases: list[CaseMetrics]


# ---------------------------------------------------------------------------
# CLAP (igual que compute_metrics.py — parámetros FIJOS)
# ---------------------------------------------------------------------------

def _load_clap_model():
    try:
        import laion_clap  # type: ignore
    except ImportError:
        print("ERROR: laion-clap no instalado. Ejecutar: uv add laion-clap>=1.1.4")
        sys.exit(1)
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    model.load_ckpt(model_id=1)  # 630k-audioset-best.pt
    model.eval()
    return model


def _cosine_sim(a, b) -> float:
    import torch
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return float((a * b).sum(dim=-1).item())


def clap_text_audio(model, text: str, audio_path: Path) -> float:
    text_emb = model.get_text_embedding([text], use_tensor=True)
    audio_emb = model.get_audio_embedding_from_filelist([str(audio_path)], use_tensor=True)
    return _cosine_sim(text_emb, audio_emb)


def clap_audio_audio(model, path_a: Path, path_b: Path) -> float:
    emb_a = model.get_audio_embedding_from_filelist([str(path_a)], use_tensor=True)
    emb_b = model.get_audio_embedding_from_filelist([str(path_b)], use_tensor=True)
    return _cosine_sim(emb_a, emb_b)


# ---------------------------------------------------------------------------
# Correlación de croma
# ---------------------------------------------------------------------------

def chroma_correlation(path_a: Path, path_b: Path, sr: int = CHROMA_SR) -> float:
    """
    Correlación de croma CQT entre dos archivos de audio.
    Retorna la media de similitudes coseno frame a frame en [0, 1].
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        raise ImportError("librosa no instalado. Ejecutar: uv add librosa")

    y_a, _ = librosa.load(str(path_a), sr=sr, mono=True)
    y_b, _ = librosa.load(str(path_b), sr=sr, mono=True)

    # Alinear longitudes al mínimo (el source puede ser más corto que el output)
    min_len = min(len(y_a), len(y_b))
    y_a = y_a[:min_len]
    y_b = y_b[:min_len]

    c_a = librosa.feature.chroma_cqt(y=y_a, sr=sr)  # (12, T)
    c_b = librosa.feature.chroma_cqt(y=y_b, sr=sr)

    # Normalizar columnas a unitario
    norm_a = np.linalg.norm(c_a, axis=0, keepdims=True) + 1e-8
    norm_b = np.linalg.norm(c_b, axis=0, keepdims=True) + 1e-8
    c_a_n = c_a / norm_a
    c_b_n = c_b / norm_b

    return float((c_a_n * c_b_n).sum(axis=0).mean())


# ---------------------------------------------------------------------------
# Utilidades de audio
# ---------------------------------------------------------------------------

def _trim_audio_prefix(input_path: Path, offset_s: float) -> Path:
    """
    Recorta los primeros offset_s segundos de input_path y escribe a un WAV temporal.
    Retorna la ruta del archivo temporal (el llamador es responsable de borrarlo si quiere).
    Si offset_s <= 0 devuelve input_path sin modificar.
    """
    if offset_s <= 0:
        return input_path

    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError:
        raise ImportError("librosa y soundfile son necesarios para recortar continuaciones.")

    y, sr = librosa.load(str(input_path), sr=None, mono=False)
    offset_samples = int(offset_s * sr)
    if y.ndim == 1:
        y_trimmed = y[offset_samples:]
    else:
        y_trimmed = y[:, offset_samples:]

    if y_trimmed.shape[-1] == 0:
        raise ValueError(
            f"El recorte de {offset_s:.1f}s dejó audio vacío en {input_path.name} "
            f"(duración total: {y.shape[-1]/sr:.1f}s)."
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y_trimmed.T if y_trimmed.ndim == 2 else y_trimmed, sr)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Descubrimiento de modelos y casos
# ---------------------------------------------------------------------------

def discover_models(edit_dir: Path, only: str = "") -> list[str]:
    if only:
        return [m.strip() for m in only.split(",") if m.strip()]
    found = []
    for model in KNOWN_EDIT_MODELS:
        model_dir = edit_dir / model
        # Solo contar outputs fuera de smoke/
        if model_dir.is_dir():
            outputs = [
                p for p in model_dir.rglob("output.wav")
                if "smoke" not in p.parts
            ]
            if outputs:
                found.append(model)
    return found


def discover_cases(model_dir: Path) -> list[tuple[str, Path]]:
    """
    Retorna [(case_id, output_wav)] ignorando el subdirectorio smoke/.
    """
    result = []
    for output_wav in sorted(model_dir.rglob("output.wav")):
        if "smoke" in output_wav.parts:
            continue
        case_id = output_wav.parent.name
        result.append((case_id, output_wav))
    return result


# ---------------------------------------------------------------------------
# Métricas de un caso
# ---------------------------------------------------------------------------

def compute_case_metrics(
    clap_model,
    case: dict,
    source: dict,
    output_wav: Path,
    eval_base: Path,
    no_chroma: bool,
) -> CaseMetrics:
    """
    Calcula las 3 métricas para un (model, case_id).
    Para casos continuation recorta el prefix del source antes de medir clap_text y chroma.
    """
    import os

    source_path = eval_base / source["file"]
    category = case["category"]
    target_prompt = case["target_prompt"]
    trimmed = False
    tmp_path: Optional[Path] = None

    try:
        # Para continuation: recortar prefix del source del output
        if category == "continuation":
            offset_s = float(source.get("duration_s") or 0.0)
            if offset_s > 0:
                tmp_path = _trim_audio_prefix(output_wav, offset_s)
                measure_path = tmp_path
                trimmed = True
            else:
                measure_path = output_wav
        else:
            measure_path = output_wav

        # 1. CLAP text→output (sobre segmento recortado para continuation)
        clap_text: Optional[float] = None
        try:
            clap_text = clap_text_audio(clap_model, target_prompt, measure_path)
        except Exception as exc:
            print(f"    CLAP text ERROR: {exc}")

        # 2. CLAP audio↔audio (source vs output completo)
        clap_audio: Optional[float] = None
        try:
            clap_audio = clap_audio_audio(clap_model, source_path, output_wav)
        except Exception as exc:
            print(f"    CLAP audio ERROR: {exc}")

        # 3. Chroma (source vs segmento recortado)
        chroma: Optional[float] = None
        if not no_chroma:
            try:
                chroma = chroma_correlation(source_path, measure_path)
            except Exception as exc:
                print(f"    Chroma ERROR: {exc}")

        return CaseMetrics(
            case_id=case["id"],
            model="",  # rellenado por el llamador
            category=category,
            target_prompt=target_prompt,
            source_id=source["id"],
            clap_text=clap_text,
            clap_audio=clap_audio,
            chroma_corr=chroma,
            output_path=str(output_wav),
            trimmed=trimmed,
        )

    finally:
        if tmp_path and tmp_path != output_wav:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Resumen de modelo
# ---------------------------------------------------------------------------

def _mean(values: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def model_summary(model: str, cases: list[CaseMetrics]) -> ModelSummary:
    return ModelSummary(
        model=model,
        n_cases=len(cases),
        mean_clap_text=_mean([c.clap_text for c in cases]),
        mean_clap_audio=_mean([c.clap_audio for c in cases]),
        mean_chroma=_mean([c.chroma_corr for c in cases]),
        cases=cases,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Métricas CLAP + chroma para el benchmark audio2audio (edit/).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--eval-dir", default="../evaluation",
                   help="Ruta al directorio evaluation/ (default: ../evaluation)")
    p.add_argument("--prompts-json", default="../evaluation/prompts_edit.json",
                   help="Ruta a prompts_edit.json (default: ../evaluation/prompts_edit.json)")
    p.add_argument("--only", default="",
                   help="Evaluar solo estos modelos (coma-separado).")
    p.add_argument("--no-chroma", action="store_true",
                   help="Omitir correlación de croma (requiere librosa).")
    p.add_argument("--json", default="", metavar="FILE",
                   help="Exportar resultados a JSON. Default: evaluation/edit/metrics_edit.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    edit_dir = eval_dir / "edit"
    prompts_path = Path(args.prompts_json).resolve()

    if not prompts_path.exists():
        print(f"ERROR: no existe {prompts_path}")
        sys.exit(1)

    with open(prompts_path) as f:
        data = json.load(f)

    cases_by_id = {c["id"]: c for c in data["cases"]}
    sources_by_id = {s["id"]: s for s in data["sources"]}

    models = discover_models(edit_dir, args.only)
    if not models:
        print(f"No se encontraron modelos con output.wav en {edit_dir}")
        print("Ejecuta primero eval_all en algún script Modal de edición.")
        sys.exit(1)

    print(f"Modelos a evaluar: {models}")
    print("")
    print("Cargando modelo CLAP (descarga ~1.2 GB en primer uso) …")
    clap_model = _load_clap_model()
    print("CLAP listo.\n")

    all_summaries: list[ModelSummary] = []

    for model_name in models:
        model_dir = edit_dir / model_name
        print(f"=== {model_name} ===")

        case_results: list[CaseMetrics] = []
        discovered = discover_cases(model_dir)

        if not discovered:
            print(f"  Sin outputs encontrados en {model_dir}")
            continue

        for case_id, output_wav in discovered:
            case = cases_by_id.get(case_id)
            if case is None:
                print(f"  [{case_id}] SKIP — case_id no encontrado en prompts_edit.json")
                continue
            source = sources_by_id.get(case["source_id"])
            if source is None:
                print(f"  [{case_id}] SKIP — source_id '{case['source_id']}' no encontrado")
                continue

            try:
                cm = compute_case_metrics(
                    clap_model, case, source, output_wav, eval_dir, args.no_chroma
                )
                cm.model = model_name

                trim_tag = " [trimmed]" if cm.trimmed else ""
                parts = []
                if cm.clap_text is not None:
                    parts.append(f"CLAP_text={cm.clap_text:.4f}")
                if cm.clap_audio is not None:
                    parts.append(f"CLAP_audio={cm.clap_audio:.4f}")
                if cm.chroma_corr is not None:
                    parts.append(f"chroma={cm.chroma_corr:.4f}")
                print(f"  [{case_id}]{trim_tag} ({cm.category}) {' | '.join(parts)}")
                case_results.append(cm)

            except Exception as exc:
                print(f"  [{case_id}] ERROR: {exc}")
                case_results.append(CaseMetrics(
                    case_id=case_id,
                    model=model_name,
                    category=case.get("category", "unknown"),
                    target_prompt=case.get("target_prompt", ""),
                    source_id=case.get("source_id", ""),
                    clap_text=None,
                    clap_audio=None,
                    chroma_corr=None,
                    output_path=str(output_wav),
                    error=str(exc),
                ))

        summary = model_summary(model_name, case_results)
        all_summaries.append(summary)

        ct = f"{summary.mean_clap_text:.4f}" if summary.mean_clap_text is not None else "  N/A "
        ca = f"{summary.mean_clap_audio:.4f}" if summary.mean_clap_audio is not None else "  N/A "
        ch = f"{summary.mean_chroma:.4f}" if summary.mean_chroma is not None else "  N/A "
        print(f"  Resumen: {summary.n_cases} casos | "
              f"CLAP_text={ct} | CLAP_audio={ca} | chroma={ch}")
        print("")

    # Tabla resumen
    col = 22
    print("=" * (col + 50))
    print(f"{'Modelo':<{col}} {'n':>3}  {'CLAP_text':>10}  {'CLAP_audio':>10}  {'chroma':>8}")
    print("-" * (col + 50))
    for s in all_summaries:
        ct = f"{s.mean_clap_text:.4f}" if s.mean_clap_text is not None else "       N/A"
        ca = f"{s.mean_clap_audio:.4f}" if s.mean_clap_audio is not None else "       N/A"
        ch = f"{s.mean_chroma:.4f}" if s.mean_chroma is not None else "     N/A"
        print(f"  {s.model:<{col-2}} {s.n_cases:>3}  {ct:>10}  {ca:>10}  {ch:>8}")
    print("=" * (col + 50))

    # Export JSON
    json_out = args.json or str(edit_dir / "metrics_edit.json")
    output_data = {
        "models": [
            {
                "model": s.model,
                "n_cases": s.n_cases,
                "mean_clap_text": s.mean_clap_text,
                "mean_clap_audio": s.mean_clap_audio,
                "mean_chroma": s.mean_chroma,
                "cases": [asdict(c) for c in s.cases],
            }
            for s in all_summaries
        ]
    }
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResultados exportados a: {json_out}")


if __name__ == "__main__":
    main()
