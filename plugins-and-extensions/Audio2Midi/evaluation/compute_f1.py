"""
compute_f1.py — Métricas F1 objetivas para la evaluación Audio2Midi.

Compara transcribed_cuda.mid contra el MIDI de referencia (ground truth)
usando mir_eval.transcription. Soporta los tests que tienen GT disponible:
  - yourmt3/test04     (Slakh2100 Track01884)
  - yourmt3/test05     (Slakh2100 Track01975)
  - yourmt3/test07     (MusicNet 2556)
  - yourmt3/test08     (MusicNet 2628)
  - compound/test03    (Slakh2100 Track01884, mismo GT que yourmt3/test04)
  - miros/test04       (Slakh2100 Track01884 — comparación directa con yourmt3/test04)
  - miros/test05       (Slakh2100 Track01975 — comparación directa con yourmt3/test05)
  - miros/test07       (MusicNet 2556 — caveat: score timing)
  - miros/test08       (MusicNet 2628 — caveat: score timing)
  - compound_v2/test04 (Slakh2100 Track01884 — Demucs+ADTOF+YourMT3+, con drum F1)
  - compound_v2/test05 (Slakh2100 Track01975)
  - compound_v2/test07 (MusicNet 2556)
  - compound_v2/test08 (MusicNet 2628)

Ground truth (creado por fetch_ground_truth.sh):
  evaluation/_ground_truth/slakh/Track01884/all_src.mid
  evaluation/_ground_truth/slakh/Track01975/all_src.mid
  evaluation/_ground_truth/musicnet/2556.mid
  evaluation/_ground_truth/musicnet/2628.mid

Uso:
  cd plugins-and-extensions/Audio2Midi/research
  uv run python ../evaluation/compute_f1.py
  uv run python ../evaluation/compute_f1.py --only yourmt3/test04
  uv run python ../evaluation/compute_f1.py --only compound_v2/
  uv run python ../evaluation/compute_f1.py --json /tmp/f1_results.json
  uv run python ../evaluation/compute_f1.py --no-drums  # omite la métrica de batería

Nota MusicNet: los MIDIs de MusicNet son las fuentes ajustadas en tiempo al
audio (no scores estáticos), por lo que el onset_tolerance=50ms es adecuado.
Si el F1 resulta anómalo (<30% en test07), la siguiente mejora sería usar
los CSV de etiquetas per-recording (disponibles en Zenodo 5120004 o Kaggle).

Métrica de drums (compound_v2):
  - Solo para tests con batería en el GT (Slakh, no MusicNet clásico).
  - Agrupa notas de canal drums (is_drum=True) en 5 clases ADTOF:
      BD (35/36), SD (38/40), HH (42/44/46), TT (43/45/47/48/50), CY (49/51/52/53/55/57/59)
  - F1 onset-only por clase (sin pitch matching) con onset_tolerance=50ms.
  - Desactivable con --no-drums para reproducir métricas previas exactamente.

TODO: mir_eval.transcription_velocity para F1 con dinámica (v2).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pretty_midi
import mir_eval

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_ROOT = Path(__file__).resolve().parent
GT_ROOT = EVAL_ROOT / "_ground_truth"


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TestSpec:
    rel_path: str
    dataset: str
    gt_path: Path


TESTS: list[TestSpec] = [
    TestSpec(
        "yourmt3/test04",
        "slakh",
        GT_ROOT / "slakh/Track01884/all_src.mid",
    ),
    TestSpec(
        "yourmt3/test05",
        "slakh",
        GT_ROOT / "slakh/Track01975/all_src.mid",
    ),
    TestSpec(
        "yourmt3/test07",
        "musicnet",
        GT_ROOT / "musicnet/2556.mid",
    ),
    TestSpec(
        "yourmt3/test08",
        "musicnet",
        GT_ROOT / "musicnet/2628.mid",
    ),
    TestSpec(
        "compound/test03",
        "slakh",
        GT_ROOT / "slakh/Track01884/all_src.mid",
    ),
    # MIROS (AMT Challenge 2025 winner) — mismos GT que yourmt3/, comparación directa
    TestSpec(
        "miros/test04",
        "slakh",
        GT_ROOT / "slakh/Track01884/all_src.mid",
    ),
    TestSpec(
        "miros/test05",
        "slakh",
        GT_ROOT / "slakh/Track01975/all_src.mid",
    ),
    TestSpec(
        "miros/test07",
        "musicnet",
        GT_ROOT / "musicnet/2556.mid",
    ),
    TestSpec(
        "miros/test08",
        "musicnet",
        GT_ROOT / "musicnet/2628.mid",
    ),
    # Compound v2 (Demucs+ADTOF+YourMT3+) — mismos GT, comparación directa
    TestSpec(
        "compound_v2/test04",
        "slakh",
        GT_ROOT / "slakh/Track01884/all_src.mid",
    ),
    TestSpec(
        "compound_v2/test05",
        "slakh",
        GT_ROOT / "slakh/Track01975/all_src.mid",
    ),
    TestSpec(
        "compound_v2/test07",
        "musicnet",
        GT_ROOT / "musicnet/2556.mid",
    ),
    TestSpec(
        "compound_v2/test08",
        "musicnet",
        GT_ROOT / "musicnet/2628.mid",
    ),
]

# ---------------------------------------------------------------------------
# Vocabulario de clases instrumental coarse (mc13_full_plus_256 de YourMT3+)
# Cada clase agrupa 8 programas GM consecutivos (0-indexed).
# ---------------------------------------------------------------------------
_GM_CLASS_MAP: dict[str, range] = {
    "Piano":                  range(0, 8),
    "Chromatic Percussion":   range(8, 16),
    "Organ":                  range(16, 24),
    "Guitar":                 range(24, 32),
    "Bass":                   range(32, 40),
    "Strings":                range(40, 48),
    "Ensemble":               range(48, 56),
    "Brass":                  range(56, 64),
    "Reed":                   range(64, 72),
    "Pipe":                   range(72, 80),
    "Synth Lead":             range(80, 88),
    "Synth Pad":              range(88, 96),
    "Synth Effects":          range(96, 104),
}
_DRUM_CLASS = "Drums"


def _program_to_class(program: int, is_drum: bool) -> str:
    if is_drum:
        return _DRUM_CLASS
    for cls, rng in _GM_CLASS_MAP.items():
        if program in rng:
            return cls
    return "Other"


# ---------------------------------------------------------------------------
# Note extraction
# ---------------------------------------------------------------------------
NotesBucket = dict[str, tuple[np.ndarray, np.ndarray]]
"""class_name → (intervals[N,2], pitches_hz[N])"""


def _extract_by_class(midi_path: Path) -> NotesBucket:
    """Load MIDI and group pitched notes by coarse instrument class.

    Drums are excluded from pitched metrics (mir_eval pitch tolerance would
    treat GM drum note numbers as pitches, producing nonsense results).
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    raw: dict[str, list[tuple[float, float, int]]] = {}
    for inst in pm.instruments:
        cls = _program_to_class(inst.program, inst.is_drum)
        if cls == _DRUM_CLASS:
            continue
        for n in inst.notes:
            if n.end <= n.start:
                continue
            raw.setdefault(cls, []).append((n.start, n.end, n.pitch))

    out: NotesBucket = {}
    for cls, notes in raw.items():
        arr = np.array(notes, dtype=float)
        intervals = arr[:, :2]
        pitches_hz = pretty_midi.note_number_to_hz(arr[:, 2].astype(int))
        order = np.argsort(intervals[:, 0])
        out[cls] = (intervals[order], pitches_hz[order])
    return out


def _flatten(bucket: NotesBucket) -> tuple[np.ndarray, np.ndarray]:
    """Merge all classes into one flat sorted array."""
    if not bucket:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    intervals = np.vstack([v[0] for v in bucket.values()])
    pitches = np.concatenate([v[1] for v in bucket.values()])
    order = np.argsort(intervals[:, 0])
    return intervals[order], pitches[order]


# ---------------------------------------------------------------------------
# Drum note extraction and onset F1
# ---------------------------------------------------------------------------
# Mapa de notas GM → 5 clases ADTOF (mismo vocabulario que el modelo)
_DRUM_CLASS_MAP: dict[str, frozenset[int]] = {
    "BD":  frozenset({35, 36}),            # Bass Drum (kick)
    "SD":  frozenset({38, 40}),            # Snare
    "HH":  frozenset({42, 44, 46}),        # Hi-Hat (closed/pedal/open)
    "TT":  frozenset({43, 45, 47, 48, 50}),# Toms
    "CY":  frozenset({49, 51, 52, 53, 55, 57, 59}),  # Cymbals + Ride
}
_PITCH_TO_DRUM_CLASS: dict[int, str] = {
    pitch: cls
    for cls, pitches in _DRUM_CLASS_MAP.items()
    for pitch in pitches
}

DrumBucket = dict[str, np.ndarray]
"""class_name → onsets_sec[N]"""


def _extract_drums(midi_path: Path) -> DrumBucket:
    """Load MIDI and group drum onsets by ADTOF class.

    Only reads instruments with is_drum=True (channel 10 / GM percussion).
    Notes with pitches outside the 5-class map are discarded.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    raw: dict[str, list[float]] = {}
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            cls = _PITCH_TO_DRUM_CLASS.get(n.pitch)
            if cls is None:
                continue
            raw.setdefault(cls, []).append(n.start)

    return {cls: np.sort(np.array(onsets, dtype=float)) for cls, onsets in raw.items()}


def _drum_onset_f1(
    ref_drums: DrumBucket,
    est_drums: DrumBucket,
    onset_tolerance: float = 0.05,
) -> dict:
    """
    Compute onset-only F1 per drum class and macro average.

    Uses a greedy 1-to-1 matching: each ref onset is matched to at most one
    est onset within onset_tolerance seconds. This mirrors standard DTW/ADT
    evaluation practice (no pitch dimension for drums).
    """
    classes = sorted(set(ref_drums) | set(est_drums))
    per_class: dict[str, dict] = {}

    for cls in classes:
        ref_onsets = ref_drums.get(cls, np.array([]))
        est_onsets = est_drums.get(cls, np.array([]))

        n_ref = len(ref_onsets)
        n_est = len(est_onsets)

        if n_ref == 0 and n_est == 0:
            per_class[cls] = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_ref": 0, "n_est": 0}
            continue
        if n_ref == 0:
            per_class[cls] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_ref": 0, "n_est": n_est}
            continue
        if n_est == 0:
            per_class[cls] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_ref": n_ref, "n_est": 0}
            continue

        # Greedy matching: sort both, use a pointer approach
        ref_sorted = np.sort(ref_onsets)
        est_sorted = np.sort(est_onsets)
        matched_ref = np.zeros(n_ref, dtype=bool)
        matched_est = np.zeros(n_est, dtype=bool)
        j_start = 0
        for i in range(n_ref):
            best_j = -1
            best_dist = onset_tolerance + 1.0
            for j in range(j_start, n_est):
                diff = abs(ref_sorted[i] - est_sorted[j])
                if diff > onset_tolerance:
                    if est_sorted[j] > ref_sorted[i] + onset_tolerance:
                        break
                    continue
                if not matched_est[j] and diff < best_dist:
                    best_dist = diff
                    best_j = j
            if best_j >= 0:
                matched_ref[i] = True
                matched_est[best_j] = True
                j_start = max(j_start, best_j)

        tp = int(matched_ref.sum())
        p = tp / n_est if n_est else 0.0
        r = tp / n_ref if n_ref else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[cls] = {"precision": p, "recall": r, "f1": f, "n_ref": n_ref, "n_est": n_est}

    f1_values = [v["f1"] for v in per_class.values() if v["n_ref"] > 0 or v["n_est"] > 0]
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    return {"per_class": per_class, "macro_f1": macro_f1}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _op_f1(
    ref_int: np.ndarray,
    ref_pit: np.ndarray,
    est_int: np.ndarray,
    est_pit: np.ndarray,
    onset_tolerance: float = 0.05,
    with_offset: bool = False,
) -> dict:
    kwargs: dict = dict(
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
    )
    if with_offset:
        kwargs["offset_ratio"] = 0.2
        kwargs["offset_min_tolerance"] = 0.05
    else:
        kwargs["offset_ratio"] = None

    if len(ref_pit) == 0 and len(est_pit) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_ref": 0, "n_est": 0}
    if len(ref_pit) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_ref": 0, "n_est": int(len(est_pit))}
    if len(est_pit) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_ref": int(len(ref_pit)), "n_est": 0}

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pit, est_int, est_pit, **kwargs
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "n_ref": int(len(ref_pit)),
        "n_est": int(len(est_pit)),
    }


def _class_presence_f1(ref_bucket: NotesBucket, est_bucket: NotesBucket) -> dict:
    """Binary F1 on the *set* of coarse instrument classes present."""
    ref_set = set(ref_bucket)
    est_set = set(est_bucket)
    tp = len(ref_set & est_set)
    fp = len(est_set - ref_set)
    fn = len(ref_set - est_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "ref_classes": sorted(ref_set),
        "est_classes": sorted(est_set),
    }


def _per_class_op_f1(
    ref_bucket: NotesBucket,
    est_bucket: NotesBucket,
    onset_tolerance: float = 0.05,
) -> dict[str, dict]:
    classes = set(ref_bucket) | set(est_bucket)
    out: dict[str, dict] = {}
    _empty = (np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float))
    for cls in sorted(classes):
        ref_int, ref_pit = ref_bucket.get(cls, _empty)
        est_int, est_pit = est_bucket.get(cls, _empty)
        out[cls] = _op_f1(ref_int, ref_pit, est_int, est_pit, onset_tolerance)
    return out


# ---------------------------------------------------------------------------
# Per-test driver
# ---------------------------------------------------------------------------
def evaluate_test(spec: TestSpec, onset_tolerance: float = 0.05, include_drums: bool = True) -> dict:
    test_path = EVAL_ROOT / spec.rel_path
    est_mid = test_path / "transcribed_cuda.mid"

    base = {"test": spec.rel_path, "dataset": spec.dataset}

    if not est_mid.exists():
        return {**base, "error": "transcribed_cuda.mid no encontrado"}
    if not spec.gt_path.exists():
        return {**base, "error": f"ground truth no encontrado: {spec.gt_path}"}

    ref_bucket = _extract_by_class(spec.gt_path)
    est_bucket = _extract_by_class(est_mid)

    ref_int_flat, ref_pit_flat = _flatten(ref_bucket)
    est_int_flat, est_pit_flat = _flatten(est_bucket)

    result = {
        **base,
        "gt": str(spec.gt_path.relative_to(EVAL_ROOT)),
        "onset_pitch": _op_f1(
            ref_int_flat, ref_pit_flat, est_int_flat, est_pit_flat,
            onset_tolerance, with_offset=False,
        ),
        "onset_offset_pitch": _op_f1(
            ref_int_flat, ref_pit_flat, est_int_flat, est_pit_flat,
            onset_tolerance, with_offset=True,
        ),
        "per_class": _per_class_op_f1(ref_bucket, est_bucket, onset_tolerance),
        "instrument_class_presence": _class_presence_f1(ref_bucket, est_bucket),
    }

    if include_drums:
        ref_drums = _extract_drums(spec.gt_path)
        est_drums = _extract_drums(est_mid)
        if ref_drums or est_drums:
            result["drums"] = _drum_onset_f1(ref_drums, est_drums, onset_tolerance)
        else:
            result["drums"] = None  # GT sin batería (MusicNet clásico)

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "   N/A"
    if pct:
        return f"{value * 100:5.1f}%"
    return f"{value:.3f}"


def print_table(results: list[dict]) -> None:
    header = f"{'test':<25} {'dataset':<10} {'F1-op':>7} {'F1-oop':>7} {'F1-cls':>7} {'F1-drm':>7} {'n_ref':>6} {'n_est':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['test']:<25} ERROR: {r['error']}")
            continue
        op = r["onset_pitch"]
        oop = r["onset_offset_pitch"]
        cls = r["instrument_class_presence"]
        drum_result = r.get("drums")
        drum_str = _fmt(drum_result["macro_f1"]) if drum_result else "   N/A"
        print(
            f"{r['test']:<25} {r['dataset']:<10}"
            f" {_fmt(op['f1'])}"
            f" {_fmt(oop['f1'])}"
            f" {_fmt(cls['f1'])}"
            f" {drum_str}"
            f" {op['n_ref']:>6}"
            f" {op['n_est']:>6}"
        )

    print()
    print("F1-op  = onset+pitch (50ms onset tol, pitched instruments)")
    print("F1-oop = onset+offset+pitch (offset_ratio=0.2)")
    print("F1-cls = instrument class presence (binary set F1)")
    print("F1-drm = drum onset macro-F1 por clase (BD/SD/HH/TT/CY), N/A si sin drums en GT")

    # Per-class detail for tests without error
    for r in results:
        if "error" in r or "per_class" not in r:
            continue
        per = r["per_class"]
        if not per:
            continue
        print(f"\n  [{r['test']}] F1 por clase instrumental:")
        for cls_name, m in per.items():
            print(f"    {cls_name:<28}  F1={_fmt(m['f1'])}  n_ref={m['n_ref']:>5}  n_est={m['n_est']:>5}")
        print(f"  GT classes:  {r['instrument_class_presence']['ref_classes']}")
        print(f"  EST classes: {r['instrument_class_presence']['est_classes']}")

        drum_result = r.get("drums")
        if drum_result:
            print(f"\n  [{r['test']}] F1 batería por clase (onset ±50ms):")
            for cls_name, m in drum_result["per_class"].items():
                print(f"    {cls_name:<6}  F1={_fmt(m['f1'])}  n_ref={m['n_ref']:>5}  n_est={m['n_est']:>5}")
            print(f"  Macro-F1 drums: {_fmt(drum_result['macro_f1'])}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute F1 metrics for Audio2Midi evaluation")
    parser.add_argument(
        "--only",
        metavar="FILTER",
        help="Filtrar tests por substring, e.g. yourmt3/test04 o slakh",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="FILE",
        help="Escribir resultados completos en JSON",
    )
    parser.add_argument(
        "--onset-tolerance",
        type=float,
        default=0.05,
        metavar="SEC",
        help="Tolerancia de onset en segundos (default: 0.05)",
    )
    parser.add_argument(
        "--no-drums",
        action="store_true",
        help="Omitir la métrica de batería (reproduce comportamiento pre-compound_v2)",
    )
    args = parser.parse_args()

    specs = TESTS
    if args.only:
        specs = [s for s in TESTS if args.only in s.rel_path]
        if not specs:
            print(f"ERROR: ningún test coincide con el filtro '{args.only}'", file=sys.stderr)
            print(f"Tests disponibles: {[s.rel_path for s in TESTS]}", file=sys.stderr)
            sys.exit(1)

    include_drums = not args.no_drums
    print(f"\nEvaluando {len(specs)} test(s) con onset_tolerance={args.onset_tolerance}s …\n")
    results = [evaluate_test(s, args.onset_tolerance, include_drums=include_drums) for s in specs]

    print_table(results)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n[compute_f1] JSON guardado: {args.json}")


if __name__ == "__main__":
    main()
