"""
compute_f1.py — Métricas F1 objetivas para la evaluación Audio2Midi.

Compara transcribed_cuda.mid contra el MIDI de referencia (ground truth)
usando mir_eval.transcription. Soporta los tests que tienen GT disponible:
  - yourmt3/test04  (Slakh2100 Track01884)
  - yourmt3/test05  (Slakh2100 Track01975)
  - yourmt3/test07  (MusicNet 2556)
  - yourmt3/test08  (MusicNet 2628)
  - compound/test03 (Slakh2100 Track01884, mismo GT que yourmt3/test04)

Ground truth (creado por fetch_ground_truth.sh):
  evaluation/_ground_truth/slakh/Track01884/all_src.mid
  evaluation/_ground_truth/slakh/Track01975/all_src.mid
  evaluation/_ground_truth/musicnet/2556.mid
  evaluation/_ground_truth/musicnet/2628.mid

Uso:
  cd plugins-and-extensions/Audio2Midi/research
  uv run python ../evaluation/compute_f1.py
  uv run python ../evaluation/compute_f1.py --only yourmt3/test04
  uv run python ../evaluation/compute_f1.py --json /tmp/f1_results.json

Nota MusicNet: los MIDIs de MusicNet son las fuentes ajustadas en tiempo al
audio (no scores estáticos), por lo que el onset_tolerance=50ms es adecuado.
Si el F1 resulta anómalo (<30% en test07), la siguiente mejora sería usar
los CSV de etiquetas per-recording (disponibles en Zenodo 5120004 o Kaggle).

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
def evaluate_test(spec: TestSpec, onset_tolerance: float = 0.05) -> dict:
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

    return {
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
    header = f"{'test':<22} {'dataset':<10} {'F1-op':>7} {'F1-oop':>7} {'F1-cls':>7} {'n_ref':>6} {'n_est':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['test']:<22} ERROR: {r['error']}")
            continue
        op = r["onset_pitch"]
        oop = r["onset_offset_pitch"]
        cls = r["instrument_class_presence"]
        print(
            f"{r['test']:<22} {r['dataset']:<10}"
            f" {_fmt(op['f1'])}"
            f" {_fmt(oop['f1'])}"
            f" {_fmt(cls['f1'])}"
            f" {op['n_ref']:>6}"
            f" {op['n_est']:>6}"
        )

    print()
    print("F1-op  = onset+pitch (50ms onset tol)")
    print("F1-oop = onset+offset+pitch (offset_ratio=0.2)")
    print("F1-cls = instrument class presence (binary set F1)")

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
    args = parser.parse_args()

    specs = TESTS
    if args.only:
        specs = [s for s in TESTS if args.only in s.rel_path]
        if not specs:
            print(f"ERROR: ningún test coincide con el filtro '{args.only}'", file=sys.stderr)
            print(f"Tests disponibles: {[s.rel_path for s in TESTS]}", file=sys.stderr)
            sys.exit(1)

    print(f"\nEvaluando {len(specs)} test(s) con onset_tolerance={args.onset_tolerance}s …\n")
    results = [evaluate_test(s, args.onset_tolerance) for s in specs]

    print_table(results)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n[compute_f1] JSON guardado: {args.json}")


if __name__ == "__main__":
    main()
