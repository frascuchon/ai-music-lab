#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-based harmonization of a MIDI melody: key -> chord per bar -> bass and
comping parts.

Used in two places:
  * as the automatic accompaniment "prompt" for Anticipatory (the model needs a
    few seconds of accompaniment before it can continue; a melody alone makes it
    start cold and produce almost nothing), and
  * as a deterministic accompaniment generator in its own right.

Chords are chosen per window of two beats (not per bar) so the result does not
depend on knowing the time signature, from all 24 major/minor triads with the
detected key only as a soft preference.

Pure Python (no numpy, no pretty_midi) for everything except `to_pretty_midi`,
so it can be unit-tested without the heavy dependencies.

A "note" here is a tuple (start_s, end_s, pitch, velocity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# General MIDI programs used for the generated parts
PROGRAM_BASS = 33      # electric bass (finger)
PROGRAM_PIANO = 4      # electric piano 1
PROGRAM_STRINGS = 48   # string ensemble 1

STYLES = ("pop", "pad", "arp")


# ── key and chords ────────────────────────────────────────────────

def _pearson(a, b) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def pitch_class_histogram(notes) -> list[float]:
    hist = [0.0] * 12
    for start, end, pitch, vel in notes:
        hist[pitch % 12] += max(end - start, 0.0) * (0.5 + vel / 254.0)
    return hist


def detect_key(notes) -> tuple[int, str]:
    """Return (tonic pitch class, 'major' | 'minor') by Krumhansl-Schmuckler."""
    hist = pitch_class_histogram(notes)
    if not any(hist):
        return 0, "major"
    best, best_r = (0, "major"), -2.0
    for tonic in range(12):
        rotated = hist[tonic:] + hist[:tonic]
        for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
            r = _pearson(rotated, profile)
            if r > best_r:
                best, best_r = (tonic, mode), r
    return best


@dataclass(frozen=True)
class Chord:
    root: int                 # pitch class
    quality: str              # 'maj' | 'min'
    pcs: tuple = field(default=())

    @property
    def name(self) -> str:
        return NOTE_NAMES[self.root] + ("" if self.quality == "maj" else "m")


_TRIAD_INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7)}


def _make_chord(root: int, quality: str) -> Chord:
    pcs = tuple((root + i) % 12 for i in _TRIAD_INTERVALS[quality])
    return Chord(root % 12, quality, pcs)


ALL_CHORDS = [_make_chord(r, q) for r in range(12) for q in ("maj", "min")]


def diatonic_set(tonic: int, mode: str) -> set:
    """(root, quality) pairs of the triads that belong to the key."""
    if mode == "major":
        scale, qualities = (0, 2, 4, 5, 7, 9, 11), ("maj", "min", "min", "maj", "maj", "min", None)
    else:
        scale, qualities = (0, 2, 3, 5, 7, 8, 10), ("min", None, "maj", "min", "maj", "maj", "maj")
        # harmonic-minor dominant is allowed too: v and V
    out = {((tonic + s) % 12, q) for s, q in zip(scale, qualities) if q}
    if mode == "minor":
        out.add(((tonic + 7) % 12, "min"))
    return out


def _window_score(chord: Chord, window_notes, w0: float, w1: float,
                  key_bonus: float, tonic: int) -> float:
    score = key_bonus
    if chord.root == tonic:
        score += 0.25
    lowest = min((n[2] for n in window_notes), default=None)
    first = min(window_notes, key=lambda n: n[0], default=None)
    for note in window_notes:
        start, end, pitch, vel = note
        overlap = min(end, w1) - max(start, w0)
        if overlap <= 0:
            continue
        weight = overlap * (0.5 + vel / 254.0)
        if note is first:
            weight *= 1.5           # the first note of a window is usually on the beat
        pc = pitch % 12
        if pc == chord.root:
            score += 1.4 * weight
        elif pc in chord.pcs:
            score += 1.0 * weight
        else:
            score -= 0.7 * weight
        if pitch == lowest and pc == chord.root:
            score += 0.6 * weight   # the bass note of the riff names the chord
    return score


def choose_chords(notes, windows, tonic: int, mode: str, change_penalty: float = 0.3):
    """One triad per window from all 24 major/minor triads, with the key as a soft
    preference and a Viterbi pass so chords do not flicker. `windows` is a list
    of (t0, t1). Returns a list of Chord (one per window)."""
    in_key = diatonic_set(tonic, mode)
    n = len(windows)
    if n == 0:
        return []
    k = len(ALL_CHORDS)

    emit, has_notes = [], []
    for w0, w1 in windows:
        window = [x for x in notes if x[1] > w0 and x[0] < w1]
        has_notes.append(bool(window))
        emit.append([
            _window_score(c, window, w0, w1,
                          0.35 if (c.root, c.quality) in in_key else 0.0, tonic)
            for c in ALL_CHORDS
        ])

    best, back = [emit[0][:]], [[-1] * k]
    for t in range(1, n):
        row, brow = [], []
        for j in range(k):
            options = []
            for i in range(k):
                move = 0.0 if i == j else (-6.0 if not has_notes[t] else -change_penalty)
                options.append((best[t - 1][i] + move, i))
            val, arg = max(options)
            row.append(val + emit[t][j])
            brow.append(arg)
        best.append(row)
        back.append(brow)

    j = max(range(k), key=lambda x: best[-1][x])
    path = [j]
    for t in range(n - 1, 0, -1):
        j = back[t][j]
        path.append(j)
    path.reverse()
    return [ALL_CHORDS[j] for j in path]


def merge_segments(windows, chords):
    """Merge consecutive windows that carry the same chord -> [(t0, t1, Chord)]."""
    segments = []
    for (w0, w1), chord in zip(windows, chords):
        if segments and segments[-1][2] == chord:
            segments[-1] = (segments[-1][0], w1, chord)
        else:
            segments.append((w0, w1, chord))
    return segments


# ── rendering ─────────────────────────────────────────────────────

def _closest_voicing(chord: Chord, previous_center: float, low: int, high: int) -> list[int]:
    """Triad voicing (3 notes) in [low, high] whose centre is closest to the last one."""
    best, best_d = None, 1e9
    for inversion in range(3):
        pcs = list(chord.pcs[inversion:]) + list(chord.pcs[:inversion])
        for octave_base in range(12 * (low // 12) - 12, high + 12, 12):   # multiples of 12: pitch classes stay intact
            voicing, prev = [], None
            for pc in pcs:
                p = octave_base + pc
                while prev is not None and p <= prev:
                    p += 12
                while p < low:
                    p += 12
                voicing.append(p)
                prev = p
            if max(voicing) > high or min(voicing) < low:
                continue
            d = abs(sum(voicing) / 3 - previous_center)
            if d < best_d:
                best, best_d = voicing, d
    return best or [low + (pc - low) % 12 for pc in chord.pcs]


def _bass_pitch(root_pc: int, low: int = 36, high: int = 47) -> int:
    p = low + (root_pc - low) % 12
    return p if p <= high else p - 12


def render(segments, beat: float, style: str = "pop"):
    """Return {'bass': [...notes], 'comp': [...notes], 'comp_program': int}.
    Patterns are laid out in beats from the start of each chord segment, so they
    do not depend on the bar length."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose one of {STYLES}")
    bass, comp = [], []
    center = 60.0
    for t0, t1, chord in segments:
        root = _bass_pitch(chord.root)
        fifth = root + 7 if root + 7 <= 52 else root - 5
        voicing = _closest_voicing(chord, center, 52, 72)
        center = sum(voicing) / len(voicing)
        n_beats = max(1, round((t1 - t0) / beat))

        if style == "pad":
            bass.append((t0, t1 - 0.02, root, 78))
            comp.extend((t0, t1 - 0.02, p, 52) for p in voicing)
            continue
        for i in range(0, n_beats, 2):
            s = t0 + i * beat
            e = min(s + 2 * beat, t1) - 0.02
            if style == "arp":
                bass.append((s, e, root if (i // 2) % 2 == 0 else fifth, 80))
                order = [voicing[0], voicing[1], voicing[2], voicing[1]]
                for h in range(4):
                    hs = s + h * beat / 2
                    if hs < t1:
                        comp.append((hs, min(hs + beat / 2, t1) - 0.02, order[h], 58 if h % 2 == 0 else 48))
            else:  # pop
                bass.append((s, min(s + 1.5 * beat, t1) - 0.02, root if (i // 2) % 2 == 0 else fifth, 82))
                comp.extend((s, e, p, 56) for p in voicing)
    program = PROGRAM_STRINGS if style == "pad" else PROGRAM_PIANO
    return {"bass": bass, "comp": comp, "comp_program": program}


# ── public entry points ───────────────────────────────────────────

def beat_windows(bpm: float, end_time: float, beats_per_window: int = 2):
    """Consecutive windows of `beats_per_window` beats covering [0, end_time]."""
    length = 60.0 / bpm * beats_per_window
    windows, t = [], 0.0
    while t < end_time - 1e-6:
        windows.append((t, t + length))
        t += length
    return windows


def harmonize(notes, bpm: float, style: str = "pop", beats_per_window: int = 2,
              until: float | None = None):
    """Harmonize a melody whose first note is at t = 0.

    `until` limits the accompaniment to [0, until] seconds (used to build the
    short prompt Anticipatory needs).

    Returns a dict: key=(tonic, mode), chords=[(t0, t1, Chord)], bass, comp, comp_program.
    """
    if not notes:
        raise ValueError("the melody has no notes")
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    tonic, mode = detect_key(notes)
    end = max(n[1] for n in notes)
    windows = beat_windows(bpm, end, beats_per_window)
    chords = choose_chords(notes, windows, tonic, mode)
    segments = merge_segments(windows, chords)
    parts = render(segments, 60.0 / bpm, style)
    out = {"key": (tonic, mode), "chords": segments, **parts}
    if until is not None:
        for name in ("bass", "comp"):
            out[name] = [(s, min(e, until), p, v) for s, e, p, v in out[name] if s < until]
    return out


def describe(result) -> str:
    tonic, mode = result["key"]
    prog = " | ".join(c.name for _, _, c in result["chords"])
    return f"{NOTE_NAMES[tonic]} {mode}: {prog}"


def to_pretty_midi(result, melody_notes=None, melody_program: int = 0):
    """Build a PrettyMIDI with bass, comping and (optionally) the melody."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()

    def add(notes, program, name):
        inst = pretty_midi.Instrument(program=program, name=name)
        for s, e, p, v in notes:
            if e > s:
                inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p), start=s, end=e))
        if inst.notes:
            pm.instruments.append(inst)

    if melody_notes:
        add(melody_notes, melody_program, "Melody")
    add(result["bass"], PROGRAM_BASS, "Bass")
    add(result["comp"], result["comp_program"], "Chords")
    return pm
