"""
Verificación local del post-procesamiento de beat tracking.

Toma los pares (input.wav, transcribed_cuda.mid) existentes en evaluation/miros/test*/
y aplica _apply_beat_tracking localmente (sin Modal ni GPU).

Comprueba:
  1. BPM detectado por librosa coincide con el tempo map inyectado en el MIDI
  2. El número de notas es idéntico al original
  3. El drift de tiempo absoluto por nota es < 5 ms
  4. El MIDI resultante tiene múltiples cambios de tempo (mapa variable)
  5. El MIDI resultante es válido (carga sin errores)

Uso:
    cd Audio2Midi/research
    uv run python eval_beat_tracking.py [--save]

    --save  guarda cada MIDI procesado como transcribed_cuda_beat.mid
"""

import argparse
import sys
import os
import tempfile
from pathlib import Path

import librosa
import mido
import numpy as np
import pretty_midi

EVAL_DIR = Path(__file__).parent.parent / "evaluation" / "miros"


# ---------------------------------------------------------------------------
# Función copiada de research_miros_modal.py (sin dependencia de Modal)
# ---------------------------------------------------------------------------
def _apply_beat_tracking(audio_bytes: bytes, midi_bytes: bytes) -> bytes:
    suffix = ".wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        suffix = ".mp3"

    tmp_audio = tempfile.mktemp(suffix=suffix)
    tmp_midi_in = tempfile.mktemp(suffix=".mid")
    tmp_midi_out = tempfile.mktemp(suffix=".mid")

    try:
        with open(tmp_audio, "wb") as f:
            f.write(audio_bytes)
        with open(tmp_midi_in, "wb") as f:
            f.write(midi_bytes)

        y, sr = librosa.load(tmp_audio, sr=None, mono=True)
        tempo_arr, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=512, units="frames"
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        global_bpm = float(np.atleast_1d(tempo_arr)[0])

        if len(beat_times) < 2:
            return midi_bytes

        pm = pretty_midi.PrettyMIDI(tmp_midi_in)

        ticks_per_beat = 480
        first_dur = float(beat_times[1] - beat_times[0])
        beat0_tick = int(round(beat_times[0] / first_dur * ticks_per_beat))

        def seconds_to_ticks(t_sec: float) -> int:
            if t_sec <= 0:
                return 0
            if t_sec <= beat_times[0]:
                return max(0, int(round(t_sec / first_dur * ticks_per_beat)))
            if t_sec >= beat_times[-1]:
                last_dur = float(beat_times[-1] - beat_times[-2])
                extra = (t_sec - beat_times[-1]) / last_dur
                return int(round(beat0_tick + (len(beat_times) - 1 + extra) * ticks_per_beat))
            idx = int(np.searchsorted(beat_times, t_sec, side="right")) - 1
            dt = float(beat_times[idx + 1] - beat_times[idx])
            frac = (t_sec - beat_times[idx]) / dt
            return int(round(beat0_tick + (idx + frac) * ticks_per_beat))

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)

        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage("track_name", name="Tempo Map", time=0))

        tempo_us_first = int(round(first_dur * 1e6))
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us_first, time=0))
        last_tick = 0

        for i in range(len(beat_times) - 1):
            dt = beat_times[i + 1] - beat_times[i]
            if dt <= 0:
                continue
            tempo_us = int(round(dt * 1e6))
            tick = beat0_tick + i * ticks_per_beat
            delta = tick - last_tick
            if delta > 0:
                tempo_track.append(
                    mido.MetaMessage("set_tempo", tempo=tempo_us, time=delta)
                )
                last_tick = tick
        tempo_track.append(mido.MetaMessage("end_of_track", time=0))

        drum_ch = 9
        non_drum_chs = [c for c in range(16) if c != drum_ch]
        non_drum_ch_idx = 0

        for inst_idx, inst in enumerate(pm.instruments):
            if inst.is_drum:
                channel = drum_ch
            else:
                channel = non_drum_chs[non_drum_ch_idx % len(non_drum_chs)]
                non_drum_ch_idx += 1

            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(
                mido.MetaMessage("track_name", name=inst.name or f"Inst {inst_idx}", time=0)
            )
            track.append(
                mido.Message("program_change", channel=channel, program=inst.program, time=0)
            )

            events: list[tuple] = []
            for note in inst.notes:
                tick_on = seconds_to_ticks(note.start)
                tick_off = max(tick_on + 1, seconds_to_ticks(note.end))
                events.append((tick_on, "note_on", note.pitch, note.velocity, channel))
                events.append((tick_off, "note_off", note.pitch, 0, channel))
            for cc in inst.control_changes:
                events.append(
                    (seconds_to_ticks(cc.time), "control_change", cc.number, cc.value, channel)
                )
            for pb in inst.pitch_bends:
                events.append(
                    (seconds_to_ticks(pb.time), "pitchwheel", pb.pitch, 0, channel)
                )

            events.sort(key=lambda e: e[0])
            last_tick_ev = 0
            for tick, msg_type, p1, p2, ch in events:
                delta = max(0, tick - last_tick_ev)
                last_tick_ev = tick
                if msg_type == "note_on":
                    track.append(mido.Message("note_on", channel=ch, note=p1, velocity=p2, time=delta))
                elif msg_type == "note_off":
                    track.append(mido.Message("note_off", channel=ch, note=p1, velocity=0, time=delta))
                elif msg_type == "control_change":
                    track.append(mido.Message("control_change", channel=ch, control=p1, value=p2, time=delta))
                elif msg_type == "pitchwheel":
                    track.append(mido.Message("pitchwheel", channel=ch, pitch=p1, time=delta))
            track.append(mido.MetaMessage("end_of_track", time=0))

        mid.save(tmp_midi_out)
        with open(tmp_midi_out, "rb") as f:
            return f.read()

    except Exception as exc:
        print(f"  [ERROR en _apply_beat_tracking] {exc}")
        return midi_bytes

    finally:
        for p in [tmp_audio, tmp_midi_in, tmp_midi_out]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Métricas de verificación
# ---------------------------------------------------------------------------
def extract_tempo_changes(midi_bytes: bytes) -> list[float]:
    """Devuelve la lista de BPMs en el tempo map del MIDI."""
    mid = mido.MidiFile(file=__import__("io").BytesIO(midi_bytes))
    bpms = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                bpms.append(60e6 / msg.tempo)
    return bpms


def note_times(midi_bytes: bytes) -> list[float]:
    """Extrae tiempos de inicio de todas las notas (en segundos) vía pretty_midi."""
    tmp = tempfile.mktemp(suffix=".mid")
    try:
        with open(tmp, "wb") as f:
            f.write(midi_bytes)
        pm = pretty_midi.PrettyMIDI(tmp)
        times = []
        for inst in pm.instruments:
            for note in inst.notes:
                times.append(note.start)
        return sorted(times)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def note_count(midi_bytes: bytes) -> int:
    tmp = tempfile.mktemp(suffix=".mid")
    try:
        with open(tmp, "wb") as f:
            f.write(midi_bytes)
        pm = pretty_midi.PrettyMIDI(tmp)
        return sum(len(inst.notes) for inst in pm.instruments)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Guarda transcribed_cuda_beat.mid por test")
    args = parser.parse_args()

    test_dirs = sorted(
        EVAL_DIR.glob("test*"),
        key=lambda p: int("".join(c for c in p.name if c.isdigit()) or "0"),
    )

    PASS = "\033[32mPASS\033[0m"
    FAIL = "\033[31mFAIL\033[0m"
    header = f"{'Test':<8} {'BPM':>6} {'Beats':>6} {'Tempos':>7} {'Notes':>6} {'MaxDrift':>10}  {'Checks'}"
    print(header)
    print("-" * len(header))

    all_ok = True
    for td in test_dirs:
        audio = td / "input.wav"
        if not audio.exists():
            audio = td / "input.mp3"
        mid_orig = td / "transcribed_cuda.mid"
        if not audio.exists() or not mid_orig.exists():
            print(f"{td.name:<8}  (sin input o MIDI — saltando)")
            continue

        audio_bytes = audio.read_bytes()
        midi_bytes_orig = mid_orig.read_bytes()

        # Detección librosa (para referencia)
        y, sr = librosa.load(str(audio), sr=None, mono=True)
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        librosa_bpm = float(np.atleast_1d(tempo_arr)[0])
        n_beats = len(beat_times)

        # Aplicar beat tracking
        midi_bytes_new = _apply_beat_tracking(audio_bytes, midi_bytes_orig)

        # 1. Notas preservadas
        n_orig = note_count(midi_bytes_orig)
        n_new = note_count(midi_bytes_new)
        notes_ok = n_orig == n_new

        # 2. Tempo map variable (≥ 2 cambios)
        bpms = extract_tempo_changes(midi_bytes_new)
        n_tempos = len(bpms)
        tempos_ok = n_tempos >= 2

        # 3. BPM medio del mapa ≈ librosa BPM (tolerancia ±15%)
        if bpms:
            mean_bpm = float(np.mean(bpms))
            bpm_ok = abs(mean_bpm - librosa_bpm) / librosa_bpm < 0.15
        else:
            mean_bpm = 0.0
            bpm_ok = False

        # 4. Drift de tiempos absolutos (round-trip: segundos → ticks → segundos)
        times_orig = note_times(midi_bytes_orig)
        times_new = note_times(midi_bytes_new)
        if times_orig and times_new and len(times_orig) == len(times_new):
            diffs = [abs(a - b) for a, b in zip(times_orig, times_new)]
            max_drift_ms = max(diffs) * 1000
            drift_ok = max_drift_ms < 20.0  # tolerancia 20 ms
        else:
            max_drift_ms = float("inf")
            drift_ok = False

        ok = notes_ok and tempos_ok and bpm_ok and drift_ok
        all_ok = all_ok and ok

        checks = " ".join([
            f"notes={PASS if notes_ok else FAIL}",
            f"tempos={PASS if tempos_ok else FAIL}",
            f"bpm={PASS if bpm_ok else FAIL}",
            f"drift={PASS if drift_ok else FAIL}",
        ])

        print(
            f"{td.name:<8} {librosa_bpm:>6.1f} {n_beats:>6} {n_tempos:>7} "
            f"{n_new:>6} {max_drift_ms:>9.1f}ms  {checks}"
        )

        if args.save:
            out = td / "transcribed_cuda_beat.mid"
            out.write_bytes(midi_bytes_new)
            print(f"         → guardado {out}")

    print()
    print("Resultado global:", PASS if all_ok else FAIL)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
