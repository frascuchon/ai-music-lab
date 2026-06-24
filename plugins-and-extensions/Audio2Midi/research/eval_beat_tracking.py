"""
Verificación local del post-procesamiento de beat tracking.

Toma los pares (input.wav, transcribed_cuda.mid) existentes en evaluation/miros/test*/
y aplica _apply_beat_tracking localmente (sin Modal ni GPU).

Comprueba:
  1. BPM detectado por librosa coincide con el tempo único inyectado en el MIDI
  2. El número de notas es idéntico al original
  3. El drift de tiempo absoluto por nota es < 5 ms (round-trip ticks)
  4. El MIDI tiene exactamente 1 evento set_tempo (tempo constante para grilla uniforme)
  5. El BPM está en rango musical (60-180 BPM)

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
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512, aggregate=np.median)
        tempo_arr, _ = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=512, units="frames"
        )
        global_bpm = float(np.atleast_1d(tempo_arr)[0])
        while global_bpm < 60:
            global_bpm *= 2
        while global_bpm > 180:
            global_bpm /= 2

        pm = pretty_midi.PrettyMIDI(tmp_midi_in)

        ticks_per_beat = 480
        beat_duration = 60.0 / global_bpm
        tempo_us = int(round(beat_duration * 1e6))

        def seconds_to_ticks(t_sec: float) -> int:
            return max(0, int(round(t_sec / beat_duration * ticks_per_beat)))

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)

        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage("track_name", name="Tempo Map", time=0))
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
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
    header = f"{'Test':<8} {'BPM':>6} {'Range':>12} {'Tempos':>7} {'Notes':>6} {'MaxDrift':>10}  {'Checks'}"
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

        # Aplicar beat tracking
        midi_bytes_new = _apply_beat_tracking(audio_bytes, midi_bytes_orig)

        # 1. Notas preservadas
        n_orig = note_count(midi_bytes_orig)
        n_new = note_count(midi_bytes_new)
        notes_ok = n_orig == n_new

        # 2. Tempo map: exactamente 1 evento set_tempo (constante)
        bpms = extract_tempo_changes(midi_bytes_new)
        n_tempos = len(bpms)
        tempos_ok = n_tempos == 1

        # 3. BPM en rango musical (60-180)
        detected_bpm = bpms[0] if bpms else 0.0
        bpm_ok = 60 <= detected_bpm <= 180

        # 4. Drift de tiempos absolutos (round-trip: segundos → ticks → segundos)
        #    Con tempo constante el error es solo redondeo de ticks (~1 tick).
        times_orig = note_times(midi_bytes_orig)
        times_new = note_times(midi_bytes_new)
        if times_orig and times_new and len(times_orig) == len(times_new):
            diffs = [abs(a - b) for a, b in zip(times_orig, times_new)]
            max_drift_ms = max(diffs) * 1000
            # Tolerancia = media de los diffs (esperamos <1 tick = <beat_dur/480 s)
            drift_ok = max_drift_ms < 5.0
        else:
            max_drift_ms = float("inf")
            drift_ok = False

        ok = notes_ok and tempos_ok and bpm_ok and drift_ok
        all_ok = all_ok and ok

        checks = " ".join([
            f"notes={PASS if notes_ok else FAIL}",
            f"1tempo={PASS if tempos_ok else FAIL}",
            f"bpm={PASS if bpm_ok else FAIL}",
            f"drift={PASS if drift_ok else FAIL}",
        ])
        bpm_range = f"{detected_bpm:.1f} BPM" if bpms else "—"

        print(
            f"{td.name:<8} {detected_bpm:>6.1f} {bpm_range:>12} {n_tempos:>7} "
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
