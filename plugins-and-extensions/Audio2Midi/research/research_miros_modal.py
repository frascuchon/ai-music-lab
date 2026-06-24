"""
Modal.com inference app para MIROS (Music Information Retrieval Osnabrück).

MIROS ganó el AMT Challenge 2025 (F-measure 0.5998 vs MT3 baseline 0.3932).
Extiende YourMT3+ sustituyendo el encoder por MusicFM (conformer pre-entrenado
con BEST-RQ) y añadiendo un adapter recurrente con instrument-group embeddings.
Decoders T5-style con cross-attention, RoPE y FlashAttention.

Repo:  https://github.com/amt-os/ai4m-miros  (Apache 2.0 no confirmado — uso interno)
Paper: "Advancing Multi-Instrument Music Transcription: Results from the 2025
       AMT Challenge" — arXiv 2603.27528 (ICLR Workshop / ICME 2025)

AVISO: El repositorio no tiene LICENSE explícita. Uso solo para investigación
interna. No redistribuir pesos ni código.

Checkpoints (Google Drive):
  MusicFM pretrain (~380 MB): gdrive id 1FqqMfcdqeiRr1v7sdrfkqPpr0Vs7e9nZ
    → model/musicfm/data/pretrained_msd.pt
  MIROS fine-tuned (~1.5 GB): gdrive id 1hp-6D1yYvPxXCXDQyXRQRJArle8R-VfB
    → logs/Multi_longer_seq_length_frozen_enc_silu/le2bzt53/checkpoints/last.ckpt

Pipeline:
    [audio.wav/mp3]
        ↓
    MusicFM encoder (conformer, pretrained BEST-RQ)
        ↓
    Recurrent adapter (instrument-group embeddings)
        ↓
    Parallel multi-T5 decoders (RoPE, FlashAttention)
        ↓
    librosa.beat.beat_track → variable tempo map (beat tracking)
        ↓
    transcribed_cuda.mid (multi-track, tempo map corregido)

Setup (descarga pesos al Volume, ejecutar una vez):
    cd Audio2Midi/research
    modal run research_miros_modal.py::setup

Smoke test:
    modal run research_miros_modal.py::main \\
        --audio-path ../evaluation/miros/test04/input.wav \\
        --out-dir ../evaluation/miros/test04

Benchmark completo:
    modal run research_miros_modal.py::eval_all \\
        --eval-dir ../evaluation/miros

Solo tests con ground truth:
    modal run research_miros_modal.py::eval_all \\
        --eval-dir ../evaluation/miros --only 4,5,7,8

GPU: A10G requerida (flash-attn 2.7.2 requiere Ampere+, T4 no sirve).
     Override: env MIROS_GPU=<gpu>.

Coste estimado (A10G):
  setup (descarga ~2 GB): ~$0.05
  10 tests × ~2-3 min/test: ~$0.20-0.40 total
"""

import os
import re
import sys
import time
import tempfile
import uuid
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("miros-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"

MIROS_REPO_URL = "https://github.com/amt-os/ai4m-miros"
REPO_DIR = "/workspace/ai4m-miros"

# Paths dentro del repo (donde main.py espera los checkpoints)
MUSICFM_PRETRAIN_REPO_PATH = "model/musicfm/data/pretrained_msd.pt"
MIROS_CKPT_REPO_PATH = "logs/Multi_longer_seq_length_frozen_enc_silu/le2bzt53/checkpoints/last.ckpt"

# Paths en el Volume (estructura limpia para persistencia)
MUSICFM_PRETRAIN_VOL_PATH = "musicfm/pretrained_msd.pt"
MIROS_CKPT_VOL_PATH = "miros/last.ckpt"

# Google Drive IDs (confirmados en main.py del repo)
MUSICFM_GDRIVE_ID = "1FqqMfcdqeiRr1v7sdrfkqPpr0Vs7e9nZ"
MIROS_FINETUNED_GDRIVE_ID = "1hp-6D1yYvPxXCXDQyXRQRJArle8R-VfB"

DEFAULT_GPU = os.environ.get("MIROS_GPU", "A10G")

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "sox", "libsndfile1", "curl"])
    .run_commands(
        f"git clone {MIROS_REPO_URL} {REPO_DIR}"
    )
    .run_commands(
        # Auditoría de la estructura clonada — aparece en los logs del build
        f"echo '=== MIROS repo .py files ===' && find {REPO_DIR} -name '*.py' "
        f"! -path '*/__pycache__/*' ! -path '*/logs/*' | sort | head -60"
    )
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "lightning==2.4.0",
        "transformers==4.48.3",
        "torchmetrics==1.4.2",
        "einops",
        "librosa==0.10.2.post1",
        "soundfile",
        "pretty_midi",
        "mido",
        "numpy",
        "scipy",
        "scikit-learn",
        "wandb",
        "gdown",                                          # descarga checkpoints Google Drive
        "git+https://github.com/craffel/mir_eval.git",
        "git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup.git",
    )
    .run_commands(
        # flash-attn 2.7.2 wheel precompilada (cu12+torch2.4+py3.10). Ampere+ only (A10G OK).
        # El build step no tiene GPU/CUDA_HOME → compilar desde fuente falla; usamos wheel binaria.
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/"
        "flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
)

app = modal.App("miros-inference", image=image)


# ---------------------------------------------------------------------------
# Diagnóstico — ejecutar si hay errores de checkpoint
# ---------------------------------------------------------------------------
@app.function(volumes={WEIGHTS_MOUNT: weights_vol}, timeout=120)
def debug_checkpoint():
    """
    Inspecciona los checkpoints descargados en el Volume.
    modal run research_miros_modal.py::debug_checkpoint
    """
    for label, vol_path in [
        ("MusicFM pretrain", MUSICFM_PRETRAIN_VOL_PATH),
        ("MIROS fine-tuned", MIROS_CKPT_VOL_PATH),
    ]:
        ckpt = f"{WEIGHTS_MOUNT}/{vol_path}"
        print(f"\n--- {label} ---")
        print(f"Path:    {ckpt}")
        print(f"Exists:  {os.path.exists(ckpt)}")
        if os.path.exists(ckpt):
            size = os.path.getsize(ckpt)
            print(f"Size:    {size} bytes ({size/1e6:.1f} MB)")
            with open(ckpt, "rb") as f:
                first = f.read(8)
            print(f"Magic:   {first.hex()!r}")
            if first[:2] == b"PK":
                print("Formato: PyTorch ZIP ✓")
            elif first[0:1] == b"\x80":
                print("Formato: PyTorch pickle ✓")
            else:
                print("Formato: DESCONOCIDO — posible archivo corrupto o LFS pointer")


# ---------------------------------------------------------------------------
# Helpers compartidos
# ---------------------------------------------------------------------------
def _download_gdrive(gdrive_id: str, dest_path: str, label: str = "") -> None:
    """Descarga un archivo de Google Drive usando gdown (primary) + curl (fallback)."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tag = f"[{label}] " if label else ""
    t0 = time.time()

    # Intentar con gdown primero (maneja confirmación automática para archivos >100MB)
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        print(f"{tag}Descargando con gdown desde {url} …")
        gdown.download(url, dest_path, quiet=False)
        size = os.path.getsize(dest_path)
        print(f"{tag}OK (gdown) — {size/1e6:.0f} MB en {time.time()-t0:.0f}s")
        return
    except Exception as e:
        print(f"{tag}gdown falló ({e}), intentando curl …")
        if os.path.exists(dest_path):
            os.unlink(dest_path)

    # Fallback: curl con confirm=t (mismo approach que main.py original)
    import subprocess
    url = f"https://drive.usercontent.google.com/download?id={gdrive_id}&export=download&confirm=t"
    print(f"{tag}Descargando con curl …")
    result = subprocess.run(
        ["curl", "-fL", "--progress-bar", "-o", dest_path, url],
        check=True,
    )
    size = os.path.getsize(dest_path)
    print(f"{tag}OK (curl) — {size/1e6:.0f} MB en {time.time()-t0:.0f}s")


def _setup_repo_env() -> None:
    """
    Configura sys.path y symlinks de checkpoints.

    MIROS espera los checkpoints en rutas relativas al root del repo:
      model/musicfm/data/pretrained_msd.pt
      logs/Multi_longer_seq_length_frozen_enc_silu/le2bzt53/checkpoints/last.ckpt

    Los almacenamos en el Volume con nombres planos y creamos symlinks.
    """
    # Añadir el repo al Python path
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    os.chdir(REPO_DIR)

    # Symlink MusicFM pretrain
    musicfm_dest = f"{REPO_DIR}/{MUSICFM_PRETRAIN_REPO_PATH}"
    musicfm_src = f"{WEIGHTS_MOUNT}/{MUSICFM_PRETRAIN_VOL_PATH}"
    os.makedirs(os.path.dirname(musicfm_dest), exist_ok=True)
    if not os.path.lexists(musicfm_dest):
        os.symlink(musicfm_src, musicfm_dest)
        print(f"[setup_env] Symlink: {musicfm_dest} → {musicfm_src}")

    # Symlink MIROS fine-tuned ckpt
    miros_dest = f"{REPO_DIR}/{MIROS_CKPT_REPO_PATH}"
    miros_src = f"{WEIGHTS_MOUNT}/{MIROS_CKPT_VOL_PATH}"
    os.makedirs(os.path.dirname(miros_dest), exist_ok=True)
    if not os.path.lexists(miros_dest):
        os.symlink(miros_src, miros_dest)
        print(f"[setup_env] Symlink: {miros_dest} → {miros_src}")


def _transcribe_one(audio_bytes: bytes, beat_tracking: bool = True) -> bytes:
    """
    Transcribe un audio (bytes) a MIDI (bytes) llamando a miros_transcribe().

    transcribe.py::transcribe() recarga el modelo en cada llamada (~30s overhead).
    Aceptable frente a los ~2-3 min de inferencia por audio.
    Para batch de 10 tests, el total de overhead de carga es ~5 min sobre ~30 min de inferencia.
    """
    from transcribe import transcribe as miros_transcribe

    # Determinamos sufijo según primeros bytes (WAV=RIFF, MP3=ID3/0xFF)
    suffix = ".wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        suffix = ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        audio_path = tmp_in.name

    midi_fd, midi_path = tempfile.mkstemp(suffix=".mid")
    os.close(midi_fd)

    try:
        miros_transcribe(audio_path, midi_path)
        if os.path.exists(midi_path) and os.path.getsize(midi_path) > 0:
            with open(midi_path, "rb") as f:
                midi_bytes = f.read()
            if beat_tracking:
                midi_bytes = _apply_beat_tracking(audio_bytes, midi_bytes)
            return midi_bytes
        else:
            raise RuntimeError(f"MIROS no generó MIDI en: {midi_path}")
    finally:
        for p in [audio_path, midi_path]:
            if os.path.exists(p):
                os.unlink(p)


def _apply_beat_tracking(audio_bytes: bytes, midi_bytes: bytes) -> bytes:
    """
    Post-procesamiento: detecta el BPM del audio con librosa y reescribe el MIDI
    con un único tempo constante global.

    Usar un tempo constante (no beat-a-beat variable) garantiza que REAPER muestre
    una grilla uniforme con barras de igual longitud, lo que hace el material
    cómodamente editable. El BPM detectado define la grilla; las notas se posicionan
    en los ticks correspondientes a sus tiempos absolutos originales.

    Los tiempos absolutos de las notas (en segundos) se conservan intactos.
    """
    import librosa
    import pretty_midi
    import mido
    import numpy as np

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

        # 1. Detección de BPM global desde el audio
        y, sr = librosa.load(tmp_audio, sr=None, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512, aggregate=np.median)
        tempo_arr, _ = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=512, units="frames"
        )
        global_bpm = float(np.atleast_1d(tempo_arr)[0])

        # Mantener en rango musical típico (60-180 BPM)
        while global_bpm < 60:
            global_bpm *= 2
        while global_bpm > 180:
            global_bpm /= 2

        print(f"[beat_tracking] BPM detectado: {global_bpm:.1f}")

        # 2. Leer notas originales (tiempos en segundos) con pretty_midi
        pm = pretty_midi.PrettyMIDI(tmp_midi_in)

        # 3. Conversor segundos → ticks con tempo constante
        #    t_sec / beat_duration * ticks_per_beat
        ticks_per_beat = 480
        beat_duration = 60.0 / global_bpm
        tempo_us = int(round(beat_duration * 1e6))

        def seconds_to_ticks(t_sec: float) -> int:
            return max(0, int(round(t_sec / beat_duration * ticks_per_beat)))

        # 4. Construir MIDI con mido: track 0 = tempo único
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)

        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage("track_name", name="Tempo Map", time=0))
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        tempo_track.append(mido.MetaMessage("end_of_track", time=0))

        # 5. Pistas de instrumento
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
                    track.append(
                        mido.Message("note_on", channel=ch, note=p1, velocity=p2, time=delta)
                    )
                elif msg_type == "note_off":
                    track.append(
                        mido.Message("note_off", channel=ch, note=p1, velocity=0, time=delta)
                    )
                elif msg_type == "control_change":
                    track.append(
                        mido.Message("control_change", channel=ch, control=p1, value=p2, time=delta)
                    )
                elif msg_type == "pitchwheel":
                    track.append(
                        mido.Message("pitchwheel", channel=ch, pitch=p1, time=delta)
                    )
            track.append(mido.MetaMessage("end_of_track", time=0))

        mid.save(tmp_midi_out)
        with open(tmp_midi_out, "rb") as f:
            result = f.read()

        print(f"[beat_tracking] MIDI reescrito: {len(midi_bytes)} → {len(result)} bytes")
        return result

    except Exception as exc:
        print(f"[beat_tracking] ERROR ({exc}) — devolviendo MIDI original sin modificar.")
        return midi_bytes

    finally:
        for p in [tmp_audio, tmp_midi_in, tmp_midi_out]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Setup — descarga checkpoints al Volume
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=3600,
    cpu=4,
    memory=8192,
)
def setup() -> None:
    """
    Descarga MusicFM pretrain + MIROS fine-tuned al Volume con gdown.
    Ejecutar una sola vez:
        modal run research_miros_modal.py::setup
    """
    def _need_download(vol_path: str, min_mb: float = 50.0) -> bool:
        full = f"{WEIGHTS_MOUNT}/{vol_path}"
        if not os.path.exists(full):
            return True
        size_mb = os.path.getsize(full) / 1e6
        if size_mb < min_mb:
            print(f"[setup] Archivo demasiado pequeño ({size_mb:.1f} MB < {min_mb} MB), re-descargando.")
            os.unlink(full)
            return True
        print(f"[setup] Ya existe: {vol_path} ({size_mb:.0f} MB) — skip.")
        return False

    weights_vol.reload()

    if _need_download(MUSICFM_PRETRAIN_VOL_PATH, min_mb=100.0):
        _download_gdrive(
            MUSICFM_GDRIVE_ID,
            f"{WEIGHTS_MOUNT}/{MUSICFM_PRETRAIN_VOL_PATH}",
            label="MusicFM",
        )

    if _need_download(MIROS_CKPT_VOL_PATH, min_mb=200.0):
        _download_gdrive(
            MIROS_FINETUNED_GDRIVE_ID,
            f"{WEIGHTS_MOUNT}/{MIROS_CKPT_VOL_PATH}",
            label="MIROS ckpt",
        )

    # Verificar magic bytes
    for label, vol_path in [
        ("MusicFM", MUSICFM_PRETRAIN_VOL_PATH),
        ("MIROS", MIROS_CKPT_VOL_PATH),
    ]:
        full = f"{WEIGHTS_MOUNT}/{vol_path}"
        with open(full, "rb") as f:
            magic = f.read(8)
        if magic[:2] in (b"PK", b"\x80\x04") or magic[0:1] == b"\x80":
            print(f"[setup] {label}: magic OK ({magic[:4].hex()!r})")
        else:
            print(f"[setup] {label}: WARNING — magic inesperado {magic[:4].hex()!r} (posible LFS pointer)")

    weights_vol.commit()
    print("[setup] Volume commiteado. Setup completado.")


# ---------------------------------------------------------------------------
# Función Modal principal — transcribe lista de audios
# ---------------------------------------------------------------------------
@app.function(
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=7200,
    gpu=DEFAULT_GPU,
)
def transcribe_batch(audio_payloads: list[bytes], beat_tracking: bool = True) -> list[bytes]:
    """
    Transcribe cada audio a MIDI multi-instrumento con MIROS.

    audio_payloads: bytes de cada archivo WAV/MP3
    beat_tracking: si True (por defecto), reescribe el tempo map del MIDI con
                   librosa.beat.beat_track para que el grid de REAPER se alinee.
    Returns: list[midi_bytes] — b"" si la transcripción falló para ese audio
    """
    _setup_repo_env()

    results = []
    for i, audio_bytes in enumerate(audio_payloads):
        t0 = time.time()
        try:
            midi_bytes = _transcribe_one(audio_bytes, beat_tracking=beat_tracking)
            elapsed = time.time() - t0
            print(f"[transcribe] [{i+1}/{len(audio_payloads)}] OK — {len(midi_bytes)} bytes ({elapsed:.1f}s)")
            results.append(midi_bytes)
        except Exception as exc:
            print(f"[transcribe] [{i+1}/{len(audio_payloads)}] ERROR: {exc}")
            results.append(b"")

    return results


# ---------------------------------------------------------------------------
# Entrypoint: un audio → out-dir
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    audio_path: str = "",
    out_dir: str = ".",
    force: bool = False,
    no_beat_tracking: bool = False,
):
    """
    Transcribe un audio con MIROS y guarda el MIDI resultante en out-dir.

    Ejemplo:
        modal run research_miros_modal.py::main \\
            --audio-path ../evaluation/miros/test04/input.wav \\
            --out-dir ../evaluation/miros/test04

        # Sin beat tracking (tempo map por defecto):
        modal run research_miros_modal.py::main \\
            --audio-path ... --no-beat-tracking
    """
    if not audio_path:
        print("ERROR: --audio-path requerido. Ejemplo:")
        print("  modal run research_miros_modal.py::main \\")
        print("      --audio-path ../evaluation/miros/test04/input.wav \\")
        print("      --out-dir ../evaluation/miros/test04")
        raise SystemExit(1)

    audio_p = Path(audio_path)
    if not audio_p.exists():
        print(f"ERROR: No existe: {audio_p}")
        raise SystemExit(1)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_mid = out_p / "transcribed_cuda.mid"

    if out_mid.exists() and not force:
        print(f"[main] Ya existe {out_mid}. Usa --force para regenerar.")
        return

    beat_tracking = not no_beat_tracking
    print(f"[main] Transcribiendo con MIROS: {audio_p.name} → {out_mid} (beat_tracking={beat_tracking})")
    audio_bytes = audio_p.read_bytes()
    [midi_bytes] = transcribe_batch.remote([audio_bytes], beat_tracking=beat_tracking)

    if not midi_bytes:
        print("[main] ERROR: la transcripción devolvió vacío.")
        raise SystemExit(1)

    out_mid.write_bytes(midi_bytes)
    print(f"[main] Guardado: {out_mid} ({len(midi_bytes)} bytes)")


# ---------------------------------------------------------------------------
# Entrypoint: todos los tests del directorio de evaluación
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/miros",
    force: bool = False,
    only: str = "",
    no_beat_tracking: bool = False,
):
    """
    Transcribe todos los tests del directorio de evaluación con MIROS.

    Cada carpeta test*/ debe contener input.wav (o input.mp3 para test10).
    Output por test: transcribed_cuda.mid

    Ejemplos:
        modal run research_miros_modal.py::eval_all

        # Solo tests con ground truth:
        modal run research_miros_modal.py::eval_all --only 4,5,7,8

        # Forzar re-transcripción:
        modal run research_miros_modal.py::eval_all --force

        # Sin beat tracking:
        modal run research_miros_modal.py::eval_all --no-beat-tracking
    """
    eval_path = Path(eval_dir)

    test_dirs = sorted(
        eval_path.glob("test*"),
        key=lambda p: int(re.sub(r"\D", "", p.name) or "0"),
    )

    if only:
        only_nums = {int(x) for x in only.split(",")}
        test_dirs = [
            d for d in test_dirs
            if int(re.sub(r"\D", "", d.name) or "0") in only_nums
        ]

    if not test_dirs:
        print(f"ERROR: No se encontraron carpetas test* en {eval_dir}")
        sys.exit(1)

    audio_bytes_list = []
    valid_dirs = []
    for td in test_dirs:
        input_wav = td / "input.wav"
        if not input_wav.exists():
            input_mp3 = td / "input.mp3"
            if input_mp3.exists():
                input_wav = input_mp3
            else:
                print(f"[warn] Sin input.wav/mp3 en {td.name} — saltando. Ejecuta fetch_fixtures.sh primero.")
                continue

        out_mid = td / "transcribed_cuda.mid"
        if out_mid.exists() and not force:
            print(f"[skip] {td.name}: ya tiene transcribed_cuda.mid. Usa --force para regenerar.")
            continue

        audio_bytes_list.append(input_wav.read_bytes())
        valid_dirs.append((td, out_mid))
        print(f"[eval_all] Encolado: {td.name} ({input_wav.name}, {input_wav.stat().st_size / 1e6:.1f} MB)")

    if not audio_bytes_list:
        print("[eval_all] Nada que transcribir (todos los tests ya tienen outputs o faltan inputs).")
        return

    beat_tracking = not no_beat_tracking
    print(f"\n[eval_all] Transcribiendo {len(audio_bytes_list)} tests en Modal ({DEFAULT_GPU}) (beat_tracking={beat_tracking}) …\n")
    t0 = time.time()
    results = transcribe_batch.remote(audio_bytes_list, beat_tracking=beat_tracking)

    ok = 0
    for (td, out_mid), midi_bytes in zip(valid_dirs, results):
        if midi_bytes:
            out_mid.write_bytes(midi_bytes)
            print(f"[eval_all] ✓ {td.name} → {out_mid.name} ({len(midi_bytes)} bytes)")
            ok += 1
        else:
            print(f"[eval_all] ✗ {td.name} — transcripción fallida")

    elapsed = time.time() - t0
    print(f"\n[eval_all] Completado: {ok}/{len(audio_bytes_list)} OK en {elapsed:.0f}s")
    print(f"[eval_all] Outputs en: {eval_path.resolve()}")
    print("[eval_all] Siguiente paso: bash ../evaluation/render_mp3.sh && python ../evaluation/compute_f1.py")
