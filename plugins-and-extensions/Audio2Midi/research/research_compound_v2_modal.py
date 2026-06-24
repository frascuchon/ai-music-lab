"""
Modal.com inference app para el pipeline compuesto v2 (Audio→MIDI).

Arquitectura: Separación de fuentes (Demucs) + ADTOF para batería + YourMT3+ para el resto

Pipeline:
    [audio.wav]
        │
        ├─→ YourMT3+ (audio completo) ──────────→ pitched.mid (multi-instrumento, sin drums)
        │
        └─→ Demucs htdemucs_6s (drums stem)
                ↓
            drums.wav
                ↓
            ADTOF ─────────────────────────────→ drums.mid (kick, snare, hihat, toms, cymbals)
                                                          ↓
                                              merge MIDI (pitched + drums)
                                                          ↓
                                              transcribed_cuda.mid

Mejora sobre compound v1 (Demucs + BasicPitch):
    v1 tenía F1=31.1% en Slakh #1884 porque BasicPitch sobre-detecta notas en stems
    y no está diseñado para batería. v2 preserva YourMT3+ (F1=77.5%) y añade ADTOF
    como especialista de drums.

Nota sobre YourMT3+: recibe el audio ORIGINAL completo (no stems). Esto evita el
leakage de Demucs que hundió v1. YourMT3+ puede detectar o no batería según el
audio, pero la batería dedicada viene de ADTOF sobre el stem limpio.

Requisitos previos:
    modal run research_yourmt3_modal.py::setup  # descarga ckpt (~562MB, una vez)
    modal run research_adtof_modal.py::setup    # verifica instalación ADTOF

Setup (verifica dependencias en compound v2):
    modal run research_compound_v2_modal.py::setup

Smoke test:
    modal run research_compound_v2_modal.py::main \\
        --audio-path ../evaluation/yourmt3/test04/input.wav \\
        --out-dir ../evaluation/compound_v2/test04

Benchmark completo (4 tests con GT):
    modal run research_compound_v2_modal.py::eval_all \\
        --eval-dir ../evaluation/compound_v2 --only 4,5,7,8

GPU: A10G para Demucs (default). YourMT3+ y ADTOF se invocan como apps Modal separadas.
     Override: env COMPOUND_V2_GPU=T4.

Coste estimado (A10G):
    - Demucs por test: ~$0.02-0.05
    - YourMT3+ (A10G): ~$0.10-0.20 por test (facturado por esa app)
    - ADTOF (T4): ~$0.01-0.02 por test (facturado por esa app)
    - 4 tests: ~$0.50-1.00 total
"""

import io
import os
import re
import sys
import time
import tempfile
import shutil
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_GPU = os.environ.get("COMPOUND_V2_GPU", "A10G")
DEMUCS_MODEL = "htdemucs_6s"

# ---------------------------------------------------------------------------
# Container image — solo necesita Demucs + pretty_midi + mido para el merge
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "sox", "libsndfile1"])
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "demucs>=4.0",
        "mido>=1.3",
        "pretty_midi>=0.2.10",
        "librosa>=0.10",
        "soundfile>=0.12",
        "numpy>=1.24",
        "psutil>=6.0",
    )
)

app = modal.App("compound-v2-pipeline", image=image)


# ---------------------------------------------------------------------------
# Helpers — separación de stems y merge MIDI
# ---------------------------------------------------------------------------
def _separate_stems(audio_bytes: bytes) -> dict[str, bytes]:
    """
    Separa un audio en stems usando Demucs htdemucs_6s.
    Returns: dict stem_name → WAV bytes
    """
    from demucs.separate import main as demucs_main

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    out_dir = tempfile.mkdtemp()
    try:
        demucs_main(["-n", DEMUCS_MODEL, "--out", out_dir,
                     "--device", "cuda", audio_path])

        base_name = Path(audio_path).stem
        stems_dir = Path(out_dir) / DEMUCS_MODEL / base_name
        if not stems_dir.exists():
            raise RuntimeError(f"Demucs no generó salida en {stems_dir}")

        result = {}
        for stem_file in sorted(stems_dir.glob("*.wav")):
            result[stem_file.stem] = stem_file.read_bytes()
        return result
    finally:
        os.unlink(audio_path)
        shutil.rmtree(out_dir, ignore_errors=True)


def _merge_pitched_drums(pitched_bytes: bytes, drums_bytes: bytes) -> bytes:
    """
    Fusiona el MIDI pitched (YourMT3+) y el MIDI de batería (ADTOF) en un único archivo.
    Los tracks de cada MIDI se conservan sin modificación.
    """
    import pretty_midi

    combined = pretty_midi.PrettyMIDI()

    for label, midi_bytes in [("pitched", pitched_bytes), ("drums", drums_bytes)]:
        if not midi_bytes:
            print(f"[merge] Advertencia: MIDI {label} vacío — omitiendo")
            continue
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            f.write(midi_bytes)
            tmp_path = f.name
        try:
            pm = pretty_midi.PrettyMIDI(tmp_path)
            combined.instruments.extend(pm.instruments)
        except Exception as exc:
            print(f"[merge] ERROR cargando MIDI {label}: {exc}")
        finally:
            os.unlink(tmp_path)

    buf = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    buf.close()
    try:
        combined.write(buf.name)
        with open(buf.name, "rb") as f:
            return f.read()
    finally:
        os.unlink(buf.name)


# ---------------------------------------------------------------------------
# Setup — verifica Demucs y que las apps YourMT3+/ADTOF son accesibles
# ---------------------------------------------------------------------------
@app.function(timeout=300, gpu=DEFAULT_GPU)
def setup():
    """
    Verifica que Demucs está instalado y que las apps remotas existen.

    Ejecutar antes del primer eval_all:
        modal run research_compound_v2_modal.py::setup
    """
    import demucs.api
    print("[setup] Demucs OK")

    sep = demucs.api.Separator(model=DEMUCS_MODEL, device="cpu")
    stems = sep.model.sources
    print(f"[setup] Demucs htdemucs_6s stems: {stems}")

    # Verificar que las apps remotas existen (lookup síncrono)
    yourmt3_fn = modal.Function.from_name("yourmt3-inference", "transcribe_batch")
    adtof_fn = modal.Function.from_name("adtof-inference", "transcribe_batch")
    print(f"[setup] YourMT3+ lookup OK: {yourmt3_fn}")
    print(f"[setup] ADTOF lookup OK:    {adtof_fn}")

    print("[setup] Todo OK — compound v2 listo.")


# ---------------------------------------------------------------------------
# Función Modal principal
# ---------------------------------------------------------------------------
@app.function(
    timeout=7200,
    gpu=DEFAULT_GPU,
    memory=16384,
)
def transcribe_batch(audio_payloads: list[bytes]) -> list[bytes]:
    """
    Pipeline completo compound v2: Demucs drums → ADTOF, audio completo → YourMT3+, merge.

    audio_payloads: bytes de cada archivo WAV/MP3
    Returns: list[midi_bytes] — b"" si el pipeline falló para ese audio
    """
    yourmt3_fn = modal.Function.from_name("yourmt3-inference", "transcribe_batch")
    adtof_fn = modal.Function.from_name("adtof-inference", "transcribe_batch")

    results = []
    for i, audio_bytes in enumerate(audio_payloads):
        t0 = time.time()
        print(f"\n[compound_v2] [{i+1}/{len(audio_payloads)}] Iniciando pipeline …")
        try:
            # 1. Lanzar YourMT3+ en paralelo mientras Demucs trabaja
            print(f"[compound_v2]   Spawnando YourMT3+ (audio completo) …")
            yt3_handle = yourmt3_fn.spawn([audio_bytes])

            # 2. Separar stems con Demucs (A10G) — transcurre en paralelo con YourMT3+
            print(f"[compound_v2]   Separando stems con Demucs htdemucs_6s …")
            t1 = time.time()
            stems = _separate_stems(audio_bytes)
            drums_wav = stems.get("drums", b"")
            print(f"[compound_v2]   Stems: {list(stems.keys())} ({time.time()-t1:.1f}s)")

            if not drums_wav:
                raise RuntimeError("Demucs no produjo stem 'drums'")

            # 3. ADTOF sobre el stem de batería
            print(f"[compound_v2]   Transcribiendo batería con ADTOF …")
            t2 = time.time()
            [drums_midi_bytes] = adtof_fn.remote([drums_wav])
            print(f"[compound_v2]   ADTOF: {len(drums_midi_bytes)} bytes ({time.time()-t2:.1f}s)")

            # 4. Recoger resultado de YourMT3+
            print(f"[compound_v2]   Esperando YourMT3+ …")
            t3 = time.time()
            [pitched_midi_bytes] = yt3_handle.get()
            print(f"[compound_v2]   YourMT3+: {len(pitched_midi_bytes)} bytes ({time.time()-t3:.1f}s)")

            # 5. Merge
            midi_bytes = _merge_pitched_drums(pitched_midi_bytes, drums_midi_bytes)
            elapsed = time.time() - t0
            print(f"[compound_v2] [{i+1}] OK — {len(midi_bytes)} bytes MIDI en {elapsed:.1f}s")
            results.append(midi_bytes)

        except Exception as exc:
            print(f"[compound_v2] [{i+1}] ERROR: {exc}")
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
):
    """
    Transcribe un audio con compound v2 y guarda el MIDI resultante.

    Ejemplo:
        modal run research_compound_v2_modal.py::main \\
            --audio-path ../evaluation/yourmt3/test04/input.wav \\
            --out-dir ../evaluation/compound_v2/test04
    """
    if not audio_path:
        print("ERROR: --audio-path requerido. Ejemplo:")
        print("  modal run research_compound_v2_modal.py::main \\")
        print("      --audio-path ../evaluation/yourmt3/test04/input.wav \\")
        print("      --out-dir ../evaluation/compound_v2/test04")
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

    print(f"[main] Compound v2: {audio_p.name} → {out_mid}")
    [midi_bytes] = transcribe_batch.remote([audio_p.read_bytes()])

    if not midi_bytes:
        print("[main] ERROR: el pipeline devolvió vacío.")
        raise SystemExit(1)

    out_mid.write_bytes(midi_bytes)
    print(f"[main] Guardado: {out_mid} ({len(midi_bytes)} bytes)")


# ---------------------------------------------------------------------------
# Entrypoint: todos los tests del directorio de evaluación
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/compound_v2",
    force: bool = False,
    only: str = "",
):
    """
    Transcribe todos los tests con compound v2.

    Ejemplos:
        # Solo tests con GT (Slakh + MusicNet):
        modal run research_compound_v2_modal.py::eval_all --only 4,5,7,8

        # Forzar re-transcripción:
        modal run research_compound_v2_modal.py::eval_all --force
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

    print(f"\n[eval_all] Procesando {len(audio_bytes_list)} tests con compound v2 ({DEFAULT_GPU}) …\n")
    t0 = time.time()
    results = transcribe_batch.remote(audio_bytes_list)

    ok = 0
    for (td, out_mid), midi_bytes in zip(valid_dirs, results):
        if midi_bytes:
            out_mid.write_bytes(midi_bytes)
            print(f"[eval_all] ✓ {td.name} → {out_mid.name} ({len(midi_bytes)} bytes)")
            ok += 1
        else:
            print(f"[eval_all] ✗ {td.name} — pipeline fallido")

    elapsed = time.time() - t0
    print(f"\n[eval_all] Completado: {ok}/{len(audio_bytes_list)} OK en {elapsed:.0f}s")
    print(f"[eval_all] Outputs en: {eval_path.resolve()}")
    print("[eval_all] Siguiente paso:")
    print("  bash ../evaluation/render_mp3.sh")
    print("  uv run python ../evaluation/compute_f1.py --only compound_v2/")
