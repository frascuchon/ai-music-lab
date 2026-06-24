"""
Modal.com inference app para el pipeline compuesto v2 (Audio→MIDI).

Arquitectura: Separación de fuentes (Demucs) + ADTOF para batería + YourMT3+ para el resto

Pipeline:
    [audio.wav]
        │
        ├─→ YourMT3+ (audio completo) ──────────→ pitched.mid (multi-instrumento)
        │
        └─→ Demucs htdemucs_6s (drums stem)
                ↓
            drums.wav
                ↓
            ADTOF ─────────────────────────────→ drums.mid (kick, snare, hihat, toms, cymbals)
                                                          ↓
                                              merge MIDI local (pitched + drums)
                                                          ↓
                                              transcribed_cuda.mid

Mejora sobre compound v1 (Demucs + BasicPitch):
    v1 tenía F1=31.1% en Slakh #1884 porque BasicPitch sobre-detecta notas en stems
    y no está diseñado para batería. v2 preserva YourMT3+ (F1=77.5%) y añade ADTOF
    como especialista de drums.

Nota sobre YourMT3+: recibe el audio ORIGINAL completo (no stems). Esto evita el
leakage de Demucs que hundió v1. ADTOF recibe el stem drums.wav de Demucs.

Nota de arquitectura Modal:
    El orquestador llama a las otras apps IMPORTANDO sus módulos directamente
    (no via modal.Function.from_name, que requiere modal deploy).
    Esto funciona dentro del local_entrypoint de modal run.

Requisitos previos:
    modal run research_yourmt3_modal.py::setup  # descarga ckpt (~562MB, una vez)
    modal run research_adtof_modal.py::setup    # verifica instalación ADTOF

Setup (verifica Demucs):
    modal run research_compound_v2_modal.py::setup

Benchmark completo (4 tests con GT):
    modal run research_compound_v2_modal.py::eval_all \\
        --eval-dir ../evaluation/compound_v2 --only 4,5,7,8

GPU: A10G para Demucs (default). YourMT3+ y ADTOF corren en sus propias imágenes/GPUs.
     Override: env COMPOUND_V2_GPU=T4.

Coste estimado:
    - Demucs A10G: ~$0.02-0.05 por test
    - YourMT3+ A10G: ~$0.10-0.20 por test
    - ADTOF T4: ~$0.01-0.02 por test
    - 4 tests: ~$0.50-1.00 total
"""

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
# Container image — solo Demucs + pretty_midi para el merge
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
# Helpers de separación — solo Demucs
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


# ---------------------------------------------------------------------------
# Merge MIDI — función Python pura (sin Modal, corre localmente)
# ---------------------------------------------------------------------------
def _merge_midis_local(pitched_bytes: bytes, drums_bytes: bytes) -> bytes:
    """
    Fusiona MIDI pitched (YourMT3+) + MIDI drums (ADTOF) en un único archivo.
    Corre en el entrypoint local (Python puro, no en container).
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
# Modal function: Demucs — extrae solo el stem de batería
# ---------------------------------------------------------------------------
@app.function(
    timeout=3600,
    gpu=DEFAULT_GPU,
    memory=16384,
)
def separate_drums_batch(audio_payloads: list[bytes]) -> list[bytes]:
    """
    Separa stem de batería de cada audio con Demucs htdemucs_6s.

    audio_payloads: bytes de cada archivo WAV
    Returns: list[drums_wav_bytes] — b"" si Demucs falló
    """
    results = []
    for i, audio_bytes in enumerate(audio_payloads):
        t0 = time.time()
        try:
            stems = _separate_stems(audio_bytes)
            drums_wav = stems.get("drums", b"")
            if not drums_wav:
                raise RuntimeError("Demucs no produjo stem 'drums'")
            elapsed = time.time() - t0
            print(f"[demucs] [{i+1}/{len(audio_payloads)}] OK — drums.wav {len(drums_wav)/1e6:.1f} MB ({elapsed:.1f}s)")
            results.append(drums_wav)
        except Exception as exc:
            print(f"[demucs] [{i+1}/{len(audio_payloads)}] ERROR: {exc}")
            results.append(b"")
    return results


# ---------------------------------------------------------------------------
# Setup — verifica Demucs
# ---------------------------------------------------------------------------
@app.function(timeout=300, gpu=DEFAULT_GPU)
def setup():
    """
    Verifica que Demucs está instalado correctamente.

    Ejecutar antes del primer eval_all:
        modal run research_compound_v2_modal.py::setup
    """
    import demucs.api
    print("[setup] Demucs OK")
    sep = demucs.api.Separator(model=DEMUCS_MODEL, device="cpu")
    stems = sep.model.sources
    print(f"[setup] Demucs htdemucs_6s stems: {stems}")
    print("[setup] Compound v2 listo. Para YourMT3+ y ADTOF, ejecutar sus setup individuales.")


# ---------------------------------------------------------------------------
# Entrypoints — orquestación en el contexto local (no en container)
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
        print("ERROR: --audio-path requerido.")
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

    # Importar funciones Modal de los otros módulos directamente
    from research_yourmt3_modal import transcribe_batch as yt3_fn
    from research_adtof_modal import transcribe_batch as adtof_fn

    audio_bytes = audio_p.read_bytes()

    print(f"[main] 1/3 Separando drum stem (Demucs)…")
    [drums_wav] = separate_drums_batch.remote([audio_bytes])

    print(f"[main] 2/3 YourMT3+ (audio completo, {len(audio_bytes)/1e6:.1f} MB)…")
    [pitched_midi_bytes] = yt3_fn.remote([audio_bytes])

    print(f"[main] 3/3 ADTOF (drum stem, {len(drums_wav)/1e6:.1f} MB)…")
    [drums_midi_bytes] = adtof_fn.remote([drums_wav])

    midi_bytes = _merge_midis_local(pitched_midi_bytes, drums_midi_bytes)
    if not midi_bytes:
        print("[main] ERROR: merge devolvió vacío.")
        raise SystemExit(1)

    out_mid.write_bytes(midi_bytes)
    print(f"[main] Guardado: {out_mid} ({len(midi_bytes)} bytes)")


@app.local_entrypoint()
def eval_all(
    eval_dir: str = "../evaluation/compound_v2",
    force: bool = False,
    only: str = "",
):
    """
    Transcribe todos los tests con compound v2 (Demucs+ADTOF+YourMT3+).

    Importa las funciones Modal de research_yourmt3_modal y research_adtof_modal
    directamente (sin modal deploy) y las llama desde el entrypoint local.

    Ejemplos:
        modal run research_compound_v2_modal.py::eval_all --only 4,5,7,8
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
        print("[eval_all] Nada que transcribir.")
        return

    n = len(audio_bytes_list)
    print(f"\n[eval_all] {n} tests a procesar con compound v2.\n")

    # Importar funciones Modal directamente (funciona sin modal deploy en modal run)
    from research_yourmt3_modal import transcribe_batch as yt3_fn
    from research_adtof_modal import transcribe_batch as adtof_fn

    t0 = time.time()

    # Paso 1: Demucs — extrae stem de batería para todos los tests
    print(f"[eval_all] PASO 1/3 — Demucs htdemucs_6s (drum stems)…")
    t1 = time.time()
    drum_wavs = separate_drums_batch.remote(audio_bytes_list)
    print(f"[eval_all] Demucs completado en {time.time()-t1:.0f}s")

    # Paso 2: YourMT3+ — transcripción pitched sobre audio completo
    print(f"\n[eval_all] PASO 2/3 — YourMT3+ (pitched, audio completo)…")
    t2 = time.time()
    pitched_midis = yt3_fn.remote(audio_bytes_list)
    print(f"[eval_all] YourMT3+ completado en {time.time()-t2:.0f}s")

    # Paso 3: ADTOF — transcripción de batería sobre drum stems
    print(f"\n[eval_all] PASO 3/3 — ADTOF (drums sobre stem separado)…")
    t3 = time.time()
    drums_midis = adtof_fn.remote(drum_wavs)
    print(f"[eval_all] ADTOF completado en {time.time()-t3:.0f}s")

    # Merge local y guardar resultados
    print(f"\n[eval_all] Merge MIDI y guardando resultados…")
    ok = 0
    for i, (td, out_mid) in enumerate(valid_dirs):
        try:
            midi_bytes = _merge_midis_local(
                pitched_midis[i] if i < len(pitched_midis) else b"",
                drums_midis[i] if i < len(drums_midis) else b"",
            )
            if midi_bytes:
                out_mid.write_bytes(midi_bytes)
                print(f"[eval_all] ✓ {td.name} → {out_mid.name} ({len(midi_bytes)} bytes)")
                ok += 1
            else:
                print(f"[eval_all] ✗ {td.name} — merge devolvió vacío")
        except Exception as exc:
            print(f"[eval_all] ✗ {td.name} — ERROR: {exc}")

    elapsed = time.time() - t0
    print(f"\n[eval_all] Completado: {ok}/{n} OK en {elapsed:.0f}s")
    print(f"[eval_all] Outputs en: {eval_path.resolve()}")
    print("[eval_all] Siguiente paso:")
    print("  bash ../evaluation/render_mp3.sh")
    print("  uv run python ../evaluation/compute_f1.py --only compound_v2/")
