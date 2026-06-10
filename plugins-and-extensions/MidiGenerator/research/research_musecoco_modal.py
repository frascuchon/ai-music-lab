"""
Modal.com inference app para MuseCoco (Microsoft/muzic, MIT License).

Generación MIDI multi-track desde texto via dos etapas:
  Stage 1 — text2attribute : texto → atributos musicales (CPU, transformers 4.26)
  Stage 2 — attribute2music: atributos → tokens REMI2 → MIDI  (GPU, fairseq 0.10.2)

Modelos en HuggingFace:
  XinXuNLPer/MuseCoco_text2attribute   (~1.4 GB)
  XinXuNLPer/MuseCoco_attribute2music  (~14.5 GB)

Dependencias clave: fairseq==0.10.2 + pytorch-fast-transformers (CUDA kernels)
→ requieren Python 3.8 + PyTorch 1.11 + CUDA 11.3 devel image.
→ Modal runner requiere Python 3.10+: se instala Python 3.11 (add_python) para Modal
  y se mantiene conda Python 3.8 para la inferencia MuseCoco via subprocess.
→ Los pesos se guardan en un Modal Volume (se descargan solo la primera vez).

Setup (primera vez, descarga ~16 GB al Volume):
    modal run research_musecoco_modal.py::setup_weights

Inferencia (::main requerido porque hay dos local_entrypoints):
    modal run research_musecoco_modal.py::main --prompt "jazz piano trio, 120 BPM" --out out_muse.mid
    modal run research_musecoco_modal.py::main --prompt "..." --n_samples 2 --out out_muse.mid

Coste estimado (L4 24GB, $0.80/hr):
  ~7 min/generación  →  ~$0.09 por track
  T4 (16GB) era más barato/hr pero ~22 min/gen por presión de VRAM → L4 más barato por generación

Alternativas a Modal si buscas más barato (sin créditos gratuitos):
  RunPod    $0.09-0.20/hr (A4000/A5000), pago desde el día 1, muy económico
  Vast.ai   $0.20-0.40/hr (A100), spot market, sin garantía de uptime

Referencia: https://github.com/microsoft/muzic/tree/main/musecoco
Paper: arXiv:2306.00110  (AAAI 2024)
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Volumes — pesos persistentes entre ejecuciones (16 GB total, descarga única)
# ---------------------------------------------------------------------------
weights_vol = modal.Volume.from_name("musecoco-weights", create_if_missing=True)
WEIGHTS_MOUNT = "/vol/weights"
A2M_CKPT = f"{WEIGHTS_MOUNT}/attribute2music.pt"
T2A_MODEL_DIR = f"{WEIGHTS_MOUNT}/text2attribute"

# ---------------------------------------------------------------------------
# Container image
# pytorch:1.11.0-cuda11.3-cudnn8-devel incluye NVCC → necesario para compilar
# pytorch-fast-transformers (kernels CUDA de linear attention).
#
# Conflicto de versiones Python:
#   - Modal runner necesita Python ≥ 3.10
#   - fairseq 0.10.2 + pytorch-fast-transformers solo funcionan con Python 3.8
# Solución: add_python="3.11" instala Python 3.11 para el runner de Modal;
# las deps de MuseCoco se instalan en conda Python 3.8 (/opt/conda/bin/pip),
# y los subprocesos de inferencia lo invocan como /opt/conda/bin/python.
# ---------------------------------------------------------------------------
MUZIC_DIR = "/opt/muzic/musecoco"
STAGE1_DIR = f"{MUZIC_DIR}/1-text2attribute_model"
STAGE2_DIR = f"{MUZIC_DIR}/2-attribute2music_model"

# Python 3.8 (conda) en el devel image — usado por los subprocesos de inferencia.
# IMPORTANTE: usar python3.8 explícito (no python/python3) porque sus symlinks
# en /opt/conda/bin/ se sobreescriben para que apunten a Python 3.11.
CONDA_PYTHON = "/opt/conda/bin/python3.8"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel",
        # Python 3.11 para el runner de Modal (requiere ≥ 3.10)
        add_python="3.11",
    )
    # Ubuntu 18.04 + CUDA 11.3 repos tienen claves GPG rotadas → apt-get update falla.
    # Eliminamos esas fuentes (CUDA ya está instalado en la imagen); sólo necesitamos
    # los repos base de Ubuntu para git/g++.
    # Además, redirigimos los symlinks python/python3 de conda a Python 3.11:
    # add_python instala 3.11 en /usr/local/bin/, pero /opt/conda/bin/ va primero
    # en PATH, por lo que sin esta corrección el runner sigue usando Python 3.8.
    .run_commands(
        "rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list",
        "ln -sf /usr/local/bin/python3.11 /opt/conda/bin/python",
        "ln -sf /usr/local/bin/python3.11 /opt/conda/bin/python3",
    )
    .apt_install(["git", "g++", "build-essential"])
    # pytorch-fast-transformers: linear attention CUDA kernels usados por MuseCoco.
    # Debe compilarse con Python 3.8 + NVCC del devel image (~5 min, cacheado).
    .run_commands(
        f"{CONDA_PYTHON} -m pip install pytorch-fast-transformers",
    )
    # Deps de inferencia MuseCoco → conda Python 3.8 (fairseq 0.10.2 solo soporta 3.8)
    .run_commands(
        f"{CONDA_PYTHON} -m pip install"
        " fairseq==0.10.2"
        " transformers==4.26.0"
        " datasets==2.14.6"
        " absl-py"
        " accelerate"
        " protobuf==3.20.3"
        " tqdm"
        " scikit-learn"
        " miditoolkit==0.1.16"
        " scipy"
        " numpy==1.23.4"
        " music21"
        " msgpack"
        " huggingface_hub"
        " pretty_midi"
        " psutil",
        # sentencepiece aparte para evitar problemas de escaping con !
        f"{CONDA_PYTHON} -m pip install 'sentencepiece!=0.1.92'",
    )
    # Deps de las funciones Modal (download_weights, logging) → Python 3.11
    .pip_install(
        "huggingface_hub",
        "pretty_midi",
        "tqdm",
    )
    # Clonar muzic repo (código personalizado fairseq + midiprocessor incluido)
    .run_commands(
        "git clone --depth=1 https://github.com/microsoft/muzic.git /opt/muzic",
    )
)

app = modal.App("musecoco-inference", image=image)


# ---------------------------------------------------------------------------
# Función de setup — descarga los pesos al Volume (ejecutar solo una vez)
# ---------------------------------------------------------------------------
@app.function(
    cpu=4,
    memory=8192,
    timeout=7200,  # 2h para los 16 GB
    volumes={WEIGHTS_MOUNT: weights_vol},
)
def download_weights() -> None:
    """Descarga ambos modelos MuseCoco al Volume. Ejecutar una sola vez."""
    import time
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(WEIGHTS_MOUNT, exist_ok=True)

    # --- text2attribute (~1.4 GB via snapshot, solo pytorch_model.bin) ---
    t0 = time.time()
    if not os.path.exists(f"{T2A_MODEL_DIR}/pytorch_model.bin"):
        print("Descargando MuseCoco_text2attribute (~1.4 GB)...")
        snapshot_download(
            repo_id="XinXuNLPer/MuseCoco_text2attribute",
            local_dir=T2A_MODEL_DIR,
            ignore_patterns=["optimizer.pt", "rng_state.pth"],  # omitir estado del optimizador
        )
        print(f"  text2attribute descargado en {time.time()-t0:.0f}s")
    else:
        print("text2attribute ya en Volume, saltando.")

    # --- attribute2music (~14.5 GB checkpoint único) ---
    t1 = time.time()
    if not os.path.exists(A2M_CKPT):
        print("Descargando MuseCoco_attribute2music (~14.5 GB)...")
        hf_hub_download(
            repo_id="XinXuNLPer/MuseCoco_attribute2music",
            filename="attribute2music.pt",
            local_dir=WEIGHTS_MOUNT,
        )
        print(f"  attribute2music descargado en {time.time()-t1:.0f}s")
    else:
        print("attribute2music ya en Volume, saltando.")

    weights_vol.commit()
    print("Volume sincronizado.")


# ---------------------------------------------------------------------------
# Función principal de inferencia
# ---------------------------------------------------------------------------
@app.function(
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=1800,
    volumes={WEIGHTS_MOUNT: weights_vol},
)
def generate(prompt: str, n_samples: int = 1, instruments: list[int] | None = None) -> list[str]:
    """
    Genera n_samples MIDIs desde un prompt. Guarda en el Volume y retorna
    rutas relativas (output/<job_id>/sample_N.mid) para que main() las
    descargue con 'modal volume get'. Así el cliente local no necesita
    mantener una conexión gRPC de 11 min (stage2 tarda ~664s en T4).
    """
    import pickle
    import shutil
    import tempfile
    import time
    from copy import deepcopy

    weights_vol.reload()

    if not os.path.exists(A2M_CKPT):
        raise RuntimeError(
            "Pesos no encontrados. Ejecuta primero:\n"
            "  modal run research_musecoco_modal.py::setup_weights"
        )

    stage2_linear = f"{STAGE2_DIR}/linear_mask"

    data_bin_dir = f"{STAGE2_DIR}/data/truncated_2560/data-bin"
    att_key_path = f"{STAGE1_DIR}/data/att_key.json"
    num_labels_path = f"{STAGE1_DIR}/num_labels.json"
    inference_script = f"{stage2_linear}/interactive_dict_v5_1billion.py"

    # Directorio de salida persistente en el Volume
    job_id = str(int(time.time()))
    vol_output_dir = f"{WEIGHTS_MOUNT}/output/{job_id}"
    os.makedirs(vol_output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # -----------------------------------------------------------------------
        # STAGE 1: text → attributes
        # Reproduce predict.sh: python main.py --do_predict ...
        # -----------------------------------------------------------------------
        t0 = time.time()
        print(f"[stage1] text→attributes | prompt='{prompt}'")

        # predict.json: formato esperado por main.py de MuseCoco
        predict_json = [{"text": prompt}]
        predict_json_path = f"{tmpdir}/predict.json"
        with open(predict_json_path, "w") as f:
            json.dump(predict_json, f)

        stage1_out_dir = f"{tmpdir}/stage1_out"
        os.makedirs(stage1_out_dir, exist_ok=True)

        cmd_stage1 = [
            CONDA_PYTHON, f"{STAGE1_DIR}/main.py",
            "--do_predict",
            f"--model_name_or_path={T2A_MODEL_DIR}",
            f"--test_file={predict_json_path}",
            f"--attributes={att_key_path}",
            f"--num_labels={num_labels_path}",
            f"--output_dir={stage1_out_dir}",
            "--overwrite_output_dir",
        ]
        result = subprocess.run(
            cmd_stage1,
            cwd=STAGE1_DIR,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("[stage1] stdout:", result.stdout[-2000:])
            print("[stage1] stderr:", result.stderr[-2000:])
            raise RuntimeError(f"Stage 1 falló con código {result.returncode}")

        predict_attrs_path = f"{stage1_out_dir}/predict_attributes.json"
        softmax_probs_path = f"{stage1_out_dir}/softmax_probs.json"
        if not os.path.exists(predict_attrs_path):
            raise RuntimeError(f"No se generó predict_attributes.json en {stage1_out_dir}")

        t_s1 = time.time() - t0
        print(f"[stage1] completado en {t_s1:.1f}s")

        # Debug: mostrar atributos predichos por stage 1
        _dbg_attrs = json.load(open(predict_attrs_path))
        _att_key_all = json.load(open(att_key_path))
        _I1s2_keys = [a for a in _att_key_all if a[:4] == "I1s2"]
        _INST_NAMES = [
            "piano","keyboard","percussion","organ","guitar","bass",
            "violin","viola","cello","harp","strings","voice",
            "trumpet","trombone","tuba","horn","brass","sax",
            "oboe","bassoon","clarinet","piccolo","flute","pipe",
            "synthesizer","ethnic","sound_effect","drum",
        ]
        active_instr = []
        for ki, k in enumerate(_I1s2_keys):
            label = _dbg_attrs.get(k, [[0]])[0]
            label_val = label[0] if isinstance(label, list) else label
            if label_val == 1:  # vector [p_yes, p_no, p_na]: posición 0 = yes → label_val=1 significa yes
                name = _INST_NAMES[ki] if ki < len(_INST_NAMES) else f"class_{ki}"
                active_instr.append(f"{name}(class={ki})")
        print(f"[stage1] instrumentos predichos: {active_instr if active_instr else 'ninguno (solo piano por defecto?)'}")
        # Mostrar los demás atributos clave
        for attr in ["R1", "R3", "S4", "B1s1", "TS1s1", "K1", "T1s1", "EM1"]:
            val = _dbg_attrs.get(attr)
            if val is not None:
                print(f"[stage1]   {attr}: {val[0]}")

        # -----------------------------------------------------------------------
        # STAGE 1→2 PRE: atributos → infer_test.bin
        # Reproduce la lógica de stage2_pre.py (hardcoded paths → aquí dinámicos)
        # -----------------------------------------------------------------------
        t1 = time.time()
        print("[pre] generando infer_test.bin...")

        test_data = json.load(open(predict_json_path))
        pred_attrs = json.load(open(predict_attrs_path))
        softmax_probs = json.load(open(softmax_probs_path))
        att_key = json.load(open(att_key_path))

        final = []
        for line in test_data:
            ins = {
                "text": line["text"],
                "pred_labels": {},
                "pred_probs": {},
            }
            final.append(deepcopy(ins))

        for k, v in pred_attrs.items():
            for j in range(len(v)):
                final[j]["pred_labels"][k] = deepcopy(v[j])
        for k, v in softmax_probs.items():
            for j in range(len(v)):
                final[j]["pred_probs"][k] = deepcopy(v[j])

        # Override de instrumentos: inyectar I1s2 directamente sin pasar por stage 1.
        # instruments = lista de class IDs (0-27) a activar. El resto se fuerzan a "no".
        if instruments:
            print(f"[override] forzando instrumentos: {instruments}")
            I1s2_key_all = [a for a in att_key if a[:4] == "I1s2"]
            for idx in range(len(final)):
                for ki, k in enumerate(I1s2_key_all):
                    # Formato del vector: [p_yes, p_no, p_na] — índice 0 = "yes"
                    # Activar = [1,0,0] (yes), desactivar = [0,1,0] (no)
                    if ki in instruments:
                        final[idx]["pred_labels"][k] = [1, 0, 0]
                        final[idx]["pred_probs"][k] = [0.90, 0.05, 0.05]
                    else:
                        final[idx]["pred_labels"][k] = [0, 1, 0]
                        final[idx]["pred_probs"][k] = [0.05, 0.90, 0.05]

        # Agrupar atributos I1s2 y S4 (igual que stage2_pre.py original)
        I1s2_key = [a for a in att_key if a[:4] == "I1s2"]
        S4_key = [a for a in att_key if a[:2] == "S4"]

        for idx in range(len(final)):
            pred_labels_I1s2 = [deepcopy(final[idx]["pred_labels"].pop(k)) for k in I1s2_key]
            pred_probs_I1s2 = [deepcopy(final[idx]["pred_probs"].pop(k)) for k in I1s2_key]
            pred_labels_S4 = [deepcopy(final[idx]["pred_labels"].pop(k)) for k in S4_key]
            pred_probs_S4 = [deepcopy(final[idx]["pred_probs"].pop(k)) for k in S4_key]
            final[idx]["pred_probs"]["I1s2"] = pred_probs_I1s2
            final[idx]["pred_probs"]["S4"] = pred_probs_S4
            final[idx]["pred_labels"]["I1s2"] = pred_labels_I1s2
            final[idx]["pred_labels"]["S4"] = pred_labels_S4

        infer_bin_path = f"{tmpdir}/infer_test.bin"
        with open(infer_bin_path, "wb") as f:
            pickle.dump(final, f)

        t_pre = time.time() - t1
        print(f"[pre] infer_test.bin generado en {t_pre:.1f}s | {len(final)} samples")

        # -----------------------------------------------------------------------
        # STAGE 2: attribute → MIDI (fairseq GPU)
        # Reproduce interactive_1billion.sh con n_samples samples
        # -----------------------------------------------------------------------
        t2 = time.time()
        print("[stage2] attributes→MIDI | GPU inference...")

        save_root = f"{tmpdir}/generation"
        os.makedirs(save_root, exist_ok=True)

        cmd_stage2 = [
            CONDA_PYTHON, "-u", inference_script,
            data_bin_dir,
            "--task", "language_modeling_control",
            "--path", A2M_CKPT,
            "--ctrl_command_path", infer_bin_path,
            "--save_root", save_root,
            "--need_num", str(n_samples),
            "--start", "0",
            "--end", "1",
            "--max-len-b", "2560",
            "--min-len", "512",
            "--sampling",
            "--beam", "1",
            "--sampling-topk", "15",
            "--temperature", "1.0",
            "--no-repeat-ngram-size", "0",
            "--buffer-size", "1",
            "--batch-size", "1",
        ]
        result = subprocess.run(
            cmd_stage2,
            cwd=stage2_linear,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": f"{STAGE2_DIR}:{stage2_linear}:{os.environ.get('PYTHONPATH', '')}",
            },
        )
        if result.returncode != 0:
            print("[stage2] stdout:", result.stdout[-3000:])
            print("[stage2] stderr:", result.stderr[-3000:])
            raise RuntimeError(f"Stage 2 falló con código {result.returncode}")

        t_s2 = time.time() - t2
        print(f"[stage2] completado en {t_s2:.1f}s")

        # -----------------------------------------------------------------------
        # STAGE 3: REMI tokens → MIDI  (midiprocessor, Python 3.8)
        # interactive_dict_v5_1billion.py tiene except:pass que silencia errores
        # de decodificación. Hacemos la conversión explícitamente aquí.
        # -----------------------------------------------------------------------
        stage3_script = textwrap.dedent(f"""\
            import sys, pathlib
            sys.path.insert(0, '{STAGE2_DIR}')
            from midiprocessor import MidiDecoder

            save_root = pathlib.Path(sys.argv[1])
            decoder = MidiDecoder("REMIGEN2")
            remi_files = sorted(save_root.rglob("remi/*.txt"))
            if not remi_files:
                print("[stage3] ERROR: no remi/*.txt files found", flush=True)
                sys.exit(1)
            ok = 0
            for remi_txt in remi_files:
                midi_dir = remi_txt.parent.parent / "midi"
                midi_dir.mkdir(exist_ok=True)
                midi_path = midi_dir / (remi_txt.stem + ".mid")
                with open(remi_txt) as f:
                    tokens = f.read().strip().split()
                # strip attribute prefix: keep tokens after <sep>
                sep_indices = [i for i, t in enumerate(tokens) if t == "<sep>"]
                if sep_indices:
                    tokens = tokens[sep_indices[-1] + 1:]
                # strip leading/trailing special tokens
                tokens = [t for t in tokens if not t.startswith("<") or t == "<unk>"]
                try:
                    midi_obj = decoder.decode_from_token_str_list(tokens)
                    midi_obj.dump(str(midi_path))
                    print(f"[stage3] OK {{midi_path}}", flush=True)
                    ok += 1
                except Exception as e:
                    print(f"[stage3] ERROR {{remi_txt}}: {{e}}", flush=True)
                    raise
            print(f"[stage3] {{ok}}/{{len(remi_files)}} converted", flush=True)
        """)
        t3 = time.time()
        print("[stage3] REMI→MIDI conversion...")
        result3 = subprocess.run(
            [CONDA_PYTHON, "-c", stage3_script, save_root],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": f"{STAGE2_DIR}:{os.environ.get('PYTHONPATH', '')}"},
        )
        print(result3.stdout)
        if result3.returncode != 0:
            print("[stage3] stderr:", result3.stderr[-2000:])
            raise RuntimeError(f"Stage 3 (REMI→MIDI) falló con código {result3.returncode}")
        print(f"[stage3] completado en {time.time() - t3:.1f}s")

        # -----------------------------------------------------------------------
        # Recoger archivos MIDI generados y copiar al Volume
        # -----------------------------------------------------------------------
        # Listar árbol completo de save_root (debug: visible en container logs)
        all_files = list(Path(save_root).rglob("*"))
        print(f"[debug] {len(all_files)} entradas en save_root:")
        for p in sorted(all_files):
            print(f"  {p}")

        # Guardar stdout de stage2 en Volume para poder revisarlo offline
        with open(f"{vol_output_dir}/stage2_stdout.txt", "w") as lf:
            lf.write(result.stdout[-5000:])

        midi_files = sorted(Path(save_root).glob("**/*.mid"))
        if not midi_files:
            midi_files = sorted(Path(save_root).glob("**/*.midi"))
        if not midi_files:
            weights_vol.commit()
            raise RuntimeError(
                f"No se encontraron archivos .mid en {save_root}. "
                f"Revisa stage2_stdout.txt en el Volume (job_id={job_id})."
            )

        # Copiar MIDIs al Volume
        rel_paths = []
        for i, mf in enumerate(midi_files):
            rel = f"output/{job_id}/sample_{i+1}.mid"
            shutil.copy(str(mf), f"{WEIGHTS_MOUNT}/{rel}")
            rel_paths.append(rel)

        weights_vol.commit()
        print(f"[output] {len(rel_paths)} MIDI(s) guardados en el Volume")

        # Análisis desde el container (pretty_midi instalado en conda py3.8)
        try:
            import io
            import pretty_midi
            for i, mf in enumerate(midi_files):
                pm = pretty_midi.PrettyMIDI(str(mf))
                n_tracks = len(pm.instruments)
                n_notes = sum(len(inst.notes) for inst in pm.instruments)
                dur = pm.get_end_time()
                instrs = [inst.program for inst in pm.instruments]
                print(f"  [{i+1}] {n_tracks} pistas | {n_notes} notas | {dur:.1f}s | instrs: {instrs}")
        except Exception:
            pass

        print(f"\n--- TIEMPOS ---")
        print(f"  Stage 1: {t_s1:.1f}s  |  Pre: {t_pre:.1f}s  |  Stage 2: {t_s2:.1f}s")
        print(f"  TOTAL:   {t_s1 + t_pre + t_s2:.1f}s")

        return rel_paths


# ---------------------------------------------------------------------------
# Local entrypoint de setup
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def setup_weights() -> None:
    """Descarga los pesos de MuseCoco al Volume (ejecutar solo una vez)."""
    print("Descargando pesos de MuseCoco al Modal Volume...")
    print("  text2attribute: ~1.4 GB (pytorch_model.bin)")
    print("  attribute2music: ~14.5 GB (attribute2music.pt)")
    print("  Total: ~16 GB  |  Tiempo estimado: 20-40 min (red depende de Modal)")
    download_weights.remote()
    print("Setup completado. Ahora puedes ejecutar inferencias.")


# ---------------------------------------------------------------------------
# Local entrypoint de inferencia
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = "upbeat jazz piano trio, 120 BPM, C major, happy mood",
    out: str = "out_muse.mid",
    n_samples: int = 1,
    instruments: str = "",
) -> None:
    """
    Genera MIDI desde texto usando MuseCoco en Modal (T4 GPU).

    Usa spawn()+polling para que el cliente local no necesite mantener
    una conexión gRPC de 11 min (stage2 en T4 tarda ~664s). Los MIDIs se
    guardan en el Volume y se descargan con 'modal volume get'.

    Ejemplo:
        modal run research_musecoco_modal.py::main \\
            --prompt "jazz piano trio, upbeat, 120 BPM" \\
            --out out_muse.mid
    """
    import time

    # instruments: "0,5,27" → [0, 5, 27] (class IDs). Vacío = dejar que stage 1 decida.
    instr_list = [int(x.strip()) for x in instruments.split(",") if x.strip()] if instruments else None

    out_path = Path(out)
    print(f"Enviando a Modal [L4] | prompt='{prompt}' | n_samples={n_samples}")
    if instr_list:
        print(f"[override] instrumentos: {instr_list}")
    print("[spawn] Stage2 tarda ~11 min; usando spawn+poll para evitar heartbeat timeout")

    call = generate.spawn(prompt=prompt, n_samples=n_samples, instruments=instr_list)
    print(f"[spawned] object_id={call.object_id}")

    # Persistir call_id localmente para poder hacer recover si el cliente cae
    _state = {"call_id": call.object_id, "out": str(out_path), "prompt": prompt}
    Path(".musecoco_last_call.json").write_text(json.dumps(_state, indent=2))
    print(f"[recovery] estado guardado en .musecoco_last_call.json")

    # Polling cada 30s para no depender de una conexión gRPC larga
    rel_paths = None
    dots = 0
    while rel_paths is None:
        try:
            rel_paths = call.get(timeout=30)
        except TimeoutError:
            print(".", end="", flush=True)
            dots += 1
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("deadline", "timed out")):
                print(".", end="", flush=True)
                dots += 1
            else:
                raise
    if dots:
        print()

    if not rel_paths:
        print("[error] La función no retornó rutas MIDI.")
        return

    _download_and_report(rel_paths, out_path, prompt)


def _download_and_report(rel_paths: list[str], out_path: Path, prompt: str = "") -> None:
    import subprocess as sp

    for i, rel_path in enumerate(rel_paths):
        local_out = out_path if i == 0 else out_path.with_stem(f"{out_path.stem}_{i+1}")
        print(f"[download] {rel_path} → {local_out}")
        sp.run(
            ["modal", "volume", "get", "--force", "musecoco-weights", rel_path, str(local_out)],
            check=True,
        )
        print(f"[saved] {local_out.resolve()}")

    print("\n--- RESULTADOS ---")
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(out_path))
        n_tracks = len(pm.instruments)
        n_notes = sum(len(inst.notes) for inst in pm.instruments)
        dur = pm.get_end_time()
        instrs = [inst.program for inst in pm.instruments]
        if prompt:
            print(f"  prompt:       {prompt}")
        print(f"  device:       Modal L4 GPU")
        print(f"  pistas:       {n_tracks}")
        print(f"  notas:        {n_notes}")
        print(f"  duración:     {dur:.1f}s")
        print(f"  instrumentos: {instrs}")
    except ImportError:
        print("  (instala pretty_midi localmente para ver detalles del MIDI)")


# ---------------------------------------------------------------------------
# Local entrypoint de recovery — retoma un call spawneado que quedó huérfano
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def recover(
    call_id: str = "",
    out: str = "",
) -> None:
    """
    Descarga el resultado de un generate() spawneado previamente.
    Sin argumentos, lee .musecoco_last_call.json del directorio actual.

    Ejemplo:
        modal run research_musecoco_modal.py::recover
        modal run research_musecoco_modal.py::recover --call-id fc-XXXX --out out.mid
    """
    import time

    if not call_id:
        state_file = Path(".musecoco_last_call.json")
        if not state_file.exists():
            print("[recover] No hay .musecoco_last_call.json y no se pasó --call-id")
            return
        state = json.loads(state_file.read_text())
        call_id = state["call_id"]
        if not out:
            out = state.get("out", "out_recovered.mid")
        prompt = state.get("prompt", "")
    else:
        prompt = ""

    out_path = Path(out or "out_recovered.mid")
    print(f"[recover] Polling call_id={call_id} → {out_path}")

    call = modal.functions.FunctionCall.from_id(call_id)
    rel_paths = None
    dots = 0
    while rel_paths is None:
        try:
            rel_paths = call.get(timeout=30)
        except TimeoutError:
            print(".", end="", flush=True)
            dots += 1
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("deadline", "timed out")):
                print(".", end="", flush=True)
                dots += 1
            else:
                raise
    if dots:
        print()

    if not rel_paths:
        print("[recover] La función no retornó rutas MIDI.")
        return

    _download_and_report(rel_paths, out_path, prompt)
