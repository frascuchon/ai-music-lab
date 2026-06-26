#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup helper CLI global para los plugins REAPER AI — stdlib only, sin deps externos.

Invocado por shared/Setup.lua como:
    python3 setup_helpers.py <cmd> [flags]

donde python3 es el Python detectado por REAPER (o system python3).
Cada subcomando escribe progreso en --progress con el protocolo:

    state|pct|msg\n

  state  : running | done | error
  pct    : 0.0–1.0

El subcomando 'check' añade líneas CHECK| tras la cabecera:

    done|1.0|done\n
    CHECK|<name>|ok|<detail>\n
    CHECK|<name>|missing|<detail>\n

Checks core (siempre): python, uv, modal-cli, modal-auth
Checks extras por plugin (si la carpeta existe junto a shared/):
  StemsSeparator → demucs (pip local), hf-secret (Modal secret)
  Audio2Midi     → (ninguno extra — pesos en Modal Volumes)

Subcommands:
  check                                  Detecta entorno, escribe CHECK lines.
  install-uv                             Descarga e instala uv.
  sync-deps                              uv sync (instala modal CLI en venv de shared/).
  install-demucs   --python <path>       pip install demucs en el Python dado.
  modal-login                            Abre browser para Modal token new.
  modal-secret-list                      Informa si huggingface-secret existe.
  modal-secret-create --token <tok>      Crea/actualiza huggingface-secret.
  prewarm-sam      [--model <hf-id>]     Precarga SAM Audio en Modal Volume.
  prewarm-miros                          Descarga pesos MIROS en Modal Volume.
  prewarm-yourmt3                        Descarga pesos YourMT3+ en Modal Volume.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent          # shared/
PLUGINS_DIR = SCRIPT_DIR.parent             # plugins-and-extensions/
HF_SECRET_NAME = "huggingface-secret"


# ---------------------------------------------------------------------------
# Progress protocol
# ---------------------------------------------------------------------------

def write(pf: Path | None, state: str, pct: float, msg: str,
          extra: list[str] | None = None) -> None:
    if not pf:
        return
    lines = [f"{state}|{pct:.3f}|{msg}\n"]
    if extra:
        lines.extend(line + "\n" for line in extra)
    try:
        pf.write_text("".join(lines))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_uv() -> Path | None:
    u = shutil.which("uv")
    if u:
        return Path(u)
    fallback = Path.home() / ".local" / "bin" / "uv"
    return fallback if fallback.is_file() else None


def _run(cmd: list[str], **kwargs) -> tuple[int, str, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def _stream(pf: Path | None, cmd: list[str], *, done_msg: str = "Completado",
            env: dict | None = None) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    assert proc.stdout
    pct = 0.1
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            write(pf, "running", min(pct, 0.95), line[:120])
            pct = min(pct + 0.04, 0.95)
    proc.wait()
    if proc.returncode == 0:
        write(pf, "done", 1.0, done_msg)
    else:
        write(pf, "error", pct, f"Error (cod {proc.returncode})")
    return proc.returncode


# ---------------------------------------------------------------------------
# Core env checks  →  (status, detail)   status = "ok" | "missing"
# ---------------------------------------------------------------------------

def _chk_python(ini_path: Path) -> tuple[str, str]:
    if not ini_path.exists():
        return "missing", f"reaper.ini no encontrado: {ini_path}"
    libpath = None
    for line in ini_path.read_text(errors="replace").splitlines():
        for key in ("pythonlibpath64", "pythonlibpath32"):
            if line.startswith(f"{key}="):
                v = line[len(key) + 1:].strip()
                if v:
                    libpath = v
                    break
        if libpath:
            break
    if not libpath:
        return "missing", "pythonlibpath no encontrado en reaper.ini"
    exe = Path(libpath).parent / "bin" / "python3"
    if not exe.exists():
        return "missing", f"Python no encontrado: {exe}"
    rc, out, _ = _run([str(exe), "--version"])
    return ("ok", f"{exe.name} ({out})") if rc == 0 else ("missing", str(exe))


def _chk_uv() -> tuple[str, str]:
    uv = _find_uv()
    if uv is None:
        return "missing", "uv no en PATH ni ~/.local/bin/uv"
    rc, out, _ = _run([str(uv), "--version"])
    return ("ok", f"{uv} ({out})") if rc == 0 else ("missing", f"{uv} (error)")


def _chk_modal_cli() -> tuple[str, str]:
    uv = _find_uv()
    if uv is None:
        return "missing", "uv requerido para modal"
    rc, out, _ = _run([str(uv), "run", "--project", str(SCRIPT_DIR),
                       "modal", "--version"])
    return ("ok", out[:60]) if rc == 0 else ("missing", "modal no disponible")


def _chk_modal_auth() -> tuple[str, str]:
    toml = Path.home() / ".modal.toml"
    if not toml.exists():
        return "missing", "~/.modal.toml no existe"
    try:
        content = toml.read_text()
        if "token" in content.lower():
            return "ok", str(toml)
        return "missing", "~/.modal.toml sin token válido"
    except Exception as e:
        return "missing", str(e)


# ---------------------------------------------------------------------------
# Plugin-specific extra checks
# ---------------------------------------------------------------------------

def _chk_demucs() -> tuple[str, str]:
    """Comprueba demucs en el intérprete que lanza este script (== Python de REAPER)."""
    rc, out, err = _run([sys.executable, "-c",
                         "import demucs; print(demucs.__version__)"])
    if rc == 0:
        return "ok", f"demucs {out.strip()}"
    last = (err or "ImportError").split("\n")[-1][:80]
    return "missing", last


def _chk_hf_secret() -> tuple[str, str]:
    uv = _find_uv()
    if uv is None:
        return "missing", "uv requerido"
    rc, out, err = _run([str(uv), "run", "--project", str(SCRIPT_DIR),
                         "modal", "secret", "list"])
    if rc != 0:
        return "missing", "modal secret list falló (¿auth?)"
    return ("ok", f"'{HF_SECRET_NAME}' encontrado") \
        if HF_SECRET_NAME in out \
        else ("missing", f"'{HF_SECRET_NAME}' no existe")


# ---------------------------------------------------------------------------
# Subcommand: check
# ---------------------------------------------------------------------------

def cmd_check(args) -> None:
    pf = Path(args.progress) if args.progress else None
    write(pf, "running", 0.1, "Comprobando entorno...")

    ini = Path.home() / "Library" / "Application Support" / "REAPER" / "reaper.ini"

    # Core checks — always
    checks: list[tuple[str, tuple[str, str]]] = [
        ("python",     _chk_python(ini)),
        ("uv",         _chk_uv()),
        ("modal-cli",  _chk_modal_cli()),
        ("modal-auth", _chk_modal_auth()),
    ]

    # StemsSeparator extras — if plugin folder exists
    stems_dir = PLUGINS_DIR / "StemsSeparator"
    if stems_dir.is_dir():
        checks += [
            ("demucs",     _chk_demucs()),
            ("hf-secret",  _chk_hf_secret()),
        ]

    # Audio2Midi extras — currently none extra beyond core
    # audio2midi_dir = PLUGINS_DIR / "Audio2Midi"
    # if audio2midi_dir.is_dir():
    #     pass  # modal volumes used; no local install needed

    extra = [f"CHECK|{n}|{s}|{d}" for n, (s, d) in checks]
    write(pf, "done", 1.0, "done", extra)


# ---------------------------------------------------------------------------
# Subcommand: install-uv
# ---------------------------------------------------------------------------

def cmd_install_uv(args) -> None:
    pf = Path(args.progress) if args.progress else None
    write(pf, "running", 0.1, "Descargando instalador de uv...")
    try:
        with urllib.request.urlopen("https://astral.sh/uv/install.sh", timeout=30) as resp:
            install_bytes = resp.read()
    except Exception as e:
        write(pf, "error", 0, f"Error descargando instalador: {e}")
        return
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False, mode="wb") as f:
        f.write(install_bytes)
        tmp = Path(f.name)
    try:
        write(pf, "running", 0.4, "Ejecutando instalador de uv...")
        _stream(pf, ["sh", str(tmp)],
                done_msg="uv instalado — reinicia REAPER o abre un nuevo terminal")
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Subcommand: sync-deps  (instala modal CLI en el venv de shared/)
# ---------------------------------------------------------------------------

def cmd_sync_deps(args) -> None:
    pf = Path(args.progress) if args.progress else None
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv no encontrado — instálalo primero")
        return
    write(pf, "running", 0.05, "Instalando dependencias del proyecto (modal, protobuf)...")
    _stream(pf, [str(uv), "sync", "--project", str(SCRIPT_DIR)],
            done_msg="Dependencias instaladas — modal CLI listo")


# ---------------------------------------------------------------------------
# Subcommand: install-demucs  (StemsSeparator)
# ---------------------------------------------------------------------------

def cmd_install_demucs(args) -> None:
    pf = Path(args.progress) if args.progress else None
    python = args.python or sys.executable
    write(pf, "running", 0.05, "Instalando demucs...")
    _stream(pf, [python, "-m", "pip", "install", "--user", "demucs"],
            done_msg="demucs instalado")


# ---------------------------------------------------------------------------
# Subcommand: modal-login
# ---------------------------------------------------------------------------

def cmd_modal_login(args) -> None:
    pf = Path(args.progress) if args.progress else None
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv no encontrado — instálalo primero")
        return

    toml = Path.home() / ".modal.toml"
    old_mtime = toml.stat().st_mtime if toml.exists() else None
    write(pf, "running", 0.05, "Abriendo navegador para login en Modal...")

    proc = subprocess.Popen(
        [str(uv), "run", "--project", str(SCRIPT_DIR), "modal", "token", "new"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.stdout
    start, timeout, pct = time.monotonic(), 300, 0.1
    while True:
        if time.monotonic() - start > timeout:
            proc.terminate()
            write(pf, "error", pct, "Timeout esperando login (5 min)")
            return
        line = proc.stdout.readline()
        if line:
            write(pf, "running", min(pct, 0.5), line.rstrip()[:120])
            pct = min(pct + 0.03, 0.5)
        if toml.exists() and toml.stat().st_mtime != old_mtime:
            proc.terminate()
            write(pf, "done", 1.0, "Login completado — ~/.modal.toml actualizado")
            return
        if proc.poll() is not None:
            rc = proc.returncode
            if toml.exists() and toml.stat().st_mtime != old_mtime:
                write(pf, "done", 1.0, "Login completado")
            elif rc == 0:
                write(pf, "done", 1.0, "modal token new completado")
            else:
                write(pf, "error", pct, f"modal token new falló (cod {rc})")
            return
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Subcommand: modal-secret-create  (StemsSeparator — HF token)
# ---------------------------------------------------------------------------

def cmd_modal_secret_create(args) -> None:
    pf = Path(args.progress) if args.progress else None
    token = (args.token or "").strip()
    if not token.startswith("hf_"):
        write(pf, "error", 0, "Token inválido (debe empezar por 'hf_')")
        return
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv no encontrado")
        return
    write(pf, "running", 0.3, f"Guardando secret '{HF_SECRET_NAME}'...")
    rc, out, err = _run([str(uv), "run", "--project", str(SCRIPT_DIR),
                         "modal", "secret", "create", HF_SECRET_NAME,
                         f"HF_TOKEN={token}", "--force"])
    if rc == 0:
        write(pf, "done", 1.0, f"Secret '{HF_SECRET_NAME}' guardado")
    else:
        write(pf, "error", 0, (err or out)[:100])


# ---------------------------------------------------------------------------
# Subcommand: prewarm-sam  (StemsSeparator — SAM Audio)
# ---------------------------------------------------------------------------

def cmd_prewarm_sam(args) -> None:
    pf = Path(args.progress) if args.progress else None
    model = args.model or "facebook/sam-audio-large"
    script = PLUGINS_DIR / "StemsSeparator" / "modal_sam_audio.py"
    if not script.exists():
        write(pf, "error", 0, f"No encontrado: {script}")
        return
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv no encontrado")
        return
    write(pf, "running", 0.05, f"Descargando {model} (puede tardar 10-20 min)...")
    _stream(pf,
            [str(uv), "run", "--project", str(SCRIPT_DIR),
             "modal", "run", f"{script}::run_download", "--model", model],
            done_msg="Modelo SAM descargado y cacheado en Modal Volume")


# ---------------------------------------------------------------------------
# Subcommand: prewarm-miros  (Audio2Midi — MIROS)
# ---------------------------------------------------------------------------

def cmd_prewarm_miros(args) -> None:
    pf = Path(args.progress) if args.progress else None
    script = PLUGINS_DIR / "Audio2Midi" / "research" / "research_miros_modal.py"
    if not script.exists():
        write(pf, "error", 0, f"No encontrado: {script}")
        return
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv no encontrado")
        return
    write(pf, "running", 0.05,
          "Descargando pesos MIROS en Modal Volume (puede tardar 10-15 min)...")
    _stream(pf,
            [str(uv), "run", "--project", str(PLUGINS_DIR / "Audio2Midi" / "research"),
             "modal", "run", f"{script}::setup"],
            done_msg="Pesos MIROS descargados y cacheados en Modal Volume")


# ---------------------------------------------------------------------------
# Subcommand: prewarm-yourmt3  (Audio2Midi — YourMT3+)
# ---------------------------------------------------------------------------

def cmd_prewarm_yourmt3(args) -> None:
    pf = Path(args.progress) if args.progress else None
    script = PLUGINS_DIR / "Audio2Midi" / "research" / "research_yourmt3_modal.py"
    if not script.exists():
        write(pf, "error", 0, f"No encontrado: {script}")
        return
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv no encontrado")
        return
    write(pf, "running", 0.05,
          "Descargando pesos YourMT3+ en Modal Volume (puede tardar 5-10 min)...")
    _stream(pf,
            [str(uv), "run", "--project", str(PLUGINS_DIR / "Audio2Midi" / "research"),
             "modal", "run", f"{script}::setup"],
            done_msg="Pesos YourMT3+ descargados y cacheados en Modal Volume")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Setup helper global para los plugins REAPER AI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_chk = sub.add_parser("check")
    p_chk.add_argument("--progress", default="")

    p_uv = sub.add_parser("install-uv")
    p_uv.add_argument("--progress", default="")

    p_syn = sub.add_parser("sync-deps")
    p_syn.add_argument("--progress", default="")

    p_dem = sub.add_parser("install-demucs")
    p_dem.add_argument("--python", default="")
    p_dem.add_argument("--progress", default="")

    p_log = sub.add_parser("modal-login")
    p_log.add_argument("--progress", default="")

    p_scr = sub.add_parser("modal-secret-create")
    p_scr.add_argument("--token", required=True)
    p_scr.add_argument("--progress", default="")

    p_pw_sam = sub.add_parser("prewarm-sam")
    p_pw_sam.add_argument("--model", default="facebook/sam-audio-large")
    p_pw_sam.add_argument("--progress", default="")

    p_pw_miros = sub.add_parser("prewarm-miros")
    p_pw_miros.add_argument("--progress", default="")

    p_pw_yt3 = sub.add_parser("prewarm-yourmt3")
    p_pw_yt3.add_argument("--progress", default="")

    args = p.parse_args()
    {
        "check":                cmd_check,
        "install-uv":           cmd_install_uv,
        "sync-deps":            cmd_sync_deps,
        "install-demucs":       cmd_install_demucs,
        "modal-login":          cmd_modal_login,
        "modal-secret-create":  cmd_modal_secret_create,
        "prewarm-sam":          cmd_prewarm_sam,
        "prewarm-miros":        cmd_prewarm_miros,
        "prewarm-yourmt3":      cmd_prewarm_yourmt3,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
