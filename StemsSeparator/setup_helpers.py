#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup helper CLI for Stem Separator — stdlib only, no external deps.

Invoked by Setup.lua as:
    python3 setup_helpers.py <cmd> [flags]

where python3 is whatever Python REAPER detected (or system python3).
Each subcommand writes progress to --progress in the protocol:

    state|pct|msg\n

  state  : running | done | error
  pct    : 0.0–1.0

The 'check' subcommand appends CHECK| lines after the header:

    done|1.0|done\n
    CHECK|<name>|ok|<detail>\n
    CHECK|<name>|missing|<detail>\n

Subcommands:
  check                                  Detect env, write CHECK lines.
  install-demucs   --python <path>       pip install demucs into given python.
  modal-login                            Run 'modal token new', wait for toml.
  modal-secret-list                      Report if huggingface-secret exists.
  modal-secret-create --token <tok>      Create/update huggingface-secret.
  prewarm          [--model <hf-id>]     Run modal_sam_audio.py::run_download.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
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
# Individual env checks  →  (status, detail)   status = "ok" | "missing"
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


def _chk_demucs() -> tuple[str, str]:
    # Check against THIS interpreter (the one running setup_helpers.py),
    # which is REAPER's detected Python — same one that runs separate_demucs.py.
    rc, out, err = _run([sys.executable, "-c",
                         "import demucs; print(demucs.__version__)"])
    if rc == 0:
        return "ok", f"demucs {out.strip()}"
    last = (err or "ImportError").split("\n")[-1][:80]
    return "missing", last


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


def _chk_hf_secret() -> tuple[str, str]:
    uv = _find_uv()
    if uv is None:
        return "missing", "uv requerido"
    rc, out, err = _run([str(uv), "run", "--project", str(SCRIPT_DIR),
                         "modal", "secret", "list"])
    if rc != 0:
        return "missing", f"modal secret list falló (¿auth?)"
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

    checks = [
        ("python",     _chk_python(ini)),
        ("uv",         _chk_uv()),
        ("demucs",     _chk_demucs()),
        ("modal-cli",  _chk_modal_cli()),
        ("modal-auth", _chk_modal_auth()),
        ("hf-secret",  _chk_hf_secret()),
    ]
    extra = [f"CHECK|{n}|{s}|{d}" for n, (s, d) in checks]
    write(pf, "done", 1.0, "done", extra)


# ---------------------------------------------------------------------------
# Subcommand: install-demucs
# ---------------------------------------------------------------------------

def cmd_install_demucs(args) -> None:
    pf = Path(args.progress) if args.progress else None
    python = args.python or sys.executable
    write(pf, "running", 0.05, f"Instalando demucs...")
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
# Subcommand: modal-secret-create
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
# Subcommand: prewarm
# ---------------------------------------------------------------------------

def cmd_prewarm(args) -> None:
    pf = Path(args.progress) if args.progress else None
    model = args.model or "facebook/sam-audio-large"
    script = SCRIPT_DIR / "modal_sam_audio.py"
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
            done_msg=f"Modelo descargado y cacheado en Modal Volume")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_chk = sub.add_parser("check")
    p_chk.add_argument("--progress", default="")

    p_dem = sub.add_parser("install-demucs")
    p_dem.add_argument("--python", default="")
    p_dem.add_argument("--progress", default="")

    p_log = sub.add_parser("modal-login")
    p_log.add_argument("--progress", default="")

    p_scr = sub.add_parser("modal-secret-create")
    p_scr.add_argument("--token", required=True)
    p_scr.add_argument("--progress", default="")

    p_pw = sub.add_parser("prewarm")
    p_pw.add_argument("--model", default="facebook/sam-audio-large")
    p_pw.add_argument("--progress", default="")

    args = p.parse_args()
    {
        "check":               cmd_check,
        "install-demucs":      cmd_install_demucs,
        "modal-login":         cmd_modal_login,
        "modal-secret-create": cmd_modal_secret_create,
        "prewarm":             cmd_prewarm,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
