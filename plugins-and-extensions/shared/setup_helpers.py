#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global setup helper CLI for REAPER AI plugins — stdlib only, no external deps.

Invoked by shared/Setup.lua as:
    python3 setup_helpers.py <cmd> [flags]

where python3 is the Python detected by REAPER (or system python3).
Each subcommand writes progress to --progress using the protocol:

    state|pct|msg\n

  state  : running | done | error
  pct    : 0.0–1.0

The 'check' subcommand appends CHECK| lines after the header:

    done|1.0|done\n
    CHECK|<name>|ok|<detail>\n
    CHECK|<name>|missing|<detail>\n

Core checks (always): python, uv, modal-cli, modal-auth
Plugin-specific extra checks (if the folder exists next to shared/):
  StemsSeparator  → demucs (local pip), hf-secret (Modal secret)
  Audio2Midi      → (no extras — weights in Modal Volumes)
  MidiGenerator   → (no extras — weights in Modal Volumes)

Subcommands:
  check                                  Detect environment, write CHECK lines.
  install-uv                             Download and install uv.
  sync-deps                              uv sync (install modal CLI in shared/ venv).
  install-demucs   --python <path>       pip install demucs in the given Python.
  modal-login                            Open browser for Modal token new.
  modal-secret-list                      Report whether huggingface-secret exists.
  modal-secret-create --token <tok>      Create/update huggingface-secret.
  prewarm-sam      [--model <hf-id>]     Pre-load SAM Audio in Modal Volume.
  prewarm-miros                          Download MIROS weights in Modal Volume.
  prewarm-yourmt3                        Download YourMT3+ weights in Modal Volume.
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


def _stream(pf: Path | None, cmd: list[str], *, done_msg: str = "Completed",
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
        write(pf, "error", pct, f"Error (code {proc.returncode})")
    return proc.returncode


# ---------------------------------------------------------------------------
# Core env checks  →  (status, detail)   status = "ok" | "missing"
# ---------------------------------------------------------------------------

def _chk_python(ini_path: Path) -> tuple[str, str]:
    if not ini_path.exists():
        return "missing", f"reaper.ini not found: {ini_path}"
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
        return "missing", "pythonlibpath not found in reaper.ini"
    exe = Path(libpath).parent / "bin" / "python3"
    if not exe.exists():
        return "missing", f"Python not found: {exe}"
    rc, out, _ = _run([str(exe), "--version"])
    return ("ok", f"{exe.name} ({out})") if rc == 0 else ("missing", str(exe))


def _chk_uv() -> tuple[str, str]:
    uv = _find_uv()
    if uv is None:
        return "missing", "uv not in PATH or ~/.local/bin/uv"
    rc, out, _ = _run([str(uv), "--version"])
    return ("ok", f"{uv} ({out})") if rc == 0 else ("missing", f"{uv} (error)")


def _chk_modal_cli() -> tuple[str, str]:
    uv = _find_uv()
    if uv is None:
        return "missing", "uv required for modal"
    rc, out, _ = _run([str(uv), "run", "--project", str(SCRIPT_DIR),
                       "modal", "--version"])
    return ("ok", out[:60]) if rc == 0 else ("missing", "modal not available")


def _chk_modal_auth() -> tuple[str, str]:
    toml = Path.home() / ".modal.toml"
    if not toml.exists():
        return "missing", "~/.modal.toml does not exist"
    try:
        content = toml.read_text()
        if "token" in content.lower():
            return "ok", str(toml)
        return "missing", "~/.modal.toml has no valid token"
    except Exception as e:
        return "missing", str(e)


# ---------------------------------------------------------------------------
# Plugin-specific extra checks
# ---------------------------------------------------------------------------

def _chk_demucs() -> tuple[str, str]:
    """Check demucs in the interpreter running this script (== REAPER's Python)."""
    rc, out, err = _run([sys.executable, "-c",
                         "import demucs; print(demucs.__version__)"])
    if rc == 0:
        return "ok", f"demucs {out.strip()}"
    last = (err or "ImportError").split("\n")[-1][:80]
    return "missing", last


def _chk_hf_secret() -> tuple[str, str]:
    uv = _find_uv()
    if uv is None:
        return "missing", "uv required"
    rc, out, err = _run([str(uv), "run", "--project", str(SCRIPT_DIR),
                         "modal", "secret", "list"])
    if rc != 0:
        return "missing", "modal secret list failed (auth?)"
    return ("ok", f"'{HF_SECRET_NAME}' found") \
        if HF_SECRET_NAME in out \
        else ("missing", f"'{HF_SECRET_NAME}' does not exist")


# ---------------------------------------------------------------------------
# Subcommand: check
# ---------------------------------------------------------------------------

def cmd_check(args) -> None:
    pf = Path(args.progress) if args.progress else None
    write(pf, "running", 0.1, "Checking environment...")

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
    write(pf, "running", 0.1, "Downloading uv installer...")
    try:
        with urllib.request.urlopen("https://astral.sh/uv/install.sh", timeout=30) as resp:
            install_bytes = resp.read()
    except Exception as e:
        write(pf, "error", 0, f"Error downloading installer: {e}")
        return
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False, mode="wb") as f:
        f.write(install_bytes)
        tmp = Path(f.name)
    try:
        write(pf, "running", 0.4, "Running uv installer...")
        _stream(pf, ["sh", str(tmp)],
                done_msg="uv installed — restart REAPER or open a new terminal")
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Subcommand: sync-deps  (install modal CLI in the shared/ venv)
# ---------------------------------------------------------------------------

def cmd_sync_deps(args) -> None:
    pf = Path(args.progress) if args.progress else None
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv not found — install it first")
        return
    write(pf, "running", 0.05, "Installing project dependencies (modal, protobuf)...")
    _stream(pf, [str(uv), "sync", "--project", str(SCRIPT_DIR)],
            done_msg="Dependencies installed — modal CLI ready")


# ---------------------------------------------------------------------------
# Subcommand: install-demucs  (StemsSeparator)
# ---------------------------------------------------------------------------

def cmd_install_demucs(args) -> None:
    pf = Path(args.progress) if args.progress else None
    python = args.python or sys.executable
    write(pf, "running", 0.05, "Installing demucs...")
    _stream(pf, [python, "-m", "pip", "install", "--user", "demucs"],
            done_msg="demucs installed")


# ---------------------------------------------------------------------------
# Subcommand: modal-login
# ---------------------------------------------------------------------------

def cmd_modal_login(args) -> None:
    pf = Path(args.progress) if args.progress else None
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv not found — install it first")
        return

    toml = Path.home() / ".modal.toml"
    old_mtime = toml.stat().st_mtime if toml.exists() else None
    write(pf, "running", 0.05, "Opening browser for Modal login...")

    proc = subprocess.Popen(
        [str(uv), "run", "--project", str(SCRIPT_DIR), "modal", "token", "new"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.stdout
    start, timeout, pct = time.monotonic(), 300, 0.1
    while True:
        if time.monotonic() - start > timeout:
            proc.terminate()
            write(pf, "error", pct, "Timeout waiting for login (5 min)")
            return
        line = proc.stdout.readline()
        if line:
            write(pf, "running", min(pct, 0.5), line.rstrip()[:120])
            pct = min(pct + 0.03, 0.5)
        if toml.exists() and toml.stat().st_mtime != old_mtime:
            proc.terminate()
            write(pf, "done", 1.0, "Login completed — ~/.modal.toml updated")
            return
        if proc.poll() is not None:
            rc = proc.returncode
            if toml.exists() and toml.stat().st_mtime != old_mtime:
                write(pf, "done", 1.0, "Login completed")
            elif rc == 0:
                write(pf, "done", 1.0, "modal token new completed")
            else:
                write(pf, "error", pct, f"modal token new failed (code {rc})")
            return
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Subcommand: modal-secret-create  (StemsSeparator — HF token)
# ---------------------------------------------------------------------------

def cmd_modal_secret_create(args) -> None:
    pf = Path(args.progress) if args.progress else None
    token = (args.token or "").strip()
    if not token.startswith("hf_"):
        write(pf, "error", 0, "Invalid token (must start with 'hf_')")
        return
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv not found")
        return
    write(pf, "running", 0.3, f"Saving secret '{HF_SECRET_NAME}'...")
    rc, out, err = _run([str(uv), "run", "--project", str(SCRIPT_DIR),
                         "modal", "secret", "create", HF_SECRET_NAME,
                         f"HF_TOKEN={token}", "--force"])
    if rc == 0:
        write(pf, "done", 1.0, f"Secret '{HF_SECRET_NAME}' saved")
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
        write(pf, "error", 0, f"Not found: {script}")
        return
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv not found")
        return
    write(pf, "running", 0.05, f"Downloading {model} (may take 10-20 min)...")
    _stream(pf,
            [str(uv), "run", "--project", str(SCRIPT_DIR),
             "modal", "run", f"{script}::run_download", "--model", model],
            done_msg="SAM model downloaded and cached in Modal Volume")


# ---------------------------------------------------------------------------
# Subcommand: prewarm-miros  (Audio2Midi — MIROS)
# ---------------------------------------------------------------------------

def cmd_prewarm_miros(args) -> None:
    pf = Path(args.progress) if args.progress else None
    script = PLUGINS_DIR / "Audio2Midi" / "research" / "research_miros_modal.py"
    if not script.exists():
        write(pf, "error", 0, f"Not found: {script}")
        return
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv not found")
        return
    write(pf, "running", 0.05,
          "Downloading MIROS weights to Modal Volume (may take 10-15 min)...")
    _stream(pf,
            [str(uv), "run", "--project", str(SCRIPT_DIR),
             "modal", "run", f"{script}::setup"],
            done_msg="MIROS weights downloaded and cached in Modal Volume")


# ---------------------------------------------------------------------------
# Subcommand: prewarm-yourmt3  (Audio2Midi — YourMT3+)
# ---------------------------------------------------------------------------

def cmd_prewarm_yourmt3(args) -> None:
    pf = Path(args.progress) if args.progress else None
    script = PLUGINS_DIR / "Audio2Midi" / "research" / "research_yourmt3_modal.py"
    if not script.exists():
        write(pf, "error", 0, f"Not found: {script}")
        return
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv not found")
        return
    write(pf, "running", 0.05,
          "Downloading YourMT3+ weights to Modal Volume (may take 5-10 min)...")
    _stream(pf,
            [str(uv), "run", "--project", str(SCRIPT_DIR),
             "modal", "run", f"{script}::setup"],
            done_msg="YourMT3+ weights downloaded and cached in Modal Volume")


# ---------------------------------------------------------------------------
# Subcommands: prewarm-midigen-*  (MidiGenerator — 6 models)
# ---------------------------------------------------------------------------

_MG_MODELS = {
    "amadeus":      ("research_amadeus_modal.py",      "setup"),
    "midi-llm":     ("research_midi_llm_modal.py",     "setup"),
    "text2midi":    ("research_text2midi_modal.py",     "setup"),
    "chatmusician": ("research_chatmusician_modal.py",  "setup"),
    "musecoco":     ("research_musecoco_modal.py",      "setup_weights"),
    "anticipatory": ("research_anticipatory_modal.py",  "setup"),
}

_MG_RESEARCH = PLUGINS_DIR / "MidiGenerator" / "research"


def _prewarm_midigen_model(model_name: str, pf) -> None:
    entry = _MG_MODELS.get(model_name)
    if entry is None:
        write(pf, "error", 0, f"Unknown MidiGenerator model: {model_name!r}")
        return
    script_name, entrypoint = entry
    script = _MG_RESEARCH / script_name
    if not script.exists():
        write(pf, "error", 0, f"Script not found: {script}")
        return
    uv = _find_uv()
    if uv is None:
        write(pf, "error", 0, "uv not found")
        return
    msg = f"Downloading {model_name} weights to Modal Volume..."
    write(pf, "running", 0.05, msg)
    _stream(pf,
            [str(uv), "run", "--project", str(SCRIPT_DIR),
             "modal", "run", f"{script}::{entrypoint}"],
            done_msg=f"{model_name} weights cached in Modal Volume")


def cmd_prewarm_midigen_amadeus(args) -> None:
    _prewarm_midigen_model("amadeus", Path(args.progress) if args.progress else None)

def cmd_prewarm_midigen_midi_llm(args) -> None:
    _prewarm_midigen_model("midi-llm", Path(args.progress) if args.progress else None)

def cmd_prewarm_midigen_text2midi(args) -> None:
    _prewarm_midigen_model("text2midi", Path(args.progress) if args.progress else None)

def cmd_prewarm_midigen_chatmusician(args) -> None:
    _prewarm_midigen_model("chatmusician", Path(args.progress) if args.progress else None)

def cmd_prewarm_midigen_musecoco(args) -> None:
    _prewarm_midigen_model("musecoco", Path(args.progress) if args.progress else None)

def cmd_prewarm_midigen_anticipatory(args) -> None:
    _prewarm_midigen_model("anticipatory", Path(args.progress) if args.progress else None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Global setup helper for REAPER AI plugins.")
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

    for mg_model in ["amadeus", "midi-llm", "text2midi", "chatmusician", "musecoco", "anticipatory"]:
        p_mg = sub.add_parser(f"prewarm-midigen-{mg_model}")
        p_mg.add_argument("--progress", default="")

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
        "prewarm-midigen-amadeus":      cmd_prewarm_midigen_amadeus,
        "prewarm-midigen-midi-llm":     cmd_prewarm_midigen_midi_llm,
        "prewarm-midigen-text2midi":    cmd_prewarm_midigen_text2midi,
        "prewarm-midigen-chatmusician": cmd_prewarm_midigen_chatmusician,
        "prewarm-midigen-musecoco":     cmd_prewarm_midigen_musecoco,
        "prewarm-midigen-anticipatory": cmd_prewarm_midigen_anticipatory,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
