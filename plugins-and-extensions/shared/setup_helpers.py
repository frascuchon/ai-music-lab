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

Note: there used to be prewarm-sam / prewarm-yourmt3 /
prewarm-midigen-* subcommands here that manually pre-downloaded each model's
weights into its Modal Volume before first real use. Removed — every real
generate/transcribe/separate Modal function now downloads+caches its own
weights lazily on first call (see each research_*_modal.py's _ensure_weights()
or equivalent), so the first real run simply absorbs the download time
instead of requiring a separate manual pre-warm step.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import deque
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


def _tail_error_message(cmd: list[str], returncode: int,
                        tail_lines: list[str]) -> str:
    """Build a descriptive error message from a failed subprocess's own
    output instead of a bare exit code — a bare 'Error (code 1)' gives the
    user nothing to act on, especially for commands (brew, pip, uv, modal)
    whose real failure reason is only visible in their own stdout/stderr."""
    cmd_name = Path(cmd[0]).name if cmd else "Command"
    last = next((ln for ln in reversed(tail_lines) if ln.strip()), "")
    if last:
        return f"{cmd_name} failed (code {returncode}): {last[:160]}"
    return (f"{cmd_name} failed (code {returncode}) with no output — "
            "see reaperai_setup.log for details.")


def _stream(pf: Path | None, cmd: list[str], *, done_msg: str = "Completed",
            env: dict | None = None) -> int:
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)
    except OSError as exc:
        write(pf, "error", 0, f"Failed to launch {Path(cmd[0]).name if cmd else cmd}: {exc}")
        return 1
    assert proc.stdout
    pct = 0.1
    tail: deque[str] = deque(maxlen=40)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            write(pf, "running", min(pct, 0.95), line[:120])
            pct = min(pct + 0.04, 0.95)
    proc.stdout.close()
    proc.wait()
    if proc.returncode == 0:
        write(pf, "done", 1.0, done_msg)
    else:
        tail_lines = list(tail)
        write(pf, "error", pct, _tail_error_message(cmd, proc.returncode, tail_lines),
              ["--- last output lines ---"] + tail_lines if tail_lines else None)
    return proc.returncode


# ---------------------------------------------------------------------------
# Core env checks  →  (status, detail)   status = "ok" | "missing"
# ---------------------------------------------------------------------------

def _find_versioned_python3(bindir: Path) -> Path | None:
    """Scan bindir for the highest-numbered python3.N when the bare
    "python3" symlink isn't there (e.g. Homebrew python@3.10 framework)."""
    best, best_ver = None, -1
    try:
        entries = list(bindir.iterdir())
    except OSError:
        return None
    for entry in entries:
        m = re.fullmatch(r"python3\.(\d+)", entry.name)
        if m:
            ver = int(m.group(1))
            if ver > best_ver:
                best_ver, best = ver, entry
    return best


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
    # If the configured path points inside lib/ (e.g. lib/python3.11/config-...)
    # instead of at lib/ itself, walk up ancestors looking for bin/python3
    # (or bin/python3.N, e.g. Homebrew framework builds that ship no bare
    # "python3" symlink — mirrors common.lua's _find_versioned_python3).
    parent = Path(libpath).parent
    exe = None
    for _ in range(4):
        bindir = parent / "bin"
        candidate = bindir / "python3"
        if candidate.exists():
            exe = candidate
            break
        versioned = _find_versioned_python3(bindir)
        if versioned is not None:
            exe = versioned
            break
        if parent.parent == parent:
            break
        parent = parent.parent
    if exe is None:
        return "missing", f"Python not found: {libpath}"
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
    """Check demucs AND soundfile in the interpreter running this script
    (== REAPER's Python). soundfile is checked here too (not just demucs)
    because separate_demucs.py/separate_sam.py import it directly for their
    own --start/--duration section-extraction helper — demucs 4.x dropped
    its own soundfile dependency in favor of `sphn`, so `pip install demucs`
    alone no longer pulls soundfile in, even though this plugin still needs
    it. A user who had demucs installed before that changed would pass this
    check yet still hit "No module named 'soundfile'" the first time they
    cut a section, so install-demucs below always installs both together."""
    rc, out, err = _run([sys.executable, "-c",
                         "import demucs; print(demucs.__version__)"])
    if rc != 0:
        last = (err or "ImportError").split("\n")[-1][:80]
        return "missing", last
    rc_sf, _, err_sf = _run([sys.executable, "-c", "import soundfile"])
    if rc_sf != 0:
        last = (err_sf or "ImportError").split("\n")[-1][:80]
        return "missing", f"soundfile: {last}"
    return "ok", f"demucs {out.strip()}"


def _chk_modal_secret(name: str) -> tuple[str, str]:
    uv = _find_uv()
    if uv is None:
        return "missing", "uv required"
    rc, out, err = _run([str(uv), "run", "--project", str(SCRIPT_DIR),
                         "modal", "secret", "list"])
    if rc != 0:
        return "missing", "modal secret list failed (auth?)"
    return ("ok", f"'{name}' found") if name in out \
        else ("missing", f"'{name}' does not exist")


def _chk_hf_secret() -> tuple[str, str]:
    return _chk_modal_secret(HF_SECRET_NAME)


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
    # Install torch/torchaudio together with demucs so pip's resolver picks
    # a single ABI-compatible pair instead of drifting apart across separate
    # installs (mismatched torch/torchaudio crashes at import time).
    # soundfile is included even though demucs 4.x no longer depends on it
    # itself (it switched to `sphn`) — separate_demucs.py/separate_sam.py
    # import soundfile directly for their own section-extraction helper.
    _stream(pf, [python, "-m", "pip", "install", "--user",
                 "demucs", "torch", "torchaudio", "soundfile"],
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
    tail: deque[str] = deque(maxlen=40)
    start, timeout, pct = time.monotonic(), 300, 0.1
    while True:
        if time.monotonic() - start > timeout:
            proc.terminate()
            write(pf, "error", pct, "Timeout waiting for login (5 min) — "
                  "the browser tab may not have completed the auth flow")
            return
        line = proc.stdout.readline()
        if line:
            clean = line.rstrip()
            if clean:
                tail.append(clean)
            write(pf, "running", min(pct, 0.5), clean[:120])
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
                tail_lines = list(tail)
                write(pf, "error", pct,
                      _tail_error_message(["modal token new"], rc, tail_lines),
                      ["--- last output lines ---"] + tail_lines if tail_lines else None)
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
        write(pf, "error", 0, "uv not found — install it first (Setup tab, 'Install uv')")
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

    args = p.parse_args()
    dispatch = {
        "check":                cmd_check,
        "install-uv":           cmd_install_uv,
        "sync-deps":            cmd_sync_deps,
        "install-demucs":       cmd_install_demucs,
        "modal-login":          cmd_modal_login,
        "modal-secret-create":  cmd_modal_secret_create,
    }
    try:
        dispatch[args.cmd](args)
    except Exception as exc:
        # Without this, an uncaught exception here (e.g. the resolved
        # REAPER Python path turning out not to exist, so subprocess.run
        # raises FileNotFoundError — the actual bug behind one real report)
        # leaves the progress file stuck on its initial "running|0.00|..."
        # line forever: panel_setup.lua's poll_check()/poll_simple() just
        # wait for a "done"/"error" state with no timeout, so the Setup tab
        # looks permanently hung instead of surfacing the real failure.
        pf = Path(getattr(args, "progress", "") or "") or None
        write(pf, "error", 0, f"{args.cmd} crashed: {exc}")
        raise


if __name__ == "__main__":
    main()
