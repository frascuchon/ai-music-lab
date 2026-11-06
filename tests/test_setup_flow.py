"""Setup tab: 'Install deps' (sync-deps) must work, report real errors, and
leave the Modal CLI configured.

Regression: setup_helpers.py used `Path | None` annotations without
`from __future__ import annotations`; under a Python < 3.10 it died at import
time before writing any progress, so the panel showed a bare "Error" (or hung)
and the user could not configure the tool.
"""
import ast
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGINS = ROOT / "plugins-and-extensions"
HELPER = PLUGINS / "shared" / "setup_helpers.py"
PANEL = PLUGINS / "shared" / "panel_setup.lua"

# Scripts launched from Lua with REAPER's configured Python (any 3.x).
REAPER_RUN_SCRIPTS = [
    PLUGINS / "shared" / "setup_helpers.py",
    PLUGINS / "Audio2Midi" / "transcribe.py",
    PLUGINS / "MidiGenerator" / "midigen.py",
    PLUGINS / "Text2Audio" / "text2audio.py",
]


def _uses_pep604_annotation(tree):
    def has_bitor(node):
        return node is not None and any(
            isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)
            for n in ast.walk(node))

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = n.args.args + n.args.kwonlyargs + n.args.posonlyargs
            if any(has_bitor(a.annotation) for a in args) or has_bitor(n.returns):
                return True
        if isinstance(n, ast.AnnAssign) and has_bitor(n.annotation):
            return True
    return False


def _has_future_annotations(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names)
               for n in tree.body)


@pytest.mark.parametrize("script", REAPER_RUN_SCRIPTS, ids=lambda p: p.name)
def test_reaper_scripts_import_on_old_python(script):
    tree = ast.parse(script.read_text())
    if _uses_pep604_annotation(tree):
        assert _has_future_annotations(tree), (
            f"{script.name} uses `X | None` annotations without "
            "`from __future__ import annotations`; it crashes at import on "
            "Python < 3.10")


@pytest.mark.parametrize("script", REAPER_RUN_SCRIPTS, ids=lambda p: p.name)
def test_reaper_scripts_import_under_real_old_python(script):
    old = next((p for v in ("3.8", "3.9") if (p := shutil.which(f"python{v}"))),
               None) or ("/usr/bin/python3" if _is_old("/usr/bin/python3") else None)
    if not old:
        pytest.skip("no Python < 3.10 available")
    code = (
        "import importlib.util,sys;sys.path.insert(0,sys.argv[2]);"
        "s=importlib.util.spec_from_file_location('m',sys.argv[1]);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)")
    r = subprocess.run([old, "-I", "-c", code, str(script), str(script.parent)],
                       capture_output=True, text=True)
    # Only import-time TypeErrors/SyntaxErrors matter; missing third-party
    # modules (ImportError) are not what this test is about.
    assert "TypeError" not in r.stderr and "SyntaxError" not in r.stderr, r.stderr


def _is_old(py):
    if not Path(py).exists():
        return False
    r = subprocess.run([py, "-c", "import sys;print(sys.version_info<(3,10))"],
                       capture_output=True, text=True)
    return r.stdout.strip() == "True"


# --- sync-deps behaviour with a fake `uv` -----------------------------------

def _fake_uv(tmp_path, body):
    d = tmp_path / "bin"
    d.mkdir()
    uv = d / "uv"
    uv.write_text("#!/bin/sh\n" + body)
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return d


def _run_helper(tmp_path, cmd, path_dir):
    pf = tmp_path / "progress.txt"
    env = {"PATH": f"{path_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    r = subprocess.run([sys.executable, "-I", str(HELPER), cmd, "--progress", str(pf)],
                       capture_output=True, text=True, env=env)
    return r, pf.read_text() if pf.exists() else ""


def test_sync_deps_success_reports_done(tmp_path):
    d = _fake_uv(tmp_path, 'echo "Installed 1 package"\nexit 0\n')
    r, out = _run_helper(tmp_path, "sync-deps", d)
    assert r.returncode == 0, r.stderr
    assert out.startswith("done|1.000|")


def test_sync_deps_failure_reports_real_reason(tmp_path):
    d = _fake_uv(tmp_path, 'echo "resolving..."\n'
                           'echo "error: Failed to build modal: no network"\nexit 2\n')
    _, out = _run_helper(tmp_path, "sync-deps", d)
    first = out.splitlines()[0]
    assert first.startswith("error|")
    assert "Failed to build modal: no network" in first, first   # not a bare code
    assert "--- last output lines ---" in out


def test_sync_deps_without_uv_says_so(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    # PATH has /usr/bin:/bin only; make sure the real uv is not picked up.
    if shutil.which("uv", path="/usr/bin:/bin"):
        pytest.skip("uv installed in system PATH")
    r, out = _run_helper(tmp_path, "sync-deps", empty)
    assert out.startswith("error|") and "uv not found" in out


def test_check_reports_modal_cli_ok_after_sync(tmp_path):
    d = _fake_uv(tmp_path, 'echo "modal client version: 1.4.3"\nexit 0\n')
    _, out = _run_helper(tmp_path, "check", d)
    assert "CHECK|uv|ok|" in out
    assert "CHECK|modal-cli|ok|" in out


def test_shared_project_is_installable():
    """The lockfile matches pyproject.toml and pins a usable modal."""
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv not installed")
    shared = PLUGINS / "shared"
    r = subprocess.run([uv, "lock", "--check", "--offline", "--project", str(shared)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "name = \"modal\"" in (shared / "uv.lock").read_text()


# --- Lua panel ---------------------------------------------------------------

def test_panel_shows_error_details_for_install_deps():
    src = PANEL.read_text()
    assert 'ST.sync_msg ~= "" and ST.sync_msg' in src, \
        "Install deps error must display its message, not just 'Error'"
    assert "text_wrapped" in src


def test_launcher_reports_interpreter_crash(tmp_path):
    """Run the shell command the panel builds against an interpreter that dies
    at import time; the progress file must end up as 'error|...' with the
    traceback's last line instead of staying on 'running'."""
    lua = shutil.which("lua")
    if not lua:
        pytest.skip("lua not installed")
    src = PANEL.read_text()
    start = src.index("local function build_cmd")
    body = src[start:src.index("\nend\n", start) + 5]
    boom = tmp_path / "boom.py"
    boom.write_text("raise TypeError('unsupported operand type(s) for |')\n")
    pf, log = tmp_path / "pf.txt", tmp_path / "log.txt"
    pf.write_text("running|0.00|Starting...")
    script = (
        f'local function q(s) return "\'"..s.."\'" end\n'
        f'local TMPDIR="{tmp_path}/"\nlocal PYTHON="{sys.executable}"\n'
        f'local HELPER="{boom}"\n{body}\n'
        f'io.write(build_cmd("sync-deps", "{pf}"))\n')
    cmd = subprocess.run([lua, "-e", script], capture_output=True, text=True,
                         check=True).stdout
    cmd = cmd.replace(f"{tmp_path}/reaperai_setup.log", str(log))
    subprocess.run(["sh", "-c", cmd.rstrip(" &") .rstrip()], check=False,
                   env={**os.environ})
    out = pf.read_text()
    assert out.startswith("error|"), out
    assert "unsupported operand" in out


# --- failure diagnostics -----------------------------------------------------

def test_failed_install_writes_diagnostic_report(tmp_path):
    d = _fake_uv(tmp_path,
                 'if [ "$1" = "--version" ]; then echo "uv 9.9.9"; exit 0; fi\n'
                 'if [ "$1" = "python" ]; then echo "/opt/py/bin/python3.11"; exit 0; fi\n'
                 'echo "Resolving dependencies"\n'
                 'echo "error: No solution found: modal requires grpclib"\nexit 1\n')
    _, out = _run_helper(tmp_path, "sync-deps", d)
    lines = out.splitlines()
    assert "No solution found: modal requires grpclib" in lines[0]
    report_line = next(ln for ln in lines if ln.startswith("REPORT|"))
    report = Path(report_line.split("|", 1)[1]).read_text()
    # what ran, how it ended, in which environment, and the full output
    assert "sync --project" in report and "exit code:       1" in report
    assert "uv 9.9.9" in report and "/opt/py/bin/python3.11" in report
    assert "helper python:" in report and "PATH:" in report
    assert "Python >=3.11" in report and "pypi.org" in report
    assert "Resolving dependencies" in report          # not just the last line
    assert "modal requires grpclib" in report


def test_missing_uv_also_writes_report(tmp_path):
    if shutil.which("uv", path="/usr/bin:/bin"):
        pytest.skip("uv installed in system PATH")
    empty = tmp_path / "empty"
    empty.mkdir()
    _, out = _run_helper(tmp_path, "sync-deps", empty)
    assert "REPORT|" in out
    assert "NOT FOUND" in Path(
        next(l for l in out.splitlines() if l.startswith("REPORT|")).split("|", 1)[1]
    ).read_text()


def test_helper_crash_writes_report_with_traceback(tmp_path):
    pf = tmp_path / "p.txt"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    # --python <missing> makes pip launch fail inside the helper
    subprocess.run([sys.executable, "-I", str(HELPER), "install-demucs",
                    "--python", str(tmp_path / "no-python"), "--progress", str(pf)],
                   capture_output=True, text=True, env=env)
    out = pf.read_text()
    assert out.startswith("error|") and "REPORT|" in out


def test_panel_exposes_report_to_user():
    src = PANEL.read_text()
    assert "REPORT|" in src and "Open report" in src
    for key in ("uv_state", "sync_state", "login_state", "demucs_state"):
        assert f'error_details("{key}"' in src


def test_modal_python_requirement_matches_project():
    """modal 1.x needs Python >=3.10; the project requires >=3.11 (the
    interpreter used on the reporting machine), so 3.11 must be accepted."""
    import tomllib
    meta = tomllib.loads((PLUGINS / "shared" / "pyproject.toml").read_text())
    assert meta["project"]["requires-python"].startswith(">=3.11")
    lock = (PLUGINS / "shared" / "uv.lock").read_text()
    assert 'requires-python = ">=3.11"' in lock


def test_real_install_on_python_311_configures_modal(tmp_path):
    """End to end on the Python of the reporting machine: a clean `uv sync`
    from the shipped pyproject/uv.lock installs a working modal CLI."""
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv not installed")
    found = subprocess.run([uv, "python", "find", "3.11"], capture_output=True, text=True)
    if found.returncode != 0:
        pytest.skip("Python 3.11 not available")
    proj = tmp_path / "shared"
    proj.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(PLUGINS / "shared" / name, proj / name)
    env = {**os.environ, "UV_PYTHON_DOWNLOADS": "never"}
    sync = subprocess.run([uv, "sync", "--frozen", "--offline", "--python", "3.11",
                           "--project", str(proj)], capture_output=True, text=True, env=env)
    if sync.returncode != 0 and "offline" in (sync.stderr + sync.stdout).lower():
        pytest.skip("packages not in the local uv cache")
    assert sync.returncode == 0, sync.stdout + sync.stderr
    ver = subprocess.run([uv, "run", "--frozen", "--offline", "--project", str(proj),
                          "modal", "--version"], capture_output=True, text=True, env=env)
    assert ver.returncode == 0 and "modal client version" in ver.stdout, ver.stderr
