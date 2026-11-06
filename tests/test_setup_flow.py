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
    sync = subprocess.run([uv, "sync", "--frozen", "--python", "3.11",
                           "--project", str(proj)], capture_output=True, text=True, env=env)
    out = (sync.stderr + sync.stdout).lower()
    if sync.returncode != 0 and any(w in out for w in (
            "dns error", "failed to fetch", "connection", "timed out", "network")):
        pytest.skip("no network access to PyPI")
    assert sync.returncode == 0, sync.stdout + sync.stderr
    ver = subprocess.run([uv, "run", "--frozen", "--project", str(proj),
                          "modal", "--version"], capture_output=True, text=True, env=env)
    assert ver.returncode == 0 and "modal client version" in ver.stdout, ver.stderr


# --- sync-deps self-healing and verification --------------------------------

def _fake_uv_logged(tmp_path, script):
    """Fake uv that logs its arguments to calls.txt, then runs `script`."""
    d = tmp_path / "bin"
    d.mkdir()
    uv = d / "uv"
    uv.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/calls.txt"\n{script}\n')
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return d


def _calls(tmp_path):
    return (tmp_path / "calls.txt").read_text().splitlines()


def test_sync_deps_recovers_from_corrupt_environment(tmp_path):
    """First `uv sync` fails (broken .venv); the helper wipes it and retries
    with --reinstall, then verifies modal, instead of giving up."""
    venv = HELPER.parent / ".venv"
    if venv.exists():
        pytest.skip("shared/.venv exists in this checkout; refusing to delete it")
    d = _fake_uv_logged(tmp_path, '''
case "$*" in
  *--reinstall*) echo "reinstalled"; exit 0;;
  sync*) mkdir -p "%s"; echo "error: failed to inspect python interpreter"; exit 2;;
  run*) echo "modal client version: 1.4.3"; exit 0;;
esac''' % venv)
    try:
        _, out = _run_helper(tmp_path, "sync-deps", d)
    finally:
        shutil.rmtree(venv, ignore_errors=True)
    assert out.startswith("done|1.000|"), out
    calls = _calls(tmp_path)
    assert any("--reinstall" in c for c in calls)
    assert calls[-1].startswith("run") and "modal --version" in calls[-1]


def test_sync_deps_fails_if_modal_cli_is_unusable(tmp_path):
    """A sync that exits 0 but leaves a broken CLI must not report success."""
    d = _fake_uv_logged(tmp_path, '''
case "$1" in
  sync) echo "Installed 1 package"; exit 0;;
  run) echo "ImportError: cannot import name 'api_pb2'" >&2; exit 1;;
esac''')
    _, out = _run_helper(tmp_path, "sync-deps", d)
    first = out.splitlines()[0]
    assert first.startswith("error|") and "cannot import name 'api_pb2'" in first
    report_path = next(l for l in out.splitlines() if l.startswith("REPORT|")).split("|", 1)[1]
    assert "verify modal" in Path(report_path).read_text()


def test_failure_report_contains_every_attempt(tmp_path):
    d = _fake_uv_logged(tmp_path, 'echo "error: network unreachable"; exit 1')
    _, out = _run_helper(tmp_path, "sync-deps", d)
    report_path = next(l for l in out.splitlines() if l.startswith("REPORT|")).split("|", 1)[1]
    report = Path(report_path).read_text()
    assert report.count("network unreachable") >= 2          # both attempts kept
    assert "### uv sync:" in report and "clean environment" in report


def test_check_modal_cli_detail_includes_reason(tmp_path):
    d = _fake_uv(tmp_path, 'if [ "$1" = "--version" ]; then echo uv 1; exit 0; fi\n'
                           'echo "ModuleNotFoundError: No module named grpclib" >&2\nexit 1\n')
    _, out = _run_helper(tmp_path, "check", d)
    line = next(l for l in out.splitlines() if l.startswith("CHECK|modal-cli|"))
    assert "missing" in line and "No module named grpclib" in line


# --- stale reaper.ini Python (Homebrew upgrade) ------------------------------

def _make_python(path, version="3.11.9"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "Python {version}"\n')
    path.chmod(0o755)


def _import_helpers():
    import importlib.util
    spec = importlib.util.spec_from_file_location("setup_helpers_under_test", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stale_brew_cellar_path_resolves_to_upgraded_python(tmp_path):
    """reaper.ini keeps python@3.11/3.11.4_1 but brew upgraded it to 3.11.9_2."""
    brew = tmp_path / "brew"
    _make_python(brew / "opt" / "python@3.11" / "bin" / "python3.11")
    ini = tmp_path / "reaper.ini"
    ini.write_text("pythonlibpath64=" + str(
        brew / "Cellar/python@3.11/3.11.4_1/Frameworks/Python.framework/Versions/3.11/lib") + "\n")
    status, detail = _import_helpers()._chk_python(ini, prefixes=[str(brew)])
    assert status == "ok", detail
    assert "python3.11" in detail and "stale" in detail


def test_stale_path_falls_back_to_newest_python_in_prefix(tmp_path):
    brew = tmp_path / "brew"
    _make_python(brew / "bin" / "python3.12", "3.12.1")
    ini = tmp_path / "reaper.ini"
    ini.write_text(f"pythonlibpath64={tmp_path}/gone/python@3.11/3.11.4/lib\n")
    status, detail = _import_helpers()._chk_python(ini, prefixes=[str(brew)])
    assert status == "ok" and "python3.12" in detail


def test_stale_path_with_no_python_anywhere_reports_missing(tmp_path):
    ini = tmp_path / "reaper.ini"
    ini.write_text(f"pythonlibpath64={tmp_path}/gone/python@3.11/3.11.4/lib\n")
    status, detail = _import_helpers()._chk_python(ini, prefixes=[str(tmp_path / "none")])
    assert status == "missing" and "Python not found" in detail


def test_lua_detect_python_survives_brew_upgrade(tmp_path):
    lua = shutil.which("lua")
    if not lua:
        pytest.skip("lua not installed")
    brew = tmp_path / "brew"
    _make_python(brew / "opt" / "python@3.11" / "bin" / "python3.11")
    res = tmp_path / "res"
    res.mkdir()
    (res / "reaper.ini").write_text(
        f"pythonlibpath64={brew}/Cellar/python@3.11/3.11.4_1/Frameworks/Python.framework/Versions/3.11/lib\n")
    script = f'''
reaper = {{ GetResourcePath = function() return "{res}" end,
            GetOS = function() return "OSX64" end }}
package.path = "{PLUGINS}/shared/lib/?.lua;" .. package.path
local common = dofile("{PLUGINS}/shared/lib/common.lua")
common.PYTHON_PREFIXES = {{ "{brew}" }}
local py, err = common.detect_reaper_python()
io.write(py, "\\n", tostring(err))
'''
    out = subprocess.run([lua, "-e", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    py, err = out.stdout.split("\n", 1)
    assert py == f"{brew}/opt/python@3.11/bin/python3.11", out.stdout
    assert "missing Python" in err


# --- every dependency installs without a compiler on every supported Mac -----

def _locked_packages():
    import tomllib
    data = tomllib.loads((PLUGINS / "shared" / "uv.lock").read_text())
    return [p for p in data["package"] if p["name"] != "reaper-ai-plugins-shared"]


def test_lock_has_prebuilt_wheels_for_intel_and_apple_silicon_macs():
    """Regression: cbor2 6.x had only arm64 wheels, so on an Intel Mac uv built
    it from source ("can't find Rust compiler" — REAPER's PATH has no
    ~/.cargo/bin) and Install deps failed. Every locked package must be
    installable from a wheel on both Mac architectures."""
    problems = []
    for pkg in _locked_packages():
        wheels = [w["url"].rsplit("/", 1)[-1] for w in pkg.get("wheels", [])]
        if any(w.endswith("-none-any.whl") for w in wheels):
            continue
        intel = any("macosx" in w and ("x86_64" in w or "universal2" in w) for w in wheels)
        arm = any("macosx" in w and ("arm64" in w or "universal2" in w) for w in wheels)
        if not (intel and arm):
            problems.append(f"{pkg['name']}=={pkg['version']} (intel={intel}, arm64={arm})")
    assert not problems, "no prebuilt macOS wheel for: " + ", ".join(problems)


def test_cbor2_is_pinned_below_6():
    import tomllib
    deps = tomllib.loads((PLUGINS / "shared" / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any(d.replace(" ", "").startswith("cbor2<6") for d in deps)
    cbor2 = next(p for p in _locked_packages() if p["name"] == "cbor2")
    assert cbor2["version"].startswith("5.")


def test_uv_subprocesses_get_cargo_and_homebrew_on_path(tmp_path, monkeypatch):
    cargo = tmp_path / ".cargo" / "bin"
    cargo.mkdir(parents=True)
    mod = _import_helpers()
    monkeypatch.setattr(mod, "EXTRA_PATH_DIRS", (cargo, tmp_path / "missing"))
    env = mod._tool_env({"PATH": "/usr/bin:/bin"})
    parts = env["PATH"].split(os.pathsep)
    assert str(cargo) in parts and str(tmp_path / "missing") not in parts
    assert parts[:2] == ["/usr/bin", "/bin"]          # existing order preserved


def test_rust_compiler_failure_gets_an_actionable_message(tmp_path):
    body = "\n".join([
        'echo "Building cbor2==6.1.1"',
        'echo "error: can\'t find Rust compiler" >&2',
        "exit 1",
    ]) + "\n"
    d = _fake_uv(tmp_path, body)
    _, out = _run_helper(tmp_path, "sync-deps", d)
    first = out.splitlines()[0]
    assert "cbor2" in first and "no prebuilt wheel" in first and "Rust" in first
