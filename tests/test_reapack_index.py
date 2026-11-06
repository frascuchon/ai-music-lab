"""Every file that index.xml tells ReaPack to download must be tracked in git.

ReaPack fetches each <source> from raw.githubusercontent.com/<repo>/main/...;
a file that is missing or gitignored there returns 404 and aborts the install.
"""
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RAW_PREFIX = "https://raw.githubusercontent.com/frascuchon/ai-music-lab/main/"


def _sources():
    tree = ET.parse(ROOT / "index.xml")
    return [(s.get("file"), (s.text or "").strip()) for s in tree.iter("source")]


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


SOURCES = _sources()


def test_index_has_sources():
    assert SOURCES


@pytest.mark.parametrize("name,url", SOURCES, ids=[s[0] for s in SOURCES])
def test_source_is_published(name, url):
    assert url.startswith(RAW_PREFIX), f"unexpected URL for {name}: {url}"
    rel = url[len(RAW_PREFIX):]
    assert (ROOT / rel).is_file(), f"{rel} does not exist in the repo"
    ignored = _git("check-ignore", "-q", rel)
    assert ignored.returncode != 0, f"{rel} is gitignored, so it 404s on GitHub"
    tracked = _git("ls-files", "--error-unmatch", rel)
    assert tracked.returncode == 0, f"{rel} is not tracked by git (404 on GitHub)"


def test_main_script_is_not_empty_and_matches_latest_version():
    """Regression: 1.2.2/1.2.3 shipped an empty ai-music-lab.lua (0 bytes) after a
    version bump truncated it. The ReaPack main script must have real content and
    carry the same @version as the newest <version> in index.xml."""
    main = ROOT / "plugins-and-extensions" / "ai-music-lab.lua"
    text = main.read_text()
    assert len(text) > 1000, f"{main.name} is nearly empty ({len(text)} bytes)"
    assert "@version" in text
    latest = next(ET.parse(ROOT / "index.xml").getroot().iter("version")).get("name")
    declared = text.split("@version", 1)[1].split()[0]
    assert declared == latest, f"@version {declared} != newest index.xml version {latest}"
    assert "function" in text or "dofile" in text or "require" in text


# ── runtime closure ──────────────────────────────────────────────────────────
# ReaPack installs only what the newest <version> lists. A file the plugin loads
# at runtime but that is not listed simply never reaches the user's machine
# (e.g. a model offered in a panel whose Modal script was not shipped).
import ast
import re

PLUGINS = ROOT / "plugins-and-extensions"


def _latest_files():
    latest = next(ET.parse(ROOT / "index.xml").getroot().iter("version"))
    return {s.get("file") for s in latest.iter("source")}


def _runtime_lua():
    for p in PLUGINS.rglob("*.lua"):
        rel = p.relative_to(PLUGINS).as_posix()
        if (p.name.startswith("test_") or "research/" in rel or "evaluation/" in rel
                or p.name in ("gen_test_smf.lua", "luac.out")):
            continue
        yield p


def _referenced_by_lua():
    """(referencing file, path relative to plugins-and-extensions)."""
    found = []
    for p in _runtime_lua():
        text = p.read_text()
        here = p.parent.relative_to(PLUGINS).as_posix()
        here = "" if here == "." else here + "/"
        for m in re.finditer(r'(?:SCRIPT_DIR|SHARED_DIR)\s*\.\.\s*"([^"]+\.(?:py|lua|toml|lock))"', text):
            base = "shared/" if "SHARED_DIR" in m.group(0) else here
            found.append((p, base + m.group(1)))
        for m in re.finditer(r'dofile\(SCRIPT_DIR\s*\.\.\s*"([^"]+)"', text):
            found.append((p, here + m.group(1)))
        # MidiGenerator builds its Modal script names from the model keys
        if '"research_" .. k .. "_modal.py"' in text:
            keys = re.search(r"MG_MODELS\s*=\s*\{([^}]*)\}", text).group(1)
            for k in re.findall(r'"(\w+)"', keys):
                found.append((p, f"{here}research/research_{k}_modal.py"))
        for m in re.finditer(r'require\("(\w+)"\)', text):
            lib = PLUGINS / "shared" / "lib" / f"{m.group(1)}.lua"
            if lib.is_file():
                found.append((p, f"shared/lib/{m.group(1)}.lua"))
    return found


def test_every_file_the_panels_load_is_declared():
    declared = _latest_files()
    missing = sorted({(p.relative_to(PLUGINS).as_posix(), rel)
                      for p, rel in _referenced_by_lua()
                      if (PLUGINS / rel).is_file() and rel not in declared})
    assert not missing, f"loaded at runtime but not in the newest index.xml version: {missing}"


def test_every_modal_script_ships_its_constraints():
    declared = _latest_files()
    missing = []
    for f in sorted(declared):
        if f.endswith("_modal.py") or f.endswith("modal_sam_audio.py"):
            stem = Path(f).stem
            for c in (PLUGINS / "shared" / "modal_constraints").glob(f"{stem}*.txt"):
                rel = f"shared/modal_constraints/{c.name}"
                if rel not in declared:
                    missing.append(rel)
            if not list((PLUGINS / "shared" / "modal_constraints").glob(f"{stem}*.txt")):
                missing.append(f"constraints for {f}")
    assert not missing, f"not shipped: {missing}"


def test_local_imports_and_add_local_file_of_shipped_python_are_declared():
    declared = _latest_files()
    missing = set()
    for f in sorted(x for x in declared if x.endswith(".py")):
        src = (PLUGINS / f).read_text()
        d = Path(f).parent.as_posix()
        for n in ast.walk(ast.parse(src)):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods = [n.module]
            for m in mods:
                for cand in (f"{d}/{m.split('.')[0]}.py", f"shared/{m.split('.')[0]}.py",
                             f"{d}/{m.replace('.', '/')}.py"):
                    if (PLUGINS / cand).is_file() and cand not in declared:
                        missing.add((f, cand))
        for name in re.findall(r'parent(?:\.parent)*\s*/\s*"([\w.]+\.py)"', src):
            for cand in (f"{d}/{name}", f"{Path(d).parent.as_posix()}/{name}"):
                if (PLUGINS / cand).is_file() and cand not in declared:
                    missing.add((f, cand))
    assert not missing, f"imported/uploaded but not shipped: {sorted(missing)}"
