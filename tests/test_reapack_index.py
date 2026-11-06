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
