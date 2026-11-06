"""Every shipped Modal image must be reproducible.

Regression (ACE-Step): images installed torch/torchaudio unpinned and cloned
repositories at HEAD. Modal caches an image per account, so they kept working
where they were built long ago and broke on a fresh build elsewhere (torch 2.15
next to torchaudio 2.11 -> torchaudio dependency error). The fix for ALL models:

* ``shared/modal_constraints/<script>.txt`` pins every package of the image that
  is known to work, and the image installs it through ``PIP_CONSTRAINT``;
* every ``git clone`` is checked out at an exact commit and every ``git+https``
  requirement carries one.
"""
import ast
import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGINS = ROOT / "plugins-and-extensions"
CONSTRAINTS = PLUGINS / "shared" / "modal_constraints"
SHA = re.compile(r"\b[0-9a-f]{40}\b")


def _index_sources():
    return [(s.get("file"), (s.text or "").strip())
            for s in next(ET.parse(ROOT / "index.xml").getroot().iter("version")).iter("source")]


def _is_modal_script(name):
    return name.endswith("_modal.py") or name.endswith("modal_sam_audio.py")


def _shipped_scripts():
    """Modal scripts in the newest ReaPack version (what users install today)."""
    return sorted(f for f, _ in _index_sources() if _is_modal_script(f))


def _all_scripts():
    """Every Modal script any ReaPack version ever shipped; retired models keep
    working as research scripts, so they must stay reproducible too."""
    return sorted({s.get("file") for s in ET.parse(ROOT / "index.xml").getroot().iter("source")
                   if _is_modal_script(s.get("file"))})


SHIPPED = _shipped_scripts()
SCRIPTS = _all_scripts()


def _constraint_files(script: str):
    stem = Path(script).stem
    if stem == "research_musecoco_modal":
        return [CONSTRAINTS / f"{stem}.conda.txt", CONSTRAINTS / f"{stem}.system.txt"]
    return [CONSTRAINTS / f"{stem}.txt"]


def _source(script):
    return (PLUGINS / script).read_text()


def _image_calls(script):
    """(method, args-as-source-text) for every call of the `image = (...)` chain —
    only real code, so comments and docstrings never count."""
    tree = ast.parse(_source(script))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "image":
            out, cur = [], node.value
            while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
                out.append((cur.func.attr, " ".join(ast.unparse(a) for a in
                                                    list(cur.args) + [k.value for k in cur.keywords])))
                cur = cur.func.value
            return list(reversed(out))
    raise AssertionError(f"{script}: no `image = (...)`")


def _image_text(script):
    return "\n".join(args for _, args in _image_calls(script))


def _pins(path: Path):
    return {line.split("==")[0]: line.split("==")[1]
            for line in path.read_text().splitlines()
            if line and not line.startswith("#")}


def test_there_are_modal_scripts():
    assert len(SHIPPED) >= 10 and len(SCRIPTS) >= 24 and set(SHIPPED) <= set(SCRIPTS)


# --- constraints files -------------------------------------------------------

@pytest.mark.parametrize("script", SCRIPTS)
def test_constraints_exist_and_are_wellformed(script):
    for path in _constraint_files(script):
        assert path.is_file(), f"{path.name} missing"
        lines = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
        assert len(lines) > 20, f"{path.name} suspiciously short"
        seen = set()
        for l in lines:
            assert re.fullmatch(r"[a-z0-9][a-z0-9._-]*==[0-9][A-Za-z0-9._!-]*", l), \
                f"{path.name}: bad constraint {l!r}"            # exact, no +local, no ranges
            name = l.split("==")[0]
            assert name not in seen, f"{path.name}: duplicate {name}"
            seen.add(name)
        assert not (seen & {"pip", "setuptools", "wheel"}), "build tooling must not be constrained"


@pytest.mark.parametrize("script", SCRIPTS)
def test_constraints_pin_the_ml_stack(script):
    """Whatever torch-family / transformers / numpy the model uses is pinned."""
    text = _source(script)
    for path in _constraint_files(script):
        pins = _pins(path)
        if "conda" in path.name or "system" in path.name:
            continue
        for pkg in ("numpy", "protobuf"):
            assert pkg in pins, f"{path.name}: {pkg} not pinned"
        if re.search(r"torch|transformers|audiocraft|stable-audio", text):
            assert "torch" in pins or "tensorflow-cpu" in pins, f"{path.name}: torch not pinned"


# --- the image really installs them ------------------------------------------

def _image_chain(script):
    """Method names (with first string arg) of the `image = (...)` chain, in order."""
    tree = ast.parse(_source(script))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "image":
            calls, cur = [], node.value
            while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
                calls.append((cur.func.attr, ast.unparse(cur)))
                cur = cur.func.value
            return list(reversed(calls))
    raise AssertionError(f"{script}: no `image = (...)`")


@pytest.mark.parametrize("script", SCRIPTS)
def test_image_applies_constraints_before_any_pip_step(script):
    chain = _image_chain(script)
    if Path(script).stem == "research_musecoco_modal":
        joined = " ".join(c for _, c in chain)
        assert "modal_constraints_conda.txt" in joined and "modal_constraints_system.txt" in joined
        assert joined.count("_PIP_CONDA") >= 3, "every conda pip call must use its constraints"
        assert "extra_options" in joined
        return
    kinds = [k for k, _ in chain]
    env_idx = next(i for i, (k, c) in enumerate(chain)
                   if k == "env" and "PIP_CONSTRAINT" in c)
    file_idx = next(i for i, (k, c) in enumerate(chain)
                    if k == "add_local_file" and "copy=True" in c and "_CONSTRAINTS" in c)
    first_pip = next(i for i, (k, c) in enumerate(chain)
                     if k == "pip_install" or (k == "run_commands" and "pip" in c))
    assert file_idx < env_idx < first_pip, f"{script}: constraints must precede pip steps: {kinds}"


@pytest.mark.parametrize("script", SCRIPTS)
def test_constraints_path_points_to_the_shipped_file(script):
    src = _source(script)
    for path in _constraint_files(script):
        assert path.name in src or path.stem.split(".")[0] in src


# --- git references are pinned ----------------------------------------------

@pytest.mark.parametrize("script", SCRIPTS)
def test_every_git_clone_is_checked_out_at_a_commit(script):
    for method, args in _image_calls(script):
        if "git clone" not in args:
            continue
        assert re.search(r"checkout --detach\s+(\{\w+\}|[0-9a-f]{40})", args), \
            f"{script}: `git clone` without checkout of an exact commit: {args[:120]!r}"
        assert "--depth" not in args, "shallow clone cannot check out a sha"


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_git_requirement_has_a_commit(script):
    for m in re.finditer(r"git\+https://[^\s'\"]+", _image_text(script)):
        assert re.search(r"@[0-9a-f]{40}$", m.group(0)), \
            f"{script}: unpinned git requirement {m.group(0)}"


@pytest.mark.parametrize("script", SCRIPTS)
def test_explicit_pins_agree_with_constraints(script):
    """A `pkg==x` literal in the script that contradicts the constraints would make
    pip fail ("conflicting constraint") — or silently mislead about what runs."""
    pins = {}
    paths = _constraint_files(script)
    if Path(script).stem == "research_musecoco_modal":
        paths = [p for p in paths if "conda" in p.name]
    for path in paths:
        pins.update(_pins(path))
    problems = []
    for m in re.finditer(r"""(?:^|["' =])([A-Za-z][A-Za-z0-9_.-]*)==([0-9][^"' ,);]*)""",
                         _image_text(script), re.M):
        name = re.sub(r"[-_.]+", "-", m.group(1)).lower()
        want, got = m.group(2).split("+")[0], pins.get(name)
        if got is not None and got != want:
            problems.append(f"{name}=={want} in script vs {got} in constraints")
    assert not problems, "; ".join(problems)


# --- shipped via ReaPack -----------------------------------------------------

def test_every_constraints_file_is_listed_in_the_index():
    """A script shipped without its constraints file would fail to build on the
    user's machine (add_local_file of a missing path)."""
    listed = {f for f, _ in _index_sources()}
    for script in SHIPPED:
        for path in _constraint_files(script):
            assert f"shared/modal_constraints/{path.name}" in listed, \
                f"{path.name} not shipped by index.xml (users would get a build error)"
    for f in listed:
        if "modal_constraints/" in f:
            assert (PLUGINS / f).is_file(), f"{f} listed but missing"


# --- the scripts still import and still build an image ------------------------

@pytest.mark.parametrize("script", SCRIPTS)
def test_script_imports_and_defines_an_image(script, monkeypatch):
    modal = pytest.importorskip("modal")
    path = PLUGINS / script
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location(f"t_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    assert isinstance(mod.image, modal.Image)
    for p in _constraint_files(script):
        assert p.is_file()


# --- ACE-Step specific pins stay in sync -------------------------------------

def test_acestep_scripts_share_pins():
    gen = _source("Text2Audio/research/research_acestep_gen_modal.py")
    edit = _source("Text2Audio/research/research_acestep_edit_modal.py")
    for const in ("ACESTEP_REF", "TORCH_PINS"):
        g = re.search(rf"^{const} = .*?(?=\n\S)", gen, re.S | re.M).group(0)
        e = re.search(rf"^{const} = .*?(?=\n\S)", edit, re.S | re.M).group(0)
        assert g == e, f"{const} differs between ACE-Step gen and edit"
    assert _pins(CONSTRAINTS / "research_acestep_gen_modal.txt")["torch"] == "2.10.0"


# --- tooling -----------------------------------------------------------------

def test_constraint_builder_normalises_a_freeze(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import build_modal_constraints as b
    monkeypatch.setattr(b, "OUT", tmp_path)
    b.write(tmp_path / "x.txt", "demo", [
        "Torch==2.4.0+cu121", "pip==25.1", "setuptools==65.5.1", "Foo_Bar==1.0",
        "ace-step @ file:///root/ACEStep", "dacvae @ git+https://x/y.git@" + "a" * 40,
    ])
    text = (tmp_path / "x.txt").read_text()
    assert "torch==2.4.0\n" in text and "foo-bar==1.0" in text
    assert "pip==" not in text and "setuptools==" not in text
    assert "# dacvae @ git+https://x/y.git@" + "a" * 40 in text
    assert "ace-step" not in text.replace("# ", "") or "file://" not in text.split("Direct")[0]


# --- images whose layers contradict each other install the exact final state ---

LOCKFILE_INSTALL = "pip install --no-deps -r /opt/modal_constraints.txt"
EXPECTED_LOCKFILE = {
    "Text2Audio/research/research_audiogen_modal.py",
    "Text2Audio/research/research_magnet_modal.py",
    "Text2Audio/research/research_inspiremusic_gen_modal.py",
    "Text2Audio/research/research_inspiremusic_modal.py",
    "StemsSeparator/modal_sam_audio.py",
}


@pytest.mark.parametrize("script", sorted(EXPECTED_LOCKFILE))
def test_conflicting_layers_use_exact_final_state(script):
    """audiocraft pins torch==2.1.0 while the image runs a newer torch; sam-audio's
    dependencies contradict the protobuf the Modal client needs; InspireMusic's
    requirements.txt is overridden by a later peft upgrade. pip's resolver rejects
    all of these once versions are constrained, so the working image's exact package
    set is installed without resolution instead."""
    assert LOCKFILE_INSTALL in _image_text(script)
    methods = [m for m, _ in _image_calls(script)]
    # nothing may re-resolve (and so re-conflict) the stack afterwards
    after = _image_calls(script)
    idx = next(i for i, (_, a) in enumerate(after) if LOCKFILE_INSTALL in a)
    for method, args in after[idx + 1:]:
        if method == "pip_install":
            assert all(re.match(r"[\w.\-\[\]]+[><=!~]=", a.strip("'\" "))
                       for a in args.replace(",", " ").split() if a.strip("'\" ")
                       and not a.startswith(("extra", "git+"))), \
                f"{script}: pip_install after the lockfile step re-resolves: {args}"


def test_lockfile_scripts_are_exactly_the_expected_ones():
    found = {s for s in SCRIPTS if LOCKFILE_INSTALL in _image_text(s)}
    assert found == EXPECTED_LOCKFILE, f"unexpected change in lockfile-mode images: {found ^ EXPECTED_LOCKFILE}"


def test_audiocraft_images_pin_audiocraft_itself():
    for name in ("audiogen", "magnet"):
        pins = _pins(CONSTRAINTS / f"research_{name}_modal.txt")
        assert pins["audiocraft"] == "1.3.0" and pins["torch"] == "2.14.0"


def test_sam_audio_git_requirements_are_pinned():
    text = _image_text("StemsSeparator/modal_sam_audio.py")
    for pkg in ("sam_audio", "dacvae", "imagebind", "laion_clap", "perception-models", "pytorchvideo"):
        assert re.search(rf"{pkg} @ git\+https://[^ ']+@[0-9a-f]{{40}}", text), f"{pkg} not pinned by commit"


# --- CUDA library collisions in whole-set installs ----------------------------

@pytest.mark.parametrize("script", sorted(EXPECTED_LOCKFILE))
def test_lockfile_has_no_cu12_next_to_cu13_libraries(script):
    """Regression: AudioGen failed on the GPU with
    `libtorch_cuda.so: undefined symbol: ncclCommResume`. Its package set held the
    nvidia-*-cu12 leftovers of audiocraft's torch 2.1.0 next to torch 2.14's cu13
    libraries; both ship libnccl.so.2 and installing the set at once let the stale
    one win. Sequential builds hid it because the newer wheels came last."""
    for path in _constraint_files(script):
        names = set(_pins(path))
        if any(n.startswith("nvidia-") and n.endswith("-cu13") for n in names):
            stale = sorted(n for n in names if n.startswith("nvidia-") and n.endswith("-cu12"))
            assert not stale, f"{path.name}: stale CUDA 12 libraries {stale}"


def test_builder_drops_stale_cuda12_only_when_torch_is_cuda13(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import build_modal_constraints as b
    pins = {"torch": "2.14.0", "nvidia-nccl-cu12": "2.18.1", "nvidia-nccl-cu13": "2.30.7",
            "nvidia-cublas-cu12": "12.1.3.1"}
    assert sorted(b.drop_stale_cuda(pins)) == ["nvidia-cublas-cu12", "nvidia-nccl-cu12"]
    assert set(pins) == {"torch", "nvidia-nccl-cu13"}
    only12 = {"torch": "2.4.0", "nvidia-nccl-cu12": "2.20.5"}
    assert b.drop_stale_cuda(only12) == [] and "nvidia-nccl-cu12" in only12   # nothing to collide with
