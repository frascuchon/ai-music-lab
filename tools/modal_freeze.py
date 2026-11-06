#!/usr/bin/env python3
"""Record what a model's Modal image actually contains.

Usage (from plugins-and-extensions/):
    uv run --project shared modal run ../tools/modal_freeze.py \
        --script Text2Audio/research/research_sao_edit_modal.py --out /tmp/sao_edit.json

Mounts the script's own ``image`` on a CPU function and returns:
  * ``pip freeze --all`` for every interpreter found (the system one and, for
    MuseCoco, its conda one);
  * the Python version;
  * every git checkout inside the image (path, remote URL, HEAD sha).

``tools/build_modal_constraints.py`` turns this into the per-model constraints
files in plugins-and-extensions/shared/modal_constraints/, which every image
installs with PIP_CONSTRAINT so a rebuild can never drift to newer releases.
Run it again (with the constraints removed) only when you deliberately want to
upgrade a model.
"""
import importlib.util
import json
import sys
from pathlib import Path

import modal

_script = None
for i, a in enumerate(sys.argv):
    if a == "--script" and i + 1 < len(sys.argv):
        _script = Path(sys.argv[i + 1]).resolve()
    elif a.startswith("--script="):
        _script = Path(a.split("=", 1)[1]).resolve()

_image = None
if _script is not None:               # in the container the image is already built
    sys.path.insert(0, str(_script.parent))
    _spec = importlib.util.spec_from_file_location("target_script", _script)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _image = _mod.image

app = modal.App("modal-freeze", image=_image)


@app.function(timeout=1800)
def probe():
    import os
    import subprocess
    import sys as _sys

    def sh(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return r.stdout.strip()

    pythons = {}
    for exe in dict.fromkeys([_sys.executable, "/opt/conda/bin/python3.8"]):
        if os.path.exists(exe):
            pythons[exe] = {
                "version": sh(f"{exe} -c 'import platform;print(platform.python_version())'"),
                "freeze": sh(f"{exe} -m pip freeze --all").splitlines(),
            }
    repos = []
    for g in sh("find / -xdev -maxdepth 4 -name .git -type d 2>/dev/null").splitlines():
        d = os.path.dirname(g)
        repos.append({"path": d,
                      "url": sh(f"git -C {d} config --get remote.origin.url"),
                      "sha": sh(f"git -C {d} rev-parse HEAD")})
    return {"pythons": pythons, "repos": repos}


@app.local_entrypoint()
def main(script: str, out: str):
    data = probe.remote()
    data["script"] = script
    Path(out).write_text(json.dumps(data, indent=1))
    print(f"[freeze] {script}: {sum(len(p['freeze']) for p in data['pythons'].values())} "
          f"packages, {len(data['repos'])} repos → {out}")
