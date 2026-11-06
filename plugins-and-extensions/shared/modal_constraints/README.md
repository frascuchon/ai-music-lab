# Modal image constraints

One file per model image: the exact version of every package the working image
contains. Each image installs it through `PIP_CONSTRAINT` (or, where the layers
contradict each other, installs it directly with `pip install --no-deps -r`), so
rebuilding an image — on another Modal account, after a cache miss — can never
drift to newer releases. Git repositories and `git+` requirements are pinned by
commit inside each script.

Why: an unpinned image kept working on the account that built it long ago and
failed on a fresh build elsewhere (torch 2.15 next to torchaudio 2.11 for
ACE-Step). `tests/test_modal_images.py` enforces all of this.

## Upgrading a model deliberately

1. Remove its constraints file reference temporarily (or build the image without it).
2. `uv run --project shared modal run ../tools/modal_freeze.py --script <script> --out /tmp/x.json`
3. `python3 tools/build_modal_constraints.py <dir with the json>`
4. Test the model end to end, then commit the new constraints file.

Files are generated — don't edit them by hand.
