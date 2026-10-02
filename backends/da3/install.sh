#!/usr/bin/env bash
# Depth Anything 3 env: Python 3.12, torch 2.10 cu128 (+ xformers).   bash backends/da3/install.sh
. "$(dirname "$0")/../../scripts/wsl/lib.sh"

ENV=$(make_env da3 3.12)
install_torch "$ENV"
SRC="$NIKO_HOME/third_party/Depth-Anything-3"
echo "DA3 commit: $(clone_pinned https://github.com/ByteDance-Seed/Depth-Anything-3 "$SRC" "${DA3_COMMIT:-}")"
# xformers from the same cu128 index so it matches torch 2.10
pip_in "$ENV" xformers --index-url "$TORCH_INDEX" || echo "xformers: no matching wheel (DA3 may still run without it)"
# addict is imported by depth_anything_3.model but missing from its pyproject
pip_in "$ENV" -e "$SRC" opencv-python-headless addict
pip_in "$ENV" -e "$REPO_DIR/backends/common" -e "$REPO_DIR/backends/da3"
freeze "$ENV" da3
"$ENV/bin/python" -m niko_da3.probe || echo "probe not green yet"
