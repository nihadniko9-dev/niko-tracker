#!/usr/bin/env bash
# CoTracker3 env: Python 3.12, torch 2.10 cu128.   bash backends/cotracker/install.sh
. "$(dirname "$0")/../../scripts/wsl/lib.sh"

ENV=$(make_env cotracker 3.12)
install_torch "$ENV"
SRC="$NIKO_HOME/third_party/co-tracker"
echo "co-tracker commit: $(clone_pinned https://github.com/facebookresearch/co-tracker "$SRC" "${COTRACKER_COMMIT:-}")"
pip_in "$ENV" -e "$SRC" numpy opencv-python-headless pillow
pip_in "$ENV" -e "$REPO_DIR/backends/common" -e "$REPO_DIR/backends/cotracker"
freeze "$ENV" cotracker
"$ENV/bin/python" -m niko_cotracker.probe
