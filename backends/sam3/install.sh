#!/usr/bin/env bash
# SAM 3 / 3.1 env: Python 3.12, torch 2.10 cu128.   bash backends/sam3/install.sh
. "$(dirname "$0")/../../scripts/wsl/lib.sh"

ENV=$(make_env sam3 3.12)
install_torch "$ENV"
SRC="$NIKO_HOME/third_party/sam3"
echo "sam3 commit: $(clone_pinned https://github.com/facebookresearch/sam3 "$SRC" "${SAM3_COMMIT:-}")"
"$ENV/bin/python" "$REPO_DIR/backends/sam3/port.py" "$SRC"
# pycocotools, psutil: imported by sam3 at inference time but missing from its pyproject
pip_in "$ENV" -e "$SRC" einops ninja opencv-python-headless pycocotools psutil
pip_in "$ENV" -e "$REPO_DIR/backends/common" -e "$REPO_DIR/backends/sam3"
freeze "$ENV" sam3
"$ENV/bin/python" -m niko_sam3.probe || echo "probe not green yet (checkpoint pending?)"
