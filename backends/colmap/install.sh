#!/usr/bin/env bash
# COLMAP 4.2 env via the pycolmap CUDA 12 wheel (no torch).   bash backends/colmap/install.sh
# If the wheel lacks sm_120 kernels, the fallback is a source build of COLMAP 4.2 with
# CMAKE_CUDA_ARCHITECTURES=120 (see PROGRESS.md).
. "$(dirname "$0")/../../scripts/wsl/lib.sh"

ENV=$(make_env colmap 3.12)
pip_in "$ENV" "pycolmap-cuda12==4.2.0" numpy scipy opencv-python-headless
pip_in "$ENV" -e "$REPO_DIR/backends/common" -e "$REPO_DIR/backends/colmap"
freeze "$ENV" colmap
"$ENV/bin/python" -m niko_colmap.probe
