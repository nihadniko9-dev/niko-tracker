#!/usr/bin/env bash
# MegaSaM env ported to Python 3.12 / torch 2.10 cu128 / sm_120.   bash backends/megasam/install.sh
# Upstream pins py3.10 + torch 2.0.1 + CUDA 11.8, which cannot run on Blackwell. See port.py.
. "$(dirname "$0")/../../scripts/wsl/lib.sh"

ENV=$(make_env megasam 3.12)
install_torch "$ENV"
SRC="$NIKO_HOME/third_party/mega-sam"
[ -d "$SRC/.git" ] || git clone -q --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/mega-sam/mega-sam "$SRC"
echo "mega-sam commit: $(git -C "$SRC" rev-parse HEAD)  base: $(git -C "$SRC/base" rev-parse HEAD)"
"$ENV/bin/python" "$REPO_DIR/backends/megasam/port.py" "$SRC"

pip_in "$ENV" "numpy<2" opencv-python-headless scipy timm einops huggingface-hub safetensors \
    imageio matplotlib tqdm ninja "setuptools<80" wheel
# No xformers here: on sm_120, xformers 0.0.35 only has the FA2 kernels (fp16/bf16), and
# DINOv2 / UniDepth run in fp32 -> "No operator found". Without it they use PyTorch attention.
uv pip uninstall --python "$ENV/bin/python" xformers 2>/dev/null || true
pip_in "$ENV" --no-deps -e "$SRC/UniDepth"
pip_in "$ENV" -e "$REPO_DIR/backends/megasam/shims"

# DROID-SLAM backends + lietorch CUDA extensions for sm_120 (TORCH_CUDA_ARCH_LIST=12.0 from env.sh)
(cd "$SRC/base" && MAX_JOBS=16 "$ENV/bin/python" setup.py install > "$NIKO_HOME/megasam_build.log" 2>&1) \
    || { tail -n 40 "$NIKO_HOME/megasam_build.log"; exit 1; }
tail -n 3 "$NIKO_HOME/megasam_build.log"

pip_in "$ENV" -e "$REPO_DIR/backends/common" -e "$REPO_DIR/backends/megasam"
freeze "$ENV" megasam
"$ENV/bin/python" -m niko_megasam.probe || echo "probe not green yet"
