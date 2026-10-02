#!/usr/bin/env bash
# System packages for Niko Tracker inside WSL2 Ubuntu 24.04. Runs as root:
#   wsl -d Ubuntu-24.04 -u root -- bash "/mnt/d/Niko Tracker/scripts/wsl/00_system_root.sh"
# Never installs a Linux NVIDIA driver: WSL uses the Windows driver (/usr/lib/wsl/lib).
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "must run as root" >&2; exit 1; }
grep -qi microsoft /proc/version || { echo "not running inside WSL" >&2; exit 1; }
. /etc/os-release
[[ "$VERSION_ID" == "24.04" ]] || { echo "expected Ubuntu 24.04, found $VERSION_ID" >&2; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config git git-lfs python3.12-dev \
    curl wget ca-certificates aria2 unzip xz-utils htop \
    ffmpeg \
    libgl1 libegl1 libglib2.0-0 libsm6 libice6 libxext6 libxrender1 libxi6 \
    libxxf86vm1 libxfixes3 libxkbcommon0 libx11-6 libxcursor1 libxinerama1 libxrandr2

# CUDA toolkit 12.8 (nvcc for sm_120) from NVIDIA's WSL-Ubuntu repository: toolkit only, no driver.
if [[ ! -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    cd /tmp
    wget -c https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y cuda-toolkit-12-8
fi

/usr/local/cuda-12.8/bin/nvcc --version
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
echo "system setup done"
