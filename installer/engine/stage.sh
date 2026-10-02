#!/bin/bash
# Niko Tracker - engine image, step 1: the engine distro's root file system as one tar, slimmed.
# Author: Nihad Jihad ("Niko").
# Run as root inside the working engine distro (it is only read):  stage.sh <out.tar>
# Left out, and why:
#   footage, solves, runs, benchmark data, test folders, logs  - Nihad's work / not the engine
#   checkpoints                     - model weights: downloaded from their official sources at install
#   uv cache                        - the envs keep their files (hard links), the cache is not needed
#   /usr/local/cuda-*, /opt/nvidia  - CUDA toolkit and Nsight: build tools, not needed to run (the
#                                     envs' PyTorch brings its CUDA libraries) and not redistributable
#   secrets.sh, shell history, ssh, gh, git credentials, caches - private
#   /usr/lib/wsl, /mnt, /proc, ...  - mounts (--one-file-system), each computer has its own
set -euo pipefail
OUT="$1"
H=/home/rudaw
N=$H/niko
cd /
x=()
for p in \
  "$N/runs" "$N/solves" "$N/bench" "$N/checkpoints" "$N/cache" "$N/pytest_tmp" "$N/smoke_tmp" \
  "$N/testclips" "$N/blender" "$N/tmp_*" "$N/*.log" "$N/hardware.json" \
  "$H/.cache" "$H/.nv" "$H/.config/niko/secrets.sh" "$H/.bash_history" "$H/.python_history" "$H/.lesshst" \
  "$H/.viminfo" "$H/.ssh" "$H/.config/gh" "$H/.git-credentials" "$H/.gitconfig" "$H/.wget-hsts" \
  "$H/.local/share/Trash" "$H/.motd_shown" \
  /root/.bash_history /root/.cache /root/.ssh "/root/*.log" \
  "/usr/local/cuda*" /opt/nvidia \
  /var/cache/apt /var/lib/apt/lists /var/log /var/tmp /var/crash /var/lib/systemd/coredump \
  "/tmp/*" /swapfile; do
  x+=("--exclude=.$p")
done
# exit 1 = "file changed as we read it" (/sys and friends while the distro runs): the archive is complete
tar --one-file-system --numeric-owner --xattrs --acls --anchored --warning=no-file-changed   -cpf "$OUT" "${x[@]}" . || [ $? -eq 1 ]
ls -la "$OUT"
