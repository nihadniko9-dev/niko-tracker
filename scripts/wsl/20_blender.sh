#!/usr/bin/env bash
# Linux Blender into $NIKO_HOME/blender (resumable download, sha256 checked).
#   wsl -d Ubuntu-24.04 --exec bash -lc 'bash "$NIKO_REPO/scripts/wsl/20_blender.sh"'
set -euo pipefail
if [ -f "$HOME/.config/niko/env.sh" ]; then . "$HOME/.config/niko/env.sh"; fi

VER="${BLENDER_VERSION:-5.2.2}"
SERIES="${VER%.*}"
BASE="https://download.blender.org/release/Blender${SERIES}"
TAR="blender-${VER}-linux-x64.tar.xz"
DEST="$NIKO_HOME/blender"
mkdir -p "$DEST/downloads"
cd "$DEST/downloads"

curl -fL -C - -o "$TAR" "$BASE/$TAR"
curl -fsSL -o "blender-${VER}.sha256" "$BASE/blender-${VER}.sha256"
grep " $TAR\$" "blender-${VER}.sha256" | sha256sum -c -

if [ ! -x "$DEST/blender-${VER}-linux-x64/blender" ]; then
    tar -xf "$TAR" -C "$DEST"
fi
"$DEST/blender-${VER}-linux-x64/blender" --version | head -n 1
