#!/bin/bash
# Niko Tracker - engine image, step 2: inside the imported build copy (never the working distro),
# make it an engine that stands on its own. Author: Nihad Jihad ("Niko").
# Run as root:  prepare.sh <engine-src.tar from git archive> <version>
#   - the engine code is the committed tree of this version, at $NIKO_HOME/engine
#   - the envs' editable installs pointed at the Windows checkout (/mnt/d/Pack/Track Nhad): repointed
#   - env.sh: NIKO_REPO = the engine copy, no CUDA toolkit, uv never re-syncs on its own
#   - machine identity and leftovers cleared
set -euo pipefail
SRC="$1"
VER="$2"
H=/home/rudaw
N=$H/niko
E=$N/engine
OLD="/mnt/d/Pack/Track Nhad"

rm -rf "$E"
mkdir -p "$E"
tar -xf "$SRC" -C "$E"
echo "$VER" > "$E/ENGINE_VERSION"   # the engine code (updated alone by the add-on's update button)
echo "$VER" > "$N/IMAGE_VERSION"    # the environments it runs on (changed only by a new setup)

for f in "$N"/envs/*/lib/python3.12/site-packages/*.pth \
         "$N"/envs/*/lib/python3.12/site-packages/*.dist-info/direct_url.json; do
  if grep -q "$OLD" "$f"; then
    sed -i "s#$OLD#$E#g" "$f"
  fi
done
sed -i "s#file://${OLD// /%20}#file://$E#g" "$N"/envs/*/lib/python3.12/site-packages/*.dist-info/direct_url.json
if grep -rlsE "/mnt/d/Pack/Track( |%20)Nhad" "$N"/envs/*/lib/python3.12/site-packages/*.pth      "$N"/envs/*/lib/python3.12/site-packages/*.dist-info/direct_url.json; then
  echo "editable installs still point at $OLD" >&2
  exit 1
fi

cat > "$H/.config/niko/env.sh" <<'EOF'
# Niko Tracker engine environment (installer/engine/prepare.sh)
export NIKO_HOME="$HOME/niko"
export NIKO_REPO="$NIKO_HOME/engine"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$NIKO_HOME/cache/uv"
export UV_LINK_MODE=hardlink
# the envs ship complete: `uv run` uses them as they are (an engine update re-syncs explicitly)
export UV_NO_SYNC=1
export HF_HOME="$NIKO_HOME/cache/hf"
# the Hugging Face token (only needed to download SAM 3) lives only here, outside the engine
if [ -f "$HOME/.config/niko/secrets.sh" ]; then . "$HOME/.config/niko/secrets.sh"; fi
EOF
# the folders stage.sh left out, empty: models are downloaded into checkpoints at install
mkdir -p "$N/checkpoints" "$N/cache/uv" "$N/cache/hf" "$N/solves"
chown -R 1000:1000 "$E" "$N/IMAGE_VERSION" "$H/.config/niko" "$N/checkpoints" "$N/cache" "$N/solves"
mkdir -p /var/log /var/tmp /var/cache/apt/archives/partial /var/lib/apt/lists/partial
chmod 1777 /var/tmp

: > /etc/machine-id
rm -f /var/lib/dbus/machine-id
apt-get clean >/dev/null 2>&1 || true
echo "prepared engine $VER at $E"
