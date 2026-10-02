# Shared helpers for backend install scripts (source me).
set -euo pipefail
if [ -f "$HOME/.config/niko/env.sh" ]; then . "$HOME/.config/niko/env.sh"; fi
: "${NIKO_HOME:?NIKO_HOME is not set (run scripts/wsl/10_user.sh)}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

# make_env <name> <python>: create $NIKO_HOME/envs/<name> once
make_env() {
    local env="$NIKO_HOME/envs/$1"
    [ -x "$env/bin/python" ] || uv venv --python "$2" "$env"
    echo "$env"
}

# pip_in <env> <args...>
pip_in() {
    local env="$1"; shift
    uv pip install --python "$env/bin/python" "$@"
}

install_torch() {
    pip_in "$1" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
}

# clone_pinned <url> <dir> [commit]: shallow clone once, check out the commit if given, print HEAD
clone_pinned() {
    local url="$1" dir="$2" commit="${3:-}"
    [ -d "$dir/.git" ] || git clone -q --depth 1 "$url" "$dir" >&2
    if [ -n "$commit" ]; then
        git -C "$dir" fetch -q --depth 1 origin "$commit" >&2 && git -C "$dir" checkout -q "$commit" >&2
    fi
    git -C "$dir" rev-parse HEAD
}

# freeze <env> <name>: record resolved versions next to the backend
freeze() {
    uv pip freeze --python "$1/bin/python" > "$REPO_DIR/backends/$2/requirements.lock.txt"
}
