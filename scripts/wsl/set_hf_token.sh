#!/usr/bin/env bash
# Read a Hugging Face token on stdin and store it in ~/.config/niko/secrets.sh (never in the repo).
# Used by set_hf_token.bat (Windows: copies the token from the clipboard).
set -euo pipefail
# keep only the last hf_... token in the input (drops CR/LF, BOM or surrounding text)
T=$(tr -c 'A-Za-z0-9_' '\n' | grep -E '^hf_[A-Za-z0-9]{20,}$' | tail -n 1 || true)
case "$T" in
    hf_*) ;;
    *) echo "That is not a Hugging Face token (it must start with hf_)." >&2; exit 1 ;;
esac
mkdir -p "$HOME/.config/niko"
umask 077
printf 'export HF_TOKEN=%s\n' "$T" > "$HOME/.config/niko/secrets.sh"
chmod 600 "$HOME/.config/niko/secrets.sh"
echo "saved (${#T} characters) in ~/.config/niko/secrets.sh"
