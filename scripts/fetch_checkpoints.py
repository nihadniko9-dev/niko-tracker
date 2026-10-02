"""Download model checkpoints listed in niko.registry (resumable; HF_TOKEN from the environment).

    uv run python scripts/fetch_checkpoints.py --list
    uv run python scripts/fetch_checkpoints.py cotracker3_offline da3_nested_giant_large_1.1
    uv run python scripts/fetch_checkpoints.py --all --ask-token   # the installer's step: every missing one

Files land in $NIKO_HOME/checkpoints/<key>/. Interrupted downloads resume on the next run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from niko.paths import checkpoints_dir
from niko.registry import CHECKPOINTS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def is_complete(ck) -> bool:
    dest = checkpoints_dir() / ck.key
    return all((dest / name).exists() and (dest / f"{name}.sha256").exists() for name in ck.files)


def ask_token() -> str | None:
    """Ask for a Hugging Face token (SAM 3 is gated: accept Meta's license on its page first) and
    store it in ~/.config/niko/secrets.sh, never in the engine. Empty answer: no token."""
    import getpass
    import re

    print()
    print("SAM 3 (the masks of moving things) is gated by Meta. To download it:")
    print("  1. sign in at https://huggingface.co and accept the license at https://huggingface.co/facebook/sam3.1")
    print("  2. make a token (Settings > Access Tokens, type Read) and paste it here.")
    print("Press Enter without a token to skip SAM 3 for now (you can run this again later).")
    m = re.search(r"hf_[A-Za-z0-9]{20,}", getpass.getpass("Hugging Face token (hidden): "))
    if not m:
        return None
    secrets = Path.home() / ".config" / "niko" / "secrets.sh"
    secrets.parent.mkdir(parents=True, exist_ok=True)
    secrets.write_text("export HF_TOKEN=" + m.group(0) + chr(10))
    secrets.chmod(0o600)
    print(f"token saved in {secrets}")
    return m.group(0)


def fetch(ck) -> None:
    from huggingface_hub import hf_hub_download

    kind, _, repo = ck.source.partition(":")
    if kind not in ("hf", "hf-space"):
        raise SystemExit(f"{ck.key}: only hf: / hf-space: sources are handled here (got {ck.source})")
    repo_type = "space" if kind == "hf-space" else "model"
    dest = checkpoints_dir() / ck.key
    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    if ck.gated and not token:
        raise SystemExit(f"{ck.key} is gated: set HF_TOKEN (and request access to {repo})")
    for name in ck.files:
        path = Path(hf_hub_download(repo_id=repo, filename=name, repo_type=repo_type,
                                    revision=ck.revision, local_dir=dest, token=token))
        digest = sha256(path)
        (dest / f"{name}.sha256").write_text(f"{digest}  {name}\n")
        print(f"{ck.key}/{name}: {path.stat().st_size / 1e9:.3f} GB  sha256 {digest[:16]}...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true", help="every checkpoint not downloaded yet")
    ap.add_argument("--ask-token", action="store_true", help="ask for a Hugging Face token when a gated one needs it")
    args = ap.parse_args()
    by_key = {c.key: c for c in CHECKPOINTS}
    if args.list or not (args.keys or args.all):
        for c in CHECKPOINTS:
            size = f"{c.size_bytes / 1e9:.2f} GB" if c.size_bytes else "size ?"
            done = "  downloaded" if is_complete(c) else ""
            print(f"{c.key:<30} {size:>10}  {c.source}  {c.license}{'  (gated)' if c.gated else ''}{done}")
        return 0
    todo = [c for c in CHECKPOINTS if not is_complete(c)] if args.all else []
    for key in args.keys:
        if key not in by_key:
            print(f"unknown checkpoint {key!r}", file=sys.stderr)
            return 2
        todo.append(by_key[key])
    if not todo:
        print("every model is downloaded")
        return 0
    total = sum(c.size_bytes or 0 for c in todo)
    print(f"{len(todo)} model(s) to download, about {total / 1e9:.1f} GB "
          "(interrupted downloads resume when this runs again)")
    if any(c.gated for c in todo) and not os.environ.get("HF_TOKEN") and args.ask_token:
        token = ask_token()
        if token:
            os.environ["HF_TOKEN"] = token
    skipped = []
    for ck in todo:
        if ck.gated and not os.environ.get("HF_TOKEN"):
            skipped.append(ck.key)
            continue
        fetch(ck)
    if skipped:
        print(f"not downloaded (needs a Hugging Face token): {', '.join(skipped)}")
        return 3
    print("every model is downloaded")
    return 0

if __name__ == "__main__":
    sys.exit(main())
