"""Download model checkpoints listed in niko.registry (resumable; HF_TOKEN from the environment).

    uv run python scripts/fetch_checkpoints.py --list
    uv run python scripts/fetch_checkpoints.py cotracker3_offline da3_nested_giant_large_1.1

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
    args = ap.parse_args()
    by_key = {c.key: c for c in CHECKPOINTS}
    if args.list or not args.keys:
        for c in CHECKPOINTS:
            size = f"{c.size_bytes / 1e9:.2f} GB" if c.size_bytes else "size ?"
            print(f"{c.key:<30} {size:>10}  {c.source}  {c.license}{'  (gated)' if c.gated else ''}")
        return 0
    for key in args.keys:
        if key not in by_key:
            print(f"unknown checkpoint {key!r}", file=sys.stderr)
            return 2
        fetch(by_key[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
