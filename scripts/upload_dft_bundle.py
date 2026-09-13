#!/usr/bin/env python3
"""Upload the consolidated DFT dataset bundle to a HuggingFace **dataset** repo.

A dataset-type repo (not the model-type ``sparc-assets`` repo the checkpoints use) so
reviewers get the interactive dataset viewer over ``materials.csv`` for free.

Nothing is uploaded unless the bundle's own verification passed -- ``--dry-run`` first,
and the upload refuses to run while ``audit/verification.csv`` contains a FAIL.

Note that uploading publishes the bundle to HuggingFace; content pushed there may be
cached or indexed even if it is deleted later. The repo is created **private** by
default; ``--public`` is an explicit opt-in.

Usage:
  export HF_TOKEN=...                      # or: huggingface-cli login
  uv run python scripts/upload_dft_bundle.py --dry-run
  uv run python scripts/upload_dft_bundle.py
  uv run python scripts/upload_dft_bundle.py --repo me/other-name --public
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.assets import _token, dft_dataset_repo_id  # noqa: E402

DEFAULT_SOURCE = ROOT / "dft_dataset"
IGNORE = ["_cache/**", "**/__pycache__/**", "*.pyc", "*.pyo"]


def _size(src: Path):
    nbytes = nfiles = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if rel.startswith("_cache/") or "__pycache__/" in rel or rel.endswith((".pyc", ".pyo")):
            continue
        nbytes += p.stat().st_size
        nfiles += 1
    return nbytes, nfiles


def _verification(src: Path):
    p = src / "audit" / "verification.csv"
    if not p.exists():
        return None, None
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows, [r for r in rows if r.get("verdict") == "FAIL"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--repo", default=None, help=f"default: {dft_dataset_repo_id()}")
    ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="upload even if a check FAILed")
    ap.add_argument("--message", default="Consolidated DFT dataset")
    args = ap.parse_args(argv)

    src = Path(args.source).resolve()
    if not (src / "materials.csv").exists():
        sys.exit(f"no bundle at {src}\nBuild it first: uv run python scripts/build_dft_bundle.py")

    repo_id = args.repo or dft_dataset_repo_id()
    nbytes, nfiles = _size(src)
    rows, fails = _verification(src)

    print(f"Source : {src}")
    print(f"Repo   : {repo_id}  (type=dataset, private={not args.public})")
    print(f"Payload: {nfiles} files, {nbytes / 1e6:.1f} MB  (excluding _cache/)")
    if not rows:
        print("Checks : NONE FOUND -- this bundle carries no verification evidence "
              "(built with --skip-verify?)")
    else:
        print(f"Checks : {len(rows) - len(fails)}/{len(rows)} PASS")
        for f in fails:
            print(f"   FAIL {f['check']}: {f.get('note', '')}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0
    if not rows and not args.force:
        sys.exit("\nRefusing to upload a bundle with no verification evidence. "
                 "Rebuild without --skip-verify, or pass --force to publish anyway.")
    if fails and not args.force:
        sys.exit("\nRefusing to upload while a verification check FAILs. "
                 "Fix it, or pass --force if the failure is understood and accepted.")
    if not _token():
        sys.exit("\nNo HuggingFace token. Set HF_TOKEN or run: huggingface-cli login")

    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    api.create_repo(repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=str(src), repo_id=repo_id, repo_type="dataset",
                      ignore_patterns=IGNORE, commit_message=args.message)
    print(f"\nUploaded -> https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
