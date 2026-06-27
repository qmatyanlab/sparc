#!/usr/bin/env python3
"""Upload SPARC large assets to the combined (private) HuggingFace repo.

Run this from the machine that holds the real asset files (e.g. the original NERSC
project dir). It mirrors the local tree so that, on the consumer side, each
``hf filename == utils.assets.REGISTRY key == local destination`` — meaning
``utils.assets.resolve_path`` drops every file back at the path the configs expect.

Files are uploaded one-by-one (HF de-duplicates by content hash, so re-running is
cheap and effectively resumable) to avoid scanning unrelated large directories such
as ``exp_res/``.

Usage:
  export HF_TOKEN=...                 # or: huggingface-cli login
  python scripts/upload_assets.py --source /global/cfs/cdirs/m2663/angush/sparc
  python scripts/upload_assets.py --source <dir> --groups core optional   # skip caches
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.assets import REGISTRY, repo_id, repo_type  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        required=True,
        help="root dir that holds the real asset files (mirrors registry paths)",
    )
    ap.add_argument(
        "--groups",
        nargs="*",
        default=["core", "cached", "optional"],
        help="which registry groups to upload (default: all)",
    )
    ap.add_argument(
        "--public",
        action="store_true",
        help="create the HF repo public (default: private)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be uploaded and exit",
    )
    args = ap.parse_args()

    src = Path(args.source).resolve()
    rid, rtype = repo_id(), repo_type()
    private = not args.public
    wanted = set(args.groups)
    rel_paths = sorted(rel for rel, g in REGISTRY.items() if g in wanted)

    present = [rel for rel in rel_paths if (src / rel).is_file()]
    missing = [rel for rel in rel_paths if not (src / rel).is_file()]

    print(f"Source root : {src}")
    print(f"Target repo : {rid}  (type={rtype}, private={private})")
    print(f"Groups      : {sorted(wanted)}")
    total = sum((src / rel).stat().st_size for rel in present)
    print(f"To upload   : {len(present)} files, {total / 1e9:.2f} GB")
    for rel in present:
        print(f"  + {rel}  ({(src / rel).stat().st_size / 1e6:.1f} MB)")
    for rel in missing:
        print(f"  ! MISSING at source: {rel}")
    if args.dry_run:
        return 0
    if not present:
        print("Nothing to upload.")
        return 1

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(rid, repo_type=rtype, private=private, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{rid}")

    for i, rel in enumerate(present, 1):
        size_mb = (src / rel).stat().st_size / 1e6
        print(f"[{i}/{len(present)}] uploading {rel} ({size_mb:.1f} MB) ...", flush=True)
        api.upload_file(
            path_or_fileobj=str(src / rel),
            path_in_repo=rel,
            repo_id=rid,
            repo_type=rtype,
        )
    print(f"Done. {len(present)} files uploaded to {rid}.")
    if missing:
        print(f"WARNING: {len(missing)} registry files were missing at source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
