#!/usr/bin/env python3
"""Upload paper RL run *visualization bundles* to the SPARC HuggingFace repo.

A bundle is a run directory under ``exp_res/`` MINUS the large ``models/`` checkpoints
(~40-65 MB/run) -- everything needed to regenerate the paper figures. Bundles go to
``runs/<name>/`` in the same repo used by ``utils.assets`` (default
``AngusHsuPhys/sparc-assets``); the consumer side fetches them with
``scripts/download_paper_runs.py``.

Usage:
  export HF_TOKEN=...                       # or: huggingface-cli login
  python scripts/upload_paper_runs.py                     # all PAPER_RUNS from exp_res/
  python scripts/upload_paper_runs.py --dry-run           # size preview, no upload
  python scripts/upload_paper_runs.py --runs <name> ...   # a subset
  python scripts/upload_paper_runs.py --source <dir>      # runs live somewhere else
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.assets import PAPER_RUNS, RUN_PREFIX, repo_id, repo_type  # noqa: E402

# Skip the RL checkpoints and python caches; keep samples/rewards/metrics/deliverables/...
IGNORE_PATTERNS = ["models/**", "**/__pycache__/**", "*.pyc", "*.pyo"]


def _bundle_size(run_dir: Path) -> tuple[int, int]:
    """(bytes, file_count) of everything that WOULD upload (i.e. excluding models/)."""
    nbytes = nfiles = 0
    for p in run_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir).as_posix()
        if rel.startswith("models/") or "__pycache__/" in rel or rel.endswith((".pyc", ".pyo")):
            continue
        nbytes += p.stat().st_size
        nfiles += 1
    return nbytes, nfiles


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", default=None, help="dir holding the run folders (default: <repo>/exp_res)")
    ap.add_argument("--runs", nargs="*", default=None, help="subset of run names (default: all PAPER_RUNS)")
    ap.add_argument("--public", action="store_true", help="create the HF repo public (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="show sizes and exit without uploading")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    src_root = Path(args.source).resolve() if args.source else (root / "exp_res")
    names = args.runs or list(PAPER_RUNS)
    rid, rtype = repo_id(), repo_type()

    present, missing, total = [], [], 0
    for n in names:
        d = src_root / n
        if d.is_dir():
            nbytes, nfiles = _bundle_size(d)
            present.append((n, d, nbytes, nfiles))
            total += nbytes
        else:
            missing.append(n)

    print(f"Source root : {src_root}")
    print(f"Target repo : {rid}  (type={rtype}, prefix={RUN_PREFIX}/, private={not args.public})")
    print(f"Bundles     : {len(present)} runs, {total / 1e6:.1f} MB (models/ excluded)")
    for n, _d, nbytes, nfiles in present:
        print(f"  + {RUN_PREFIX}/{n}  ({nbytes / 1e6:.1f} MB, {nfiles} files)")
    for n in missing:
        print(f"  ! MISSING at source: {n}")
    if args.dry_run:
        return 0
    if not present:
        print("Nothing to upload.")
        return 1

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(rid, repo_type=rtype, private=not args.public, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{rid}")

    for i, (n, d, nbytes, _nfiles) in enumerate(present, 1):
        print(f"[{i}/{len(present)}] uploading {RUN_PREFIX}/{n} ({nbytes / 1e6:.1f} MB) ...", flush=True)
        api.upload_folder(
            folder_path=str(d),
            path_in_repo=f"{RUN_PREFIX}/{n}",
            repo_id=rid,
            repo_type=rtype,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"Add paper run bundle: {n}",
        )
    print(f"Done. {len(present)} run bundle(s) uploaded to {rid} under {RUN_PREFIX}/.")
    if missing:
        print(f"WARNING: {len(missing)} runs missing at source: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
