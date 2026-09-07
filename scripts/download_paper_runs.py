#!/usr/bin/env python3
"""Download the paper RL run *visualization bundles* from HuggingFace.

Each bundle is a run directory without the large ``models/`` checkpoints (~40-65 MB/run) --
enough to regenerate every paper figure. Runs land in ``exp_res/<name>/`` (the location the
plotting scripts expect). Each bundle also ships the pre-rendered figures under
``deliverables*/`` (and ``figure4_bundle/`` for the dielectric run).

Usage:
  huggingface-cli login          # only if the repo is still private
  python scripts/download_paper_runs.py                 # all 5 runs
  python scripts/download_paper_runs.py --list          # show run names
  python scripts/download_paper_runs.py --runs <name>   # a subset
Then visualize, e.g.:
  python scripts/plot_dielectric_story_composite.py exp_res/<name>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.assets import PAPER_RUNS, download_paper_runs, repo_id  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="*", default=None, help="subset of run names (default: all)")
    ap.add_argument("--dest", default=None, help="destination dir (default: <repo>/exp_res)")
    ap.add_argument("--force", action="store_true", help="re-download even if the run dir already exists")
    ap.add_argument("--list", action="store_true", help="list available run bundles and exit")
    args = ap.parse_args()

    if args.list:
        print("Available paper run bundles:")
        for n in PAPER_RUNS:
            print(f"  {n}")
        return 0

    names = args.runs or list(PAPER_RUNS)
    print(f"Fetching {len(names)} run bundle(s) from {repo_id()} ...")
    results = download_paper_runs(names, dest=args.dest, force=args.force)
    for n in names:
        print(f"  {n} -> {results[n]}")
    print("\nDone. Regenerate a figure, e.g.:")
    print(f"  python scripts/plot_dielectric_story_composite.py {results[names[0]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
