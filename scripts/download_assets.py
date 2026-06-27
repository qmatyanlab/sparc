#!/usr/bin/env python3
"""Prefetch SPARC assets from HuggingFace.

Thin wrapper around ``utils.assets`` so assets can be warmed before submitting a job
(useful when compute nodes lack outbound internet — prefetch on the login/head node
which shares the filesystem).

Examples:
  python scripts/download_assets.py            # core + small files (~4.2 GB)
  python scripts/download_assets.py --all       # also the training caches (~13 GB total)
  python scripts/download_assets.py --groups core
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.assets import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
