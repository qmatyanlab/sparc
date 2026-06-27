#!/usr/bin/env python3
"""Fig-5 candidate (C): novelty, uniqueness and diversity over RL steps.

Answers the reflexive generative-model question -- "is it just memorising MP-20 /
collapsing to one structure?" -- with metrics.csv columns:
  (a) fraction of generated structures that are novel / unique / novel+unique+stable;
  (b) batch diversity (div_ratio) and the stable fraction, vs RL step.

Usage:
  python scripts/plot_novelty_diversity.py <run_dir>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_distribution_shift  # noqa: F401  (applies publication.mplstyle)
from plot_run import moving_average


def _cols(run_dir: Path, names):
    rows = list(csv.DictReader(open(run_dir / "metrics.csv")))

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(name, "")))
            except ValueError:
                out.append(np.nan)
        return np.array(out)

    step = col("step")
    if np.all(np.isnan(step)):
        step = np.arange(len(rows), dtype=float)
    return step, {n: col(n) for n in names}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--smooth", type=int, default=5, help="moving-average window")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    names = ["frac_novel_structures", "frac_unique_structures",
             "frac_novel_unique_structures", "frac_novel_unique_stable_structures",
             "frac_stable_structures", "div_ratio"]
    step, c = _cols(run_dir, names)
    w = max(1, args.smooth)

    def ma(y):
        m = np.isfinite(y)
        return step[m], moving_average(y[m], w)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)

    series_a = [
        ("frac_novel_structures", "Novel", "#000dfc"),
        ("frac_unique_structures", "Unique", "#1fb312"),
        ("frac_novel_unique_structures", "Novel & unique", "#ff7f0e"),
        ("frac_novel_unique_stable_structures", "Novel, unique & stable", "#d62728"),
    ]
    for key, lab, col in series_a:
        if np.isfinite(c[key]).any():
            x, y = ma(c[key])
            axA.plot(x, 100.0 * y, color=col, lw=1.8, label=lab)
    axA.set_xlabel("RL step"); axA.set_ylabel("Fraction of generated (%)")
    axA.set_ylim(0, 100)
    axA.set_title("Not memorising: novelty & uniqueness")
    axA.legend(loc="lower left", fontsize=7)

    # (b) diversity + stability
    if np.isfinite(c["div_ratio"]).any():
        x, y = ma(c["div_ratio"]); axB.plot(x, y, color="#9467bd", lw=1.8,
                                            label="Diversity (div_ratio)")
    axB.set_xlabel("RL step"); axB.set_ylabel("Diversity ratio")
    axB.set_title("Stays diverse while improving stability")
    axB.legend(loc="upper left", fontsize=7)
    axB2 = axB.twinx()
    if np.isfinite(c["frac_stable_structures"]).any():
        x, y = ma(c["frac_stable_structures"])
        axB2.plot(x, 100.0 * y, color="#1f6fb4", lw=1.8, ls="--",
                  label="Stable fraction")
    axB2.set_ylabel("Stable fraction (%)")
    axB2.legend(loc="lower right", fontsize=7)

    out = args.output or (run_dir / "deliverables" / "novelty_diversity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext))
    plt.close(fig)
    fin = lambda k: (c[k][np.isfinite(c[k])][-1] if np.isfinite(c[k]).any() else float("nan"))
    print(f"Wrote {out.with_suffix('.png')} / .pdf")
    print(f"  final: novel={100*fin('frac_novel_structures'):.0f}%  "
          f"unique={100*fin('frac_unique_structures'):.0f}%  "
          f"novel&unique&stable={100*fin('frac_novel_unique_stable_structures'):.0f}%  "
          f"div_ratio={fin('div_ratio'):.2f}")


if __name__ == "__main__":
    main()
