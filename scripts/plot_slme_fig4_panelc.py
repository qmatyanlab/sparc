#!/usr/bin/env python3
"""Fig 4, panel (c): before/after for the SLME task, stacked.

  top : element condensation -- prior (first N steps) vs best-stage periodic tables
        (count/unweighted occurrence; white = absent; eff. # elements e^H annotated).
  bottom : space-group condensation -- top-N SGs, prior vs steered grouped bars.

Usage:
  python scripts/plot_slme_fig4_panelc.py <run_dir> [--window 10] [--top 12]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_distribution_shift  # noqa: F401  (applies publication.mplstyle)
from demo_element_migration_gif import build_layout, load_density
from plot_run import moving_average
from plot_eval_symmetry_inset import _hm
from pymatgen.core.periodic_table import Element

C_PRIOR = "#9ecae1"   # light blue = prior (first steps)
C_STEER = "#d62728"   # red = steered best stage


def _windows(run_dir, samples, w):
    all_steps = sorted(int(f.stem.split("_")[1]) for f in samples.glob("step_*_eval.pt"))
    early = all_steps[:w]
    rs, rew = [], []
    for r in csv.DictReader(open(run_dir / "metrics.csv")):
        try:
            rs.append(int(float(r["step"]))); rew.append(float(r["reward mean"]))
        except (ValueError, KeyError):
            pass
    peak = int(np.array(rs)[int(np.nanargmax(moving_average(np.array(rew), 5)))])
    lo = max(0, all_steps.index(min(all_steps, key=lambda s: abs(s - peak))) - w // 2)
    best = all_steps[lo:lo + w]
    return early, best


def _sg_counts(samples, steps):
    counts: Counter = Counter()
    for s in steps:
        p = samples / f"step_{int(s):04d}_eval.pt"
        if not p.exists():
            continue
        for it in torch.load(p, map_location="cpu"):
            if isinstance(it, dict) and it.get("spacegroup") is not None:
                counts[int(it["spacegroup"])] += 1
    return counts


def _eff_n(vec, present):
    p = np.array([vec[z] for z in present]); p = p / max(p.sum(), 1e-12)
    return float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum()))


def _periodic_table(ax, vec, cells, layout, vmin, vmax, cmap, max_row, title):
    def grid():
        g = np.full((10, 18), np.nan)
        for z in cells:
            r, c = layout[z]
            if vec[z] > 0:
                g[r, c] = vec[z]
        return g
    im = ax.imshow(grid(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    for z in cells:
        r, c = layout[z]
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="0.6", lw=0.5))
        norm = (vec[z] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        txt = "black" if (vec[z] <= 0 or norm > 0.6) else "white"
        ax.text(c, r, Element.from_Z(z).symbol, ha="center", va="center", fontsize=8.0, color=txt)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(max_row + 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9)
    return im


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--top", type=int, default=10, help="space groups in the bottom panel")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    samples = run_dir / "samples"
    early, best = _windows(run_dir, samples, args.window)
    early_lbl = f"Prior (first {len(early)} steps)"
    best_lbl = f"Best stage (steps {best[0]}–{best[-1]})"

    # --- element condensation ---
    steps_d, dens = load_density(run_dir, "occurrence", "none")
    steps_d = np.asarray(steps_d)
    e_idx = [i for i, s in enumerate(steps_d) if s in set(early)]
    b_idx = [i for i, s in enumerate(steps_d) if s in set(best)]
    # as a percentage of structures (consistent with the space-group panel below)
    e_vec, b_vec = dens[e_idx].mean(0) * 100.0, dens[b_idx].mean(0) * 100.0
    layout = build_layout()
    present = [z for z in range(1, 119) if z in layout and dens[:, z].max() > 0]
    fblock = set(range(57, 72)) | set(range(89, 104))
    cells = [z for z in layout if z not in fblock or any(z in fblock for z in present)]
    max_row = max(layout[z][0] for z in cells)
    pos = np.concatenate([e_vec[present], b_vec[present]]); pos = pos[pos > 0]
    vmin, vmax = float(pos.min()), float(pos.max())
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")

    # --- space-group condensation ---
    ce, cb = _sg_counts(samples, early), _sg_counts(samples, best)
    ne, nb = sum(ce.values()) or 1, sum(cb.values()) or 1
    sgs = sorted(set(ce) | set(cb), key=lambda s: (cb[s] / nb, ce[s] / ne), reverse=True)[:args.top]

    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8], hspace=0.28, wspace=0.06)
    axL, axR = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axSG = fig.add_subplot(gs[1, :])

    _periodic_table(axL, e_vec, cells, layout, vmin, vmax, cmap, max_row,
                    f"{early_lbl}\n$e^H$ = {_eff_n(e_vec, present):.1f} elements")
    im = _periodic_table(axR, b_vec, cells, layout, vmin, vmax, cmap, max_row,
                         f"{best_lbl}\n$e^H$ = {_eff_n(b_vec, present):.1f} elements")
    cbar = fig.colorbar(im, ax=[axL, axR], fraction=0.025, pad=0.02)
    cbar.set_label("Fraction of structures (%)")

    x = np.arange(len(sgs))
    axSG.bar(x - 0.2, [100 * ce[s] / ne for s in sgs], width=0.4, color=C_PRIOR,
             label=f"{early_lbl} (N={ne})")
    axSG.bar(x + 0.2, [100 * cb[s] / nb for s in sgs], width=0.4, color=C_STEER,
             label=f"{best_lbl} (N={nb})")
    axSG.set_xticks(x); axSG.set_xticklabels([_hm(s) for s in sgs], rotation=35, ha="right", fontsize=8)
    axSG.set_ylabel("Fraction of structures (%)")
    axSG.set_title(f"Space-group condensation (top {len(sgs)})", fontsize=9)
    axSG.legend(fontsize=8, loc="upper right")

    out = args.output or (run_dir / "deliverables" / "slme_fig4_panelc.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} / .pdf")
    print(f"  elements e^H: {_eff_n(e_vec, present):.1f} -> {_eff_n(b_vec, present):.1f}")
    print("  top SG steered: " + ", ".join(f"{s}:{100*cb[s]/nb:.0f}%" for s in sgs[:6]))


if __name__ == "__main__":
    main()
