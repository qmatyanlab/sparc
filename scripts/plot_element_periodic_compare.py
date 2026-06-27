#!/usr/bin/env python3
"""Fig-5 candidate (B, periodic-table form): species condensation as two periodic
tables -- EARLY (first N steps) vs LATE/BEST stage -- coloured by element occurrence
(reward-weighted by default, matching element_migration_rewardwt.gif).

Shows the condensation as a spatial collapse onto a region of the periodic table,
with the effective number of elements exp(H) annotated on each panel.

Usage:
  python scripts/plot_element_periodic_compare.py <run_dir> \
      [--weight reward|eta|none] [--window 10] [--late last|best]
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
from matplotlib.patches import Rectangle

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_distribution_shift  # noqa: F401  (applies publication.mplstyle)
from demo_element_migration_gif import build_layout, load_density
from plot_run import moving_average
from pymatgen.core.periodic_table import Element


def _eff_n_elements(vec, present):
    p = np.array([vec[z] for z in present], dtype=float)
    p = p / max(p.sum(), 1e-12)
    H = -(p * np.log(np.clip(p, 1e-12, None))).sum()
    return float(np.exp(H))


def _reward_by_step(run_dir):
    out = {}
    path = run_dir / "metrics.csv"
    if path.exists():
        for r in csv.DictReader(open(path)):
            try:
                out[int(float(r["step"]))] = float(r["reward mean"])
            except (ValueError, KeyError):
                pass
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--weight", default="reward", choices=["reward", "eta", "none"])
    p.add_argument("--window", type=int, default=10, help="steps averaged per panel")
    p.add_argument("--late", default="best", choices=["last", "best"],
                   help="late panel = final `window` steps, or window around peak reward")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    steps, dens = load_density(run_dir, "occurrence", args.weight)
    steps = np.asarray(steps)
    w = args.window

    early_idx = np.arange(min(w, len(steps)))
    if args.late == "best":
        rw = _reward_by_step(run_dir)
        rvec = np.array([rw.get(int(s), np.nan) for s in steps])
        ma = moving_average(rvec, 5)
        c = int(np.nanargmax(ma))
        lo = max(0, c - w // 2)
        late_idx = np.arange(lo, min(len(steps), lo + w))
        late_lbl = f"Best stage (steps {int(steps[late_idx[0]])}–{int(steps[late_idx[-1]])})"
    else:
        late_idx = np.arange(max(0, len(steps) - w), len(steps))
        late_lbl = f"Last {len(late_idx)} steps"
    early_lbl = f"First {len(early_idx)} steps"

    layout = build_layout()
    present = [z for z in range(1, 119) if z in layout and dens[:, z].max() > 0]
    # draw the FULL periodic table (every real element is a cell, white if absent);
    # only include the f-block rows if a lanthanide/actinide actually appears
    fblock = set(range(57, 72)) | set(range(89, 104))
    fblock_present = any(z in fblock for z in present)
    cells = [z for z in layout if z not in fblock or fblock_present]
    early_vec = dens[early_idx].mean(0)
    late_vec = dens[late_idx].mean(0)
    # colour scale spans only the POSITIVE occurrences (zero == white, off-scale),
    # so the colourbar starts at the smallest real occurrence rather than 0.00
    posvals = np.concatenate([early_vec[present], late_vec[present]])
    posvals = posvals[posvals > 0]
    vmin, vmax = float(posvals.min()), float(posvals.max())

    def grid(vec):
        g = np.full((10, 18), np.nan)        # zero / absent -> NaN -> white
        for z in cells:
            r, c = layout[z]
            if vec[z] > 0:
                g[r, c] = vec[z]
        return g

    # white-at-zero colormap: empty / zero-occurrence cells are white, colour grows
    # with occurrence (matches the reward-weighted GIF's white-background look)
    cmap = plt.get_cmap("viridis").copy()   # colour grows with occurrence
    cmap.set_bad(color="white")             # zero / absent cells render white
    max_row = max(layout[z][0] for z in cells)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), constrained_layout=True)
    for ax, vec, lbl in ((axes[0], early_vec, early_lbl), (axes[1], late_vec, late_lbl)):
        im = ax.imshow(grid(vec), cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        for z in cells:
            r, c = layout[z]
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="0.6", lw=0.5))
            # white text on the dark (low/mid) viridis cells, black on white/bright cells
            val = vec[z]
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            txt = "black" if (val <= 0 or norm > 0.6) else "white"
            ax.text(c, r, Element.from_Z(z).symbol, ha="center", va="center",
                    fontsize=6.5, color=txt)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(max_row + 0.5, -0.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{lbl}\neff. # elements $e^H$ = {_eff_n_elements(vec, present):.1f}",
                     fontsize=9)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Occurrence" + ("" if args.weight == "none" else f" ({args.weight}-weighted)"))

    out = args.output or (run_dir / "deliverables"
                          / f"element_periodic_compare_{args.weight}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} / .pdf")
    for lbl, vec in ((early_lbl, early_vec), (late_lbl, late_vec)):
        top = sorted(present, key=lambda z: vec[z], reverse=True)[:8]
        print(f"  {lbl}: eff#={_eff_n_elements(vec, present):.1f}  top: "
              + ", ".join(f"{Element.from_Z(z).symbol}:{vec[z]*100:.0f}%" for z in top))


if __name__ == "__main__":
    main()
