#!/usr/bin/env python3
"""Evaluated-set space-group distribution (fraction of structures, single colour,
no per-SG colour key) paired with a ranked top-N panel -- either layout:

  --layout full-main : full SG distribution is the mother, top-N is the inset
  --layout top-main  : ranked top-N is the mother, full distribution is the inset

Usage:
  python scripts/plot_eval_symmetry_inset.py <run_dir> [--top 15] [--layout ...]
"""
from __future__ import annotations

import argparse
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
from generate_layered_uniaxial_deliverables import load_symmetry_counts
from pymatgen.symmetry.groups import SpaceGroup

BAR_COLOR = "#000dfc"  # style blue (publication.mplstyle prop_cycle)

# Font sizes, standardized: FS_BASE inherits publication.mplstyle base (main panel)
# so it tracks the global style (currently 12); the inset runs ~3 pt smaller; value
# labels on bars one notch smaller than their panel; inset title +1.
FS_BASE = plt.rcParams["font.size"]
FS_INSET = FS_BASE - 3
FS_INSET_TITLE = FS_INSET + 1


def _hm_sym(n: int) -> str:
    """Just the Hermann-Mauguin symbol as mathtext, e.g. r"$\\mathrm{P\\bar{6}m2}$"."""
    try:
        sym = SpaceGroup.from_int_number(n).symbol
    except Exception:  # noqa: BLE001
        return str(n)
    out, i = [], 0
    while i < len(sym):
        if sym[i] == "-" and i + 1 < len(sym) and sym[i + 1].isdigit():
            out.append(r"\bar{%s}" % sym[i + 1]); i += 2
        elif sym[i] == "_" and i + 1 < len(sym) and sym[i + 1].isdigit():
            out.append(r"_{%s}" % sym[i + 1]); i += 2
        else:
            out.append(sym[i]); i += 1
    return r"$\mathrm{%s}$" % "".join(out)


def _hm(n: int) -> str:
    return r"%d  %s" % (n, _hm_sym(n))


def _draw_full(ax, sgs, frac, *, small=False, fs=None):
    """Full SG distribution: fraction (%) vs space-group number.
    `fs` overrides the axis-label/tick font size (default: FS_INSET/FS_BASE)."""
    ax.bar(sgs, frac, width=1.6, color=BAR_COLOR, linewidth=0)
    ax.set_xlim(0, 231)
    ax.set_ylim(0, max(20.0, frac.max() * 1.10))
    ax.set_yticks([0, 5, 10, 15, 20])
    fs = (FS_INSET if small else FS_BASE) if fs is None else fs
    ax.set_xticks([1, 63, 125, 187, 229] if small
                  else [1, 21, 42, 63, 83, 104, 125, 146, 166, 187, 208, 229])
    ax.tick_params(labelsize=fs, length=0)
    ax.set_xlabel("Space group", fontsize=fs)
    ax.set_ylabel("Fraction of structures (%)", fontsize=fs)


def _draw_top(ax, ranked, total, *, small=False, split_labels=False,
              axis_fs=None, tick_fs=None):
    """Ranked top-N space groups, horizontal bars, % of structures.

    `split_labels` draws the SG number and Hermann-Mauguin symbol as two separate
    left-aligned columns (a clean tabular look) instead of one combined tick label.
    `axis_fs`/`tick_fs` override the axis-label/tick font sizes (default: both `fs`, so
    existing callers are unchanged)."""
    top_sgs = [s for s, _ in ranked]
    top_pct = [100.0 * c / total for _, c in ranked]
    y = np.arange(len(top_sgs))[::-1]
    ax.barh(y, top_pct, color=BAR_COLOR, edgecolor="white", linewidth=0.3)
    fs = FS_INSET if small else FS_BASE
    ann_fs = (fs - 1) if tick_fs is None else tick_fs   # bar value labels
    axis_fs = fs if axis_fs is None else axis_fs
    tick_fs = fs if tick_fs is None else tick_fs
    for yi, p in zip(y, top_pct):
        ax.annotate(f"{p:.0f}", (p, yi), xytext=(1.5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=ann_fs)
    ax.set_yticks(y)
    if split_labels:
        # two left-aligned columns: numbers align at NUM_X, symbols align at SYM_X
        # (x = axes fraction, y = data) -- robust to the proportional sans-serif font
        ax.set_yticklabels([])
        trans = ax.get_yaxis_transform()
        # compact two-column block, shifted toward the bars to close the left-hand gap;
        # small blank between the number and symbol columns
        NUM_X, SYM_X = -0.150, -0.100
        for yi, s in zip(y, top_sgs):
            ax.text(NUM_X, yi, str(s), transform=trans, ha="left", va="center",
                    fontsize=tick_fs, clip_on=False)
            ax.text(SYM_X, yi, _hm_sym(s), transform=trans, ha="left", va="center",
                    fontsize=tick_fs, clip_on=False)
        # rotated y-title just left of the number column (manual text so it tracks
        # the columns' axes-fraction x rather than the now-empty tick labels)
        ax.text(NUM_X - 0.040, 0.5, "Space group", transform=ax.transAxes,
                rotation=90, ha="center", va="center", fontsize=axis_fs, clip_on=False)
    else:
        ax.set_yticklabels([_hm(s) for s in top_sgs], fontsize=tick_fs)
        ax.set_ylabel("Space group", fontsize=axis_fs)
    ax.set_xlim(0, max(top_pct) * 1.18)
    if not small:
        ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(axis="both", length=0, labelsize=tick_fs)
    ax.set_xlabel("Fraction of structures (%)", fontsize=axis_fs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp_dir", type=Path)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--artifact", default="eval", choices=["eval", "valid"])
    p.add_argument("--layout", default="full-main", choices=["full-main", "top-main"])
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = args.exp_dir.resolve()
    counts = load_symmetry_counts(d / "samples", args.artifact)
    total = sum(counts.values())
    sgs = np.array(sorted(counts))
    frac = np.array([100.0 * counts[s] / total for s in sgs])
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
    cap = args.artifact.capitalize()

    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)

    if args.layout == "full-main":
        _draw_full(ax, sgs, frac)
        ins = ax.inset_axes([0.12, 0.30, 0.52, 0.56])
        _draw_top(ins, ranked, total, small=True)
        default_name = f"{args.artifact}_symmetry_distribution_frac.png"
    else:  # top-main
        _draw_top(ax, ranked, total)
        ax.set_title(f"Top {args.top} space groups")
        ins = ax.inset_axes([0.46, 0.12, 0.50, 0.46])
        _draw_full(ins, sgs, frac, small=True)
        ins.set_title("Overall distribution", fontsize=FS_INSET_TITLE)
        default_name = f"{args.artifact}_top{args.top}_spacegroups.png"

    out = args.output or (d / "deliverables_layered_uniaxial" / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".eps"):
        fig.savefig(out.with_suffix(ext))
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} / .pdf / .eps  (N={total})")
    print("  top: " + ", ".join(f"{s}:{100*c/total:.0f}%" for s, c in ranked))


if __name__ == "__main__":
    main()
