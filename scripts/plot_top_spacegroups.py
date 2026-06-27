#!/usr/bin/env python3
"""Top-N space groups as a percentage, ranked (publication style).

A cleaner alternative to the full eval/valid symmetry-distribution bar charts:
ranks the space groups by frequency, shows the top N as % of the scored set, and
labels each with its Hermann-Mauguin symbol -- no per-SG colour key needed.

Usage:
  python scripts/plot_top_spacegroups.py \
      exp_res/tsenn_static_dielectric_layered_uniaxial_symmcd_v3_uniform_anchor_54304128 \
      [--top 15]
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

BAR_COLOR = "#1fb312"  # style green (= SymmCD in the paper palette)


def _hm(n: int) -> str:
    """SG number -> 'N  $\\mathrm{symbol}$' with bar over rotoinversions."""
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
    return r"%d  $\mathrm{%s}$" % (n, "".join(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp_dir", type=Path)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _panel(ax, counts, top, title):
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
    sgs = [sg for sg, _ in ranked]
    pct = [100.0 * c / total for _, c in ranked]
    y = np.arange(len(sgs))[::-1]  # highest at the top
    ax.barh(y, pct, color=BAR_COLOR, edgecolor="white", linewidth=0.4)
    for yi, p in zip(y, pct):
        ax.annotate(f"{p:.0f}%", (p, yi), xytext=(2, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([_hm(s) for s in sgs])
    ax.set_xlabel("% of scored structures")
    ax.set_title(f"{title}  (N={total})")
    ax.set_xlim(0, max(pct) * 1.18)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    args = parse_args()
    d = args.exp_dir.resolve()
    samples = d / "samples"
    valid = load_symmetry_counts(samples, "valid")
    eval_ = load_symmetry_counts(samples, "eval")

    fig, (axv, axe) = plt.subplots(1, 2, figsize=(7.4, 4.2), constrained_layout=True)
    _panel(axv, valid, args.top, f"Valid: top {args.top} space groups")
    _panel(axe, eval_, args.top, f"Evaluated: top {args.top} space groups")

    out = args.output or (d / "deliverables_layered_uniaxial" / "top_spacegroups.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".eps"):
        fig.savefig(out.with_suffix(ext))
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} / .pdf / .eps")
    for name, c in [("valid", valid), ("eval", eval_)]:
        tot = sum(c.values())
        top = sorted(c.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
        print(f"  {name} (N={tot}): " + ", ".join(f"{sg}:{100*n/tot:.0f}%" for sg, n in top))


if __name__ == "__main__":
    main()
