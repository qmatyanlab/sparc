#!/usr/bin/env python3
"""Fig-5 candidate (A): the symmetry -> property mechanism for the layered-uniaxial
dielectric reward.

Two panels make the causal chain visible:
  (a) crystal symmetry CAPS achievable anisotropy -- cubic tensors are isotropic
      (eps_max/eps_min ~ 1), only lower-symmetry systems can be uniaxial;
  (b) reward rises with that anisotropy -- so cubic is structurally locked out of
      high reward (it can't express the eps_zz != eps_xx the reward selects for).

Usage:
  python scripts/plot_symmetry_property_link.py <run_dir> [--eps-max 100]
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
from plot_layered_uniaxial_samples import load_step_records

# crystal systems in ascending symmetry, each with a fixed colour
CSYS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
        "trigonal", "hexagonal", "cubic"]
CMAP = dict(zip(CSYS, ["#777777", "#8c564b", "#1f77b4", "#2ca02c",
                       "#9467bd", "#ff7f0e", "#d62728"]))


def _crystal_system(sg: int) -> str:
    if sg <= 2:   return "triclinic"
    if sg <= 15:  return "monoclinic"
    if sg <= 74:  return "orthorhombic"
    if sg <= 142: return "tetragonal"
    if sg <= 167: return "trigonal"
    if sg <= 194: return "hexagonal"
    return "cubic"


def _anisotropy(row) -> float:
    """eps_max / eps_min of the symmetric predicted tensor (1.0 == isotropic)."""
    xx, yy, zz = row["eps_xx"], row["eps_yy"], row["eps_zz"]
    xy, xz, yz = row["eps_xy"], row["eps_xz"], row["eps_yz"]
    M = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)
    ev = np.linalg.eigvalsh(M)
    return float(ev[-1] / max(ev[0], 1e-6))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--eps-max", type=float, default=100.0)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    recs = [r for r in load_step_records(run_dir, requested_step=None)
            if bool(r["positive_definite"])
            and max(r["eps_xx"], r["eps_yy"], r["eps_zz"]) <= args.eps_max]
    for r in recs:
        r["_sys"] = _crystal_system(int(r["spacegroup"]))
        r["_aniso"] = _anisotropy(r)

    present = [s for s in CSYS if any(r["_sys"] == s for r in recs)]
    pos = {s: i for i, s in enumerate(present)}

    fig, (axA, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)

    def _violin(ax, valuefn, ylabel, logy=False):
        data = [[valuefn(r) for r in recs if r["_sys"] == s] for s in present]
        parts = ax.violinplot(data, positions=range(len(present)), widths=0.8,
                              showmeans=False, showextrema=False)
        for body, s in zip(parts["bodies"], present):
            body.set_facecolor(CMAP[s]); body.set_alpha(0.45)
            body.set_edgecolor(CMAP[s]); body.set_linewidth(0.8)
        # median bars + jittered points
        rng = np.random.default_rng(0)
        for i, (s, d) in enumerate(zip(present, data)):
            if not d:
                continue
            ax.hlines(np.median(d), i - 0.34, i + 0.34, color=CMAP[s], lw=1.8, zorder=4)
            x = i + (rng.random(len(d)) - 0.5) * 0.28
            ax.scatter(x, d, s=4, color=CMAP[s], alpha=0.35, edgecolors="none", zorder=3)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")

    # (a) anisotropy by crystal system -- symmetry caps it
    _violin(axA, lambda r: r["_aniso"],
            r"Dielectric anisotropy $\varepsilon_{\max}/\varepsilon_{\min}$", logy=True)
    axA.axhline(1.0, ls="--", color="0.4", lw=1.0, zorder=1)
    axA.text(len(present) - 0.5, 1.02, "isotropic", color="0.4", fontsize=7,
             ha="right", va="bottom")
    axA.set_title("Symmetry caps achievable anisotropy")

    # (b) reward vs anisotropy, coloured by system -- reward tracks anisotropy
    for s in present:
        xs = [r["_aniso"] for r in recs if r["_sys"] == s]
        ys = [r["reward"] for r in recs if r["_sys"] == s]
        ax2.scatter(xs, ys, s=9, color=CMAP[s], alpha=0.55, edgecolors="none", label=s)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Dielectric anisotropy $\varepsilon_{\max}/\varepsilon_{\min}$")
    ax2.set_ylabel("Reward")
    ax2.set_title("Reward tracks anisotropy")
    ax2.legend(loc="upper left", fontsize=6.5, handletextpad=0.2, labelspacing=0.2,
               borderpad=0.2)

    out = args.output or (run_dir / "deliverables_layered_uniaxial"
                          / "symmetry_property_link.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext))
    plt.close(fig)
    n_cub = sum(1 for r in recs if r["_sys"] == "cubic")
    print(f"Wrote {out.with_suffix('.png')} / .pdf  (N={len(recs)}, cubic={n_cub})")
    for s in present:
        d = [r for r in recs if r["_sys"] == s]
        rew = np.array([r["reward"] for r in d]); an = np.array([r["_aniso"] for r in d])
        print(f"  {s:13s} n={len(d):4d}  reward med={np.median(rew):.2f}  aniso med={np.median(an):.2f}")


if __name__ == "__main__":
    main()
