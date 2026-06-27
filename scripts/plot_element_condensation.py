#!/usr/bin/env python3
"""Fig-5 candidate (B): species condensation over RL steps (static, GIF-free).

Element x step occurrence heatmap -- rows ordered by the step of peak prevalence so
the condensation reads as a funnel (transient species fade at the bottom, the
survivors persist as bright bands) -- with a Shannon-entropy strip on top that
quantifies how much the element distribution condensed.

Usage:
  python scripts/plot_element_condensation.py <run_dir> [--weight none|eta|reward]
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
from demo_element_migration_gif import load_density
from pymatgen.core.periodic_table import Element


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--weight", default="none", choices=["none", "eta", "reward"],
                   help="weight each structure equally (none) or by eta/reward")
    p.add_argument("--min-occ", type=float, default=0.03,
                   help="keep elements that reach this occurrence fraction at any step")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    steps, dens = load_density(run_dir, "occurrence", args.weight)  # dens: (nstep, 119)
    steps = np.asarray(steps, dtype=float)

    keep = [z for z in range(1, 119)
            if dens[:, z].max() >= args.min_occ]
    M = dens[:, keep].T                       # (n_elem, n_step)
    syms = [Element.from_Z(z).symbol for z in keep]
    order = np.argsort(M.argmax(axis=1))      # by step of peak prevalence
    M, syms = M[order], [syms[i] for i in order]

    p = dens[:, keep]
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.20, 1.0], hspace=0.07)
    axe = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=axe)

    axe.plot(steps, entropy, color="#000dfc", lw=1.6)
    axe.set_ylabel("Element\nentropy (nats)")
    axe.tick_params(labelbottom=False)
    axe.set_title("Species condensation over RL fine-tuning")

    im = axh.imshow(M, aspect="auto", origin="lower", cmap="magma",
                    extent=[steps.min(), steps.max(), 0, len(syms)],
                    vmin=0.0, vmax=float(M.max()))
    axh.set_yticks(np.arange(len(syms)) + 0.5)
    axh.set_yticklabels(syms, fontsize=6)
    axh.set_xlabel("RL step")
    axh.set_ylabel("Element (ordered by peak step)")
    cb = fig.colorbar(im, ax=[axe, axh], fraction=0.05, pad=0.02)
    cb.set_label("Occurrence (fraction of structures)")

    out = args.output or (run_dir / "deliverables" / "element_condensation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} / .pdf  "
          f"({len(syms)} elements, {len(steps)} steps, "
          f"entropy {entropy[0]:.2f}->{entropy[-1]:.2f} nats)")
    top_final = np.argsort(M[:, -1])[::-1][:8]
    print("  final-step top elements: "
          + ", ".join(f"{syms[i]}:{M[i,-1]*100:.0f}%" for i in top_final))


if __name__ == "__main__":
    main()
