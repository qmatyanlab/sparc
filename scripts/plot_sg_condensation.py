#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# International-tables SG-number ranges per crystal system.
SYSTEMS = {
    "triclinic": (1, 2),
    "monoclinic": (3, 15),
    "orthorhombic": (16, 74),
    "tetragonal": (75, 142),
    "trigonal": (143, 167),
    "hexagonal": (168, 194),
    "cubic": (195, 230),
}
UNIAXIAL = ("tetragonal", "trigonal", "hexagonal")  # optically uniaxial: eps_xx=eps_yy != eps_zz
# colour-blind-safe; uniaxial family in warm hues, everything else muted/cool
SYS_COLOR = {
    "triclinic": "#9aa0a6", "monoclinic": "#5f6b7a", "orthorhombic": "#7b8794",
    "tetragonal": "#e8853a", "trigonal": "#d1495b", "hexagonal": "#f4c430",
    "cubic": "#4c78a8",
}


def _load(run_dir: str):
    files = sorted(
        glob.glob(os.path.join(run_dir, "adaptive_spacegroup", "step_*.csv")),
        key=lambda p: int(re.search(r"step_(\d+)", p).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"no adaptive_spacegroup/step_*.csv under {run_dir}")
    steps, ent, sysmass = [], [], {k: [] for k in SYSTEMS}
    for f in files:
        step = int(re.search(r"step_(\d+)", f).group(1))
        df = pd.read_csv(f)
        p = df.set_index("spacegroup")["current_prob"].reindex(range(1, 231)).fillna(0.0).to_numpy()
        s = p.sum()
        if s <= 0:
            continue
        p = p / s
        nz = p[p > 0]
        steps.append(step)
        ent.append(float(-(nz * np.log(nz)).sum()))
        for name, (lo, hi) in SYSTEMS.items():
            sysmass[name].append(float(p[lo - 1:hi].sum()))
    order = np.argsort(steps)
    steps = np.asarray(steps)[order]
    ent = np.asarray(ent)[order]
    sysmass = {k: np.asarray(v)[order] for k, v in sysmass.items()}
    uni = sum(sysmass[k] for k in UNIAXIAL)
    return steps, ent, sysmass, uni


def _summary(label, steps, ent, uni):
    print(f"[{label}] steps {steps[0]}..{steps[-1]} ({len(steps)} pts)")
    print(f"    entropy   {ent[0]:.3f} -> {ent[-1]:.3f} nats")
    print(f"    uniaxial  {uni[0]*100:4.1f}% -> {uni[-1]*100:4.1f}% of SG proposal mass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--compare", default=None, help="second run_dir to overlay")
    ap.add_argument("--labels", default=None, help="comma-separated labels for the two runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [args.run_dir] + ([args.compare] if args.compare else [])
    labels = (args.labels.split(",") if args.labels
              else [os.path.basename(r.rstrip("/"))[:24] for r in runs])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    for ri, (run, lab) in enumerate(zip(runs, labels)):
        steps, ent, sysmass, uni = _load(run)
        _summary(lab, steps, ent, uni)
        ls = "-" if ri == 0 else "--"
        axA.plot(steps, ent, ls, color="#333", lw=2, label=lab)
        if len(runs) == 1:
            # single run: full crystal-system breakdown
            for name in SYSTEMS:
                axB.plot(steps, sysmass[name] * 100, lw=2.2 if name in UNIAXIAL else 1.2,
                         color=SYS_COLOR[name], alpha=0.95 if name in UNIAXIAL else 0.55,
                         label=name + (" *" if name in UNIAXIAL else ""))
        else:
            axB.plot(steps, uni * 100, ls, color="#d1495b", lw=2, label=lab)

    axA.axhline(np.log(169), color="#bbb", ls=":", lw=1)
    axA.text(steps[-1], np.log(169), " ~uniform (169 SGs)", va="bottom", ha="right", fontsize=8, color="#888")
    axA.set(xlabel="RL step", ylabel="SG proposal entropy (nats)", title="(a) Distribution entropy — lower = condensing")
    axA.legend(fontsize=8, frameon=False)

    axB.set(xlabel="RL step", ylabel="proposal probability mass (%)",
            title="(b) Uniaxial* mass" if len(runs) > 1 else "(b) Crystal-system mass (* = uniaxial)")
    axB.legend(fontsize=8, frameon=False, ncol=2)
    for ax in (axA, axB):
        ax.spines[["top", "right"]].set_visible(False)

    out = args.out or os.path.join(args.run_dir, "sg_condensation.png")
    fig.savefig(out, dpi=200)
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print("wrote", out, "and", os.path.splitext(out)[0] + ".pdf")


if __name__ == "__main__":
    main()
