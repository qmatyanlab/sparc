#!/usr/bin/env python3
"""Visualize that DiffCSP (continuous, unconstrained) cannot produce genuine
high-symmetry crystals while SymmCD (symmetry-constrained) does — even though
both reach comparable reward.

Three panels:
  (a) reward mean vs step              — both models score similarly
  (b) symmetry-order distribution      — # symmetry operations per scored
      (at strict symprec)                structure; SymmCD spreads to high order,
                                         DiffCSP pinned at 1 (P1)
  (c) P1 fraction: loose vs strict     — DiffCSP's apparent symmetry is a
      symprec                            tolerance artifact; SymmCD is invariant

Example:
    python scripts/plot_symmetry_architecture.py \
        exp_res/..._symmcd_v1_carryover_53154012 \
        exp_res/..._diffcsp_v1_54289718 --labels SymmCD DiffCSP
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

LOOSE, STRICT = 0.1, 0.01
LAST_STEPS = 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", type=Path, nargs=2)
    p.add_argument("--labels", nargs=2, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--last-steps", type=int, default=LAST_STEPS)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def _analyze(path: Path):
    """Per scored structure: (n_ops@strict, sg@loose, sg@strict)."""
    atoms_list = ase_read(path, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    out = []
    for atoms in atoms_list:
        s = AseAtomsAdaptor.get_structure(atoms)
        rec = {"n_ops": None, "sg_loose": None, "sg_strict": None}
        try:
            ds = SpacegroupAnalyzer(s, symprec=STRICT).get_symmetry_dataset()
            rec["n_ops"] = int(len(ds.rotations))
            rec["sg_strict"] = int(ds.number)
        except Exception:
            pass
        try:
            rec["sg_loose"] = int(
                SpacegroupAnalyzer(s, symprec=LOOSE).get_space_group_number())
        except Exception:
            pass
        out.append(rec)
    return out


def load_run(run_dir: Path, last_steps: int, workers: int):
    paths = sorted((run_dir / "samples").glob("step_*_eval.extxyz"))[-last_steps:]
    with Pool(workers) as pool:
        recs = [r for chunk in pool.map(_analyze, paths) for r in chunk]
    return recs


def load_reward(run_dir: Path):
    steps, rew = [], []
    with (run_dir / "metrics.csv").open() as fh:
        for r in csv.DictReader(fh):
            try:
                steps.append(int(float(r["step"]))); rew.append(float(r["reward mean"]))
            except (KeyError, ValueError):
                pass
    return steps, rew


def main() -> None:
    args = parse_args()
    labels = args.labels or [d.name for d in args.run_dirs]
    out = args.output_dir or (args.run_dirs[1] / "deliverables" / "symmetry_architecture")
    out.mkdir(parents=True, exist_ok=True)
    colors = {labels[0]: "tab:blue", labels[1]: "tab:red"}

    data = {}
    for lab, d in zip(labels, args.run_dirs):
        print(f"[{lab}] analyzing symmetry (strict symprec={STRICT}) ...")
        data[lab] = {"recs": load_run(d, args.last_steps, args.workers),
                     "reward": load_reward(d)}

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # (a) reward vs step
    for lab in labels:
        st, rw = data[lab]["reward"]
        axes[0].plot(st, rw, color=colors[lab], alpha=0.85, label=lab)
    axes[0].set_xlabel("RL step"); axes[0].set_ylabel("reward mean")
    axes[0].set_title("(a) both models reach comparable reward")
    axes[0].legend()

    # (b) symmetry-order distribution at strict symprec
    # discrete operation counts -> bar by tier
    tiers = [1, 2, 4, 8, 12, 16, 24, 48, 96, 192]
    def tier_idx(n):
        # bucket n_ops into the nearest tier at or below
        idx = 0
        for i, t in enumerate(tiers):
            if n >= t:
                idx = i
        return idx
    width = 0.4
    x = np.arange(len(tiers))
    for k, lab in enumerate(labels):
        nops = [r["n_ops"] for r in data[lab]["recs"] if r["n_ops"]]
        counts = Counter(tier_idx(n) for n in nops)
        frac = np.array([counts.get(i, 0) for i in range(len(tiers))]) / max(1, len(nops))
        axes[1].bar(x + (k - 0.5) * width, frac, width, color=colors[lab],
                    label=f"{lab} (median {int(np.median(nops))} ops)")
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(t) for t in tiers])
    axes[1].set_xlabel("# symmetry operations  (P1 = 1  →  high symmetry →)")
    axes[1].set_ylabel("fraction of scored structures")
    axes[1].set_title(f"(b) realized symmetry order @ symprec {STRICT}")
    axes[1].legend(fontsize=9)

    # (c) P1 fraction loose vs strict
    grp = np.arange(2)
    for k, lab in enumerate(labels):
        recs = data[lab]["recs"]
        p1_loose = np.mean([r["sg_loose"] == 1 for r in recs if r["sg_loose"]])
        p1_strict = np.mean([r["sg_strict"] == 1 for r in recs if r["sg_strict"]])
        bars = axes[2].bar(grp + (k - 0.5) * width, [p1_loose, p1_strict], width,
                           color=colors[lab], label=lab)
        for b, v in zip(bars, [p1_loose, p1_strict]):
            axes[2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                         ha="center", fontsize=9)
    axes[2].set_xticks(grp)
    axes[2].set_xticklabels([f"loose\n(symprec {LOOSE})", f"strict\n(symprec {STRICT})"])
    axes[2].set_ylabel("fraction labeled P1 (triclinic)")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("(c) DiffCSP's symmetry is a tolerance artifact")
    axes[2].legend()

    fig.suptitle("Symmetry is architectural: SymmCD constructs it; DiffCSP only approximates it",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "symmetry_architecture.png", dpi=150)

    # console summary
    for lab in labels:
        nops = [r["n_ops"] for r in data[lab]["recs"] if r["n_ops"]]
        hi = np.mean([n >= 8 for n in nops])
        print(f"[{lab}] n={len(nops)} median_ops={int(np.median(nops))} "
              f"frac(≥8 ops)={hi:.1%} max_ops={max(nops)}")
    print(f"saved {out / 'symmetry_architecture.png'}")


if __name__ == "__main__":
    main()
