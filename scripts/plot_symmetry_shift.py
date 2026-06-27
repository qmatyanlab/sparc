#!/usr/bin/env python3
"""Before/after symmetry distribution of an RL run: what kind of structures does the
reward favour?  Compares the prior (first N steps) with the steered best stage
(window around peak reward), as the fraction of structures in each crystal system
and each highest-rotational-axis class.

Usage:
  python scripts/plot_symmetry_shift.py <run_dir> [--window 10] [--late best|last]
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

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_distribution_shift  # noqa: F401  (applies publication.mplstyle)
from plot_run import moving_average
from plot_eval_symmetry_inset import _hm

C_PRIOR = "#9ecae1"   # light blue = prior (first steps)
C_STEER = "#d62728"   # red = steered best stage

SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
           "trigonal", "hexagonal", "cubic"]
FOLDS = ["1 (none)", "2-fold", "3-fold", "4-fold", "6-fold", "cubic"]


def _system(sg: int) -> str:
    if sg <= 2:   return "triclinic"
    if sg <= 15:  return "monoclinic"
    if sg <= 74:  return "orthorhombic"
    if sg <= 142: return "tetragonal"
    if sg <= 167: return "trigonal"
    if sg <= 194: return "hexagonal"
    return "cubic"


def _fold(sg: int) -> str:
    if sg <= 2:   return "1 (none)"
    if sg <= 74:  return "2-fold"
    if sg <= 142: return "4-fold"
    if sg <= 167: return "3-fold"
    if sg <= 194: return "6-fold"
    return "cubic"


def _sg_counts(samples_dir: Path, steps) -> Counter:
    counts: Counter = Counter()
    for s in steps:
        p = samples_dir / f"step_{int(s):04d}_eval.pt"
        if not p.exists():
            continue
        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, list):
            continue
        for it in payload:
            if isinstance(it, dict) and it.get("spacegroup") is not None:
                counts[int(it["spacegroup"])] += 1
    return counts


def _reward_steps(run_dir: Path):
    steps, rew = [], []
    path = run_dir / "metrics.csv"
    if path.exists():
        for r in csv.DictReader(open(path)):
            try:
                steps.append(int(float(r["step"]))); rew.append(float(r["reward mean"]))
            except (ValueError, KeyError):
                pass
    return np.array(steps), np.array(rew)


def _fracs(counts: Counter, keyfn, order):
    total = sum(counts.values()) or 1
    g: Counter = Counter()
    for sg, n in counts.items():
        g[keyfn(sg)] += n
    return np.array([100.0 * g.get(k, 0) / total for k in order]), total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--late", default="best", choices=["best", "last"])
    p.add_argument("--mode", default="sg", choices=["sg", "group"],
                   help="sg = top-N space groups before/after; group = crystal-system/fold")
    p.add_argument("--top", type=int, default=20, help="number of space groups (mode=sg)")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _plot_top_sg(run_dir, c_early, c_late, early_lbl, late_lbl, top, output):
    ne, nl = sum(c_early.values()) or 1, sum(c_late.values()) or 1
    # rank by steered fraction, then prior, so the reward-favoured SGs sit on top
    sgs = sorted(set(c_early) | set(c_late),
                 key=lambda s: (c_late[s] / nl, c_early[s] / ne), reverse=True)[:top]
    fe = [100.0 * c_early[s] / ne for s in sgs]
    fl = [100.0 * c_late[s] / nl for s in sgs]
    y = np.arange(len(sgs))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 7.0), constrained_layout=True)
    ax.barh(y + 0.2, fe, height=0.4, color=C_PRIOR, label=f"{early_lbl} (N={ne})")
    ax.barh(y - 0.2, fl, height=0.4, color=C_STEER, label=f"{late_lbl} (N={nl})")
    ax.set_yticks(y); ax.set_yticklabels([_hm(s) for s in sgs])
    ax.set_xlabel("Fraction of structures (%)")
    ax.set_title(f"Top {len(sgs)} space groups: prior vs steered")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right")
    for ext in (".png", ".pdf"):
        fig.savefig(output.with_suffix(ext))
    plt.close(fig)
    print(f"Wrote {output.with_suffix('.png')} / .pdf")
    print("  SG: prior% -> steered%")
    for s in sgs:
        print(f"   {s:4d} {_hm(s).split('$')[0].strip():>10s}: "
              f"{100*c_early[s]/ne:4.0f}% -> {100*c_late[s]/nl:4.0f}%")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    samples = run_dir / "samples"
    all_steps = sorted(int(f.stem.split("_")[1])
                       for f in samples.glob("step_*_eval.pt"))
    w = args.window
    early = all_steps[:w]
    if args.late == "best":
        rs, rew = _reward_steps(run_dir)
        ma = moving_average(rew, 5)
        peak = int(rs[int(np.nanargmax(ma))])
        lo = max(0, all_steps.index(min(all_steps, key=lambda s: abs(s - peak))) - w // 2)
        late = all_steps[lo:lo + w]
        late_lbl = f"Best stage (steps {late[0]}–{late[-1]})"
    else:
        late = all_steps[-w:]
        late_lbl = f"Last {len(late)} steps"
    early_lbl = f"Prior (first {len(early)} steps)"

    c_early = _sg_counts(samples, early)
    c_late = _sg_counts(samples, late)

    if args.mode == "sg":
        out = args.output or (run_dir / "deliverables" / f"symmetry_shift_top{args.top}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        _plot_top_sg(run_dir, c_early, c_late, early_lbl, late_lbl, args.top, out)
        return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for ax, keyfn, order, title in (
            (axA, _system, SYSTEMS, "Crystal system"),
            (axB, _fold, FOLDS, "Highest rotational axis")):
        fe, ne = _fracs(c_early, keyfn, order)
        fl, nl = _fracs(c_late, keyfn, order)
        x = np.arange(len(order))
        ax.bar(x - 0.2, fe, width=0.4, color=C_PRIOR, label=f"{early_lbl} (N={ne})")
        ax.bar(x + 0.2, fl, width=0.4, color=C_STEER, label=f"{late_lbl} (N={nl})")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel("Fraction of structures (%)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")

    out = args.output or (run_dir / "deliverables" / "symmetry_shift.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext))
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} / .pdf")
    for lbl, c in ((early_lbl, c_early), (late_lbl, c_late)):
        f, n = _fracs(c, _system, SYSTEMS)
        print(f"  {lbl} (N={n}): " + ", ".join(f"{s}:{v:.0f}%" for s, v in zip(SYSTEMS, f) if v > 0))


if __name__ == "__main__":
    main()
