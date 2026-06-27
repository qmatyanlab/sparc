#!/usr/bin/env python3
"""Compare SLME reward schemes (ascending vs eta_heavy/centered band gap) on a
run's actual generated (band_gap, eta) pairs, to see which scheme rewards the
degenerate band_gap->0 corner.

Schemes (band_gap term, eta term; reduce=weighted sum), matching the configs:
  ascending  : BG ascending [0.5,3.0] w=0.4 ; eta ascending [0,0.35] w=0.6
               (configs/reward/tsenn_slme_optimate.yaml and tsenn_slme.yaml)
  eta_heavy  : BG centered@1.25 [0,1.5] w=0.2 ; eta ascending [0,0.35] w=0.8
               (configs/reward/tsenn_slme_optimate_eta_heavy.yaml)

Example:
    python scripts/compare_slme_reward_schemes.py \
        exp_res/tsenn_slme_03um_optimate_etaheavy_adaptive_v1_54218097
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir", type=Path)
    p.add_argument("--bg-dir", default="rewards/bandgap")
    p.add_argument("--eta-dir", default="rewards/tsenn_slme_optimate_eta")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def ascending(v, minv, maxv):
    return np.clip((v - minv) / (maxv - minv), 0.0, 1.0)


def centered(v, target, minv, maxv):
    # rewards/reward.py linear_target: linear_scaling(-|v-target|, -maxv, -minv)
    diff = np.abs(v - target)
    return np.clip((maxv - diff) / (maxv - minv), 0.0, 1.0)


def centered_floor(v, target, minv, maxv, gap_floor):
    # centered@target, but hard-zeroed below gap_floor (kills metal-corner credit)
    return np.where(v < gap_floor, 0.0, centered(v, target, minv, maxv))


SCHEMES = {
    "ascending (optimate / old)": {
        "bg": lambda bg: ascending(bg, 0.5, 3.0),
        "w_bg": 0.4,
        "eta": lambda eta: ascending(eta, 0.0, 0.35),
        "w_eta": 0.6,
        "color": "tab:blue",
    },
    "eta_heavy (centered@1.25)": {
        "bg": lambda bg: centered(bg, 1.25, 0.0, 1.5),
        "w_bg": 0.2,
        "eta": lambda eta: ascending(eta, 0.0, 0.35),
        "w_eta": 0.8,
        "color": "tab:red",
    },
    "smooth tent (centered@1.25, maxv=0.75)": {
        "bg": lambda bg: centered(bg, 1.25, 0.0, 0.75),
        "w_bg": 0.2,
        "eta": lambda eta: ascending(eta, 0.0, 0.35),
        "w_eta": 0.8,
        "color": "tab:purple",
    },
}


def load_pairs(exp_dir: Path, bg_sub: str, eta_sub: str):
    bg_dir, eta_dir = exp_dir / bg_sub, exp_dir / eta_sub
    bg_steps = {int(p.stem.split("_")[-1]): p for p in bg_dir.glob("step_*.txt")}
    eta_steps = {int(p.stem.split("_")[-1]): p for p in eta_dir.glob("step_*.txt")}
    bgs, etas = [], []
    for step in sorted(set(bg_steps) & set(eta_steps)):
        b = np.loadtxt(bg_steps[step], ndmin=1)
        e = np.loadtxt(eta_steps[step], ndmin=1)
        n = min(len(b), len(e))
        bgs.append(b[:n]); etas.append(e[:n])
    return np.concatenate(bgs), np.concatenate(etas)


def main() -> None:
    args = parse_args()
    out = args.output_dir or (args.exp_dir / "deliverables" / "reward_scheme_compare")
    out.mkdir(parents=True, exist_ok=True)

    bg, eta = load_pairs(args.exp_dir, args.bg_dir, args.eta_dir)
    # pipeline scores NaN predictions as 0.0 (rewards/reward.py np.nan_to_num)
    bg = np.nan_to_num(bg, nan=0.0)
    eta = np.clip(np.nan_to_num(eta, nan=0.0), 0.0, None)
    print(f"loaded {len(bg)} (band_gap, eta) pairs; "
          f"band_gap median={np.median(bg):.3f} eV, eta median={np.median(eta):.3f}")

    low = bg < 0.5  # the degenerate near-metallic corner
    print(f"\nfraction with band_gap < 0.5 eV: {low.mean():.1%} "
          f"(eta there: median={np.median(eta[low]):.3f})")

    # 1-D band_gap term comparison + total reward vs band_gap on real data
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    grid = np.linspace(0, 3.5, 400)
    for name, s in SCHEMES.items():
        axes[0].plot(grid, s["w_bg"] * s["bg"](grid), color=s["color"],
                     label=f"{name}  (w={s['w_bg']})")
    axes[0].axvspan(0, 0.5, color="gray", alpha=0.12, label="BG<0.5 (collapse corner)")
    axes[0].axvline(1.25, color="k", ls=":", lw=1, label="eta peak ~1.25 eV")
    axes[0].set_xlabel("band gap (eV)"); axes[0].set_ylabel("weighted band-gap reward term")
    axes[0].set_title("band-gap reward TERM"); axes[0].legend(fontsize=8)

    for name, s in SCHEMES.items():
        total = s["w_bg"] * s["bg"](bg) + s["w_eta"] * s["eta"](eta)
        order = np.argsort(bg)
        axes[1].scatter(bg, total, s=5, alpha=0.25, color=s["color"], label=name)
        # binned median
        bins = np.linspace(0, 3.5, 18)
        idx = np.digitize(bg, bins)
        bx = [bg[idx == k].mean() for k in range(1, len(bins)) if (idx == k).any()]
        by = [np.median(total[idx == k]) for k in range(1, len(bins)) if (idx == k).any()]
        axes[1].plot(bx, by, color=s["color"], lw=2)
        print(f"\n[{name}] total reward: overall mean={total.mean():.3f}; "
              f"BG<0.5 mean={total[low].mean():.3f}; BG in [1.0,1.5] mean={total[(bg>=1)&(bg<=1.5)].mean():.3f}")
    axes[1].axvspan(0, 0.5, color="gray", alpha=0.12)
    axes[1].set_xlabel("band gap (eV)"); axes[1].set_ylabel("total reward")
    axes[1].set_title("total reward vs band gap (this run's structures)")
    axes[1].legend(fontsize=8)

    # histogram of band gaps actually generated
    axes[2].hist(bg, bins=np.linspace(0, 3.5, 50), color="gray", alpha=0.7)
    axes[2].axvspan(0, 0.5, color="red", alpha=0.12)
    axes[2].axvline(1.25, color="k", ls=":", lw=1)
    axes[2].set_xlabel("band gap (eV)"); axes[2].set_ylabel("count")
    axes[2].set_title("generated band-gap distribution (all steps)")

    fig.suptitle(f"SLME reward-scheme comparison — {args.exp_dir.name}")
    fig.tight_layout()
    fig.savefig(out / "reward_scheme_compare.png", dpi=150)
    print(f"\nfigure: {out / 'reward_scheme_compare.png'}")


if __name__ == "__main__":
    main()
