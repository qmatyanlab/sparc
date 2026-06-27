#!/usr/bin/env python3
"""Sweep the band_gap/eta weight split on the ASCENDING SLME reward and plot
reward-vs-band_gap for each profile, using a run's actual generated structures.

All profiles use: band_gap ascending [0.5,3.0], eta ascending [0,0.35]; only the
(w_bg, w_eta) split changes. Also reports the top-k advantage allocation
(corner BG<0.5 vs escaped) that actually drives RL drift (baseline=min(top-k)).

Example:
    python scripts/compare_ascending_weights.py \
        exp_res/tsenn_slme_03um_optimate_etaheavy_adaptive_v1_54218097
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (w_bg, w_eta) profiles to compare
PROFILES = [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.0, 1.0)]
TOPK_RATIO = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir", type=Path)
    p.add_argument("--bg-dir", default="rewards/bandgap")
    p.add_argument("--eta-dir", default="rewards/tsenn_slme_optimate_eta")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def ascending(v, minv, maxv):
    return np.clip((v - minv) / (maxv - minv), 0.0, 1.0)


def reward(bg, eta, w_bg, w_eta):
    return w_bg * ascending(bg, 0.5, 3.0) + w_eta * ascending(eta, 0.0, 0.35)


def load_pairs(exp_dir, bg_sub, eta_sub):
    bg_dir, eta_dir = exp_dir / bg_sub, exp_dir / eta_sub
    bg_steps = {int(p.stem.split("_")[-1]): p for p in bg_dir.glob("step_*.txt")}
    eta_steps = {int(p.stem.split("_")[-1]): p for p in eta_dir.glob("step_*.txt")}
    bg, eta, step_id = [], [], []
    for s in sorted(set(bg_steps) & set(eta_steps)):
        b = np.loadtxt(bg_steps[s], ndmin=1)
        e = np.loadtxt(eta_steps[s], ndmin=1)
        n = min(len(b), len(e))
        bg.append(b[:n]); eta.append(e[:n]); step_id.append(np.full(n, s))
    return (np.nan_to_num(np.concatenate(bg), nan=0.0),
            np.clip(np.nan_to_num(np.concatenate(eta), nan=0.0), 0.0, None),
            np.concatenate(step_id))


def advantage_split(bg, eta, steps, w_bg, w_eta):
    """top-k advantage going to corner (BG<0.5) vs escaped, per RL mechanics."""
    cs, es = [], []
    for s in np.unique(steps):
        m = steps == s
        g, e = bg[m], eta[m]
        if len(g) < 4:
            continue
        r = reward(g, e, w_bg, w_eta)
        k = max(1, int(len(r) * TOPK_RATIO))
        top = np.argsort(r)[::-1][:k]
        rt, gt = r[top], g[top]
        adv = rt - rt.min()
        if adv.sum() <= 0:
            continue
        corner = gt < 0.5
        cs.append(adv[corner].sum() / adv.sum())
        es.append(adv[~corner].sum() / adv.sum())
    return float(np.mean(cs)), float(np.mean(es))


def main() -> None:
    args = parse_args()
    out = args.output_dir or (args.exp_dir / "deliverables" / "ascending_weight_sweep")
    out.mkdir(parents=True, exist_ok=True)

    bg, eta, steps = load_pairs(args.exp_dir, args.bg_dir, args.eta_dir)
    eta_pct = eta * 100.0
    print(f"loaded {len(bg)} structures; BG median={np.median(bg):.3f} eV")

    n = len(PROFILES)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 9),
                             gridspec_kw={"height_ratios": [3, 2]})
    bins = np.linspace(0, 3.5, 18)
    print(f"\n{'profile':>14} {'corner<0.5':>11} {'escaped':>9} "
          f"{'R@BG<0.5':>9} {'R@[1,1.5]':>10} {'R@BG>2':>8}")
    for j, (w_bg, w_eta) in enumerate(PROFILES):
        r = reward(bg, eta, w_bg, w_eta)
        ax = axes[0, j]
        sc = ax.scatter(bg, r, c=eta_pct, s=8, alpha=0.5, cmap="viridis",
                        vmin=0, vmax=35)
        idx = np.digitize(bg, bins)
        bx = [bg[idx == k].mean() for k in range(1, len(bins)) if (idx == k).any()]
        by = [np.median(r[idx == k]) for k in range(1, len(bins)) if (idx == k).any()]
        ax.plot(bx, by, "r-", lw=2, label="binned median")
        ax.axvspan(0, 0.5, color="gray", alpha=0.12)
        ax.axvline(1.25, color="k", ls=":", lw=1)
        ax.set_xlim(0, 3.5); ax.set_ylim(0, 1.0)
        ax.set_title(f"{w_bg:.1f}·BG + {w_eta:.1f}·η")
        ax.set_xlabel("band gap (eV)")
        if j == 0:
            ax.set_ylabel("total reward")
            ax.legend(fontsize=8, loc="upper right")

        # advantage allocation bar
        c_share, e_share = advantage_split(bg, eta, steps, w_bg, w_eta)
        axes[1, j].bar(["corner\n<0.5", "escaped\n>=0.5"], [c_share, e_share],
                       color=["tab:red", "tab:green"])
        axes[1, j].set_ylim(0, 1.0)
        for i, v in enumerate([c_share, e_share]):
            axes[1, j].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
        if j == 0:
            axes[1, j].set_ylabel("top-k advantage share")

        corner = bg < 0.5
        tgt = (bg >= 1.0) & (bg <= 1.5)
        wide = bg > 2.0
        print(f"{w_bg:.1f}/{w_eta:.1f}".rjust(14)
              + f"{c_share:>11.1%}{e_share:>9.1%}"
              + f"{r[corner].mean():>9.3f}{r[tgt].mean():>10.3f}{r[wide].mean():>8.3f}")

    cbar = fig.colorbar(sc, ax=axes[0, :].tolist(), shrink=0.8, pad=0.01)
    cbar.set_label("predicted η (%)")
    fig.suptitle(f"Ascending SLME reward — band_gap/η weight sweep — {args.exp_dir.name}",
                 y=0.99)
    fig.savefig(out / "ascending_weight_sweep.png", dpi=150, bbox_inches="tight")
    print(f"\nfigure: {out / 'ascending_weight_sweep.png'}")


if __name__ == "__main__":
    main()
