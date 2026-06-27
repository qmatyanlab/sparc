#!/usr/bin/env python3
"""Fig 4 -- SLME composite story (v2).

Reuses the settled (a) reward and (b) prior->steered (band gap, eta) distribution
shift from plot_run_story_composite, and replaces the old structure/spectra panel
with the before/after CONDENSATION view:
  (c-top)    element condensation: prior vs best-stage periodic tables (fraction of
             structures containing each element; white = absent; e^H annotated);
  (c-bottom) space-group condensation: top-N SGs, prior vs steered.

The SAME early/best windows drive (a) red boxes, (b) scatters, and (c).

Usage:
  python scripts/plot_slme_story_composite_v2.py <run_dir> [--window 5] [--top 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_distribution_shift as pds
from plot_distribution_shift import load_bg_eta, _sq_curve
from plot_run import moving_average
from plot_run_story_composite import _load_reward_series, _plot_reward, _plot_bg_eta_panel
from plot_slme_fig4_panelc import _periodic_table, _eff_n, _sg_counts, C_PRIOR, C_STEER
from demo_element_migration_gif import build_layout, load_density
from plot_eval_symmetry_inset import _hm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--window", type=int, default=5, help="reward moving-average window")
    p.add_argument("--win-steps", type=int, default=10, help="steps per early/best window")
    p.add_argument("--top", type=int, default=10, help="space groups in the (c) bar panel")
    p.add_argument("--target-eta-min", type=float, default=0.25)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    pds.TARGET_ETA_MIN = float(args.target_eta_min)
    eta_min = pds.TARGET_ETA_MIN

    # ---- windows (shared by a/b/c): early = first win-steps; best = peak-MA window
    steps, reward = _load_reward_series(run_dir)
    W = args.win_steps
    early_rng = (0, W)
    fin = np.isfinite(reward)
    fin_steps = steps[fin]
    ma = moving_average(reward[fin], args.window)
    best_step = int(fin_steps[int(np.nanargmax(ma))])
    half = W // 2
    lo, hi = best_step - half, best_step + (W - half)
    max_step = int(np.nanmax(fin_steps))
    if hi > max_step + 1:
        lo -= hi - (max_step + 1); hi = max_step + 1
    steered_rng = (max(0, lo), hi)

    # ---- (b) bg/eta for the two windows
    e_bg, e_eta = load_bg_eta(run_dir, early_rng)
    s_bg, s_eta = load_bg_eta(run_dir, steered_rng)
    sq_x = np.linspace(0.0, 4.0, 800)
    sq_y = _sq_curve(sq_x)
    from scipy.stats import gaussian_kde
    zmax = 0.0
    for bg, eta in ((e_bg, e_eta), (s_bg, s_eta)):
        if bg.size >= 5:
            try:
                zmax = max(zmax, float(gaussian_kde(np.vstack([bg, eta]))(np.vstack([bg, eta])).max()))
            except Exception:  # noqa: BLE001
                zmax = max(zmax, 1.0)

    # ---- (c) element + SG condensation over the same windows
    samples = run_dir / "samples"
    early_steps = list(range(early_rng[0], early_rng[1]))
    best_steps = list(range(steered_rng[0], steered_rng[1]))
    steps_d, dens = load_density(run_dir, "occurrence", "none")
    steps_d = np.asarray(steps_d)
    e_idx = [i for i, s in enumerate(steps_d) if s in set(early_steps)]
    b_idx = [i for i, s in enumerate(steps_d) if s in set(best_steps)]
    e_vec, b_vec = dens[e_idx].mean(0) * 100.0, dens[b_idx].mean(0) * 100.0
    layout = build_layout()
    present = [z for z in range(1, 119) if z in layout and dens[:, z].max() > 0]
    fblock = set(range(57, 72)) | set(range(89, 104))
    cells = [z for z in layout if z not in fblock or any(p in fblock for p in present)]
    max_row = max(layout[z][0] for z in cells)
    pos = np.concatenate([e_vec[present], b_vec[present]]); pos = pos[pos > 0]
    vmin, vmax = float(pos.min()), float(pos.max())
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")

    ce, cb_ = _sg_counts(samples, early_steps), _sg_counts(samples, best_steps)
    ne, nb = sum(ce.values()) or 1, sum(cb_.values()) or 1
    sgs = sorted(set(ce) | set(cb_), key=lambda s: (cb_[s] / nb, ce[s] / ne), reverse=True)[:args.top]

    # ---- layout: 3 row-bands
    fig = plt.figure(figsize=(13.5, 14.0))
    outer = GridSpec(3, 1, height_ratios=[1.0, 0.95, 0.7], hspace=0.33, figure=fig)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.30)
    midd = outer[1].subgridspec(1, 2, wspace=0.04)
    botm = outer[2].subgridspec(1, 1)

    ax_reward = fig.add_subplot(top[0])
    ax_early = fig.add_subplot(top[1])
    ax_steered = fig.add_subplot(top[2], sharex=ax_early, sharey=ax_early)
    axL = fig.add_subplot(midd[0]); axR = fig.add_subplot(midd[1])
    ax_sg = fig.add_subplot(botm[0])

    # (a) reward with both windows boxed; compact title + legend for the narrow panel
    _plot_reward(ax_reward, steps, reward, args.window, early_rng, steered_rng)
    ax_reward.set_title("Mean reward vs RL step", fontsize=11, pad=6)
    lg = ax_reward.get_legend()
    if lg is not None:
        lg.remove()
    ax_reward.legend(loc="upper center", ncol=1, fontsize=7.5, framealpha=0.9,
                     handlelength=1.3, borderaxespad=0.3)
    # (b) prior vs steered scatter
    sc = _plot_bg_eta_panel(ax_early, "Prior", e_bg, e_eta, zmax, sq_x, sq_y, eta_min, True)
    _plot_bg_eta_panel(ax_steered, "Steered (best)", s_bg, s_eta, zmax, sq_x, sq_y, eta_min, False)
    fig.colorbar(sc, ax=[ax_early, ax_steered], label="KDE density", fraction=0.046, pad=0.02)

    # (c-top) element periodic tables
    _periodic_table(axL, e_vec, cells, layout, vmin, vmax, cmap, max_row,
                    f"Prior (first {len(early_steps)} steps)\n$e^H$ = {_eff_n(e_vec, present):.1f} elements")
    im = _periodic_table(axR, b_vec, cells, layout, vmin, vmax, cmap, max_row,
                         f"Steered (steps {best_steps[0]}–{best_steps[-1]})\n$e^H$ = {_eff_n(b_vec, present):.1f} elements")
    cbar = fig.colorbar(im, ax=[axL, axR], fraction=0.022, pad=0.02)
    cbar.set_label("Fraction of structures (%)")

    # (c-bottom) space-group before/after
    x = np.arange(len(sgs))
    ax_sg.bar(x - 0.2, [100 * ce[s] / ne for s in sgs], width=0.4, color=C_PRIOR,
              label=f"Prior (N={ne})")
    ax_sg.bar(x + 0.2, [100 * cb_[s] / nb for s in sgs], width=0.4, color=C_STEER,
              label=f"Steered (N={nb})")
    ax_sg.set_xticks(x); ax_sg.set_xticklabels([_hm(s) for s in sgs], rotation=35, ha="right")
    ax_sg.set_ylabel("Fraction of structures (%)")
    ax_sg.set_title("Space-group condensation")
    ax_sg.legend(loc="upper right")

    # panel letters
    for ax, lab, dx in ((ax_reward, "(a)", -0.055), (ax_early, "(b)", -0.07),
                        (axL, "(c)", -0.045)):
        p = ax.get_position()
        fig.text(p.x0 + dx, p.y1 + 0.008, lab, fontsize=16, fontweight="bold",
                 ha="left", va="bottom")

    out = args.output or (run_dir / "deliverables" / "slme_story_composite_v2.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} / .pdf")
    print(f"  windows: early {early_rng}, steered {steered_rng}")
    print(f"  elements e^H: {_eff_n(e_vec, present):.1f} -> {_eff_n(b_vec, present):.1f}")
    print("  top SG steered: " + ", ".join(f"{s}:{100*cb_[s]/nb:.0f}%" for s in sgs[:5]))


if __name__ == "__main__":
    main()
