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
from plot_distribution_shift import load_bg_eta, _sq_curve, detect_eta_dir
import torch
from collections import Counter
from plot_run import moving_average
from plot_run_story_composite import _load_reward_series, _plot_reward
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# light-grey mark for the in-target region (was salmon "#fb9a99"); grey needs a
# touch more alpha than salmon did to read as a shaded band under the viridis points
_TARGET_SHADE = "#d9d9d9"
_TARGET_ALPHA = 0.55


def _bg_eta(ax, name, bg, eta, zmax, sq_x, sq_y, eta_min, show_ylabel):
    """(band gap, eta) scatter: KDE-coloured points, shaded target region, and a
    single legend carrying N, in-target%% and the SQ limit (no stray annotations)."""
    from scipy.stats import gaussian_kde
    ax.fill_between(sq_x, eta_min, sq_y, where=(sq_y > eta_min), color=_TARGET_SHADE,
                    alpha=_TARGET_ALPHA, lw=0)
    ax.plot(sq_x, sq_y, "k--", lw=1.2)
    if bg.size >= 5:
        try:
            z = gaussian_kde(np.vstack([bg, eta]))(np.vstack([bg, eta]))
        except Exception:  # noqa: BLE001
            z = np.ones(bg.size)
    else:
        z = np.ones(bg.size)
    o = z.argsort()
    sc = ax.scatter(bg[o], eta[o], c=z[o], s=14, cmap="viridis", vmin=0.0, vmax=zmax,
                    edgecolors="none", zorder=3)
    cap = np.interp(bg, sq_x, sq_y)
    frac = 100.0 * ((eta > eta_min) & (eta <= cap)).mean() if bg.size else 0.0
    ax.set_xlim(-0.05, 4.0); ax.set_ylim(0, 0.40)
    ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([0.05, 0.15, 0.25, 0.35])
    # bare tick numbers (the unit lives in the axis label "SLME eta (%)"); and hide the
    # y-tick labels on the Steered panel, which shares the y-axis with Prior.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_xlabel("Band gap (eV)")
    if show_ylabel:
        ax.set_ylabel(r"SLME $\eta$ (%)")
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(name, fontsize=13)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4c4c4c",
               markersize=5, label=f"N = {bg.size}"),
        Patch(facecolor=_TARGET_SHADE, alpha=_TARGET_ALPHA, label=f"In target region: {frac:.0f}%"),
        Line2D([0], [0], color="black", ls="--", lw=1.2, label="SQ limit"),
    ], loc="upper right", fontsize=8.5, handlelength=1.3, handletextpad=0.5,
        labelspacing=0.3, borderaxespad=0.3)
    return sc
from plot_slme_fig4_panelc import _periodic_table, _eff_n, _sg_counts, C_PRIOR, C_STEER
from demo_element_migration_gif import build_layout, load_density
from plot_eval_symmetry_inset import _hm


def _ema_debiased(x, w):
    """TensorBoard-style debiased exponential moving average (weight w in [0,1))."""
    out = np.empty(len(x), dtype=float); last = 0.0; deb = 0.0
    for i, v in enumerate(x):
        last = w * last + (1 - w) * v
        deb = w * deb + (1 - w)
        out[i] = last / deb
    return out


def _plot_reward_ema(ax, steps, reward, window, max_step=None):
    """TensorBoard-style: faint raw reward trace + bold exponential moving average
    (debiased, span≈`window`), no band. EMA is defined at every point, so the line
    runs cleanly to the last step with no truncated-window endpoint artifact.
    `max_step` caps the x-axis (the plateaued/drifting tail is cut for display)."""
    fin = np.isfinite(reward)
    s, r = steps[fin], reward[fin]
    ema = _ema_debiased(r, (window - 1) / (window + 1))   # span-`window` decay
    ax.plot(s, r, color="0.78", lw=0.8, label="Reward mean (raw)", zorder=1)
    ax.plot(s, ema, color="black", lw=2.0, label=f"EMA (span {window})", zorder=3)
    ax.set_xlabel("RL step"); ax.set_ylabel("Reward mean")
    xhi = s.max() + 1 if max_step is None else min(s.max() + 1, max_step)
    ax.set_xlim(s.min(), xhi)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.18 * (yhi - ylo))       # headroom for legend / labels
    ax.legend(loc="upper center", ncol=1, fontsize=12, framealpha=0.9,
              handlelength=1.3, borderaxespad=0.3)


def _window_stats(run_dir, steps):
    """Element occurrence (% of structures, length-119) and space-group counts over
    the SLME-SCOREABLE structures (finite band gap AND eta) in `steps`. Returns
    (occ_pct, sg_counter, n) so every panel uses the same finite-(E_g,eta) population
    as the (b) scatter."""
    eta_dir = detect_eta_dir(run_dir)
    bg_dir = run_dir / "rewards" / "bandgap"
    dens = np.zeros(119); sg: Counter = Counter(); n = 0
    for s in steps:
        ev = run_dir / "samples" / f"step_{int(s):04d}_eval.pt"
        ef, bf = eta_dir / f"step_{int(s):04d}.txt", bg_dir / f"step_{int(s):04d}.txt"
        if not (ev.exists() and ef.exists() and bf.exists()):
            continue
        payload = torch.load(ev, map_location="cpu")
        eta = np.array([float(x) for x in ef.read_text().splitlines() if x.strip()])
        bg = np.array([float(x) for x in bf.read_text().splitlines() if x.strip()])
        fin = np.isfinite(eta) & np.isfinite(bg)
        for i, it in enumerate(payload):
            if i >= len(fin) or not fin[i] or not isinstance(it, dict):
                continue
            n += 1
            for z in {int(z) for z in it["atom_types"].tolist()}:
                if z < 119:
                    dens[z] += 1
            if it.get("spacegroup") is not None:
                sg[int(it["spacegroup"])] += 1
    occ = dens / n * 100.0 if n else dens
    return occ, sg, max(n, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--window", type=int, default=5, help="reward moving-average window")
    p.add_argument("--win-steps", type=int, default=10, help="steps per early/best window")
    p.add_argument("--top", type=int, default=10, help="space groups in the (c) bar panel")
    p.add_argument("--max-step", type=int, default=None,
                   help="cap the (a) reward x-axis at this RL step (trims the drifting tail)")
    p.add_argument("--target-eta-min", type=float, default=0.25)
    p.add_argument("--steered-window", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                   help="override the EMA-peak steered window with an explicit [LO, HI) step range "
                        "(e.g. --steered-window 48 58 for a late/condensed region when the reward "
                        "peaks early but the chemistry keeps evolving)")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    # bump the rcParams-driven fonts (axis labels, ticks, default titles/legends) +2
    plt.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 14,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 13})
    pds.TARGET_ETA_MIN = float(args.target_eta_min)
    eta_min = pds.TARGET_ETA_MIN

    # ---- windows (shared by a/b/c): early = first win-steps; best = peak-MA window
    steps, reward = _load_reward_series(run_dir)
    W = args.win_steps
    early_rng = (0, W)
    fin = np.isfinite(reward)
    fin_steps = steps[fin]
    # locate the peak on the SAME EMA that panel (a) shows (consistent with display)
    sm = _ema_debiased(reward[fin], (args.window - 1) / (args.window + 1))
    best_step = int(fin_steps[int(np.nanargmax(sm))])
    half = W // 2
    lo, hi = best_step - half, best_step + (W - half)
    max_step = int(np.nanmax(fin_steps))
    if hi > max_step + 1:
        lo -= hi - (max_step + 1); hi = max_step + 1
    steered_rng = (max(0, lo), hi)
    if args.steered_window is not None:
        steered_rng = (int(args.steered_window[0]), int(args.steered_window[1]))

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
    early_steps = list(range(early_rng[0], early_rng[1]))
    best_steps = list(range(steered_rng[0], steered_rng[1]))
    # elements + space groups over the SAME finite-(E_g,eta) scoreable population as (b)
    e_vec, ce, ne = _window_stats(run_dir, early_steps)
    b_vec, cb_, nb = _window_stats(run_dir, best_steps)
    layout = build_layout()
    present = [z for z in range(1, 119) if z in layout and (e_vec[z] > 0 or b_vec[z] > 0)]
    fblock = set(range(57, 72)) | set(range(89, 104))
    cells = [z for z in layout if z not in fblock or any(p in fblock for p in present)]
    max_row = max(layout[z][0] for z in cells)
    pos = np.concatenate([e_vec[present], b_vec[present]]); pos = pos[pos > 0]
    vmin, vmax = float(pos.min()), float(pos.max())
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")

    sgs = sorted(set(ce) | set(cb_), key=lambda s: (cb_[s] / nb, ce[s] / ne), reverse=True)[:args.top]

    # ---- layout: LEFT column = (a) reward over (b) scatters; RIGHT column = (c)
    # two wide panels (element periodic-table pair over space-group bars)
    # left and right share the SAME row split so (a)≡(c) and (b)≡(d) align in height;
    # top row taller than bottom so panel (b) isn't stretched
    fig = plt.figure(figsize=(15.0, 9.2))
    outer = GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.12, figure=fig)
    left = outer[0].subgridspec(2, 1, height_ratios=[1.15, 0.85], hspace=0.34)
    # small wspace: the two scatters share the y-axis (Steered hides its tick labels),
    # so they can sit close together and each span more width to fill the (b) row
    bsub = left[1].subgridspec(1, 2, wspace=0.08)
    # right: (c) two periodic tables stacked vertically, then (d) SG bars
    right = outer[1].subgridspec(2, 1, height_ratios=[1.35, 0.75], hspace=0.28)
    csub = right[0].subgridspec(2, 1, hspace=0.08)

    ax_reward = fig.add_subplot(left[0])
    ax_early = fig.add_subplot(bsub[0])
    ax_steered = fig.add_subplot(bsub[1], sharex=ax_early, sharey=ax_early)
    axL = fig.add_subplot(csub[0]); axR = fig.add_subplot(csub[1])
    # nudge (d) rightward / narrower with a small left spacer so it lines up under
    # the (centred) periodic tables
    botsub = right[1].subgridspec(1, 2, width_ratios=[0.06, 0.94], wspace=0.0)
    ax_sg = fig.add_subplot(botsub[1])

    # (a) reward: raw + EMA (no truncated-window endpoint artifact) + colour bands
    _plot_reward_ema(ax_reward, steps, reward, args.window, args.max_step)
    ylo_r, yhi_r = ax_reward.get_ylim()
    for rng, name, band, txt in ((early_rng, "Prior", C_PRIOR, "#1f6fb4"),
                                 (steered_rng, "Steered", C_STEER, "#d62728")):
        ax_reward.axvspan(rng[0], rng[1], color=band, alpha=0.32, lw=0, zorder=0)
        ax_reward.text(0.5 * (rng[0] + rng[1]), yhi_r - 0.03 * (yhi_r - ylo_r), name,
                       ha="center", va="top", color=txt, fontsize=9.5, fontweight="bold")
    # (b) prior vs steered scatter (clean local panel; few ticks)
    sc = _bg_eta(ax_early, "Prior", e_bg, e_eta, zmax, sq_x, sq_y, eta_min, True)
    _bg_eta(ax_steered, "Steered (best)", s_bg, s_eta, zmax, sq_x, sq_y, eta_min, False)
    fig.colorbar(sc, ax=[ax_early, ax_steered], label="KDE density", fraction=0.046, pad=0.02)

    # (c) element periodic tables, stacked vertically; label goes in the empty
    # top-middle block (groups 3-12, periods 1-3), centred at col 6.5, row ~1
    n_prior = int((e_vec > 0).sum())   # distinct elements appearing in the window
    n_steer = int((b_vec > 0).sum())
    _periodic_table(axL, e_vec, cells, layout, vmin, vmax, cmap, max_row, "")
    axL.text(6.5, 1.0, f"Prior\n(first {len(early_steps)} steps)\n{n_prior} elements",
             ha="center", va="center", fontsize=11, linespacing=1.4)
    im = _periodic_table(axR, b_vec, cells, layout, vmin, vmax, cmap, max_row, "")
    axR.text(6.5, 1.0, f"Steered\n(steps {best_steps[0]}–{best_steps[-1]})\n{n_steer} elements",
             ha="center", va="center", fontsize=11, linespacing=1.4)
    cbar = fig.colorbar(im, ax=[axL, axR], fraction=0.03, pad=0.02)
    cbar.set_label("% of structures")

    # (c-bottom) space-group before/after
    x = np.arange(len(sgs))
    ax_sg.bar(x - 0.2, [100 * ce[s] / ne for s in sgs], width=0.4, color=C_PRIOR,
              label="Prior")
    ax_sg.bar(x + 0.2, [100 * cb_[s] / nb for s in sgs], width=0.4, color=C_STEER,
              label="Steered")
    ax_sg.set_xticks(x); ax_sg.set_xticklabels([_hm(s) for s in sgs], rotation=35, ha="right")
    ax_sg.set_ylabel("% of structures")
    ax_sg.set_title(f"Top {len(sgs)} space groups")
    ax_sg.legend(loc="upper right")

    # raise (d) to align horizontally with (b) (same y-band) WITHOUT resizing (c)
    pe = ax_early.get_position()
    psd = ax_sg.get_position()
    ax_sg.set_position([psd.x0, pe.y0, psd.width, pe.height])

    # (a),(b) letters at their top-left
    for ax, lab in ((ax_reward, "(a)"), (ax_early, "(b)")):
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(-44, 8),
                    textcoords="offset points", fontsize=18, fontweight="bold",
                    ha="left", va="bottom", annotation_clip=False)
    # (c),(d) share a common x so the two right-column letters align vertically
    pc, pd = axL.get_position(), ax_sg.get_position()
    xcd = pc.x0 - 0.024
    for y, lab in ((pc.y1 + 0.012, "(c)"), (pd.y1 + 0.012, "(d)")):
        fig.text(xcd, y, lab, fontsize=18, fontweight="bold", ha="left", va="bottom")

    out = args.output or (run_dir / "deliverables" / "slme_story_composite_v3.png")
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
