#!/usr/bin/env python3
"""Side-by-side Figure-4 panel (a) for two runs -- the batch-filter gate ablation.

Renders the design-space drift KDE for N runs on IDENTICAL axes, KDE bandwidth, contour
levels and colour ramp, so the only thing that differs between panels is the run. Built
to answer "what did the strict S.U.N. gate actually do?" by putting the pre-S.U.N. and
S.U.N. panels next to each other.

Panel rendering mirrors `figure_4.py::draw_drift` (KDE bw=0.55, gridn=200, truncated
Reds, target rectangle at gap>=0.3 / reward>=0.5) and reads the same cached
`drift_iso_data.npz` that `scripts/export_fig4_drift_data.py` writes.

Density is one quantity, so both panels share one sequential ramp; the runs are told
apart by position and title, never by hue. The two median markers are the only
categorical encoding and use a CVD-validated pair.

Usage:
  uv run python scripts/plot_fig4_drift_compare.py \
      exp_res/<run_a>/figure4_bundle/drift_iso_data.npz "pre-S.U.N. gate (V.U.N)" \
      exp_res/<run_b>/figure4_bundle/drift_iso_data.npz "strict S.U.N. gate (V.U.N.S)" \
      -o exp_res/_abrepr/fig4_drift_gate_ablation.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Validated against the light chart surface (all pairs): CVD dE 22.9 (protan),
# normal-vision dE 31.0, contrast >= 3:1 -- all six checks PASS.
EARLY, LATE = "#0A7BBF", "#D55E00"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#dcdcd8"
LEVELS = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]


def stats(d, n=10, gt=0.3, st=0.5):
    gap, reward, step = d["gap"], d["reward"], d["step"]
    gate_min = float(d["gate_min"])
    steps = np.unique(step)
    k = min(n, max(1, len(steps) // 2))
    em, lm = np.isin(step, steps[:k]), np.isin(step, steps[-k:])
    pg = gap >= gate_min                       # above the band-gap gate (non-metallic)
    # Medians and fractions are taken over the FULL early/late population, with no gate
    # mask -- the same convention as `plot_fig4_drift.py --aggregate`, which is what
    # Figure 4 panel (a) mirrors. Masking by the gate here would silently report
    # different numbers from the published panel.
    med = lambda m: (float(np.median(gap[m])), float(np.median(reward[m])))  # noqa: E731
    frac = lambda m, c: 100 * float(np.mean(c[m])) if m.sum() else 0.0       # noqa: E731
    intgt = (gap >= gt) & (reward >= st)
    return {
        "n": int(gap.size), "loops": int(len(steps)),
        "early": med(em), "late": med(lm),
        "metals": (frac(em, ~pg), frac(lm, ~pg)),
        "in_target": (frac(em, intgt), frac(lm, intgt)),
        "gate_min": gate_min,
    }


def draw(ax, d, title, xlim, ylim, n=10, gt=0.3, st=0.5):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    gap, reward = d["gap"], d["reward"]
    ramp = LinearSegmentedColormap.from_list("rt", plt.cm.Reds(np.linspace(0.35, 1.0, 256)))
    gn = 200
    XX, YY = np.meshgrid(np.linspace(*xlim, gn), np.linspace(*ylim, gn))
    ZZ = gaussian(gap, reward, XX, YY)
    Zn = ZZ / ZZ.max() if ZZ.max() > 0 else ZZ
    ax.contour(XX, YY, Zn, levels=LEVELS, cmap=ramp, linewidths=1.5, zorder=3)

    # target region: what the reward is actually asking for
    ax.plot([gt, gt, xlim[1]], [ylim[1], st, st], linestyle=(0, (5, 4)),
            color=MUTED, linewidth=1.2, zorder=2)

    s = stats(d, n, gt, st)
    (ex, ey), (lx, ly) = s["early"], s["late"]
    ax.annotate("", xy=(lx, ly), xytext=(ex, ey), zorder=4,
                arrowprops=dict(arrowstyle="-|>", linewidth=2.0, color=INK,
                                shrinkA=6, shrinkB=8))
    ax.scatter([ex], [ey], s=95, c=EARLY, edgecolors="white", linewidths=1.6, zorder=5,
               label=f"first {n} loops (median)")
    ax.scatter([lx], [ly], s=95, c=LATE, edgecolors="white", linewidths=1.6, zorder=5,
               label=f"last {n} loops (median)")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Predicted band gap (eV)", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=9)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

    ax.text(0.03, 0.035,
            f"N = {s['n']:,} over {s['loops']} loops\n"
            f"below gap gate: {s['metals'][0]:.0f}% $\\rightarrow$ {s['metals'][1]:.0f}%\n"
            f"in target: {s['in_target'][0]:.0f}% $\\rightarrow$ {s['in_target'][1]:.0f}%",
            transform=ax.transAxes, fontsize=8, color=INK, va="bottom", ha="left", zorder=6,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=GRID,
                      linewidth=0.6, alpha=0.96))
    return s


def gaussian(x, y, XX, YY, bw=0.55):
    from scipy.stats import gaussian_kde
    return gaussian_kde(np.vstack([x, y]), bw_method=bw)(
        np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="<npz> <label> [<npz> <label> ...]")
    ap.add_argument("-o", "--output", type=Path,
                    default=ROOT / "exp_res" / "_abrepr" / "fig4_drift_gate_ablation.png")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--gap-max", type=float, default=4.0)
    args = ap.parse_args(argv)

    if len(args.pairs) % 2:
        raise SystemExit("pairs must be <npz> <label> ...")
    items = [(Path(args.pairs[i]), args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)]
    data = [(np.load(p), lab) for p, lab in items]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # shared limits so the panels are directly comparable
    x_hi = min(args.gap_max, max(float(np.percentile(d["gap"], 99)) for d, _ in data) + 0.3)
    xlim, ylim = (-0.1, x_hi), (0.0, 1.03)

    fig, axes = plt.subplots(1, len(data), figsize=(4.6 * len(data), 4.0),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    out = []
    for ax, (d, lab) in zip(axes, data):
        out.append((lab, draw(ax, d, lab, xlim, ylim, args.n)))
    axes[0].set_ylabel(r"Reward $r_\mathrm{uni}$", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([0], [0], linestyle=(0, (5, 4)), color=MUTED, linewidth=1.2))
    l.append("target region")
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3,
               fontsize=8, labelcolor=INK, frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")

    print(f"{'run':38s} {'N':>7s} {'median (gap,reward)':>26s} {'below gate':>14s} {'in target':>13s}")
    for lab, s in out:
        print(f"  {lab:36s} {s['n']:7,d}  ({s['early'][0]:.2f},{s['early'][1]:.2f}) -> "
              f"({s['late'][0]:.2f},{s['late'][1]:.2f})  "
              f"{s['metals'][0]:4.0f}% -> {s['metals'][1]:3.0f}%  "
              f"{s['in_target'][0]:4.0f}% -> {s['in_target'][1]:3.0f}%")
    print(f"\nwrote {args.output} + {pdf.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
