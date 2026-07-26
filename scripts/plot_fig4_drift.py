#!/usr/bin/env python3
"""Figure 4 -- RL drift in the (band gap, reward) plane: first-N vs last-N loops.

A single-panel figure for one dielectric run. It overlays the EARLY (first-N RL steps, Blues)
and LATE (last-N RL steps, Reds) 2-D KDE density of the sampled population in the (predicted
E3NN band gap, combined RL reward r_uni) plane, with a grey centroid drift arrow (early -> late
medians) marking the RL "iterative move", and a dashed rectangular target region
(E_g >= gap-thr & r_uni >= score-thr).

The y value is the ACTUAL combined RL reward that steers the run -- exactly the "reward mean"
plotted in the composite-story panel (b): each reward term is linear-scaled to [0,1] with the
run's own minv/maxv (rewards/reward.py:linear_scaling) and combined with the run's reduce
(min). Terms are reconstructed from on-disk per-step artifacts: the dielectric scalar from the
dielectric reward dir's step_*.txt, the E3NN band gap from rewards/bandgap/step_*.txt, both 1:1
aligned. No model re-run, no CSV.

Example:
  python scripts/plot_fig4_drift.py \
      exp_res/dielectric_inplane_isotropy_gapgate_mprime_newbg_b96 \
      --n 10 --label "in-plane isotropy" --output exp_res/fig4_drift_ipe_vs_ipiso.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from plot_gap_score_kde import _diel_reward_dir, _kde_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="run directory")
    p.add_argument("--n", type=int, default=10, help="window size: first-N vs last-N steps")
    p.add_argument("--label", default=None, help="panel title (default: run-dir short name)")
    p.add_argument("--gap-thr", type=float, default=0.3, help="target rectangle gap threshold (eV)")
    p.add_argument("--score-thr", type=float, default=0.5, help="target rectangle r_uni threshold")
    p.add_argument("--eps-cap", type=float, default=100.0, help="drop |eps|>cap blow-ups")
    p.add_argument("--gap-max", type=float, default=4.0, help="x-axis upper cap (eV)")
    p.add_argument("--include-below-gate", action="store_true",
                   help="include gap<gate_min (metallic / gate-failing) structures in the "
                        "density -- the RAW full population. With two separate panels the "
                        "metallic pile no longer overlaps, so its shrink is visible. Default "
                        "excludes it.")
    p.add_argument("--aggregate", action="store_true",
                   help="single-panel view: aggregate the density over ALL loops (raw full "
                        "population, line-only contours colored by level) with a dotted "
                        "early->late population-median shift arrow, instead of the two-panel "
                        "early|late drift.")
    p.add_argument("--mark-csv", type=Path, default=None,
                   help="aggregate mode: overlay starred, labeled points from a CSV "
                        "(columns: gap, score, label) — e.g. externally-scored structures")
    p.add_argument("--output", type=Path,
                   default=ROOT / "exp_res" / "fig4_drift_ipe_vs_ipiso.png",
                   help="output path (.pdf written alongside)")
    return p.parse_args()


def _lin(v, minv, maxv):
    """rewards/reward.py linear_scaling: (v-minv)/(maxv-minv) clipped to [0,1]."""
    return np.clip((v - minv) / (maxv - minv + 1e-12), 0.0, 1.0)


def load_reward_cfg(run_dir: Path):
    """Per-term reward scaling (minv/maxv, ascending) + reduce, from the run config
    (hparams.yaml / .hydra/config.yaml). Returns {'reduce', 'diel', 'bandgap'} where the
    dielectric term is the TSENNStaticDielectric prop and 'bandgap' is the band_gap prop
    (or None)."""
    import yaml
    run_dir = Path(run_dir)
    for cand in (run_dir / "hparams.yaml", run_dir / ".hydra" / "config.yaml"):
        if not cand.exists():
            continue
        try:
            cfg = yaml.safe_load(cand.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(cfg, dict):
            continue
        rew = cfg.get("reward") or {}
        diel = bg = None
        for prop in rew.get("prop_cfg") or []:
            if not isinstance(prop, dict):
                continue
            calc = prop.get("calculator") or {}
            tgt = str(calc.get("_target_", ""))
            entry = {"minv": float(prop.get("minv", 0.0)), "maxv": float(prop.get("maxv", 1.0)),
                     "target": str(prop.get("target", "ascending"))}
            if "TSENNStaticDielectric" in tgt:
                diel = entry
            elif "E3NNBandGap" in tgt or prop.get("name") == "band_gap":
                bg = entry
        if diel is not None:
            return {"reduce": str(rew.get("reduce", "min")), "diel": diel, "bandgap": bg}
    return None


def load_gap_raw(run: Path, eps_cap: float):
    """Per-structure (gap, raw dielectric scalar, step) over ALL steps, from on-disk per-step
    artifacts, kept where the tensor valid_mask is set, finite, and within the eps cap."""
    diel = _diel_reward_dir(run)
    if diel is None:
        raise FileNotFoundError(f"{run.name}: need rewards/*/step_*_tensor.npz")
    bg_dir = run / "rewards" / "bandgap"
    g_all, r_all, s_all = [], [], []
    for tp in sorted(diel.glob("step_*_tensor.npz")):
        bp = bg_dir / tp.name.replace("_tensor.npz", ".txt")
        rp = diel / tp.name.replace("_tensor.npz", ".txt")
        if not bp.exists() or not rp.exists():
            continue
        step = int(tp.name.split("_")[1])
        d = np.load(tp, allow_pickle=True)
        ten = d["tensor"]
        mask = d["valid_mask"] if "valid_mask" in d else np.ones(len(ten), bool)
        gaps = np.array([float(x) for x in bp.read_text().split() if x.strip()])
        rew = np.array([float(x) for x in rp.read_text().split() if x.strip()])
        n = min(len(ten), len(mask), len(gaps), len(rew))
        if n == 0:
            continue
        exx, eyy, ezz = ten[:n, 0, 0], ten[:n, 1, 1], ten[:n, 2, 2]
        gap, r = gaps[:n], rew[:n]
        ok = (mask[:n] & np.isfinite(gap) & np.isfinite(r)
              & (np.maximum.reduce([np.abs(exx), np.abs(eyy), np.abs(ezz)]) <= eps_cap))
        if not ok.any():
            continue
        g_all.append(np.maximum(gap[ok], 0.0))
        r_all.append(r[ok])
        s_all.append(np.full(int(ok.sum()), step))
    if not g_all:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(g_all), np.concatenate(r_all), np.concatenate(s_all)


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style = ROOT / "publication.mplstyle"
    if style.exists():
        plt.style.use(str(style))

    run = args.run_dir.resolve()
    label = args.label or run.name.split("_")[1]
    cfg = load_reward_cfg(run)
    if cfg is None:
        raise SystemExit(f"{run.name}: could not read reward config")
    gap, raw, step = load_gap_raw(run, args.eps_cap)
    if gap.size == 0:
        raise SystemExit(f"{run.name}: no finite reward/gap structures")

    # combined RL reward = reduce over each term's linear-scaled value (matches panel (b))
    sd = _lin(raw, cfg["diel"]["minv"], cfg["diel"]["maxv"])
    if cfg["bandgap"] is not None:
        sg = _lin(gap, cfg["bandgap"]["minv"], cfg["bandgap"]["maxv"])
        reward = np.minimum(sd, sg) if cfg["reduce"] == "min" else 0.5 * (sd + sg)
    else:
        reward = sd

    steps = np.unique(step)
    n = min(args.n, max(1, len(steps) // 2))
    early_s, late_s = set(steps[:n].tolist()), set(steps[-n:].tolist())
    em, lm = np.isin(step, list(early_s)), np.isin(step, list(late_s))

    # Below-gate (metallic / gate-failing, E_g < band-gap gate minv) structures have their
    # combined reward pinned at ~0 by the gate, so they clump BOTH distributions into the
    # bottom-left corner and swamp the drift. Exclude them from the density + centroids; keep
    # them only as a reported fraction (they SHRINK over the run -- a result worth stating).
    gate_min = float(cfg["bandgap"]["minv"]) if cfg.get("bandgap") else args.gap_thr
    pg = gap >= gate_min
    gt, st = args.gap_thr, args.score_thr

    def _frac(mask, cond):
        return 100 * np.mean(cond[mask]) if mask.sum() else 0.0

    if args.aggregate:
        # SINGLE panel: density aggregated over ALL loops (raw full population), line-only
        # contours colored by density level (F style) + a dotted early->late median shift arrow
        from matplotlib.colors import LinearSegmentedColormap
        reds_trunc = LinearSegmentedColormap.from_list(
            "rt", plt.cm.Reds(np.linspace(0.35, 1.0, 256)))
        from scipy.stats import gaussian_kde
        x_hi = min(args.gap_max, float(np.percentile(gap, 99)) + 0.3)
        xlim, ylim = (-0.1, x_hi), (-0.02, 1.03)
        # match panel F of the sweep: smoother bandwidth (bw=0.55), gridn=200, peak-normalized
        _gn = 200
        _xs = np.linspace(*xlim, _gn); _ys = np.linspace(*ylim, _gn)
        XX, YY = np.meshgrid(_xs, _ys)
        ZZ = gaussian_kde(np.vstack([gap, reward]), bw_method=0.55)(
            np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        Zn = ZZ / ZZ.max() if ZZ.max() > 0 else ZZ
        levels = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]

        fig, ax = plt.subplots(figsize=(7.4, 5.7))
        cs = ax.contour(XX, YY, Zn, levels=levels, cmap=reds_trunc, linewidths=1.3, zorder=3)
        cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.02, ticks=levels)
        cb.set_label("relative KDE density", fontsize=9)
        cb.ax.tick_params(labelsize=8)

        ax.plot([gt, gt], [st, ylim[1]], color="0.3", ls="--", lw=1.8, zorder=4)
        ax.plot([gt, xlim[1]], [st, st], color="0.3", ls="--", lw=1.8, zorder=4)

        # solid (thinner) early->late population-median shift arrow; dot labels live in the legend
        pA = (float(np.median(gap[em])), float(np.median(reward[em])))
        pB = (float(np.median(gap[lm])), float(np.median(reward[lm])))
        ax.annotate("", xy=pB, xytext=pA,
                    arrowprops=dict(arrowstyle="-|>", color="0.1", lw=1.8, mutation_scale=18,
                                    shrinkA=5, shrinkB=6), zorder=6)
        ax.scatter([pA[0]], [pA[1]], s=80, fc="#3b3b3b", ec="k", lw=1, zorder=7)
        ax.scatter([pB[0]], [pB[1]], s=80, fc="#2ca25f", ec="k", lw=1, zorder=7)

        # overlay externally-scored structures as gold NUMBERED stars (names go in the legend)
        has_marks = args.mark_csv is not None and Path(args.mark_csv).exists()
        mark_pts = []
        if has_marks:
            import csv as _csv
            import matplotlib.patheffects as pe
            rows = list(_csv.DictReader(open(args.mark_csv)))
            mark_pts = sorted(((float(r["gap"]), float(r["score"]), r.get("label", ""))
                               for r in rows), key=lambda t: t[0])
            if mark_pts:
                ax.scatter([p[0] for p in mark_pts], [p[1] for p in mark_pts], marker="*",
                           s=340, facecolor="gold", edgecolor="k", linewidths=1.0, zorder=8)
                for i, (x, y, lab) in enumerate(mark_pts, 1):
                    ax.annotate(str(i), (x, y), xytext=(7, 0), textcoords="offset points",
                                fontsize=9, fontweight="bold", zorder=9, ha="left", va="center",
                                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

        tgt = (gap > gt) & (reward > st)   # kept for the stdout summary only

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlabel("Predicted band gap (eV)")
        ax.set_ylabel(r"Reward $r_{\mathrm{uni}}$")
        leg = [Line2D([0], [0], marker="o", ls="", markerfacecolor="#3b3b3b",
                      markeredgecolor="k", markersize=9, label="first 10 (median)"),
               Line2D([0], [0], marker="o", ls="", markerfacecolor="#2ca25f",
                      markeredgecolor="k", markersize=9, label="last 10 (median)")]
        for i, (x, y, lab) in enumerate(mark_pts, 1):
            pretty = (f"{lab.split('_SG')[0]} (SG{lab.split('_SG')[1]})"
                      if "_SG" in lab else lab)
            leg.append(Line2D([0], [0], marker="*", ls="", markerfacecolor="gold",
                              markeredgecolor="k", markersize=12, label=f"{i}  {pretty}"))
        ax.legend(handles=leg, loc="lower right", fontsize=8.5, frameon=True, framealpha=0.9)

        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {run.name}: N={gap.size} over {len(steps)} loops (aggregate)  "
              f"shift median ({pA[0]:.2f},{pA[1]:.2f})->({pB[0]:.2f},{pB[1]:.2f})  "
              f"metals {_frac(em, ~pg):.0f}%->{_frac(lm, ~pg):.0f}%  "
              f"in-target {_frac(em, tgt):.0f}%->{_frac(lm, tgt):.0f}%")
        print(f"-> {out.with_suffix('.png')}")
        return

    # default: exclude below-gate (metallic) from the density; --include-below-gate keeps the
    # RAW full population (two separate panels then show the metallic pile shrink directly)
    plot_base = np.ones(len(gap), bool) if args.include_below_gate else pg

    winplot = (em | lm) & plot_base
    x_hi = (min(args.gap_max, float(np.percentile(gap[winplot], 99)) + 0.3)
            if winplot.any() else args.gap_max)
    x_lo = -0.1 if args.include_below_gate else (gate_min - 0.1)
    xlim, ylim = (x_lo, x_hi), (-0.02, 1.03)

    gt, st = args.gap_thr, args.score_thr
    tgt = (gap > gt) & (reward > st)

    def _frac(mask, cond):
        return 100 * np.mean(cond[mask]) if mask.sum() else 0.0
    in_e, in_l = _frac(em, tgt), _frac(lm, tgt)
    bg_e, bg_l = _frac(em, ~pg), _frac(lm, ~pg)
    bg_word = "below-gate (metallic)" if args.include_below_gate else "below-gate excluded"

    # two panels: early KDE (left) vs late KDE (right), shared axes for a direct comparison
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.3), sharex=True, sharey=True)
    centroids = {}
    for ax, m, cmap, line, letter, lbl, itf, bgf in (
            (axes[0], em, "Blues", "#1f5fa8", "a",
             f"early — loops {int(steps[0])}–{int(steps[n - 1])}", in_e, bg_e),
            (axes[1], lm, "Reds", "#c1272d", "b",
             f"late — loops {int(steps[-n])}–{int(steps[-1])}", in_l, bg_l)):
        mp = m & plot_base   # past-gate population, or the raw full set if --include-below-gate
        if mp.sum() >= 5:
            XX, YY, ZZ = _kde_grid(gap[mp], reward[mp], xlim, ylim)
            ax.contourf(XX, YY, ZZ, levels=[0.2, 0.4, 0.6, 0.8, 1.0], cmap=cmap,
                        alpha=0.55, zorder=2)
            ax.contour(XX, YY, ZZ, levels=[0.2, 0.4, 0.6, 0.8], colors=line,
                       linewidths=0.8, zorder=3)
            cx, cy = float(np.median(gap[mp])), float(np.median(reward[mp]))
            ax.scatter([cx], [cy], s=60, facecolor=line, edgecolor="k", linewidths=1.0,
                       zorder=7)
            centroids[letter] = (cx, cy)
        # band-gap gate marker + dashed target rectangle (identical on both panels)
        ax.axvline(gate_min, color="0.6", ls=":", lw=1.4, zorder=1)
        ax.plot([gt, gt], [st, ylim[1]], color="0.3", ls="--", lw=2.0, zorder=5)
        ax.plot([gt, xlim[1]], [st, st], color="0.3", ls="--", lw=2.0, zorder=5)
        ax.text(0.035, 0.965,
                f"{bg_word}: {bgf:.0f}%\nin target: {itf:.0f}% of samples",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="0.2",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.8, alpha=0.9))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Predicted band gap $E_g$ (eV, E3NN)")
        ax.set_title(f"({letter}) {lbl}", fontsize=10)
    axes[0].set_ylabel(r"Combined RL reward $r_{\mathrm{uni}}$")
    leg = [Line2D([0], [0], marker="o", ls="", markerfacecolor="0.4", markeredgecolor="k",
                  markersize=8, label="window median"),
           Line2D([0], [0], color="0.3", ls="--", lw=2.0,
                  label=fr"target: $E_g\!\geq\!{gt:g}$ & $r_{{\mathrm{{uni}}}}\!\geq\!{st:g}$"),
           Line2D([0], [0], color="0.6", ls=":", lw=1.4,
                  label=fr"band-gap gate $E_g\!=\!{gate_min:g}$")]
    axes[1].legend(handles=leg, loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    fig.suptitle(f"{label} — RL reward drift in the (band gap, reward) plane: "
                 f"first {n} vs last {n} loops", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    med_e = centroids.get("a", (float("nan"),) * 2)[1]
    med_l = centroids.get("b", (float("nan"),) * 2)[1]
    print(f"  {run.name}: N={gap.size} over {len(steps)} steps  "
          f"past-gate median reward {med_e:.2f}->{med_l:.2f}  in-target {in_e:.0f}%->{in_l:.0f}%  "
          f"below-gate {bg_e:.0f}%->{bg_l:.0f}% (first/last {n})")
    print(f"-> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
