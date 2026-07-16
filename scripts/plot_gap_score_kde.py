#!/usr/bin/env python3
"""All-steps gap-vs-anisotropy-score density scatter for layered-uniaxial dielectric runs.

The single-panel, KDE-coloured analogue of plot_gap_score_ehull_beforeafter.py: SAME axes
(x = predicted E3NN band gap, y = raw layered-uniaxial anisotropy scalar from the predicted
eps tensor), but AGGREGATED over every RL step and coloured by 2-D KDE density (z), exactly
like the SLME composite (b) panels. Reveals where the whole run's population concentrates in
the (gap, score) plane rather than a before/after snapshot.

Gap + eps are read straight from the run's on-disk per-step reward artifacts (no model re-run,
no CSV needed): gaps from rewards/bandgap/step_*.txt and the dielectric tensor from
rewards/<diel-reward>/step_*_tensor.npz (the subdir carrying *_tensor.npz), which are 1:1
aligned per step via the tensor's valid_mask.

Example:
  python scripts/plot_gap_score_kde.py exp_res/<run> [exp_res/<run2> ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from plot_gap_score_ehull_beforeafter import aniso, detect_gate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+", help="one or more run directories")
    p.add_argument("--mode", choices=["kde", "drift"], default="kde",
                   help="'kde' = all-steps density scatter; 'drift' = early-vs-late density "
                        "contours with a centroid drift arrow (the RL 'iterative move')")
    p.add_argument("--n", type=int, default=20,
                   help="drift mode: window size (first-N vs last-N steps), like the paper's "
                        "0-20 vs 220-240 loops")
    p.add_argument("--fom-level", type=float, default=0.2,
                   help="drift mode: iso-FoM (E_g * score) level for the target-region frontier")
    p.add_argument("--score-thr", type=float, default=0.2)
    p.add_argument("--gap-thr", type=float, default=0.5)
    p.add_argument("--eps-cap", type=float, default=100.0, help="drop |eps|>cap blow-ups")
    p.add_argument("--gap-max", type=float, default=4.0, help="x-axis upper limit (eV)")
    p.add_argument("--gate", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="override the gate ramp (default: read band_gap minv/maxv from config)")
    p.add_argument("--hide-gate", action="store_true",
                   help="omit the gate-ramp band/line (kde mode)")
    p.add_argument("--color", choices=["kde", "ehull"], default="kde",
                   help="kde mode z/colour: 2-D KDE density (default) or per-structure E_hull")
    p.add_argument("--ehull-clip", type=float, default=0.1,
                   help="--color ehull: clip the colour scale at this ehull (eV/atom)")
    p.add_argument("--target", choices=["fom", "rect"], default="fom",
                   help="target region: FoM hyperbola E_g*score>=fom-level (default), or "
                        "rectangle E_g>gap-thr & score>score-thr")
    p.add_argument("--mark-csv", type=Path, default=None,
                   help="overlay starred, labeled points (CSV columns: gap, score, label)")
    return p.parse_args()


def _overlay_marks(ax, path):
    """Overlay starred, labeled points from a CSV (columns gap, score, label). Returns a
    legend proxy handle, or None if no path/rows."""
    if path is None:
        return None
    import csv as _csv
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    rows = list(_csv.DictReader(open(path)))
    if not rows:
        return None
    xs = [float(r["gap"]) for r in rows]
    ys = [float(r["score"]) for r in rows]
    ax.scatter(xs, ys, marker="*", s=340, facecolor="gold", edgecolor="k", linewidths=1.0,
               zorder=7, label="DFPT candidate")
    for x, y, r in zip(xs, ys, rows):
        ax.annotate(r.get("label", ""), (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    return Line2D([0], [0], marker="*", color="none", markerfacecolor="gold",
                  markeredgecolor="k", markersize=15, label="DFPT candidate")


def _kde_grid(x, y, xlim, ylim, gridn=160):
    """Evaluate a 2-D gaussian KDE on a grid; return (XX, YY, ZZ) normalised to peak=1."""
    from scipy.stats import gaussian_kde
    xs = np.linspace(xlim[0], xlim[1], gridn)
    ys = np.linspace(ylim[0], ylim[1], gridn)
    XX, YY = np.meshgrid(xs, ys)
    try:
        ZZ = gaussian_kde(np.vstack([x, y]))(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    except Exception:  # noqa: BLE001
        ZZ = np.zeros_like(XX)
    m = ZZ.max()
    return XX, YY, (ZZ / m if m > 0 else ZZ)


def _fom_frontier(ax, xlim, ylim, C, label=None):
    """Dashed-grey iso-FoM frontier score = C / E_g (i.e. FoM = E_g * score = C); everything
    to its upper-right has FoM >= C. Returns True if any part is visible."""
    gx = np.linspace(1e-3, xlim[1], 600)
    sy = C / gx
    vis = (sy >= ylim[0]) & (sy <= ylim[1])
    if vis.any():
        ax.plot(gx[vis], sy[vis], color="0.35", ls="--", lw=2.2, zorder=5, label=label)
    return bool(vis.any())


def _draw_drift(ax, gap, score, step, args, gmin, gmax, gated):
    """Early-vs-late density contours (Fig-5a-style before->after) + a dashed-grey curve
    outlining the DFT target region (2-sigma ellipse of the late 'both-good' cluster)."""
    steps = np.unique(step)
    N = min(args.n, max(1, len(steps) // 2))
    early_s, late_s = set(steps[:N].tolist()), set(steps[-N:].tolist())
    em = np.isin(step, list(early_s))
    lm = np.isin(step, list(late_s))
    smax = max(float(np.nanmax(score)), 0.2)
    # crop to where the contoured (early+late) data actually lives -- the full gap/score range
    # is mostly empty.
    win = em | lm
    x_hi = min(args.gap_max, max(float(np.percentile(gap[win], 98)) + 0.2, gmax + 0.3))
    y_hi = min(smax * 1.1, max(float(np.percentile(score[win], 99)) + 0.05, args.score_thr * 1.5))
    xlim, ylim = (-0.1, x_hi), (-0.02, y_hi)

    levels = [0.2, 0.4, 0.6, 0.8]
    from matplotlib.lines import Line2D
    handles = []
    cf_early = cf_late = None
    for m, cmap, line, lbl in (
            (em, "Blues", "#1f5fa8", f"early (steps {int(steps[0])}–{int(steps[N-1])})"),
            (lm, "Reds", "#c1272d", f"late (steps {int(steps[-N])}–{int(steps[-1])})")):
        if m.sum() < 5:
            continue
        XX, YY, ZZ = _kde_grid(gap[m], score[m], xlim, ylim)
        cf = ax.contourf(XX, YY, ZZ, levels=levels + [1.0], cmap=cmap, alpha=0.45, zorder=2)
        ax.contour(XX, YY, ZZ, levels=levels, colors=line, linewidths=0.8, zorder=3)
        handles.append(Line2D([0], [0], color=line, lw=6, alpha=0.6, label=lbl))
        if m is em:
            cf_early = cf
        else:
            cf_late = cf

    # DFT target region: rectangle (E_g >= gap_thr & score >= score_thr, --target rect) OR the
    # iso-FoM frontier score = C / E_g (FoM = E_g*score >= C, --target fom; C = --fom-level).
    if args.target == "rect":
        GAP_THR, SCORE_THR = args.gap_thr, args.score_thr
        ax.plot([GAP_THR, GAP_THR], [SCORE_THR, ylim[1]], color="0.3", ls="--", lw=2.2, zorder=5)
        ax.plot([GAP_THR, xlim[1]], [SCORE_THR, SCORE_THR], color="0.3", ls="--", lw=2.2, zorder=5)
        handles.append(Line2D([0], [0], color="0.3", ls="--", lw=2.2,
                              label=fr"target: $E_g\!\geq\!{GAP_THR:g}$ & score$\,\geq\,{SCORE_THR:g}$"))
    else:
        C = args.fom_level
        if _fom_frontier(ax, xlim, ylim, C):
            handles.append(Line2D([0], [0], color="0.35", ls="--", lw=2.2,
                                  label=fr"target: $E_g\!\times\!$score $\geq$ {C:g}"))

    mh = _overlay_marks(ax, args.mark_csv)
    if mh is not None:
        handles.append(mh)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.9)

    # Two slim colour bars (early Blues inner + late Reds outer) sharing the SAME level ticks
    # [0.2..1.0]; blue=early/red=late is the legend. No per-bar titles (that cramped the text);
    # one clean shared "relative KDE density" label on the outer bar.
    _cbs = [cf for cf in (cf_early, cf_late) if cf is not None]
    for _i, _cf in enumerate(_cbs):
        _outer = _i == len(_cbs) - 1
        _cb = ax.figure.colorbar(
            _cf, ax=ax, fraction=0.046, pad=(0.04 if _i == 0 else 0.09),
            ticks=(levels + [1.0]),
        )
        _cb.ax.tick_params(labelsize=7)
        if _outer:
            _cb.set_label("KDE density", labelpad=32)

    def _bg(m):
        return 100 * np.mean((gap[m] > args.gap_thr) & (score[m] > args.score_thr)) if m.sum() else 0.0
    return N, _bg(em), _bg(lm)


def _diel_reward_dir(run: Path) -> Path | None:
    """The rewards/ subdir carrying the per-step dielectric tensors (*_tensor.npz)."""
    rew = run / "rewards"
    if not rew.exists():
        return None
    for d in sorted(rew.iterdir()):
        if d.is_dir() and next(d.glob("step_*_tensor.npz"), None) is not None:
            return d
    return None


def _ehull_by_energy(stab: Path) -> dict[float, float]:
    """Map round(total_energy_ev,3) -> energy_above_hull from a step stability CSV
    (same total-energy match used by plot_candidates_gap_score_ehull)."""
    import csv as _csv
    out: dict[float, float] = {}
    if not stab.exists():
        return out
    for r in _csv.DictReader(stab.open()):
        try:
            out[round(float(r["total_energy_ev"]), 3)] = float(
                r["energy_above_hull_ev_per_atom"])
        except (TypeError, ValueError, KeyError):
            pass
    return out


def load_all(run: Path, eps_cap: float):
    """Per-point (gap, score, ehull, step) over ALL steps from on-disk per-step artifacts.

    Reads the dielectric tensor (n,3,3) + valid_mask from <diel>/step_*_tensor.npz and the
    1:1-aligned E3NN gaps from rewards/bandgap/step_*.txt; keeps only valid, finite, within-cap
    entries. score = layered-uniaxial anisotropy scalar of the eps tensor (rewards.aniso). ehull
    is matched per structure via samples/step_*_eval.extxyz total_energy -> step_*_stability.csv
    (NaN where unavailable). Returns (gap, score, ehull, step) arrays + n_steps."""
    from ase.io import read as ase_read
    diel = _diel_reward_dir(run)
    bg_dir = run / "rewards" / "bandgap"
    samples = run / "samples"
    # gap source: in-loop rewards/bandgap/step_*.txt (gate runs) OR, for runs without an
    # in-loop gap term (e.g. uniform_anchor), the re-predicted E3NN gaps in the scored CSV.
    gap_csv = run / "deliverables_bandgap_filtered" / "scored_by_reward_with_bandgap.csv"
    use_csv = not bg_dir.exists()
    gap_lookup: dict[tuple[int, int], float] = {}
    if use_csv:
        if not gap_csv.exists():
            raise FileNotFoundError(
                f"{run.name}: no rewards/bandgap/ and no {gap_csv.name}; "
                f"run scripts/filter_static_dielectric_by_bandgap.py first")
        import pandas as pd
        _gdf = pd.read_csv(gap_csv)
        gap_lookup = {(int(r.step), int(r.index)): float(r.band_gap_ev)
                      for r in _gdf.itertuples()}
    if diel is None:
        raise FileNotFoundError(f"{run.name}: need a rewards/*/step_*_tensor.npz")
    g_all, s_all, e_all, st_all, n_steps = [], [], [], [], 0
    for tp in sorted(diel.glob("step_*_tensor.npz")):
        step = int(tp.name.split("_")[1])
        d = np.load(tp, allow_pickle=True)
        ten = d["tensor"]                          # (n,3,3)
        mask = d["valid_mask"] if "valid_mask" in d else np.ones(len(ten), bool)
        if use_csv:
            gaps = np.array([gap_lookup.get((step, i), np.nan) for i in range(len(ten))])
        else:
            bp = bg_dir / (tp.name.replace("_tensor.npz", ".txt"))
            if not bp.exists():
                continue
            gaps = np.array([float(x) for x in bp.read_text().split() if x.strip()])
        n = min(len(ten), len(mask), len(gaps))
        if n == 0:
            continue
        exx, eyy, ezz = ten[:n, 0, 0], ten[:n, 1, 1], ten[:n, 2, 2]
        gap = gaps[:n]
        ok = (mask[:n]
              & np.isfinite(gap) & np.isfinite(exx) & np.isfinite(eyy) & np.isfinite(ezz)
              & (np.maximum.reduce([np.abs(exx), np.abs(eyy), np.abs(ezz)]) <= eps_cap))
        if not ok.any():
            continue
        # ehull per structure (total-energy match; NaN if eval.extxyz/stability missing)
        eh = np.full(n, np.nan)
        ev = samples / f"step_{step:04d}_eval.extxyz"
        if ev.exists() and ev.stat().st_size:
            eh_map = _ehull_by_energy(samples / f"step_{step:04d}_stability.csv")
            atoms = ase_read(ev, index=":")
            if not isinstance(atoms, list):
                atoms = [atoms]
            for i in range(min(n, len(atoms))):
                te = round(float(atoms[i].info.get("total_energy", 1e9)), 3)
                eh[i] = eh_map.get(te, np.nan)
        g_all.append(np.maximum(gap[ok], 0.0))
        s_all.append(np.asarray(aniso(exx[ok], eyy[ok], ezz[ok]), dtype=float))
        e_all.append(eh[ok])
        st_all.append(np.full(int(ok.sum()), step))
        n_steps += 1
    if not g_all:
        return (np.array([]), np.array([]), np.array([]), np.array([]), 0)
    return (np.concatenate(g_all), np.concatenate(s_all), np.concatenate(e_all),
            np.concatenate(st_all), n_steps)


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    style = ROOT / "publication.mplstyle"
    if style.exists():
        plt.style.use(str(style))

    SCORE_THR, GAP_THR, CAP = args.score_thr, args.gap_thr, args.eps_cap
    for rdir in args.run_dirs:
        rdir = rdir.resolve()
        gmin, gmax, gated = (args.gate[0], args.gate[1], True) if args.gate else detect_gate(rdir)
        gap, score, ehull, step, n_steps = load_all(rdir, CAP)
        if gap.size == 0:
            print(f"  {rdir.name}: no finite-gap structures; skipping")
            continue

        smax = max(float(np.nanmax(score)), 0.2)
        gmax_x = min(args.gap_max, float(np.nanmax(gap)) + 0.2)
        runid = rdir.name.split("_")[-1]
        pass_lbl = "past gate" if gated else "non-metallic"
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        ax.set_xlabel("Predicted band gap $E_g$ (eV, E3NN)")
        ax.set_ylabel("Layered-uniaxial dielectric score")

        if args.mode == "drift":
            # _draw_drift owns the gate/threshold/box + the cropped axis limits
            N, r0, r1 = _draw_drift(ax, gap, score, step, args, gmin, gmax, gated)
            ax.set_title(f"{runid} — gap vs anisotropy score, RL drift (early vs late {N} steps)\n"
                         f"both-good (score>{SCORE_THR} & gap>{GAP_THR}): {r0:.0f}% → {r1:.0f}%",
                         fontsize=10)
            stem = "gap_score_drift"
            summary = f"drift both-good {r0:.0f}%->{r1:.0f}% (early/late {N} steps)"
        else:
            y_top = smax * 1.1
            # optional gate-ramp band (hidden with --hide-gate)
            if not args.hide_gate:
                ax.axvspan(gmin, gmax, color="0.88", zorder=0,
                           label=(f"gate ramp [{gmin},{gmax}]" if gated
                                  else f"gap≥{gmin} (DFPT-safe ref)"))
                ax.axvline(gmin, color="0.5", ls="--", lw=1, zorder=1)

            # z / colour: E_hull per structure, or 2-D KDE density (default)
            if args.color == "ehull":
                fin = np.isfinite(ehull)
                if (~fin).any():  # ehull-less points drawn faint grey underneath
                    ax.scatter(gap[~fin], score[~fin], c="0.8", s=14,
                               edgecolors="none", zorder=2)
                o = np.argsort(ehull[fin])[::-1]         # most-stable (low ehull) drawn last/on top
                gx, gy = gap[fin][o], score[fin][o]
                sc = ax.scatter(gx, gy, c=np.clip(ehull[fin][o], 0.0, args.ehull_clip),
                                cmap="viridis_r", vmin=0.0, vmax=args.ehull_clip, s=16,
                                edgecolors="none", zorder=3)
                cbar_label = fr"$E_\mathrm{{hull}}$ (eV/atom, clip {args.ehull_clip:g})"
            else:
                if gap.size >= 5:
                    try:
                        z = gaussian_kde(np.vstack([gap, score]))(np.vstack([gap, score]))
                    except Exception:  # noqa: BLE001
                        z = np.ones(gap.size)
                else:
                    z = np.ones(gap.size)
                o = z.argsort()
                sc = ax.scatter(gap[o], score[o], c=z[o], cmap="viridis", s=16,
                                edgecolors="none", zorder=3)
                cbar_label = "KDE density"

            # target region: rectangle E_g>gap_thr & score>score_thr, or FoM hyperbola
            if args.target == "rect":
                ax.fill_between([GAP_THR, gmax_x], SCORE_THR, y_top, color="0.6",
                                alpha=0.14, zorder=0)
                ax.plot([GAP_THR, GAP_THR], [SCORE_THR, y_top], color="0.3", ls="--",
                        lw=1.8, zorder=5)
                ax.plot([GAP_THR, gmax_x], [SCORE_THR, SCORE_THR], color="0.3", ls="--",
                        lw=1.8, zorder=5,
                        label=fr"target: $E_g>${GAP_THR:g} & score$>${SCORE_THR:g}")
                inside = (gap > GAP_THR) & (score > SCORE_THR)
                tgt_txt = fr"inside target ($E_g>${GAP_THR:g} & score$>${SCORE_THR:g}):"
                summary = f"inside rect(gap>{GAP_THR:g},score>{SCORE_THR:g}) "
            else:
                C = args.fom_level
                _fom_frontier(ax, (-0.2, gmax_x), (-0.02, y_top), C,
                              label=fr"target: $E_g\!\times\!$score $\geq$ {C:g}")
                inside = (gap * score) >= C
                tgt_txt = fr"inside target ($E_g\!\times\!$score $\geq$ {C:g}):"
                summary = f"inside FoM≥{C:g} "

            tgt_pct = 100 * np.mean(inside)
            summary += f"{tgt_pct:.0f}%"
            ax.text(0.97, 0.82, tgt_txt + "\n"
                    f"{tgt_pct:.0f}% ({int(inside.sum())}/{gap.size})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    color="0.2", fontweight="bold")
            ax.set_title(f"{runid} — gap vs anisotropy score (all {n_steps} steps)\n"
                         f"N={gap.size}", fontsize=10)
            _overlay_marks(ax, args.mark_csv)
            ax.legend(loc="upper left", fontsize=8, frameon=False)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cbar_label)
            ax.set_xlim(-0.2, gmax_x); ax.set_ylim(-0.02, y_top)
            stem = "gap_score_kde_allsteps"

        out = rdir / "deliverables_layered_uniaxial"
        out.mkdir(parents=True, exist_ok=True)
        for extn in ("png", "pdf"):
            fig.savefig(out / f"{stem}.{extn}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {rdir.name}: N={gap.size} over {n_steps} steps  {summary}  -> {out.name}/{stem}.png")


if __name__ == "__main__":
    main()
