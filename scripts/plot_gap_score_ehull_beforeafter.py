#!/usr/bin/env python3
"""Before/after (first-N vs last-N RL steps) gap-vs-anisotropy-score-vs-Ehull scatter for
layered-uniaxial dielectric runs -- works for BOTH gap-gated and ungated runs.

For each run dir a 2-panel figure: x = predicted E3NN band gap, y = raw layered-uniaxial
anisotropy scalar (computed from the predicted eps tensor), colour = Ehull (clipped). Left =
first N steps ("before" RL steers), right = last N steps ("after"). The band-gap GATE ramp
[minv,maxv] (read from the run's reward config) is shaded; for an UNGATED run the same band is
drawn as a "DFPT-safe reference". A red box marks the both-good corner (score>thr AND gap>thr).

Gap + eps are read from deliverables_bandgap_filtered/scored_by_reward_with_bandgap.csv (so the
E3NN gaps are identical/consistent across runs -- run scripts/filter_static_dielectric_by_bandgap.py
first if it is missing). Ehull is matched from samples/step_*_stability.csv via extxyz total_energy.

Example:
  python scripts/plot_gap_score_ehull_beforeafter.py \\
      exp_res/<ungated_run> exp_res/<gated_run> --n 10
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ase.io import read as ase_read

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+", help="one or more run directories")
    p.add_argument("--n", type=int, default=10, help="first-N vs last-N steps (default 10)")
    p.add_argument("--score-thr", type=float, default=0.2)
    p.add_argument("--gap-thr", type=float, default=0.5)
    p.add_argument("--ehull-clip", type=float, default=0.1)
    p.add_argument("--eps-cap", type=float, default=100.0, help="drop |eps|>cap blow-ups")
    p.add_argument("--gate", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="override the gate ramp (default: read band_gap minv/maxv from config)")
    return p.parse_args()


def detect_gate(run: Path):
    """Return (minv, maxv, gated) from the run's reward config (band_gap term)."""
    for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
        cfg_path = run / ".hydra" / name
        if not cfg_path.exists():
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(cfg, dict):   # e.g. overrides.yaml is a list
            continue
        props = (((cfg or {}).get("reward") or {}).get("prop_cfg")) or []
        for prop in props:
            if isinstance(prop, dict) and prop.get("name") == "band_gap":
                return float(prop.get("minv", 0.5)), float(prop.get("maxv", 1.2)), True
    return 0.5, 1.2, False   # ungated -> reference band


def aniso(exx, eyy, ezz):
    perp = 0.5 * (exx + eyy)
    lay = np.abs(ezz - perp) / (np.abs(ezz) + np.abs(perp) + 1e-8)
    mis = np.abs(exx - eyy) / (np.abs(exx) + np.abs(eyy) + 1e-8)
    return lay * (1.0 - mis)


def ehull_map(stab: Path) -> dict[float, float]:
    out: dict[float, float] = {}
    if not stab.exists():
        return out
    for r in csv.DictReader(stab.open()):
        try:
            out[round(float(r["total_energy_ev"]), 3)] = float(r["energy_above_hull_ev_per_atom"])
        except (TypeError, ValueError, KeyError):
            pass
    return out


def load_window(run: Path, df: pd.DataFrame, steps, eps_cap: float) -> dict[str, np.ndarray]:
    g, s, e = [], [], []
    by_step = {st: d for st, d in df.groupby("step")}
    for step in steps:
        if step not in by_step:
            continue
        ext = run / "samples" / f"step_{int(step):04d}_eval.extxyz"
        if not ext.exists():
            continue
        atoms = ase_read(ext, index=":")
        if not isinstance(atoms, list):
            atoms = [atoms]
        eh = ehull_map(run / "samples" / f"step_{int(step):04d}_stability.csv")
        for _, r in by_step[step].iterrows():
            if max(abs(r.eps_xx), abs(r.eps_yy), abs(r.eps_zz)) > eps_cap:
                continue
            i = int(r["index"])
            te = round(atoms[i].info.get("total_energy", 1e9), 3) if i < len(atoms) else 1e9
            g.append(max(float(r.band_gap_ev), 0.0))
            s.append(float(aniso(r.eps_xx, r.eps_yy, r.eps_zz)))
            e.append(eh.get(te, np.nan))
    return {"gap": np.array(g), "score": np.array(s), "ehull": np.array(e)}


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    style = ROOT / "publication.mplstyle"
    if style.exists():
        plt.style.use(str(style))

    N, SCORE_THR, GAP_THR, CLIP = args.n, args.score_thr, args.gap_thr, args.ehull_clip
    for rdir in args.run_dirs:
        rdir = rdir.resolve()
        csvp = rdir / "deliverables_bandgap_filtered" / "scored_by_reward_with_bandgap.csv"
        if not csvp.exists():
            print(f"  {rdir.name}: missing {csvp.name} -- run filter_static_dielectric_by_bandgap.py "
                  f"first; skipping")
            continue
        gmin, gmax, gated = (args.gate[0], args.gate[1], True) if args.gate else detect_gate(rdir)
        df = pd.read_csv(csvp)
        df = df[np.isfinite(df["band_gap_ev"])]
        steps = sorted(df["step"].unique().tolist())
        first, last = steps[:N], steps[-N:]
        before = load_window(rdir, df, first, args.eps_cap)
        after = load_window(rdir, df, last, args.eps_cap)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
        smax = max(np.nanmax(before["score"]) if before["score"].size else 0.2,
                   np.nanmax(after["score"]) if after["score"].size else 0.2)
        gmax_x = min(4.0, max(np.nanmax(before["gap"]) if before["gap"].size else 1,
                              np.nanmax(after["gap"]) if after["gap"].size else 1) + 0.2)
        gate_lbl = f"gate ramp [{gmin},{gmax}]" if gated else f"gap≥{gmin} (DFPT-safe ref)"
        pass_lbl = "past gate" if gated else "non-metallic"
        sc = None
        for ax, win, lbl in ((axes[0], before, f"BEFORE — first {len(first)} steps"),
                             (axes[1], after, f"AFTER — last {len(last)} steps")):
            ax.axvspan(gmin, gmax, color="0.88", zorder=0, label=gate_lbl)
            ax.axvline(gmin, color="0.5", ls="--", lw=1, zorder=1)
            c = np.clip(win["ehull"], 0.0, CLIP)
            sc = ax.scatter(win["gap"], win["score"], c=c, cmap="viridis_r",
                            vmin=0, vmax=CLIP, s=22, alpha=0.8, linewidths=0)
            pct_pass = 100 * np.mean(win["gap"] >= gmin) if win["gap"].size else 0
            gd, sd = win["gap"], win["score"]
            good = (gd > GAP_THR) & (sd > SCORE_THR)
            ratio = 100 * np.mean(good) if gd.size else 0.0
            ax.axhline(SCORE_THR, color="firebrick", ls=":", lw=0.8, zorder=1)
            ax.add_patch(Rectangle((GAP_THR, SCORE_THR), gmax_x - GAP_THR, smax * 1.1 - SCORE_THR,
                                   facecolor="firebrick", alpha=0.07, edgecolor="firebrick",
                                   lw=1.0, zorder=1))
            ax.text(0.97, 0.80,
                    f"score>{SCORE_THR} &\ngap>{GAP_THR}:\n{ratio:.0f}% ({int(good.sum())}/{gd.size})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color="firebrick", fontweight="bold")
            ax.set_title(f"{lbl}\nN={win['gap'].size}  |  {pct_pass:.0f}% {pass_lbl} (gap≥{gmin})",
                         fontsize=9)
            ax.set_xlabel("Predicted band gap $E_g$ (eV, E3NN)")
            ax.set_xlim(-0.2, gmax_x)
            ax.set_ylim(-0.02, smax * 1.1)
            ax.legend(loc="upper left", fontsize=7, frameon=False)
        axes[0].set_ylabel("Layered-uniaxial dielectric score")
        cbar = fig.colorbar(sc, ax=list(axes), fraction=0.046, pad=0.02)
        cbar.set_label(f"$E_\\mathrm{{hull}}$ (eV/atom, clipped {CLIP})")
        tag = "gap-gated" if gated else "ungated (no gap term)"
        fig.suptitle(f"{rdir.name.split('_')[-1]} {tag} — gap vs anisotropy score vs stability, "
                     f"before vs after RL", fontsize=11)
        out = rdir / "deliverables_layered_uniaxial"
        out.mkdir(parents=True, exist_ok=True)
        for extn in ("png", "pdf"):
            fig.savefig(out / f"gap_score_ehull_first{N}_vs_last{N}.{extn}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)

        def gr(w):
            return 100 * np.mean((w["gap"] > GAP_THR) & (w["score"] > SCORE_THR)) if w["gap"].size else 0
        print(f"  {rdir.name}: {tag}  N {before['gap'].size}->{after['gap'].size}  "
              f"gap>={gmin} {100*np.mean(before['gap']>=gmin):.0f}%->{100*np.mean(after['gap']>=gmin):.0f}%  "
              f"both-good {gr(before):.0f}%->{gr(after):.0f}%  -> {out.name}/gap_score_ehull_first{N}_vs_last{N}.png")


if __name__ == "__main__":
    main()
