#!/usr/bin/env python3
"""Composite "story" figure for an SLME RL run.

Combines three panels into one publication figure:
  (a) reward mean vs RL step, W-step moving average +/- moving std, with the
      EARLY and STEERED(best) windows boxed in red and arrows pointing to (b);
  (b) prior->finetuned (band gap, eta) distribution shift, two panels (Early vs
      Steered), with the SQ-limit dome "target region" (reuses plot_distribution_shift);
  (c) TSENN/OptiMate predicted absorption of a few DIVERSE good structures, each
      drawn as a crystal-structure inset inside the absorption panel.

Usage:
  python scripts/plot_run_story_composite.py \
      exp_res/tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1_54346667 \
      [--early-steps 0:10] [--steered-steps 94:104] [--n-structures 4] [--window 5]

Run from the project root so `rewards...` and sibling scripts import cleanly.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.ticker import PercentFormatter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reused building blocks (importing plot_distribution_shift also applies
# publication.mplstyle, keeping this figure consistent with the rest of the paper).
import plot_distribution_shift as pds
from plot_distribution_shift import (
    ETA_PLOT_MAX,
    _annotate_good_region,
    _plot_sq_limit,
    _sq_curve,
    load_bg_eta,
    parse_range,
)
from plot_good_structure_absorption import load_run_config
from plot_run import moving_average
from plot_tsenn_slme_results import (
    _latex_formula,
    _load_tsenn_thickness_um,
    _tsenn_slme_from_run_config,
)
from rewards.calculators.tsenn_slme.calc import (
    _absorption_coef_um_inv,
    _kk_eps1_from_eps2,
)

from ase.io import read as ase_read  # noqa: F401  (kept for parity / future use)
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.visualize.plot import plot_atoms
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors

HALOGENS = {"F", "Cl", "Br", "I"}
CURVE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
PANEL_LETTER_SIZE = 16  # single size for (a)/(b)/(c)
# One standardized font scheme for the whole composite (overrides the imported
# publication.mplstyle defaults so every panel matches).
FONT_RC = {
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path, default=None,
                   help="output PNG (default: <run>/deliverables/run_story_composite.png)")
    p.add_argument("--early-steps", type=str, default="0:10")
    p.add_argument("--steered-steps", type=str, default=None,
                   help="steered window 'lo:hi'; default = 10 steps ending at the max-MA reward step")
    p.add_argument("--n-structures", type=int, default=4)
    p.add_argument("--max-halide", type=int, default=2,
                   help="cap how many of the sampled structures may contain a halogen")
    p.add_argument("--window", type=int, default=5, help="moving-average / moving-std window")
    p.add_argument("--target-eta-min", type=float, default=0.25)
    return p.parse_args()


# ----------------------------------------------------------------------------- (a)
def _load_reward_series(run_dir: Path):
    """Return (steps, reward_mean) from metrics.csv."""
    path = run_dir / "metrics.csv"
    rows = list(csv.DictReader(open(path)))

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(name, "")))
            except ValueError:
                out.append(np.nan)
        return np.array(out)

    steps = col("step")
    if np.all(np.isnan(steps)):
        steps = np.arange(len(rows), dtype=float)
    return steps, col("reward mean")


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling std matching moving_average's centered window."""
    n = len(values)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = values[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.std()
    return out


def _plot_reward(ax, steps, reward, window, early_rng, steered_rng):
    finite = np.isfinite(reward)
    steps_f, reward_f = steps[finite], reward[finite]
    ma = moving_average(reward_f, window)
    sd = _rolling_std(reward_f, window)

    ax.plot(steps_f, reward_f, color="0.7", lw=0.8, label="reward mean (raw)", zorder=1)
    ax.plot(steps_f, ma, color="#1f6fb4", lw=2.0, label=f"moving avg (W={window})", zorder=3)
    ax.fill_between(steps_f, ma - sd, ma + sd, color="#9ecae1", alpha=0.45,
                    label=f"$\\pm$ moving std (W={window})", zorder=2)
    ax.set_xlabel("RL step")
    ax.set_ylabel("Reward mean")
    ax.set_title(f"Mean reward vs RL step — {window}-step moving average $\\pm$ moving std")
    # headroom so the legend sits above the curve, centered between the early/steered
    # red boxes (a single horizontal row clears both the left and right windows)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.18 * (yhi - ylo))
    ax.legend(loc="upper center", ncol=3, framealpha=0.9, fontsize=10,
              handlelength=1.4, columnspacing=1.0)

    # red windows for early / steered
    ylo, yhi = ax.get_ylim()
    for (lo, hi) in (early_rng, steered_rng):
        ax.add_patch(Rectangle((lo, ylo), hi - lo, yhi - ylo, fill=False,
                                edgecolor="#d62728", lw=1.8, zorder=5))
    return (early_rng, steered_rng)


# ----------------------------------------------------------------------------- (b)
def _frac_in_target(bg, eta, sq_x, sq_y, eta_min):
    if bg.size == 0:
        return 0.0
    cap = (np.interp(bg, sq_x, sq_y) if sq_y is not None
           else np.full_like(bg, ETA_PLOT_MAX))
    return 100.0 * ((eta > eta_min) & (eta <= cap)).mean()


def _plot_bg_eta_panel(ax, name, bg, eta, zmax, sq_x, sq_y, eta_min, show_ylabel):
    from scipy.stats import gaussian_kde
    if bg.size >= 5:
        try:
            z = gaussian_kde(np.vstack([bg, eta]))(np.vstack([bg, eta]))
        except Exception:  # noqa: BLE001
            z = np.ones(bg.size)
    else:
        z = np.ones(bg.size)
    _annotate_good_region(ax, label_loc="left", fontsize=10.5, label_dx=-0.45)  # +2; shifted left
    order = z.argsort()
    sc = ax.scatter(bg[order], eta[order], c=z[order], s=14, cmap="viridis",
                    vmin=0.0, vmax=zmax, edgecolors="none", zorder=2)
    _plot_sq_limit(ax, eg_max=4.0)
    # extend the band-gap range to 4.0 eV and add eta headroom so the upper-right
    # legend sits in empty space above/right of the target region.
    ax.set_xlim(-0.05, 4.0)  # slight negative start so BG~0 scatter clears the spine
    ax.set_xticks(np.arange(0.0, 4.01, 0.5))
    ax.set_ylim(0, 0.40)
    # η is stored as a fraction (0–0.40); show it as a percentage on the axis
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Band gap (eV)")
    if show_ylabel:
        ax.set_ylabel("SLME $\\eta$ (%)")
    ax.set_title(name, pad=10)
    frac = _frac_in_target(bg, eta, sq_x, sq_y, eta_min)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4c4c4c",
               markeredgecolor="none", markersize=5, label=f"N = {bg.size}"),
        Patch(facecolor="#fb9a99", edgecolor="#d7301f", alpha=0.20,
              label=f"In target region: {frac:.0f}%"),
        Line2D([0], [0], color="black", ls="--", lw=1.25, label="SQ limit"),
    ], loc="upper right", handlelength=1.1, handletextpad=0.5, borderaxespad=0.3)
    return sc


# ----------------------------------------------------------------------------- (c)
def _elset(formula: str) -> frozenset:
    return frozenset(re.findall(r"[A-Z][a-z]?", formula))


def _select_diverse(run_dir: Path, n: int, max_halide: int):
    """Pick n chemically diverse good structures from candidate_shortlist.csv."""
    path = run_dir / "deliverables" / "trajectory" / "candidate_shortlist.csv"
    import pandas as pd
    df = pd.read_csv(path).sort_values("eta", ascending=False)

    def greedy(require_new):
        chosen, seen, covered, n_hal = [], set(), set(), 0
        for _, r in df.iterrows():
            es = _elset(str(r["formula"]))
            if es in seen:
                continue
            is_hal = bool(es & HALOGENS)
            if is_hal and n_hal >= max_halide:
                continue
            if require_new and chosen and not (es - covered):
                continue
            chosen.append(r)
            seen.add(es)
            covered |= es
            n_hal += int(is_hal)
            if len(chosen) >= n:
                break
        return chosen

    chosen = greedy(require_new=True)
    if len(chosen) < n:  # relax the new-element rule if diversity was too strict
        have = {_elset(str(r["formula"])) for r in chosen}
        for r in greedy(require_new=False):
            if _elset(str(r["formula"])) not in have:
                chosen.append(r)
                have.add(_elset(str(r["formula"])))
            if len(chosen) >= n:
                break
    return chosen[:n]


def _format_sg(symbol: str) -> str:
    """Hermann-Mauguin symbol -> LaTeX mathtext: '-N' -> bar over N, '_N' -> subscript.
    e.g. 'R-3' -> R$\\bar{3}$, 'P2_1/c' -> P2$_1$/c, 'Fm-3m' -> Fm$\\bar{3}$m."""
    out, i = [], 0
    while i < len(symbol):
        ch = symbol[i]
        if ch == "-" and i + 1 < len(symbol) and symbol[i + 1].isdigit():
            out.append(r"\bar{%s}" % symbol[i + 1])
            i += 2
        elif ch == "_" and i + 1 < len(symbol) and symbol[i + 1].isdigit():
            out.append(r"_{%s}" % symbol[i + 1])
            i += 2
        else:
            out.append(ch)
            i += 1
    return r"$\mathrm{%s}$" % "".join(out)


def _reduced(formula: str) -> str:
    from pymatgen.core import Composition
    try:
        return Composition(formula).reduced_formula
    except Exception:  # noqa: BLE001
        return str(formula)


_EHULL_CACHE: dict = {}


def _ehull_lookup(run_dir: Path, step: int, formula: str, nsites: int | None = None):
    """e_hull (eV/atom) for a picked structure, from samples/step_<step>_stability.csv,
    matched by reduced formula (+ nsites when available). None if not found."""
    import pandas as pd
    key = (str(run_dir), int(step))
    if key not in _EHULL_CACHE:
        p = run_dir / "samples" / f"step_{int(step):04d}_stability.csv"
        _EHULL_CACHE[key] = pd.read_csv(p) if p.exists() else None
    df = _EHULL_CACHE[key]
    if df is None:
        return None
    target = _reduced(formula)
    sub = df[df["reduced_formula"].map(_reduced) == target]
    if nsites is not None and "nsites" in sub.columns:
        exact = sub[sub["nsites"] == nsites]
        if not exact.empty:
            sub = exact
    if sub.empty:
        return None
    return float(sub["energy_above_hull_ev_per_atom"].min())


def _symmetry_view(struct, symprec: float = 0.01):
    """Conventional standard cell + a rotation that looks straight down the principal
    symmetry axis + a space-group label. Reveals the symmetry better than the raw
    (P1-written) primitive cell viewed at an arbitrary tilt."""
    try:
        sga = SpacegroupAnalyzer(struct, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        system = sga.get_crystal_system()
        label = f"{_format_sg(sga.get_space_group_symbol())} (#{sga.get_space_group_number()})"
    except Exception:  # noqa: BLE001
        return struct, "30x,20y,0z", None
    # Look straight DOWN the principal axis: c (z) for tri/tetra/hex/cubic/ortho;
    # b for monoclinic (unique axis); a perspective tilt for triclinic (no axis).
    rotation = {
        "monoclinic": "-90x,0y,0z",
        "triclinic": "30x,20y,0z",
    }.get(system, "0x,0y,0z")
    return conv, rotation, label


def _render_structure(ax, struct, color, title=None, rotation="35x,25y,10z", radii=0.5):
    """Render a crystal structure into an existing axes, with colored border + title."""
    atoms = AseAtomsAdaptor.get_atoms(struct)
    try:
        plot_atoms(atoms, ax, rotation=rotation, radii=radii, show_unit_cell=2)
    except Exception:  # noqa: BLE001
        ax.text(0.5, 0.5, "(render failed)", ha="center", va="center", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.0)
    if title:
        ax.set_title(title, color=color, pad=2.0)


def _compute_absorption(calc, picks, thickness_um):
    structs = [Structure.from_str(str(r["cif"]), fmt="cif") for r in picks]
    energies_ev, eps2, valid = calc.tsenn.predict_epsilon2_iso(structs)
    eps2 = np.nan_to_num(eps2, nan=0.0)
    eps1 = _kk_eps1_from_eps2(energies_ev, eps2)
    alpha_um = _absorption_coef_um_inv(energies_ev, eps1, eps2)
    alpha_cm = alpha_um * 1.0e4
    absorptance = 1.0 - np.exp(-2.0 * np.clip(alpha_um, 0.0, None) * thickness_um)
    return structs, energies_ev, alpha_cm, absorptance, valid


def _plot_curves(ax_alpha, ax_abs, picks, energies_ev, alpha_cm, absorptance, valid, thickness_um):
    for i, r in enumerate(picks):
        if not bool(valid[i]):
            continue
        c = CURVE_COLORS[i % len(CURVE_COLORS)]
        lbl = (rf"${_latex_formula(str(r['formula']))}$ "
               rf"($E_g={r['band_gap']:.2f}$ eV, $\eta={100*r['eta']:.1f}\%$)")
        ax_alpha.plot(energies_ev, alpha_cm[i], color=c, lw=1.7, label=lbl)
        ax_abs.plot(energies_ev, absorptance[i], color=c, lw=1.8)
    ax_alpha.set_yscale("log")
    ax_alpha.set_ylabel(r"$\alpha(E)\ (\mathrm{cm}^{-1})$")
    ax_alpha.set_title(rf"TSENN/OptiMate absorption ($L={thickness_um:g}\,\mu$m)")
    ax_alpha.legend(loc="lower right")
    ax_abs.set_ylabel(r"Absorptance $A(E)=1-e^{-2\alpha(E)L}$")
    ax_abs.set_xlabel("Photon energy $E$ (eV)")
    ax_abs.set_xlim(0, 5)  # spectra are ~identical beyond 5 eV; truncate to save width
    ax_abs.set_ylim(0, 1.02)


def _element_sublegend(ax, struct, border_color):
    """Per-structure element sub-legend (to the right of its render): one colored
    swatch per element in THIS structure, using the SAME jmol colors ASE renders."""
    ax.axis("off")
    els = sorted({str(el) for el in struct.composition.elements},
                 key=lambda s: atomic_numbers[s])
    handles = [Line2D([0], [0], marker="o", color="none", markersize=8,
                      markerfacecolor=jmol_colors[atomic_numbers[s]],
                      markeredgecolor="0.35", label=s) for s in els]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.28, 0.5),
              frameon=False, handletextpad=0.2, labelspacing=0.35, borderpad=0.1,
              borderaxespad=0.0)


def _plot_structure_column(struct_axes, legend_axes, structs, picks, valid, run_dir):
    for i, (ax, lax) in enumerate(zip(struct_axes, legend_axes)):
        if i >= len(structs) or not bool(valid[i]):
            ax.axis("off")
            lax.axis("off")
            continue
        c = CURVE_COLORS[i % len(CURVE_COLORS)]
        r = picks[i]
        conv, rotation, sg_label = _symmetry_view(structs[i])
        eh = _ehull_lookup(run_dir, int(r["step"]), str(r["formula"]), nsites=len(structs[i]))
        # line 1: formula + space group;  line 2: E_g (+ E_hull when found)
        head = rf"${_latex_formula(str(r['formula']))}$"
        if sg_label:
            head += f"  {sg_label}"
        info = rf"$E_g={r['band_gap']:.2f}$ eV"
        if eh is not None:
            info += "\n" + rf"$E_\mathrm{{hull}}={eh:.2f}$ eV/atom"
        _render_structure(ax, conv, c, title=f"{head}\n{info}", rotation=rotation, radii=0.5)
        _element_sublegend(lax, conv, c)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    plt.rcParams.update(FONT_RC)  # standardize all panel fonts
    pds.TARGET_ETA_MIN = float(args.target_eta_min)
    eta_min = pds.TARGET_ETA_MIN

    # (a) reward series + windows
    steps, reward = _load_reward_series(run_dir)
    early_rng = parse_range(args.early_steps)
    if args.steered_steps:
        steered_rng = parse_range(args.steered_steps)
    else:
        # center a window on the peak-MA reward step, with the SAME width as the
        # early window (so both summarize an equal number of RL steps).
        fin_steps = steps[np.isfinite(reward)]
        ma = moving_average(reward[np.isfinite(reward)], args.window)
        best_step = int(fin_steps[int(np.nanargmax(ma))])
        width = early_rng[1] - early_rng[0]          # e.g. 10
        half = width // 2                             # +/- 5
        lo, hi = best_step - half, best_step + (width - half)
        max_step = int(np.nanmax(fin_steps))
        if hi > max_step + 1:                         # clamp at the end, keep width
            lo -= hi - (max_step + 1)
            hi = max_step + 1
        steered_rng = (max(0, lo), hi)

    # (b) bg/eta for the two windows
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

    # (c) calculator + diverse structures + absorption
    cfg = load_run_config(run_dir)
    thickness_um = _load_tsenn_thickness_um(cfg)
    calc = _tsenn_slme_from_run_config(cfg)
    picks = _select_diverse(run_dir, args.n_structures, args.max_halide)
    structs, energies_ev, alpha_cm, absorptance, valid = _compute_absorption(
        calc, picks, thickness_um)
    n_struct = len(structs)

    # ---- layout: 3 columns | left (a reward + b scatter pair) | middle (structures) | right (spectra)
    fig = plt.figure(figsize=(19.0, 8.6))
    outer = GridSpec(1, 3, width_ratios=[1.25, 0.52, 0.62], wspace=0.05, figure=fig)
    left = outer[0].subgridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.22, wspace=0.24)
    # middle column: one row per structure, each split into [render | element sub-legend]
    mid = outer[1].subgridspec(n_struct, 2, width_ratios=[1.0, 0.32],
                               wspace=0.02, hspace=0.55)
    right = outer[2].subgridspec(2, 1, height_ratios=[1.05, 0.95], hspace=0.10)

    ax_reward = fig.add_subplot(left[0, :])
    ax_early = fig.add_subplot(left[1, 0])
    ax_steered = fig.add_subplot(left[1, 1], sharex=ax_early, sharey=ax_early)
    struct_axes = [fig.add_subplot(mid[i, 0]) for i in range(n_struct)]
    legend_axes = [fig.add_subplot(mid[i, 1]) for i in range(n_struct)]
    ax_alpha = fig.add_subplot(right[0])
    ax_abs = fig.add_subplot(right[1], sharex=ax_alpha)

    _plot_reward(ax_reward, steps, reward, args.window, early_rng, steered_rng)
    sc = _plot_bg_eta_panel(ax_early, "Early", e_bg, e_eta, zmax, sq_x, sq_y, eta_min, True)
    _plot_bg_eta_panel(ax_steered, "Steered (best)", s_bg, s_eta, zmax, sq_x, sq_y, eta_min, False)
    fig.colorbar(sc, ax=[ax_early, ax_steered], label="KDE density", fraction=0.046, pad=0.02)
    _plot_structure_column(struct_axes, legend_axes, structs, picks, valid, run_dir)
    _plot_curves(ax_alpha, ax_abs, picks, energies_ev, alpha_cm, absorptance, valid, thickness_um)

    # short vertical RED arrows hanging straight down below each window box
    # (axis-anchored data coords -> robust to the tight bbox; same x = vertical).
    yr0, yr1 = ax_reward.get_ylim()
    span = yr1 - yr0
    for rng in (early_rng, steered_rng):
        cx = 0.5 * (rng[0] + rng[1])
        ax_reward.annotate("", xy=(cx, yr0 - 0.20 * span), xytext=(cx, yr0 - 0.09 * span),
                           xycoords="data", textcoords="data", annotation_clip=False,
                           arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.2,
                                           mutation_scale=18))

    # manuscript panel letters. (a)=reward, (b)=trajectory scatter (left column,
    # so (a)/(b) share a left edge and align); (c)=structures+spectra group, so
    # its letter sits on the first structure panel. Offset in points keeps the
    # letters consistently placed regardless of panel width.
    # panel letters in FIGURE coords for exact alignment: (a)/(b) share a left x
    # (vertical alignment, stacked left column); (a)/(c) share a top y (horizontal
    # alignment, top row). Placed left of each panel so they clear the titles.
    pa = ax_reward.get_position()
    pb = ax_early.get_position()
    pc = struct_axes[0].get_position()
    left_x = pa.x0 - 0.030          # common x for (a) and (b)
    top_y = pa.y1 + 0.05            # common y for (a) and (c)
    for x, y, s in [(left_x, top_y, "(a)"),
                    (left_x, pb.y1 + 0.010, "(b)"),
                    (pc.x0 - 0.035, top_y, "(c)")]:  # left into the margin, clear of titles
        fig.text(x, y, s, fontsize=PANEL_LETTER_SIZE, fontweight="bold",
                 ha="left", va="bottom")

    out = args.output or (run_dir / "deliverables" / "run_story_composite.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print("Sampled structures:", ", ".join(str(r["formula"]) for r in picks))


if __name__ == "__main__":
    main()
