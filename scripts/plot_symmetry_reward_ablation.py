#!/usr/bin/env python3
"""Publication ablation figure: "equal reward, unequal symmetry".

Under the SAME reward and RL settings, only the generator differs (SymmCD = ours,
symmetry-constrained; DiffCSP = MatInvent, unconstrained). Both reach the same
reward, but only SymmCD produces the rotational symmetry a real material should
possess (2/3/4/6-fold axes) -- and a post-hoc MLIP relaxation of the SAME
generated structures cannot give DiffCSP the high-order (4-/6-fold) axes it lacks.

  (a) reward mean vs RL step (moving average) -> SymmCD and DiffCSP score the same
  (b) highest rotational axis, strict symprec, BEFORE and AFTER MLIP relaxation of
      the generated structures -> SymmCD is rich and relaxation-stable; DiffCSP is
      P1 as-generated and only reaches 2-fold after relaxation (never 4-/6-fold)

Panel (b) reads the cached before/after space groups written by
`scripts/relax_symmetry_recovery.py` into <run>/deliverables/relax_recovery/records_*.csv.

Usage:
  python scripts/plot_symmetry_reward_ablation.py \
      exp_res/..._symmcd_v1_carryover_53154012 \
      exp_res/..._diffcsp_v1_54289718
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# importing plot_distribution_shift applies publication.mplstyle (paper-consistent)
import plot_distribution_shift  # noqa: F401
from plot_run import moving_average
from plot_symmetry_architecture import load_reward
from plot_eval_symmetry_inset import _hm

from ase.data.colors import jmol_colors
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

FS_BASE = plt.rcParams["font.size"]

FOLD_ORDER = ["None\n(P1/P$\\bar{1}$)", "2-fold", "3-fold", "4-fold", "6-fold", "cubic"]


def _lighten(c, f=0.58):
    """Blend a colour toward white (f in [0,1]) for the MLIP-relaxed variants."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


# colours from publication.mplstyle's prop_cycle (paper-consistent): SymmCD = the
# style green, DiffCSP = the style blue; a lighter tint marks the MLIP-relaxed set.
C_SYMM_GEN = "#1fb312"   # style green (ours)
C_DIFF_GEN = "#000dfc"   # style blue (DiffCSP)
C_SYMM_RLX = _lighten(C_SYMM_GEN)
C_DIFF_RLX = _lighten(C_DIFF_GEN)


def _fold(sg: int | None) -> str:
    """Highest rotational axis order of the principal axis, from the space-group #."""
    if sg is None or sg <= 2:      # triclinic: no rotation axis
        return FOLD_ORDER[0]
    if sg <= 74:                   # monoclinic + orthorhombic: 2-fold
        return "2-fold"
    if sg <= 142:                  # tetragonal: 4-fold
        return "4-fold"
    if sg <= 167:                  # trigonal: 3-fold
        return "3-fold"
    if sg <= 194:                  # hexagonal: 6-fold
        return "6-fold"
    return "cubic"                 # cubic: multi-axis (no unique uniaxial axis)


def _fold_dist(sgs) -> np.ndarray:
    n = len(sgs)
    c = Counter(_fold(int(s)) for s in sgs)
    return np.array([100.0 * c.get(k, 0) / n for k in FOLD_ORDER])


def _records(run_dir: Path, subdir: str):
    """before/after fold distributions from the cached relax_recovery records csv."""
    hits = sorted((run_dir / subdir).glob("records_*.csv"))
    if not hits:
        raise SystemExit(f"no records_*.csv in {run_dir / subdir} "
                         "(run scripts/relax_symmetry_recovery.py first)")
    df = pd.read_csv(hits[0])
    return _fold_dist(df["sg_before"]), _fold_dist(df["sg_after"]), len(df)


# ------------------------------------------------------------------ panel (c):
# WHY SymmCD reaches the high-order axes that DiffCSP cannot. SymmCD samples a
# space group + Wyckoff sites and REPLICATES the asymmetric unit by the group's
# operations. We show this on a REAL generated sample that possesses a 6-fold axis
# (WSe3S, SG 176 P6_3/m), decomposed exactly like paper_materials/
# symmcd_real_sample_decomp.py: (c) the asymmetric unit, (d) the 6-fold operation
# applied (one Wyckoff site fans into a regular hexagon), (e) the full cell.
#
# Key projection detail: view DOWN c (the 6-fold axis is c), in CARTESIAN, and
# recentre on the axis at fractional origin -- wrap frac to [-1/2,1/2) then
# `frac @ lattice.matrix`. This makes the 6h orbit a true regular hexagon; a naive
# fractional (a,b) projection would skew the 120 deg cell and hide the 6-fold.

ORBIT_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def load_sample(pt: Path, idx: int) -> Structure:
    s = torch.load(pt, map_location="cpu", weights_only=False)[idx]
    species = [Element.from_Z(int(z)).symbol for z in s["atom_types"].tolist()]
    L = s["lengths"][0].tolist(); A = s["angles"][0].tolist()
    return Structure(Lattice.from_parameters(*L, *A), species, s["frac_coords"].numpy())


def _jmol(symbol: str):
    return tuple(float(x) for x in jmol_colors[Element(symbol).Z])


def _decomp_data(struct: Structure, symprec: float):
    """Asymmetric unit + a LOCAL MOTIF viewed straight down the c (6-fold) axis.

    The motif is the set of atoms within rmax of the axis (at the fractional
    origin), built from periodic images and de-duplicated by keeping the FRONT
    atom (largest c-height) at each projected point -- exactly what you see looking
    down c in VESTA. This recovers the full coordination: the ring orbit forms its
    hexagon, and special-position sites complete their triangles around it.

    Returns: sg_num/sg_sym, orbits/wyckoff/site_syms (per Wyckoff orbit), and the
    motif arrays cxy (Nx2), corb (orbit index per motif atom), csp (species),
    gen (one motif index per orbit = the asymmetric-unit representative),
    ring (orbit index that forms the largest off-axis ring), and the unit cell."""
    spa = SpacegroupAnalyzer(struct, symprec=symprec)
    sym = spa.get_symmetrized_structure()
    ds = spa.get_symmetry_dataset()
    try:
        site_syms_all = ds["site_symmetry_symbols"]
    except (KeyError, TypeError):
        site_syms_all = getattr(ds, "site_symmetry_symbols", None)
    orbits = [list(g) for g in sym.equivalent_indices]
    wyck = list(sym.wyckoff_symbols)
    site_syms = ([site_syms_all[g[0]] for g in orbits]
                 if site_syms_all is not None else ["?"] * len(orbits))
    species = [s.specie.symbol for s in sym]
    orbit_of = {k: oi for oi, g in enumerate(orbits) for k in g}
    M = struct.lattice.matrix
    frnear = (sym.frac_coords + 0.5) % 1.0 - 0.5
    rad_cell = np.linalg.norm((frnear @ M)[:, :2], axis=1)
    rmax = max(rad_cell[g].mean() for g in orbits) * 1.06

    groups: dict = {}
    for k in range(len(sym)):
        for i in range(-2, 3):
            for j in range(-2, 3):
                fr = sym.frac_coords[k] + np.array([i, j, 0.0])
                xy = (fr @ M)[:2]
                if np.hypot(xy[0], xy[1]) <= rmax + 1e-6:
                    key = (round(float(xy[0]), 2), round(float(xy[1]), 2))
                    z = fr[2] % 1.0
                    if key not in groups or z > groups[key][1]:   # keep the FRONT atom
                        groups[key] = (xy, z, species[k], orbit_of[k])
    items = list(groups.values())
    cxy = np.array([t[0] for t in items])
    corb = np.array([t[3] for t in items])
    csp = [t[2] for t in items]
    radc = np.linalg.norm(cxy, axis=1)
    ang = np.arctan2(cxy[:, 1], cxy[:, 0])
    N = len(items)
    ring = max(range(len(orbits)),
               key=lambda oi: sum(1 for t in range(N) if corb[t] == oi and radc[t] > 0.3))

    def adist(x, y):
        return abs((x - y + np.pi) % (2 * np.pi) - np.pi)

    # Build the 60-deg fundamental wedge so its two outer CORNERS land on one atom
    # of each off-axis non-ring orbit (e.g. Cr and W), which are 60 deg apart, with
    # the ring atom (Te 6h) inside. Generators = those two corner atoms + that ring
    # atom. Atoms stay at their real positions -- the wedge is drawn around them.
    off = [t for t in range(N) if radc[t] > 0.3 and corb[t] != ring]
    corona_orbits = sorted(set(int(corb[t]) for t in off))
    gen = [None] * len(orbits)
    wedge = None
    if len(corona_orbits) >= 2:
        A_ts = [t for t in off if corb[t] == corona_orbits[0]]
        B_ts = [t for t in off if corb[t] == corona_orbits[1]]
        sc, ta, tb = min(((adist(ang[p], ang[q]) - np.pi / 3, p, q)
                          for p in A_ts for q in B_ts), key=lambda z: abs(z[0]))
        if abs(sc) < 0.3:                       # a ~60-deg corona pair exists
            wedge = (ta, tb)
    if wedge is not None:
        ta, tb = wedge
        gen[corb[ta]] = ta
        gen[corb[tb]] = tb
        mid = np.arctan2(np.sin(ang[ta]) + np.sin(ang[tb]),
                         np.cos(ang[ta]) + np.cos(ang[tb]))
    else:
        mid = float(ang[next(t for t in range(N) if corb[t] == ring)])
    for oi in range(len(orbits)):               # remaining orbits: rep nearest mid
        if gen[oi] is None:
            cand = [t for t in range(N) if corb[t] == oi]
            gen[oi] = min(cand, key=lambda t: adist(ang[t], mid)) if cand else None
    a, b = M[0, :2], M[1, :2]
    cell = np.array([(a + b) / 2, (a - b) / 2, -(a + b) / 2, (b - a) / 2])
    return dict(sg_num=spa.get_space_group_number(), sg_sym=spa.get_space_group_symbol(),
                orbits=orbits, wyckoff=wyck, site_syms=site_syms,
                cxy=cxy, corb=corb, csp=csp, gen=gen, ring=ring, wedge=wedge, cell=cell,
                rmax=rmax)


def _project_motif_xy(struct: Structure, rmax: float):
    """Project a structure straight down c into the same axis-centred window used
    by _decomp_data (periodic images within rmax, front-atom de-dup). Returns
    cxy (Nx2) + csp (species). Used for the panel-(c) DiffCSP sample, which shares
    the SAME lattice/composition as the SymmCD crystal but places atoms freely."""
    M = struct.lattice.matrix
    species = [s.specie.symbol for s in struct]
    groups: dict = {}
    for k in range(len(struct)):
        for i in range(-2, 3):
            for j in range(-2, 3):
                fr = struct.frac_coords[k] + np.array([i, j, 0.0])
                xy = (fr @ M)[:2]
                if np.hypot(xy[0], xy[1]) <= rmax + 1e-6:
                    key = (round(float(xy[0]), 2), round(float(xy[1]), 2))
                    z = fr[2] % 1.0
                    if key not in groups or z > groups[key][1]:   # keep FRONT atom
                        groups[key] = (xy, z, species[k])
    items = list(groups.values())
    cxy = np.array([t[0] for t in items])
    csp = [t[2] for t in items]
    return cxy, csp


def _incell_xy(struct: Structure):
    """Project the actual UNIT-CELL contents straight down c, axis-centred (each
    atom mapped once into the rhombus via (frac+0.5)%1-0.5). For panel (c) this
    shows the literal DiffCSP-generated cell (no periodic-image duplication)."""
    M = struct.lattice.matrix
    frw = (struct.frac_coords + 0.5) % 1.0 - 0.5
    xy = (frw @ M)[:, :2]
    csp = [s.specie.symbol for s in struct]
    return xy, csp


# glossy-sphere atom rendering (base disc + lighter offset core + white specular highlight),
# matching the Fig-4 / crystals_nc look; used by both _atom (panels c-e) and _GlossyLegendHandler.
_GLOSS_CORE_LIGHTEN = 0.30   # how far the core is pulled toward white
_GLOSS_CORE_FRAC = 0.55      # core diameter as a fraction of the sphere
_GLOSS_CORE_ALPHA = 0.38
_GLOSS_SPEC_FRAC = 0.20      # white specular glint diameter fraction
_GLOSS_SPEC_ALPHA = 0.70
_GLOSS_CORE_OFF = (-0.26, 0.26)   # upper-left, in units of the sphere radius
_GLOSS_SPEC_OFF = (-0.32, 0.32)


def _gloss_core(color):
    import matplotlib.colors as _mc
    cr, cg, cb = _mc.to_rgb(color)
    f = _GLOSS_CORE_LIGHTEN
    return (cr + (1 - cr) * f, cg + (1 - cg) * f, cb + (1 - cb) * f)


def _atom(ax, xy, color, size=200, ec="black", lw=0.9, alpha=1.0, symbol=None):
    """Glossy sphere: base disc + lighter offset core + white specular highlight. Highlights are
    offset by a fraction of the marker radius via a points->inch transform so they track the sphere
    at any marker size."""
    from matplotlib.transforms import offset_copy
    ms = np.sqrt(size) * 1.5                      # marker diameter in points
    r = ms / 2.0

    def _off(fx, fy):
        return offset_copy(ax.transData, fig=ax.figure, x=fx * r / 72.0, y=fy * r / 72.0,
                           units="inches")
    core = _gloss_core(color)
    ax.plot(xy[0], xy[1], "o", ms=ms, mfc=color, mec=ec, mew=lw, alpha=alpha, zorder=5)
    ax.plot(xy[0], xy[1], "o", ms=ms * _GLOSS_CORE_FRAC, mfc=core, mec="none",
            alpha=_GLOSS_CORE_ALPHA * alpha, zorder=5.1, transform=_off(*_GLOSS_CORE_OFF))
    ax.plot(xy[0], xy[1], "o", ms=ms * _GLOSS_SPEC_FRAC, mfc="white", mec="none",
            alpha=_GLOSS_SPEC_ALPHA * alpha, zorder=5.2, transform=_off(*_GLOSS_SPEC_OFF))
    if symbol is not None:
        ax.text(xy[0], xy[1], symbol, ha="center", va="center", fontsize=7,
                fontweight="bold", color="white", zorder=6)


class _GlossyLegendHandler(HandlerBase):
    """Draw a legend key marker as a glossy sphere (base + lighter core + specular), using the
    same recipe as _atom so the key matches the rendered atoms."""

    def __init__(self, color, **kw):
        super().__init__(**kw)
        self.color = color

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        cx, cy = width / 2.0 - xdescent, height / 2.0 - ydescent
        r = 0.62 * height
        core = _gloss_core(self.color)
        base = Circle((cx, cy), r, facecolor=self.color, edgecolor="black", lw=0.7, transform=trans)
        corec = Circle((cx + _GLOSS_CORE_OFF[0] * r, cy + _GLOSS_CORE_OFF[1] * r),
                       r * _GLOSS_CORE_FRAC, facecolor=core, edgecolor="none",
                       alpha=_GLOSS_CORE_ALPHA, transform=trans)
        spec = Circle((cx + _GLOSS_SPEC_OFF[0] * r, cy + _GLOSS_SPEC_OFF[1] * r),
                      r * _GLOSS_SPEC_FRAC, facecolor="white", edgecolor="none",
                      alpha=_GLOSS_SPEC_ALPHA, transform=trans)
        return [base, corec, spec]


def _draw_cell_poly(ax, cell, lw=1.0, color="0.45"):
    ax.add_patch(Polygon(cell, closed=True, fill=False, edgecolor=color, lw=lw,
                         ls=(0, (5, 3)), zorder=1))


def _style_decomp(ax, R, title, color="black", sub=None):
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=FS_BASE, pad=(15 if sub else 6), color=color)
    if sub:
        ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_BASE - 3, color="0.35")


def _axis_symbol(ax, n=6, faded=False, zorder=7):
    ax.plot([0], [0], marker=(n, 0, 0), ms=13, mfc="white",
            mec=("0.62" if faded else "0.1"), mew=(1.2 if faded else 1.5),
            zorder=zorder)


def _sym_tex(sym: str) -> str:
    """Hermann-Mauguin symbol (e.g. 'P6_3/m') -> mathtext with sub/bar, no number."""
    out, i = [], 0
    while i < len(sym):
        if sym[i] == "-" and i + 1 < len(sym) and sym[i + 1].isdigit():
            out.append(r"\bar{%s}" % sym[i + 1]); i += 2
        elif sym[i] == "_" and i + 1 < len(sym) and sym[i + 1].isdigit():
            out.append(r"_{%s}" % sym[i + 1]); i += 2
        else:
            out.append(sym[i]); i += 1
    return r"$\mathrm{%s}$" % "".join(out)


def _tag(ax, text):
    """Small boxed formula + space-group tag in a panel's lower-left corner."""
    ax.text(0.02, 0.02, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS_BASE - 4, zorder=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.6))


def _draw_decomp(ax_c, ax_d, ax_e, d, jmol_of, jitter=0.18, seed=0,
                 diffcsp_struct=None, label_c=None, label_e=None):
    cxy, corb, csp, gen = d["cxy"], d["corb"], d["csp"], d["gen"]
    orbits, ring, cell = d["orbits"], d["ring"], d["cell"]
    Rm = float(np.abs(cxy).max()) * 1.30                 # motif framing (c, d)
    Re = max(Rm, float(np.abs(cell).max()) * 1.08)       # cell framing (e)

    # --- (c) DiffCSP: the SAME composition + cell, but placed by free coordinates.
    # When a real DiffCSP CSP sample is supplied we project it down c into the same
    # window as (e); otherwise we fall back to jittering the motif. Either way there
    # is no Wyckoff sharing / enforced axis, so the symmetry is only approximate (P1).
    # Same frame + cell as (e) -> a direct apples-to-apples comparison.
    _style_decomp(ax_c, Re, "DiffCSP\ngenerated crystals", color=C_DIFF_GEN)
    _draw_cell_poly(ax_c, cell, lw=1.3, color="0.35")
    # the would-be axis (greyed), BEHIND the atoms (zorder<atoms' 5) since DiffCSP
    # has no real axis -> the free-coordinate atoms sit in front of it
    _axis_symbol(ax_c, faded=True, zorder=2)
    if diffcsp_struct is not None:                       # real DiffCSP coordinates
        dcxy, dcsp = _incell_xy(diffcsp_struct)           # the literal unit cell
        for t in range(len(dcxy)):
            _atom(ax_c, dcxy[t], jmol_of(dcsp[t]), size=120, lw=0.7)
    else:                                                # jitter fallback
        pert = cxy + np.random.default_rng(seed).normal(0.0, jitter, cxy.shape)
        for t in range(len(cxy)):
            _atom(ax_c, pert[t], jmol_of(csp[t]), size=120, lw=0.7)
    if label_c:
        _tag(ax_c, label_c)

    # --- (d) SymmCD: asymmetric unit + the symmetry operations on one panel ---
    _style_decomp(ax_d, Re, "Asymmetric unit with\nP6$_3$/m operations",
                  color="black")
    if d["wedge"] is not None:                           # 60-deg fundamental wedge
        ta, tb = d["wedge"]
        tri = np.array([[0.0, 0.0], cxy[ta], cxy[tb]])
    else:
        thg = np.arctan2(cxy[gen[ring]][1], cxy[gen[ring]][0])
        Rw = Rm * 0.92
        tri = np.array([[0.0, 0.0],
                        [Rw * np.cos(thg - np.pi / 6), Rw * np.sin(thg - np.pi / 6)],
                        [Rw * np.cos(thg + np.pi / 6), Rw * np.sin(thg + np.pi / 6)]])
    ax_d.add_patch(Polygon(tri, closed=True, facecolor="#ffe7a8",
                           edgecolor="#d9a93a", lw=1.0, zorder=0))
    for oi in range(len(orbits)):                        # dashed generator->image links
        col = ORBIT_PALETTE[oi % len(ORBIT_PALETTE)]
        gi = gen[oi]
        for t in np.where(corb == oi)[0]:
            ax_d.plot([cxy[gi, 0], cxy[t, 0]], [cxy[gi, 1], cxy[t, 1]], color=col,
                      lw=0.9, ls=(0, (4, 2)), zorder=2)
    for t in range(len(cxy)):                            # image atoms (small + ring)
        _atom(ax_d, cxy[t], jmol_of(csp[t]), size=85, lw=1.2,
              ec=ORBIT_PALETTE[corb[t] % len(ORBIT_PALETTE)])
    _axis_symbol(ax_d)
    for oi, gi in enumerate(gen):                        # asymmetric unit (bold) + label
        _atom(ax_d, cxy[gi], jmol_of(csp[gi]), size=230, lw=2.0, symbol=csp[gi],
              ec=ORBIT_PALETTE[oi % len(ORBIT_PALETTE)])
        lab = f"{d['wyckoff'][oi]} ({d['site_syms'][oi]})"
        ax_d.annotate(lab, xy=cxy[gi], xytext=(-18, 0), textcoords="offset points",
                      fontsize=6.5, color="#222", ha="right", va="center",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999",
                                lw=0.4), zorder=8)

    # --- (e) the resulting SymmCD crystal, with the unit cell drawn out -------
    _style_decomp(ax_e, Re, "SymmCD\ngenerated crystals", color=C_SYMM_GEN)
    _draw_cell_poly(ax_e, cell, lw=1.3, color="0.35")
    for t in range(len(cxy)):
        _atom(ax_e, cxy[t], jmol_of(csp[t]), size=150, lw=0.7)
    _axis_symbol(ax_e)
    if label_e:
        _tag(ax_e, label_e)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("symmcd_dir", type=Path)
    p.add_argument("diffcsp_dir", type=Path)
    p.add_argument("--labels", nargs=2, default=["SymmCD", "DiffCSP"])
    p.add_argument("--records-subdir", default="deliverables/relax_recovery")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--max-step", type=int, default=None,
                   help="crop the reward panel (a) x-axis to this RL step")
    # panel (c)/(d)/(e): real 6-fold sample decomposition
    p.add_argument("--decomp-pt", type=Path, default=None,
                   help="explicit step_*_eval.pt for the decomposition sample "
                        "(default: <symmcd_dir>/samples/step_<decomp-step>_eval.pt)")
    p.add_argument("--decomp-step", type=int, default=137,
                   help="step of the decomposition sample (default CrTe3W step 137)")
    p.add_argument("--decomp-index", type=int, default=19,
                   help="index within that step (default CrTe3W idx 19: 6-fold axis "
                        "EMPTY, Te 6h forms a regular hexagon around it)")
    p.add_argument("--decomp-symprec", type=float, default=0.1,
                   help="symprec for the decomposition's symmetry analysis")
    p.add_argument("--diffcsp-decomp-cif", type=Path, default=None,
                   help="panel (c): a REAL DiffCSP CSP sample (same composition + "
                        "cell as the showcase crystal, see scripts/diffcsp_csp_sample.py). "
                        "If given, panel (c) renders these real coordinates instead "
                        "of the jitter fallback")
    p.add_argument("--diffcsp-jitter", type=float, default=0.16,
                   help="panel (c) FALLBACK (no --diffcsp-decomp-cif): per-atom "
                        "free-coordinate jitter in Angstrom (approximate symmetry)")
    p.add_argument("--diffcsp-seed", type=int, default=0,
                   help="random seed for the panel-(c) DiffCSP jitter fallback")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _ema_debiased(x, w):
    """TensorBoard-style debiased exponential moving average (weight w in [0,1))."""
    out = np.empty(len(x), dtype=float); last = 0.0; deb = 0.0
    for i, v in enumerate(x):
        last = w * last + (1 - w) * v
        deb = w * deb + (1 - w)
        out[i] = last / deb
    return out


def main() -> None:
    args = parse_args()
    # publication.mplstyle is already applied (via the plot_distribution_shift import)
    symm_dir, diff_dir = args.symmcd_dir.resolve(), args.diffcsp_dir.resolve()
    symm_lab, diff_lab = args.labels

    # panel (a): reward of the two as-generated RL runs (relax=false)
    rewards = [(symm_lab, load_reward(symm_dir), C_SYMM_GEN),
               (diff_lab, load_reward(diff_dir), C_DIFF_GEN)]
    # panel (b): before/after MLIP relaxation of the generated structures
    s_gen, s_rlx, s_n = _records(symm_dir, args.records_subdir)
    d_gen, d_rlx, d_n = _records(diff_dir, args.records_subdir)
    series = [
        (f"{symm_lab}",            s_gen, C_SYMM_GEN),
        (f"{symm_lab} + MLIP-relaxed", s_rlx, C_SYMM_RLX),
        (f"{diff_lab}",            d_gen, C_DIFF_GEN),
        (f"{diff_lab} + MLIP-relaxed", d_rlx, C_DIFF_RLX),
    ]

    fig = plt.figure(figsize=(10.5, 6.7))
    outer = GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.44, figure=fig)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.27)
    bot = outer[1].subgridspec(1, 3, wspace=0.10)
    axr = fig.add_subplot(top[0])
    axb = fig.add_subplot(top[1])
    axc = fig.add_subplot(bot[0])
    axd = fig.add_subplot(bot[1])
    axe = fig.add_subplot(bot[2])

    # (a) reward vs step -- moving-average curves only, labelled by model
    rmax = 0.0
    for lab, (steps, rew), c in rewards:
        steps, rew = np.asarray(steps, float), np.asarray(rew, float)
        fin = np.isfinite(rew)
        ema = _ema_debiased(rew[fin], (args.window - 1) / (args.window + 1))  # span-window EMA
        axr.plot(steps[fin], ema, color=c, lw=1.8, label=lab)
        rmax = max(rmax, np.nanmax(rew))
    axr.set_xlabel("RL step")             # keep (a) at its default x-label height
    axr.set_ylabel("Reward mean")
    axr.set_title("Comparable reward", fontsize=FS_BASE)
    axr.legend(loc="lower right")
    axr.set_ylim(top=rmax * 1.08)
    axr.set_xlim(left=0, right=args.max_step)   # crop to --max-step if given
    if args.max_step is not None:
        axr.set_xticks([0, 40, 80, args.max_step])

    # (b) rotational-fold distribution, before/after MLIP relaxation
    x = np.arange(len(FOLD_ORDER))
    k = len(series)
    w = 0.82 / k
    for i, (lab, f, c) in enumerate(series):
        off = (i - (k - 1) / 2.0) * w
        bars = axb.bar(x + off, f, w, color=c, label=lab, edgecolor="white", linewidth=0.3)
        for b, v in zip(bars, f):
            if v >= 1.0:
                axb.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                             textcoords="offset points", xytext=(0, 1.4),
                             ha="center", va="bottom", fontsize=7)
    # bracket the high-order axes relaxation cannot give DiffCSP (4-/6-fold) with
    # neutral dashed boundaries instead of a green fill (keeps green = SymmCD only,
    # not the region). Solid grey dashed lines -> EPS/PS export stays clean.
    for xb in (2.5, 4.5):
        axb.axvline(xb, color="0.45", lw=1.2, ls=(0, (6, 3)), zorder=0)
    # white bbox masks the dashed line where it would otherwise cross the text
    axb.annotate("4-/6-fold:\nSymmCD only",
                 xy=(3.5, 52), ha="center", va="center", zorder=4,
                 color=C_SYMM_GEN, fontstyle="italic", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    axb.set_xticks(x)
    axb.set_xticklabels(FOLD_ORDER)
    axb.set_ylabel("% of generated crystals")
    axb.set_xlabel("Highest rotational axis")
    # (b)'s x-label is aligned to (a)'s "RL step" height after layout -- see the
    # _align_xlabel_to call just before savefig.
    axb.set_title("Rotational symmetry, before & after MLIP relaxation",
                  fontsize=FS_BASE)
    axb.set_ylim(0, 105)
    # opaque white background so the dashed region lines don't show through it
    axb.legend(loc="upper right", ncol=1, fontsize=FS_BASE - 3,  # match (c-e) legend
               frameon=True, facecolor="white", framealpha=1.0, edgecolor="0.8"
               ).set_zorder(6)

    # (c)/(d)/(e) SymmCD mechanism on a REAL 6-fold sample: asymmetric unit ->
    # 6-fold operation applied -> replicated cell (see paper_materials reference).
    decomp_pt = args.decomp_pt or (
        symm_dir / "samples" / f"step_{args.decomp_step:04d}_eval.pt")
    dstruct = load_sample(decomp_pt, args.decomp_index)
    d = _decomp_data(dstruct, args.decomp_symprec)
    print(f"  panel (c-e) sample: {dstruct.composition.reduced_formula}  "
          f"SG #{d['sg_num']} {d['sg_sym']}  orbits={[len(o) for o in d['orbits']]}")
    # formula (with subscripts) + space-group tags for panels (c) and (e)
    formula_tex = re.sub(r"(\d+)", r"$_{\1}$", dstruct.composition.reduced_formula)
    label_e = f"{formula_tex}\nSG {d['sg_num']} ({_sym_tex(d['sg_sym'])})"
    diffcsp_struct = None
    label_c = None
    if args.diffcsp_decomp_cif is not None:
        diffcsp_struct = Structure.from_file(str(args.diffcsp_decomp_cif))
        spa_c = SpacegroupAnalyzer(diffcsp_struct, symprec=args.decomp_symprec)
        sg_c, sym_c = spa_c.get_space_group_number(), spa_c.get_space_group_symbol()
        label_c = f"{formula_tex}\nSG {sg_c} ({_sym_tex(sym_c)})"
        print(f"  panel (c) DiffCSP sample: {diffcsp_struct.composition.reduced_formula} "
              f"SG {sg_c} {sym_c}  ({args.diffcsp_decomp_cif})")
    _draw_decomp(axc, axd, axe, d, _jmol, jitter=args.diffcsp_jitter,
                 seed=args.diffcsp_seed, diffcsp_struct=diffcsp_struct,
                 label_c=label_c, label_e=label_e)

    # bold panel letters (a)-(e), outside each box at the upper-left corner
    for ax, lab, dxy in [(axr, "(a)", (-34, 4)), (axb, "(b)", (-34, 4)),
                         (axc, "(c)", (-6, 8)), (axd, "(d)", (-6, 8)),
                         (axe, "(e)", (-6, 8))]:
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=dxy,
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=11, fontweight="bold", annotation_clip=False)

    # shared key for the decomposition row
    species_u = sorted(set(d["csp"]))
    handles = [Line2D([0], [0], marker="o", ls="", mfc=_jmol(sp), mec="black",
                      mew=0.7, ms=9, label=sp) for sp in species_u]
    handles += [
        Line2D([0], [0], marker="^", ls="", mfc="#ffe7a8", mec="#d9a93a",
               mew=1.0, ms=11, label="asymmetric unit (wedge)"),
        Line2D([0], [0], color="#666", lw=1.1, ls=(0, (4, 2)),
               label="symmetry-image link"),
        Line2D([0], [0], marker=(6, 0, 0), ls="", mfc="white", mec="0.1",
               mew=1.3, ms=11, label="6-fold axis"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=FS_BASE - 3, bbox_to_anchor=(0.5, 0.04))

    # Align (b)'s x-label to (a)'s "RL step" height. (a) keeps its default
    # position; (b) is raised to the SAME display-y (they share the row, so the
    # centred (b) label clears the far-left two-line "None (P1/P1̄)" tick).
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ext = axr.xaxis.label.get_window_extent(rend)
    y_disp = 0.5 * (ext.y0 + ext.y1)
    y_axb = axb.transAxes.inverted().transform((0, y_disp))[1]
    axb.xaxis.set_label_coords(0.5, y_axb)

    out = args.output or (symm_dir / "deliverables" / "symmcd_diffcsp_ablation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)  # dpi (600) + tight bbox come from publication.mplstyle
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".eps"))  # vector EPS for journal submission
    plt.close(fig)
    print(f"Wrote {out}  (SymmCD N={s_n}, DiffCSP N={d_n})")
    for lab, f, _ in series:
        print(f"  {lab}: " + ", ".join(
            f"{k.splitlines()[0]}:{v:.0f}%" for k, v in zip(FOLD_ORDER, f)))


if __name__ == "__main__":
    main()
