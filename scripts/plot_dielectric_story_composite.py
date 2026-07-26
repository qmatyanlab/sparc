#!/usr/bin/env python3
"""Composite "story" figure for a static-dielectric (layered-uniaxial) RL run.

The dielectric analogue of plot_run_story_composite.py (SLME). Three panels:
  (a) reward mean vs RL step, W-step moving average +/- moving std;
  (b) the OVERALL evaluated-set space-group distribution (fraction of structures),
      reusing plot_eval_symmetry_inset._draw_full;
  (c) a few chemically/symmetry-diverse high-reward structures, each rendered as a
      crystal inset paired with its PREDICTED 3x3 dielectric tensor (annotated
      matrix) -- the layered-uniaxial anisotropy is the eps_zz vs eps_xx=eps_yy split.

Usage:
  python scripts/plot_dielectric_story_composite.py \
      exp_res/tsenn_static_dielectric_layered_uniaxial_symmcd_v3_uniform_anchor_54304128 \
      [--n-structures 4] [--window 5] [--artifact eval]

Run from the project root so sibling scripts import cleanly.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Arc

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _PROJECT_ROOT / "scripts"
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Importing plot_eval_symmetry_inset applies publication.mplstyle (via its own
# plot_distribution_shift import) and gives us the full-distribution drawer + fonts.
import plot_eval_symmetry_inset as pei
from plot_eval_symmetry_inset import _draw_full, _draw_top, FS_BASE, FS_INSET_TITLE

# global amplification: every figure font is derived from FS_BASE, so bumping it here
# scales the whole composite by +FONT_BUMP.  Set FONT_BUMP=0 to retract to the original.
FONT_BUMP = 2
_FS_BASE0 = FS_BASE            # original base, kept for the dense reward eq (must not overflow)
FS_BASE = FS_BASE + FONT_BUMP

# one standard size for ALL schematic labels/captions/axis-text/ticks (panel a). The
# dense reward equation is the only deliberate exception (kept smaller so the box stays
# compact).  FS_BASE is 14 -> FS_LABEL is 11.
FS_LABEL = FS_BASE - 3
# shared standard for the RHS results panels (b)/(c): MAIN axis labels + titles at FS_AXIS,
# tick labels/legend/value-labels at FS_LABEL, insets at FS_LABEL -- so all three panels read
# at one consistent scale.
FS_AXIS = FS_BASE - 1
from generate_layered_uniaxial_deliverables import load_symmetry_counts
from plot_run import moving_average
from plot_layered_uniaxial_samples import load_step_records
from plot_tsenn_slme_results import _latex_formula

from itertools import product
from ase import Atoms
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from ase.visualize.plot import plot_atoms
from ase.io.utils import rotate as _ase_rotate
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path, default=None,
                   help="output path (default: <run>/deliverables_layered_uniaxial/"
                        "dielectric_story_composite.png)")
    p.add_argument("--n-structures", type=int, default=4,
                   help="number of top space groups to showcase in panel (c)")
    p.add_argument("--top", type=int, default=15, help="ranked SGs shown in panel (b)")
    p.add_argument("--ehull-max", type=float, default=0.1,
                   help="metastability cutoff (eV/atom) for panel (c) representatives")
    p.add_argument("--eps-max", type=float, default=100.0,
                   help="drop panel (c) candidates whose max predicted dielectric "
                        "component exceeds this (filters surrogate blow-ups)")
    p.add_argument("--exclude-sg", type=int, nargs="*", default=[],
                   help="space groups to drop from the panel (c) showcase (e.g. 225)")
    p.add_argument("--only-sg", type=int, nargs="*", default=None,
                   help="pin the panel (c) showcase to these space groups (overrides "
                        "the auto top-N); may repeat a SG if fewer than --n-structures")
    p.add_argument("--rank-by", default="sgpop", choices=["sgpop", "reward"],
                   help="panel (c): 'sgpop' = best of each top space group (by popularity); "
                        "'reward' = the highest-reward stable structures (distinct SGs)")
    p.add_argument("--window", type=int, default=10, help="EMA span for the reward curve")
    # NEW DEFAULT LAYOUT (2026-06): left = method schematic, right = reward (b) + SG bars (c)
    p.add_argument("--schematic-sg", type=int, default=None,
                   help="pin the schematic's example crystal to this space group "
                        "(default: auto-pick a high-reward, stable, uniaxial structure)")
    p.add_argument("--schematic-rank", type=int, default=0,
                   help="0=best, 1=2nd-best, ... among the schematic example candidates")
    p.add_argument("--schematic-formula", default="NbMoSe4",
                   help="pin the schematic example to this exact composition (default: "
                        "NbMoSe4); keeps the figure reproducible/consistent with the "
                        "mock-up as the eval set grows and the auto rank-0 pick drifts. "
                        "Pass an empty string to fall back to the auto rank-0 pick.")
    p.add_argument("--legacy-showcase", action="store_true",
                   help="restore the OLD layout (left a/b, right 2x2 structure showcase) "
                        "instead of the new schematic-left / results-right layout")
    p.add_argument("--best-window", type=int, default=0,
                   help="if >0, restrict panels (b)/(c) to this many RL steps centered "
                        "on the peak moving-average reward (boxed on panel a)")
    p.add_argument("--artifact", default="eval", choices=["eval", "valid"])
    p.add_argument("--split-panels", action="store_true",
                   help="emit panels (a)/(b)/(c) as SEPARATE png/pdf/eps (no letters), "
                        "cropped from the composite so they tile back; skips the combined "
                        "figure. NEW layout only.")
    p.add_argument("--sg-after-step", type=int, default=None,
                   help="adaptive-spacegroup step for the schematic 'after' histogram "
                        "(default: latest available step)")
    p.add_argument("--schematic-only", action="store_true",
                   help="FAST path: redraw ONLY panel (a) from the cached example "
                        "(deliverables/schematic_example.json) without reloading the full "
                        "eval set. Run once without it to populate the cache.")
    return p.parse_args()


# ----------------------------------------------------------------------------- (a)
def _load_reward_series(run_dir: Path):
    rows = list(csv.DictReader(open(run_dir / "metrics.csv")))

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


def _best_window(steps, reward, ma_window, width):
    """[lo, hi) RL-step window of `width` steps centered on the peak moving-average
    reward step (clamped to the run's step range). Returns (lo, hi, best_step)."""
    fin = np.isfinite(reward)
    steps_f, reward_f = steps[fin], reward[fin]
    ma = moving_average(reward_f, ma_window)
    best_step = int(steps_f[int(np.nanargmax(ma))])
    lo_step, hi_step = int(steps_f.min()), int(steps_f.max())
    half = width // 2
    lo, hi = best_step - half, best_step + (width - half)
    if hi > hi_step + 1:
        lo -= hi - (hi_step + 1); hi = hi_step + 1
    if lo < lo_step:
        hi += lo_step - lo; lo = lo_step
    return lo, min(hi, hi_step + 1), best_step


def _windowed_sg_counts(samples_dir: Path, artifact: str, steps):
    """Space-group counts from the eval/valid .pt payloads of just `steps`."""
    import torch
    from collections import Counter
    counts: Counter = Counter()
    for s in steps:
        p = samples_dir / f"step_{int(s):04d}_{artifact}.pt"
        if not p.exists():
            continue
        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict) and item.get("spacegroup") is not None:
                counts[int(item["spacegroup"])] += 1
    return counts


def _ema_debiased(x, w):
    """TensorBoard-style debiased exponential moving average (weight w in [0,1))."""
    out = np.empty(len(x), dtype=float); last = 0.0; deb = 0.0
    for i, v in enumerate(x):
        last = w * last + (1 - w) * v
        deb = w * deb + (1 - w)
        out[i] = last / deb
    return out


def _plot_reward(ax, steps, reward, window, box=None):
    # TensorBoard-style: faint raw trace + bold debiased EMA, no band (EMA is defined
    # at every point, so no truncated-window endpoint artifact)
    finite = np.isfinite(reward)
    steps_f, reward_f = steps[finite], reward[finite]
    ema = _ema_debiased(reward_f, (window - 1) / (window + 1))   # span-`window` decay
    ax.plot(steps_f, reward_f, color="0.78", lw=0.8, label="Reward mean (raw)", zorder=1)
    ax.plot(steps_f, ema, color="black", lw=2.0, label=f"EMA (span {window})", zorder=3)
    ax.set_xlabel("RL step")
    ax.set_ylabel("Reward mean")
    ax.set_xlim(steps_f.min(), steps_f.max() + 1)
    if box is not None:
        lo, hi = box
        ylo, yhi = ax.get_ylim()
        ax.add_patch(Rectangle((lo, ylo), hi - lo, yhi - ylo, fill=False,
                               edgecolor="#d62728", lw=1.8, zorder=5,
                               label=f"best {hi - lo} steps ({lo}–{hi - 1})"))
    ax.legend(loc="lower right", framealpha=0.9, handlelength=1.4)


# ----------------------------------------------------------------------------- (c)
_AXIS_COLORS = {"a": "#d62728", "b": "#2ca02c", "c": "#1f4fd6"}  # VESTA-ish a/b/c

# jmol gives Nb and Mo near-identical teal, so the two trilayers (NbSe2 vs MoSe2)
# would be indistinguishable; override to distinct VESTA-like hues (matches screenshot).
_ELEM_COLOR = {"Mo": (0.34, 0.71, 0.44), "Nb": (0.72, 0.53, 0.82)}  # green, purple


def _species_color(z):
    """RGB for atomic number `z`: a distinct override where jmol is ambiguous, else jmol."""
    return _ELEM_COLOR.get(chemical_symbols[int(z)], tuple(jmol_colors[int(z)]))


def _axis_triad(ax, cell, rotation, origin=(0.15, 0.15), length_in=0.40, dot_ms=11):
    """Draw a small a/b/c axis triad (VESTA-style) at `origin` (axes fraction) for the
    given lattice `cell` (rows a,b,c) under ASE rotation string `rotation`. Arrows have
    a fixed DISPLAY length `length_in` (inches) in every direction AND across panels --
    so equal-length lattice axes read as equal-length arrows regardless of the axes'
    aspect ratio (plain axes-fraction would stretch horizontal vs diagonal/vertical).
    The two most in-plane axes become arrows; the ~viewing axis becomes a dot (filled =
    toward viewer, x = away)."""
    R = _ase_rotate(rotation)
    proj = np.asarray(cell, dtype=float) @ R   # rows a,b,c -> (sx, sy, depth)
    ox, oy = origin
    fig = ax.figure
    pos = ax.get_position()
    win = max(pos.width * fig.get_figwidth(), 1e-6)    # axes width  (inches)
    hin = max(pos.height * fig.get_figheight(), 1e-6)  # axes height (inches)
    gap = 0.05                                          # label gap past the tip (inches)
    # arrows start at the EDGE of the central viewing-axis dot, not its centre, so the
    # dot's filled circle doesn't cover the arrow tails (dot radius pt->inch + clearance)
    r_in = 0.5 * dot_ms / 72.0 + 0.03
    for name, v in zip("abc", proj):
        sx, sy, depth = float(v[0]), float(v[1]), float(v[2])
        plen = np.hypot(sx, sy)
        vlen = float(np.linalg.norm(v)) + 1e-9
        col = _AXIS_COLORS[name]
        if plen / vlen < 0.32:                 # ~perpendicular to screen -> dot
            ax.plot([ox], [oy], marker="o", ms=dot_ms, mfc="white", mec=col, mew=2.0,
                    transform=ax.transAxes, zorder=10, clip_on=False)
            ax.plot([ox], [oy], marker=("." if depth >= 0 else "x"),
                    ms=(0.82 * dot_ms if depth >= 0 else 0.55 * dot_ms), color=col,
                    transform=ax.transAxes, zorder=11, clip_on=False)
            ax.text(ox, oy - 0.13 / hin, name, color=col, ha="center", va="top",
                    fontsize=FS_BASE - 4, fontweight="bold",
                    transform=ax.transAxes, zorder=11)
        else:
            ux, uy = sx / plen, sy / plen
            ax.annotate("", xy=(ox + length_in / win * ux, oy + length_in / hin * uy),
                        xytext=(ox + r_in / win * ux, oy + r_in / hin * uy),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                        mutation_scale=9),
                        zorder=10, annotation_clip=False)
            ax.text(ox + (length_in + gap) / win * ux, oy + (length_in + gap) / hin * uy,
                    name, color=col, ha="center", va="center", fontsize=FS_BASE - 4,
                    fontweight="bold", transform=ax.transAxes, zorder=11)


def _symmetry_view(struct, symprec: float = 0.1, tilt: float = 0.0, side: bool = False,
                   view: str | None = None):
    """Conventional standard cell + a viewing rotation/supercell.

    side=False (default): look straight down the principal symmetry axis so the
      IN-PLANE rotational symmetry is visible (tri/tetra/hex/cubic/ortho -> down c;
      monoclinic -> down b; triclinic -> generic tilt). Axial systems get a 2x2x1
      ab-tiling so the rotational tiling reads.
    side=True: a side perspective with c ~vertical and a c-extended slab, so the
      out-of-plane LAYER ALTERNATION (e.g. As/Se stacking) is visible."""
    try:
        sga = SpacegroupAnalyzer(struct, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        system = sga.get_crystal_system()
    except Exception:  # noqa: BLE001
        return struct, "35x,25y,10z"
    if view == "c":                  # look straight DOWN c (in-plane symmetry)
        rotation = "0x,0y,0z"        # a->right, b->120deg, c out of page
        reps = [2, 2, 1]
    elif view == "a":                # look straight DOWN a (layers stack along c, vertical)
        rotation = "90z,-90x"        # c->up, b->horizontal, a into page
        reps = [1, 2, 2]
    elif side:
        rotation = "-83x,8y,0z"      # c near-vertical (edge-on), slight azimuth for depth
        reps = [2, 1, 2]             # slab: 2 cells along a, 2 layers along c
    else:
        rotation = {
            "monoclinic": f"-{90 - tilt:g}x,0y,0z",
            "triclinic": "30x,20y,0z",
        }.get(system, f"{tilt:g}x,0y,0z")
        reps = [2, 2, 1] if system in {"hexagonal", "trigonal", "tetragonal"} else [1, 1, 1]
    conv = conv.copy()
    conv.make_supercell(reps)
    return conv, rotation


def _vesta_atoms(struct, view, margin: float = 0.45):
    """VESTA-style boundary view (the Atoms cell stays the single conventional cell,
    so plot_atoms draws ONE cell outline with the motif centred).

    view='c' (look DOWN c): the conventional cell PLUS periodic images within
      `margin` (fractional) in a,b, so corner atoms show full in-plane coordination.
    view='a' (look DOWN a, c vertical): an edge-on slab -- ONE c period (the layer
      stacking along c, +/- a small margin so the neighbouring layers peek in) tiled
      `nb` cells along b, so the layers -- which lie PERPENDICULAR to c -- read as
      wide horizontal sheets instead of the old one-atom-wide vertical sliver. `nb`
      is auto-sized so the tall c-axis slab roughly fills a portrait panel."""
    conv = SpacegroupAnalyzer(struct, symprec=0.01).get_conventional_standard_structure()
    atoms = AseAtomsAdaptor.get_atoms(conv)
    cell = np.array(atoms.cell)
    frac = atoms.get_scaled_positions(wrap=True)
    syms = atoms.get_chemical_symbols()
    out_f, out_s = [], []
    if view == "c":
        for shift in product((-1, 0, 1), repeat=3):
            if shift[2] != 0:                       # don't replicate the viewing axis c
                continue
            for f, s in zip(frac, syms):
                g = f + np.array(shift, dtype=float)
                if all(-margin <= g[d] <= 1.0 + margin for d in (0, 1)):
                    out_f.append(g); out_s.append(s)
    else:  # view == "a": VESTA-style edge-on view of ONE cell, c vertical. Expand the
           # two IN-VIEW-PLANE axes (b, c) by a boundary margin -- exactly like the
           # down-c view expands a, b -- so ~2 metals per layer show with their Se
           # coordination, instead of tiling many duplicate Nb/Mo columns. c is shifted
           # so the widest van-der-Waals gap lands on the cell edge, so the two trilayers
           # (e.g. NbSe2 + MoSe2) of one period render, not a periodic-image 3rd layer.
        z = np.sort(frac[:, 2] % 1.0)
        gaps = np.diff(np.concatenate([z, [z[0] + 1.0]]))
        k = int(np.argmax(gaps))
        zshift = (-(z[k] + 0.5 * gaps[k])) % 1.0
        bmarg, cmarg = 0.45, 0.04
        for ib in (-1, 0, 1):
            for ic in (-1, 0, 1):
                for f, s in zip(frac, syms):
                    gy = f[1] + ib
                    gz = (f[2] + zshift) % 1.0 + ic
                    if (-bmarg <= gy <= 1.0 + bmarg) and (-cmarg <= gz <= 1.0 + cmarg):
                        out_f.append([f[0], gy, gz]); out_s.append(s)
    pos = np.array(out_f) @ cell
    return Atoms(symbols=out_s, positions=pos, cell=cell, pbc=True)


def _render_structure(ax, struct, title, side: bool = False, view: str | None = None,
                      triad: bool = False, legend: bool = True):
    if view in ("c", "a"):
        # small down-c margin -> just the unit cell (corner metals completed + the
        # interior atoms), without the ring of extra boundary Se (margin unused for "a")
        atoms = _vesta_atoms(struct, view, margin=0.10)
        # spin about c so lattice vector a is horizontal (VESTA-style, any cell)
        a0 = np.array(atoms.cell)[0]
        zalign = -np.degrees(np.arctan2(a0[1], a0[0]))
        if view == "c":               # look down c (in-plane); +c toward viewer (dot OUT)
            rotation = f"0x,0y,{zalign:.2f}z"
        else:                         # look down a: c up, b horizontal, +a toward viewer
            # -90y,-90z (not 90y,90z) so the VIEWING axis a points OUT of the page like
            # c does in down-c -- both views use the VESTA "look down the +axis" handedness
            # (the only change vs into-page is a harmless left-right mirror of the slab).
            rotation = f"{zalign:.2f}z,-90y,-90z"
    else:
        struct, rotation = _symmetry_view(struct, side=side, view=view)
        atoms = AseAtomsAdaptor.get_atoms(struct)
    try:
        cols = np.array([_species_color(z) for z in atoms.numbers])
        plot_atoms(atoms, ax, rotation=rotation, radii=0.6, show_unit_cell=2, colors=cols)
    except Exception:  # noqa: BLE001
        ax.text(0.5, 0.5, "(render failed)", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([])
    # adjustable="datalim" keeps the axes BOX at its allocated size (the default
    # "box" shrinks it to the data aspect, which broke the triad's inch-based arrow
    # lengths and left no room for them); the data limits stretch instead.
    ax.set_aspect("equal", adjustable="datalim")
    # pad the data limits, with extra room at the BOTTOM for a horizontal species
    # legend below the structure (clear of the atoms regardless of cell shape).
    # side views (tall, narrow slabs) get tighter padding so they zoom in to fill
    # the panel rather than leaving it mostly blank.
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    rx, ry = x1 - x0, y1 - y0
    if view == "a":                            # zoom IN: tight crop so the slab fills it
        ml, mr, mb, mt = 0.03, 0.01, 0.16, 0.01  # (keep a bottom gap for the a/b/c triad)
    elif view == "c":                          # padding around the cell (kept a bottom gap for triad)
        ml, mr, mb, mt = 0.22, 0.14, 0.24, 0.12
    elif not legend:
        ml = mr = mb = mt = 0.08               # no per-view legend -> tight crop
    elif side:
        ml = mr = 0.06; mb, mt = 0.24, 0.05
    else:
        ml = mr = 0.18; mb, mt = 0.42, 0.12
    ax.set_xlim(x0 - ml * rx, x1 + mr * rx)
    ax.set_ylim(y0 - mb * ry, y1 + mt * ry)
    if triad:                       # drawn after the box/limits are final so the
        ax.apply_aspect()           # inch-based arrow lengths use the real axes box
        org = (0.08, 0.15) if view == "a" else (0.03, 0.08)
        _axis_triad(ax, np.array(atoms.cell), rotation, origin=org, length_in=0.42)
    for s in ax.spines.values():  # no box around the structure render
        s.set_visible(False)
    ax.set_title(title, fontsize=FS_BASE + 1, pad=3.0)
    if legend:
        # species legend: a horizontal row in the blank strip below the structure
        nums = sorted({int(n) for n in atoms.numbers})
        handles = [Line2D([0], [0], marker="o", ls="", markersize=7,
                          markerfacecolor=_species_color(n), markeredgecolor="0.3",
                          markeredgewidth=0.4, label=chemical_symbols[n]) for n in nums]
        leg = ax.legend(handles=handles, loc="lower center",
                        bbox_to_anchor=(0.5, 0.0), fontsize=FS_BASE + 1,
                        frameon=False, handletextpad=0.2, columnspacing=0.9,
                        borderpad=0.0, borderaxespad=0.0, ncol=len(nums))
        leg.set_in_layout(False)


def _tensor_matrix(row):
    """Reconstruct the symmetric predicted 3x3 dielectric tensor from a record."""
    xx, yy, zz = row["eps_xx"], row["eps_yy"], row["eps_zz"]
    xy, xz, yz = row["eps_xy"], row["eps_xz"], row["eps_yz"]
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)


def _plot_tensor(ax, M, fs=None, title_fs=None):
    """Annotated 3x3 dielectric-tensor matrix (per-matrix colour scale).
    `fs`/`title_fs` override the cell/label and title sizes (smaller in the compact
    schematic inset; default keeps the larger legacy-showcase sizing)."""
    fs = FS_BASE - 1 if fs is None else fs
    title_fs = FS_BASE + 1 if title_fs is None else title_fs
    im = ax.imshow(M, cmap="Blues", vmin=0.0, vmax=M.max())
    labels = ["x", "y", "z"]
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=fs)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=fs)
    ax.tick_params(length=0)
    thresh = 0.55 * M.max()
    for i in range(3):
        for j in range(3):
            label = f"{M[i, j]:.1f}".replace("-0.0", "0.0")
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=fs,
                    color="white" if M[i, j] > thresh else "0.15")
    ax.set_title(r"$\varepsilon_{\mathrm{pred}}$", fontsize=title_fs, pad=2.0)
    for s in ax.spines.values():
        s.set_visible(False)


def _best_in_sg(records, sg, ehull_max, used_ids):
    """Best representative in one space group: positive-definite and STABLE
    (e_hull <= ehull_max), then HIGHEST reward; fall back to most-stable, then to
    highest reward if no e_hull. Skips records already in `used_ids`. None if empty."""
    cand = [r for r in records
            if r.get("spacegroup") == int(sg) and bool(r["positive_definite"])
            and id(r) not in used_ids]
    if not cand:
        return None
    with_eh = [r for r in cand if r.get("ehull") is not None]
    if not with_eh:  # no stability data -> highest reward (e_hull shows as n/a)
        return max(cand, key=lambda r: float(r["reward"]))
    stable = [r for r in with_eh if r["ehull"] <= ehull_max]
    if stable:
        return max(stable, key=lambda r: float(r["reward"]))
    return min(with_eh, key=lambda r: r["ehull"])  # nothing stable -> most stable


def _pick_stable(records, allowed_sgs, n, ehull_max):
    """Up to `n` stable, high-reward representatives drawn from `allowed_sgs`
    (in order). One per space group first (diversity), then fill remaining slots
    by cycling back through the SGs for their next-best member -- so a single SG
    (e.g. --only-sg 187) can supply all n structures."""
    picks, used = [], set()
    for _ in range(n):  # at most n passes; each pass adds <=1 per SG
        added = False
        for sg in allowed_sgs:
            rep = _best_in_sg(records, sg, ehull_max, used)
            if rep is not None:
                picks.append(rep); used.add(id(rep)); added=True
                if len(picks) >= n:
                    return picks
        if not added:
            break
    return picks


def _pick_by_reward(records, allowed_sgs, n, ehull_max):
    """Up to `n` HIGHEST-reward stable structures from `allowed_sgs`, with DISTINCT
    formulas (so the same compound in two SGs isn't shown twice). Relaxes to
    distinct-formula-only, then to anything, if it can't fill n slots."""
    allowed = set(int(s) for s in allowed_sgs)
    pool = [r for r in records
            if r.get("spacegroup") in allowed and bool(r["positive_definite"])]
    stable = [r for r in pool if r.get("ehull") is not None and r["ehull"] <= ehull_max]
    use = sorted(stable or pool, key=lambda r: -float(r["reward"]))
    picks, seen_f, seen_sg, used = [], set(), set(), set()
    for r in use:  # pass 1: distinct formula AND distinct SG
        f, sg = str(r["formula"]), int(r["spacegroup"])
        if f in seen_f or sg in seen_sg:
            continue
        picks.append(r); seen_f.add(f); seen_sg.add(sg); used.add(id(r))
        if len(picks) >= n:
            return picks
    for r in use:  # pass 2: distinct formula only
        if id(r) in used or str(r["formula"]) in seen_f:
            continue
        picks.append(r); seen_f.add(str(r["formula"])); used.add(id(r))
        if len(picks) >= n:
            return picks
    for r in use:  # pass 3: fill with anything left
        if id(r) in used:
            continue
        picks.append(r)
        if len(picks) >= n:
            break
    return picks


def _plot_structure_cells(render_axes, matrix_axes, records, side_idx=()):
    for k, (rax, max_) in enumerate(zip(render_axes, matrix_axes)):
        if k >= len(records):
            rax.axis("off"); max_.axis("off"); continue
        row = records[k]
        struct = Structure.from_str(row["cif"], fmt="cif")
        eh = row.get("ehull")
        eh_txt = "n/a" if eh is None else f"{eh:.2f}"
        title = (rf"${_latex_formula(str(row['formula']))}$  SG {row['spacegroup']}"
                 f"\nscore {row['reward']:.2f} | $E_{{\\mathrm{{hull}}}}$ {eh_txt} eV/atom")
        _render_structure(rax, struct, title, side=(k in side_idx))
        _plot_tensor(max_, _tensor_matrix(row))


# --------------------------------------------------------------------- schematic (a)
# uniaxial point groups (a unique principal axis can host eps_zz != eps_in-plane):
# tetragonal (75-142), trigonal (143-167), hexagonal (168-194). Cubic is isotropic.
UNIAXIAL_SGS = set(range(75, 195))


def _pick_schematic_example(records, ehull_max, sg=None, rank=0, formula=None):
    """One real uniaxial, stable, high-reward structure to anchor the schematic.
    `formula` pins the example to an exact composition (e.g. 'NbMoSe4') so the figure
    stays consistent with the paper mock-up even as the eval set grows and the
    auto rank-0 pick drifts."""
    pool = [r for r in records if bool(r.get("positive_definite", True))]
    if sg is not None:
        pool = [r for r in pool if r.get("spacegroup") == int(sg)]
    else:
        pool = [r for r in pool if r.get("spacegroup") in UNIAXIAL_SGS]
    if formula:
        want = formula.replace(" ", "")
        pool = [r for r in pool if str(r["formula"]).replace(" ", "") == want]
    stable = [r for r in pool if r.get("ehull") is not None and r["ehull"] <= ehull_max]
    use = sorted(stable or pool, key=lambda r: -float(r["reward"]))
    if not use:
        return None
    return use[min(max(rank, 0), len(use) - 1)]


# ---- before/after space-group distributions for the schematic flow -------------
def _load_sg_anchor(run_dir: Path) -> np.ndarray:
    """The uniform space-group anchor the run STARTS from -- the `base_prob` column of the
    adaptive-SG snapshot (MP-20 prior tempered to near-uniform via sg_temperature, so no
    space group dominates).  Length-230 probability vector over SG 1..230."""
    path = Path(run_dir) / "adaptive_spacegroup" / "initial.csv"
    probs = np.zeros(230, dtype=float)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            sg = int(row["spacegroup"])
            if 1 <= sg <= 230:
                probs[sg - 1] = float(row["base_prob"])
    s = probs.sum()
    return probs / s if s > 0 else probs


# fields of the picked example needed to redraw the schematic without reloading the
# (slow) full eval set -- cached so `--schematic-only` iterations are instant.
_SCHEMATIC_CACHE_KEYS = ("formula", "spacegroup", "cif", "eps_xx", "eps_yy", "eps_zz",
                         "eps_xy", "eps_xz", "eps_yz", "reward", "ehull")


def _cache_schematic_example(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: rec.get(k) for k in _SCHEMATIC_CACHE_KEYS}))


def _load_reward_updated_sg_dist(run_dir: Path, step: int | None = None):
    """Reward-updated SG sampling policy as (step, length-230 prob vector).

    Reads the adaptive-spacegroup snapshot CSVs written during RL (column `current_prob`
    = the policy after reward reweighting); default = the latest available step."""
    policy_dir = Path(run_dir) / "adaptive_spacegroup"
    step_files = sorted(policy_dir.glob("step_*.csv"))
    if not step_files:
        raise FileNotFoundError(f"no adaptive_spacegroup/step_*.csv under {policy_dir}")
    path = step_files[-1] if step is None else policy_dir / f"step_{int(step):04d}.csv"
    used = int(path.stem.split("_")[1])
    probs = np.zeros(230, dtype=float)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            sg = int(row["spacegroup"])
            if 1 <= sg <= 230:
                probs[sg - 1] = float(row["current_prob"])
    s = probs.sum()
    return used, (probs / s if s > 0 else probs)


def _draw_sg_hist(ax, probs, color, title, show_xlabel=True, show_peak=True,
                  ylim_top=None, ylabel="probability"):
    """Compact space-group distribution: probability (y) vs SG number 1..230 (x).
    Solid colour (EPS-safe), names the dominant SG, minimal frame for an inset.
    `ylim_top` pins the y-axis ceiling (shared across panels for honest comparison)."""
    sg = np.arange(1, 231)
    ax.bar(sg, probs, width=1.8, color=color, edgecolor="none", zorder=3)
    ax.set_xlim(0, 231)
    ymax = max(float(np.max(probs)), 1e-6)
    if ylim_top is None:
        mag = 10 ** np.floor(np.log10(max(ymax, 1e-9)))
        nice_top = np.ceil(ymax / mag) * mag
    else:
        nice_top = ylim_top
    ax.set_ylim(0, nice_top * 1.15)
    ax.set_xticks([1, 115, 230])
    if show_peak:
        from pymatgen.symmetry.groups import SpaceGroup as _SG
        peak_sg = int(np.argmax(probs)) + 1
        sg_symbol = _SG.from_int_number(peak_sg).symbol
        peak_height = float(probs[peak_sg - 1])
        ax.annotate(f"{peak_sg} ({sg_symbol})",
                    xy=(peak_sg, peak_height),
                    xytext=(peak_sg + 28, nice_top * 0.72),
                    fontsize=FS_LABEL - 1, color=color, fontweight="bold",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                    mutation_scale=6),
                    clip_on=False)
    ax.tick_params(labelsize=FS_LABEL, length=2, pad=1)
    ax.set_yticks([0.0, nice_top])
    ax.set_yticklabels(["0", f"{nice_top:.3g}"])
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("0.45")
    if show_xlabel:
        ax.set_xlabel("space group $G$", fontsize=FS_LABEL, labelpad=1)
    if title:
        ax.set_title(title, fontsize=FS_LABEL, pad=2.5, color=color)


def _node(ax, cx, cy, w, h, text, fc, ec, fs):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.018",
                                facecolor=fc, edgecolor=ec, lw=1.8, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)


def _varrow(ax, x, y0, y1, label=None):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color="0.2", lw=2.0,
                                mutation_scale=18), zorder=2)
    if label:
        ax.text(x + 0.025, 0.5 * (y0 + y1), label, ha="left", va="center",
                fontsize=FS_LABEL, color="0.3")


def _inplane_rotation_glyph(ax, ox, oy, s=1.0, relation=None):
    """Vertical c-axis with a flattened (perspective) rotation ring ENCIRCLING it in the
    ab-plane, captioned '3, 4, 6-fold in-plane rotation' -- the n-fold axis along c that
    makes the in-plane dielectric response isotropic (eps_xx = eps_yy).
    `relation` is an optional LaTeX string placed below the caption (e.g. the anisotropy
    relation eps_zz >> eps_perp)."""
    import math
    L = 0.090 * s
    # vertical c-axis = the rotation axis (drawn through the ring so the ring wraps it)
    ax.annotate("", xy=(ox, oy + L), xytext=(ox, oy - 0.028 * s),
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.8,
                                mutation_scale=12), zorder=6)
    ax.text(ox + 0.018 * s, oy + L, "$c$", ha="left", va="center",
            fontsize=FS_LABEL, color="0.3", zorder=6)
    # in-plane rotation ring: a flattened ellipse (the ab-plane seen edge-on) centred on
    # the c-axis; back half dashed + front half solid gives the 3-D "wraps around" read
    cy = oy + 0.40 * L
    ew, eh = 0.135 * s, 0.050 * s
    ax.add_patch(Arc((ox, cy), ew, eh, angle=0, theta1=180, theta2=360,
                     color="0.3", lw=1.7, zorder=7))               # front half (solid)
    ax.add_patch(Arc((ox, cy), ew, eh, angle=0, theta1=18, theta2=180,
                     color="0.3", lw=1.4, linestyle=(0, (3, 2)), zorder=4))  # back (dash)
    # two in-plane arrows (a and b axes) radiating from ring centre at 0° and 120°
    for ang_deg in (0, 120):
        th_ip = math.radians(ang_deg)
        tip_x = ox + ew / 2 * math.cos(th_ip)
        tip_y = cy + eh / 2 * math.sin(th_ip)
        ax.annotate("", xy=(tip_x, tip_y), xytext=(ox, cy),
                    arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.5,
                                    mutation_scale=9), zorder=5)
    # arrowhead on the front arc (right end, ~ -22 deg), tangent in the sweep direction
    th = math.radians(-22)
    px, py = ox + ew / 2 * math.cos(th), cy + eh / 2 * math.sin(th)
    tx, ty = math.sin(th) * ew / 2, -math.cos(th) * eh / 2          # clockwise tangent
    nrm = math.hypot(tx, ty)
    ax.annotate("", xy=(px + 0.022 * s * tx / nrm, py + 0.022 * s * ty / nrm),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.7,
                                mutation_scale=11), zorder=7)
    cap_y = oy - 0.050 * s
    ax.text(ox, cap_y, "3, 4, 6-fold\nin-plane rotation",
            ha="center", va="top", fontsize=FS_LABEL, color="0.3")
    if relation:
        ax.text(ox, cap_y - 0.072 * s, relation,
                ha="center", va="top", fontsize=FS_LABEL, color="0.1")


def _detect_scalar_mode(run_dir) -> str:
    """The dielectric calculator's scalar_mode from the run config (hparams.yaml or
    .hydra/config.yaml), so the SAME script renders both the layered-uniaxial reward and
    the inplane-isotropy reward with the correct schematic formula + tensor relation.
    Defaults to 'layered_uniaxial' when nothing is found."""
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
        props = (((cfg or {}).get("reward") or {}).get("prop_cfg")) or []
        for prop in props:
            if not isinstance(prop, dict):
                continue
            calc = prop.get("calculator") or {}
            if "TSENNStaticDielectric" in str(calc.get("_target_", "")) or "scalar_mode" in calc:
                mode = calc.get("scalar_mode")
                if mode:
                    return str(mode)
    return "layered_uniaxial"


def _draw_schematic(fig, ax, record, run_dir, after_step=None):
    """Left-column method schematic as a single TOP->BOTTOM flow, integrated with REAL
    space-group data:
        MP-20 SG prior (before)  ->  diffusion model samples a crystal  ->  predicted
        eps tensor + in-plane rotation  ->  uniaxial-anisotropy reward  ->  reward-updated
        SG policy (after).
    The before/after histograms are the actual MP-20 prior and the adaptive-SG policy the
    reward converges to, so the panel literally shows the loop reshaping the space-group
    sampling distribution.  EPS-safe: solid colours, no alpha."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # shared column geometry: the blue generator card and the green reward card use the
    # SAME centre + width so their left/right edges line up (was: 0.375 vs 0.40 centres).
    BOX_CX, BOX_W = 0.40, 0.74          # x in [0.03, 0.77]
    colx = BOX_CX
    orange = "#ff7f0e"
    scalar_mode = _detect_scalar_mode(run_dir)

    # ----- BEFORE: the uniform space-group anchor (feeds the generator; no SG dominates)
    prior = _load_sg_anchor(run_dir)
    bax = ax.inset_axes([0.130, 0.892, 0.540, 0.072])
    _draw_sg_hist(bax, prior, "0.55", "initial SG prior $\\pi^0$",
                  show_xlabel=False, show_peak=False, ylabel="$\\pi^0$")
    _varrow(ax, colx, 0.866, 0.828, r"sample $G \sim \pi^0$")

    # ----- 1+2) generator card: model header + sampled crystal (two labelled views) --
    ax.add_patch(FancyBboxPatch((BOX_CX - BOX_W / 2, 0.520), BOX_W, 0.305,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor="none", edgecolor="k", lw=1.4,
                                zorder=2))
    struct = Structure.from_str(record["cif"], fmt="cif")
    sg_tex = f"${_latex_formula(str(record['formula']))}$  SG {record['spacegroup']}"
    if scalar_mode == "inplane_isotropy":
        # big down-c view on the LEFT; model title + formula + species key stacked on the RIGHT
        # so the crystal can fill the card (title/SG moved off the top).
        cax_c = ax.inset_axes([0.075, 0.540, 0.40, 0.285])
        _render_structure(cax_c, struct, "", view="c", triad=True, legend=False)
        rx = 0.645
        ax.text(rx, 0.775, "SG-conditioned\ndiffusion model\n$p_\\theta(\\mathcal{M}'|G)$",
                ha="center", va="center", fontsize=FS_LABEL, color="0.1",
                linespacing=1.35, zorder=4)
        ax.text(rx, 0.638, sg_tex, ha="center", va="center", fontsize=FS_LABEL, color="0.15")
        key_anchor, key_ncol = (rx, 0.572), 2
    else:
        # plain black title on white (no filled header band -- the outline is enough)
        ax.text(BOX_CX, 0.802,
                "SG-conditioned diffusion model  $p_\\theta(\\mathcal{M}'|G)$",
                ha="center", va="center", fontsize=FS_LABEL, color="0.1", zorder=4)
        ax.text(BOX_CX, 0.762, sg_tex, ha="center", va="bottom",
                fontsize=FS_LABEL, color="0.15")
        # down c: in-plane symmetry.  down a: layer stacking edge-on (PORTRAIT box).
        cax_c = ax.inset_axes([0.055, 0.582, 0.300, 0.150])
        _render_structure(cax_c, struct, "", view="c", triad=True, legend=False)
        cax_a = ax.inset_axes([0.452, 0.562, 0.300, 0.195])
        _render_structure(cax_a, struct, "", view="a", triad=True, legend=False)
        key_anchor, key_ncol = (colx, 0.537), None
    nums = sorted({int(n) for n in AseAtomsAdaptor.get_atoms(struct).numbers})
    handles = [Line2D([0], [0], marker="o", ls="", markersize=8,
                      markerfacecolor=_species_color(n), markeredgecolor="0.3",
                      markeredgewidth=0.5, label=chemical_symbols[n]) for n in nums]
    key = ax.legend(handles=handles, loc="center", bbox_to_anchor=key_anchor,
                    ncol=(key_ncol or len(nums)), frameon=False, fontsize=FS_LABEL,
                    handletextpad=0.2, columnspacing=0.8, borderpad=0.0,
                    borderaxespad=0.0)
    key.set_in_layout(False)
    _varrow(ax, colx, 0.508, 0.470, "predict $\\varepsilon$ (TSENN)")

    # ----- 3) predicted dielectric tensor + in-plane rotation glyph -----------------
    zz = float(record["eps_zz"])
    perp = 0.5 * (float(record["eps_xx"]) + float(record["eps_yy"]))
    tlabel = ("$\\varepsilon_{xx} = \\varepsilon_{yy}$" if scalar_mode == "inplane_isotropy"
              else "$\\varepsilon_{xx} = \\varepsilon_{yy} > \\varepsilon_{zz}$")
    tax = ax.inset_axes([0.305, 0.312, 0.215, 0.130])
    _plot_tensor(tax, _tensor_matrix(record), fs=FS_LABEL, title_fs=FS_LABEL)
    tax.set_xticks([])     # drop the bottom x/y/z labels so the reward arrow is clear
    # in-plane rotation symmetry glyph with the anisotropy relation below its caption
    _inplane_rotation_glyph(ax, 0.140, 0.400, s=0.95, relation=tlabel)
    _varrow(ax, colx, 0.300, 0.262, r"compute reward $r(\mathcal{M}')$")

    # ----- 4) reward (the dielectric objective) -------------------------------------
    # Full pure-math formula (no English prose), matched to the run's scalar_mode in
    # rewards/calculators/tsenn_static_dielectric.py:209-250 (layered_uniaxial vs
    # inplane_isotropy).  One line, sized to fit the ~5.1 in box.
    ax.add_patch(FancyBboxPatch((colx - BOX_W / 2, 0.180), BOX_W, 0.066,
                                boxstyle="round,pad=0.004,rounding_size=0.018",
                                facecolor="white", edgecolor="k", lw=1.4, zorder=3))
    if scalar_mode == "inplane_isotropy":
        # keep the in-plane-isotropy term explicit; the out-of-plane gate is just the symbol
        # g_z (its full clip definition can be expanded later).
        ax.text(colx, 0.213,
                r"$r = \left(1-\frac{|\varepsilon_{xx}-\varepsilon_{yy}|}{|\varepsilon_{xx}|+|\varepsilon_{yy}|}\right)"
                r"\cdot g_{z}$",
                ha="center", va="center", fontsize=_FS_BASE0 + 1, zorder=4)
    else:
        ax.text(colx, 0.213,
                r"$r = \frac{|\varepsilon_{\parallel}-\varepsilon_{\perp}|}{|\varepsilon_{\parallel}|+|\varepsilon_{\perp}|}"
                r"\!\left(1-\frac{|\varepsilon_{xx}-\varepsilon_{yy}|}{|\varepsilon_{xx}|+|\varepsilon_{yy}|}\right),"
                r"\ \ \varepsilon_{\parallel} = \varepsilon_{zz},\ \varepsilon_{\perp} = \frac{1}{2}(\varepsilon_{xx}+\varepsilon_{yy})$",
                ha="center", va="center", fontsize=_FS_BASE0 + 1, zorder=4)
    # unlabeled arrow -- the histogram title below ("reward-updated ...") names the result,
    # so a separate "update SG policy" caption would only collide with it.
    _varrow(ax, colx, 0.175, 0.137)

    # ----- RL feedback loop: arrow spans the gap between reward box and generator box;
    # sits right of the "compute reward r(M)" label (which ends ~0.64) so they don't touch
    arrow_x = 0.70
    ax.annotate("", xy=(arrow_x, 0.520), xytext=(arrow_x, 0.246),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=2.0,
                                mutation_scale=18, linestyle="--"),
                annotation_clip=False)
    ax.text(arrow_x + 0.028, 0.381, r"RL update $\theta$",
            ha="left", va="center",
            fontsize=FS_LABEL, color="0.35", rotation=0, clip_on=False)

    # ----- AFTER: the reward-updated SG policy (the reward reshapes pi_theta(SG)) ----
    after_used, learned = _load_reward_updated_sg_dist(run_dir, after_step)
    aax = ax.inset_axes([0.130, 0.044, 0.540, 0.070])
    _draw_sg_hist(aax, learned, orange,
                  "reward-updated SG distribution $\\pi^{(j)}$", show_xlabel=True,
                  show_peak=False, ylabel="$\\pi^{(j)}$")


def _draw_sg_panel(ax_dist, ranked, total, sgs, frac, dist_label, top):
    """Top-N space-group bars (mother) + full distribution (inset)."""
    _draw_top(ax_dist, ranked, total, small=False, split_labels=True,
              axis_fs=FS_AXIS, tick_fs=FS_LABEL)
    ax_dist.set_xticks([5, 15, 25])
    ax_dist.set_title(f"Top {min(top, len(ranked))} space groups", fontsize=FS_AXIS)
    ins = ax_dist.inset_axes([0.40, 0.14, 0.58, 0.58])
    _draw_full(ins, sgs, frac, small=True, fs=FS_LABEL)
    ins.set_yticks([5, 15, 25])
    ins.set_xticks([1, 115, 230])
    ins.set_title(dist_label, fontsize=FS_LABEL)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.split_panels and args.legacy_showcase:
        raise SystemExit("--split-panels is only supported with the NEW layout "
                         "(drop --legacy-showcase)")
    # shared font standard. panels (a)/(c) set explicit sizes (unaffected by this); panel
    # (b)=_plot_reward relies on rcParams, so this is what lands it on the standard:
    # main axis labels/titles at FS_AXIS (11), ticks + legend at FS_LABEL (9).
    plt.rcParams.update({"axes.labelsize": FS_AXIS, "axes.titlesize": FS_AXIS,
                         "xtick.labelsize": FS_LABEL, "ytick.labelsize": FS_LABEL,
                         "legend.fontsize": FS_LABEL,
                         # calligraphic \mathcal{M} (material) in the generator title;
                         # the publication style maps cal->Liberation Sans (plain M), so
                         # point it at the bundled CM symbol font for a true script glyph
                         "mathtext.cal": "cmsy10"})

    # FAST path: redraw only panel (a) from the cached example, skipping the slow full
    # eval-set load -- it reuses the SAME figure/gridspec geometry so the cropped panel
    # is identical to the composite's panel (a).
    if args.schematic_only:
        cache = run_dir / "deliverables_layered_uniaxial" / "schematic_example.json"
        if not cache.exists():
            raise SystemExit(
                f"--schematic-only needs {cache}; run once without it (e.g. "
                f"--schematic-formula {args.schematic_formula or 'NbMoSe4'}) to cache it.")
        example = json.loads(cache.read_text())
        fig = plt.figure(figsize=(15.0, 8.0))
        # same margins/ratios as the composite NEW layout so the cropped panel is identical
        outer = GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.10,
                         left=0.04, right=0.978, top=0.94, bottom=0.085, figure=fig)
        ax_schematic = fig.add_subplot(outer[0])
        _draw_schematic(fig, ax_schematic, example, run_dir, after_step=args.sg_after_step)
        fig.canvas.draw()
        inv = fig.dpi_scale_trans.inverted()
        pad = 0.06
        (x0, y0) = ax_schematic.transAxes.transform((0.0, 0.0))
        (x1, y1) = ax_schematic.transAxes.transform((0.89, 1.0))
        bb = Bbox.from_extents(x0, y0, x1, y1).transformed(inv)
        bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
        outdir = (args.output.parent if args.output
                  else run_dir / "deliverables_layered_uniaxial")
        outdir.mkdir(parents=True, exist_ok=True)
        for ext in (".png", ".pdf", ".eps"):
            fig.savefig(outdir / f"dielectric_story_panel_a{ext}", bbox_inches=bb)
        print(f"[schematic-only] wrote dielectric_story_panel_a.{{png,pdf,eps}} to {outdir}")
        return

    # (a) reward
    steps, reward = _load_reward_series(run_dir)

    # best-reward window mode: restrict panels (b)/(c) to a few peak-reward steps
    box = None
    if args.best_window > 0:
        lo, hi, best_step = _best_window(steps, reward, args.window, args.best_window)
        box = (lo, hi)
        window_steps = list(range(lo, hi))
        counts = _windowed_sg_counts(run_dir / "samples", args.artifact, window_steps)
        records = []
        for s in window_steps:
            try:
                records.extend(load_step_records(run_dir, requested_step=s))
            except ValueError:  # no valid records this step
                pass
        dist_label = f"Distribution (best {hi - lo} steps)"
    else:
        counts = load_symmetry_counts(run_dir / "samples", args.artifact)
        records = load_step_records(run_dir, requested_step=None)
        dist_label = "Overall distribution"

    # drop surrogate dielectric blow-ups from the panel (c) candidate pool
    records = [r for r in records
               if max(r["eps_xx"], r["eps_yy"], r["eps_zz"]) <= args.eps_max]

    # (b) ranked top-N space groups (main) + the distribution (inset)
    total = sum(counts.values())
    sgs = np.array(sorted(counts))
    frac = np.array([100.0 * counts[s] / total for s in sgs])
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
    # space groups showcased in panel (c): explicit --only-sg, else the ranked SGs
    # minus any --exclude-sg (e.g. drop cubic 225, which can't be uniaxial).
    excl = set(args.exclude_sg)
    if args.only_sg:
        allowed_sgs = list(args.only_sg)
    elif args.rank_by == "reward":
        # all space groups (minus excluded) so high-reward SGs outside the popularity
        # top-N can still be showcased
        allowed_sgs = [sg for sg in sorted(counts) if sg not in excl]
    else:
        allowed_sgs = [sg for sg, _ in ranked if sg not in excl]

    # (c) representatives from the allowed space groups
    if args.rank_by == "reward":
        picks = _pick_by_reward(records, allowed_sgs, args.n_structures, args.ehull_max)
    else:
        picks = _pick_stable(records, allowed_sgs, args.n_structures, args.ehull_max)

    if args.legacy_showcase:
        # ---- OLD layout: left column (a)/(b); right block (c) = 2x2 structures,
        # each split into [render | tensor matrix].
        fig = plt.figure(figsize=(16.0, 8.0))
        outer = GridSpec(1, 2, width_ratios=[1.0, 1.12], wspace=0.16, figure=fig)
        left = outer[0].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.34)
        right = outer[1].subgridspec(2, 4, width_ratios=[1.0, 0.55, 1.0, 0.55],
                                     wspace=0.20, hspace=0.42)
        ax_reward = fig.add_subplot(left[0])
        ax_dist = fig.add_subplot(left[1])
        render_axes, matrix_axes = [], []
        for r in range(2):
            for cpair in (0, 2):
                render_axes.append(fig.add_subplot(right[r, cpair]))
                matrix_axes.append(fig.add_subplot(right[r, cpair + 1]))
        _plot_reward(ax_reward, steps, reward, args.window, box=box)
        _draw_sg_panel(ax_dist, ranked, total, sgs, frac, dist_label, args.top)
        _plot_structure_cells(render_axes, matrix_axes, picks, side_idx=(0, 1))
        pa = ax_reward.get_position(); pb = ax_dist.get_position()
        pc = render_axes[0].get_position()
        top_y = pa.y1 + 0.03; left_x = pa.x0 - 0.068
        for x, y, s in [(left_x, top_y, "(a)"), (left_x, pb.y1 + 0.012, "(b)"),
                        (pc.x0 - 0.080, top_y, "(c)")]:
            fig.text(x, y, s, fontsize=FS_BASE + 4, fontweight="bold",
                     ha="left", va="bottom")
    else:
        # ---- NEW layout: left = method schematic (a); right = reward (b) + SG bars (c)
        example = _pick_schematic_example(records, args.ehull_max,
                                          sg=args.schematic_sg,
                                          rank=args.schematic_rank,
                                          formula=args.schematic_formula)
        if example is None:
            raise SystemExit("No uniaxial example structure found for the schematic; "
                             "relax --schematic-formula/--schematic-sg or use "
                             "--legacy-showcase.")
        picks = [example]   # CIF export below writes the schematic example
        # cache the picked example so --schematic-only can redraw panel (a) instantly
        _cache_schematic_example(
            run_dir / "deliverables_layered_uniaxial" / "schematic_example.json", example)
        # explicit, deterministic margins (no manual re-crop) + a balanced right-column
        # split. (a)'s width fraction is kept so the carefully-tuned schematic isn't
        # stretched; the b/c split + gaps are what we tune.
        fig = plt.figure(figsize=(15.0, 8.0))
        outer = GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.10,
                         left=0.04, right=0.978, top=0.94, bottom=0.085, figure=fig)
        ax_schematic = fig.add_subplot(outer[0])
        right = outer[1].subgridspec(2, 1, height_ratios=[0.80, 1.20], hspace=0.22)
        ax_reward = fig.add_subplot(right[0])
        ax_dist = fig.add_subplot(right[1])
        _draw_schematic(fig, ax_schematic, example, run_dir,
                        after_step=args.sg_after_step)
        _plot_reward(ax_reward, steps, reward, args.window, box=box)
        _draw_sg_panel(ax_dist, ranked, total, sgs, frac, dist_label, args.top)
        ps = ax_schematic.get_position()
        pa = ax_reward.get_position(); pb = ax_dist.get_position()
        # split-panels crops by figure rectangle; the letters sit just left of each axes and
        # would land inside a crop, so omit them in that mode (user adds labels on recombine)
        if not args.split_panels:
            for x, y, s in [(0.006, ps.y1, "(a)"),
                            (pa.x0 - 0.068, pa.y1 + 0.010, "(b)"),
                            (pb.x0 - 0.068, pb.y1 + 0.010, "(c)")]:
                fig.text(x, y, s, fontsize=FS_BASE + 4, fontweight="bold",
                         ha="left", va="bottom")

    tag = ""
    if args.best_window > 0:
        tag += f"_best{args.best_window}steps"
    if args.rank_by == "reward":
        tag += "_byreward"
    if args.only_sg:
        tag += "_sg" + "-".join(str(s) for s in args.only_sg)
    elif args.exclude_sg:
        tag += "_no" + "-".join(str(s) for s in args.exclude_sg)
    default_name = f"dielectric_story_composite{tag}.png"
    out = args.output or (run_dir / "deliverables_layered_uniaxial" / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.split_panels:
        # crop each panel so there is no bleed from neighbours and no clipped labels.
        # (b)/(c) are real plots -> their own tight bbox (axes + ticks/labels/legend/inset).
        # (a) is an axis-off schematic whose tight bbox is the FULL subplot rect (reaching the
        # gap, pulling in the neighbouring SG-bar y-labels), so crop it to an explicit content
        # box in axes fraction: x in [0, 0.84] drops the empty right + the gap; full height.
        # (the schematic is a single vertical column, all content within x<=0.77.)
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = fig.dpi_scale_trans.inverted()
        pad = 0.06   # inches of breathing room around each panel
        bb_b = ax_reward.get_tightbbox(rend).transformed(inv)
        bb_c = ax_dist.get_tightbbox(rend).transformed(inv)
        divider_y = (bb_b.y0 + bb_c.y1) / 2   # midpoint gap between b and c
        for name, ax in (("a", ax_schematic), ("b", ax_reward), ("c", ax_dist)):
            if name == "a":
                (x0, y0) = ax.transAxes.transform((0.0, 0.0))
                (x1, y1) = ax.transAxes.transform((0.89, 1.0))
                bb = Bbox.from_extents(x0, y0, x1, y1).transformed(inv)
            elif name == "b":
                bb = Bbox.from_extents(bb_b.x0, divider_y, bb_b.x1, bb_b.y1)
            else:
                bb = Bbox.from_extents(bb_c.x0, bb_c.y0, bb_c.x1, divider_y)
            bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
            for ext in (".png", ".pdf", ".eps"):
                fig.savefig(out.parent / f"dielectric_story_panel_{name}{tag}{ext}",
                            bbox_inches=bb)
        plt.close(fig)
        print(f"Wrote dielectric_story_panel_{{a,b,c}}{tag}.png/.pdf/.eps to {out.parent} "
              f"(eval N={total})")
    else:
        for ext in (".png", ".pdf", ".eps"):
            fig.savefig(out.with_suffix(ext), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out.with_suffix('.png')} / .pdf / .eps  (eval N={total})")

    # export the panel (c) structures as .cif (for Crystal Toolkit etc.)
    cif_dir = out.parent / "story_structures_cif"
    cif_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(picks):
        fn = f"c{i+1}_{r['formula']}_SG{r['spacegroup']}.cif"
        (cif_dir / fn).write_text(str(r["cif"]))
    print(f"CIFs: {cif_dir}")

    def _summ(r):
        eh = r.get("ehull")
        eh_s = "n/a" if eh is None else f"{eh:.2f}"
        return f"{r['formula']} (SG{r['spacegroup']}, score {r['reward']:.2f}, Ehull {eh_s})"

    print("Structures:", ", ".join(_summ(r) for r in picks))


if __name__ == "__main__":
    main()
