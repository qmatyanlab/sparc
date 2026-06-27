#!/usr/bin/env python3
"""Animate the chemical-species migration of an RL run: a periodic-table
density heatmap (how often each element appears in the scored structures at
each step) alongside the reward-vs-step curve with a moving marker.

Shows the search "hopping" through chemistries as the reward evolves.

Reads samples/step_*_eval.pt (atom_types) + metrics.csv (reward mean).

Example:
    python scripts/demo_element_migration_gif.py \
        exp_res/tsenn_slme_03um_symmcd_v3_seed1_optimate_adaptive_53904266
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from ase.visualize.plot import plot_atoms
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor

# periodic-table layout: Z -> (row, col), 18 columns (groups), f-block split out
def build_layout() -> dict[int, tuple[int, int]]:
    pos: dict[int, tuple[int, int]] = {}
    main = {  # period: list of (Z, group) ; group 1..18
        1: [(1, 1), (2, 18)],
        2: [(3, 1), (4, 2), (5, 13), (6, 14), (7, 15), (8, 16), (9, 17), (10, 18)],
        3: [(11, 1), (12, 2), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18)],
        4: [(z, g) for z, g in zip(range(19, 37), range(1, 19))],
        5: [(z, g) for z, g in zip(range(37, 55), range(1, 19))],
        6: [(55, 1), (56, 2)] + [(z, g) for z, g in zip(range(72, 87), range(4, 19))],
        7: [(87, 1), (88, 2)] + [(z, g) for z, g in zip(range(104, 119), range(4, 19))],
    }
    for period, items in main.items():
        for z, group in items:
            pos[z] = (period - 1, group - 1)
    # lanthanides 57-71 -> row 8 (cols 2..16), actinides 89-103 -> row 9
    for i, z in enumerate(range(57, 72)):
        pos[z] = (8, 2 + i)
    for i, z in enumerate(range(89, 104)):
        pos[z] = (9, 2 + i)
    return pos


def item_to_atoms(item):
    """Build an ASE Atoms (for rendering) + pymatgen Structure from a .pt item."""
    L = item["lengths"][0].tolist()
    A = item["angles"][0].tolist()
    lat = Lattice.from_parameters(L[0], L[1], L[2], A[0], A[1], A[2])
    sp = [Element.from_Z(int(z)) for z in item["atom_types"].tolist()]
    st = Structure(lat, sp, item["frac_coords"].numpy(), coords_are_cartesian=False)
    return AseAtomsAdaptor.get_atoms(st), st


def load_best_structures(run_dir: Path, n_best: int = 2):
    """Per step, the top-n highest-eta scored structures [(atoms, label), ...]."""
    samples = run_dir / "samples"
    eta_dir = next((run_dir / "rewards" / d for d in
                    ["tsenn_slme_optimate_eta", "tsenn_slme_eta"]
                    if (run_dir / "rewards" / d).exists()), None)
    best = {}
    for pt in sorted(samples.glob("step_*_eval.pt")):
        step = int(pt.stem.split("_")[1])
        payload = torch.load(pt, map_location="cpu")
        if not isinstance(payload, list) or not payload:
            continue
        eta = (np.loadtxt(eta_dir / f"step_{step:04d}.txt", ndmin=1)
               if eta_dir and (eta_dir / f"step_{step:04d}.txt").exists()
               else np.full(len(payload), np.nan))
        order = np.argsort(np.nan_to_num(eta, nan=-1.0))[::-1]
        picks = []
        for idx in order:
            item = payload[idx]
            if not isinstance(item, dict) or "atom_types" not in item:
                continue
            try:
                atoms, st = item_to_atoms(item)
            except Exception:
                continue
            sg = item.get("spacegroup")
            e = float(eta[idx]) if idx < len(eta) else float("nan")
            picks.append((atoms,
                          f"{st.composition.reduced_formula}  SG{sg}  η={e:.2f}"))
            if len(picks) >= n_best:
                break
        if picks:
            best[step] = picks
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA smoothing of element density (0=raw). Higher alpha "
                        "= smoother but more lag / washes out migration.")
    p.add_argument("--metric", choices=["occurrence", "atom_frac"],
                   default="occurrence",
                   help="occurrence=frac of structures containing element; "
                        "atom_frac=frac of atoms")
    p.add_argument("--weight", choices=["none", "reward", "eta"], default="none",
                   help="weight each structure's contribution by its combined "
                        "reward / eta (so density = where reward mass concentrates)")
    return p.parse_args()


def _ascending(v, minv, maxv):
    return float(np.clip((v - minv) / (maxv - minv), 0.0, 1.0))


def load_reward_terms(run_dir: Path):
    """Read per-prop (minv,maxv,weight) for band_gap & eta from the run's hydra
    config, to reconstruct each structure's combined reward."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    terms = {}
    for prop in cfg.reward.prop_cfg:
        name = str(prop.name)
        key = "band_gap" if name == "band_gap" else ("eta" if name.endswith("_eta") else None)
        if key and float(prop.get("weight", 0.0)) > 0:
            terms[key] = (float(prop.minv), float(prop.maxv), float(prop.weight))
    return terms


def load_density(run_dir: Path, metric: str, weight: str):
    samples = run_dir / "samples"
    bg_dir = run_dir / "rewards" / "bandgap"
    # detect eta dir
    eta_dir = next((run_dir / "rewards" / d for d in
                    ["tsenn_slme_optimate_eta", "tsenn_slme_eta"]
                    if (run_dir / "rewards" / d).exists()), None)
    terms = load_reward_terms(run_dir) if weight == "reward" else {}
    rows, steps = [], []
    for pt in sorted(samples.glob("step_*_eval.pt")):
        step = int(pt.stem.split("_")[1])
        payload = torch.load(pt, map_location="cpu")
        if not isinstance(payload, list) or not payload:
            continue
        eta = (np.loadtxt(eta_dir / f"step_{step:04d}.txt", ndmin=1)
               if eta_dir and (eta_dir / f"step_{step:04d}.txt").exists()
               else np.full(len(payload), np.nan))
        bg = (np.loadtxt(bg_dir / f"step_{step:04d}.txt", ndmin=1)
              if (bg_dir / f"step_{step:04d}.txt").exists()
              else np.full(len(payload), np.nan))
        dens = np.zeros(119)
        wsum = 0.0
        for i, item in enumerate(payload):
            if not isinstance(item, dict) or "atom_types" not in item:
                continue
            zs = [int(z) for z in item["atom_types"].tolist()]
            if not zs:
                continue
            e = float(eta[i]) if i < len(eta) else np.nan
            if weight == "none":
                w = 1.0
            elif weight == "eta":
                w = max(0.0, e) if np.isfinite(e) else 0.0
            else:  # combined reward
                w = 0.0
                if "band_gap" in terms and i < len(bg) and np.isfinite(bg[i]):
                    mn, mx, wt = terms["band_gap"]; w += wt * _ascending(bg[i], mn, mx)
                if "eta" in terms and np.isfinite(e):
                    mn, mx, wt = terms["eta"]; w += wt * _ascending(e, mn, mx)
            if w <= 0:
                continue
            wsum += w
            if metric == "occurrence":
                for z in set(zs):
                    if z < 119:
                        dens[z] += w
            else:
                c = Counter(zs); tot = sum(c.values())
                for z, n in c.items():
                    if z < 119:
                        dens[z] += w * n / tot
        if wsum > 0:
            rows.append(dens / wsum)
            steps.append(step)
    return steps, np.array(rows)


def main() -> None:
    args = parse_args()
    suffix = "" if args.weight == "none" else f"_{args.weight}wt"
    out = args.output or (args.run_dir / "deliverables" / f"element_migration{suffix}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)

    steps, dens = load_density(args.run_dir, args.metric, args.weight)
    if args.ema > 0:
        sm = dens.copy()
        for i in range(1, len(sm)):
            sm[i] = args.ema * sm[i - 1] + (1 - args.ema) * dens[i]
        dens = sm

    rw = {}
    with (args.run_dir / "metrics.csv").open() as fh:
        for r in csv.DictReader(fh):
            try:
                rw[int(float(r["step"]))] = float(r["reward mean"])
            except (KeyError, ValueError):
                pass
    reward = np.array([rw.get(s, np.nan) for s in steps])

    best = load_best_structures(args.run_dir)

    layout = build_layout()
    nrows, ncols = 10, 18
    present_Z = [z for z in range(1, 119) if z in layout and dens[:, z].max() > 0]
    vmax = float(np.percentile([dens[:, z].max() for z in present_Z], 95)) or 1.0

    fig = plt.figure(figsize=(14, 9.5))
    # tight left-column gap (element density <-> reward); keep right-column gap
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1.4], width_ratios=[3, 1.4],
                          hspace=0.10, wspace=0.16)
    axpt = fig.add_subplot(gs[0, 0])   # element density
    axpt.set_anchor("S")  # bottom-align the equal-aspect table toward the reward
    axr = fig.add_subplot(gs[1, 0])    # reward — same width as element density
    right = gs[:, 1].subgridspec(2, 1, hspace=0.28)  # two sample structures
    axst1 = fig.add_subplot(right[0, 0])
    axst2 = fig.add_subplot(right[1, 0])

    def grid_for(step_i):
        g = np.full((nrows, ncols), np.nan)
        for z, (r, c) in layout.items():
            if z < 119 and dens[step_i, z] > 0:
                g[r, c] = dens[step_i, z]
        return np.ma.masked_invalid(g)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")
    im = axpt.imshow(grid_for(0), cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    # static element symbol labels
    for z, (r, c) in layout.items():
        try:
            sym = Element.from_Z(z).symbol
        except Exception:
            continue
        axpt.text(c, r, sym, ha="center", va="center", fontsize=7, color="black")
    axpt.set_xticks([]); axpt.set_yticks([])
    wlabel = {"none": "scored structures",
              "reward": "reward-weighted (where reward mass sits)",
              "eta": "η-weighted"}[args.weight]
    axpt.set_title(f"element density — {wlabel}")
    cb = fig.colorbar(im, ax=axpt, shrink=0.7, pad=0.01)
    cb.set_label(f"{args.metric} frequency")

    axr.plot(steps, reward, color="black", lw=1.2)
    axr.set_xlabel("RL step"); axr.set_ylabel("reward mean")
    marker = axr.axvline(steps[0], color="crimson", lw=2)
    dot, = axr.plot([steps[0]], [reward[0]], "o", color="crimson", ms=8)

    ROT = "-75x,10y,0z"  # fixed view so the structure rotation is consistent
    struct_axes = [axst1, axst2]
    rank_label = ["best sample", "2nd best sample"]

    def draw_structure(step):
        picks = best.get(step, [])
        for k, ax in enumerate(struct_axes):
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            if k < len(picks):
                atoms, label = picks[k]
                try:
                    plot_atoms(atoms, ax, rotation=ROT, radii=0.5)
                except Exception:
                    pass
                ax.set_title(f"{rank_label[k]}\n{label}", fontsize=9)
            else:
                ax.set_title(rank_label[k], fontsize=9)

    draw_structure(steps[0])

    def top_elems(step_i, k=5):
        order = np.argsort(dens[step_i])[::-1]
        out = []
        for z in order[:k]:
            if dens[step_i, z] <= 0:
                break
            try:
                out.append(f"{Element.from_Z(int(z)).symbol}:{dens[step_i, z]:.0%}")
            except Exception:
                pass
        return ", ".join(out)

    sup = fig.suptitle("", fontsize=13)

    def update(i):
        im.set_data(grid_for(i))
        marker.set_xdata([steps[i], steps[i]])
        dot.set_data([steps[i]], [reward[i]])
        draw_structure(steps[i])
        sup.set_text(f"step {steps[i]}   reward={reward[i]:.3f}   top: {top_elems(i)}")
        return im, marker, dot, sup

    anim = FuncAnimation(fig, update, frames=len(steps), blit=False)
    # margins only — keep the per-gridspec hspace set above
    fig.subplots_adjust(left=0.05, right=0.93, top=0.92, bottom=0.08)
    anim.save(str(out), writer=PillowWriter(fps=args.fps), dpi=90)
    print(f"frames={len(steps)} | tracked elements={len(present_Z)} | vmax={vmax:.2f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
