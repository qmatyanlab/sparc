#!/usr/bin/env python3
"""Before/after MLIP-relaxation symmetry recovery, as a paper figure.

Relaxes a sample of as-generated scored structures from each run (MatterSim,
reusing the pipeline's batch relaxer), then compares space group / crystal
system / symmetry order at strict symprec before vs after relaxation.

Shows that relaxation only PARTIALLY rescues DiffCSP (≈half stay P1, recovered
ones reach only low monoclinic — rarely the uniaxial symmetry the task needs),
whereas SymmCD is already at the symmetric minimum (relaxation is a near no-op).

Saves a per-run before/after CSV (so re-plotting needs no re-relaxation) and
the figure.

Example:
    python scripts/relax_recovery_figure.py \
        exp_res/..._symmcd_v1_carryover_53154012 \
        exp_res/..._diffcsp_v1_54289718 --labels SymmCD DiffCSP --n 150
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
from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

from pipeline.filters.opt_filter import _relax_structures_with_progress

STRICT = 0.01
UNIAXIAL = {"tetragonal", "trigonal", "hexagonal"}
SYS_ORDER = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
             "trigonal", "hexagonal", "cubic"]
SYS_COLORS = dict(zip(SYS_ORDER,
    ["#999999", "#8c6bb1", "#4393c3", "#2ca25f", "#f4a460", "#d6604d", "#b2182b"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", type=Path, nargs=2)
    p.add_argument("--labels", nargs=2, default=None)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--last-steps", type=int, default=20)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-n-steps", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--potential", default="MatterSim-v1.0.0-5M.pth")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def crystal_system(sg):
    if sg is None:
        return None
    if sg <= 2: return "triclinic"
    if sg <= 15: return "monoclinic"
    if sg <= 74: return "orthorhombic"
    if sg <= 142: return "tetragonal"
    if sg <= 167: return "trigonal"
    if sg <= 194: return "hexagonal"
    return "cubic"


def sg_and_order(s):
    try:
        ds = SpacegroupAnalyzer(s, symprec=STRICT).get_symmetry_dataset()
        return int(ds.number), int(len(ds.rotations))
    except Exception:
        return None, None


def load_sample(run_dir: Path, n: int, last_steps: int, rng):
    strucs = []
    for pt in sorted((run_dir / "samples").glob("step_*_eval.extxyz"))[-last_steps:]:
        al = ase_read(pt, index=":")
        if not isinstance(al, list):
            al = [al]
        strucs.extend(AseAtomsAdaptor.get_structure(a) for a in al)
    if len(strucs) > n:
        strucs = [strucs[i] for i in rng.choice(len(strucs), n, replace=False)]
    return strucs


def _energies_per_atom(strucs, potential, device):
    """Single-point MatterSim energy/atom for the as-generated structures."""
    from mattersim.forcefield import MatterSimCalculator
    calc = MatterSimCalculator(load_path=potential, device=device)
    out = []
    for s in strucs:
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.calc = calc
        try:
            out.append(float(atoms.get_potential_energy()) / len(atoms))
        except Exception:  # noqa: BLE001
            out.append(float("nan"))
    return out


# before/after is the SAME structure (identical composition & atom count), so no
# supercell search is needed -- keep it fast. NB: get_rms_dist is volume-normalised,
# so this captures internal/fractional rearrangement, not absolute cell rescaling.
_MATCHER = StructureMatcher(primitive_cell=False, attempt_supercell=False)


def _displacement_rmsd(before, after):
    """RMSD (A) between a structure and its relaxed self (cell-aware, scaled)."""
    try:
        rms = _MATCHER.get_rms_dist(before, after)
        if rms is None:
            return float("nan")
        return float(rms[0]) * float(after.lattice.volume / len(after)) ** (1 / 3)
    except Exception:  # noqa: BLE001
        return float("nan")


def relax_and_record(run_dir, label, args, rng):
    cache = (args.output_dir or run_dir / "deliverables" / "relax_recovery") / f"records_{label}.csv"
    strucs = load_sample(run_dir, args.n, args.last_steps, rng)
    print(f"[{label}] relaxing {len(strucs)} structures ...")
    before = [sg_and_order(s) for s in strucs]
    e_before = _energies_per_atom(strucs, args.potential, args.device)
    relaxed, e_after_total = _relax_structures_with_progress(
        strucs, device=torch.device(args.device), potential_load_path=args.potential,
        fmax=args.fmax, max_n_steps=args.max_n_steps, max_natoms_per_batch=128,
        optimizer="FIRE", cell_filter="EXPCELLFILTER")
    relaxed = [a if isinstance(a, Structure) else AseAtomsAdaptor.get_structure(a)
               for a in relaxed]
    after = [sg_and_order(s) for s in relaxed]
    recs = []
    for i, ((sgb, ob), (sga, oa)) in enumerate(zip(before, after)):
        nat = len(relaxed[i])
        ea = float(e_after_total[i]) / nat if np.isfinite(e_after_total[i]) else float("nan")
        recs.append({"sg_before": sgb, "ops_before": ob, "sys_before": crystal_system(sgb),
                     "sg_after": sga, "ops_after": oa, "sys_after": crystal_system(sga),
                     "natoms": nat, "e_before_per_atom": e_before[i], "e_after_per_atom": ea,
                     "de_per_atom": ea - e_before[i],
                     "rmsd": _displacement_rmsd(strucs[i], relaxed[i])})
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    # energetics + displacement summary (the "how far did they move" numbers)
    de = np.array([-r["de_per_atom"] for r in recs if np.isfinite(r["de_per_atom"])])  # energy DROP
    rm = np.array([r["rmsd"] for r in recs if np.isfinite(r["rmsd"])])
    if len(de):
        print(f"[{label}] relaxation energy DROP (eV/atom): median={np.median(de):.4f} "
              f"mean={de.mean():.4f}  |  displacement RMSD (A): median={np.median(rm):.4f} "
              f"mean={rm.mean():.4f}")
    return recs


def frac(recs, key, pred):
    vals = [r for r in recs if r[key] is not None]
    return np.mean([pred(r[key]) for r in vals]) if vals else float("nan")


def main() -> None:
    args = parse_args()
    labels = args.labels or [d.name for d in args.run_dirs]
    out = args.output_dir or (args.run_dirs[1] / "deliverables" / "relax_recovery")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    colors = {labels[0]: "tab:blue", labels[1]: "tab:red"}

    data = {lab: relax_and_record(d, lab, args, rng)
            for lab, d in zip(labels, args.run_dirs)}

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    width = 0.38
    x = np.arange(2)  # before, after

    # (a) P1 fraction before/after
    for k, lab in enumerate(labels):
        recs = data[lab]
        p1 = [frac(recs, "sg_before", lambda v: v == 1),
              frac(recs, "sg_after", lambda v: v == 1)]
        bars = axes[0].bar(x + (k - 0.5) * width, p1, width, color=colors[lab], label=lab)
        for b, v in zip(bars, p1):
            axes[0].text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["as-generated", "after relax"])
    axes[0].set_ylabel("fraction P1 (triclinic)"); axes[0].set_ylim(0, 1.08)
    axes[0].set_title(f"(a) P1 fraction @ symprec {STRICT}\nrelaxation only half-rescues DiffCSP")
    axes[0].legend()

    # (b) uniaxial fraction before/after (the symmetry the task needs)
    for k, lab in enumerate(labels):
        recs = data[lab]
        uni = [frac(recs, "sys_before", lambda v: v in UNIAXIAL),
               frac(recs, "sys_after", lambda v: v in UNIAXIAL)]
        bars = axes[1].bar(x + (k - 0.5) * width, uni, width, color=colors[lab], label=lab)
        for b, v in zip(bars, uni):
            axes[1].text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(["as-generated", "after relax"])
    axes[1].set_ylabel("fraction uniaxial (tet/trig/hex)"); axes[1].set_ylim(0, 1.08)
    axes[1].set_title("(b) uniaxial symmetry (protects ε_xx=ε_yy)\nDiffCSP barely reaches it even relaxed")
    axes[1].legend()

    # (c) crystal-system composition AFTER relax (stacked) per run
    for k, lab in enumerate(labels):
        recs = data[lab]
        sysc = Counter(r["sys_after"] for r in recs if r["sys_after"])
        tot = sum(sysc.values())
        bottom = 0.0
        for sysname in SYS_ORDER:
            f = sysc.get(sysname, 0) / tot if tot else 0
            axes[2].bar(k, f, 0.6, bottom=bottom, color=SYS_COLORS[sysname],
                        label=sysname if k == 0 else None)
            bottom += f
    axes[2].set_xticks(range(len(labels))); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("fraction"); axes[2].set_ylim(0, 1.0)
    axes[2].set_title("(c) crystal system after relaxation\nDiffCSP stuck in triclinic+monoclinic")
    axes[2].legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    fig.suptitle("MLIP relaxation only partially rescues DiffCSP's symmetry; SymmCD is born symmetric",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "relax_recovery.png", dpi=150, bbox_inches="tight")

    for lab in labels:
        recs = data[lab]
        ob = [r["ops_before"] for r in recs if r["ops_before"]]
        oa = [r["ops_after"] for r in recs if r["ops_after"]]
        print(f"[{lab}] median symmetry ops {int(np.median(ob))}→{int(np.median(oa))} | "
              f"P1 {frac(recs,'sg_before',lambda v:v==1):.0%}→{frac(recs,'sg_after',lambda v:v==1):.0%} | "
              f"uniaxial {frac(recs,'sys_before',lambda v:v in UNIAXIAL):.0%}→"
              f"{frac(recs,'sys_after',lambda v:v in UNIAXIAL):.0%}")
    print(f"saved {out / 'relax_recovery.png'}")


if __name__ == "__main__":
    main()
