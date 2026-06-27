#!/usr/bin/env python3
"""Test the hypothesis: does MLIP (MatterSim) relaxation remove DiffCSP's
~0.065 A pseudo-symmetry distortion, snapping its P1 structures back onto a
higher-symmetry parent?

Relaxes a sample of as-generated eval structures (reusing the pipeline's
MatterSim BatchRelaxer) and compares space group at strict symprec (0.01)
and the RMSD-to-refined-parent, before vs after relaxation.

Example:
    python scripts/relax_symmetry_recovery.py \
        exp_res/..._diffcsp_v1_54289718 --label diffcsp --n 60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

from pipeline.filters.opt_filter import _relax_structures_with_progress


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir", type=Path)
    p.add_argument("--label", default=None)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--last-steps", type=int, default=10)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-n-steps", type=int, default=500)
    p.add_argument("--symprec", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--potential", default="MatterSim-v1.0.0-5M.pth"
    )
    return p.parse_args()


def sg(structure, symprec):
    try:
        return SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number()
    except Exception:
        return None


def rmsd_to_parent(structure, symprec=0.1):
    """RMSD (A) between structure and its own symmetry-refined cell."""
    try:
        refined = SpacegroupAnalyzer(structure, symprec=symprec).get_refined_structure()
    except Exception:
        return np.nan
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=True)
    rms = matcher.get_rms_dist(structure, refined)
    if rms is None:
        return np.nan
    # scale the (dimensionless) rms by mean nearest-neighbour-ish length
    return float(rms[0]) * float(structure.lattice.volume / len(structure)) ** (1 / 3)


def main() -> None:
    args = parse_args()
    label = args.label or args.exp_dir.name
    samples = args.exp_dir / "samples"
    steps = sorted(
        int(p.stem.split("_")[1]) for p in samples.glob("step_*_eval.extxyz")
    )[-args.last_steps :]

    strucs = []
    for step in steps:
        al = ase_read(samples / f"step_{step:04d}_eval.extxyz", index=":")
        if not isinstance(al, list):
            al = [al]
        strucs.extend(AseAtomsAdaptor.get_structure(a) for a in al)
    rng = np.random.default_rng(0)
    if len(strucs) > args.n:
        strucs = [strucs[i] for i in rng.choice(len(strucs), args.n, replace=False)]
    print(f"[{label}] relaxing {len(strucs)} structures "
          f"(fmax={args.fmax}, max_n_steps={args.max_n_steps}) ...")

    sg_before = [sg(s, args.symprec) for s in strucs]
    rmsd_before = [rmsd_to_parent(s) for s in strucs]

    relaxed, _ = _relax_structures_with_progress(
        strucs,
        device=torch.device(args.device),
        potential_load_path=args.potential,
        fmax=args.fmax,
        max_n_steps=args.max_n_steps,
        max_natoms_per_batch=128,
        optimizer="FIRE",
        cell_filter="EXPCELLFILTER",
    )
    relaxed = [
        a if isinstance(a, Structure) else AseAtomsAdaptor.get_structure(a)
        for a in relaxed
    ]
    sg_after = [sg(s, args.symprec) for s in relaxed]
    rmsd_after = [rmsd_to_parent(s) for s in relaxed]

    def p1_frac(sgs):
        v = [x for x in sgs if x]
        return sum(x == 1 for x in v) / len(v) if v else float("nan")

    def nonp1_frac(sgs):
        v = [x for x in sgs if x]
        return sum(x != 1 for x in v) / len(v) if v else float("nan")

    n = len(strucs)
    print(f"\n===== {label}: n={n}, symprec={args.symprec} =====")
    print(f"P1 fraction   before relax: {p1_frac(sg_before):.1%}   "
          f"after relax: {p1_frac(sg_after):.1%}")
    print(f"non-P1 fraction before:     {nonp1_frac(sg_before):.1%}   "
          f"after:        {nonp1_frac(sg_after):.1%}")
    rb = np.array([x for x in rmsd_before if np.isfinite(x)])
    ra = np.array([x for x in rmsd_after if np.isfinite(x)])
    if len(rb):
        print(f"RMSD-to-parent (A)  before: median={np.median(rb):.4f}  "
              f"after: median={np.median(ra):.4f}" if len(ra) else "")
    # transition breakdown
    stayed_p1 = sum(1 for b, a in zip(sg_before, sg_after) if b == 1 and a == 1)
    p1_to_sym = sum(1 for b, a in zip(sg_before, sg_after) if b == 1 and a and a != 1)
    sym_stayed = sum(1 for b, a in zip(sg_before, sg_after) if b and b != 1 and a and a != 1)
    print(f"\ntransitions @ symprec={args.symprec}:")
    print(f"  P1 -> P1 (stayed triclinic):     {stayed_p1}/{n}")
    print(f"  P1 -> higher symmetry (recovered): {p1_to_sym}/{n}")
    print(f"  non-P1 -> non-P1:                {sym_stayed}/{n}")
    # show some recovered examples
    recovered = [(strucs[i].composition.reduced_formula, sg_after[i])
                 for i in range(n) if sg_before[i] == 1 and sg_after[i] and sg_after[i] != 1]
    if recovered:
        print(f"  recovered examples (formula -> SG@{args.symprec}): "
              + ", ".join(f"{f}->{g}" for f, g in recovered[:8]))

    # crystal-system classification: only tetragonal/trigonal/hexagonal protect
    # the uniaxial eps_xx=eps_yy degeneracy the reward targets
    def crystal_system(sgn):
        if sgn is None:
            return "none"
        if sgn == 1 or sgn == 2:
            return "triclinic"
        if sgn <= 15:
            return "monoclinic"
        if sgn <= 74:
            return "orthorhombic"
        if sgn <= 142:
            return "tetragonal"
        if sgn <= 167:
            return "trigonal"
        if sgn <= 194:
            return "hexagonal"
        return "cubic"

    UNIAXIAL = {"tetragonal", "trigonal", "hexagonal"}
    from collections import Counter
    sys_after = Counter(crystal_system(s) for s in sg_after)
    sys_before = Counter(crystal_system(s) for s in sg_before)
    uni_b = sum(sys_before[k] for k in UNIAXIAL)
    uni_a = sum(sys_after[k] for k in UNIAXIAL)
    print(f"\ncrystal system (uniaxial = tet/trig/hex protects eps_xx=eps_yy):")
    print(f"  uniaxial fraction  before: {uni_b}/{n} ({uni_b/n:.1%})   "
          f"after: {uni_a}/{n} ({uni_a/n:.1%})")
    print(f"  systems after relax: "
          + ", ".join(f"{k}:{v}" for k, v in sys_after.most_common()))
    sgc_after = Counter(s for s in sg_after if s)
    print(f"  top SG@{args.symprec} after relax: "
          + ", ".join(f"SG{sg_}:{c}" for sg_, c in sgc_after.most_common(8)))


if __name__ == "__main__":
    main()
