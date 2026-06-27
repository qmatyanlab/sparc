#!/usr/bin/env python
"""Fixed-composition (CSP) DiffCSP sampling for the Fig 3 panel (c) showcase.

Generates *real* DiffCSP structures for the same composition AND unit cell as the
SymmCD showcase crystal (CrTe3W, SG 176), so panel (c) "DiffCSP generated
crystals" is an actual model output rather than a jitter of the SymmCD motif.

Mechanism: DiffCSP places fractional coordinates freely; with the composition
held fixed (keep_atom_types) and the cell held fixed (keep_lattice), the only
thing the model decides is *where the atoms sit* — which, unlike SymmCD, is not
constrained to a space group, so the result has only approximate / no symmetry.

This is a standalone driver. It does NOT touch the RL pipeline; it only calls the
backward-compatible `keep_atom_types` branch added to DiffCSPModule.sample().
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pymatgen.core import Element, Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.diffcsp.diffusion import MAX_ATOMIC_NUM
from models.diffcsp.sample import (
    DEFAULT_STEP_LR,
    data2struc,
    lattices_to_params_shape,
)
from models.suite.diffcsp import DiffCSPSuite

DEFAULT_SYMMCD_PT = (
    "exp_res/tsenn_static_dielectric_layered_uniaxial_symmcd_v1_carryover_53154012"
    "/samples/step_0137_eval.pt"
)


def load_showcase(pt: Path, idx: int):
    """Return (atomic_numbers list, lengths list[3], angles list[3]) of the
    SymmCD showcase crystal — the fixed composition + cell DiffCSP must match."""
    s = torch.load(pt, map_location="cpu", weights_only=False)[idx]
    z = [int(x) for x in s["atom_types"].tolist()]
    lengths = [float(x) for x in s["lengths"][0].tolist()]
    angles = [float(x) for x in s["angles"][0].tolist()]
    return z, lengths, angles


def load_target(pt: Path, idx: int) -> Structure:
    """Full SymmCD target structure DiffCSP must accidentally reproduce."""
    s = torch.load(pt, map_location="cpu", weights_only=False)[idx]
    species = [Element.from_Z(int(x)).symbol for x in s["atom_types"].tolist()]
    L = s["lengths"][0].tolist()
    A = s["angles"][0].tolist()
    return Structure(Lattice.from_parameters(*L, *A), species,
                     s["frac_coords"].numpy())


def build_batch(z, lengths, angles, n_samples):
    """N identical Data objects: fixed one-hot composition + fixed cell."""
    natoms = len(z)
    onehot = torch.zeros(natoms, MAX_ATOMIC_NUM, dtype=torch.float)
    for i, zi in enumerate(z):
        onehot[i, zi - 1] = 1.0  # readout is argmax(...)+1, so col z-1
    data_list = []
    for _ in range(n_samples):
        data_list.append(
            Data(
                num_atoms=torch.LongTensor([natoms]),
                num_nodes=natoms,
                atom_types=onehot.clone(),
                frac_coords=torch.zeros(natoms, 3),  # placeholder; resampled
                lengths=torch.tensor([lengths], dtype=torch.float),
                angles=torch.tensor([angles], dtype=torch.float),
            )
        )
    return data_list


def sample_structures(model, batch, step_lr):
    outputs, _ = model.sample(batch, step_lr=step_lr)
    frac_coords = outputs["frac_coords"].detach().cpu()
    num_atoms = outputs["num_atoms"].detach().cpu()
    atom_types = outputs["atom_types"].detach().cpu()
    lattices = outputs["lattices"].detach().cpu()
    lengths, angles = lattices_to_params_shape(lattices)
    atom_types = torch.argmax(atom_types, dim=-1) + 1
    offset = torch.cumsum(num_atoms, dim=0).tolist()
    offset = torch.LongTensor([0] + offset)
    strucs = []
    for i in range(len(num_atoms)):
        data = Data(
            frac_coords=frac_coords[offset[i] : offset[i + 1]],
            atom_types=atom_types[offset[i] : offset[i + 1]],
            lengths=lengths[i].view(1, -1),
            angles=angles[i].view(1, -1),
            num_atoms=num_atoms[i],
            num_nodes=num_atoms[i],
        )
        strucs.append(data2struc(data))
    return strucs


def space_group(struct, symprec):
    try:
        return SpacegroupAnalyzer(struct, symprec=symprec).get_space_group_number()
    except Exception:
        return None


def generate_many(model, z, lengths, angles, n, batch_size, step_lr, device):
    """Generate n structures in chunks of batch_size (keeps GPU memory bounded)."""
    strucs = []
    done = 0
    while done < n:
        bs = min(batch_size, n - done)
        loader = DataLoader(build_batch(z, lengths, angles, bs), batch_size=bs)
        batch = next(iter(loader)).to(device)
        strucs.extend(sample_structures(model, batch, step_lr))
        done += bs
        print(f"    ... {done}/{n}", flush=True)
    return strucs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symmcd-pt", default=DEFAULT_SYMMCD_PT)
    ap.add_argument("--index", type=int, default=19)
    ap.add_argument(
        "--model-path",
        default=None,
        help="DiffCSP model dir (hparams.yaml + *.ckpt). Default None -> "
        "base pretrained DiffCSP-mp20 from HF (most defensible 'this is "
        "intrinsic to DiffCSP'). Pass a run's models/final for the RL model.",
    )
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="generation batch size (chunks --num-samples to fit GPU)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--symprec", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--match",
        action="store_true",
        help="MATCH MODE: sample --num-samples full-CSP structures (composition "
        "fixed, lattice + coords free) and count how many reproduce the SymmCD "
        "target via pymatgen StructureMatcher; report the rate + efficiency factor.",
    )
    ap.add_argument(
        "--output-cif",
        default=(
            "exp_res/tsenn_static_dielectric_layered_uniaxial_symmcd_v1_carryover_"
            "53154012/deliverables/CrTe3W_diffcsp_panelC.cif"
        ),
    )
    ap.add_argument(
        "--free-lattice-check",
        action="store_true",
        help="Also sample with the lattice free (full CSP) and report the SG "
        "distribution DiffCSP lands in for this composition.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z, lengths, angles = load_showcase(Path(args.symmcd_pt), args.index)
    species = [Element.from_Z(zi).symbol for zi in z]
    print(f"Showcase: {species}  cell a,b,c={lengths} angles={angles}")

    suite = DiffCSPSuite(
        model_name="diffcsp",
        sample_cfg=OmegaConf.create({}),
        finetune_cfg=OmegaConf.create({}),
        model_path=args.model_path,
        device=args.device,
    )
    model = suite.load_model()
    model = model.to(suite.device)
    model.eval()
    model.keep_atom_types = True
    step_lr = DEFAULT_STEP_LR["csp"]["mp_20"]

    if args.match:
        # Full CSP: composition fixed, lattice + coords free ("changing the
        # lattice and fractional"). Count accidental reproductions of the target.
        model.keep_lattice = False
        target = load_target(Path(args.symmcd_pt), args.index)
        print(f"\nMATCH MODE: {args.num_samples} full-CSP samples vs target "
              f"{target.composition.reduced_formula} "
              f"SG {space_group(target, args.symprec)}")
        gen = generate_many(model, z, lengths, angles, args.num_samples,
                            args.batch_size, step_lr, suite.device)
        matcher = StructureMatcher()
        matches = sum(1 for g in gen if matcher.fit(g, target))
        n = len(gen)
        sgs = [space_group(g, args.symprec) for g in gen]
        nonp1 = sum(1 for s in sgs if s and s > 2)
        sixfold = sum(1 for s in sgs if s and 168 <= s <= 194)
        print(f"\n=== RESULT (StructureMatcher default tol) ===")
        print(f"  exact-structure matches : {matches} / {n}")
        print(f"  non-P1 (SG>2)           : {nonp1} / {n}")
        print(f"  6-fold (SG 168-194)     : {sixfold} / {n}")
        if matches > 0:
            p = matches / n
            print(f"  match rate p = {p:.2e}  ->  ~{1/p:.0f} samples per hit; "
                  f"SymmCD yields it per sample -> ~{1/p:.0f}x more efficient")
        else:
            ub = 3.0 / n  # rule of three, 95% upper bound
            print(f"  0 matches -> 95% upper bound p <= 3/{n} = {ub:.2e}  ->  "
                  f">= ~{1/ub:.0f}x more efficient (lower bound)")
        return

    # Default: fixed-composition CSP in the showcase cell, write a panel-(c) CIF.
    model.keep_lattice = True

    loader = DataLoader(
        build_batch(z, lengths, angles, args.num_samples),
        batch_size=args.num_samples,
    )
    batch = next(iter(loader)).to(suite.device)
    strucs = sample_structures(model, batch, step_lr)

    sgs = [space_group(s, args.symprec) for s in strucs]
    print(f"\nFixed cell + fixed composition (symprec={args.symprec}):")
    print("  per-sample SG:", sgs)
    counts = Counter(s if s is not None else 0 for s in sgs)
    print("  SG distribution:", dict(counts))
    # representative = most common SG (the typical DiffCSP outcome, not extreme)
    modal_sg = counts.most_common(1)[0][0]
    rep_idx = next(i for i, s in enumerate(sgs) if (s or 0) == modal_sg)
    rep = strucs[rep_idx]
    print(
        f"  representative: sample {rep_idx}, SG {sgs[rep_idx]}, "
        f"formula {rep.composition.reduced_formula}"
    )

    out = Path(args.output_cif)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to(filename=str(out), fmt="cif")
    print(f"  wrote {out}")

    if args.free_lattice_check:
        model.keep_lattice = False
        loader = DataLoader(
            build_batch(z, lengths, angles, args.num_samples),
            batch_size=args.num_samples,
        )
        batch = next(iter(loader)).to(suite.device)
        free = sample_structures(model, batch, DEFAULT_STEP_LR["csp"]["mp_20"])
        free_sgs = [space_group(s, args.symprec) for s in free]
        print("\nFree lattice (full CSP) SG distribution:")
        print("  per-sample SG:", free_sgs)
        print(
            "  SG distribution:",
            dict(Counter(s if s is not None else 0 for s in free_sgs)),
        )


if __name__ == "__main__":
    main()
