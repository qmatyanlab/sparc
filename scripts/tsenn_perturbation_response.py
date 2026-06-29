#!/usr/bin/env python3
"""Perturbation-response probe for the TSENN layered_uniaxial score.

Takes genuinely symmetric symmcd structures (SG retained at symprec 0.01,
tetragonal/trigonal/hexagonal so eps_xx=eps_yy is symmetry-protected), applies
graded controlled perturbations, and measures the TSENN score response with
standardize_structure=none:

  - uniaxial strain along c (positive control: proven sensitivity in
    arXiv:2505.04862)
  - shear strain (xz)
  - in-plane lattice asymmetry a(1+d), b(1-d) (directly breaks eps_xx=eps_yy)
  - random Gaussian atomic displacements (DiffCSP-like incoherent noise)

Outputs response curves and marks DiffCSP's measured noise floor (0.065 A
median symmetrization threshold).

Example:
    python scripts/tsenn_perturbation_response.py \
        exp_res/tsenn_static_dielectric_layered_uniaxial_symmcd_v1_carryover_53154012
"""
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from rewards.calculators.tsenn_static_dielectric import TSENNStaticDielectric

STRAINS = [0.0025, 0.005, 0.01, 0.02]
SIGMAS = [0.01, 0.02, 0.05, 0.08, 0.10]
N_DRAWS = 2
DIFFCSP_FLOOR_ANG = 0.065
DIFFCSP_FLOOR_STRAIN = 0.01
RNG = np.random.default_rng(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--last-steps", type=int, default=20)
    parser.add_argument("--n-refs", type=int, default=50)
    parser.add_argument("--min-prop", type=float, default=0.05)
    parser.add_argument("--max-prop", type=float, default=np.inf)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-path",
        default="data/surrogates/TSENN_static_dielectric_tensor.torch",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def make_calculator(model_path: str, device: str, root_dir: str):
    # mirror configs/reward/tsenn_static_dielectric_layered_uniaxial.yaml,
    # but score the structure exactly as given (no symmetry idealization)
    return TSENNStaticDielectric(
        root_dir=root_dir,
        task="static_dielectric",
        model_path=model_path,
        device=device,
        batch_size=16,
        r_max=6.0,
        out_dim=1,
        em_dim=64,
        lmax=2,
        layers=2,
        mul=32,
        num_neighbors=59.902574690065045,
        scale_0e=8.20143833581954,
        scale_2e=0.6969301341467804,
        dropout_prob=0.4,
        use_batch_norm=False,
        scalar_mode="layered_uniaxial",
        standardize_structure="none",
        standardize_symprec=0.1,
    )


def select_references(
    exp_dir: Path, last_steps: int, n_refs: int, min_prop: float, max_prop: float
):
    samples = exp_dir / "samples"
    steps = sorted(
        int(p.stem.split("_")[1]) for p in samples.glob("step_*_eval.extxyz")
    )[-last_steps:]
    candidates = []
    for step in steps:
        atoms_list = ase_read(samples / f"step_{step:04d}_eval.extxyz", index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        rew = exp_dir / "rewards" / "tsenn_static_dielectric_layered_uniaxial" / f"step_{step:04d}.txt"
        props = [float(x) for x in rew.read_text().split()] if rew.exists() else []
        for i, atoms in enumerate(atoms_list):
            prop = props[i] if i < len(props) else np.nan
            if not np.isfinite(prop) or prop < min_prop or prop > max_prop:
                continue
            s = AseAtomsAdaptor.get_structure(atoms)
            try:
                sg = SpacegroupAnalyzer(s, symprec=0.01).get_space_group_number()
            except Exception:
                continue
            # tetragonal/trigonal/hexagonal: in-plane degeneracy symmetry-protected
            if 75 <= sg <= 194:
                candidates.append((prop, sg, s))
    candidates.sort(key=lambda t: -t[0])
    return candidates[:n_refs]


def shear_xz(structure: Structure, gamma: float) -> Structure:
    F = np.eye(3)
    F[0, 2] = gamma
    new_matrix = structure.lattice.matrix @ F.T
    return Structure(
        Lattice(new_matrix),
        structure.species,
        structure.frac_coords,
        coords_are_cartesian=False,
    )


def displace(structure: Structure, sigma: float) -> Structure:
    cart = structure.cart_coords + RNG.normal(0.0, sigma, structure.cart_coords.shape)
    return Structure(
        structure.lattice, structure.species, cart, coords_are_cartesian=True
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.exp_dir / "deliverables" / "tsenn_perturbation_response"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    refs = select_references(
        args.exp_dir, args.last_steps, args.n_refs, args.min_prop, args.max_prop
    )
    print(f"selected {len(refs)} symmetric references "
          f"(prop {refs[-1][0]:.3f}..{refs[0][0]:.3f}, "
          f"SGs {sorted(set(sg for _, sg, _ in refs))})")

    # build the full perturbation batch
    tasks = []  # (ref_idx, mode, magnitude, structure)
    for idx, (_, _, s) in enumerate(refs):
        tasks.append((idx, "base", 0.0, s))
        for d in STRAINS:
            su = s.copy(); su.apply_strain([0.0, 0.0, d])
            tasks.append((idx, "uniaxial_c", d, su))
            tasks.append((idx, "shear_xz", d, shear_xz(s, d)))
            sp = s.copy(); sp.apply_strain([d, -d, 0.0])
            tasks.append((idx, "inplane_asym", d, sp))
        for sig in SIGMAS:
            for _ in range(N_DRAWS):
                tasks.append((idx, "displacement", sig, displace(s, sig)))

    print(f"scoring {len(tasks)} structures ...")
    with tempfile.TemporaryDirectory() as tmp:
        calc = make_calculator(args.model_path, args.device, tmp)
        tensors, valid = calc.predict_static_tensor([t[3] for t in tasks])
        scores = calc._reduce_tensor_to_scalar(tensors, valid)

    base_score = {}
    rows = []
    for (idx, mode, mag, _), sc in zip(tasks, scores):
        if mode == "base":
            base_score[idx] = sc
        rows.append((idx, mode, mag, sc))
    records = []
    for idx, mode, mag, sc in rows:
        b = base_score.get(idx, np.nan)
        if mode == "base" or not np.isfinite(sc) or not np.isfinite(b):
            continue
        records.append({"ref": idx, "mode": mode, "mag": mag,
                        "base": b, "score": sc, "delta": sc - b})

    with (output_dir / "responses.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    def agg(mode, mag):
        d = np.array([r["delta"] for r in records
                      if r["mode"] == mode and r["mag"] == mag])
        return d

    lattice_modes = ["uniaxial_c", "shear_xz", "inplane_asym"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode, color in zip(lattice_modes, ["tab:blue", "tab:orange", "tab:red"]):
        med = [np.median(agg(mode, d)) for d in STRAINS]
        q1 = [np.percentile(agg(mode, d), 25) for d in STRAINS]
        q3 = [np.percentile(agg(mode, d), 75) for d in STRAINS]
        x = np.array(STRAINS) * 100
        axes[0].plot(x, med, "-o", color=color, label=mode)
        axes[0].fill_between(x, q1, q3, color=color, alpha=0.15)
    axes[0].axvline(DIFFCSP_FLOOR_STRAIN * 100, color="gray", ls=":",
                    label="diffcsp ~1% scale")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("strain (%)")
    axes[0].set_ylabel("Δ layered_uniaxial score")
    axes[0].set_title("coherent lattice perturbations")
    axes[0].legend(fontsize=8)

    med = [np.median(agg("displacement", s)) for s in SIGMAS]
    q1 = [np.percentile(agg("displacement", s), 25) for s in SIGMAS]
    q3 = [np.percentile(agg("displacement", s), 75) for s in SIGMAS]
    axes[1].plot(SIGMAS, med, "-o", color="tab:green", label="random displacement")
    axes[1].fill_between(SIGMAS, q1, q3, color="tab:green", alpha=0.15)
    axes[1].axvline(DIFFCSP_FLOOR_ANG, color="gray", ls=":",
                    label=f"diffcsp floor {DIFFCSP_FLOOR_ANG} Å")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("displacement σ (Å)")
    axes[1].set_ylabel("Δ layered_uniaxial score")
    axes[1].set_title("incoherent atomic displacements")
    axes[1].legend(fontsize=8)
    fig.suptitle("TSENN response to controlled perturbations "
                 "(symmetric symmcd references, standardize=none)")
    fig.tight_layout()
    fig.savefig(output_dir / "perturbation_response.png", dpi=150)

    print(f"\n{'mode':<14}" + "".join(f"{d*100:>9.2f}%" for d in STRAINS))
    for mode in lattice_modes:
        meds = [np.median(agg(mode, d)) for d in STRAINS]
        print(f"{mode:<14}" + "".join(f"{m:>10.4f}" for m in meds))
    print(f"{'displacement':<14}" + "".join(f"{s:>9.2f}Å" for s in SIGMAS))
    print(f"{'':<14}" + "".join(
        f"{np.median(agg('displacement', s)):>10.4f}" for s in SIGMAS))
    base_vals = np.array(list(base_score.values()))
    print(f"\nbase scores: n={np.isfinite(base_vals).sum()} "
          f"median={np.nanmedian(base_vals):.3f}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
