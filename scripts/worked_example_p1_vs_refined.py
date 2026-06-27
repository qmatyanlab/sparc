#!/usr/bin/env python3
"""Worked example: for a few DiffCSP structures, show side-by-side what the
generator actually emitted (as-generated, P1 at strict tolerance) versus the
symmetry-idealized cell the reward pipeline actually scored (refined at
symprec 0.1). Demonstrates that the reward is computed on a different object
than the deliverable.

Example:
    python scripts/worked_example_p1_vs_refined.py \
        exp_res/tsenn_static_dielectric_layered_uniaxial_diffcsp_v1_54289718 \
        --step 100
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from rewards.calculators.tsenn_static_dielectric import TSENNStaticDielectric

PROP = "tsenn_static_dielectric_layered_uniaxial"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir", type=Path)
    p.add_argument("--step", type=int, default=None, help="default: last step")
    p.add_argument("--n", type=int, default=4, help="number of examples to show")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--model-path",
        default="data/dielectric/indep_e3_dielectric_DDP_Lmax2_Lr0.01_bs16_em64_layers2_mul32_best.torch",
    )
    return p.parse_args()


def make_calc(mode, model_path, device, root):
    return TSENNStaticDielectric(
        root_dir=root, task="static_dielectric", model_path=model_path,
        device=device, batch_size=16, r_max=6.0, out_dim=1, em_dim=64,
        lmax=2, layers=2, mul=32, num_neighbors=59.902574690065045,
        scale_0e=8.20143833581954, scale_2e=0.6969301341467804,
        dropout_prob=0.4, use_batch_norm=False, scalar_mode="layered_uniaxial",
        standardize_structure=mode, standardize_symprec=0.1,
    )


def sg(structure, symprec):
    try:
        return SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    samples = args.exp_dir / "samples"
    if args.step is None:
        args.step = max(int(p.stem.split("_")[1]) for p in samples.glob("step_*_eval.extxyz"))

    atoms_list = ase_read(samples / f"step_{args.step:04d}_eval.extxyz", index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    strucs = [AseAtomsAdaptor.get_structure(a) for a in atoms_list]
    rew = args.exp_dir / "rewards" / PROP / f"step_{args.step:04d}.txt"
    recorded = [float(x) for x in rew.read_text().split()] if rew.exists() else [np.nan] * len(strucs)

    # pick examples that look symmetric at 0.1 but are P1 at strict tolerance,
    # ordered by recorded score
    picks = []
    for i, s in enumerate(strucs):
        sg01, sg001 = sg(s, 0.1), sg(s, 0.001)
        if sg01 and sg01 != 1 and (sg001 or 1) == 1:
            picks.append((recorded[i] if i < len(recorded) else np.nan, i, s, sg01, sg001))
    picks.sort(key=lambda t: -(t[0] if np.isfinite(t[0]) else -1))
    picks = picks[: args.n]

    with tempfile.TemporaryDirectory() as tmp:
        calc_refined = make_calc("refined", args.model_path, args.device, tmp)
        calc_none = make_calc("none", args.model_path, args.device, tmp)
        sel = [s for _, _, s, _, _ in picks]
        t_ref, v_ref = calc_refined.predict_static_tensor(sel)
        s_ref = calc_refined._reduce_tensor_to_scalar(t_ref, v_ref)
        t_raw, v_raw = calc_none.predict_static_tensor(sel)
        s_raw = calc_none._reduce_tensor_to_scalar(t_raw, v_raw)

    print(f"\nStep {args.step}, {args.exp_dir.name}")
    print("Each row: one generated structure. 'AS-GENERATED' = what DiffCSP emitted "
          "(scored with standardize=none).")
    print("'REFINED' = symmetry-idealized cell the pipeline actually scored "
          "(standardize=refined, symprec=0.1).\n")
    hdr = f"{'#':>3} {'formula':>12} | {'SG@0.1':>7} {'SG@0.001':>9} | " \
          f"{'score_refined':>13} {'score_raw':>10} {'recorded':>9}"
    print(hdr)
    print("-" * len(hdr))
    for k, (rec, idx, s, sg01, sg001) in enumerate(picks):
        eps_ref = t_ref[k]
        diag = np.diag(eps_ref) if v_ref[k] else [np.nan] * 3
        print(f"{idx:>3} {s.composition.reduced_formula:>12} | "
              f"{sg01:>7} {str(sg001):>9} | "
              f"{s_ref[k]:>13.4f} {s_raw[k]:>10.4f} {rec:>9.4f}")
        print(f"      refined eps diag (xx,yy,zz): "
              f"{diag[0]:.2f}, {diag[1]:.2f}, {diag[2]:.2f}  "
              f"-> in-plane |exx-eyy|={abs(diag[0]-diag[1]):.3f}")

    print(f"\nInterpretation: SG@0.1 is the label the pipeline assigns; SG@0.001=1 (P1) "
          f"is the true symmetry of the\ngenerated cell. 'recorded' matches 'score_refined' "
          f"(pipeline scores the refined cell). 'score_raw' is what\nthe as-generated P1 "
          f"structure scores — the deliverable's actual value.")


if __name__ == "__main__":
    main()
