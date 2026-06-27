#!/usr/bin/env python3
"""Quantify how much of the TSENN reward is contributed by the scoring
pipeline's symmetry refinement (standardize_structure: refined, symprec 0.1)
versus the structure as actually generated (standardize_structure: none).

Scores the last N steps of eval structures from each run dir under both
standardization modes with the same TSENN dielectric model and compares
property distributions and in-band hit rates.

Example:
    python scripts/rescore_standardize_ablation.py \
        exp_res/..._symmcd_v1_carryover_53154012 \
        exp_res/..._diffcsp_v1_54289718 \
        --labels symmcd diffcsp
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
from pymatgen.io.ase import AseAtomsAdaptor

from rewards.calculators.tsenn_static_dielectric import TSENNStaticDielectric

PROP = "tsenn_static_dielectric_layered_uniaxial"
HIT_LOW, HIT_HIGH = 0.08, 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--last-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-path",
        default="data/dielectric/indep_e3_dielectric_DDP_Lmax2_Lr0.01_bs16_em64_layers2_mul32_best.torch",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def make_calculator(mode: str, model_path: str, device: str, root_dir: str):
    # mirror configs/reward/tsenn_static_dielectric_layered_uniaxial.yaml
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
        standardize_structure=mode,
        standardize_symprec=0.1,
    )


def load_run(run_dir: Path, last_steps: int):
    samples = run_dir / "samples"
    steps = sorted(
        int(p.stem.split("_")[1]) for p in samples.glob("step_*_eval.extxyz")
    )[-last_steps:]
    structures, recorded = [], []
    for step in steps:
        atoms_list = ase_read(samples / f"step_{step:04d}_eval.extxyz", index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        strucs = [AseAtomsAdaptor.get_structure(a) for a in atoms_list]
        rew_path = run_dir / "rewards" / PROP / f"step_{step:04d}.txt"
        props = (
            [float(x) for x in rew_path.read_text().split()]
            if rew_path.exists()
            else [np.nan] * len(strucs)
        )
        structures.extend(strucs)
        recorded.extend(props[: len(strucs)] + [np.nan] * (len(strucs) - len(props)))
    return steps, structures, np.array(recorded)


def score(calc, structures) -> np.ndarray:
    tensors, valid_mask = calc.predict_static_tensor(structures)
    return calc._reduce_tensor_to_scalar(tensors, valid_mask)


def hit_rate(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(((finite >= HIT_LOW) & (finite <= HIT_HIGH)).mean())


def main() -> None:
    args = parse_args()
    labels = args.labels or [d.name for d in args.run_dirs]
    output_dir = args.output_dir or (
        args.run_dirs[-1] / "deliverables" / "standardize_rescore"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        calc_refined = make_calculator("refined", args.model_path, args.device, tmp)
        calc_none = make_calculator("none", args.model_path, args.device, tmp)

        results = {}
        for label, run_dir in zip(labels, args.run_dirs):
            steps, structures, recorded = load_run(run_dir, args.last_steps)
            print(f"[{label}] scoring {len(structures)} structures "
                  f"from steps {steps[0]}..{steps[-1]} ...")
            refined = score(calc_refined, structures)
            raw = score(calc_none, structures)
            results[label] = (recorded, refined, raw)

            with (output_dir / f"rescore_{label}.csv").open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["recorded", "refined", "raw_as_generated"])
                writer.writerows(zip(recorded, refined, raw))

    # summary + plot
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, (label, (recorded, refined, raw)) in zip(axes, results.items()):
        both = np.isfinite(refined) & np.isfinite(raw)
        drop = refined[both] - raw[both]
        print(
            f"\n[{label}] n={both.sum()} scored in both modes\n"
            f"  sanity (recorded vs re-scored refined): "
            f"corr={np.corrcoef(recorded[both & np.isfinite(recorded)], refined[both & np.isfinite(recorded)])[0,1]:.3f}\n"
            f"  property mean:  refined={np.nanmean(refined):.4f}  as-generated={np.nanmean(raw):.4f}\n"
            f"  in-band hit rate [{HIT_LOW},{HIT_HIGH}]:  refined={hit_rate(refined):.1%}  as-generated={hit_rate(raw):.1%}\n"
            f"  refinement subsidy (refined - raw): median={np.median(drop):+.4f}  mean={np.mean(drop):+.4f}"
        )
        ax.scatter(raw[both], refined[both], s=8, alpha=0.4)
        lim = max(np.nanmax(refined[both]), np.nanmax(raw[both])) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.axhspan(HIT_LOW, HIT_HIGH, color="green", alpha=0.08)
        ax.axvspan(HIT_LOW, HIT_HIGH, color="green", alpha=0.08)
        ax.set_xlabel("property, scored as-generated (standardize=none)")
        ax.set_ylabel("property, scored refined (symprec=0.1)")
        ax.set_title(label)
    fig.suptitle("TSENN layered-uniaxial property: refined vs as-generated scoring")
    fig.tight_layout()
    fig.savefig(output_dir / "refined_vs_raw.png", dpi=150)
    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
