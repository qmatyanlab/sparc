#!/usr/bin/env python3
"""Compare realized space-group distributions and reward trajectories between
two runs (e.g. the symmcd vs diffcsp ablation under the same reward).

Space groups are computed with SpacegroupAnalyzer from samples/step_*_eval.extxyz
(the structures actually scored each loop), so it works for any model suite.

Example:
    python scripts/plot_sg_ablation_compare.py \
        exp_res/tsenn_static_dielectric_layered_uniaxial_symmcd_v1_carryover_53154012 \
        exp_res/tsenn_static_dielectric_layered_uniaxial_diffcsp_v1_54289718 \
        --labels symmcd diffcsp
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs=2)
    parser.add_argument("--labels", nargs=2, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--top-sg", type=int, default=15)
    parser.add_argument(
        "--hist-last-steps",
        type=int,
        default=20,
        help="Number of most recent steps aggregated into the SG histogram",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _step_spacegroups(task: tuple[Path, float]) -> tuple[int, list[int]]:
    path, symprec = task
    step = int(path.stem.split("_")[1])
    atoms_list = ase_read(path, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    sgs = []
    for atoms in atoms_list:
        try:
            structure = AseAtomsAdaptor.get_structure(atoms)
            sgs.append(
                int(
                    SpacegroupAnalyzer(
                        structure, symprec=symprec
                    ).get_space_group_number()
                )
            )
        except Exception:
            continue
    return step, sgs


def load_run_spacegroups(
    run_dir: Path, symprec: float, workers: int
) -> dict[int, list[int]]:
    paths = sorted((run_dir / "samples").glob("step_*_eval.extxyz"))
    if not paths:
        raise FileNotFoundError(f"No step_*_eval.extxyz under {run_dir}/samples")
    with Pool(workers) as pool:
        results = pool.map(_step_spacegroups, [(p, symprec) for p in paths])
    return dict(sorted(results))


def load_reward_trajectory(run_dir: Path) -> tuple[list[int], list[float]]:
    steps, rewards = [], []
    with (run_dir / "metrics.csv").open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                steps.append(int(float(row["step"])))
                rewards.append(float(row["reward mean"]))
            except (KeyError, ValueError):
                continue
    return steps, rewards


def p1_fraction(sgs: list[int]) -> float:
    return sgs.count(1) / len(sgs) if sgs else float("nan")


def save_per_step_csv(
    sg_by_step: dict[int, list[int]], output_path: Path
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "n_structures", "p1_fraction", "top_sg", "sg_counts"])
        for step, sgs in sg_by_step.items():
            counts = Counter(sgs)
            top_sg = counts.most_common(1)[0][0] if counts else ""
            writer.writerow(
                [
                    step,
                    len(sgs),
                    f"{p1_fraction(sgs):.4f}",
                    top_sg,
                    ";".join(f"{sg}:{n}" for sg, n in sorted(counts.items())),
                ]
            )


def main() -> None:
    args = parse_args()
    labels = args.labels or [d.name for d in args.run_dirs]
    output_dir = args.output_dir or (
        args.run_dirs[1] / "deliverables" / "sg_ablation_compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for label, run_dir in zip(labels, args.run_dirs):
        print(f"[{label}] computing space groups for {run_dir} ...")
        sg_by_step = load_run_spacegroups(run_dir, args.symprec, args.workers)
        steps, rewards = load_reward_trajectory(run_dir)
        runs[label] = {"sg_by_step": sg_by_step, "reward": (steps, rewards)}
        save_per_step_csv(sg_by_step, output_dir / f"sg_per_step_{label}.csv")

    colors = {labels[0]: "tab:blue", labels[1]: "tab:red"}

    # P1 fraction vs step
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, data in runs.items():
        steps = list(data["sg_by_step"])
        fracs = [p1_fraction(sgs) for sgs in data["sg_by_step"].values()]
        ax.plot(steps, fracs, label=label, color=colors[label], alpha=0.8)
    ax.set_xlabel("RL step")
    ax.set_ylabel("P1 fraction of scored structures")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.set_title(f"P1 fraction per step (symprec={args.symprec})")
    fig.tight_layout()
    fig.savefig(output_dir / "p1_fraction_vs_step.png", dpi=150)
    plt.close(fig)

    # SG histogram over the last K steps of each run
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (label, data) in zip(axes, runs.items()):
        recent_steps = list(data["sg_by_step"])[-args.hist_last_steps :]
        counts = Counter(
            sg for step in recent_steps for sg in data["sg_by_step"][step]
        )
        total = sum(counts.values())
        top = counts.most_common(args.top_sg)
        xs = np.arange(len(top))
        ax.bar(
            xs,
            [n / total for _, n in top],
            color=[("dimgray" if sg == 1 else colors[label]) for sg, _ in top],
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([str(sg) for sg, _ in top])
        ax.set_xlabel("space group number")
        ax.set_title(
            f"{label}: last {len(recent_steps)} steps (n={total}), "
            f"P1={counts.get(1, 0) / total:.1%}"
        )
    axes[0].set_ylabel("fraction of scored structures")
    fig.suptitle("Realized space-group distribution (P1 bar shown in gray)")
    fig.tight_layout()
    fig.savefig(output_dir / "sg_histogram_recent.png", dpi=150)
    plt.close(fig)

    # Reward trajectories
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, data in runs.items():
        steps, rewards = data["reward"]
        ax.plot(steps, rewards, label=label, color=colors[label], alpha=0.8)
    ax.set_xlabel("RL step")
    ax.set_ylabel("reward mean")
    ax.legend()
    ax.set_title("Reward trajectory")
    fig.tight_layout()
    fig.savefig(output_dir / "reward_vs_step.png", dpi=150)
    plt.close(fig)

    # Console summary
    for label, data in runs.items():
        recent_steps = list(data["sg_by_step"])[-args.hist_last_steps :]
        counts = Counter(
            sg for step in recent_steps for sg in data["sg_by_step"][step]
        )
        total = sum(counts.values())
        top = ", ".join(f"SG{sg}: {n / total:.1%}" for sg, n in counts.most_common(5))
        print(
            f"[{label}] last {len(recent_steps)} steps, n={total}, "
            f"P1={counts.get(1, 0) / total:.1%} | top: {top}"
        )
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
