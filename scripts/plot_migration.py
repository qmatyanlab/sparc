#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--top-sg", type=int, default=15)
    parser.add_argument("--step", type=int, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def detect_spacegroup_number(structure, symprec: float) -> int | None:
    try:
        return int(
            SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number()
        )
    except Exception:
        return None


def load_eval_structures(path: Path) -> list:
    atoms_list = ase_read(path, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]


def load_step_migrations(samples_dir: Path, step: int, symprec: float) -> dict:
    stem = f"step_{step:04d}"
    valid_payload = torch.load(samples_dir / f"{stem}_valid.pt", map_location="cpu")
    stability_rows = read_rows(samples_dir / f"{stem}_stability.csv")
    eval_structures = load_eval_structures(samples_dir / f"{stem}_eval.extxyz")

    valid_count = len(valid_payload)
    eval_count = len(eval_structures)
    relaxed_sg_by_kept_index = {
        idx: detect_spacegroup_number(structure, symprec)
        for idx, structure in enumerate(eval_structures)
    }

    transitions: list[tuple[int | None, int | None]] = []
    kept_count = 0
    no_reference_count = 0
    rejected_count = 0

    for row in stability_rows:
        valid_index = int(row["index"])
        if valid_index >= valid_count:
            continue
        source_sg = valid_payload[valid_index].get("spacegroup")
        kept = row["kept"].strip().lower() == "true"
        kept_index = int(row["kept_index"])
        if kept and kept_index >= 0:
            kept_count += 1
            transitions.append((source_sg, relaxed_sg_by_kept_index.get(kept_index)))
        else:
            rejected_count += 1

    no_reference_path = samples_dir / f"{stem}_stability_no_reference.txt"
    if no_reference_path.exists():
        with no_reference_path.open("r") as handle:
            no_reference_count = sum(
                1
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            )

    return {
        "step": step,
        "valid_count": valid_count,
        "eval_count": eval_count,
        "kept_count": kept_count,
        "rejected_count": rejected_count,
        "no_reference_count": no_reference_count,
        "transitions": transitions,
    }


def discover_steps(samples_dir: Path) -> list[int]:
    steps = []
    for path in sorted(samples_dir.glob("step_*_valid.pt")):
        step = int(path.stem.split("_")[1])
        stem = f"step_{step:04d}"
        if not (samples_dir / f"{stem}_stability.csv").exists():
            continue
        if not (samples_dir / f"{stem}_eval.extxyz").exists():
            continue
        steps.append(step)
    if not steps:
        raise FileNotFoundError(f"No step_*_valid.pt files found under {samples_dir}")
    return steps


def save_step_summary(step_data: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "valid_count",
                "kept_count",
                "eval_count",
                "rejected_count",
                "no_reference_count",
            ],
        )
        writer.writeheader()
        for row in step_data:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def save_transition_csv(step_data: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "source_spacegroup", "relaxed_spacegroup", "count"],
        )
        writer.writeheader()
        for row in step_data:
            counts = Counter(row["transitions"])
            for (source_sg, relaxed_sg), count in sorted(
                counts.items(),
                key=lambda item: (
                    -item[1],
                    -1 if item[0][0] is None else item[0][0],
                    -1 if item[0][1] is None else item[0][1],
                ),
            ):
                writer.writerow(
                    {
                        "step": row["step"],
                        "source_spacegroup": source_sg,
                        "relaxed_spacegroup": relaxed_sg,
                        "count": count,
                    }
                )


def plot_step_counts(step_data: list[dict], output_path: Path) -> None:
    steps = np.array([row["step"] for row in step_data], dtype=int)
    valid = np.array([row["valid_count"] for row in step_data], dtype=int)
    kept = np.array([row["kept_count"] for row in step_data], dtype=int)
    rejected = np.array([row["rejected_count"] for row in step_data], dtype=int)
    no_reference = np.array([row["no_reference_count"] for row in step_data], dtype=int)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, valid, marker="o", label="valid generated")
    ax.plot(steps, kept, marker="o", label="kept / eval")
    ax.plot(steps, rejected, marker="o", label="rejected by filter")
    if np.any(no_reference > 0):
        ax.plot(steps, no_reference, marker="o", label="no reference")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Structure count")
    ax.set_title("Valid → eval migration counts by step")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_transition_matrix(
    step_data: list[dict], output_path: Path, top_sg: int
) -> None:
    all_transitions = []
    source_counts = Counter()
    relaxed_counts = Counter()
    for row in step_data:
        all_transitions.extend(row["transitions"])
        for source_sg, relaxed_sg in row["transitions"]:
            source_counts[source_sg] += 1
            relaxed_counts[relaxed_sg] += 1

    if not all_transitions:
        raise ValueError("No kept transitions found to plot")

    top_sources = [sg for sg, _ in source_counts.most_common(top_sg)]
    top_relaxed = [sg for sg, _ in relaxed_counts.most_common(top_sg)]
    matrix = np.zeros((len(top_sources), len(top_relaxed)), dtype=int)

    source_index = {sg: idx for idx, sg in enumerate(top_sources)}
    relaxed_index = {sg: idx for idx, sg in enumerate(top_relaxed)}
    for source_sg, relaxed_sg in all_transitions:
        if source_sg in source_index and relaxed_sg in relaxed_index:
            matrix[source_index[source_sg], relaxed_index[relaxed_sg]] += 1

    fig_w = max(8, 0.6 * len(top_relaxed) + 4)
    fig_h = max(8, 0.45 * len(top_sources) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(top_relaxed)))
    ax.set_xticklabels([str(sg) for sg in top_relaxed], rotation=45, ha="right")
    ax.set_yticks(range(len(top_sources)))
    ax.set_yticklabels([str(sg) for sg in top_sources])
    ax.set_xlabel("Relaxed / eval space group")
    ax.set_ylabel("Generated valid space group")
    ax.set_title("Space-group migration among kept structures")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_transition_matrix_ordered(step_data: list[dict], output_path: Path) -> None:
    all_transitions = []
    for row in step_data:
        all_transitions.extend(row["transitions"])

    if not all_transitions:
        raise ValueError("No kept transitions found to plot")

    sg_values = sorted(
        {sg for transition in all_transitions for sg in transition if sg is not None}
    )
    matrix = np.zeros((len(sg_values), len(sg_values)), dtype=int)
    sg_index = {sg: idx for idx, sg in enumerate(sg_values)}

    for source_sg, relaxed_sg in all_transitions:
        if source_sg is None or relaxed_sg is None:
            continue
        matrix[sg_index[source_sg], sg_index[relaxed_sg]] += 1

    fig_size = max(10, 0.22 * len(sg_values) + 4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap="magma", aspect="equal")
    tick_step = max(1, len(sg_values) // 20)
    tick_positions = list(range(0, len(sg_values), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [str(sg_values[idx]) for idx in tick_positions], rotation=45, ha="right"
    )
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(sg_values[idx]) for idx in tick_positions])
    ax.set_xlabel("Relaxed / eval space group")
    ax.set_ylabel("Generated valid space group")
    ax.set_title("Space-group migration (ordered by SG number)")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    samples_dir = exp_dir / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing samples directory: {samples_dir}")

    output_dir = args.output_dir or (exp_dir / "deliverables" / "migration")
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [args.step] if args.step is not None else discover_steps(samples_dir)
    step_data = [
        load_step_migrations(samples_dir, step, args.symprec) for step in steps
    ]

    save_step_summary(step_data, output_dir / "step_summary.csv")
    save_transition_csv(step_data, output_dir / "spacegroup_transitions.csv")
    plot_step_counts(step_data, output_dir / "valid_to_eval_counts.png")
    plot_transition_matrix(
        step_data,
        output_dir / "spacegroup_migration_matrix.png",
        top_sg=args.top_sg,
    )
    plot_transition_matrix_ordered(
        step_data,
        output_dir / "spacegroup_migration_matrix_ordered.png",
    )

    print(f"Wrote migration outputs to {output_dir}")


if __name__ == "__main__":
    main()
