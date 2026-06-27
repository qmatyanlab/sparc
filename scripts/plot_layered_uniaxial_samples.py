#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import warnings
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from ase.io import read as ase_read
from ase.visualize.plot import plot_atoms
from matplotlib.lines import Line2D
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


SUBSCRIPT_TABLE = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-structures", type=int, default=9)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--spacegroups",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit space groups to sample from.",
    )
    parser.add_argument(
        "--per-spacegroup",
        type=int,
        default=None,
        help="Number of samples to select per requested space group.",
    )
    parser.add_argument(
        "--allow-non-positive-definite",
        action="store_true",
        help="Include samples whose predicted dielectric tensor has a non-positive eigenvalue.",
    )
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def format_formula(formula: str) -> str:
    return formula.translate(SUBSCRIPT_TABLE)


def add_subplot_element_legend(ax: Any, atoms: Any) -> None:
    unique_numbers = sorted({int(number) for number in atoms.numbers})
    handles = []
    for atomic_number in unique_numbers:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=jmol_colors[atomic_number],
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=6.5,
                label=chemical_symbols[atomic_number],
            )
        )
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        fontsize=8,
        frameon=True,
        title="Elements",
        title_fontsize=9,
        borderpad=0.4,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    legend.set_in_layout(False)


def add_subplot_title(ax: Any, title: str) -> None:
    ax.text(
        0.5,
        1.04,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        linespacing=1.2,
    )


def read_scalar_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            values.append(float(text))
    return values


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if np.isnan(number):
        return None
    return number


def parse_int(value: str | None) -> int | None:
    number = parse_float(value)
    if number is None:
        return None
    return int(number)


def load_step_ehull_map(stability_path: Path, eval_count: int) -> np.ndarray:
    values = np.full(eval_count, np.nan, dtype=float)
    if not stability_path.exists() or stability_path.stat().st_size == 0:
        return values
    for row in read_csv_rows(stability_path):
        kept = str(row.get("kept", "")).strip().lower() == "true"
        kept_index = parse_int(row.get("kept_index"))
        ehull = parse_float(row.get("energy_above_hull_ev_per_atom"))
        if (
            kept
            and kept_index is not None
            and 0 <= kept_index < eval_count
            and ehull is not None
        ):
            values[kept_index] = ehull
    return values


def load_step_records(
    exp_dir: Path, requested_step: int | None
) -> list[dict[str, Any]]:
    samples_dir = require_path(exp_dir / "samples", "samples directory")
    reward_dir = require_path(
        exp_dir / "rewards" / "tsenn_static_dielectric_layered_uniaxial",
        "layered-uniaxial reward directory",
    )
    records: list[dict[str, Any]] = []

    for tensor_path in sorted(reward_dir.glob("step_*_tensor.npz")):
        step = int(tensor_path.stem.split("_")[1])
        if requested_step is not None and step != requested_step:
            continue
        score_path = reward_dir / f"step_{step:04d}.txt"
        extxyz_path = samples_dir / f"step_{step:04d}_eval.extxyz"
        payload_path = samples_dir / f"step_{step:04d}_eval.pt"
        stability_path = samples_dir / f"step_{step:04d}_stability.csv"
        if (
            not score_path.exists()
            or not extxyz_path.exists()
            or not payload_path.exists()
        ):
            continue

        score_values = read_scalar_values(score_path)
        tensor_payload = np.load(tensor_path)
        tensors = tensor_payload["tensor"]
        valid_mask = tensor_payload["valid_mask"]
        payload = torch.load(payload_path, map_location="cpu")
        if not isinstance(payload, list):
            continue
        atoms_list = ase_read(extxyz_path, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        structures = [
            AseAtomsAdaptor.get_structure(cast(Any, atoms)) for atoms in atoms_list
        ]
        ehull_values = load_step_ehull_map(stability_path, len(payload))

        n = min(
            len(score_values),
            len(tensors),
            len(valid_mask),
            len(payload),
            len(structures),
            len(ehull_values),
        )
        for index in range(n):
            item = payload[index]
            if not isinstance(item, dict):
                continue
            if not bool(valid_mask[index]):
                continue
            tensor = np.asarray(tensors[index], dtype=float)
            structure = structures[index]
            reward = float(score_values[index])
            eigvals = np.linalg.eigvalsh(tensor)
            records.append(
                {
                    "step": step,
                    "index": index,
                    "spacegroup": int(item.get("spacegroup"))
                    if item.get("spacegroup") is not None
                    else None,
                    "reward": reward,
                    "ehull": float(ehull_values[index])
                    if not np.isnan(ehull_values[index])
                    else None,
                    "formula": structure.composition.reduced_formula,
                    "cif": structure.to(fmt="cif"),
                    "eps_xx": float(tensor[0, 0]),
                    "eps_yy": float(tensor[1, 1]),
                    "eps_zz": float(tensor[2, 2]),
                    "eps_xy": float(tensor[0, 1]),
                    "eps_xz": float(tensor[0, 2]),
                    "eps_yz": float(tensor[1, 2]),
                    "min_eig": float(np.min(eigvals)),
                    "positive_definite": bool(np.all(eigvals > 0.0)),
                }
            )
    if not records:
        raise ValueError(
            f"No valid layered-uniaxial sample/tensor records found under {exp_dir}"
        )
    return records


def select_diverse_records(
    records: list[dict[str, Any]], max_items: int
) -> list[dict[str, Any]]:
    candidates = sorted(records, key=lambda row: float(row["reward"]), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[int, int]] = set()
    seen_spacegroups: set[int] = set()
    seen_formulas: set[str] = set()
    seen_steps: set[int] = set()

    for row in candidates:
        row_id = (int(row["step"]), int(row["index"]))
        spacegroup = row.get("spacegroup")
        if (
            spacegroup is None
            or row_id in selected_ids
            or int(spacegroup) in seen_spacegroups
        ):
            continue
        selected.append(row)
        selected_ids.add(row_id)
        seen_spacegroups.add(int(spacegroup))
        seen_formulas.add(str(row["formula"]))
        seen_steps.add(int(row["step"]))
        if len(selected) >= max_items:
            return selected

    for row in candidates:
        row_id = (int(row["step"]), int(row["index"]))
        formula = str(row["formula"])
        step = int(row["step"])
        if row_id in selected_ids or formula in seen_formulas or step in seen_steps:
            continue
        selected.append(row)
        selected_ids.add(row_id)
        seen_formulas.add(formula)
        seen_steps.add(step)
        if len(selected) >= max_items:
            return selected

    for row in candidates:
        row_id = (int(row["step"]), int(row["index"]))
        if row_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row_id)
        if len(selected) >= max_items:
            return selected
    return selected


def select_records_by_spacegroup(
    records: list[dict[str, Any]],
    spacegroups: list[int],
    per_spacegroup: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for spacegroup in spacegroups:
        group_records = [
            record for record in records if record.get("spacegroup") == int(spacegroup)
        ]
        if len(group_records) < per_spacegroup:
            raise ValueError(
                f"Requested {per_spacegroup} samples for SG {spacegroup}, but only {len(group_records)} are available"
            )
        rng.shuffle(group_records)
        group_records.sort(
            key=lambda row: (float(row["reward"]), float(row["min_eig"])),
            reverse=True,
        )
        selected.extend(group_records[:per_spacegroup])
    return selected


def choose_ncols(num_records: int) -> int:
    if num_records <= 6:
        return min(3, num_records)
    if num_records <= 12:
        return 4
    return 4


def save_structure_grid(records: list[dict[str, Any]], output_path: Path) -> None:
    ncols = choose_ncols(len(records))
    nrows = math.ceil(len(records) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 5.6 * nrows), squeeze=False
    )
    fig.subplots_adjust(
        left=0.04, right=0.98, bottom=0.04, top=0.93, wspace=0.12, hspace=0.28
    )
    for ax in axes.flat:
        ax.axis("off")
        ax.set_box_aspect(1)
    for ax, row in zip(axes.flat, records):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            structure = Structure.from_str(row["cif"], fmt="cif")
        atoms = AseAtomsAdaptor.get_atoms(structure)
        plot_atoms(atoms, ax, rotation=("35x,25y,10z"), radii=0.35, show_unit_cell=2)
        add_subplot_element_legend(ax, atoms)
        ehull_text = (
            "n/a" if row.get("ehull") is None else f"{row['ehull']:.3f} eV/atom"
        )
        title = (
            f"{format_formula(str(row['formula']))}\n"
            f"SG {row['spacegroup']} | step {row['step']}\n"
            f"score={row['reward']:.3f} | Ehull={ehull_text}"
        )
        add_subplot_title(ax, title)
        ax.set_axis_off()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_diagonal_tensor_grid(records: list[dict[str, Any]], output_path: Path) -> None:
    ncols = choose_ncols(len(records))
    nrows = math.ceil(len(records) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 4.8 * nrows), squeeze=False, sharey=True
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.08, top=0.92, wspace=0.18, hspace=0.35
    )
    y_values = np.array(
        [row[key] for row in records for key in ("eps_xx", "eps_yy", "eps_zz")],
        dtype=float,
    )
    ymin = float(np.min(y_values))
    ymax = float(np.max(y_values))
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    margin = 0.08 * (ymax - ymin)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    labels = [r"$\varepsilon_{xx}$", r"$\varepsilon_{yy}$", r"$\varepsilon_{zz}$"]

    for ax in axes.flat:
        ax.axis("off")

    for ax, row in zip(axes.flat, records):
        values = [float(row["eps_xx"]), float(row["eps_yy"]), float(row["eps_zz"])]
        ax.bar(labels, values, color=colors, width=0.72)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_ylabel("Dielectric component")
        ax.grid(axis="y", alpha=0.2)
        ehull_text = (
            "n/a" if row.get("ehull") is None else f"{row['ehull']:.3f} eV/atom"
        )
        ax.set_title(
            f"{format_formula(str(row['formula']))}\nSG {row['spacegroup']} | step {row['step']} | score={row['reward']:.3f} | Ehull={ehull_text}",
            fontsize=10,
        )
        ax.axis("on")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_sample_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "step",
        "index",
        "formula",
        "spacegroup",
        "reward",
        "ehull",
        "positive_definite",
        "min_eig",
        "eps_xx",
        "eps_yy",
        "eps_zz",
        "eps_xy",
        "eps_xz",
        "eps_yz",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fieldnames})


def group_records_by_spacegroup(
    records: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        spacegroup = record.get("spacegroup")
        if spacegroup is None:
            continue
        grouped.setdefault(int(spacegroup), []).append(record)
    return grouped


def main() -> None:
    args = parse_args()
    exp_dir = require_path(args.exp_dir.resolve(), "experiment directory")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else exp_dir / "deliverables_layered_uniaxial_samples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_step_records(exp_dir, requested_step=args.step)
    if not args.allow_non_positive_definite:
        records = [record for record in records if bool(record["positive_definite"])]
        if not records:
            raise ValueError(
                "No positive-definite layered-uniaxial sample/tensor records found under the requested selection"
            )
    if args.spacegroups is not None:
        if args.per_spacegroup is None:
            raise ValueError(
                "--per-spacegroup is required when --spacegroups is provided"
            )
        selected = select_records_by_spacegroup(
            records,
            spacegroups=[int(sg) for sg in args.spacegroups],
            per_spacegroup=max(1, int(args.per_spacegroup)),
            seed=int(args.seed),
        )
    else:
        selected = select_diverse_records(
            records, max_items=max(1, int(args.num_structures))
        )

    if args.spacegroups is not None:
        grouped = group_records_by_spacegroup(selected)
        for spacegroup in [int(sg) for sg in args.spacegroups]:
            sg_records = grouped.get(spacegroup, [])
            if not sg_records:
                continue
            save_structure_grid(
                sg_records, output_dir / f"sample_structures_sg{spacegroup}.png"
            )
            save_diagonal_tensor_grid(
                sg_records, output_dir / f"sample_xx_yy_zz_sg{spacegroup}.png"
            )
    else:
        save_structure_grid(selected, output_dir / "sample_structures.png")
        save_diagonal_tensor_grid(selected, output_dir / "sample_xx_yy_zz.png")

    save_sample_csv(selected, output_dir / "sample_tensor_summary.csv")
    print(f"Saved layered-uniaxial sample deliverables to {output_dir}")


if __name__ == "__main__":
    main()
