#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import torch
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from rewards.calculators.tsenn_static_dielectric import TSENNStaticDielectric


EPS_NUM_DEFAULT = 1e-8
FAMILY_RANGES: tuple[tuple[str, int, int], ...] = (
    ("triclinic", 1, 2),
    ("monoclinic", 3, 15),
    ("orthorhombic", 16, 74),
    ("tetragonal", 75, 142),
    ("trigonal", 143, 167),
    ("hexagonal", 168, 194),
    ("cubic", 195, 230),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV outputs. Defaults to <exp_dir>/dielectric_layered_anisotropy_eval for a single run, or ./dielectric_layered_anisotropy_eval for multiple runs.",
    )
    parser.add_argument(
        "--artifact-stem",
        type=str,
        default="eval",
        help="Artifact stem to load from samples/step_*_<stem>.extxyz (default: eval).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit RL steps to include.",
    )
    parser.add_argument(
        "--max-steps-per-run",
        type=int,
        default=None,
        help="Optional cap on the number of step files to load per run.",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Optional global cap on the number of structures scored.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Number of refined structures to score per TSENN call (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="TSENN inference batch size (default: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override TSENN dielectric checkpoint path.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=0.01,
        help="Symmetry tolerance for realized SG detection and refinement (default: 0.01).",
    )
    parser.add_argument(
        "--eps-num",
        type=float,
        default=EPS_NUM_DEFAULT,
        help="Numerical stabilizer in anisotropy formulas (default: 1e-8).",
    )
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def resolve_output_dir(exp_dirs: list[Path], output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    if len(exp_dirs) == 1:
        return (exp_dirs[0] / "dielectric_layered_anisotropy_eval").resolve()
    return Path.cwd().resolve() / "dielectric_layered_anisotropy_eval"


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root
        / "data"
        / "dielectric"
        / "indep_e3_dielectric_DDP_Lmax2_Lr0.01_bs16_em64_layers2_mul32_best.torch"
    ).resolve()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def parse_step_from_path(path: Path, artifact_stem: str) -> int:
    suffix = f"_{artifact_stem}"
    stem = path.stem
    if not stem.startswith("step_") or not stem.endswith(suffix):
        raise ValueError(f"Unexpected artifact name: {path.name}")
    return int(stem[len("step_") : -len(suffix)])


def crystal_family_from_spacegroup(spacegroup: int | None) -> str:
    if spacegroup is None:
        return "unknown"
    for family, start, end in FAMILY_RANGES:
        if start <= spacegroup <= end:
            return family
    return "unknown"


def family_order_key(name: str) -> tuple[int, str]:
    for index, (family, _, _) in enumerate(FAMILY_RANGES):
        if family == name:
            return index, name
    if name == "all":
        return len(FAMILY_RANGES), name
    return len(FAMILY_RANGES) + 1, name


def load_structures(path: Path) -> list[Any]:
    atoms_list = ase_read(path, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [AseAtomsAdaptor.get_structure(cast(Any, atoms)) for atoms in atoms_list]


def detect_realized_spacegroup(structure: Any, symprec: float) -> int | None:
    try:
        return int(
            SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number()
        )
    except Exception:
        return None


def refine_structure(structure: Any, symprec: float) -> Any | None:
    try:
        return SpacegroupAnalyzer(structure, symprec=symprec).get_refined_structure()
    except Exception:
        return None


def iter_artifact_paths(
    exp_dir: Path,
    artifact_stem: str,
    steps: set[int] | None,
    max_steps_per_run: int | None,
) -> list[Path]:
    samples_dir = require_path(exp_dir / "samples", f"samples directory for {exp_dir}")
    paths = sorted(samples_dir.glob(f"step_*_{artifact_stem}.extxyz"))
    if steps is not None:
        paths = [
            path for path in paths if parse_step_from_path(path, artifact_stem) in steps
        ]
    if max_steps_per_run is not None:
        paths = paths[:max_steps_per_run]
    return paths


def compute_layered_metrics(
    tensor: np.ndarray, eps_num: float
) -> tuple[float, float, float]:
    eps_xx = float(tensor[0, 0])
    eps_yy = float(tensor[1, 1])
    eps_zz = float(tensor[2, 2])
    eps_perp = 0.5 * (eps_xx + eps_yy)
    layered = abs(eps_zz - eps_perp) / (abs(eps_zz) + abs(eps_perp) + eps_num)
    inplane_mismatch = abs(eps_xx - eps_yy) / (abs(eps_xx) + abs(eps_yy) + eps_num)
    uniaxial = layered * (1.0 - inplane_mismatch)
    return layered, inplane_mismatch, uniaxial


def chunked(values: list[int], chunk_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def summarize_metric(values: np.ndarray) -> dict[str, float | int | str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "p90": np.nan}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    header = (
        "family",
        "count",
        "A_mean",
        "A_median",
        "A_p90",
        "U_mean",
        "U_median",
        "U_p90",
        "B_mean",
        "B_median",
        "B_p90",
    )
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["family"]),
                    str(row["count"]),
                    format_float(row["A_mean"]),
                    format_float(row["A_median"]),
                    format_float(row["A_p90"]),
                    format_float(row["U_mean"]),
                    format_float(row["U_median"]),
                    format_float(row["U_p90"]),
                    format_float(row["B_mean"]),
                    format_float(row["B_median"]),
                    format_float(row["B_p90"]),
                ]
            )
        )


def format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.6f}"


def main() -> None:
    args = parse_args()
    exp_dirs = [
        require_path(path.resolve(), f"experiment directory {path}")
        for path in args.exp_dirs
    ]
    output_dir = resolve_output_dir(exp_dirs, args.output_dir)
    model_path = require_path(
        resolve_model_path(args.model_path), "TSENN model checkpoint"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    calculator = TSENNStaticDielectric(
        root_dir=str(output_dir / "tsenn_runtime"),
        task="static_dielectric",
        model_path=str(model_path),
        device=resolve_device(args.device),
        batch_size=int(args.batch_size),
        scalar_mode="trace_mean",
        standardize_structure="none",
    )

    selected_steps = set(args.steps) if args.steps is not None else None
    records: list[dict[str, Any]] = []
    refined_structures: list[Any] = []
    refined_record_indices: list[int] = []

    total_loaded = 0
    stop_loading = False
    for exp_dir in exp_dirs:
        artifact_paths = iter_artifact_paths(
            exp_dir=exp_dir,
            artifact_stem=args.artifact_stem,
            steps=selected_steps,
            max_steps_per_run=args.max_steps_per_run,
        )
        for artifact_path in artifact_paths:
            step = parse_step_from_path(artifact_path, args.artifact_stem)
            structures = load_structures(artifact_path)
            for index, structure in enumerate(structures):
                spacegroup = detect_realized_spacegroup(structure, args.symprec)
                record: dict[str, Any] = {
                    "exp_dir": str(exp_dir),
                    "artifact_path": str(artifact_path),
                    "step": step,
                    "index": index,
                    "formula": structure.composition.reduced_formula,
                    "realized_spacegroup": spacegroup,
                    "family": crystal_family_from_spacegroup(spacegroup),
                    "refined": False,
                    "predicted": False,
                    "eps_xx": np.nan,
                    "eps_yy": np.nan,
                    "eps_zz": np.nan,
                    "layered_anisotropy_A": np.nan,
                    "inplane_mismatch_B": np.nan,
                    "uniaxial_score_U": np.nan,
                }
                refined = refine_structure(structure, args.symprec)
                if refined is not None:
                    record["refined"] = True
                    refined_record_indices.append(len(records))
                    refined_structures.append(refined)
                records.append(record)
                total_loaded += 1
                if (
                    args.max_structures is not None
                    and total_loaded >= args.max_structures
                ):
                    stop_loading = True
                    break
            if stop_loading:
                break
        if stop_loading:
            break

    if not records:
        raise RuntimeError("No saved structures found for the requested inputs.")

    if not refined_structures:
        raise RuntimeError("All loaded structures failed refinement; nothing to score.")

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    all_refined_indices = list(range(len(refined_structures)))
    for chunk in chunked(all_refined_indices, args.chunk_size):
        chunk_structures = [refined_structures[index] for index in chunk]
        tensors, valid_mask = calculator.predict_static_tensor(chunk_structures)
        for local_index, refined_index in enumerate(chunk):
            record = records[refined_record_indices[refined_index]]
            if not bool(valid_mask[local_index]):
                continue
            tensor = tensors[local_index]
            layered, inplane_mismatch, uniaxial = compute_layered_metrics(
                tensor=tensor,
                eps_num=float(args.eps_num),
            )
            record["predicted"] = True
            record["eps_xx"] = float(tensor[0, 0])
            record["eps_yy"] = float(tensor[1, 1])
            record["eps_zz"] = float(tensor[2, 2])
            record["layered_anisotropy_A"] = float(layered)
            record["inplane_mismatch_B"] = float(inplane_mismatch)
            record["uniaxial_score_U"] = float(uniaxial)

    summary_rows: list[dict[str, Any]] = []
    families = [family for family, _, _ in FAMILY_RANGES] + ["unknown"]
    for family in families:
        family_records = [record for record in records if record["family"] == family]
        if not family_records:
            continue
        a_stats = summarize_metric(
            np.array(
                [record["layered_anisotropy_A"] for record in family_records],
                dtype=float,
            )
        )
        b_stats = summarize_metric(
            np.array(
                [record["inplane_mismatch_B"] for record in family_records], dtype=float
            )
        )
        u_stats = summarize_metric(
            np.array(
                [record["uniaxial_score_U"] for record in family_records], dtype=float
            )
        )
        summary_rows.append(
            {
                "family": family,
                "total_structures": len(family_records),
                "count": a_stats["count"],
                "A_mean": a_stats["mean"],
                "A_median": a_stats["median"],
                "A_p90": a_stats["p90"],
                "B_mean": b_stats["mean"],
                "B_median": b_stats["median"],
                "B_p90": b_stats["p90"],
                "U_mean": u_stats["mean"],
                "U_median": u_stats["median"],
                "U_p90": u_stats["p90"],
            }
        )

    all_a = summarize_metric(
        np.array([record["layered_anisotropy_A"] for record in records], dtype=float)
    )
    all_b = summarize_metric(
        np.array([record["inplane_mismatch_B"] for record in records], dtype=float)
    )
    all_u = summarize_metric(
        np.array([record["uniaxial_score_U"] for record in records], dtype=float)
    )
    summary_rows.append(
        {
            "family": "all",
            "total_structures": len(records),
            "count": all_a["count"],
            "A_mean": all_a["mean"],
            "A_median": all_a["median"],
            "A_p90": all_a["p90"],
            "B_mean": all_b["mean"],
            "B_median": all_b["median"],
            "B_p90": all_b["p90"],
            "U_mean": all_u["mean"],
            "U_median": all_u["median"],
            "U_p90": all_u["p90"],
        }
    )
    summary_rows.sort(key=lambda row: family_order_key(str(row["family"])))

    per_structure_path = output_dir / "per_structure_scores.csv"
    family_summary_path = output_dir / "family_summary.csv"
    write_csv(
        per_structure_path,
        rows=records,
        fieldnames=[
            "exp_dir",
            "artifact_path",
            "step",
            "index",
            "formula",
            "realized_spacegroup",
            "family",
            "refined",
            "predicted",
            "eps_xx",
            "eps_yy",
            "eps_zz",
            "layered_anisotropy_A",
            "inplane_mismatch_B",
            "uniaxial_score_U",
        ],
    )
    write_csv(
        family_summary_path,
        rows=summary_rows,
        fieldnames=[
            "family",
            "total_structures",
            "count",
            "A_mean",
            "A_median",
            "A_p90",
            "B_mean",
            "B_median",
            "B_p90",
            "U_mean",
            "U_median",
            "U_p90",
        ],
    )

    predicted_count = sum(bool(record["predicted"]) for record in records)
    refined_count = sum(bool(record["refined"]) for record in records)
    print(f"Loaded {len(records)} structures from {len(exp_dirs)} run(s).")
    print(
        f"Refined {refined_count} structures; predicted tensors for {predicted_count}."
    )
    print(f"Per-structure CSV: {per_structure_path}")
    print(f"Family summary CSV: {family_summary_path}")
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
