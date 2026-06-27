#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from ase.io import read as ase_read
from ase.visualize.plot import plot_atoms
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from PIL import Image
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


SUBSCRIPT_TABLE = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-structures", type=int, default=9)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument(
        "--band-gap-ma-window",
        type=int,
        default=5,
        help="Moving-average window for the band-gap-vs-step plot (default: 5).",
    )
    parser.add_argument(
        "--compare-valid-steps",
        type=int,
        nargs="+",
        default=None,
        help="RL steps whose valid.pt space-group distributions should be plotted side-by-side.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip animated GIF exports for faster per-run deliverable generation.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
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
    if math.isnan(number):
        return None
    return number


def parse_int(value: str | None) -> int | None:
    number = parse_float(value)
    if number is None:
        return None
    return int(number)


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_run_config(exp_dir: Path) -> Any:
    config_path = exp_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    return OmegaConf.load(require_path(exp_dir / "hparams.yaml", "run config"))


def load_band_gap_reward_settings(cfg: Any) -> dict[str, Any]:
    for prop_cfg in cfg.reward.prop_cfg:
        if str(prop_cfg.name) == "band_gap":
            return {
                "target": str(prop_cfg.target),
                "minv": float(prop_cfg.minv),
                "maxv": float(prop_cfg.maxv),
            }
    raise ValueError("Run config does not define a band_gap reward target")


def linear_scaling(values: np.ndarray, minv: float, maxv: float) -> np.ndarray:
    scaled = (values - minv) / (maxv - minv)
    return np.clip(scaled, 0.0, 1.0)


def compute_band_gap_reward(
    band_gaps: np.ndarray, target: Any, minv: float, maxv: float
) -> np.ndarray:
    target_mode = str(target).strip().lower()
    if target_mode == "ascending":
        return linear_scaling(band_gaps, minv=minv, maxv=maxv)
    target_value = float(target)
    diff = np.abs(band_gaps - target_value)
    return linear_scaling(-diff, minv=-maxv, maxv=-minv)


def load_metrics(metrics_path: Path) -> dict[str, np.ndarray]:
    rows = read_csv_rows(require_path(metrics_path, "metrics file"))
    if not rows:
        raise ValueError(f"No rows found in {metrics_path}")
    if "step" in rows[0]:
        deduped: dict[int, dict[str, str]] = {}
        for row in rows:
            step = parse_int(row.get("step"))
            if step is None:
                continue
            deduped[step] = row
        rows = [deduped[step] for step in sorted(deduped)]
    columns = rows[0].keys()
    data: dict[str, np.ndarray] = {}
    for column in columns:
        values = [parse_float(row.get(column)) for row in rows]
        if any(value is not None for value in values):
            data[column] = np.array(
                [np.nan if value is None else value for value in values], dtype=float
            )
    return data


def load_band_gap_values(
    reward_dir: Path,
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    files = sorted(reward_dir.glob("step_*.txt"))
    if not files:
        raise FileNotFoundError(f"No band gap reward files found under {reward_dir}")
    per_step_values: list[np.ndarray] = []
    steps: list[int] = []
    for path in files:
        values: list[float] = []
        with path.open("r") as handle:
            for line in handle:
                value = parse_float(line)
                if value is not None:
                    values.append(value)
        if values:
            per_step_values.append(np.array(values, dtype=float))
            steps.append(int(path.stem.split("_")[-1]))
    if not per_step_values:
        raise ValueError(f"No numeric band gap values found under {reward_dir}")
    return np.concatenate(per_step_values), steps, per_step_values


def load_step_ehull_map(stability_path: Path, eval_count: int) -> np.ndarray:
    values = np.full(eval_count, np.nan, dtype=float)
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
    exp_dir: Path, reward_settings: dict[str, float]
) -> list[dict[str, Any]]:
    samples_dir = exp_dir / "samples"
    reward_dir = exp_dir / "rewards" / "bandgap"
    records: list[dict[str, Any]] = []
    for eval_pt_path in sorted(samples_dir.glob("step_*_eval.pt")):
        step = int(eval_pt_path.stem.split("_")[1])
        payload = torch.load(eval_pt_path, map_location="cpu")
        if not isinstance(payload, list):
            continue
        extxyz_path = samples_dir / f"step_{step:04d}_eval.extxyz"
        stability_path = samples_dir / f"step_{step:04d}_stability.csv"
        band_gap_path = reward_dir / f"step_{step:04d}.txt"
        if not extxyz_path.exists() or extxyz_path.stat().st_size == 0:
            continue
        if not band_gap_path.exists() or band_gap_path.stat().st_size == 0:
            continue
        if not stability_path.exists() or stability_path.stat().st_size == 0:
            continue
        atoms_list = ase_read(extxyz_path, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        structures = [
            AseAtomsAdaptor.get_structure(cast(Any, atom)) for atom in atoms_list
        ]
        band_gap_values = []
        with band_gap_path.open("r") as handle:
            for line in handle:
                value = parse_float(line)
                if value is not None:
                    band_gap_values.append(value)
        if len(payload) != len(structures) or len(payload) != len(band_gap_values):
            raise ValueError(
                f"Artifact length mismatch for step {step}: eval.pt={len(payload)}, extxyz={len(structures)}, bandgap={len(band_gap_values)}"
            )
        ehull_values = load_step_ehull_map(stability_path, len(payload))
        band_gap_array = np.array(band_gap_values, dtype=float)
        reward_array = compute_band_gap_reward(
            band_gap_array,
            target=reward_settings["target"],
            minv=reward_settings["minv"],
            maxv=reward_settings["maxv"],
        )
        for index, (item, structure, band_gap, reward, ehull) in enumerate(
            zip(payload, structures, band_gap_array, reward_array, ehull_values)
        ):
            if not isinstance(item, dict):
                continue
            spacegroup = parse_int(str(item.get("spacegroup")))
            records.append(
                {
                    "step": step,
                    "index": index,
                    "spacegroup": spacegroup,
                    "band_gap": float(band_gap),
                    "reward": float(reward),
                    "ehull": float(ehull) if not np.isnan(ehull) else None,
                    "formula": structure.composition.reduced_formula,
                    "cif": structure.to(fmt="cif"),
                }
            )
    if not records:
        raise ValueError(f"No step records found under {exp_dir}")
    return records


def load_symmetry_counts(samples_dir: Path, artifact_stem: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    files = sorted(samples_dir.glob(f"step_*_{artifact_stem}.pt"))
    if not files:
        raise FileNotFoundError(
            f"No {artifact_stem}.pt files found under {samples_dir}"
        )
    for path in files:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            spacegroup = parse_int(str(item.get("spacegroup")))
            if spacegroup is not None:
                counts[spacegroup] += 1
    if not counts:
        raise ValueError(f"No spacegroup data found under {samples_dir}")
    return counts


def load_step_symmetry_counts(
    samples_dir: Path, artifact_stem: str, step: int
) -> Counter[int]:
    path = require_path(
        samples_dir / f"step_{step:04d}_{artifact_stem}.pt",
        f"{artifact_stem}.pt artifact for step {step}",
    )
    counts: Counter[int] = Counter()
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected payload type in {path}: {type(payload)!r}")
    for item in payload:
        if not isinstance(item, dict):
            continue
        spacegroup = parse_int(str(item.get("spacegroup")))
        if spacegroup is not None:
            counts[spacegroup] += 1
    if not counts:
        raise ValueError(f"No spacegroup data found in {path}")
    return counts


def load_adaptive_sg_policy(policy_dir: Path) -> dict[str, Any]:
    initial_path = require_path(
        policy_dir / "initial.csv", "initial adaptive SG snapshot"
    )
    initial_rows = read_csv_rows(initial_path)
    if not initial_rows:
        raise ValueError(f"No rows found in {initial_path}")
    base_prob = np.array([float(row["base_prob"]) for row in initial_rows], dtype=float)
    initial_prob = np.array(
        [float(row["current_prob"]) for row in initial_rows], dtype=float
    )

    step_files = sorted(policy_dir.glob("step_*.csv"))
    if not step_files:
        raise FileNotFoundError(
            f"No adaptive SG step snapshots found under {policy_dir}"
        )

    steps: list[int] = [-1]
    labels: list[str] = ["initial"]
    current_prob = [initial_prob]
    reward_ema = [
        np.array([float(row["reward_ema"]) for row in initial_rows], dtype=float)
    ]
    step_count = [np.array([int(row["step_count"]) for row in initial_rows], dtype=int)]

    for path in step_files:
        rows = read_csv_rows(path)
        step = int(path.stem.split("_")[1])
        steps.append(step)
        labels.append(f"step {step}")
        current_prob.append(
            np.array([float(row["current_prob"]) for row in rows], dtype=float)
        )
        reward_ema.append(
            np.array([float(row["reward_ema"]) for row in rows], dtype=float)
        )
        step_count.append(np.array([int(row["step_count"]) for row in rows], dtype=int))

    return {
        "steps": np.array(steps, dtype=int),
        "labels": labels,
        "base_prob": base_prob,
        "current_prob": np.stack(current_prob),
        "reward_ema": np.stack(reward_ema),
        "step_count": np.stack(step_count),
    }


def summarize_step_records(
    step_records: list[dict[str, Any]], key: str
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    grouped: dict[int, list[float]] = {}
    for record in step_records:
        value = record.get(key)
        if value is None:
            continue
        grouped.setdefault(int(record["step"]), []).append(float(value))
    if not grouped:
        raise ValueError(f"No values found for {key}")
    steps = sorted(grouped)
    per_step = [np.array(grouped[step], dtype=float) for step in steps]
    return np.concatenate(per_step), steps, per_step


def try_summarize_step_records(
    step_records: list[dict[str, Any]], key: str
) -> tuple[np.ndarray, list[int], list[np.ndarray]] | None:
    try:
        return summarize_step_records(step_records, key)
    except ValueError:
        return None


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
                markersize=7,
                label=chemical_symbols[atomic_number],
            )
        )
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        fontsize=9,
        frameon=True,
        title="Elements",
        title_fontsize=10,
        borderpad=0.45,
        handletextpad=0.5,
        labelspacing=0.3,
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
        fontsize=13,
        linespacing=1.25,
    )


def save_histogram(
    values: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    bins: int | np.ndarray,
    extra_vlines: list[tuple[float, str, str]] | None = None,
    xlim: tuple[float, float] | None = None,
    crop_to_xlim: bool = False,
) -> None:
    plot_values = values
    if crop_to_xlim and xlim is not None:
        plot_values = values[(values >= xlim[0]) & (values <= xlim[1])]
        if plot_values.size == 0:
            plot_values = values
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.hist(plot_values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
    mean_value = float(np.mean(plot_values))
    median_value = float(np.median(plot_values))
    ax.axvline(mean_value, color="#F58518", linewidth=2, label=f"mean={mean_value:.3f}")
    ax.axvline(
        median_value,
        color="#54A24B",
        linewidth=2,
        linestyle="--",
        label=f"median={median_value:.3f}",
    )
    if extra_vlines:
        for value, label, color in extra_vlines:
            ax.axvline(value, color=color, linewidth=2, linestyle=":", label=label)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_rl_curve(metrics: dict[str, np.ndarray], output_path: Path) -> None:
    steps = metrics["step"]
    reward_mean = metrics["reward mean"]
    reward_std = metrics.get("reward std")
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(steps, reward_mean, color="#4C78A8", linewidth=2, label="reward mean")
    if reward_std is not None:
        lower = reward_mean - reward_std
        upper = reward_mean + reward_std
        ax.fill_between(
            steps, lower, upper, color="#4C78A8", alpha=0.2, label="reward ± std"
        )
    ax.set_title("Mean reward vs RL step (deduped by final row per step)")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    effective_window = min(window, len(values))
    kernel = np.ones(effective_window, dtype=float)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values, dtype=float), kernel, mode="same")
    return numerator / denominator


def save_band_gap_curve(
    metrics: dict[str, np.ndarray], output_path: Path, ma_window: int
) -> None:
    steps = metrics["step"]
    band_gap_mean = metrics["band_gap mean"]
    smoothed = moving_average(band_gap_mean, ma_window)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(
        steps,
        smoothed,
        color="#E45756",
        linewidth=2.5,
        label=f"band gap MA({ma_window})",
    )
    ax.set_title(f"Band gap moving average vs RL step (window={ma_window})")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Band gap (eV)")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_symmetry_distribution(
    counts: Counter[int], output_path: Path, title: str
) -> None:
    spacegroups = np.array(sorted(counts), dtype=int)
    frequencies = np.array([counts[sg] for sg in spacegroups], dtype=int)
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = plt.colormaps["viridis"]
    bar_colors = [cmap(norm(sg)) for sg in spacegroups]
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.bar(spacegroups, frequencies, color=bar_colors, width=0.9)
    ax.set_title(title)
    ax.set_xlabel("Space group")
    ax.set_ylabel("Count")
    ax.set_xlim(
        max(0.5, float(spacegroups.min()) - 1), min(230.5, float(spacegroups.max()) + 1)
    )
    ticks = np.linspace(
        spacegroups.min(), spacegroups.max(), num=min(12, len(spacegroups)), dtype=int
    )
    ax.set_xticks(sorted(set(int(tick) for tick in ticks)))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Space group color key")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_step_symmetry_comparison(
    step_counts: list[tuple[int, Counter[int]]], output_path: Path, title: str
) -> None:
    if not step_counts:
        raise ValueError("No step-specific symmetry counts provided")
    ncols = len(step_counts)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(6.4 * ncols, 5.5),
        constrained_layout=True,
        squeeze=False,
    )
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = plt.colormaps["viridis"]
    for ax, (step, counts) in zip(axes[0], step_counts):
        spacegroups = np.array(sorted(counts), dtype=int)
        frequencies = np.array([counts[sg] for sg in spacegroups], dtype=float)
        frequencies = frequencies / frequencies.sum()
        bar_colors = [cmap(norm(sg)) for sg in spacegroups]
        ax.bar(spacegroups, frequencies, color=bar_colors, width=0.9)
        ax.set_title(f"RL step {step}")
        ax.set_xlabel("Space group")
        ax.set_ylabel("Fraction of valid samples")
        ax.set_xlim(
            max(0.5, float(spacegroups.min()) - 1),
            min(230.5, float(spacegroups.max()) + 1),
        )
        ticks = np.linspace(
            spacegroups.min(),
            spacegroups.max(),
            num=min(12, len(spacegroups)),
            dtype=int,
        )
        ax.set_xticks(sorted(set(int(tick) for tick in ticks)))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist())
    cbar.set_label("Space group color key")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _step_tick_positions(
    steps: np.ndarray, max_ticks: int = 10
) -> tuple[list[int], list[str]]:
    if len(steps) <= max_ticks:
        positions = list(range(len(steps)))
    else:
        positions = sorted(
            set(np.linspace(0, len(steps) - 1, num=max_ticks, dtype=int).tolist())
        )
    labels = ["init" if steps[pos] < 0 else str(int(steps[pos])) for pos in positions]
    return positions, labels


def save_adaptive_sg_heatmap(policy: dict[str, Any], output_path: Path) -> list[int]:
    steps = cast(np.ndarray, policy["steps"])
    current_prob = cast(np.ndarray, policy["current_prob"])
    base_prob = cast(np.ndarray, policy["base_prob"])
    drift = current_prob - base_prob[None, :]
    top_changed = np.argsort(np.abs(drift[-1]))[::-1][:10] + 1

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    image = ax.imshow(
        current_prob.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    xticks, xlabels = _step_tick_positions(steps)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("RL step")
    ax.set_ylabel("Space group")
    ax.set_title("Adaptive SG proposal probability over RL step")
    yticks = np.linspace(0, 229, num=12, dtype=int)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(tick + 1)) for tick in yticks])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("current_prob")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return [int(sg) for sg in top_changed]


def save_adaptive_sg_all_trajectories(
    policy: dict[str, Any], output_path: Path
) -> list[int]:
    steps = cast(np.ndarray, policy["steps"])
    current_prob = cast(np.ndarray, policy["current_prob"])
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = plt.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    plotted_steps = steps.astype(float)
    plotted_steps[plotted_steps < 0] = -1
    for index in range(current_prob.shape[1]):
        sg = int(index + 1)
        color = cmap(norm(sg))
        ax.plot(
            plotted_steps,
            current_prob[:, index],
            color=color,
            linewidth=0.8,
            alpha=0.35,
        )
    ax.set_xlabel("RL step")
    ax.set_ylabel("current_prob")
    ax.set_title("Adaptive SG drift for all space groups")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Space group color key")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return list(range(1, current_prob.shape[1] + 1))


def save_adaptive_sg_gif(
    policy: dict[str, Any], output_path: Path, selected_spacegroups: list[int]
) -> None:
    steps = cast(np.ndarray, policy["steps"])
    labels = cast(list[str], policy["labels"])
    current_prob = cast(np.ndarray, policy["current_prob"])
    indices = np.array([sg - 1 for sg in selected_spacegroups], dtype=int)
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = plt.colormaps["viridis"]
    ymax = float(current_prob[:, indices].max(initial=0.0) * 1.15)
    frames: list[Image.Image] = []

    for frame_idx in range(len(steps)):
        fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
        frame_probs = current_prob[frame_idx, indices]
        sg_numbers = [int(sg) for sg in selected_spacegroups]
        bar_colors = [cmap(norm(sg)) for sg in sg_numbers]
        ax.bar(sg_numbers, frame_probs, color=bar_colors, width=0.85)
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel("Space group")
        ax.set_ylabel("current_prob")
        step_label = (
            labels[frame_idx] if frame_idx < len(labels) else str(int(steps[frame_idx]))
        )
        ax.set_title(f"Adaptive SG drift — {step_label}")
        tick_values = np.linspace(
            min(sg_numbers), max(sg_numbers), num=min(12, len(sg_numbers)), dtype=int
        )
        ax.set_xticks(sorted(set(int(tick) for tick in tick_values)))
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140)
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P"))

    if not frames:
        raise ValueError("No adaptive SG frames generated for GIF")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
    )


def save_bandgap_adaptive_sg_gif(
    metrics: dict[str, np.ndarray], policy: dict[str, Any], output_path: Path
) -> None:
    metric_steps = metrics["step"]
    band_gap_mean = metrics["band_gap mean"]
    band_gap_std = metrics.get("band_gap std")
    policy_steps = cast(np.ndarray, policy["steps"])
    policy_labels = cast(list[str], policy["labels"])
    current_prob = cast(np.ndarray, policy["current_prob"])
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = plt.colormaps["viridis"]
    metric_step_to_index = {
        int(step): idx for idx, step in enumerate(metric_steps.astype(int).tolist())
    }
    frames: list[Image.Image] = []
    adaptive_ymax = float(current_prob.max(initial=0.0) * 1.15)
    bandgap_ymax = float(np.nanmax(band_gap_mean) if len(band_gap_mean) else 1.0)
    if band_gap_std is not None:
        bandgap_ymax = float(
            np.nanmax(band_gap_mean + band_gap_std)
            if len(band_gap_mean)
            else bandgap_ymax
        )
    bandgap_ymax *= 1.1

    sg_numbers = np.arange(1, current_prob.shape[1] + 1, dtype=int)
    tick_values = np.linspace(1, current_prob.shape[1], num=12, dtype=int)
    bar_colors = [cmap(norm(int(sg))) for sg in sg_numbers]

    for frame_idx, policy_step in enumerate(policy_steps):
        if policy_step < 0:
            continue
        step_int = int(policy_step)
        if step_int not in metric_step_to_index:
            continue
        metric_idx = metric_step_to_index[step_int]

        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(14, 5.8), constrained_layout=True
        )

        ax_left.bar(
            sg_numbers,
            current_prob[frame_idx],
            color=bar_colors,
            width=0.9,
        )
        ax_left.set_xlabel("Space group")
        ax_left.set_ylabel("current_prob")
        ax_left.set_ylim(0.0, adaptive_ymax)
        ax_left.set_xlim(0.5, 230.5)
        ax_left.set_xticks(sorted(set(int(tick) for tick in tick_values)))
        ax_left.set_title("Adaptive SG drift")
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_left)
        cbar.set_label("Space group color key")

        ax_right.plot(
            metric_steps,
            band_gap_mean,
            color="#E45756",
            linewidth=2,
            label="band gap mean",
        )
        if band_gap_std is not None:
            ax_right.fill_between(
                metric_steps,
                band_gap_mean - band_gap_std,
                band_gap_mean + band_gap_std,
                color="#E45756",
                alpha=0.18,
                label="band gap ± std",
            )
        ax_right.scatter(
            [metric_steps[metric_idx]],
            [band_gap_mean[metric_idx]],
            color="black",
            s=45,
            zorder=3,
        )
        ax_right.axvline(step_int, color="black", linewidth=2)
        ax_right.set_xlabel("RL step")
        ax_right.set_ylabel("Band gap (eV)")
        ax_right.set_ylim(0.0, bandgap_ymax)
        ax_right.set_title("Band gap objective over RL step")
        ax_right.legend(loc="upper left")

        step_label = (
            policy_labels[frame_idx]
            if frame_idx < len(policy_labels)
            else f"step {step_int}"
        )
        fig.suptitle(f"Adaptive SG drift + band gap objective — {step_label}")

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140)
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P"))

    if not frames:
        raise ValueError("No frames generated for bandgap/adaptive SG GIF")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
    )


def select_diverse_structure_records(
    step_records: list[dict[str, Any]], max_items: int
) -> list[dict[str, Any]]:
    candidates = list(step_records)
    candidates.sort(
        key=lambda row: (
            float(row["reward"]),
            -float(row["ehull"]) if row.get("ehull") is not None else float("-inf"),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[int, int]] = set()
    seen_spacegroups: set[int] = set()
    seen_formulas: set[str] = set()
    seen_steps: set[int] = set()

    for row in candidates:
        spacegroup = row.get("spacegroup")
        row_id = (int(row["step"]), int(row["index"]))
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

    if len(selected) < max_items:
        for row in candidates:
            row_id = (int(row["step"]), int(row["index"]))
            if row_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id)
            if len(selected) >= max_items:
                return selected

    return selected


def save_structure_grid(
    step_records: list[dict[str, Any]], output_path: Path, max_items: int
) -> list[int]:
    selected = select_diverse_structure_records(step_records, max_items=max_items)
    if not selected:
        raise ValueError("No structures available for visualization")
    ncols = min(3, len(selected))
    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 5.6 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        bottom=0.04,
        top=0.93,
        wspace=0.12,
        hspace=0.28,
    )
    for ax in axes.flat:
        ax.axis("off")
        ax.set_box_aspect(1)
    for ax, row in zip(axes.flat, selected):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            structure = Structure.from_str(row["cif"], fmt="cif")
        atoms = AseAtomsAdaptor.get_atoms(structure)
        plot_atoms(atoms, ax, rotation=("35x,25y,10z"), radii=0.35, show_unit_cell=2)
        add_subplot_element_legend(ax, atoms)
        ehull_text = "n/a" if row.get("ehull") is None else f"{row['ehull']:.3f}"
        title = (
            f"{format_formula(row['formula'])}\n"
            f"SG {row['spacegroup']} | step {row['step']}\n"
            f"reward={row['reward']:.3f}, Eg={row['band_gap']:.3f} eV, Ehull={ehull_text} eV/atom"
        )
        add_subplot_title(ax, title)
        ax.set_axis_off()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return [
        int(row["spacegroup"]) for row in selected if row.get("spacegroup") is not None
    ]


def save_step_summary(
    output_path: Path,
    band_gap_steps: list[int],
    band_gap_values: list[np.ndarray],
    ehull_steps: list[int],
    ehull_values: list[np.ndarray],
) -> None:
    rows: list[dict[str, Any]] = []
    band_gap_map = {
        step: values for step, values in zip(band_gap_steps, band_gap_values)
    }
    ehull_map = {step: values for step, values in zip(ehull_steps, ehull_values)}
    all_steps = sorted(set(band_gap_map) | set(ehull_map))
    for step in all_steps:
        bg_values = band_gap_map.get(step)
        eh_values = ehull_map.get(step)
        rows.append(
            {
                "step": step,
                "band_gap_count": 0 if bg_values is None else len(bg_values),
                "band_gap_mean": "" if bg_values is None else float(np.mean(bg_values)),
                "band_gap_median": ""
                if bg_values is None
                else float(np.median(bg_values)),
                "ehull_count": 0 if eh_values is None else len(eh_values),
                "ehull_mean": "" if eh_values is None else float(np.mean(eh_values)),
                "ehull_median": ""
                if eh_values is None
                else float(np.median(eh_values)),
            }
        )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {exp_dir}")

    output_dir = (args.output_dir or exp_dir / "deliverables").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_symmetry_path = output_dir / "symmetry_distribution.png"
    if legacy_symmetry_path.exists():
        legacy_symmetry_path.unlink()
    for legacy_name in [
        "adaptive_sg_topk_trajectories.png",
        "adaptive_sg_topk_trajectories.gif",
    ]:
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    run_cfg = load_run_config(exp_dir)
    reward_settings = load_band_gap_reward_settings(run_cfg)
    metrics = load_metrics(exp_dir / "metrics.csv")
    band_gap_values, band_gap_steps, band_gap_per_step = load_band_gap_values(
        exp_dir / "rewards" / "bandgap"
    )
    step_records = load_step_records(exp_dir, reward_settings)
    ehull_summary = try_summarize_step_records(step_records, "ehull")
    if ehull_summary is None:
        ehull_values = np.array([], dtype=float)
        ehull_steps = []
        ehull_per_step = []
    else:
        ehull_values, ehull_steps, ehull_per_step = ehull_summary
    valid_symmetry_counts = load_symmetry_counts(
        exp_dir / "samples", artifact_stem="valid"
    )
    eval_symmetry_counts = load_symmetry_counts(
        exp_dir / "samples", artifact_stem="eval"
    )
    ehull_bins = np.arange(0.0, 0.1000001 + 0.002, 0.002)

    save_histogram(
        band_gap_values,
        output_dir / "band_gap_aggregation.png",
        title="Band gap aggregation",
        xlabel="Band gap (eV)",
        bins=args.bins,
        extra_vlines=(
            [
                (
                    float(reward_settings["target"]),
                    f"target={float(reward_settings['target']):.2f} eV",
                    "#E45756",
                )
            ]
            if str(reward_settings["target"]).strip().lower() != "ascending"
            else []
        ),
    )
    if ehull_values.size:
        save_histogram(
            ehull_values,
            output_dir / "energy_above_hull_aggregation.png",
            title="Energy above hull aggregation (cropped to 0.1 eV/atom)",
            xlabel="Energy above hull (eV/atom)",
            bins=ehull_bins,
            extra_vlines=[(0.1, "stability threshold=0.1", "#E45756")],
            xlim=(0.0, 0.1),
            crop_to_xlim=True,
        )
    save_symmetry_distribution(
        valid_symmetry_counts,
        output_dir / "valid_symmetry_distribution.png",
        title="Valid symmetry distribution",
    )
    if args.compare_valid_steps:
        compare_steps = sorted(set(args.compare_valid_steps))
        step_counts = [
            (
                step,
                load_step_symmetry_counts(exp_dir / "samples", "valid", step),
            )
            for step in compare_steps
        ]
        step_slug = "_vs_".join(f"step_{step:04d}" for step in compare_steps)
        save_step_symmetry_comparison(
            step_counts,
            output_dir / f"valid_symmetry_distribution_{step_slug}.png",
            title="Valid symmetry distribution by RL step",
        )
    save_symmetry_distribution(
        eval_symmetry_counts,
        output_dir / "eval_symmetry_distribution.png",
        title="Evaluated symmetry distribution",
    )
    save_rl_curve(metrics, output_dir / "rl_vs_step.png")
    save_band_gap_curve(
        metrics,
        output_dir / "band_gap_vs_step.png",
        ma_window=max(1, int(args.band_gap_ma_window)),
    )
    sampled_spacegroups = save_structure_grid(
        step_records,
        output_dir / "structure_samples.png",
        max_items=max(1, args.num_structures),
    )
    save_step_summary(
        output_dir / "aggregation_step_summary.csv",
        band_gap_steps,
        band_gap_per_step,
        ehull_steps,
        ehull_per_step,
    )

    adaptive_top_changed: list[int] = []
    adaptive_trajectory_spacegroups: list[int] = []
    adaptive_policy_dir = exp_dir / "adaptive_spacegroup"
    if adaptive_policy_dir.exists():
        adaptive_policy = load_adaptive_sg_policy(adaptive_policy_dir)
        adaptive_top_changed = save_adaptive_sg_heatmap(
            adaptive_policy,
            output_dir / "adaptive_sg_policy_heatmap.png",
        )
        adaptive_trajectory_spacegroups = save_adaptive_sg_all_trajectories(
            adaptive_policy,
            output_dir / "adaptive_sg_all_trajectories.png",
        )
        if not args.no_gif:
            save_adaptive_sg_gif(
                adaptive_policy,
                output_dir / "adaptive_sg_all_trajectories.gif",
                selected_spacegroups=adaptive_trajectory_spacegroups,
            )
        if not args.no_gif and "band_gap mean" in metrics:
            save_bandgap_adaptive_sg_gif(
                metrics,
                adaptive_policy,
                output_dir / "bandgap_adaptive_sg_combo.gif",
            )

    with (output_dir / "plot_metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "band_gap_target_ev",
                "ehull_crop_max_ev_per_atom",
                "ehull_bin_width_ev_per_atom",
                "structure_sampling_strategy",
                "sampled_spacegroups",
                "valid_symmetry_source",
                "eval_symmetry_source",
                "adaptive_policy_source",
                "adaptive_top_changed_spacegroups",
                "adaptive_trajectory_spacegroups",
                "bandgap_adaptive_combo_gif",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "band_gap_target_ev": reward_settings["target"],
                "ehull_crop_max_ev_per_atom": 0.1,
                "ehull_bin_width_ev_per_atom": 0.002,
                "structure_sampling_strategy": "diverse_across_spacegroups_then_formula_then_step",
                "sampled_spacegroups": ",".join(str(sg) for sg in sampled_spacegroups),
                "valid_symmetry_source": "valid.pt",
                "eval_symmetry_source": "eval.pt",
                "adaptive_policy_source": "adaptive_spacegroup/*.csv"
                if adaptive_policy_dir.exists()
                else "",
                "adaptive_top_changed_spacegroups": ",".join(
                    str(sg) for sg in adaptive_top_changed
                ),
                "adaptive_trajectory_spacegroups": ",".join(
                    str(sg) for sg in adaptive_trajectory_spacegroups
                ),
                "bandgap_adaptive_combo_gif": "bandgap_adaptive_sg_combo.gif"
                if adaptive_policy_dir.exists()
                and "band_gap mean" in metrics
                and not args.no_gif
                else "",
            }
        )

    print(f"Saved deliverables to {output_dir}")


if __name__ == "__main__":
    main()
