#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors, colormaps
from omegaconf import OmegaConf
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--duration-ms", type=int, default=500)
    parser.add_argument("--bins", type=int, default=40)
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


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


def load_run_config(exp_dir: Path) -> Any:
    config_path = exp_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    return OmegaConf.load(require_path(exp_dir / "hparams.yaml", "run config"))


def load_metrics(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv_rows(require_path(path, "metrics file"))
    if not rows:
        raise ValueError(f"No rows found in {path}")
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


def load_scalar_values(
    reward_dir: Path,
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    files = sorted(reward_dir.glob("step_*.txt"))
    if not files:
        raise FileNotFoundError(f"No reward files found under {reward_dir}")
    per_step_values: list[np.ndarray] = []
    steps: list[int] = []
    for path in files:
        values: list[float] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = parse_float(line)
                if value is not None:
                    values.append(value)
        if values:
            per_step_values.append(np.array(values, dtype=float))
            steps.append(int(path.stem.split("_")[-1]))
    if not per_step_values:
        raise ValueError(f"No numeric reward values found under {reward_dir}")
    return np.concatenate(per_step_values), steps, per_step_values


def load_non_pd_stats(reward_dir: Path) -> dict[str, np.ndarray]:
    step_files = sorted(reward_dir.glob("step_*_tensor.npz"))
    if not step_files:
        raise FileNotFoundError(f"No tensor npz files found under {reward_dir}")
    steps: list[int] = []
    valid_counts: list[int] = []
    non_pd_counts: list[int] = []
    non_pd_rates: list[float] = []
    for path in step_files:
        step = int(path.stem.split("_")[1])
        payload = np.load(path)
        tensors = np.asarray(payload["tensor"], dtype=float)
        valid_mask = np.asarray(payload["valid_mask"], dtype=bool)
        step_valid = 0
        step_non_pd = 0
        for tensor, valid in zip(tensors, valid_mask):
            if not bool(valid):
                continue
            step_valid += 1
            eigvals = np.linalg.eigvalsh(tensor)
            if float(np.min(eigvals)) <= 0.0:
                step_non_pd += 1
        steps.append(step)
        valid_counts.append(step_valid)
        non_pd_counts.append(step_non_pd)
        non_pd_rates.append(np.nan if step_valid == 0 else step_non_pd / step_valid)
    return {
        "step": np.array(steps, dtype=float),
        "valid_count": np.array(valid_counts, dtype=float),
        "non_pd_count": np.array(non_pd_counts, dtype=float),
        "non_pd_rate": np.array(non_pd_rates, dtype=float),
    }


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
    initial_prior = np.array(
        [float(row.get("prior_prob_used", row["base_prob"])) for row in initial_rows],
        dtype=float,
    )

    step_files = sorted(policy_dir.glob("step_*.csv"))
    if not step_files:
        raise FileNotFoundError(
            f"No adaptive SG step snapshots found under {policy_dir}"
        )

    steps: list[int] = [-1]
    labels: list[str] = ["initial"]
    current_prob = [initial_prob]
    prior_prob_used = [initial_prior]
    reward_ema = [
        np.array([float(row["reward_ema"]) for row in initial_rows], dtype=float)
    ]

    for path in step_files:
        rows = read_csv_rows(path)
        step = int(path.stem.split("_")[1])
        steps.append(step)
        labels.append(f"step {step}")
        current_prob.append(
            np.array([float(row["current_prob"]) for row in rows], dtype=float)
        )
        prior_prob_used.append(
            np.array(
                [float(row.get("prior_prob_used", row["base_prob"])) for row in rows],
                dtype=float,
            )
        )
        reward_ema.append(
            np.array([float(row["reward_ema"]) for row in rows], dtype=float)
        )

    return {
        "steps": np.array(steps, dtype=int),
        "labels": labels,
        "base_prob": base_prob,
        "prior_prob_used": np.stack(prior_prob_used),
        "current_prob": np.stack(current_prob),
        "reward_ema": np.stack(reward_ema),
    }


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


def save_score_curve(metrics: dict[str, np.ndarray], output_path: Path) -> None:
    steps = metrics["step"]
    mean = metrics["tsenn_static_dielectric_layered_uniaxial mean"]
    std = metrics["tsenn_static_dielectric_layered_uniaxial std"]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(steps, mean, color="#4C78A8", linewidth=2, label="layered-uniaxial mean")
    ax.fill_between(
        steps,
        mean - std,
        mean + std,
        color="#4C78A8",
        alpha=0.2,
        label="layered-uniaxial ± std",
    )
    ax.set_title("Layered-uniaxial dielectric score over RL step")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Layered-uniaxial score")
    ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_reward_curve(metrics: dict[str, np.ndarray], output_path: Path) -> None:
    steps = metrics["step"]
    mean = metrics["reward mean"]
    std = metrics["reward std"]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(steps, mean, color="#E45756", linewidth=2, label="reward mean")
    ax.fill_between(
        steps, mean - std, mean + std, color="#E45756", alpha=0.18, label="reward ± std"
    )
    ax.set_title("Reward over RL step")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_hit_curve(metrics: dict[str, np.ndarray], output_path: Path) -> None:
    steps = metrics["step"]
    hit_rate = metrics["tsenn_static_dielectric_layered_uniaxial_hit_0.08_0.20_rate"]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(steps, hit_rate, color="#54A24B", linewidth=2)
    ax.set_title("Layered-uniaxial hit rate over RL step")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Hit rate in [0.08, 0.20]")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_non_pd_curve(non_pd_stats: dict[str, np.ndarray], output_path: Path) -> None:
    steps = non_pd_stats["step"]
    non_pd_rate = non_pd_stats["non_pd_rate"] * 100.0
    non_pd_count = non_pd_stats["non_pd_count"]
    valid_count = non_pd_stats["valid_count"]

    fig, ax_left = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax_left.plot(steps, non_pd_rate, color="#B279A2", linewidth=2)
    ax_left.set_title("Non-positive-definite rate over RL step")
    ax_left.set_xlabel("RL step")
    ax_left.set_ylabel("Non-PD rate (%)")
    ax_left.set_ylim(bottom=0.0)
    ax_left.grid(axis="y", alpha=0.2)

    ax_right = ax_left.twinx()
    ax_right.plot(steps, non_pd_count, color="#9C755F", linewidth=1.5, linestyle="--")
    ax_right.set_ylabel("Non-PD count")
    if len(valid_count):
        max_valid = float(np.nanmax(valid_count))
        ax_right.set_ylim(0.0, max(max_valid, float(np.nanmax(non_pd_count))) * 1.1)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_layered_histogram(values: np.ndarray, output_path: Path, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.92)
    ax.axvline(0.08, color="#E45756", linewidth=2, linestyle="--")
    ax.axvline(0.20, color="#E45756", linewidth=2, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Layered-uniaxial score aggregation")
    ax.set_xlabel("Layered-uniaxial score")
    ax.set_ylabel("Count")
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


def save_adaptive_sg_heatmap(policy: dict[str, Any], output_path: Path) -> None:
    steps = cast(np.ndarray, policy["steps"])
    current_prob = cast(np.ndarray, policy["current_prob"])
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


def save_adaptive_sg_all_trajectories(
    policy: dict[str, Any], output_path: Path
) -> None:
    steps = cast(np.ndarray, policy["steps"])
    current_prob = cast(np.ndarray, policy["current_prob"])
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = plt.colormaps["viridis"]
    plotted_steps = steps.astype(float)
    plotted_steps[plotted_steps < 0] = -1

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index in range(current_prob.shape[1]):
        sg = int(index + 1)
        ax.plot(
            plotted_steps,
            current_prob[:, index],
            color=cmap(norm(sg)),
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


def save_adaptive_sg_decomposition(
    policy: dict[str, Any], output_path: Path, step: int, rho: float
) -> None:
    steps = cast(np.ndarray, policy["steps"])
    base_prob = cast(np.ndarray, policy["base_prob"])
    current_prob = cast(np.ndarray, policy["current_prob"])
    if step not in steps:
        raise ValueError(f"Requested step {step} not present in adaptive SG policy")
    index = int(np.where(steps == step)[0][-1])
    sg_numbers = np.arange(1, len(base_prob) + 1, dtype=int)
    base_term = rho * base_prob
    mixed_policy = current_prob[index]
    adaptive_term = mixed_policy - base_term
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = plt.colormaps["viridis"]
    bar_colors = [cmap(norm(int(sg))) for sg in sg_numbers]
    ymax = float(
        max(
            base_term.max(initial=0.0),
            adaptive_term.max(initial=0.0),
            mixed_policy.max(initial=0.0),
        )
        * 1.15
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.5), sharey=True, constrained_layout=True
    )
    panels = [
        (base_term, f"base term: rho*pi0(G) [rho={rho:.1f}]"),
        (adaptive_term, f"adaptive term: (1-rho)*pi_tilde(G) [1-rho={1 - rho:.1f}]"),
        (mixed_policy, f"mixed policy: pi_t(G) at step {step}"),
    ]
    ticks = [1, 21, 42, 63, 84, 105, 125, 146, 167, 188, 209, 230]
    for ax, (values, title) in zip(axes, panels):
        ax.bar(sg_numbers, values, color=bar_colors, width=0.85)
        ax.set_title(title)
        ax.set_xlabel("Space group")
        ax.set_xlim(0.5, 230.5)
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(ticks)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Probability mass")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label="Space group color key")
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle(f"Adaptive SG decomposition — step {step}")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_drift_frame(
    policy: dict[str, Any], frame_idx: int, ymax: float
) -> Image.Image:
    labels = cast(list[str], policy["labels"])
    base_prob = cast(np.ndarray, policy["base_prob"])
    current_prob = cast(np.ndarray, policy["current_prob"])
    sg_numbers = np.arange(1, len(base_prob) + 1, dtype=int)
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = colormaps["viridis"]
    bar_colors = [cmap(norm(int(sg))) for sg in sg_numbers]
    ticks = [1, 21, 42, 63, 84, 105, 125, 146, 167, 188, 209, 230]
    step_label = labels[frame_idx]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.5), sharey=True, constrained_layout=True
    )
    for ax, values, panel in zip(
        axes, [base_prob, current_prob[frame_idx]], ["base_prob", "current_prob"]
    ):
        ax.bar(sg_numbers, values, color=bar_colors, width=0.85)
        ax.set_xlim(0.5, 230.5)
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(ticks)
        ax.set_xlabel("Space group")
        ax.set_title(f"Adaptive SG drift — {step_label}")
        ax.grid(axis="y", alpha=0.2)
        ax.text(
            0.02,
            0.96,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
        )
    axes[0].set_ylabel("current_prob")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Space group color key")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).copy()


def save_adaptive_sg_drift_gif(
    policy: dict[str, Any], output_path: Path, duration_ms: int
) -> None:
    base_prob = cast(np.ndarray, policy["base_prob"])
    current_prob = cast(np.ndarray, policy["current_prob"])
    ymax = float(max(base_prob.max(initial=0.0), current_prob.max(initial=0.0)) * 1.08)
    frames = [
        render_drift_frame(policy, frame_idx, ymax)
        for frame_idx in range(len(policy["steps"]))
    ]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def render_plus_reward_frame(
    metrics: dict[str, np.ndarray],
    policy: dict[str, Any],
    frame_idx: int,
    adaptive_ymax: float,
    reward_ymax: float,
) -> Image.Image | None:
    metric_steps = metrics["step"]
    reward_mean = metrics["reward mean"]
    reward_std = metrics["reward std"]
    policy_steps = cast(np.ndarray, policy["steps"])
    labels = cast(list[str], policy["labels"])
    current_prob = cast(np.ndarray, policy["current_prob"])
    step = int(policy_steps[frame_idx])
    if step < 0:
        return None
    step_to_idx = {
        int(s): idx for idx, s in enumerate(metric_steps.astype(int).tolist())
    }
    if step not in step_to_idx:
        return None
    metric_idx = step_to_idx[step]
    sg_numbers = np.arange(1, current_prob.shape[1] + 1, dtype=int)
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = colormaps["viridis"]
    bar_colors = [cmap(norm(int(sg))) for sg in sg_numbers]
    ticks = [1, 21, 42, 63, 84, 105, 125, 146, 167, 188, 209, 230]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 5.8), constrained_layout=True
    )
    ax_left.bar(sg_numbers, current_prob[frame_idx], color=bar_colors, width=0.85)
    ax_left.set_xlabel("Space group")
    ax_left.set_ylabel("current_prob")
    ax_left.set_ylim(0.0, adaptive_ymax)
    ax_left.set_xlim(0.5, 230.5)
    ax_left.set_xticks(ticks)
    ax_left.set_title("Adaptive SG drift")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_left, label="Space group color key")

    ax_right.plot(
        metric_steps, reward_mean, color="#E45756", linewidth=2, label="reward mean"
    )
    ax_right.fill_between(
        metric_steps,
        reward_mean - reward_std,
        reward_mean + reward_std,
        color="#E45756",
        alpha=0.18,
        label="reward ± std",
    )
    ax_right.scatter(
        [metric_steps[metric_idx]],
        [reward_mean[metric_idx]],
        color="black",
        s=45,
        zorder=3,
    )
    ax_right.axvline(step, color="black", linewidth=2)
    ax_right.set_xlabel("RL step")
    ax_right.set_ylabel("Reward")
    ax_right.set_ylim(0.0, reward_ymax)
    ax_right.set_title("Reward objective over RL step")
    ax_right.legend(loc="upper left")

    fig.suptitle(f"Adaptive SG drift + reward objective — {labels[frame_idx]}")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).copy()


def save_adaptive_sg_plus_reward_gif(
    metrics: dict[str, np.ndarray],
    policy: dict[str, Any],
    output_path: Path,
    duration_ms: int,
) -> None:
    current_prob = cast(np.ndarray, policy["current_prob"])
    reward_mean = metrics["reward mean"]
    reward_std = metrics["reward std"]
    adaptive_ymax = float(current_prob.max(initial=0.0) * 1.15)
    reward_ymax = float(np.nanmax(reward_mean + reward_std) * 1.1)
    frames: list[Image.Image] = []
    for frame_idx in range(len(policy["steps"])):
        frame = render_plus_reward_frame(
            metrics, policy, frame_idx, adaptive_ymax, reward_ymax
        )
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise ValueError("No frames generated for adaptive SG + reward GIF")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def main() -> None:
    args = parse_args()
    exp_dir = require_path(args.exp_dir.resolve(), "experiment directory")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else exp_dir / "deliverables_layered_uniaxial"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = load_run_config(exp_dir)
    metrics = load_metrics(exp_dir / "metrics.csv")
    reward_dir = require_path(
        exp_dir / "rewards" / "tsenn_static_dielectric_layered_uniaxial",
        "layered-uniaxial reward directory",
    )
    scalar_values, _, _ = load_scalar_values(reward_dir)
    non_pd_stats = load_non_pd_stats(reward_dir)
    valid_symmetry_counts = load_symmetry_counts(exp_dir / "samples", "valid")
    eval_symmetry_counts = load_symmetry_counts(exp_dir / "samples", "eval")
    adaptive_policy = load_adaptive_sg_policy(
        require_path(exp_dir / "adaptive_spacegroup", "adaptive spacegroup directory")
    )

    latest_step = int(np.max(cast(np.ndarray, adaptive_policy["steps"])))
    prior_mix = float(run_cfg.sample_cfg.adaptive_spacegroup.prior_mix)

    save_layered_histogram(
        scalar_values,
        output_dir / "layered_uniaxial_aggregation.png",
        bins=max(10, int(args.bins)),
    )
    save_score_curve(metrics, output_dir / "layered_uniaxial_vs_step.png")
    save_hit_curve(metrics, output_dir / "layered_uniaxial_hit_vs_step.png")
    save_reward_curve(metrics, output_dir / "reward_vs_step.png")
    save_non_pd_curve(non_pd_stats, output_dir / "non_pd_rate_vs_step.png")
    save_symmetry_distribution(
        valid_symmetry_counts,
        output_dir / "valid_symmetry_distribution.png",
        "Valid symmetry distribution",
    )
    save_symmetry_distribution(
        eval_symmetry_counts,
        output_dir / "eval_symmetry_distribution.png",
        "Evaluated symmetry distribution",
    )
    save_adaptive_sg_heatmap(
        adaptive_policy, output_dir / "adaptive_sg_policy_heatmap.png"
    )
    save_adaptive_sg_all_trajectories(
        adaptive_policy, output_dir / "adaptive_sg_all_trajectories.png"
    )
    save_adaptive_sg_decomposition(
        adaptive_policy,
        output_dir / f"adaptive_sg_decomposition_step_{latest_step:04d}.png",
        step=latest_step,
        rho=prior_mix,
    )
    save_adaptive_sg_drift_gif(
        adaptive_policy,
        output_dir
        / f"adaptive_sg_drift_initial_to_step_{latest_step:04d}_{int(args.duration_ms)}ms.gif",
        duration_ms=int(args.duration_ms),
    )
    save_adaptive_sg_plus_reward_gif(
        metrics,
        adaptive_policy,
        output_dir
        / f"adaptive_sg_plus_reward_to_step_{latest_step:04d}_{int(args.duration_ms)}ms.gif",
        duration_ms=int(args.duration_ms),
    )
    print(f"Saved layered-uniaxial deliverables to {output_dir}")


if __name__ == "__main__":
    main()
