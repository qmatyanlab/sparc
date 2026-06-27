#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors, colormaps
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--duration-ms", type=int, default=500)
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if text == "":
        return float("nan")
    return float(text)


def parse_step(path: Path) -> int:
    if path.name == "initial.csv":
        return -1
    return int(path.stem.split("_")[-1])


def discover_snapshot_paths(adaptive_dir: Path) -> list[Path]:
    paths = [
        path
        for path in adaptive_dir.glob("*.csv")
        if path.name == "initial.csv" or path.name.startswith("step_")
    ]
    if not paths:
        raise FileNotFoundError(
            f"No adaptive SG CSV snapshots found under {adaptive_dir}"
        )
    return sorted(paths, key=parse_step)


def load_snapshots(paths: list[Path]) -> list[dict]:
    snapshots: list[dict] = []
    for path in paths:
        rows = read_rows(path)
        if not rows:
            continue
        snapshots.append(
            {
                "path": path,
                "step": parse_step(path),
                "spacegroups": np.array(
                    [int(row["spacegroup"]) for row in rows], dtype=int
                ),
                "base_prob": np.array(
                    [parse_float(row.get("base_prob")) for row in rows], dtype=float
                ),
                "prior_prob_used": np.array(
                    [
                        parse_float(row.get("prior_prob_used", row.get("base_prob")))
                        for row in rows
                    ],
                    dtype=float,
                ),
                "current_prob": np.array(
                    [parse_float(row.get("current_prob")) for row in rows], dtype=float
                ),
                "reward_ema": np.array(
                    [parse_float(row.get("reward_ema")) for row in rows], dtype=float
                ),
            }
        )
    if not snapshots:
        raise ValueError("No non-empty adaptive SG snapshots found")
    return snapshots


def compute_global_ymax(snapshots: list[dict]) -> float:
    stacked = np.concatenate(
        [
            np.stack(
                [
                    snapshot["base_prob"],
                    snapshot["prior_prob_used"],
                    snapshot["current_prob"],
                ],
                axis=0,
            ).reshape(-1)
            for snapshot in snapshots
        ]
    )
    ymax = float(np.nanmax(stacked))
    return ymax * 1.08 if np.isfinite(ymax) and ymax > 0 else 1.0


def render_frame(snapshot: dict, ymax: float) -> Image.Image:
    spacegroups = snapshot["spacegroups"]
    prior = snapshot["prior_prob_used"]
    current = snapshot["current_prob"]

    norm = colors.Normalize(vmin=int(spacegroups.min()), vmax=int(spacegroups.max()))
    cmap = colormaps["viridis"]
    bar_colors = cmap(norm(spacegroups))

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), sharey=True, constrained_layout=True
    )
    tick_positions = [1, 21, 42, 63, 84, 105, 125, 146, 167, 188, 209, 230]

    step = snapshot["step"]
    title = (
        "Adaptive SG drift — initial"
        if step < 0
        else f"Adaptive SG drift — step {step:04d}"
    )

    for ax, values, series_label in zip(
        axes,
        [prior, current],
        ["prior_prob_used", "current_prob"],
    ):
        ax.bar(spacegroups, values, color=bar_colors, width=0.85)
        ax.set_xlim(0.5, 230.5)
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(tick_positions)
        ax.set_xlabel("Space group")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.text(
            0.02,
            0.96,
            series_label,
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
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer)
    return image.copy()


def save_gif(frames: list[Image.Image], output_path: Path, duration_ms: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    adaptive_dir = require_path(
        exp_dir / "adaptive_spacegroup", "adaptive spacegroup directory"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else exp_dir / "adaptive_spacegroup"
    )
    snapshot_paths = discover_snapshot_paths(adaptive_dir)
    snapshots = load_snapshots(snapshot_paths)
    ymax = compute_global_ymax(snapshots)
    frames = [render_frame(snapshot, ymax) for snapshot in snapshots]
    output_path = output_dir / "adaptive_spacegroup.gif"
    save_gif(frames, output_path, args.duration_ms)
    print(f"Saved adaptive SG GIF to {output_path}")


if __name__ == "__main__":
    main()
