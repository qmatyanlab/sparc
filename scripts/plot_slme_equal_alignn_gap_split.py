#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compare_slme_equal_alignn_gap import (
    _apply_publication_style,
    _metrics,
    _plot_scatter,
)


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["train", "valid", "test"]


def _load_split_indices(cache_path: Path, split: str) -> list[int]:
    payload = torch.load(cache_path, map_location="cpu")
    key = f"idx_{split}"
    if key not in payload:
        available = sorted(k for k in payload if str(k).startswith("idx_"))
        raise KeyError(f"{key} not found in {cache_path}; available={available}")
    return [int(i) for i in payload[key]]


def _load_split_index_map(cache_path: Path, splits: list[str]) -> dict[str, set[int]]:
    payload = torch.load(cache_path, map_location="cpu")
    out: dict[str, set[int]] = {}
    for split in splits:
        key = f"idx_{split}"
        if key not in payload:
            available = sorted(k for k in payload if str(k).startswith("idx_"))
            raise KeyError(f"{key} not found in {cache_path}; available={available}")
        out[split] = {int(i) for i in payload[key]}
    return out


def _fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    fit_slope, fit_intercept = np.polyfit(x, y, 1)
    fit_y = fit_slope * x + fit_intercept
    ss_res = float(np.sum((y - fit_y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")


def _finite_xy(df: pd.DataFrame, pred_col: str) -> tuple[np.ndarray, np.ndarray]:
    truth_col = "ground_truth_equal_alignn_gap_slme_percent"
    mask = np.isfinite(df[truth_col]) & np.isfinite(df[pred_col])
    x = df.loc[mask, truth_col].to_numpy(dtype=float)
    y = df.loc[mask, pred_col].to_numpy(dtype=float)
    return x, y


def _plot_single_color(
    df: pd.DataFrame,
    pred_col: str,
    color_index: int,
    fallback_color: str,
    out_path: Path,
) -> None:
    x, y = _finite_xy(df, pred_col)
    fit_r2 = _fit_r2(x, y)

    _apply_publication_style()
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    scatter_color = (
        cycle_colors[color_index] if len(cycle_colors) > color_index else fallback_color
    )

    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(
        x,
        y,
        color=scatter_color,
        s=4,
        alpha=0.22,
        linewidths=0,
        rasterized=True,
    )
    upper = float(np.nanmax([np.nanmax(x), np.nanmax(y), 1.0]))
    xx = np.array([0.0, upper], dtype=float)
    ax.plot(xx, xx, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(0.0, upper * 1.03)
    ax.set_ylim(0.0, upper * 1.03)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("Target η (%)")
    ax.set_ylabel("Predicted η (%)")
    ax.legend(
        handles=[
            Line2D([], [], linestyle="none", label=f"R² = {fit_r2:.3f}"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
        borderpad=0.0,
        handlelength=0.0,
        handletextpad=0.0,
        fontsize=8,
    )
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".eps"))
    plt.close(fig)


def _plot_horizontal_splits(
    split_frames: dict[str, pd.DataFrame],
    pred_col: str,
    out_path: Path,
    png_dpi: int = 600,
    eps_dpi: int = 300,
) -> dict[str, dict[str, float | int]]:
    _apply_publication_style()
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    split_colors = {
        "train": cycle_colors[1] if len(cycle_colors) > 1 else "#000dfc",
        "valid": cycle_colors[2] if len(cycle_colors) > 2 else "#ff7f0e",
        "test": cycle_colors[3] if len(cycle_colors) > 3 else "#1fb312",
    }

    split_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summary: dict[str, dict[str, float | int]] = {}
    finite_values: list[np.ndarray] = []
    for split in SPLITS:
        x, y = _finite_xy(split_frames[split], pred_col)
        split_xy[split] = (x, y)
        summary[split] = _metrics(x, y)
        if len(x):
            finite_values.extend([x, y])

    if finite_values:
        upper = float(np.nanmax([np.nanmax(values) for values in finite_values]))
    else:
        upper = 30.0
    upper = max(30.0, upper)
    lim_upper = upper * 1.03
    ticks = [0, 10, 20, 30]

    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.45), constrained_layout=False)
    fig.subplots_adjust(
        left=0.068,
        right=0.995,
        bottom=0.18,
        top=0.78,
        wspace=0.05,
    )
    for ax, split in zip(axes, SPLITS):
        x, y = split_xy[split]
        ax.scatter(
            x,
            y,
            color=split_colors[split],
            s=4,
            alpha=0.25,
            linewidths=0,
            rasterized=True,
        )
        ax.plot(
            [0.0, upper],
            [0.0, upper],
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_xlim(0.0, lim_upper)
        ax.set_ylim(0.0, lim_upper)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(split)
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    label=(
                        f"R² = {summary[split]['fit_r2']:.3f}\n"
                        f"MAE = {summary[split]['mae']:.3f} %"
                    ),
                ),
            ],
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0,
            borderpad=0.0,
            handlelength=0.0,
            handletextpad=0.0,
            fontsize=8,
        )

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=split_colors[split],
                markersize=4,
                label=f"{split} ({summary[split]['count']})",
            )
            for split in SPLITS
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handletextpad=0.4,
        columnspacing=1.8,
        fontsize=8,
    )
    fig.text(0.5, 0.06, "Target η (%)", ha="center", va="center", fontsize=8)
    fig.text(
        0.047,
        0.50,
        "Predicted η (%)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8,
    )
    fig.savefig(out_path, dpi=png_dpi)
    fig.savefig(out_path.with_suffix(".eps"), dpi=eps_dpi)
    plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged",
        type=Path,
        default=ROOT
        / "runtime/slme_equal_alignn_gap_clean_compare_nid001101/slme_equal_alignn_gap_benchmark_merged.csv",
    )
    parser.add_argument(
        "--optimate-cache",
        type=Path,
        default=ROOT / "data/dielectric/cached_dielectric_spectra_preprocessed_data.pt",
    )
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "runtime/slme_equal_alignn_gap_clean_compare_nid001101/test_set",
    )
    parser.add_argument("--png-dpi", type=int, default=600)
    parser.add_argument("--eps-dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_indices = set(_load_split_indices(args.optimate_cache, args.split))
    merged = pd.read_csv(args.merged)
    split_df = merged.loc[merged["row_index"].astype(int).isin(split_indices)].copy()
    split_df = split_df.sort_values("row_index")
    split_index_map = _load_split_index_map(args.optimate_cache, SPLITS)
    split_frames = {
        split_name: merged.loc[
            merged["row_index"].astype(int).isin(indices)
        ]
        .copy()
        .sort_values("row_index")
        for split_name, indices in split_index_map.items()
    }

    trace_truth = split_df["ground_truth_equal_alignn_gap_slme_percent"].to_numpy(
        dtype=float
    )
    summary: dict[str, Any] = {
        "split": args.split,
        "split_index_count": len(split_indices),
        "merged_row_count": int(len(merged)),
        "filtered_row_count": int(len(split_df)),
        "source_merged": str(args.merged),
        "source_optimate_cache": str(args.optimate_cache),
        "ground_truth_source": (
            "dataset spectrum scored with shared ALIGNN gap, from merged equal-gap CSV"
        ),
        "trace": _metrics(
            trace_truth,
            split_df["trace_equal_alignn_gap_slme_percent"].to_numpy(dtype=float),
        ),
        "tensor": _metrics(
            trace_truth,
            split_df["tensor_equal_alignn_gap_slme_percent"].to_numpy(dtype=float),
        ),
    }

    csv_path = args.output_dir / f"slme_equal_alignn_gap_{args.split}_merged.csv"
    split_df.to_csv(csv_path, index=False)
    with open(args.output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_single_color(
        df=merged,
        pred_col="trace_equal_alignn_gap_slme_percent",
        color_index=1,
        fallback_color="#000dfc",
        out_path=args.output_dir / "gt_vs_trace_equal_alignn_gap_all_blue.png",
    )
    _plot_single_color(
        df=split_df,
        pred_col="trace_equal_alignn_gap_slme_percent",
        color_index=3,
        fallback_color="#1fb312",
        out_path=args.output_dir / f"gt_vs_trace_equal_alignn_gap_{args.split}.png",
    )
    _plot_single_color(
        df=split_df,
        pred_col="trace_equal_alignn_gap_slme_percent",
        color_index=3,
        fallback_color="#1fb312",
        out_path=args.output_dir
        / f"gt_vs_trace_equal_alignn_gap_{args.split}_green.png",
    )
    _plot_scatter(
        split_df,
        pred_col="tensor_equal_alignn_gap_slme_percent",
        title="Tensor spectrum",
        out_path=args.output_dir / f"gt_vs_tensor_equal_alignn_gap_{args.split}.png",
    )
    horizontal_path = (
        args.output_dir / "gt_vs_trace_equal_alignn_gap_train_valid_test_horizontal.png"
    )
    horizontal_summary = _plot_horizontal_splits(
        split_frames=split_frames,
        pred_col="trace_equal_alignn_gap_slme_percent",
        out_path=horizontal_path,
        png_dpi=args.png_dpi,
        eps_dpi=args.eps_dpi,
    )
    horizontal_summary_path = (
        args.output_dir / "gt_vs_trace_equal_alignn_gap_train_valid_test_horizontal.json"
    )
    with open(horizontal_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_merged": str(args.merged),
                "source_optimate_cache": str(args.optimate_cache),
                "truth_col": "ground_truth_equal_alignn_gap_slme_percent",
                "pred_col": "trace_equal_alignn_gap_slme_percent",
                "outputs": {
                    "png": str(horizontal_path),
                    "eps": str(horizontal_path.with_suffix(".eps")),
                },
                "export_dpi": {
                    "png": int(args.png_dpi),
                    "eps": int(args.eps_dpi),
                },
                "splits": horizontal_summary,
            },
            f,
            indent=2,
        )

    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {horizontal_path}", flush=True)


if __name__ == "__main__":
    main()
