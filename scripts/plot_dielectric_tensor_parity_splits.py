#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = [
    ("xx", 0, 0),
    ("yy", 1, 1),
    ("zz", 2, 2),
    ("xy", 0, 1),
    ("yz", 1, 2),
    ("xz", 0, 2),
]
SPLITS = ["train", "valid", "test"]


def _component_label(component: str) -> str:
    return rf"$\varepsilon_{{{component}}}$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data/dielectric/cached_dielectric_preprocessed_data.pt",
    )
    parser.add_argument("--truth-col", default="dielectric_tensor")
    parser.add_argument("--pred-col", default="cart_pred")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime/dielectric_tensor_parity_splits",
    )
    parser.add_argument("--png-name", default="dielectric_tensor_parity_splits.png")
    parser.add_argument("--png-dpi", type=int, default=600)
    parser.add_argument("--eps-dpi", type=int, default=300)
    return parser.parse_args()


def _apply_style() -> None:
    style_path = ROOT / "publication.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "figure.figsize": (7.2, 4.8),
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )


def _to_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _stack_tensor_column(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([_to_array(value) for value in df[col].to_numpy()]).astype(float)


def _metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(x) & np.isfinite(y)
    if not bool(mask.any()):
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "r2": float("nan"),
        }
    diff = y[mask] - x[mask]
    if int(mask.sum()) > 1:
        ss_res = float(np.sum(diff * diff))
        ss_tot = float(np.sum((x[mask] - np.mean(x[mask])) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    else:
        r2 = float("nan")
    return {
        "count": int(mask.sum()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "bias": float(np.mean(diff)),
        "r2": r2,
    }


def _split_indices(payload: dict[str, Any], split: str) -> np.ndarray:
    key = f"idx_{split}"
    if key not in payload:
        raise KeyError(f"Missing split index key: {key}")
    return np.asarray(payload[key], dtype=int)


def _component_limits(values: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()])
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = high - low
    pad = 0.06 * span if span > 0.0 else 0.5
    low -= pad
    high += pad
    if low >= 0.0:
        low = 0.0
    return low, high


def _shared_ticks(low: float, high: float, nbins: int = 5) -> np.ndarray:
    ticks = MaxNLocator(nbins=nbins).tick_values(low, high)
    ticks = ticks[np.isfinite(ticks)]
    ticks = ticks[(ticks >= low) & (ticks <= high)]
    if ticks.size < 2:
        ticks = MaxNLocator(nbins=nbins).tick_values(low, high)
        ticks = ticks[np.isfinite(ticks)]
    if ticks.size < 2:
        ticks = np.linspace(low, high, nbins)
    return ticks.astype(float)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.cache, map_location="cpu")
    df = payload["df"]
    truth = _stack_tensor_column(df, args.truth_col)
    pred = _stack_tensor_column(df, args.pred_col)

    _apply_style()
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    split_colors = {
        "train": cycle[1] if len(cycle) > 1 else "#000dfc",
        "valid": cycle[2] if len(cycle) > 2 else "#ff7f0e",
        "test": cycle[3] if len(cycle) > 3 else "#1fb312",
    }
    split_alpha = {"train": 0.13, "valid": 0.42, "test": 0.48}
    split_size = {"train": 4, "valid": 5, "test": 5}

    metrics_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(
        left=0.062,
        right=0.99,
        bottom=0.095,
        top=0.89,
        wspace=0.18,
        hspace=0.28,
    )
    for ax, (component, i, j) in zip(axes.flat, COMPONENTS):
        component_truth = truth[:, i, j]
        component_pred = pred[:, i, j]
        low, high = _component_limits([component_truth, component_pred])
        ticks = _shared_ticks(low, high)
        low, high = float(ticks[0]), float(ticks[-1])

        annotation_lines = []
        for split in SPLITS:
            idx = _split_indices(payload, split)
            x = component_truth[idx]
            y = component_pred[idx]
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[mask],
                y[mask],
                s=split_size[split],
                color=split_colors[split],
                alpha=split_alpha[split],
                linewidths=0,
                rasterized=True,
            )
            stats = _metrics(x, y)
            metrics_rows.append(
                {
                    "component": component,
                    "split": split,
                    **stats,
                }
            )
            annotation_lines.append(
                f"{split}: MAE {stats['mae']:.3f}, R² {stats['r2']:.3f}"
            )

        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(_component_label(component))
        ax.text(
            0.04,
            0.96,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
        )

    fig.text(
        0.5,
        0.035,
        "Target dielectric tensor component",
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.text(
        0.034,
        0.5,
        "Predicted dielectric tensor component",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8,
    )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=split_colors[split],
            markersize=4,
            label=f"{split} (n={len(_split_indices(payload, split))})",
        )
        for split in SPLITS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )

    png_path = args.output_dir / args.png_name
    eps_path = png_path.with_suffix(".eps")
    fig.savefig(png_path, dpi=args.png_dpi)
    fig.savefig(eps_path, dpi=args.eps_dpi)
    plt.close(fig)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = args.output_dir / "dielectric_tensor_parity_splits_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "cache": str(args.cache),
        "truth_col": args.truth_col,
        "pred_col": args.pred_col,
        "rows": int(len(df)),
        "splits": {
            split: int(len(_split_indices(payload, split))) for split in SPLITS
        },
        "component_order": [component for component, _, _ in COMPONENTS],
        "outputs": {
            "png": str(png_path),
            "eps": str(eps_path),
            "metrics_csv": str(metrics_path),
        },
        "export_dpi": {
            "png": int(args.png_dpi),
            "eps": int(args.eps_dpi),
        },
    }
    summary_path = args.output_dir / "dielectric_tensor_parity_splits_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
