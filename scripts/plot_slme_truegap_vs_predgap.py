#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(truth) & np.isfinite(pred)
    if not bool(mask.any()):
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "median_ae": float("nan"),
            "bias": float("nan"),
            "corr": float("nan"),
            "fit_slope": float("nan"),
            "fit_intercept": float("nan"),
            "fit_r2": float("nan"),
        }

    x = truth[mask]
    y = pred[mask]
    diff = y - x
    fit_slope, fit_intercept = np.polyfit(x, y, 1)
    fit_y = fit_slope * x + fit_intercept
    ss_res = float(np.sum((y - fit_y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    fit_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return {
        "count": int(mask.sum()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "median_ae": float(np.median(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "corr": float(np.corrcoef(x, y)[0, 1]),
        "fit_slope": float(fit_slope),
        "fit_intercept": float(fit_intercept),
        "fit_r2": fit_r2,
    }


def _plot_scatter(
    merged: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    mask = np.isfinite(merged[truth_col]) & np.isfinite(merged[pred_col])
    x = merged.loc[mask, truth_col].to_numpy(dtype=float)
    y = merged.loc[mask, pred_col].to_numpy(dtype=float)
    ae = np.abs(y - x)
    fit_slope, fit_intercept = np.polyfit(x, y, 1)
    fit_y = fit_slope * x + fit_intercept
    ss_res = float(np.sum((y - fit_y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    fit_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    sc = ax.scatter(x, y, c=ae, s=9, cmap="viridis", alpha=0.55, linewidths=0)
    upper = float(np.nanmax([np.nanmax(x), np.nanmax(y), 1.0]))
    xx = np.array([0.0, upper], dtype=float)
    ax.plot(xx, xx, color="black", linewidth=1.0, label="identity")
    ax.plot(
        xx,
        fit_slope * xx + fit_intercept,
        color="#d62728",
        linewidth=1.2,
        label=f"fit: y={fit_slope:.3f}x+{fit_intercept:.3f}, R^2={fit_r2:.3f}",
    )
    ax.set_xlim(0.0, upper * 1.03)
    ax.set_ylim(0.0, upper * 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(alpha=0.18, linewidth=0.6)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Absolute error (% points)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_summary(merged: pd.DataFrame, summary: dict[str, Any], out_path: Path) -> None:
    truth_col = "ground_truth_truegap_slme_percent"
    pairs = [
        ("tensor_predgap_slme_percent", "Tensor"),
        ("trace_predgap_slme_percent", "Trace"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)

    axes[0].bar(
        [label for _, label in pairs],
        [summary[label.lower()]["mae"] for _, label in pairs],
        color=["#2f6f9f", "#d98735"],
    )
    axes[0].set_ylabel("MAE (% points)")
    axes[0].set_title("True-vs-predicted-gap SLME MAE")
    axes[0].grid(axis="y", alpha=0.18, linewidth=0.6)

    for pred_col, label in pairs:
        mask = np.isfinite(merged[truth_col]) & np.isfinite(merged[pred_col])
        err = (
            merged.loc[mask, pred_col].to_numpy(dtype=float)
            - merged.loc[mask, truth_col].to_numpy(dtype=float)
        )
        axes[1].hist(np.abs(err), bins=60, histtype="step", linewidth=1.7, label=label)
        axes[2].hist(err, bins=60, histtype="step", linewidth=1.7, label=label)

    axes[1].set_xlabel("Absolute error (% points)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Absolute error")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18, linewidth=0.6)

    axes[2].set_xlabel("Prediction - truth (% points)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Signed error")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.18, linewidth=0.6)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surrogate-merged",
        type=Path,
        default=ROOT
        / "runtime/slme_mae_fully_surrogate_nid001001/slme_surrogate_benchmark_merged.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime/slme_truegap_predgap_compare_nid001101",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.surrogate_merged)
    merged = raw.rename(
        columns={
            "ground_truth_slme_percent": "ground_truth_truegap_slme_percent",
            "tensor_surrogate_slme_percent": "tensor_predgap_slme_percent",
            "trace_surrogate_slme_percent": "trace_predgap_slme_percent",
        }
    ).copy()
    for pred_col, err_col in [
        ("tensor_predgap_slme_percent", "tensor_abs_error_percent_points"),
        ("trace_predgap_slme_percent", "trace_abs_error_percent_points"),
    ]:
        merged[err_col] = np.abs(
            merged[pred_col] - merged["ground_truth_truegap_slme_percent"]
        )

    summary: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "comparison": (
            "x = dataset ground-truth spectrum with true IPA direct/indirect gaps; "
            "y = predicted spectrum with ALIGNN-predicted gap used as direct/indirect gap"
        ),
        "source": str(args.surrogate_merged),
        "ground_truth_source": (
            "imag_dielectric_interp + ipa_direct_gap/ipa_indirect_gap via "
            "KK -> absorption -> pymatgen.slme"
        ),
        "predicted_bandgap_source": (
            "ALIGNN band_gap prediction used as both direct and indirect gap"
        ),
        "tensor": _metrics(
            merged["ground_truth_truegap_slme_percent"].to_numpy(dtype=float),
            merged["tensor_predgap_slme_percent"].to_numpy(dtype=float),
        ),
        "trace": _metrics(
            merged["ground_truth_truegap_slme_percent"].to_numpy(dtype=float),
            merged["trace_predgap_slme_percent"].to_numpy(dtype=float),
        ),
    }

    merged_path = args.output_dir / "slme_truegap_predgap_benchmark_merged.csv"
    merged.to_csv(merged_path, index=False)
    worst_cols = [
        "row_index",
        "formula",
        "mat_id",
        "ipa_direct_gap",
        "ipa_indirect_gap",
        "alignn_pred_gap",
        "ground_truth_truegap_slme_percent",
    ]
    for label, pred_col, err_col in [
        ("tensor", "tensor_predgap_slme_percent", "tensor_abs_error_percent_points"),
        ("trace", "trace_predgap_slme_percent", "trace_abs_error_percent_points"),
    ]:
        (
            merged.dropna(subset=[err_col])
            .sort_values(err_col, ascending=False)
            .head(50)[worst_cols + [pred_col, err_col]]
            .to_csv(args.output_dir / f"{label}_worst_errors.csv", index=False)
        )

    with open(args.output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_scatter(
        merged,
        truth_col="ground_truth_truegap_slme_percent",
        pred_col="tensor_predgap_slme_percent",
        title="Tensor spectrum, predicted ALIGNN gap",
        xlabel="Ground truth spectrum, true-gap SLME (%)",
        ylabel="Predicted spectrum, predicted-gap SLME (%)",
        out_path=args.output_dir / "gt_truegap_vs_tensor_predgap.png",
    )
    _plot_scatter(
        merged,
        truth_col="ground_truth_truegap_slme_percent",
        pred_col="trace_predgap_slme_percent",
        title="Trace spectrum, predicted ALIGNN gap",
        xlabel="Ground truth spectrum, true-gap SLME (%)",
        ylabel="Predicted spectrum, predicted-gap SLME (%)",
        out_path=args.output_dir / "gt_truegap_vs_trace_predgap.png",
    )
    _plot_summary(merged, summary, args.output_dir / "slme_truegap_predgap_errors.png")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {merged_path}", flush=True)


if __name__ == "__main__":
    main()
