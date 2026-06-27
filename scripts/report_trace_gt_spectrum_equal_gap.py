#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_pyg_compat() -> None:
    """Compatibility for the gpu_deeph PyG version and PyG-2 pickled caches."""
    import torch_geometric.data as pyg_data_pkg
    import torch_geometric.data.data as pyg_data

    if "torch_geometric.loader" not in sys.modules:
        loader = types.ModuleType("torch_geometric.loader")
        loader.DataLoader = pyg_data_pkg.DataLoader
        sys.modules["torch_geometric.loader"] = loader

    class _Compat:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

        def __setstate__(self, state: Any) -> None:
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    for name in ("DataEdgeAttr", "DataTensorAttr"):
        if not hasattr(pyg_data, name):
            setattr(pyg_data, name, type(name, (_Compat,), {}))

    if "torch_geometric.data.storage" not in sys.modules:
        storage = types.ModuleType("torch_geometric.data.storage")
        for name in ("BaseStorage", "NodeStorage", "EdgeStorage", "GlobalStorage"):
            setattr(storage, name, type(name, (_Compat,), {}))
        sys.modules["torch_geometric.data.storage"] = storage


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
    diff = pred[mask] - truth[mask]
    x = truth[mask]
    y = pred[mask]
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
        "fit_r2": float(fit_r2),
    }


def _load_optimate_df(cache_path: Path) -> pd.DataFrame:
    _install_pyg_compat()
    return torch.load(cache_path, map_location="cpu")["df"]


def _plot_trace_slme(report: pd.DataFrame, out_dir: Path) -> None:
    truth_col = "ground_truth_equal_alignn_gap_slme_percent"
    pred_col = "trace_equal_alignn_gap_slme_percent"
    err_col = "trace_abs_error_percent_points"
    mask = np.isfinite(report[truth_col]) & np.isfinite(report[pred_col])
    x = report.loc[mask, truth_col].to_numpy(dtype=float)
    y = report.loc[mask, pred_col].to_numpy(dtype=float)
    err = report.loc[mask, err_col].to_numpy(dtype=float)
    signed = y - x
    gap = report.loc[mask, "alignn_pred_gap"].to_numpy(dtype=float)
    fit_slope, fit_intercept = np.polyfit(x, y, 1)
    fit_y = fit_slope * x + fit_intercept
    ss_res = float(np.sum((y - fit_y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    fit_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    sc = ax.scatter(x, y, c=gap, s=9, alpha=0.55, cmap="plasma", linewidths=0)
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
    ax.set_xlabel("Ground-truth spectrum SLME, shared gap (%)")
    ax.set_ylabel("Trace spectrum SLME, shared gap (%)")
    ax.set_title("Trace vs dataset spectrum, same ALIGNN gap")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(alpha=0.18, linewidth=0.6)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("ALIGNN gap (eV)")
    fig.savefig(out_dir / "trace_vs_ground_truth_equal_gap_scatter.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    axes[0].hist(err, bins=65, color="#2f6f9f", alpha=0.82)
    axes[0].set_xlabel("Absolute error (% points)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Trace SLME absolute error")
    axes[0].grid(alpha=0.18, linewidth=0.6)

    axes[1].hist(signed, bins=65, color="#d98735", alpha=0.82)
    axes[1].axvline(0.0, color="black", linewidth=1.0)
    axes[1].set_xlabel("Trace - ground truth (% points)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Signed error")
    axes[1].grid(alpha=0.18, linewidth=0.6)

    axes[2].scatter(gap, err, s=8, alpha=0.45, color="#3b7f5f", linewidths=0)
    axes[2].set_xlabel("Shared ALIGNN gap (eV)")
    axes[2].set_ylabel("Absolute error (% points)")
    axes[2].set_title("Error vs shared gap")
    axes[2].grid(alpha=0.18, linewidth=0.6)
    fig.savefig(out_dir / "trace_equal_gap_error_panels.png", dpi=220)
    plt.close(fig)


def _representative_rows(report: pd.DataFrame, max_rows: int = 6) -> list[int]:
    valid = report.dropna(
        subset=[
            "ground_truth_equal_alignn_gap_slme_percent",
            "trace_equal_alignn_gap_slme_percent",
            "trace_abs_error_percent_points",
            "alignn_pred_gap",
        ]
    )
    if valid.empty:
        return []
    err = valid["trace_abs_error_percent_points"].to_numpy(dtype=float)
    quantiles = np.linspace(0.05, 0.95, max_rows)
    chosen: list[int] = []
    for q in quantiles:
        target = float(np.quantile(err, q))
        row = valid.iloc[int(np.argmin(np.abs(err - target)))]
        idx = int(row["row_index"])
        if idx not in chosen:
            chosen.append(idx)
    return chosen


def _predict_trace_spectra(optimate_df: pd.DataFrame, row_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    _install_pyg_compat()
    from rewards.calculators.tsenn.calc import TSENN

    structures = [optimate_df.iloc[i]["pmg_structure"] for i in row_indices]
    calc = TSENN(
        root_dir=str(ROOT / "runtime/_trace_gt_equal_gap_tsenn_cache"),
        task="tsenn_dielectric",
        model_path=str(
            ROOT
            / "data/dielectric/e3_dielectric_optimate_DDP_Lmax2_Lr0.01_bs24_em128_layers4_mul64_best.torch"
        ),
        device="cpu",
        batch_size=16,
        r_max=6.0,
        out_dim=201,
        em_dim=128,
        lmax=2,
        layers=4,
        mul=64,
        num_neighbors=38.85585538243254,
        scale_0e=7.948065,
        dropout_prob=0.4,
        use_batch_norm=False,
        output_mode="trace",
    )
    energies, eps2_trace, valid = calc.predict_epsilon2_iso(
        structures, energy_min=0.0, energy_max=20.0
    )
    if not bool(np.all(valid)):
        eps2_trace[~valid] = np.nan
    return energies, eps2_trace


def _plot_representative_spectra(
    optimate_df: pd.DataFrame, report: pd.DataFrame, out_dir: Path
) -> None:
    rows = _representative_rows(report)
    if not rows:
        return
    energies_trace, eps2_trace = _predict_trace_spectra(optimate_df, rows)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), sharex=True, constrained_layout=True)
    axes_flat = list(axes.ravel())
    for ax, row_index, pred in zip(axes_flat, rows, eps2_trace):
        row = report.loc[report["row_index"] == row_index].iloc[0]
        gt_energy = np.asarray(optimate_df.iloc[row_index]["energies_interp"], dtype=float)
        gt_eps2 = np.asarray(optimate_df.iloc[row_index]["imag_dielectric_interp"], dtype=float)
        gap = float(row["alignn_pred_gap"])
        ax.plot(gt_energy, gt_eps2, color="black", linewidth=1.6, label="dataset")
        ax.plot(energies_trace, pred, color="#2f6f9f", linewidth=1.4, label="trace")
        ax.axvline(gap, color="#d98735", linewidth=1.0, linestyle="--")
        ax.set_xlim(0.0, 8.0)
        ymax = np.nanpercentile(np.concatenate([gt_eps2, pred]), 98)
        if math.isfinite(float(ymax)) and ymax > 0.0:
            ax.set_ylim(bottom=-0.05 * ymax, top=1.15 * ymax)
        ax.set_title(
            f"{row['formula']}  gap={gap:.2f} eV  AE={row['trace_abs_error_percent_points']:.2f}",
            fontsize=10,
        )
        ax.grid(alpha=0.16, linewidth=0.6)
    axes_flat[0].legend(frameon=False, loc="upper right")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\epsilon_2(E)$")
    for ax in axes[-1, :]:
        ax.set_xlabel("Energy (eV)")
    fig.savefig(out_dir / "representative_trace_vs_dataset_spectra.png", dpi=220)
    plt.close(fig)


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime/trace_gt_spectrum_equal_alignn_gap_nid001101",
    )
    parser.add_argument("--skip-spectra", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv(args.merged)
    cols = [
        "row_index",
        "formula",
        "mat_id",
        "ipa_direct_gap",
        "ipa_indirect_gap",
        "alignn_pred_gap",
        "ground_truth_equal_alignn_gap_slme_percent",
        "trace_equal_alignn_gap_slme_percent",
        "trace_abs_error_percent_points",
    ]
    report = merged[cols].copy()
    report["trace_signed_error_percent_points"] = (
        report["trace_equal_alignn_gap_slme_percent"]
        - report["ground_truth_equal_alignn_gap_slme_percent"]
    )

    summary = {
        "output_dir": str(args.output_dir),
        "comparison": "trace predicted dielectric spectrum vs dataset ground-truth dielectric spectrum",
        "shared_bandgap_source": "ALIGNN band_gap prediction used as both direct and indirect gap",
        "integration_source": str(args.merged),
        "metrics": _metrics(
            report["ground_truth_equal_alignn_gap_slme_percent"].to_numpy(dtype=float),
            report["trace_equal_alignn_gap_slme_percent"].to_numpy(dtype=float),
        ),
    }

    report.to_csv(args.output_dir / "trace_gt_spectrum_equal_gap_comparison.csv", index=False)
    (
        report.dropna(subset=["trace_abs_error_percent_points"])
        .sort_values("trace_abs_error_percent_points", ascending=False)
        .head(50)
        .to_csv(args.output_dir / "trace_gt_spectrum_equal_gap_worst_errors.csv", index=False)
    )
    with open(args.output_dir / "trace_gt_spectrum_equal_gap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_trace_slme(report, args.output_dir)

    if not args.skip_spectra:
        print("loading OPTIMATE cache for representative spectra", flush=True)
        optimate_df = _load_optimate_df(args.optimate_cache)
        print("predicting representative trace spectra", flush=True)
        _plot_representative_spectra(optimate_df, report, args.output_dir)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
