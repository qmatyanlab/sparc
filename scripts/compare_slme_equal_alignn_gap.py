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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.solar.slme import slme


ROOT = Path(__file__).resolve().parents[1]

INV_HBARC_EV_UM = 5.067726
PUBLICATION_STYLE = ROOT / "publication.mplstyle"


def _apply_publication_style() -> None:
    if PUBLICATION_STYLE.exists():
        plt.style.use(str(PUBLICATION_STYLE))
        # NERSC gpu_deeph does not currently provide the external `latex`
        # executable required by publication.mplstyle's text.usetex=True.
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
                "mathtext.fontset": "custom",
                "mathtext.rm": "Liberation Sans",
                "mathtext.it": "Liberation Sans:italic",
                "mathtext.bf": "Liberation Sans:bold",
                "ps.fonttype": 42,
                "pdf.fonttype": 42,
            }
        )


def _install_torch_geometric_pickle_shim() -> None:
    """Allow PyG-2 pickled Data objects to load in older gpu_deeph PyG."""
    import torch_geometric.data.data as pyg_data

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


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    w = np.zeros_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def _kk_eps1_from_eps2(omega_eV: np.ndarray, eps2: np.ndarray) -> np.ndarray:
    omega = omega_eV.astype(float).copy()
    omega[0] = max(float(omega[0]), 1e-12)
    weights = _trapz_weights(omega)
    omega_sq = omega * omega
    denom = omega_sq[None, :] - omega_sq[:, None]
    np.fill_diagonal(denom, np.inf)
    matrix = (omega[None, :] / denom) * weights[None, :]
    np.fill_diagonal(matrix, 0.0)
    return 1.0 + (2.0 / math.pi) * (eps2 @ matrix.T)


def _absorption_cm_inv(
    energies_ev: np.ndarray, eps1: np.ndarray, eps2: np.ndarray
) -> np.ndarray:
    abs_eps = np.sqrt(eps1 * eps1 + eps2 * eps2)
    term = np.sqrt(np.maximum(abs_eps - eps1, 0.0))
    alpha_um_inv = math.sqrt(2.0) * INV_HBARC_EV_UM * energies_ev[None, :] * term
    return alpha_um_inv * 1.0e4


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
    corr = float(np.corrcoef(x, y)[0, 1]) if int(mask.sum()) > 1 else float("nan")
    if int(mask.sum()) > 1:
        fit_slope, fit_intercept = np.polyfit(x, y, 1)
        fit_y = fit_slope * x + fit_intercept
        ss_res = float(np.sum((y - fit_y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        fit_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    else:
        fit_slope = fit_intercept = fit_r2 = float("nan")
    return {
        "count": int(mask.sum()),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "median_ae": float(np.median(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "corr": corr,
        "fit_slope": float(fit_slope),
        "fit_intercept": float(fit_intercept),
        "fit_r2": float(fit_r2),
    }


def _load_optimate_df(cache_path: Path) -> pd.DataFrame:
    _install_torch_geometric_pickle_shim()
    payload = torch.load(cache_path, map_location="cpu")
    return payload["df"]


def _compute_ground_truth_equal_gap(
    optimate_df: pd.DataFrame,
    alignn_gap: np.ndarray,
    thickness_um: float,
    temperature_k: float,
) -> np.ndarray:
    energies = np.asarray(optimate_df.iloc[0]["energies_interp"], dtype=float)
    eps2 = np.stack(optimate_df["imag_dielectric_interp"].to_numpy()).astype(float)
    eps2 = np.nan_to_num(eps2, nan=0.0, posinf=0.0, neginf=0.0)
    eps1 = _kk_eps1_from_eps2(energies, eps2)
    alpha_cm_inv = _absorption_cm_inv(energies, eps1, eps2)

    out = np.full(len(optimate_df), np.nan, dtype=float)
    thickness_m = float(thickness_um) * 1.0e-6
    for i, gap in enumerate(alignn_gap):
        if i and i % 1000 == 0:
            print(f"computed {i}/{len(out)} ground-truth SLME rows", flush=True)
        if not math.isfinite(float(gap)) or float(gap) <= 0.0:
            continue
        try:
            out[i] = float(
                slme(
                    energies,
                    alpha_cm_inv[i],
                    float(gap),
                    float(gap),
                    thickness=thickness_m,
                    temperature=float(temperature_k),
                    absorbance_in_inverse_centimeters=True,
                    cut_off_absorbance_below_direct_allowed_gap=True,
                    plot_current_voltage=False,
                )
            )
        except Exception:
            out[i] = np.nan
    return out


def _plot_scatter(
    merged: pd.DataFrame,
    pred_col: str,
    title: str,
    out_path: Path,
) -> None:
    truth_col = "ground_truth_equal_alignn_gap_slme_percent"
    mask = np.isfinite(merged[truth_col]) & np.isfinite(merged[pred_col])
    x = merged.loc[mask, truth_col].to_numpy(dtype=float)
    y = merged.loc[mask, pred_col].to_numpy(dtype=float)
    fit_slope, fit_intercept = np.polyfit(x, y, 1)
    fit_y = fit_slope * x + fit_intercept
    ss_res = float(np.sum((y - fit_y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    fit_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    _apply_publication_style()
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    scatter_color = cycle_colors[1] if len(cycle_colors) > 1 else "#000dfc"

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
    legend_handles = [
        Line2D([], [], linestyle="none", label=f"R² = {fit_r2:.3f}"),
    ]
    ax.legend(
        handles=legend_handles,
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
    plt.close(fig)


def _plot_summary(merged: pd.DataFrame, summary: dict[str, Any], out_path: Path) -> None:
    truth_col = "ground_truth_equal_alignn_gap_slme_percent"
    pairs = [
        ("tensor_equal_alignn_gap_slme_percent", "Tensor"),
        ("trace_equal_alignn_gap_slme_percent", "Trace"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)

    axes[0].bar(
        [label for _, label in pairs],
        [summary[label.lower()]["mae"] for _, label in pairs],
        color=["#2f6f9f", "#d98735"],
    )
    axes[0].set_ylabel("MAE (% points)")
    axes[0].set_title("Equal-gap SLME MAE")
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
        "--optimate-cache",
        type=Path,
        default=ROOT / "data/dielectric/cached_dielectric_spectra_preprocessed_data.pt",
    )
    parser.add_argument(
        "--surrogate-merged",
        type=Path,
        default=ROOT
        / "runtime/slme_mae_fully_surrogate_nid001001/slme_surrogate_benchmark_merged.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime/slme_equal_alignn_gap_clean_compare_nid001101",
    )
    parser.add_argument("--thickness-um", type=float, default=0.3)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("loading OPTIMATE dielectric cache", flush=True)
    optimate_df = _load_optimate_df(args.optimate_cache)
    surrogate = pd.read_csv(args.surrogate_merged)
    alignn_gap = surrogate["alignn_pred_gap"].to_numpy(dtype=float)

    print("computing equal-ALIGNN-gap ground truth SLME", flush=True)
    gt_equal = _compute_ground_truth_equal_gap(
        optimate_df=optimate_df,
        alignn_gap=alignn_gap,
        thickness_um=args.thickness_um,
        temperature_k=args.temperature_k,
    )

    gt = surrogate[
        [
            "row_index",
            "formula",
            "mat_id",
            "ipa_direct_gap",
            "ipa_indirect_gap",
            "alignn_pred_gap",
        ]
    ].copy()
    gt["ground_truth_equal_alignn_gap_slme_percent"] = gt_equal
    gt["valid_equal_alignn_gap"] = np.isfinite(gt_equal)
    gt_path = args.output_dir / "ground_truth_equal_alignn_gap_slme.csv"
    gt.to_csv(gt_path, index=False)

    merged = surrogate.copy()
    merged["ground_truth_equal_alignn_gap_slme_percent"] = gt_equal
    merged = merged.rename(
        columns={
            "tensor_surrogate_slme_percent": "tensor_equal_alignn_gap_slme_percent",
            "trace_surrogate_slme_percent": "trace_equal_alignn_gap_slme_percent",
        }
    )
    for col, out_col in [
        ("tensor_equal_alignn_gap_slme_percent", "tensor_abs_error_percent_points"),
        ("trace_equal_alignn_gap_slme_percent", "trace_abs_error_percent_points"),
    ]:
        merged[out_col] = np.abs(
            merged[col] - merged["ground_truth_equal_alignn_gap_slme_percent"]
        )

    summary: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "ground_truth_source": "imag_dielectric_interp -> KK -> absorption -> pymatgen.slme",
        "shared_bandgap_source": "ALIGNN band_gap predictor used as direct and indirect gap for ground truth and surrogate spectra",
        "surrogate_source": str(args.surrogate_merged),
        "thickness_um": float(args.thickness_um),
        "temperature_k": float(args.temperature_k),
        "num_rows": int(len(merged)),
        "num_valid_ground_truth_equal_alignn_gap": int(np.isfinite(gt_equal).sum()),
        "tensor": _metrics(
            merged["ground_truth_equal_alignn_gap_slme_percent"].to_numpy(dtype=float),
            merged["tensor_equal_alignn_gap_slme_percent"].to_numpy(dtype=float),
        ),
        "trace": _metrics(
            merged["ground_truth_equal_alignn_gap_slme_percent"].to_numpy(dtype=float),
            merged["trace_equal_alignn_gap_slme_percent"].to_numpy(dtype=float),
        ),
    }

    merged_path = args.output_dir / "slme_equal_alignn_gap_benchmark_merged.csv"
    merged.to_csv(merged_path, index=False)
    worst_cols = [
        "row_index",
        "formula",
        "mat_id",
        "ipa_direct_gap",
        "ipa_indirect_gap",
        "alignn_pred_gap",
        "ground_truth_equal_alignn_gap_slme_percent",
    ]
    for label, pred_col, err_col in [
        (
            "tensor",
            "tensor_equal_alignn_gap_slme_percent",
            "tensor_abs_error_percent_points",
        ),
        (
            "trace",
            "trace_equal_alignn_gap_slme_percent",
            "trace_abs_error_percent_points",
        ),
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
        pred_col="tensor_equal_alignn_gap_slme_percent",
        title="Tensor spectrum",
        out_path=args.output_dir / "gt_vs_tensor_equal_alignn_gap.png",
    )
    _plot_scatter(
        merged,
        pred_col="trace_equal_alignn_gap_slme_percent",
        title="Trace spectrum",
        out_path=args.output_dir / "gt_vs_trace_equal_alignn_gap.png",
    )
    _plot_summary(merged, summary, args.output_dir / "slme_equal_alignn_gap_errors.png")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {merged_path}", flush=True)


if __name__ == "__main__":
    main()
