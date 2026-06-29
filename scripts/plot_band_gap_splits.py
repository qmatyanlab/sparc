#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data.dielectric.utils.utils_model_scalar import Network


SPLITS = ["train", "valid", "test"]


def _resolve_config_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _load_config_defaults(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", {}) or {}
    inference_cfg = cfg.get("inference", {}) or {}
    output_cfg = cfg.get("outputs", {}) or {}
    defaults: dict[str, Any] = {"config": config_path}

    if "cache_path" in inference_cfg:
        defaults["cache"] = _resolve_config_path(config_path, inference_cfg["cache_path"])
    if "checkpoint_path" in inference_cfg:
        defaults["model"] = _resolve_config_path(
            config_path, inference_cfg["checkpoint_path"]
        )
    if "artifact_dir" in output_cfg:
        defaults["output_dir"] = _resolve_config_path(config_path, output_cfg["artifact_dir"])

    key_map = {
        "batch_size": ("batch_size", inference_cfg),
        "device": ("device", inference_cfg),
        "num_neighbors": ("num_neighbors", model_cfg),
        "r_max": ("r_max", model_cfg),
        "em_dim": ("em_dim", model_cfg),
        "layers": ("layers", model_cfg),
        "mul": ("mul", model_cfg),
        "lmax": ("lmax", model_cfg),
        "png_dpi": ("png_dpi", output_cfg),
        "eps_dpi": ("eps_dpi", output_cfg),
    }
    for arg_name, (cfg_name, section) in key_map.items():
        if cfg_name in section:
            defaults[arg_name] = section[cfg_name]

    return defaults


class NetWrapper(Network):
    def __init__(self, in_dim: int, em_dim: int, **kwargs: Any) -> None:
        self.pool = False
        if kwargs.get("reduce_output", False):
            kwargs["reduce_output"] = False
            self.pool = True
        super().__init__(**kwargs)
        self.em_z = nn.Linear(in_dim, em_dim)
        self.em_x = nn.Linear(in_dim, em_dim)

    def forward(self, data: Data) -> torch.Tensor:
        data.z = F.relu(self.em_z(data.z))
        data.x = F.relu(self.em_x(data.x))
        output = super().forward(data)
        if self.pool:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)
        return output


class CompatGraphDataset:
    def __init__(self, graphs: np.ndarray, targets: np.ndarray, indices: list[int]) -> None:
        self.graphs = graphs
        self.targets = targets
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Data:
        row_index = self.indices[item]
        old = self.graphs[row_index].__dict__
        return Data(
            pos=old["pos"].to(dtype=torch.float64),
            x=old["x"].to(dtype=torch.float64),
            z=old["z"].to(dtype=torch.float64),
            edge_index=old["edge_index"].to(dtype=torch.long),
            edge_vec=old["edge_vec"].to(dtype=torch.float64),
            y=torch.tensor([[self.targets[row_index]]], dtype=torch.float64),
            row_index=torch.tensor([row_index], dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument(
        "--config",
        type=Path,
        default=ROOT / "data/surrogates/TSENN_bandgap.yaml",
    )
    config_args, _ = base.parse_known_args()
    defaults = _load_config_defaults(config_args.config)

    parser = argparse.ArgumentParser(parents=[base])
    parser.add_argument(
        "--cache",
        type=Path,
        default=defaults.get(
            "cache", ROOT / "data/dielectric/cached_band_gap_preprocessed_data.pt"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=defaults.get(
            "model", ROOT / "data/surrogates/TSENN_bandgap.torch"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.get("output_dir", ROOT / "runtime/band_gap_parity_splits"),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Reuse an existing predictions CSV and only regenerate plot/summary.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 128))
    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument("--num-neighbors", type=float, default=defaults.get("num_neighbors"))
    parser.add_argument("--r-max", type=float, default=defaults.get("r_max", 6.0))
    parser.add_argument("--em-dim", type=int, default=defaults.get("em_dim", 128))
    parser.add_argument("--layers", type=int, default=defaults.get("layers", 4))
    parser.add_argument("--mul", type=int, default=defaults.get("mul", 16))
    parser.add_argument("--lmax", type=int, default=defaults.get("lmax", 2))
    parser.add_argument("--png-dpi", type=int, default=defaults.get("png_dpi", 600))
    parser.add_argument("--eps-dpi", type=int, default=defaults.get("eps_dpi", 300))
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
            "figure.figsize": (6.4, 2.45),
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )


def _compute_num_neighbors(graphs: np.ndarray, train_indices: list[int]) -> float:
    counts: list[int] = []
    for row_index in train_indices:
        old = graphs[int(row_index)].__dict__
        edge_src = old["edge_index"][0]
        num_nodes = int(old["pos"].shape[0])
        for atom_index in range(num_nodes):
            counts.append(int((edge_src == atom_index).sum()))
    return float(np.mean(counts))


def _load_model(args: argparse.Namespace, payload: dict[str, Any], device: str) -> NetWrapper:
    graphs = payload["df"]["data"].to_numpy()
    num_neighbors = (
        float(args.num_neighbors)
        if args.num_neighbors is not None
        else _compute_num_neighbors(graphs, payload["idx_train"])
    )
    model = NetWrapper(
        in_dim=118,
        em_dim=int(args.em_dim),
        irreps_in=f"{int(args.em_dim)}x0e",
        irreps_out=f"{int(payload.get('out_dim', 1))}x0e",
        irreps_node_attr=f"{int(args.em_dim)}x0e",
        layers=int(args.layers),
        mul=int(args.mul),
        lmax=int(args.lmax),
        max_radius=float(payload.get("r_max", args.r_max)),
        num_neighbors=num_neighbors,
        reduce_output=True,
        dropout_prob=0.0,
        use_batch_norm=False,
    ).to(device, dtype=torch.float64)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    model.num_neighbors_used = num_neighbors  # type: ignore[attr-defined]
    return model


def _predict_split(
    model: NetWrapper,
    dataset: CompatGraphDataset,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[pd.DataFrame] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).detach().cpu().view(-1).numpy()
            target = batch.y.detach().cpu().view(-1).numpy()
            row_index = batch.row_index.detach().cpu().view(-1).numpy()
            rows.append(
                pd.DataFrame(
                    {
                        "row_index": row_index.astype(int),
                        "target_band_gap_ev": target.astype(float),
                        "predicted_band_gap_ev": pred.astype(float),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(x) & np.isfinite(y)
    if not bool(mask.any()):
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "r2": float("nan"),
            "fit_r2": float("nan"),
        }
    x = x[mask]
    y = y[mask]
    diff = y - x
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        fit_y = slope * x + intercept
        fit_ss_res = float(np.sum((y - fit_y) ** 2))
        fit_ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        fit_r2 = (
            float(1.0 - fit_ss_res / fit_ss_tot) if fit_ss_tot > 0.0 else float("nan")
        )
    else:
        fit_r2 = float("nan")
    return {
        "count": int(len(x)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "bias": float(np.mean(diff)),
        "r2": r2,
        "fit_r2": fit_r2,
    }


def _ticks_for(values: list[np.ndarray]) -> tuple[list[float], float]:
    upper = float(np.nanmax([np.nanmax(v) for v in values if len(v)] + [1.0]))
    upper = max(1.0, upper * 1.03)
    step = max(1, int(np.ceil(upper / 4.0)))
    lim_upper = float(step * int(np.ceil(upper / step)))
    ticks = np.arange(0.0, lim_upper + 0.5 * step, float(step))
    return [float(t) for t in ticks], lim_upper


def _plot_horizontal(
    split_frames: dict[str, pd.DataFrame],
    summary: dict[str, dict[str, float | int]],
    out_path: Path,
    png_dpi: int,
    eps_dpi: int,
) -> None:
    _apply_style()
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    split_colors = {
        "train": cycle[1] if len(cycle) > 1 else "#000dfc",
        "valid": cycle[2] if len(cycle) > 2 else "#ff7f0e",
        "test": cycle[3] if len(cycle) > 3 else "#1fb312",
    }
    values = []
    for split in SPLITS:
        values.append(split_frames[split]["target_band_gap_ev"].to_numpy(dtype=float))
        values.append(split_frames[split]["predicted_band_gap_ev"].to_numpy(dtype=float))
    ticks, lim_upper = _ticks_for(values)

    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.45), constrained_layout=False)
    fig.subplots_adjust(left=0.068, right=0.995, bottom=0.18, top=0.78, wspace=0.05)

    for ax, split in zip(axes, SPLITS):
        df = split_frames[split]
        x = df["target_band_gap_ev"].to_numpy(dtype=float)
        y = df["predicted_band_gap_ev"].to_numpy(dtype=float)
        ax.scatter(
            x,
            y,
            color=split_colors[split],
            s=4,
            alpha=0.25,
            linewidths=0,
            rasterized=True,
        )
        ax.plot([0.0, lim_upper], [0.0, lim_upper], color="black", linestyle="--", linewidth=1.0)
        ax.set_xlim(0.0, lim_upper)
        ax.set_ylim(0.0, lim_upper)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(split)
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    label=(
                        f"R² = {summary[split]['r2']:.3f}\n"
                        f"MAE = {summary[split]['mae']:.3f} eV"
                    ),
                )
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
    fig.text(0.5, 0.06, "Target band gap (eV)", ha="center", va="center", fontsize=8)
    fig.text(
        0.047,
        0.50,
        "Predicted band gap (eV)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8,
    )
    fig.savefig(out_path, dpi=png_dpi)
    fig.savefig(out_path.with_suffix(".eps"), dpi=eps_dpi)
    plt.close(fig)


def _split_frames_from_predictions(
    predictions: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float | int]]]:
    split_frames: dict[str, pd.DataFrame] = {}
    summary: dict[str, dict[str, float | int]] = {}
    for split in SPLITS:
        split_df = predictions.loc[predictions["split"] == split].copy()
        split_frames[split] = split_df
        summary[split] = _metrics(
            split_df["target_band_gap_ev"].to_numpy(dtype=float),
            split_df["predicted_band_gap_ev"].to_numpy(dtype=float),
        )
    return split_frames, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    predictions_path = args.output_dir / "band_gap_train_valid_test_predictions.csv"
    payload = None
    model = None
    if args.predictions_csv is not None:
        predictions_path = args.predictions_csv
        predictions = pd.read_csv(predictions_path)
        split_frames, summary = _split_frames_from_predictions(predictions)
    else:
        payload = torch.load(args.cache, map_location="cpu")
        df = payload["df"]
        targets = df[str(payload.get("target_col", "band_gap"))].to_numpy(dtype=float)
        graphs = df["data"].to_numpy()
        model = _load_model(args, payload, device)

        split_frames = {}
        summary = {}
        for split in SPLITS:
            dataset = CompatGraphDataset(graphs, targets, payload[f"idx_{split}"])
            split_df = _predict_split(model, dataset, args.batch_size, device)
            split_df.insert(0, "split", split)
            split_frames[split] = split_df
            summary[split] = _metrics(
                split_df["target_band_gap_ev"].to_numpy(dtype=float),
                split_df["predicted_band_gap_ev"].to_numpy(dtype=float),
            )

        predictions = pd.concat(
            [split_frames[split] for split in SPLITS], ignore_index=True
        )
        predictions.to_csv(predictions_path, index=False)

    plot_path = args.output_dir / "band_gap_train_valid_test_horizontal.png"
    _plot_horizontal(
        split_frames=split_frames,
        summary=summary,
        out_path=plot_path,
        png_dpi=args.png_dpi,
        eps_dpi=args.eps_dpi,
    )

    summary_path = args.output_dir / "band_gap_train_valid_test_horizontal.json"
    payload_summary = {
        "config": str(args.config),
        "cache": str(args.cache),
        "model": str(args.model),
        "target_col": "band_gap"
        if payload is None
        else str(payload.get("target_col", "band_gap")),
        "device": device,
        "model_hparams": {
            "em_dim": int(args.em_dim),
            "layers": int(args.layers),
            "mul": int(args.mul),
            "lmax": int(args.lmax),
            "r_max": float(args.r_max)
            if payload is None
            else float(payload.get("r_max", args.r_max)),
            "num_neighbors": float(args.num_neighbors)
            if model is None
            else float(model.num_neighbors_used),  # type: ignore[attr-defined]
        },
        "reused_predictions_csv": None
        if args.predictions_csv is None
        else str(args.predictions_csv),
        "outputs": {
            "png": str(plot_path),
            "eps": str(plot_path.with_suffix(".eps")),
            "predictions_csv": str(predictions_path),
        },
        "export_dpi": {
            "png": int(args.png_dpi),
            "eps": int(args.eps_dpi),
        },
        "splits": summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload_summary, f, indent=2)

    print(json.dumps(payload_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
