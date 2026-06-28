#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from pyxtal.symmetry import Group

from models.symmcd.pl_modules.diff_utils import BetaScheduler


DEFAULT_MARGINALS_PATH = Path("data/mp_20/train_site_symm_marginals_per_sg.pt")
DEFAULT_OUTPUT_DIR = Path("runtime/site_symm_plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot site-symmetry transition matrices Q_t or Q_t_bar over diffusion time."
    )
    parser.add_argument("--sg", type=int, default=225, help="Space group number.")
    parser.add_argument(
        "--axes",
        type=int,
        nargs="+",
        default=[1],
        help="1-based site-symmetry axis indices to plot.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 250, 500, 1000],
        help="Diffusion steps to visualize.",
    )
    parser.add_argument(
        "--style",
        choices=["summary", "matrix"],
        default="summary",
        help="Use the more interpretable summary view or the raw matrix heatmap view.",
    )
    parser.add_argument(
        "--mode",
        choices=["q_t", "q_t_bar"],
        default="q_t_bar",
        help="Plot single-step Q_t or cumulative Q_t_bar.",
    )
    parser.add_argument(
        "--source-class",
        type=int,
        default=None,
        help="1-based source class i for row-distribution plots. Defaults to argmax(m).",
    )
    parser.add_argument(
        "--marginals-path",
        type=Path,
        default=DEFAULT_MARGINALS_PATH,
        help="Path to train_site_symm_marginals_per_sg.pt.",
    )
    parser.add_argument(
        "--hparams",
        type=Path,
        default=None,
        help="Optional hparams.yaml to read timesteps and beta scheduler config from.",
    )
    parser.add_argument(
        "--scheduler-mode",
        choices=["cosine", "linear", "quadratic", "sigmoid"],
        default="cosine",
        help="Used when --hparams is not provided.",
    )
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--nu", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path. Defaults under runtime/site_symm_plots/.",
    )
    return parser.parse_args()


def load_scheduler(args: argparse.Namespace) -> BetaScheduler:
    if args.hparams is None:
        timesteps = max(args.timesteps)
        return BetaScheduler(
            timesteps=timesteps,
            scheduler_mode=args.scheduler_mode,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            nu=args.nu,
        )

    cfg = OmegaConf.load(args.hparams)
    beta_cfg: Any = cfg.model.beta_scheduler
    timesteps = int(cfg.model.timesteps)
    return BetaScheduler(
        timesteps=timesteps,
        scheduler_mode=str(beta_cfg.scheduler_mode),
        beta_start=float(beta_cfg.get("beta_start", args.beta_start)),
        beta_end=float(beta_cfg.get("beta_end", args.beta_end)),
        nu=int(beta_cfg.get("nu", args.nu)),
    )


def validate_args(args: argparse.Namespace, scheduler: BetaScheduler) -> None:
    if args.sg < 1 or args.sg > 230:
        raise ValueError(f"Space group must be in [1, 230], got {args.sg}")
    for axis in args.axes:
        if axis < 1 or axis > 15:
            raise ValueError(f"Axis must be in [1, 15], got {axis}")
    if args.style == "summary" and len(args.axes) != 1:
        raise ValueError("Summary style currently supports exactly one axis.")
    for timestep in args.timesteps:
        if timestep < 0 or timestep > scheduler.timesteps:
            raise ValueError(
                f"Timestep must be in [0, {scheduler.timesteps}], got {timestep}"
            )
    if args.source_class is not None and not 1 <= args.source_class <= 13:
        raise ValueError(f"Source class must be in [1, 13], got {args.source_class}")


def load_marginals(path: Path) -> list[torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Missing marginal file: {path}")
    marginals = torch.load(path, map_location="cpu")
    if not isinstance(marginals, list) or len(marginals) != 15:
        raise ValueError(
            f"Expected a list of 15 axis tensors in {path}, got {type(marginals)}"
        )
    return marginals


def transition_matrix(
    marginal: torch.Tensor, scheduler: BetaScheduler, t: int, mode: str
) -> torch.Tensor:
    if mode == "q_t":
        alpha = scheduler.alphas[t]
    else:
        alpha = scheduler.alphas_cumprod[t]
    beta = 1.0 - alpha
    num_classes = marginal.shape[0]
    eye = torch.eye(num_classes, dtype=marginal.dtype)
    prior = marginal.unsqueeze(0).expand(num_classes, -1)
    return alpha * eye + beta * prior


def select_reference_classes(marginal: torch.Tensor) -> list[int]:
    sorted_indices = torch.argsort(marginal, descending=True).tolist()
    chosen = [index for index in sorted_indices if float(marginal[index]) > 0.0][:3]
    while len(chosen) < 3:
        for index in sorted_indices:
            if index not in chosen:
                chosen.append(index)
                break
    return [index + 1 for index in chosen[:3]]


def make_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        return args.output
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    axis_tag = "-".join(str(axis) for axis in args.axes)
    time_tag = "-".join(str(timestep) for timestep in args.timesteps)
    return (
        output_dir
        / f"sg{args.sg}_{args.style}_{args.mode}_axes{axis_tag}_t{time_tag}.png"
    )


def plot_summary(
    marginal: torch.Tensor,
    scheduler: BetaScheduler,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    axis = args.axes[0]
    source_class = (
        args.source_class
        if args.source_class is not None
        else int(torch.argmax(marginal).item()) + 1
    )
    reference_classes = select_reference_classes(marginal)
    time_grid = torch.arange(scheduler.timesteps + 1, dtype=torch.long)
    alpha_curve = scheduler.alphas[time_grid].cpu().numpy()
    alpha_bar_curve = scheduler.alphas_cumprod[time_grid].cpu().numpy()

    try:
        sg_symbol = Group(args.sg).symbol
    except Exception:
        sg_symbol = "?"

    fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])

    class_positions = np.arange(1, marginal.numel() + 1)

    ax_marginal = fig.add_subplot(grid[0, 0])
    marginal_np = marginal.cpu().numpy()
    ax_marginal.bar(class_positions, marginal_np, color="#4C72B0")
    ax_marginal.set_title(rf"Marginal $m_{{\Sigma,u,G}}$ for axis $u={axis}$")
    ax_marginal.set_xlabel("class k")
    ax_marginal.set_ylabel("probability")
    ax_marginal.set_xticks(class_positions)
    ax_marginal.set_ylim(0.0, max(0.65, float(marginal.max()) * 1.12))

    ax_scheduler = fig.add_subplot(grid[0, 1])
    ax_scheduler.plot(time_grid.numpy(), alpha_curve, label=r"$\alpha_t$", lw=2)
    ax_scheduler.plot(
        time_grid.numpy(), alpha_bar_curve, label=r"$\bar{\alpha}_t$", lw=2
    )
    for timestep in args.timesteps:
        ax_scheduler.axvline(timestep, color="0.85", lw=1, ls="--")
    ax_scheduler.scatter(
        args.timesteps,
        [float(alpha_bar_curve[timestep]) for timestep in args.timesteps],
        color="#DD8452",
        label=r"selected $\bar{\alpha}_t$",
        zorder=3,
    )
    ax_scheduler.set_title("Scheduler weights over diffusion time")
    ax_scheduler.set_xlabel("timestep t")
    ax_scheduler.set_ylabel("weight")
    ax_scheduler.set_xlim(0, scheduler.timesteps)
    ax_scheduler.set_ylim(0.0, 1.02)
    ax_scheduler.legend(frameon=False)

    ax_self = fig.add_subplot(grid[1, 0])
    for ref_class in reference_classes:
        ref_prob = float(marginal[ref_class - 1])
        curve = alpha_bar_curve + (1.0 - alpha_bar_curve) * ref_prob
        ax_self.plot(time_grid.numpy(), curve, lw=2, label=f"class {ref_class}")
    for timestep in args.timesteps:
        ax_self.axvline(timestep, color="0.88", lw=1, ls="--")
    ax_self.set_title(
        r"Self-transition $\overline{Q}_{t}[i,i]$ for representative classes"
    )
    ax_self.set_xlabel("timestep t")
    ax_self.set_ylabel("probability of staying in class i")
    ax_self.set_xlim(0, scheduler.timesteps)
    ax_self.set_ylim(0.0, 1.02)
    ax_self.legend(frameon=False)

    ax_rows = fig.add_subplot(grid[1, 1])
    width = 0.18 if len(args.timesteps) >= 4 else 0.8 / max(len(args.timesteps), 1)
    base = np.arange(1, marginal.numel() + 1, dtype=float)
    offsets = np.linspace(
        -width * (len(args.timesteps) - 1) / 2.0,
        width * (len(args.timesteps) - 1) / 2.0,
        len(args.timesteps),
    )
    for offset, timestep in zip(offsets, args.timesteps):
        row = transition_matrix(marginal, scheduler, timestep, args.mode)[
            source_class - 1
        ]
        alpha_used = (
            float(scheduler.alphas[timestep])
            if args.mode == "q_t"
            else float(scheduler.alphas_cumprod[timestep])
        )
        ax_rows.bar(
            base + offset,
            row.cpu().numpy(),
            width=width,
            label=f"t={timestep} (α={alpha_used:.3f})",
        )
    mode_label = r"\overline{Q}_t" if args.mode == "q_t_bar" else r"Q_t"
    ax_rows.set_title(
        rf"Row distribution $e_i^\top {mode_label}$ for source class $i={source_class}$"
    )
    ax_rows.set_xlabel("destination class j")
    ax_rows.set_ylabel("transition probability")
    ax_rows.set_xticks(class_positions)
    ax_rows.set_ylim(0.0, 1.02)
    ax_rows.legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"Site-symmetry diffusion summary for SG {args.sg} ({sg_symbol}), axis u={axis}",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_transition_matrices(
    marginals: list[torch.Tensor],
    scheduler: BetaScheduler,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    n_rows = len(args.axes)
    n_cols = len(args.timesteps)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows + 0.8),
        constrained_layout=True,
        squeeze=False,
    )

    try:
        sg_symbol = Group(args.sg).symbol
    except Exception:
        sg_symbol = "?"

    last_image = None
    for row, axis in enumerate(args.axes):
        marginal = marginals[axis - 1][args.sg].to(torch.float32)
        for col, timestep in enumerate(args.timesteps):
            matrix = transition_matrix(marginal, scheduler, timestep, args.mode).numpy()
            ax = axes[row][col]
            last_image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
            alpha = (
                float(scheduler.alphas[timestep])
                if args.mode == "q_t"
                else float(scheduler.alphas_cumprod[timestep])
            )
            beta = 1.0 - alpha
            ax.set_title(f"t={timestep}\nα={alpha:.3f}, β={beta:.3f}")
            ax.set_xticks(np.arange(matrix.shape[1]))
            ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1), fontsize=8)
            ax.set_yticks(np.arange(matrix.shape[0]))
            ax.set_yticklabels(np.arange(1, matrix.shape[0] + 1), fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("to class j")
            if col == 0:
                ax.set_ylabel(f"axis u={axis}\nfrom class i")

    assert last_image is not None
    cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("transition probability")
    mode_label = (
        "Single-step transition" if args.mode == "q_t" else "Cumulative transition"
    )
    fig.suptitle(
        f"{mode_label} matrices for SG {args.sg} ({sg_symbol})",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scheduler = load_scheduler(args)
    validate_args(args, scheduler)
    marginals = load_marginals(args.marginals_path)
    output_path = make_output_path(args)
    if args.style == "matrix":
        plot_transition_matrices(marginals, scheduler, args, output_path)
    else:
        plot_summary(
            marginals[args.axes[0] - 1][args.sg].to(torch.float32),
            scheduler,
            args,
            output_path,
        )
    print(output_path)


if __name__ == "__main__":
    main()
