from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors


def parse_args():
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(__file__).resolve().parent
    parser = ArgumentParser()
    parser.add_argument(
        "--sg-info",
        type=Path,
        default=root / "data" / "mp_20" / "sg_info.pt",
    )
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=out_dir / "sg_marginal_temperature_reweighting.png",
    )
    return parser.parse_args()


def load_distribution(sg_info_path: Path):
    _, sg_dist, _ = torch.load(sg_info_path, map_location="cpu")
    sg_dist = np.asarray(sg_dist, dtype=float)
    mask = sg_dist > 0
    spacegroups = np.arange(1, 231, dtype=int)[mask]
    probabilities = sg_dist[mask]
    return spacegroups, probabilities


def temperature_reweight(probabilities: np.ndarray, temperature: float):
    reweighted = probabilities.copy()
    reweighted = reweighted ** (1.0 / temperature)
    return reweighted / reweighted.sum()


def axis_ticks(spacegroups: np.ndarray):
    ticks = np.linspace(
        spacegroups.min(), spacegroups.max(), num=min(12, len(spacegroups))
    )
    return sorted({int(round(tick)) for tick in ticks})


def plot_panel(ax, spacegroups: np.ndarray, values: np.ndarray, title: str):
    norm = colors.Normalize(vmin=1, vmax=230)
    cmap = plt.colormaps["viridis"]
    bar_colors = [cmap(norm(sg)) for sg in spacegroups]
    ax.bar(spacegroups, values, color=bar_colors, width=0.9)
    ax.set_title(title)
    ax.set_xlabel("Space group")
    ax.set_xlim(
        max(0.5, float(spacegroups.min()) - 1), min(230.5, float(spacegroups.max()) + 1)
    )
    ax.set_xticks(axis_ticks(spacegroups))
    return norm, cmap


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    spacegroups, probabilities = load_distribution(args.sg_info.resolve())
    reweighted = temperature_reweight(probabilities, args.temperature)
    ymax = max(float(probabilities.max()), float(reweighted.max())) * 1.05

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6), constrained_layout=True, sharey=True
    )
    norm, cmap = plot_panel(
        axes[0],
        spacegroups,
        probabilities,
        r"Dataset marginal $p_{marg}(G)$",
    )
    plot_panel(
        axes[1],
        spacegroups,
        reweighted,
        rf"Temperature reweighted $p_T(G)$, $T_{{sg}}={args.temperature:g}$",
    )
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0.0, ymax)
    fig.suptitle("Sampling space groups from the dataset marginal", fontsize=18)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist())
    cbar.set_label("Space group color key")
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
