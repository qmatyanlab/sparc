#!/usr/bin/env python3
"""
Generate site symmetry visualization plots for GaAs (Spacegroup 216: F-43m)

This script creates three visualizations:
1. gaas_wyckoff_site_symm_matrices.png - All 9 Wyckoff positions with their 15×13 site symmetry encodings
2. gaas_site_symm_marginals.png - Training data probability distribution per axis
3. gaas_ideal_vs_marginals.png - Side-by-side comparison of ideal vs marginals

Colormap: bwr (blue-white-red)
  White = 0 (not occupied)
  Red = 1 (occupied)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pyxtal.symmetry import Group


def main():
    sg = 216
    group = Group(sg)

    print("=" * 80)
    print("Generating GaAs (SG 216) Site Symmetry Visualizations")
    print("=" * 80)
    print(f"\nSpacegroup: {sg} (F-43m)")
    print(f"Total Wyckoff positions: {len(group.Wyckoff_positions)}")
    print(f"Matrix size per position: 15 axes × 13 point group symbols")

    # =========================================================================
    # Figure 1: All 9 Wyckoff Positions (3×3 grid)
    # =========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        f"GaAs (Spacegroup 216: F-43m) - Wyckoff Position Site Symmetries\n15 axes × 13 point group symbols",
        fontsize=14,
        fontweight="bold",
    )

    axes = axes.flatten()

    for i, wp in enumerate(group.Wyckoff_positions):
        ss_obj = wp.get_site_symmetry_object()
        one_hot = ss_obj.to_one_hot()
        if torch.is_tensor(one_hot):
            one_hot = one_hot.numpy()

        ax = axes[i]
        im = ax.imshow(one_hot, cmap="bwr", aspect="auto", vmin=0, vmax=1)

        # Title with Wyckoff letter, multiplicity, and site symmetry
        title = f"Wyckoff {wp.letter} (mult={len(wp)})\nSite symm: {ss_obj.symbols}"
        ax.set_title(title, fontsize=10, fontweight="bold")

        ax.set_xticks(range(13))
        ax.set_yticks(range(15))
        ax.set_xticklabels(range(13), fontsize=7)
        ax.set_yticklabels(range(15), fontsize=7)

    plt.tight_layout()
    plt.savefig("gaas_wyckoff_site_symm_matrices.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: gaas_wyckoff_site_symm_matrices.png")
    plt.close()

    # =========================================================================
    # Figure 2: Site Symmetry Marginals (Training Data Distribution)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 12))

    ss_marginals = torch.load("../data/mp_20/train_site_symm_marginals_per_sg.pt")
    marginals_sg216 = torch.stack([ss_marginals[i][sg - 1] for i in range(15)])
    marginals_sg216_np = marginals_sg216.numpy()

    im = ax.imshow(marginals_sg216_np, cmap="bwr", aspect="auto", vmin=0, vmax=1)
    ax.set_title(
        f"GaAs (SG 216) - Site Symmetry Marginals\n(Probability distribution per axis)",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xticks(range(13))
    ax.set_yticks(range(15))
    ax.set_xticklabels(range(13))
    ax.set_yticklabels(range(15))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Probability")

    plt.tight_layout()
    plt.savefig("gaas_site_symm_marginals.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: gaas_site_symm_marginals.png")
    plt.close()

    # =========================================================================
    # Figure 3: Ideal vs Marginals Comparison (2×5 grid)
    # =========================================================================
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)

    fig.suptitle(
        f"GaAs (SG 216): Ideal Wyckoff Site Symmetries vs Training Marginals",
        fontsize=14,
        fontweight="bold",
    )

    # Top row: Wyckoff ideal encodings (first 5)
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        wp = group.Wyckoff_positions[i]
        ss_obj = wp.get_site_symmetry_object()
        one_hot = ss_obj.to_one_hot()
        if torch.is_tensor(one_hot):
            one_hot = one_hot.numpy()

        im = ax.imshow(one_hot, cmap="bwr", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"WP {wp.letter}", fontsize=10, fontweight="bold")
        ax.set_xticks(range(13))
        ax.set_yticks(range(15))
        ax.set_xticklabels(range(13), fontsize=7)
        ax.set_yticklabels(range(15), fontsize=7)

    # Bottom row: Marginals comparison
    for i in range(5):
        ax = fig.add_subplot(gs[1, i])

        im = ax.imshow(marginals_sg216_np, cmap="bwr", aspect="auto", vmin=0, vmax=1)
        ax.set_title("Marginals", fontsize=10, fontweight="bold")
        ax.set_xticks(range(13))
        ax.set_yticks(range(15))
        ax.set_xticklabels(range(13), fontsize=7)
        ax.set_yticklabels(range(15), fontsize=7)

    plt.savefig("gaas_ideal_vs_marginals.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: gaas_ideal_vs_marginals.png")
    plt.close()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nColormap: bwr (blue-white-red)")
    print(f"  White = 0 (not occupied)")
    print(f"  Red = 1 (occupied)")
    print(f"\nAspect ratio: auto")
    print(f"\nData shape: 15 axes × 13 point group symbols")
    print(f"Marginals shape: {marginals_sg216_np.shape}")
    print("\nFiles generated:")
    print(f"  1. gaas_wyckoff_site_symm_matrices.png")
    print(f"  2. gaas_site_symm_marginals.png")
    print(f"  3. gaas_ideal_vs_marginals.png")


if __name__ == "__main__":
    main()
