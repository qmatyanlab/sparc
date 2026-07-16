#!/usr/bin/env python3
"""
Before/after parity on the ORIGINAL MP test set (4260): base surrogate vs the
active-learning fine-tuned soup, same points. Directly shows whether fine-tuning
degraded in-distribution accuracy (catastrophic forgetting). Matches the style of
the project's train/valid/test parity figure.

Reads artifacts/mp_test_parity.npz (y, base, soup) from measure_forgetting.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "artifacts", "mp_test_parity.npz")
OUT = os.path.join(HERE, "artifacts", "mp_test_parity_before_after.png")

C_BASE = "#2a78d6"   # blue  (original)
C_FT = "#e34948"     # red   (fine-tuned)  -- validated diverging pair
INK = "#0b0b0b"


def _mae_r2(pred, y):
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    ss = float(np.sum((y - y.mean())**2))
    r2 = float(1 - np.sum(err**2) / ss) if ss > 0 else float("nan")
    return mae, r2


def main():
    d = np.load(NPZ)
    y, base, soup = d["y"], d["base"], d["soup"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax, pred, color, title in [
        (axes[0], base, C_BASE, "Original (base)"),
        (axes[1], soup, C_FT, "Fine-tuned (AL soup)"),
    ]:
        ax.scatter(y, pred, s=10, c=color, alpha=0.35, edgecolors="none", zorder=3)
        ax.plot([0, 12], [0, 12], "--", color="k", lw=1.2, zorder=2)
        mae, r2 = _mae_r2(pred, y)
        ax.text(0.05, 0.95, f"R$^2$ = {r2:.3f}\nMAE = {mae:.3f} eV",
                transform=ax.transAxes, va="top", ha="left", fontsize=12, color=INK)
        ax.set_xlim(0, 12); ax.set_ylim(0, 12)
        ax.set_xticks([0, 3, 6, 9, 12]); ax.set_yticks([0, 3, 6, 9, 12])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13, color=INK)
        ax.set_xlabel("Target band gap (eV)")
    axes[0].set_ylabel("Predicted band gap (eV)")
    fig.suptitle(f"MP test set (n={len(y)}): band-gap surrogate before vs after "
                 f"AL fine-tuning", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=150)
    print(f"base  MAE {_mae_r2(base,y)[0]:.3f}  R2 {_mae_r2(base,y)[1]:.3f}")
    print(f"soup  MAE {_mae_r2(soup,y)[0]:.3f}  R2 {_mae_r2(soup,y)[1]:.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
