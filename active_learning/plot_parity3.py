#!/usr/bin/env python3
"""
3-panel MP-test parity: base | naive fine-tune | rehearsal (MP UNION AL).
Shows that rehearsal keeps in-distribution accuracy while the naive fine-tune
forgets. Reads artifacts/mp_test_parity3.npz (y, base, soup, rehearsal).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "artifacts", "mp_test_parity3.npz")
OUT = os.path.join(HERE, "artifacts", "mp_test_parity_3panel.png")
INK = "#0b0b0b"


def _mr(p, y):
    e = p - y
    ss = float(np.sum((y - y.mean())**2))
    return float(np.mean(np.abs(e))), (1 - float(np.sum(e**2)) / ss if ss > 0 else float("nan"))


def main():
    d = np.load(NPZ)
    y = d["y"]
    panels = [("Original (base)", d["base"], "#2a78d6"),
              ("Naive fine-tune", d["soup"], "#e34948"),
              ("Rehearsal (MP ∪ AL)", d["rehearsal"], "#008300")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (title, pred, color) in zip(axes, panels):
        ax.scatter(y, pred, s=9, c=color, alpha=0.32, edgecolors="none", zorder=3)
        ax.plot([0, 12], [0, 12], "--", color="k", lw=1.2, zorder=2)
        mae, r2 = _mr(pred, y)
        ax.text(0.05, 0.95, f"R$^2$ = {r2:.3f}\nMAE = {mae:.3f} eV",
                transform=ax.transAxes, va="top", ha="left", fontsize=12, color=INK)
        ax.set_xlim(0, 12); ax.set_ylim(0, 12)
        ax.set_xticks([0, 3, 6, 9, 12]); ax.set_yticks([0, 3, 6, 9, 12])
        ax.set_aspect("equal"); ax.set_title(title, fontsize=13, color=INK)
        ax.set_xlabel("Target band gap (eV)")
    axes[0].set_ylabel("Predicted band gap (eV)")
    fig.suptitle(f"MP test set (n={len(y)}): band-gap surrogate before vs after AL "
                 f"fine-tuning", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=150)
    for t, p, _ in panels:
        mae, r2 = _mr(p, y)
        print(f"{t:24s} MAE {mae:.3f}  R2 {r2:.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
