#!/usr/bin/env python3
"""TSENN/OptiMate absorption-spectra top-N plots for the good-structure deliverables.

For a run's deliverables_good_structures(_corrected) folder, takes the top-N selected
"good" and "solar_window" structures, computes alpha(E) via the run's TSENN/OptiMate
dielectric model, and writes the absorption-spectra PNG + CSV at the run's thickness.

Reuses the absorption machinery from scripts/plot_tsenn_slme_results.py.

Example:
  python scripts/plot_good_structure_absorption.py exp_res/<run> \
      --deliverables <run>/deliverables_good_structures_corrected
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read as ase_read
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from pymatgen.io.ase import AseAtomsAdaptor

from rewards.calculators.tsenn_slme.calc import _absorption_coef_um_inv, _kk_eps1_from_eps2
from plot_tsenn_slme_results import (
    _latex_formula,
    _load_tsenn_thickness_um,
    _tsenn_slme_from_run_config,
)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deliverables", type=Path, default=None)
    p.add_argument("--top-n", type=int, default=4)
    return p.parse_args()


def load_run_config(run_dir: Path):
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if cfg_path.exists():
        return OmegaConf.load(cfg_path)
    return OmegaConf.load(run_dir / "hparams.yaml")


def plot_one(calc, deliv: Path, kind: str, thickness_um: float, top_n: int) -> dict:
    base = "good" if kind == "good" else "solar_window"
    csv_in = deliv / (
        "selected_good_structures.csv" if kind == "good" else "selected_solar_window_structures.csv"
    )
    xyz_in = deliv / (
        "selected_good_structures.extxyz" if kind == "good" else "selected_solar_window_structures.extxyz"
    )
    meta = pd.read_csv(csv_in).head(top_n)
    atoms = ase_read(xyz_in, index=":")
    if not isinstance(atoms, list):
        atoms = [atoms]
    structures = [AseAtomsAdaptor.get_structure(a) for a in atoms[:top_n]]

    energies_ev, eps2_iso, valid_mask = calc.tsenn.predict_epsilon2_iso(structures)
    eps2_iso = np.nan_to_num(eps2_iso, nan=0.0)
    eps1 = _kk_eps1_from_eps2(energies_ev, eps2_iso)
    alpha_um_inv = _absorption_coef_um_inv(energies_ev, eps1, eps2_iso)
    alpha_cm_inv = alpha_um_inv * 1.0e4
    absorptance = 1.0 - np.exp(-2.0 * np.clip(alpha_um_inv, 0.0, None) * thickness_um)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True)
    for i in range(len(structures)):
        if not bool(valid_mask[i]):
            continue
        r = meta.iloc[i]
        label = (
            rf"${_latex_formula(str(r['formula']))}$ "
            rf"($E_g={r['band_gap_ev']:.2f}\,\mathrm{{eV}},\ \eta={r['slme_eta_percent']:.1f}\%$, step {int(r['step'])})"
        )
        axes[0].plot(energies_ev, alpha_cm_inv[i], color=COLORS[i % len(COLORS)], lw=1.7, label=label)
        axes[1].plot(energies_ev, absorptance[i], color=COLORS[i % len(COLORS)], lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\alpha(E)\ (\mathrm{cm}^{-1})$")
    axes[0].set_title(rf"TSENN/OptiMate absorption — {kind} top-{top_n} ($L={thickness_um:g}\,\mu$m)")
    axes[0].grid(axis="y", alpha=0.12)
    axes[0].legend(loc="center right", frameon=True, fontsize=8)
    axes[1].set_xlabel(r"Photon energy $E$ (eV)")
    axes[1].set_ylabel(r"Absorptance $A(E)=1-e^{-2\alpha(E)L}$")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlim(0.0, 10.0)
    axes[1].grid(axis="y", alpha=0.12)
    axes[1].legend(
        handles=[Line2D([0], [0], color="black", lw=1.8, label=rf"$L={thickness_um:g}\,\mu$m")],
        title="Thickness", loc="center right", frameon=True,
    )
    fig.tight_layout()

    tag = f"{int(round(thickness_um*10)):02d}um"
    png = deliv / f"tsenn_absorption_spectra_{base}_top{top_n}_{tag}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    csv_out = deliv / f"tsenn_absorption_spectra_{base}_top{top_n}_{tag}.csv"
    meta.to_csv(csv_out, index=False)
    print(f"  wrote {png.name} and {csv_out.name}")
    return {
        "selection": f"{base}_top{top_n}",
        "png": str(png),
        "csv": str(csv_out),
        "structures": meta.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    deliv = (args.deliverables or run_dir / "deliverables_good_structures_corrected").resolve()
    cfg = load_run_config(run_dir)
    thickness_um = _load_tsenn_thickness_um(cfg)
    print(f"Building TSENN/OptiMate calculator (thickness {thickness_um} um, cpu)...")
    calc = _tsenn_slme_from_run_config(cfg)

    outputs = [plot_one(calc, deliv, k, thickness_um, args.top_n) for k in ("good", "solar_window")]
    summary = {"thickness_um": thickness_um, "top_n": args.top_n, "outputs": outputs}
    tag = f"{int(round(thickness_um*10)):02d}um"
    (deliv / f"tsenn_absorption_spectra_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote absorption summary to {deliv}")


if __name__ == "__main__":
    main()
