#!/usr/bin/env python3
"""Plot predicted band gap vs reward/score vs Ehull for RL run candidates.

For each run, every eval sample is shown as a point: x = E3NN-predicted band gap
(optuna_bandgap_trial_2, loaded with the correct utils_model_scalar.Network -- see
scripts/filter_static_dielectric_by_bandgap.py), y = the run's score (SLME eta or the
dielectric-anisotropy reward), colour = Ehull (clipped). Band gaps are recomputed with
the SAME E3NN model across runs for consistency.

NOTE: SLME eta stored in the run was computed with ALIGNN gaps as the integration bound,
so the y(eta)-vs-x(E3NN gap) panel has a small built-in model inconsistency (annotated).

Example:
  .venv/bin/python scripts/plot_candidates_gap_score_ehull.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.filter_static_dielectric_by_bandgap import (  # noqa: E402
    load_bandgap_model, predict_band_gaps, DEFAULT_MODEL, DEFAULT_CONFIG)

EXP = ROOT / "exp_res"
OUT = ROOT / "exp_res/_candidate_plots"

RUNS = [
    {
        "key": "slme_54346667",
        "title": "SLME (optimate adaptive) 54346667",
        "dir": EXP / "tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1_54346667",
        "score_dir": "tsenn_slme_optimate_eta",
        "score_label": r"SLME $\eta$ (%)",
        "score_mult": 100.0,
    },
    {
        "key": "staticdie_54584243",
        "title": "Static dielectric (layered uniaxial) 54584243",
        "dir": EXP / "tsenn_static_dielectric_layered_uniaxial_symmcd_v3_uniform_anchor_54584243",
        "score_dir": "tsenn_static_dielectric_layered_uniaxial",
        "score_label": "Dielectric anisotropy reward",
        "score_mult": 1.0,
    },
]

EHULL_CLIP = 0.1          # eV/atom; points above are clipped to the colourbar top
SOLAR_LO, SOLAR_HI = 1.0, 1.5  # eV reference window


def fline(p: Path) -> list[float]:
    try:
        return [float(x) for x in p.read_text().split()]
    except (OSError, ValueError):
        return []


def ehull_by_energy(stab: Path) -> dict[float, float]:
    out: dict[float, float] = {}
    if not stab.exists():
        return out
    for r in csv.DictReader(stab.open()):
        try:
            out[round(float(r["total_energy_ev"]), 3)] = float(
                r["energy_above_hull_ev_per_atom"])
        except (TypeError, ValueError, KeyError):
            pass
    return out


def build_df(run: dict[str, Any], model, r_max: float, device: str):
    import pandas as pd
    samples = run["dir"] / "samples"
    sdir = run["dir"] / "rewards" / run["score_dir"]
    rows: list[dict[str, Any]] = []
    structs: list[Any] = []
    for ext in sorted(samples.glob("step_*_eval.extxyz")):
        step = int(ext.stem.split("_")[1])
        if not ext.stat().st_size:
            continue
        atoms = ase_read(ext, index=":")
        if not isinstance(atoms, list):
            atoms = [atoms]
        score = fline(sdir / f"step_{step:04d}.txt")
        eh = ehull_by_energy(samples / f"step_{step:04d}_stability.csv")
        n = min(len(atoms), len(score) if score else len(atoms))
        for i in range(n):
            s = AseAtomsAdaptor.get_structure(atoms[i])
            te = round(atoms[i].info.get("total_energy", 1e9), 3)
            rows.append({
                "step": step, "index": i,
                "formula": s.composition.reduced_formula,
                "score": float(score[i]) * run["score_mult"],
                "ehull": eh.get(te, np.nan),
            })
            structs.append(s)
    df = pd.DataFrame(rows)
    print(f"  {run['key']}: {len(df)} eval samples; "
          f"ehull matched {df['ehull'].notna().sum()}/{len(df)}; predicting E3NN gap...")
    df["band_gap_ev"] = predict_band_gaps(structs, model, r_max, device, 256)
    return df


def plot(dfs: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = ROOT / "publication.mplstyle"
    if style.exists():
        plt.style.use(str(style))

    n = len(RUNS)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4))
    axes = np.atleast_1d(axes).ravel()
    sc = None
    for ax, run in zip(axes, RUNS):
        df = dfs[run["key"]]
        c = np.clip(df["ehull"].to_numpy(), 0.0, EHULL_CLIP)
        ax.axvspan(SOLAR_LO, SOLAR_HI, color="0.85", zorder=0,
                   label=f"{SOLAR_LO}-{SOLAR_HI} eV")
        sc = ax.scatter(df["band_gap_ev"], df["score"], c=c, cmap="viridis_r",
                        vmin=0.0, vmax=EHULL_CLIP, s=14, alpha=0.7, linewidths=0)
        ax.set_xlabel("Predicted band gap $E_g$ (eV, E3NN optuna_bandgap)")
        ax.set_ylabel(run["score_label"])
        ax.set_title(run["title"], fontsize=9)
        ax.set_xlim(-0.3, min(6.0, float(np.nanmax(df["band_gap_ev"])) + 0.3))
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    cbar = fig.colorbar(sc, ax=list(axes), fraction=0.046, pad=0.02)
    cbar.set_label(f"$E_\\mathrm{{hull}}$ (eV/atom, clipped at {EHULL_CLIP})")
    fig.suptitle("Candidate band gap vs score, coloured by stability", fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"candidates_gap_score_ehull.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {OUT}/candidates_gap_score_ehull.png")


def main() -> None:
    device = "cuda"
    print(f"Loading E3NN band-gap model {DEFAULT_MODEL.name} ...")
    model, r_max = load_bandgap_model(DEFAULT_CONFIG, DEFAULT_MODEL, device)
    dfs = {}
    for run in RUNS:
        dfs[run["key"]] = build_df(run, model, r_max, device)
        OUT.mkdir(parents=True, exist_ok=True)
        dfs[run["key"]].to_csv(OUT / f"{run['key']}_candidates.csv", index=False)
    plot(dfs)


if __name__ == "__main__":
    main()
