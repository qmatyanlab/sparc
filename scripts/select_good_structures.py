#!/usr/bin/env python3
"""Reconstructed "good structures" selector for SymmCD/SLME runs.

Reproduces the deliverables_good_structures(_corrected) outputs: from the whole RL
trajectory it scores every generated sample (NaN-aligned per-step reward files keep
row alignment with structures), filters to "good" and "solar-window" candidates,
diversity-selects the top N, and writes CSVs, CIFs, an extxyz, a structure grid, and
a selection summary.

Scores (verified against the original outputs):
  eta_score      = clip(eta / 0.35, 0, 1)
  ehull_score    = clip(1 - max(ehull, 0) / 0.1, 0, 1)
  good:   band_gap_reward = clip((Eg - 0.5) / (3.0 - 0.5), 0, 1)
  solar:  band_gap_reward = triangular peak at 1.55 eV, 0 at 1.0 and 2.5 eV
  selection_score = 0.55*eta_score + 0.25*band_gap_reward + 0.20*ehull_score
Filters: good = ehull<=0.1 & Eg>=0.5 ; solar = ehull<=0.1 & 1.0<=Eg<=2.5.

Example:
  python scripts/select_good_structures.py exp_res/<run> --output-dir <run>/deliverables_good_structures_corrected
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read as ase_read
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor

ETA_DENOM = 0.35
EHULL_MAX = 0.1
GOOD_BG_MIN = 0.5
SOLAR_BG_LO, SOLAR_BG_HI, SOLAR_BG_PEAK = 1.0, 2.5, 1.55
W_ETA, W_BG, W_EHULL = 0.55, 0.25, 0.20
N_SELECT = 24


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-select", type=int, default=N_SELECT)
    return p.parse_args()


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def fline(path: Path) -> list[float]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        try:
            out.append(float(s))
        except ValueError:
            out.append(float("nan"))
    return out


def detect_eta_subdir(rewards: Path) -> str:
    for name in ("tsenn_slme_optimate", "tsenn_slme"):
        if (rewards / f"{name}_eta").is_dir():
            return name
    raise FileNotFoundError(f"No *_eta reward dir under {rewards}")


def ehull_map(stability_path: Path, n: int) -> np.ndarray:
    vals = np.full(n, np.nan)
    if not stability_path.exists():
        return vals
    with stability_path.open() as fh:
        for row in csv.DictReader(fh):
            if str(row.get("kept", "")).strip().lower() != "true":
                continue
            try:
                ki = int(row["kept_index"])
                eh = float(row["energy_above_hull_ev_per_atom"])
            except (TypeError, ValueError, KeyError):
                continue
            if 0 <= ki < n:
                vals[ki] = eh
    return vals


def load_samples(run_dir: Path) -> list[dict[str, Any]]:
    samples = run_dir / "samples"
    rewards = run_dir / "rewards"
    pre = detect_eta_subdir(rewards)
    rows: list[dict[str, Any]] = []
    for eval_pt in sorted(samples.glob("step_*_eval.pt")):
        step = int(eval_pt.stem.split("_")[1])
        extxyz = samples / f"step_{step:04d}_eval.extxyz"
        if not (extxyz.exists() and extxyz.stat().st_size):
            continue
        payload = torch.load(eval_pt, map_location="cpu")
        if not isinstance(payload, list):
            continue
        atoms = ase_read(extxyz, index=":")
        if not isinstance(atoms, list):
            atoms = [atoms]
        n = len(atoms)
        eta = fline(rewards / f"{pre}_eta" / f"step_{step:04d}.txt")
        bg = fline(rewards / "bandgap" / f"step_{step:04d}.txt")
        jsc = fline(rewards / f"{pre}_jsc" / f"step_{step:04d}.txt")
        voc = fline(rewards / f"{pre}_voc" / f"step_{step:04d}.txt")
        eh = ehull_map(samples / f"step_{step:04d}_stability.csv", n)
        for i in range(n):
            if i >= len(payload) or i >= len(eta) or i >= len(bg):
                break
            e = eta[i]
            if not np.isfinite(e) or not np.isfinite(bg[i]):
                continue
            struct = AseAtomsAdaptor.get_structure(cast(Any, atoms[i]))
            sg = None
            if isinstance(payload[i], dict):
                try:
                    sg = int(payload[i].get("spacegroup"))
                except (TypeError, ValueError):
                    sg = None
            rows.append(
                {
                    "step": step,
                    "index": i,
                    "formula": struct.composition.reduced_formula,
                    "spacegroup": sg,
                    "band_gap_ev": float(bg[i]),
                    "ehull_ev_per_atom": float(eh[i]) if i < len(eh) else float("nan"),
                    "slme_eta_frac": float(e),
                    "slme_eta_percent": float(e) * 100.0,
                    "jsc": float(jsc[i]) if i < len(jsc) and np.isfinite(jsc[i]) else float("nan"),
                    "voc": float(voc[i]) if i < len(voc) and np.isfinite(voc[i]) else float("nan"),
                    "_atoms": atoms[i],
                    "cif": struct.to(fmt="cif"),
                }
            )
    return rows


def triangular_bg(eg: np.ndarray) -> np.ndarray:
    left = (eg - SOLAR_BG_LO) / (SOLAR_BG_PEAK - SOLAR_BG_LO)
    right = (SOLAR_BG_HI - eg) / (SOLAR_BG_HI - SOLAR_BG_PEAK)
    return clip01(np.where(eg <= SOLAR_BG_PEAK, left, right))


def score(df, kind: str):
    eta_s = clip01(df["slme_eta_frac"].to_numpy() / ETA_DENOM)
    eh = df["ehull_ev_per_atom"].to_numpy()
    ehull_s = clip01(1.0 - np.maximum(np.nan_to_num(eh, nan=1e9), 0.0) / EHULL_MAX)
    eg = df["band_gap_ev"].to_numpy()
    bg_s = clip01((eg - 0.5) / (3.0 - 0.5)) if kind == "good" else triangular_bg(eg)
    return eta_s, bg_s, ehull_s, W_ETA * eta_s + W_BG * bg_s + W_EHULL * ehull_s


def diversity_select(df, k: int):
    """unique (formula, spacegroup) pass, then unique formula, then score-fill."""
    order = df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    picked: list[int] = []
    used_f: set = set()
    used_sg: set = set()
    for i, r in order.iterrows():
        if len(picked) >= k:
            break
        if r["formula"] not in used_f and r["spacegroup"] not in used_sg:
            picked.append(i)
            used_f.add(r["formula"])
            used_sg.add(r["spacegroup"])
    for i, r in order.iterrows():
        if len(picked) >= k:
            break
        if i not in picked and r["formula"] not in used_f:
            picked.append(i)
            used_f.add(r["formula"])
    for i in range(len(order)):
        if len(picked) >= k:
            break
        if i not in picked:
            picked.append(i)
    return order.iloc[picked].reset_index(drop=True)


CSV_COLS = ["rank", "step", "index", "formula", "spacegroup", "band_gap_ev",
            "ehull_ev_per_atom", "slme_eta_frac", "slme_eta_percent",
            "band_gap_reward", "eta_score", "ehull_score", "selection_score"]


def write_csv(df, path: Path) -> None:
    out = df.copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    out[CSV_COLS].to_csv(path, index=False)


def render_grid(df, path: Path, title: str) -> None:
    n = len(df)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, df.iterrows()):
        try:
            plot_atoms(r["_atoms"], ax, radii=0.5, rotation="10x,8y,0z")
        except Exception:  # noqa: BLE001
            pass
        ax.set_title(
            f"{r['formula']} (SG{r['spacegroup']})\n"
            f"$\\eta$={r['slme_eta_percent']:.1f}%  $E_g$={r['band_gap_ev']:.2f}",
            fontsize=7,
        )
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_extxyz(df, path: Path) -> None:
    from ase.io import write as ase_write

    ase_write(path, [r["_atoms"] for _, r in df.iterrows()], format="extxyz")


def write_cifs(df, cif_dir: Path) -> None:
    cif_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        (cif_dir / f"{rank:02d}_{r['formula']}_sg{r['spacegroup']}.cif").write_text(r["cif"])


def main() -> None:
    import pandas as pd

    args = parse_args()
    run_dir = args.run_dir.resolve()
    out = (args.output_dir or run_dir / "deliverables_good_structures_corrected").resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = load_samples(run_dir)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} finite samples from {run_dir.name}")

    summary: dict[str, Any] = {
        "note": "Reconstructed; reward files preserve nan lines so structure indices remain aligned.",
        "run": str(run_dir),
        "output_dir": str(out),
    }
    for kind, tag in (("good", "good"), ("solar_window", "solar_window")):
        if kind == "good":
            cand = df[(df["ehull_ev_per_atom"] <= EHULL_MAX) & (df["band_gap_ev"] >= GOOD_BG_MIN)].copy()
        else:
            cand = df[(df["ehull_ev_per_atom"] <= EHULL_MAX)
                      & (df["band_gap_ev"] >= SOLAR_BG_LO)
                      & (df["band_gap_ev"] <= SOLAR_BG_HI)].copy()
        eta_s, bg_s, ehull_s, sc = score(cand, kind)
        cand["eta_score"], cand["band_gap_reward"], cand["ehull_score"], cand["selection_score"] = eta_s, bg_s, ehull_s, sc
        cand = cand.sort_values("selection_score", ascending=False).reset_index(drop=True)
        sel = diversity_select(cand, args.n_select)

        base = "good_structure" if kind == "good" else "solar_window"
        write_csv(cand, out / f"all_{base}_candidates.csv")
        write_csv(sel, out / (f"selected_good_structures.csv" if kind == "good" else "selected_solar_window_structures.csv"))
        write_extxyz(sel, out / (f"selected_good_structures.extxyz" if kind == "good" else "selected_solar_window_structures.extxyz"))
        write_cifs(sel, out / ("selected_cifs" if kind == "good" else "solar_window_cifs"))
        render_grid(sel, out / (f"selected_good_structures.png" if kind == "good" else "selected_solar_window_structures.png"),
                    f"{kind} structures — {run_dir.name}")
        top = sel.head(4)[["step", "index", "formula", "spacegroup", "band_gap_ev",
                           "ehull_ev_per_atom", "slme_eta_percent", "jsc", "voc", "selection_score"]]
        summary[f"candidate_count_{kind}"] = int(len(cand))
        summary[f"selected_count_{kind}"] = int(len(sel))
        summary[f"top_{kind}"] = top.to_dict(orient="records")
        print(f"  {kind}: {len(cand)} candidates -> {len(sel)} selected; best={sel.iloc[0]['formula']} "
              f"score={sel.iloc[0]['selection_score']:.4f}")

    (out / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote deliverables to {out}")


if __name__ == "__main__":
    main()
