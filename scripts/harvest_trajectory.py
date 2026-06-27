#!/usr/bin/env python3
"""Story B: harvest the RL trajectory as a property-guided *search*.

For a SymmCD/SLME run, joins per-step (composition, eta, band_gap, spacegroup)
via the row-aligned artifacts (validated the same way as scripts/plot_run.py),
then produces:
  - cumulative_hits_vs_step.png : cumulative count of UNIQUE-composition materials
        with eta>0.25 and eta>0.30 discovered up to each RL step (the
        "exploration discovers a diverse pool" plot).
  - diversity_vs_step.png       : unique compositions per step + space-group
        Shannon entropy per step (shows the policy keeps exploring, not collapsing).
  - candidate_shortlist.csv     : every unique-composition eta>0.30 hit
        (max-eta representative), pooled over the trajectory and any --extra-samples
        run dirs (e.g. the large relaxed best/early sampling runs).

CPU-only: per-step eta is already on disk.

Example:
  python scripts/harvest_trajectory.py exp_res/<train_run> \
      --extra-samples exp_res/<best_sample_run> exp_res/<early_sample_run>
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor

ETA_HI = 0.30
ETA_MID = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="training run dir (has samples/, rewards/)")
    p.add_argument(
        "--extra-samples",
        nargs="*",
        type=Path,
        default=[],
        help="additional run dirs whose samples to fold into the shortlist pool",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--eta-hi", type=float, default=ETA_HI)
    p.add_argument("--eta-mid", type=float, default=ETA_MID)
    return p.parse_args()


def parse_float(s: str) -> float | None:
    try:
        v = float(s.strip())
        return v
    except (TypeError, ValueError):
        return None


def detect_eta_dir(run_dir: Path) -> Path:
    rewards = run_dir / "rewards"
    for name in ("tsenn_slme_optimate_eta", "tsenn_slme_eta"):
        if (rewards / name).is_dir():
            return rewards / name
    cands = [d for d in rewards.iterdir() if d.is_dir() and d.name.endswith("_eta")]
    if not cands:
        raise FileNotFoundError(f"No *_eta reward dir under {rewards}")
    return cands[0]


def read_floats(path: Path) -> list[float]:
    out: list[float] = []
    with path.open("r") as fh:
        for line in fh:
            v = parse_float(line)
            out.append(float("nan") if v is None else v)
    return out


def load_run_records(run_dir: Path) -> pd.DataFrame:
    """Per-sample (step, formula, spacegroup, band_gap, eta) for one run dir."""
    samples = run_dir / "samples"
    bg_dir = run_dir / "rewards" / "bandgap"
    eta_dir = detect_eta_dir(run_dir)
    rows: list[dict[str, Any]] = []
    for eval_pt in sorted(samples.glob("step_*_eval.pt")):
        step = int(eval_pt.stem.split("_")[1])
        extxyz = samples / f"step_{step:04d}_eval.extxyz"
        bg_path = bg_dir / f"step_{step:04d}.txt"
        eta_path = eta_dir / f"step_{step:04d}.txt"
        if not (extxyz.exists() and extxyz.stat().st_size and eta_path.exists()):
            continue
        payload = torch.load(eval_pt, map_location="cpu")
        if not isinstance(payload, list):
            continue
        atoms = ase_read(extxyz, index=":")
        if not isinstance(atoms, list):
            atoms = [atoms]
        structures = [AseAtomsAdaptor.get_structure(cast(Any, a)) for a in atoms]
        eta = read_floats(eta_path)
        bg = read_floats(bg_path) if bg_path.exists() else [float("nan")] * len(structures)
        n = len(structures)
        if not (len(payload) == n == len(eta)):
            # tolerate trailing mismatch by truncating to the common length
            n = min(len(payload), n, len(eta))
        for i in range(n):
            item = payload[i]
            sg = None
            if isinstance(item, dict):
                try:
                    sg = int(item.get("spacegroup"))
                except (TypeError, ValueError):
                    sg = None
            rows.append(
                {
                    "step": step,
                    "formula": structures[i].composition.reduced_formula,
                    "spacegroup": sg,
                    "band_gap": bg[i] if i < len(bg) else float("nan"),
                    "eta": eta[i],
                    "cif": structures[i].to(fmt="cif"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No per-step records found under {run_dir}")
    return df


def plot_cumulative_hits(df: pd.DataFrame, out: Path, eta_mid: float, eta_hi: float) -> None:
    steps = sorted(df["step"].unique())
    seen_mid: set[str] = set()
    seen_hi: set[str] = set()
    cmid, chi = [], []
    for s in steps:
        sub = df[df["step"] == s]
        seen_mid |= set(sub.loc[sub["eta"] > eta_mid, "formula"])
        seen_hi |= set(sub.loc[sub["eta"] > eta_hi, "formula"])
        cmid.append(len(seen_mid))
        chi.append(len(seen_hi))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, cmid, "-", color="#2563EB", lw=2, label=f"unique comps, $\\eta>{eta_mid:g}$")
    ax.plot(steps, chi, "-", color="#DC2626", lw=2, label=f"unique comps, $\\eta>{eta_hi:g}$")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Cumulative unique compositions discovered")
    ax.set_title("Trajectory discovery: diverse high-$\\eta$ materials accumulate")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cumulative_hits_vs_step.png", dpi=200)
    plt.close(fig)
    return cmid[-1] if cmid else 0, chi[-1] if chi else 0


def plot_diversity(df: pd.DataFrame, out: Path) -> None:
    steps = sorted(df["step"].unique())
    n_uniq, sg_entropy = [], []
    for s in steps:
        sub = df[df["step"] == s]
        n_uniq.append(sub["formula"].nunique())
        sgs = [x for x in sub["spacegroup"].tolist() if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if sgs:
            counts = np.array(list(Counter(sgs).values()), dtype=float)
            p = counts / counts.sum()
            sg_entropy.append(float(-(p * np.log(p)).sum()))
        else:
            sg_entropy.append(float("nan"))
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    c1 = "#059669"
    ax1.plot(steps, n_uniq, "-", color=c1, lw=2)
    ax1.set_xlabel("RL step")
    ax1.set_ylabel("Unique compositions / step", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    c2 = "#7C3AED"
    ax2.plot(steps, sg_entropy, "--", color=c2, lw=1.5)
    ax2.set_ylabel("Space-group entropy (nats) / step", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax1.set_title("Exploration is sustained across the trajectory")
    fig.tight_layout()
    fig.savefig(out / "diversity_vs_step.png", dpi=200)
    plt.close(fig)


def build_shortlist(frames: list[tuple[str, pd.DataFrame]], eta_hi: float) -> pd.DataFrame:
    parts = []
    for source, df in frames:
        hit = df[df["eta"] > eta_hi].copy()
        hit["source"] = source
        parts.append(hit)
    pooled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if pooled.empty:
        return pooled
    # keep the max-eta representative per composition
    pooled = pooled.sort_values("eta", ascending=False)
    short = pooled.drop_duplicates(subset=["formula"], keep="first").reset_index(drop=True)
    cols = ["formula", "eta", "band_gap", "step", "source", "cif"]
    return short[[c for c in cols if c in short.columns]]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out = (args.output_dir or run_dir / "deliverables" / "trajectory").resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectory records from {run_dir.name} ...")
    traj = load_run_records(run_dir)
    print(f"  {len(traj)} samples over {traj['step'].nunique()} steps")

    nmid, nhi = plot_cumulative_hits(traj, out, args.eta_mid, args.eta_hi)
    plot_diversity(traj, out)
    print(f"  cumulative unique comps: eta>{args.eta_mid:g} -> {nmid}, eta>{args.eta_hi:g} -> {nhi}")

    frames = [("trajectory", traj)]
    for extra in args.extra_samples:
        extra = extra.resolve()
        try:
            df = load_run_records(extra)
            frames.append((extra.name, df))
            print(f"  pooled {len(df)} samples from {extra.name}")
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: skipped {extra}: {e}")

    short = build_shortlist(frames, args.eta_hi)
    csv_path = out / "candidate_shortlist.csv"
    short.to_csv(csv_path, index=False)
    print(
        f"\nShortlist: {len(short)} unique-composition materials with eta>{args.eta_hi:g} "
        f"-> {csv_path}"
    )
    print(f"Plots: {out/'cumulative_hits_vs_step.png'}")
    print(f"       {out/'diversity_vs_step.png'}")


if __name__ == "__main__":
    main()
