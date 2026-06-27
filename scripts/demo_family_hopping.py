#!/usr/bin/env python3
"""Demonstrate that reward oscillation in an adaptive-SG SLME run is the
population hopping between crystal (space-group) families, exploring local
minima and harvesting candidates across many of them.

Reads per-step scored structures from samples/step_*_eval.pt (SymmCD-assigned
spacegroup + atom_types) and per-step eta from rewards/, reward mean from
metrics.csv. No structure rebuilding needed (fast).

Produces:
  family_hopping.png   - reward(step) | family streamgraph | (step,SG) strip by eta
  family_harvest.png   - cumulative distinct high-eta candidates stacked by family
and prints the turnover<->reward correlation.

Example:
    python scripts/demo_family_hopping.py \
        exp_res/tsenn_slme_03um_symmcd_v3_seed1_optimate_adaptive_53904266
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Z -> symbol for reduced-formula identity of candidates
from pymatgen.core.periodic_table import Element

TOP_N_FAMILIES = 9
ETA_HI = 0.25  # "candidate" threshold (eta fraction)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--eta-dir", default="rewards/tsenn_slme_optimate_eta")
    p.add_argument("--eta-hi", type=float, default=ETA_HI)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def crystal_system(sg: int) -> str:
    if sg <= 2: return "triclinic"
    if sg <= 15: return "monoclinic"
    if sg <= 74: return "orthorhombic"
    if sg <= 142: return "tetragonal"
    if sg <= 167: return "trigonal"
    if sg <= 194: return "hexagonal"
    return "cubic"


def formula_of(atom_types) -> str:
    counts = Counter(int(z) for z in atom_types.tolist())
    parts = []
    for z in sorted(counts):
        try:
            sym = Element.from_Z(z).symbol
        except Exception:
            sym = f"Z{z}"
        parts.append(f"{sym}{counts[z]}")
    return "".join(parts)


def load_steps(run_dir: Path, eta_sub: str):
    samples = run_dir / "samples"
    eta_dir = run_dir / eta_sub
    recs = []
    for pt in sorted(samples.glob("step_*_eval.pt")):
        step = int(pt.stem.split("_")[1])
        eta_path = eta_dir / f"step_{step:04d}.txt"
        payload = torch.load(pt, map_location="cpu")
        if not isinstance(payload, list):
            continue
        eta = (np.loadtxt(eta_path, ndmin=1) if eta_path.exists()
               else np.full(len(payload), np.nan))
        for i, item in enumerate(payload):
            sg = item.get("spacegroup") if isinstance(item, dict) else None
            if sg is None:
                continue
            recs.append({
                "step": step,
                "sg": int(sg),
                "system": crystal_system(int(sg)),
                "formula": formula_of(item["atom_types"]),
                "eta": float(eta[i]) if i < len(eta) else np.nan,
            })
    return recs


def reward_by_step(run_dir: Path):
    out = {}
    with (run_dir / "metrics.csv").open() as fh:
        for r in csv.DictReader(fh):
            try:
                out[int(float(r["step"]))] = float(r["reward mean"])
            except (KeyError, ValueError):
                pass
    return out


def js_divergence(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main() -> None:
    args = parse_args()
    out = args.output_dir or (args.run_dir / "deliverables" / "family_hopping")
    out.mkdir(parents=True, exist_ok=True)

    recs = load_steps(args.run_dir, args.eta_dir)
    steps = sorted({r["step"] for r in recs})
    rw = reward_by_step(args.run_dir)
    rewards = np.array([rw.get(s, np.nan) for s in steps])

    # family = top-N space groups by total frequency, rest -> "other"
    sg_counts = Counter(r["sg"] for r in recs)
    top_sgs = [sg for sg, _ in sg_counts.most_common(TOP_N_FAMILIES)]
    fam_labels = [f"SG{sg} ({crystal_system(sg)[:4]})" for sg in top_sgs] + ["other"]

    def fam_index(sg):
        return top_sgs.index(sg) if sg in top_sgs else len(top_sgs)

    # per-step family fraction matrix
    nfam = len(top_sgs) + 1
    frac = np.zeros((len(steps), nfam))
    for si, s in enumerate(steps):
        c = Counter(fam_index(r["sg"]) for r in recs if r["step"] == s)
        tot = sum(c.values())
        for f, n in c.items():
            frac[si, f] = n / tot if tot else 0.0

    # turnover = JS divergence of family distribution vs previous step
    turnover = np.zeros(len(steps))
    for si in range(1, len(steps)):
        turnover[si] = js_divergence(frac[si], frac[si - 1])

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(nfam - 1)] + [(0.7, 0.7, 0.7, 1.0)]

    # ---- Figure 1: reward | streamgraph | (step,SG) strip ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1.6, 2.2]})
    axes[0].plot(steps, rewards, color="black", lw=1.5)
    axes[0].fill_between(steps, rewards, alpha=0.1, color="black")
    axes[0].set_ylabel("reward mean")
    axes[0].set_title("(a) reward oscillates")

    axes[1].stackplot(steps, frac.T, labels=fam_labels, colors=colors, alpha=0.9)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("family fraction\n(scored structures)")
    axes[1].set_title("(b) population composition by space-group family — churns step to step")
    axes[1].legend(ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.04))

    # strip: each scored structure at (step, sg), colored by eta
    xs = [r["step"] for r in recs]
    ys = [r["sg"] for r in recs]
    cs = [r["eta"] for r in recs]
    sc = axes[2].scatter(xs, ys, c=cs, cmap="viridis", s=14, alpha=0.7,
                         vmin=0, vmax=0.35)
    axes[2].set_ylabel("space group of scored structure")
    axes[2].set_xlabel("RL step")
    axes[2].set_title("(c) every scored structure: hops across SG families; high-η hits (yellow) land in many")
    cb = fig.colorbar(sc, ax=axes[2], pad=0.01); cb.set_label("predicted η")
    fig.tight_layout()
    fig.savefig(out / "family_hopping.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: cumulative distinct high-eta candidates, stacked by family ----
    fig2, ax = plt.subplots(figsize=(12, 6))
    seen = set()
    cum = np.zeros((len(steps), nfam))
    for si, s in enumerate(steps):
        if si:
            cum[si] = cum[si - 1]
        for r in recs:
            if r["step"] != s or not (r["eta"] >= args.eta_hi):
                continue
            key = r["formula"]
            if key in seen:
                continue
            seen.add(key)
            cum[si, fam_index(r["sg"])] += 1
    ax.stackplot(steps, cum.T, labels=fam_labels, colors=colors, alpha=0.9)
    ax.set_xlabel("RL step")
    ax.set_ylabel(f"cumulative distinct candidates (η ≥ {args.eta_hi})")
    ax.set_title("Exploration harvests distinct high-η candidates from MANY families")
    ax.legend(ncol=5, fontsize=7, loc="upper left")
    fig2.tight_layout()
    fig2.savefig(out / "family_harvest.png", dpi=150)
    plt.close(fig2)

    # ---- quantitative: turnover vs reward ----
    dr = np.abs(np.diff(rewards))
    t1 = turnover[1:]
    ok = np.isfinite(dr) & np.isfinite(t1)
    corr = np.corrcoef(t1[ok], dr[ok])[0, 1] if ok.sum() > 2 else float("nan")
    print(f"steps={len(steps)} | reward mean range {np.nanmin(rewards):.3f}-{np.nanmax(rewards):.3f}")
    print(f"families tracked (top {TOP_N_FAMILIES} SG): {top_sgs}")
    print(f"mean step-to-step family turnover (JS div, bits): {turnover[1:].mean():.3f}")
    print(f"corr( family turnover , |Δreward| ) = {corr:.3f}")
    n_cand = int(cum[-1].sum())
    fams_with_cand = int((cum[-1] > 0).sum())
    print(f"distinct η≥{args.eta_hi} candidates: {n_cand} across {fams_with_cand}/{nfam} families")
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
