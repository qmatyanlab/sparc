#!/usr/bin/env python3
"""Replay a run's recorded adaptive-SG observation stream (step_count,
step_mean_reward per spacegroup per step) through alternative update-rule
configurations (anchor temperature / uniform, reward_scale, mix mode), to
study collapse/exploration dynamics on identical data.

Off-policy caveat: the observation stream was generated under the original
run's proposal policy, so variants are compared on what THAT run observed;
groups it rarely sampled carry sparse reward signal.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MIN_PROB = 1e-4
EMA_DECAY = 0.9
TRACK_SGS = [225, 123, 156, 191, 183]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_stream(adaptive_dir: Path):
    paths = sorted(
        (p for p in adaptive_dir.glob("step_*.csv")),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    base = None
    stream = []
    for path in paths:
        counts = np.zeros(230)
        means = np.full(230, np.nan)
        b = np.zeros(230)
        for row in csv.DictReader(open(path)):
            sg = int(row["spacegroup"]) - 1
            b[sg] = float(row["base_prob"])
            counts[sg] = float(row["step_count"])
            try:
                means[sg] = float(row["step_mean_reward"])
            except ValueError:
                pass
        base = b
        stream.append((int(path.stem.split("_")[1]), counts, means))
    return base, stream


def make_anchor(base: np.ndarray, kind: str) -> np.ndarray:
    support = base > 0
    if kind == "base":
        return base / base.sum()
    if kind.startswith("T"):
        t = float(kind[1:])
        a = np.zeros_like(base)
        a[support] = base[support] ** (1.0 / t)
        return a / a.sum()
    if kind == "uniform":
        a = np.where(support, 1.0, 0.0)
        return a / a.sum()
    raise ValueError(kind)


def replay(base, stream, anchor_kind, scale, mix_previous, prior_mix):
    """Mirror of pipeline.sparc.SPARC._update_adaptive_sg_policy."""
    anchor = make_anchor(base, anchor_kind)
    support = anchor > 0
    ema = np.zeros(230)
    dist = anchor.copy()
    history = []
    for step, counts, means in stream:
        observed = counts > 0
        if observed.any():
            ema *= EMA_DECAY
            ema[observed] += (1.0 - EMA_DECAY) * means[observed]
            scores = scale * ema
            proposal = anchor * np.exp(scores - scores.max())
            proposal[~support] = 0.0
            proposal = proposal / proposal.sum()
            proposal[support] = np.maximum(proposal[support], MIN_PROB)
            proposal[~support] = 0.0
            proposal = proposal / proposal.sum()
            if mix_previous:
                dist = prior_mix * dist + (1.0 - prior_mix) * proposal
            else:
                dist = prior_mix * anchor + (1.0 - prior_mix) * proposal
            dist = dist / dist.sum()
        h = -(dist[dist > 0] * np.log(dist[dist > 0])).sum()
        history.append(
            {
                "step": step,
                "top_share": float(dist.max()),
                "top_sg": int(dist.argmax()) + 1,
                "eff_sgs": float(np.exp(h)),
                **{f"sg{g}": float(dist[g - 1]) for g in TRACK_SGS},
            }
        )
    return dist, np.array([(r["step"], r["top_share"], r["eff_sgs"]) for r in history]), history


VARIANTS = {
    "A: run-A actual (base,s=3,mix_prev,0.4)": dict(
        anchor_kind="base", scale=3.0, mix_previous=True, prior_mix=0.4
    ),
    "B: uniform,s=5,mix_prev,0.4 (succession)": dict(
        anchor_kind="uniform", scale=5.0, mix_previous=True, prior_mix=0.4
    ),
    "C: uniform,s=5,mix_base,0.15": dict(
        anchor_kind="uniform", scale=5.0, mix_previous=False, prior_mix=0.15
    ),
    "D: uniform,s=3,mix_base,0.15": dict(
        anchor_kind="uniform", scale=3.0, mix_previous=False, prior_mix=0.15
    ),
    "E: T2,s=5,mix_base,0.15": dict(
        anchor_kind="T2", scale=5.0, mix_previous=False, prior_mix=0.15
    ),
}


def main() -> None:
    args = parse_args()
    adaptive_dir = args.exp_dir / "adaptive_spacegroup"
    output_dir = args.output_dir or (args.exp_dir / "deliverables" / "adaptive_sg_replay")
    output_dir.mkdir(parents=True, exist_ok=True)

    base, stream = load_stream(adaptive_dir)
    print(f"loaded {len(stream)} steps, support={int((base > 0).sum())} SGs")

    results = {}
    for name, cfg in VARIANTS.items():
        final, traj, history = replay(base, stream, **cfg)
        results[name] = (final, traj, history)

    # sanity: variant A vs recorded current_prob at final step
    last_csv = sorted(
        adaptive_dir.glob("step_*.csv"), key=lambda p: int(p.stem.split("_")[1])
    )[-1]
    recorded = np.zeros(230)
    for row in csv.DictReader(open(last_csv)):
        recorded[int(row["spacegroup"]) - 1] = float(row["current_prob"])
    diff = np.abs(results[list(VARIANTS)[0]][0] - recorded).max()
    print(f"sanity replay-vs-recorded (variant A, {last_csv.name}): max|diff|={diff:.2e}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for name, (final, traj, history) in results.items():
        steps = traj[:, 0]
        axes[0, 0].plot(steps, traj[:, 1], label=name, alpha=0.85)
        axes[0, 1].plot(steps, traj[:, 2], label=name, alpha=0.85)
        axes[1, 0].plot(steps, [h["sg183"] for h in history], alpha=0.85)
        axes[1, 1].plot(steps, [h["sg225"] for h in history], alpha=0.85)
    axes[0, 0].set_title("top-group proposal share")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].set_title("effective #SGs (exp entropy)")
    axes[1, 0].set_title("SG183 (P6mm, rare-but-good) proposal prob")
    axes[1, 1].set_title("SG225 (Fm-3m, zero-reward) proposal prob")
    for ax in axes.flat:
        ax.set_xlabel("RL step")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "replay_variants.png", dpi=150)

    print(f"\n{'variant':<42} {'top_share':>9} {'top_sg':>6} {'effSG':>6} "
          f"{'sg183':>7} {'sg225':>7}")
    for name, (final, traj, history) in results.items():
        h = history[-1]
        print(f"{name:<42} {h['top_share']:9.3f} {h['top_sg']:>6} "
              f"{h['eff_sgs']:6.1f} {h['sg183']:7.4f} {h['sg225']:7.4f}")
    print(f"\nfigure: {output_dir / 'replay_variants.png'}")


if __name__ == "__main__":
    main()
