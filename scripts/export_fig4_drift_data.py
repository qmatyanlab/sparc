#!/usr/bin/env python3
"""Export a run's Figure-4 panel-(a) drift data to `drift_iso_data.npz`.

Panel (a) of Figure 4 is a KDE over (predicted band gap, combined RL reward) for every
evaluated structure across all loops. `figure_4.py` consumes that as a cached `.npz` so
the figure can be rebuilt with no GPU and no surrogate recompute -- but nothing in the
repo *wrote* that cache; it came from an ad-hoc session. This script is that missing
step, so the panel can be regenerated for any run.

The gap/reward arrays and the reward combination are taken from
`scripts/plot_fig4_drift.py` (`load_gap_raw`, `load_reward_cfg`, `_lin`) so this cache is
by construction the same data that script plots -- verified by `--verify` against an
existing npz.

Usage:
  uv run python scripts/export_fig4_drift_data.py exp_res/<run> -o <run>/figure4_bundle/drift_iso_data.npz
  uv run python scripts/export_fig4_drift_data.py exp_res/<run> --verify <existing.npz>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_fig4_drift import _lin, load_gap_raw, load_reward_cfg  # noqa: E402


def build(run_dir: Path, eps_cap: float = 100.0) -> dict:
    run = Path(run_dir).resolve()
    cfg = load_reward_cfg(run)
    if cfg is None:
        raise SystemExit(f"{run.name}: could not read the reward config")
    gap, raw, step = load_gap_raw(run, eps_cap)
    if gap.size == 0:
        raise SystemExit(f"{run.name}: no finite reward/gap structures")

    # identical to plot_fig4_drift.main(): per-term linear scaling, then the run's reduce
    sd = _lin(raw, cfg["diel"]["minv"], cfg["diel"]["maxv"])
    if cfg["bandgap"] is not None:
        sg = _lin(gap, cfg["bandgap"]["minv"], cfg["bandgap"]["maxv"])
        reward = np.minimum(sd, sg) if cfg["reduce"] == "min" else 0.5 * (sd + sg)
    else:
        reward = sd

    gate_min = float(cfg["bandgap"]["minv"]) if cfg.get("bandgap") else 0.3
    return {
        "gap": gap, "reward": reward, "step": step,
        "gate_min": np.array(gate_min),
        "reduce": np.array(cfg["reduce"]),
        "diel_minv": np.array(float(cfg["diel"]["minv"])),
        "diel_maxv": np.array(float(cfg["diel"]["maxv"])),
        "bg_minv": np.array(float(cfg["bandgap"]["minv"]) if cfg.get("bandgap") else np.nan),
        "bg_maxv": np.array(float(cfg["bandgap"]["maxv"]) if cfg.get("bandgap") else np.nan),
    }


def summarize(d: dict, n: int = 10) -> str:
    gap, reward, step = d["gap"], d["reward"], d["step"]
    steps = np.unique(step)
    k = min(n, max(1, len(steps) // 2))
    em, lm = np.isin(step, steps[:k]), np.isin(step, steps[-k:])
    pg = gap >= float(d["gate_min"])
    f = lambda m: (float(np.median(gap[m & pg])), float(np.median(reward[m & pg])))  # noqa: E731
    e, l = f(em), f(lm)
    met = lambda m: 100 * float(np.mean(~pg[m]))  # noqa: E731
    tgt = lambda m: 100 * float(np.mean(((gap >= 0.3) & (reward >= 0.5))[m & pg]))  # noqa: E731
    return (f"N={gap.size} over {len(steps)} loops | median ({e[0]:.2f},{e[1]:.2f}) -> "
            f"({l[0]:.2f},{l[1]:.2f}) | metals {met(em):.0f}% -> {met(lm):.0f}% | "
            f"in-target {tgt(em):.0f}% -> {tgt(lm):.0f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--eps-cap", type=float, default=100.0)
    ap.add_argument("--verify", type=Path, default=None,
                    help="compare against an existing npz instead of writing")
    args = ap.parse_args(argv)

    d = build(args.run_dir, args.eps_cap)
    print(f"{Path(args.run_dir).name}: {summarize(d)}")

    if args.verify:
        ref = np.load(args.verify)
        bad = []
        for k in sorted(set(ref.files) | set(d)):
            if k not in ref.files or k not in d:
                bad.append(f"{k}: only in {'reference' if k in ref.files else 'rebuild'}")
                continue
            a, b = np.asarray(ref[k]), np.asarray(d[k])
            if a.dtype.kind in "US" or b.dtype.kind in "US":
                ok = str(a) == str(b)
            else:
                ok = a.shape == b.shape and np.allclose(a, b, equal_nan=True)
            if not ok:
                bad.append(f"{k}: differs (ref {a.shape} vs new {b.shape})")
        print("VERIFY:", "identical to the reference npz" if not bad else "MISMATCH")
        for m in bad:
            print("   ", m)
        return 1 if bad else 0

    out = args.output or (Path(args.run_dir) / "figure4_bundle" / "drift_iso_data.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
