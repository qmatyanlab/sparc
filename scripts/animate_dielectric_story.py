#!/usr/bin/env python3
"""Animate the dielectric composite story as the RL loop tightens (step 0 -> max) into a GIF.

Each frame = one RL step j, laid out like scripts/plot_dielectric_story_composite.py (panel a
schematic on the left, reward (b) + space-group (c) on the right):

  (a) the composite schematic (reused via plot_dielectric_story_composite._draw_schematic),
      redrawn with step j's HIGHEST-combined-reward sample (crystal + predicted eps) and the
      reward-updated SG proposal policy pi^(j) (initial pi^0 stays fixed). The shown sample's
      reward r is printed in the banner.
  (b) the reward-mean curve REVEALED up to step j, with a red dot at the growing tip.
  (c) the CUMULATIVE realized space-group distribution through step j, on a fixed top-N SG axis
      (bars grow in place, no re-ranking jitter).

Frames are stitched into a looping GIF with Pillow (matching the repo's other GIF builders).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import plot_dielectric_story_composite as C  # noqa: E402  (also applies publication style)
from plot_layered_uniaxial_samples import load_step_records  # noqa: E402
try:
    from plot_eval_symmetry_inset import _hm  # SG number -> Hermann-Mauguin symbol
except Exception:  # noqa: BLE001
    def _hm(sg):  # fallback
        return ""


def _lin(v, a, b):
    return float(np.clip((v - a) / (b - a + 1e-12), 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--stride", type=int, default=2, help="render every Nth RL step")
    p.add_argument("--max-step", type=int, default=None, help="last step (default: last on disk)")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--window", type=int, default=10, help="reward EMA span")
    p.add_argument("--top", type=int, default=12, help="top-N space groups in panel (c)")
    p.add_argument("--diel-scale", type=float, nargs=2, default=(0.9, 1.0))
    p.add_argument("--gap-scale", type=float, nargs=2, default=(0.3, 0.8))
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--output", type=Path, default=None,
                   help="gif path (default: <run>/deliverables_layered_uniaxial/"
                        "dielectric_story_animation.gif)")
    return p.parse_args()


def _step_spacegroups(samples: Path, step: int) -> list[int]:
    pt = samples / f"step_{step:04d}_eval.pt"
    if not pt.exists():
        return []
    try:
        data = torch.load(pt, map_location="cpu")
    except Exception:  # noqa: BLE001
        return []
    out = []
    for d in (data if isinstance(data, list) else []):
        sg = d.get("spacegroup") if isinstance(d, dict) else None
        if sg is not None:
            out.append(int(sg))
    return out


def _best_sample(run: Path, step: int, ga, gb, da, db, r_uni_min=0.5):
    """Step j's most INTERESTING sample: among the in-plane-isotropic + past-gate structures
    (combined r_uni >= r_uni_min, positive-definite, no eps blow-up), the one with the LARGEST
    out-of-plane anisotropy |eps_zz - eps_par|/(...) -- a clearly uniaxial eps_zz != eps_xx=eps_yy,
    not a near-cubic one. Falls back to the max-r_uni sample when none clear the target (early)."""
    try:
        recs = load_step_records(run, requested_step=step)
    except Exception:  # noqa: BLE001
        return None
    if not recs:
        return None
    bg = run / "rewards" / "bandgap" / f"step_{step:04d}.txt"
    gaps = (np.array([float(x) for x in bg.read_text().split() if x.strip()])
            if bg.exists() else np.array([]))
    scored = []
    for r in recs:
        gap = float(gaps[r["index"]]) if r["index"] < len(gaps) else 0.0
        r_uni = min(_lin(gap, ga, gb), _lin(float(r["reward"]), da, db))
        perp = 0.5 * (r["eps_xx"] + r["eps_yy"])
        az = abs(r["eps_zz"] - perp) / (abs(r["eps_zz"]) + abs(perp) + 1e-8)
        emax = max(abs(r["eps_xx"]), abs(r["eps_yy"]), abs(r["eps_zz"]))
        c = dict(r); c.update(r_uni=r_uni, gap=gap, az=az, emax=emax)
        scored.append(c)
    good = [c for c in scored
            if c["r_uni"] >= r_uni_min and c["positive_definite"] and c["emax"] < 60]
    return max(good, key=lambda c: c["az"]) if good else max(scored, key=lambda c: c["r_uni"])


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    samples = run / "samples"
    out = (args.output or (run / "deliverables_layered_uniaxial"
                           / "dielectric_story_animation.gif")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "_anim_frames"
    tmp.mkdir(exist_ok=True)
    plt.rcParams["mathtext.cal"] = "cmsy10"  # calligraphic script M' in the schematic

    steps_all = sorted(int(p.stem.split("_")[1]) for p in samples.glob("step_*_eval.pt"))
    max_step = args.max_step if args.max_step is not None else steps_all[-1]
    frame_steps = [s for s in steps_all if s <= max_step and s % args.stride == 0]

    # reward curve + debiased EMA (revealed progressively)
    rsteps, reward = C._load_reward_series(run)
    reward = np.asarray(reward, dtype=float)
    ema = np.full_like(reward, np.nan)
    m = np.isfinite(reward)
    if m.any():
        import pandas as pd
        e = pd.Series(reward[m]).ewm(span=args.window, adjust=True).mean().to_numpy()
        ema[m] = e
    ylim_b = (float(np.nanmin(reward)) - 0.03, float(np.nanmax(reward)) + 0.05)

    # cumulative SG counts over ALL steps 0..max (snapshot at each frame step)
    da, db = args.diel_scale
    ga, gb = args.gap_scale
    cum = Counter()
    frames_data = {}   # frame_step -> (cum_counts_copy, best_record)
    last_best = None
    for s in [x for x in steps_all if x <= max_step]:
        cum.update(_step_spacegroups(samples, s))
        if s in frame_steps:
            best = _best_sample(run, s, ga, gb, da, db) or last_best
            last_best = best
            frames_data[s] = (Counter(cum), best)

    # fixed top-N SG axis + fixed x-max from the FINAL cumulative distribution
    final_cum = frames_data[frame_steps[-1]][0]
    top_sgs = [sg for sg, _ in final_cum.most_common(args.top)]
    ftot = sum(final_cum.values()) or 1
    xmax_c = max(1.0, 100 * max(final_cum[sg] for sg in top_sgs) / ftot * 1.15)
    sg_labels = [_hm(sg) or str(sg) for sg in top_sgs]

    frame_paths = []
    for k, s in enumerate(frame_steps):
        cum_counts, best = frames_data[s]
        fig = plt.figure(figsize=(15.0, 8.0))
        outer = GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.10,
                         left=0.04, right=0.978, top=0.92, bottom=0.085, figure=fig)
        ax_a = fig.add_subplot(outer[0])
        right = outer[1].subgridspec(2, 1, height_ratios=[0.80, 1.20], hspace=0.38)
        ax_b = fig.add_subplot(right[0])
        ax_c = fig.add_subplot(right[1])

        # (a) schematic for this step's best sample + pi^(j)
        if best is not None:
            C._draw_schematic(fig, ax_a, best, run, after_step=s)

        # (b) reward curve revealed up to step s
        ridx = int(np.searchsorted(rsteps, s, side="right"))
        ax_b.plot(rsteps[:ridx], reward[:ridx], color="0.75", lw=1.0, zorder=2)
        ax_b.plot(rsteps[:ridx], ema[:ridx], color="k", lw=2.2, zorder=3)
        if ridx > 0:
            ax_b.scatter([rsteps[ridx - 1]], [ema[ridx - 1]], color="#e11", s=70,
                         edgecolor="k", linewidths=0.8, zorder=5)
        ax_b.set_xlim(0, max_step)
        ax_b.set_ylim(*ylim_b)
        ax_b.set_xlabel("RL step", fontsize=12)
        ax_b.set_ylabel("Reward mean", fontsize=12)
        ax_b.tick_params(labelsize=10)

        # (c) cumulative realized SG distribution (fixed top-N axis, bars grow)
        tot = sum(cum_counts.values()) or 1
        fracs = [100 * cum_counts.get(sg, 0) / tot for sg in top_sgs]
        y = np.arange(len(top_sgs))
        ax_c.barh(y, fracs, color="#1f4fd8", edgecolor="none")
        ax_c.set_yticks(y)
        ax_c.set_yticklabels(sg_labels, fontsize=10)
        ax_c.invert_yaxis()
        ax_c.set_xlim(0, xmax_c)
        ax_c.tick_params(axis="x", labelsize=10)
        ax_c.set_xlabel("Fraction of structures (%)", fontsize=12)
        ax_c.set_title("Cumulative space groups", fontsize=12)

        fp = tmp / f"frame_{k:04d}.png"
        fig.savefig(fp, dpi=args.dpi)
        plt.close(fig)
        frame_paths.append(fp)
        if (k + 1) % 10 == 0 or k == len(frame_steps) - 1:
            print(f"  rendered {k + 1}/{len(frame_steps)} frames (step {s})")

    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    dur = int(1000 / args.fps)
    durations = [dur] * len(imgs)
    durations[-1] = dur * 12   # hold on the final frame
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=durations,
                 loop=0, optimize=True, disposal=2)
    print(f"-> {out}  ({len(imgs)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
