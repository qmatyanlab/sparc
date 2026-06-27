#!/usr/bin/env python3
"""Rescore a finished SLME run's rewards with a different reward config (e.g. E3NN gap).

Drives `rewards.reward.Reward` built from the chosen reward config to recompute, for every
per-step *eval* set, every property (band gap, SLME eta/jsc/voc) and the combined reward --
writing per-prop `step_XXXX.txt` files in the standard layout under a dedicated output subdir,
plus a rescored `metrics.csv` (the property/reward columns recomputed from the rescored values,
all gap-independent columns copied from the original metrics.csv aligned by step). Also builds a
self-contained "run view" (symlinks to samples/.hydra/adaptive_spacegroup) so the existing plot
scripts work unmodified against the rescored folder.

Reward semantics are reused verbatim (same Reward class + config), so the only thing that changes
is the band-gap source (eg_mode=e3nn) and therefore eta (gap = SLME integration lower bound).

Example:
  .venv/bin/python scripts/rescore_run_rewards.py \
      exp_res/tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1_54346667 \
      --reward-config configs/reward/tsenn_slme_optimate_bg02_eta08_e3nngap.yaml \
      --output-subdir e3nn_bandgap
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pymatgen.io.ase import AseAtomsAdaptor

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--reward-config", type=Path,
                   default=ROOT / "configs/reward/tsenn_slme_optimate_bg02_eta08_e3nngap.yaml")
    p.add_argument("--output-subdir", default="e3nn_bandgap")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-steps", type=int, default=None,
                   help="only process the first N steps (smoke test)")
    return p.parse_args()


def build_view(run_dir: Path, out_root: Path) -> None:
    """Symlink the unchanged run inputs into the rescored folder."""
    out_root.mkdir(parents=True, exist_ok=True)
    for name in ("samples", ".hydra", "adaptive_spacegroup", "hparams.yaml"):
        src = run_dir / name
        if not src.exists():
            continue
        link = out_root / name
        if link.is_symlink() or link.exists():
            continue
        os.symlink(src.resolve(), link)


def load_reward(cfg_path: Path, out_root: Path, device: str):
    """Instantiate the Reward from the config, pointing every calculator at out_root/rewards."""
    text = cfg_path.read_text()
    text = text.replace("${hydra:runtime.cwd}", str(ROOT)).replace("${device}", device)
    cfg = OmegaConf.create(text)
    cfg.root_dir = str(out_root / "rewards")
    for prop in cfg.prop_cfg:
        # original root_dir is relative, e.g. "rewards/bandgap"
        prop.calculator.root_dir = str(out_root / prop.calculator.root_dir)
    reward = instantiate(cfg, _recursive_=True)
    # name -> the txt directory each calculator writes into
    txt_dirs = {prop.name: Path(prop.calculator.root_dir) for prop in cfg.prop_cfg}
    return reward, txt_dirs


def step_metrics(rewards: np.ndarray, prop_dict: dict, failed: np.ndarray) -> dict:
    """Reproduce pipeline/base.py metric semantics: means/std over non-failed structures
    (prop_dict already has nan->0). Keys match metrics.csv columns ("<name> mean/std")."""
    ok = ~np.asarray(failed, dtype=bool)
    sr = np.asarray(rewards, dtype=float)[ok]
    m: dict[str, float] = {
        "reward mean": float(sr.mean()) if sr.size else float("nan"),
        "reward std": float(sr.std()) if sr.size else float("nan"),
    }
    for name, v in prop_dict.items():
        vv = np.asarray(v, dtype=float)[ok]
        m[f"{name} mean"] = float(vv.mean()) if vv.size else float("nan")
        m[f"{name} std"] = float(vv.std()) if vv.size else float("nan")
    return m


def write_metrics(run_dir: Path, out_root: Path,
                  metrics_by_step: dict[int, dict]) -> None:
    """Override recomputed property/reward columns; copy every other column by step."""
    orig = run_dir / "metrics.csv"
    rows = list(csv.DictReader(orig.open())) if orig.exists() else []
    fieldnames = list(rows[0].keys()) if rows else []
    for row in rows:
        step = int(float(row["step"]))
        m = metrics_by_step.get(step, {})
        for col, val in m.items():
            if col in row:
                row[col] = f"{val:.6f}"
    with (out_root / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_root = (run_dir / args.output_subdir).resolve()
    build_view(run_dir, out_root)

    reward, txt_dirs = load_reward(args.reward_config, out_root, args.device)
    print(f"Rescoring {run_dir.name} -> {out_root}/ with {args.reward_config.name}")
    print(f"  props: {list(txt_dirs)}")

    samples = run_dir / "samples"
    ada = AseAtomsAdaptor()
    steps = sorted(int(p.stem.split("_")[1]) for p in samples.glob("step_*_eval.extxyz"))
    if args.limit_steps:
        steps = steps[: args.limit_steps]
    metrics_by_step: dict[int, dict] = {}
    per_sample: list[dict] = []
    prop_names = [p.name for p in reward.prop_cfg]
    for step in steps:
        ext = samples / f"step_{step:04d}_eval.extxyz"
        if not ext.stat().st_size:
            continue
        atoms = ase_read(ext, index=":")
        if not isinstance(atoms, list):
            atoms = [atoms]
        structs = [ada.get_structure(a) for a in atoms]
        rewards, prop_dict, failed = reward.scoring((structs, ""), label=f"step_{step:04d}")
        m = step_metrics(rewards, prop_dict, failed)
        metrics_by_step[step] = m
        # per-sample dump: combined reward + raw and scaled value of every property
        for i, st in enumerate(structs):
            row = {"step": step, "index": i,
                   "formula": st.composition.reduced_formula,
                   "reward": float(rewards[i]), "failed": bool(failed[i])}
            for pc in reward.prop_cfg:
                raw = float(prop_dict[pc.name][i])
                row[f"{pc.name}__raw"] = raw
                row[f"{pc.name}__scaled"] = float(
                    reward._score_property(pc, np.array([raw]))[0])
            per_sample.append(row)
        extra = "  ".join(f"{n}_mean={m[f'{n} mean']:.3f}" for n in prop_names
                          if f"{n} mean" in m)
        print(f"  step {step:04d}: n={len(structs):3d} ok={int((~failed).sum()):3d}  "
              f"reward_mean={m['reward mean']:.4f}  {extra}", flush=True)

    write_metrics(run_dir, out_root, metrics_by_step)
    import csv as _csv
    if per_sample:
        cols = list(per_sample[0].keys())
        with (out_root / "rescored_per_sample.csv").open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(per_sample)
    print(f"Wrote rescored rewards + metrics.csv + rescored_per_sample.csv under {out_root}")


if __name__ == "__main__":
    main()
