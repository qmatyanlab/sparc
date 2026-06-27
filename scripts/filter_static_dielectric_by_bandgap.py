#!/usr/bin/env python3
"""Filter a TSENNStaticDielectric run's samples by predicted band gap.

The static-dielectric reward maximizes epsilon, which diverges as Eg -> 0, so the
best-reward samples drift toward (near-)metallic structures that DFPT cannot treat.
This script scores the per-step *eval* pool (the evaluated top-k that the reward txt /
dielectric-tensor npz are 1:1 aligned with), predicts the band gap with the project's
own E3NN band-gap model (the sibling of the dielectric model, optuna_bandgap_trial_2),
and writes a band-gap-annotated table plus a non-metallic subset (CSV + CIFs + extxyz)
suitable for DFPT.

The band-gap net MUST be loaded with the same Network class it was trained with
(data.dielectric.utils.utils_model_scalar via scripts.plot_band_gap_splits.NetWrapper);
the TSENN Network is a different architecture that silently loads the same state dict
but predicts a ~0.69 eV floor. Validated against the cached test split:
MAE 0.19 eV, R^2 0.91, and ~89% of true metals predicted at Eg <= 0.1 eV.

Example:
  .venv/bin/python scripts/filter_static_dielectric_by_bandgap.py \
      exp_res/tsenn_static_dielectric_layered_uniaxial_symmcd_v3_uniform_anchor_54584243 \
      --top-k 500 --min-gap 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "data/dielectric/optuna_bandgap_trial_2_gpu0_best.torch"
DEFAULT_CONFIG = ROOT / "data/dielectric/e3_band_gap_inference_config.yaml"
ONEHOT_CACHE = ROOT / "rewards/tsenn_cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <run_dir>/deliverables_bandgap_filtered")
    p.add_argument("--reward-name", default=None,
                   help="reward subdir name; auto-detected if omitted")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--min-gap", type=float, default=0.5,
                   help="predicted band gap (eV) cutoff for the non-metallic subset")
    p.add_argument("--top-k", type=int, default=None,
                   help="only score the top-K eval samples by dielectric reward "
                        "(default: score all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bandgap-batch-size", type=int, default=256)
    p.add_argument("--no-cifs", dest="write_cifs", action="store_false")
    p.set_defaults(write_cifs=True)
    return p.parse_args()


def detect_reward_name(rewards: Path) -> str:
    cands = [d.name for d in rewards.iterdir()
             if d.is_dir() and list(d.glob("step_*.txt"))]
    if not cands:
        raise FileNotFoundError(f"No reward subdir with step_*.txt under {rewards}")
    return sorted(cands, key=len, reverse=True)[0]


def fline(path: Path) -> list[float]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(float(line.strip()))
        except ValueError:
            out.append(float("nan"))
    return out


def load_eval_pool(run_dir: Path, reward_name: str) -> list[dict[str, Any]]:
    samples = run_dir / "samples"
    rdir = run_dir / "rewards" / reward_name
    rows: list[dict[str, Any]] = []
    for ext in sorted(samples.glob("step_*_eval.extxyz")):
        step = int(ext.stem.split("_")[1])
        if not ext.stat().st_size:
            continue
        atoms = ase_read(ext, index=":")
        if not isinstance(atoms, list):
            atoms = [atoms]
        reward = fline(rdir / f"step_{step:04d}.txt")
        npz_path = rdir / f"step_{step:04d}_tensor.npz"
        tens = mask = None
        if npz_path.exists():
            d = np.load(npz_path)
            tens, mask = d["tensor"], d.get("valid_mask")
        n = min(len(atoms), len(reward) if reward else len(atoms))
        for i in range(n):
            r = reward[i] if reward else float("nan")
            # predicted static tensor is strongly anisotropic (and can be sign-flipped),
            # so report the diagonal components rather than a cancelling trace/3 average.
            exx = eyy = ezz = float("nan")
            if tens is not None and i < len(tens) and (mask is None or bool(mask[i])):
                exx, eyy, ezz = (float(tens[i][0, 0]),
                                 float(tens[i][1, 1]),
                                 float(tens[i][2, 2]))
            struct = AseAtomsAdaptor.get_structure(cast(Any, atoms[i]))
            rows.append({
                "step": step,
                "index": i,
                "formula": struct.composition.reduced_formula,
                "nsites": len(struct),
                "dielectric_reward": float(r),
                "eps_xx": exx,
                "eps_yy": eyy,
                "eps_zz": ezz,
                "_struct": struct,
                "_atoms": atoms[i],
            })
    return rows


def load_bandgap_model(config_path: Path, model_path: Path, device: str):
    import torch
    import yaml
    from scripts.plot_band_gap_splits import NetWrapper  # official Network class

    cfg = (yaml.safe_load(config_path.read_text()) or {}).get("model", {})
    em = int(cfg.get("em_dim", 128))
    r_max = float(cfg.get("r_max", 6.0))
    model = NetWrapper(
        in_dim=118, em_dim=em,
        irreps_in=f"{em}x0e", irreps_out="1x0e", irreps_node_attr=f"{em}x0e",
        layers=int(cfg.get("layers", 4)), mul=int(cfg.get("mul", 16)),
        lmax=int(cfg.get("lmax", 2)), max_radius=r_max,
        num_neighbors=float(cfg.get("num_neighbors", 55.328226741470544)),
        reduce_output=True, dropout_prob=0.0, use_batch_norm=False,
    ).to(device, dtype=torch.float64)
    model.load_state_dict(torch.load(model_path, map_location=device)["state"])
    model.eval()
    return model, r_max


def predict_band_gaps(structs: list, model, r_max: float, device: str,
                      batch_size: int) -> np.ndarray:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from rewards.calculators.tsenn.calc import build_graph, load_or_build_onehot

    t_oh, m_oh, d_oh, r_oh, enc = load_or_build_onehot(str(ONEHOT_CACHE),
                                                       dtype=torch.float64)
    out = np.full(len(structs), np.nan, dtype=float)
    data_list, valid = [], []
    for i, s in enumerate(structs):
        try:
            g = build_graph(s, t_oh, m_oh, d_oh, r_oh, enc, r_max, dtype=torch.float64)
        except Exception:  # noqa: BLE001
            continue
        data_list.append(Data(
            pos=g.pos.double(), x=g.x.double(), z=g.z.double(),
            edge_index=g.edge_index.long(), edge_vec=g.edge_vec.double()))
        valid.append(i)
    if not data_list:
        return out
    preds = []
    with torch.no_grad():
        for batch in DataLoader(data_list, batch_size=batch_size, shuffle=False):
            preds.append(model(batch.to(device)).view(-1).cpu().numpy())
    for i, p in zip(valid, np.concatenate(preds)):
        out[i] = float(p)
    return out


def main() -> None:
    import pandas as pd

    args = parse_args()
    run_dir = args.run_dir.resolve()
    out = (args.output_dir or run_dir / "deliverables_bandgap_filtered").resolve()
    out.mkdir(parents=True, exist_ok=True)

    reward_name = args.reward_name or detect_reward_name(run_dir / "rewards")
    rows = load_eval_pool(run_dir, reward_name)
    df = pd.DataFrame(rows).sort_values("dielectric_reward",
                                        ascending=False).reset_index(drop=True)
    print(f"Loaded {len(df)} eval samples from {run_dir.name} (reward='{reward_name}')")

    scored = (df.head(args.top_k) if args.top_k else df).reset_index(drop=True)
    print(f"Predicting E3NN band gap ({args.model.name}) for {len(scored)} samples"
          + (f" (top-{args.top_k} by reward)" if args.top_k else "") + " ...")

    model, r_max = load_bandgap_model(args.config, args.model, args.device)
    scored["band_gap_ev"] = predict_band_gaps(
        list(scored["_struct"]), model, r_max, args.device, args.bandgap_batch_size)

    cols = ["step", "index", "formula", "nsites", "dielectric_reward",
            "eps_xx", "eps_yy", "eps_zz", "band_gap_ev"]
    scored[cols].to_csv(out / "scored_by_reward_with_bandgap.csv", index=False)

    keep = scored[scored["band_gap_ev"] >= args.min_gap].sort_values(
        "dielectric_reward", ascending=False).reset_index(drop=True)
    keep[cols].to_csv(out / f"non_metallic_gap_ge_{args.min_gap:g}.csv", index=False)

    g = scored["band_gap_ev"].to_numpy()
    g = g[np.isfinite(g)]
    pct = {p: float(np.percentile(g, p)) for p in (10, 25, 50, 75, 90)}
    print(f"\n=== predicted band gap ({args.model.name}) over scored pool ===")
    print(f"  metallic-ish (Eg < {args.min_gap:g} eV): "
          f"{int((g < args.min_gap).sum())}/{len(g)} "
          f"({100*(g < args.min_gap).mean():.1f}%)")
    print(f"  kept Eg >= {args.min_gap:g} eV: {len(keep)}")
    print("  percentiles eV: " + ", ".join(f"p{p}={v:.2f}" for p, v in pct.items()))

    if len(keep):
        if args.write_cifs:
            import shutil
            cif_dir = out / "non_metallic_cifs"
            if cif_dir.exists():
                shutil.rmtree(cif_dir)  # avoid mixing stale CIFs across runs
            cif_dir.mkdir(parents=True)
            for rank, (_, r) in enumerate(keep.iterrows(), 1):
                fn = f"{rank:03d}_{r['formula']}_Eg{r['band_gap_ev']:.2f}_step{r['step']}.cif"
                (cif_dir / fn).write_text(r["_struct"].to(fmt="cif"))
        from ase.io import write as ase_write
        ase_write(out / "non_metallic.extxyz",
                  [r["_atoms"] for _, r in keep.iterrows()], format="extxyz")
        print("\n  top non-metallic by reward:")
        for _, r in keep.head(10).iterrows():
            print(f"    {r['formula']:<10} Eg={r['band_gap_ev']:.2f}  "
                  f"eps_diag=[{r['eps_xx']:.1f},{r['eps_yy']:.1f},{r['eps_zz']:.1f}]  "
                  f"reward={r['dielectric_reward']:.3f}  "
                  f"(step{r['step']} idx{r['index']})")
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
