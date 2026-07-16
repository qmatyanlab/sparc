#!/usr/bin/env python3
"""
Measure catastrophic forgetting: evaluate the base vs fine-tuned (soup) band-gap
surrogate on the ORIGINAL Materials Project held-out TEST split, to see whether
fine-tuning on the OOD generated labels degraded in-distribution accuracy.

MP cache (sparc): dict with df + idx_test + target_col. We build graphs for the
test rows and compare MAE(base) vs MAE(soup) on MP. Reference base MP MAE ~0.20.
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pymatgen.core import Structure

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from rewards.calculators.e3nn_bandgap import E3NNBandGap
from rewards.calculators.tsenn.calc import build_graph

CACHE = "/global/cfs/cdirs/m2663/angush/sparc/data/dielectric/cached_band_gap_preprocessed_data.pt"


def _to_structure(v):
    if isinstance(v, Structure):
        return v
    if isinstance(v, dict):
        return Structure.from_dict(v)
    if isinstance(v, str):
        return Structure.from_str(v, fmt="json") if v.lstrip().startswith("{") else Structure.from_str(v, fmt="cif")
    raise TypeError(type(v))


def _find_struct_col(df):
    for c in df.columns:
        v = df.iloc[0][c]
        if isinstance(v, Structure):
            return c
        if isinstance(v, dict) and v.get("@class") == "Structure":
            return c
    for c in ("structure", "final_structure", "atoms"):
        if c in df.columns:
            return c
    raise RuntimeError(f"no structure column found in {list(df.columns)}")


def _eval(model, data_list, device, bs=256):
    model.eval(); preds = []
    with torch.no_grad():
        for b in DataLoader(data_list, batch_size=bs, shuffle=False):
            preds.append(model(b.to(device)).view(-1).cpu().numpy())
    return np.clip(np.concatenate(preds), 0.0, None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--max-n", type=int, default=4000, help="cap MP-test rows for speed")
    ap.add_argument("--soup", default=os.path.join(ROOT, "data/surrogates/TSENN_bandgap_ft_soup.torch"))
    args = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading MP cache ...", flush=True)
    c = torch.load(args.cache, map_location="cpu")
    df = c["df"]; idx = list(c["idx_test"]); target = c["target_col"]
    print(f"MP test rows: {len(idx)}  target={target}  cols={list(df.columns)[:12]}",
          flush=True)
    test = df.iloc[idx]

    base = E3NNBandGap(root_dir="/tmp/fg_base", device=dev)
    soup = E3NNBandGap(root_dir="/tmp/fg_soup", model_path=args.soup, device=dev)

    # 'structure' column holds ASE Atoms -> pymatgen -> build_graph
    data_list, y = [], []
    for atoms, gap in zip(test["structure"].values, test[target].values):
        try:
            s = AseAtomsAdaptor.get_structure(atoms) if isinstance(atoms, Atoms) \
                else _to_structure(atoms)
            g = build_graph(s, base.type_onehot, base.mass_onehot, base.dipole_onehot,
                            base.radius_onehot, base.type_encoding, base.r_max,
                            dtype=base.dtype)
            data_list.append(Data(pos=g.pos.double(), x=g.x.double(), z=g.z.double(),
                                  edge_index=g.edge_index.long(),
                                  edge_vec=g.edge_vec.double()))
            y.append(float(gap))
        except Exception:
            continue
    y = np.array(y)
    print(f"MP-test graphs ready: {len(data_list)} / {len(idx)}", flush=True)

    def mae(p): return float(np.mean(np.abs(p - y)))
    pb = _eval(base.model, data_list, dev)
    ps = _eval(soup.model, data_list, dev)
    print(f"\nMP-test MAE   base={mae(pb):.3f} eV   soup(fine-tuned)={mae(ps):.3f} eV   "
          f"(n={len(y)})")
    print(f"MP-test degradation from fine-tuning: {mae(ps)-mae(pb):+.3f} eV")
    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    np.savez(os.path.join(HERE, "artifacts", "mp_test_parity.npz"),
             y=y, base=pb, soup=ps)
    print("saved artifacts/mp_test_parity.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
