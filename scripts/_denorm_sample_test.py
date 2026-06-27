#!/usr/bin/env python3
"""Seeded SymmCD sampling for the denormalization before/after test.
Samples N structures with a fixed seed, saves extxyz, prints density stats."""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

from models.suite.symmcd import SymmCDSampler, _load_symmcd_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="runtime/symmcd_pretrained/mp_20")
    a = ap.parse_args()

    model, _ = _load_symmcd_model(a.model)
    model = model.to("cuda")
    sampler = SymmCDSampler(model_path=a.model, generation_batch_size=64)

    set_seed(a.seed)
    _, structs = sampler.generate(model, batch_size=a.n, num_batches=1)

    vpa, mind, atoms = [], [], []
    for s in structs:
        n = len(s)
        vpa.append(s.volume / n)
        dm = s.distance_matrix; np.fill_diagonal(dm, np.inf)
        mind.append(dm.min() if n > 1 else np.nan)
        atoms.append(AseAtomsAdaptor.get_atoms(s))
    vpa, mind = np.array(vpa), np.array(mind)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write(a.out, atoms)
    print(f"WROTE {a.out}  N={len(structs)} (seed={a.seed})")
    print(f"  vol/atom median={np.median(vpa):.2f} A^3 (p10={np.percentile(vpa,10):.2f}, "
          f"p90={np.percentile(vpa,90):.2f})")
    print(f"  min_dist median={np.nanmedian(mind):.2f} A  "
          f"frac(<1.5A)={100*np.nanmean(mind<1.5):.0f}%  frac(<1.0A)={100*np.nanmean(mind<1.0):.0f}%")


if __name__ == "__main__":
    main()
