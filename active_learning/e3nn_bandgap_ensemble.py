#!/usr/bin/env python3
"""
Deep-ensemble wrapper around the fine-tuned E3NN band-gap surrogate.

Each fine-tune seed (finetune_bandgap.py --seed N -> TSENN_bandgap_ft_seedN.torch)
is a member; the ensemble returns predictive **mean** (the improved gap) and
**std** (epistemic uncertainty / OOD signal). High std flags candidates the
surrogate shouldn't be trusted on -> route them to DFT (the active-learning gate).

`predict` returns the mean gap (drop-in with E3NNBandGap so it can feed the SLME
reward via eg_model path); `predict_with_uncertainty` returns (mean, std).

Usage (CLI, prints mean±std for CIFs):
    python e3nn_bandgap_ensemble.py struct1.cif struct2.cif ...
"""

import os
import sys
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rewards.calculators.e3nn_bandgap import E3NNBandGap

DEFAULT_GLOB = os.path.join(ROOT, "data", "surrogates", "TSENN_bandgap_ft_seed*.torch")


class E3NNBandGapEnsemble:
    def __init__(self, root_dir, ckpt_glob=DEFAULT_GLOB, device=None,
                 include_base=False, **kw):
        paths = sorted(glob.glob(ckpt_glob))
        if include_base:
            paths = [os.path.join(ROOT, "data/surrogates/TSENN_bandgap.torch")] + paths
        if not paths:
            raise FileNotFoundError(f"no ensemble checkpoints match {ckpt_glob}")
        self.members = [
            E3NNBandGap(root_dir=os.path.join(root_dir, f"_m{i}"),
                        model_path=p, device=device, **kw)
            for i, p in enumerate(paths)
        ]
        self.paths = paths

    def predict_with_uncertainty(self, struc_list):
        preds = np.stack([m.predict(struc_list) for m in self.members], axis=0)
        return np.nanmean(preds, axis=0), np.nanstd(preds, axis=0)

    def predict(self, struc_list):
        return self.predict_with_uncertainty(struc_list)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cifs", nargs="+")
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    from pymatgen.core import Structure
    ens = E3NNBandGapEnsemble(root_dir="/tmp/albg_ens", ckpt_glob=args.glob,
                              device=args.device)
    print(f"ensemble: {len(ens.members)} members")
    structs = [Structure.from_file(c) for c in args.cifs]
    mean, std = ens.predict_with_uncertainty(structs)
    for c, m, s in zip(args.cifs, mean, std):
        flag = "  <-- HIGH-UNCERTAINTY (send to DFT)" if s > 0.3 else ""
        print(f"  {os.path.basename(c):40s} gap = {m:5.3f} +/- {s:5.3f} eV{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
