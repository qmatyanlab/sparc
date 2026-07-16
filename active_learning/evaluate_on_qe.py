#!/usr/bin/env python3
"""
Evaluate the base vs fine-tuned (ensemble) band-gap surrogate against the QE
DFT verification set -- the real OOD test: did fine-tuning move the surrogate's
gaps toward DFT, and does the ensemble std flag the reward-hacked/metallic ones?

Prints base_gap, ft_ensemble mean±std, QE_gap per candidate, and MAE(base vs QE)
vs MAE(ft vs QE). QE metals are treated as gap=0.
"""
import os
import sys
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pymatgen.core import Structure
from rewards.calculators.e3nn_bandgap import E3NNBandGap
from e3nn_bandgap_ensemble import E3NNBandGapEnsemble  # noqa: E402  (same dir)

QE_PROJ = "/global/cfs/cdirs/m2663/angush/sparc_dft_verifications/slme_quantum_espresso"
CANDS = ("/global/cfs/cdirs/m2663/angush/sparc_clean/exp_res/"
         "tsenn_slme_03um_optimate_bgcenter13_eta08_e3nngap_mprime_lrdecay098_b128_55752527/"
         "deliverables_slme_candidates/cifs")


def qe_gap(mat_id):
    jp = os.path.join(QE_PROJ, "HT_IPA", "database", f"{mat_id}.json")
    if not os.path.exists(jp):
        return None
    d = json.load(open(jp)).get("data", {})
    st = d.get("status")
    if st == "done":
        return float(d.get("indirect_gap"))
    if isinstance(st, str) and st.startswith("failed"):
        return 0.0  # metallic in QE-PBE
    return None


def main():
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    base = E3NNBandGap(root_dir=os.path.join(HERE, "artifacts", "_evbase"), device=dev)
    ens = E3NNBandGapEnsemble(root_dir=os.path.join(HERE, "artifacts", "_evens"), device=dev)
    print(f"ensemble members: {len(ens.members)}")

    rows = []
    for cif in sorted(glob.glob(os.path.join(CANDS, "*.cif"))):
        mid = os.path.splitext(os.path.basename(cif))[0]
        qg = qe_gap(mid)
        if qg is None:
            continue
        s = Structure.from_file(cif)
        bg = float(base.predict([s])[0])
        m, sd = ens.predict_with_uncertainty([s])
        rows.append((mid.split("_")[1], bg, float(m[0]), float(sd[0]), qg))

    print(f"{'formula':12s} {'base':>6} {'ft_mean':>8} {'ft_std':>7} {'QE':>6}  flag")
    be, fe = [], []
    for f, bg, m, sd, qg in rows:
        be.append(abs(bg - qg)); fe.append(abs(m - qg))
        flag = "HIGH-UNC->DFT" if sd > 0.3 else ""
        print(f"{f:12s} {bg:6.2f} {m:8.2f} {sd:7.2f} {qg:6.2f}  {flag}")
    if rows:
        print(f"\nMAE vs QE  base={np.mean(be):.3f}  finetuned={np.mean(fe):.3f} eV  (n={len(rows)})")


if __name__ == "__main__":
    main()
