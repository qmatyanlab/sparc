#!/usr/bin/env python3
"""
Coverage control: how does the THIRD-PARTY pretrained ALIGNN band-gap model
(NIST mp_bandgap_hf, MP-PBE) do on the OOD generated structures, vs our base
E3NN and the rehearsal-fine-tuned E3NN? If ALIGNN also fails OOD, the coverage
gap is a general property of MP-trained gap surrogates, not e3nn-specific.

Evaluates all three on:
  (1) the OOD test set (91 dielectric_DFPT held-out families) vs VASP-PBE gap
  (2) the QE-verified SLME candidates vs QE-PBE gap (metals -> 0)

Run on a LOGIN NODE (ALIGNN downloads its weights from figshare; compute nodes
have no internet).  python active_learning/eval_alignn_ood.py
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
from rewards.calculators.alignn.calc import ALIGNN

DATA = os.path.join(HERE, "data")
ART = os.path.join(HERE, "artifacts")
SURR = os.path.join(ROOT, "data", "surrogates")
QE_PROJ = "/global/cfs/cdirs/m2663/angush/sparc_dft_verifications/slme_quantum_espresso"
CANDS = ("/global/cfs/cdirs/m2663/angush/sparc_clean/exp_res/"
         "tsenn_slme_03um_optimate_bgcenter13_eta08_e3nngap_mprime_lrdecay098_b128_55752527/"
         "deliverables_slme_candidates/cifs")


def _mae(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean(np.abs(p[m] - y[m]))) if m.any() else float("nan")


def qe_gap(mat_id):
    jp = os.path.join(QE_PROJ, "HT_IPA", "database", f"{mat_id}.json")
    if not os.path.exists(jp):
        return None
    d = json.load(open(jp)).get("data", {})
    st = d.get("status")
    if st == "done":
        return float(d.get("indirect_gap"))
    if isinstance(st, str) and st.startswith("failed"):
        return 0.0
    return None


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = E3NNBandGap(root_dir=os.path.join(ART, "_al_base"), device=dev)
    rehe = E3NNBandGap(root_dir=os.path.join(ART, "_al_reh"),
                       model_path=os.path.join(SURR, "TSENN_bandgap_ft_rehearsal.torch"), device=dev)
    alignn = ALIGNN(root_dir=os.path.join(ART, "_al_alignn"), task="band_gap", device=dev)

    def predict_all(structs):
        b = np.asarray(base.predict(structs), float)
        r = np.asarray(rehe.predict(structs), float)
        a = np.asarray(alignn.calc((structs, ""), label="al"), float)
        return b, r, a

    # (1) OOD test
    recs = json.load(open(os.path.join(DATA, "gap_labels_ood_test.json")))
    ood_s = [Structure.from_dict(r["structure"]) for r in recs]
    ood_y = np.array([float(r["gap"]) for r in recs])
    ob, orr, oa = predict_all(ood_s)
    print(f"=== OOD test (n={len(ood_y)}, VASP-PBE) MAE ===")
    print(f"  ALIGNN {_mae(oa,ood_y):.3f}   base-E3NN {_mae(ob,ood_y):.3f}   "
          f"rehearsal-E3NN {_mae(orr,ood_y):.3f}")

    # (2) QE candidates
    cand_s, cand_y, names = [], [], []
    for cif in sorted(glob.glob(os.path.join(CANDS, "*.cif"))):
        mid = os.path.splitext(os.path.basename(cif))[0]
        g = qe_gap(mid)
        if g is None:
            continue
        cand_s.append(Structure.from_file(cif)); cand_y.append(g); names.append(mid.split("_")[1])
    cand_y = np.array(cand_y)
    cb, cr, ca = predict_all(cand_s)
    print(f"\n=== QE candidates (n={len(cand_y)}, QE-PBE) MAE ===")
    print(f"  ALIGNN {_mae(ca,cand_y):.3f}   base-E3NN {_mae(cb,cand_y):.3f}   "
          f"rehearsal-E3NN {_mae(cr,cand_y):.3f}")
    print(f"\n{'formula':12s} {'QE':>6} {'ALIGNN':>7} {'baseE3':>7} {'reheE3':>7}")
    for n, y, a, b, r in zip(names, cand_y, ca, cb, cr):
        print(f"{n:12s} {y:6.2f} {a:7.2f} {b:7.2f} {r:7.2f}")

    np.savez(os.path.join(ART, "alignn_ood.npz"), ood_y=ood_y, ood_alignn=oa,
             ood_base=ob, ood_reh=orr, cand_y=cand_y, cand_alignn=ca,
             cand_base=cb, cand_reh=cr)
    print("\nsaved artifacts/alignn_ood.npz")


if __name__ == "__main__":
    main()
