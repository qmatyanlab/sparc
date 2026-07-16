#!/usr/bin/env python3
"""
Phase A: SLME-error attribution -- is the surrogate error gap-driven or
spectrum-driven?

For each QE-verified candidate, evaluate SLME(η) under the 4 corners of
{gap: ML|QE} x {eps2: ML|QE} using ONE consistent detailed-balance evaluator
(the pipeline's native backend, reused from rewards.calculators.tsenn_slme.calc),
on a common energy grid at 0.3 um / 300 K:

  ML,ML  = surrogate (reproduces the pipeline's reported slme_eta_percent)
  QE,ML  = swap in the DFT gap        -> error attributable to the GAP surrogate
  ML,QE  = swap in the DFT spectrum   -> error attributable to the SPECTRA surrogate
  QE,QE  = DFT reference

Aggregate: how much of |η(ML,ML) - η(QE,QE)| each single swap removes.

ML gap  = E3NNBandGap; ML eps2 = TSENN 'optimate' spectra surrogate.
QE gap  = HT_IPA database (indirect_gap; 0 for metallic runs);
QE eps2 = archived calc/{id}/results/epsi_{id}.dat (trace).

Usage: python attribution.py [--thickness-um 0.3]
"""

import os
import sys
import csv
import glob
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from pymatgen.core import Structure

from rewards.calculators.tsenn_slme.calc import (
    TSENNSLME, _kk_eps1_from_eps2, _absorption_coef_um_inv, _phi_bb_eV, CONST,
)

QE_PROJ = "/global/cfs/cdirs/m2663/angush/sparc_dft_verifications/slme_quantum_espresso"
CANDS = ("/global/cfs/cdirs/m2663/angush/sparc_clean/exp_res/"
         "tsenn_slme_03um_optimate_bgcenter13_eta08_e3nngap_mprime_lrdecay098_b128_55752527/"
         "deliverables_slme_candidates")
SPECTRA_CKPT = os.path.join(ROOT, "data/surrogates/TSENN_dielectric_spectra.torch")
SPECTRA_YAML = os.path.join(ROOT, "data/surrogates/TSENN_dielectric_spectra.yaml")
GAP_CKPT = os.path.join(ROOT, "data/surrogates/TSENN_bandgap.torch")
COMMON_E = np.linspace(0.0, 20.0, 401)


def eta_native(E, eps2, gap, slme, thickness_um):
    """Detailed-balance η (percent) from an eps2 spectrum + a gap, reusing the
    pipeline's native math. Metallic (gap<=0) -> 0."""
    if gap is None or not np.isfinite(gap) or gap <= 0:
        return 0.0
    eps2 = np.nan_to_num(np.asarray(eps2, float))
    eps1 = _kk_eps1_from_eps2(E, eps2[None, :])[0]
    alpha = _absorption_coef_um_inv(E, eps1[None, :], eps2[None, :])[0]
    A = 1.0 - np.exp(-2.0 * np.clip(alpha, 0.0, None) * float(thickness_um))
    phi_solar = np.interp(E, slme.solar_energies_ev, slme.solar_phi_ev, left=0, right=0)
    i_e = (np.interp(E, slme.solar_energies_ev, slme.solar_I_ev, left=0, right=0)
           if slme.solar_I_ev is not None else (E * CONST.q) * phi_solar)
    psolar = float(np.trapz(i_e, E))
    phi_bb = _phi_bb_eV(E, slme.temperature_k)
    beta = CONST.q / (CONST.kB * slme.temperature_k)
    m = (E >= gap)
    jsc = CONST.q * float(np.trapz(A[m] * phi_solar[m], E[m]))
    j0 = CONST.q * float(np.trapz(A[m] * phi_bb[m], E[m]))
    if jsc <= 0 or j0 <= 0 or psolar <= 0:
        return 0.0
    voc = (1.0 / beta) * np.log1p(jsc / j0)
    V = np.linspace(0, voc, 2000)
    pmax = float(np.max((jsc - j0 * (np.exp(beta * V) - 1.0)) * V))
    return 100.0 * pmax / psolar if pmax > 0 else 0.0


def load_qe(mat_id):
    """Return (status, qe_gap, qe_eps2_on_common) for a candidate."""
    jp = os.path.join(QE_PROJ, "HT_IPA", "database", f"{mat_id}.json")
    if not os.path.exists(jp):
        return None, None, None
    d = json.load(open(jp)).get("data", {})
    st = d.get("status")
    if st == "done":
        ig = d.get("indirect_gap")
        dat = os.path.join(QE_PROJ, "HT_IPA", "calc", mat_id, "results", f"epsi_{mat_id}.dat")
        if os.path.exists(dat):
            a = np.loadtxt(dat, comments="#")
            eps2 = a[:, 1:4].mean(axis=1)
            return st, ig, np.interp(COMMON_E, a[:, 0], eps2, left=0, right=0)
        return st, ig, None
    if isinstance(st, str) and st.startswith("failed"):
        # our QE metals fail the gap parse -> DFT says metallic
        return st, 0.0, None
    return st, None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--thickness-um", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    scfg = (yaml.safe_load(open(SPECTRA_YAML)) or {}).get("model", {}) if os.path.exists(SPECTRA_YAML) else {}
    slme = TSENNSLME(
        root_dir=os.path.join(HERE, "artifacts", "_attr_cache"), task="eta",
        model_path=SPECTRA_CKPT, device=args.device,
        thickness_um=args.thickness_um, temperature_k=300.0,
        eg_mode="e3nn", eg_model_path=GAP_CKPT,
        output_mode=scfg.get("output_mode", "trace"),
        out_dim=int(scfg.get("out_dim", 201)),
        em_dim=int(scfg.get("em_dim", 128)), mul=int(scfg.get("mul", 64)),
        lmax=int(scfg.get("lmax", 2)), layers=int(scfg.get("layers", 4)),
        num_neighbors=float(scfg.get("num_neighbors", 38.86)),
        scale_0e=float(scfg.get("scale_0e", 1.0)),
        energy_min=0.0, energy_max=20.0,
    )
    from rewards.calculators.e3nn_bandgap import E3NNBandGap
    gapcalc = E3NNBandGap(root_dir=os.path.join(HERE, "artifacts", "_attr_gap"),
                          model_path=GAP_CKPT, device=args.device)

    # ML summary CSV (for reference / expected slme_eta_percent)
    ml = {}
    with open(os.path.join(CANDS, "slme_verification_candidates_summary.csv")) as f:
        for r in csv.DictReader(f):
            ml[r["cif"]] = r

    rows = []
    for cif in sorted(glob.glob(os.path.join(CANDS, "cifs", "*.cif"))):
        base = os.path.basename(cif)
        mat_id = os.path.splitext(base)[0]
        st, qe_gap, qe_eps2 = load_qe(mat_id)
        if st is None or st not in ("done",) and not str(st).startswith("failed"):
            continue  # QE not finished for this one yet
        s = Structure.from_file(cif)
        ml_gap = float(gapcalc.predict([s])[0])
        E_ml, eps2_ml_arr, valid = slme.tsenn.predict_epsilon2_iso([s], 0.0, 20.0)
        ml_eps2 = np.interp(COMMON_E, E_ml, np.nan_to_num(eps2_ml_arr[0]), left=0, right=0)

        eta_mlml = eta_native(COMMON_E, ml_eps2, ml_gap, slme, args.thickness_um)
        eta_qeml = eta_native(COMMON_E, ml_eps2, qe_gap, slme, args.thickness_um)
        eta_mlqe = eta_native(COMMON_E, qe_eps2, ml_gap, slme, args.thickness_um) if qe_eps2 is not None else None
        eta_qeqe = eta_native(COMMON_E, qe_eps2, qe_gap, slme, args.thickness_um) if qe_eps2 is not None else (0.0 if qe_gap == 0.0 else None)
        rows.append(dict(id=mat_id, formula=ml.get(base, {}).get("formula", ""),
                         status=st, ml_gap=ml_gap, qe_gap=qe_gap,
                         eta_MLML=eta_mlml, eta_QEML=eta_qeml,
                         eta_MLQE=eta_mlqe, eta_QEQE=eta_qeqe,
                         ml_eta_csv=ml.get(base, {}).get("slme_eta_percent", "")))

    # report
    print(f"{'formula':12s} {'stat':6s} {'MLgap':>6} {'QEgap':>6} "
          f"{'MLML':>6} {'QEML':>6} {'MLQE':>6} {'QEQE':>6}  (gapΔ  specΔ)")
    gaps, specs = [], []
    for r in rows:
        def f(x): return f"{x:6.2f}" if isinstance(x, (int, float)) else f"{'-':>6}"
        gapd = (abs(r["eta_QEML"] - r["eta_MLML"]) if r["eta_QEML"] is not None else None)
        specd = (abs(r["eta_MLQE"] - r["eta_MLML"]) if r["eta_MLQE"] is not None else None)
        if r["eta_QEQE"] is not None and gapd is not None:
            gaps.append(gapd)
        if r["eta_QEQE"] is not None and specd is not None:
            specs.append(specd)
        print(f"{r['formula']:12s} {str(r['status'])[:6]:6s} {f(r['ml_gap'])} {f(r['qe_gap'])} "
              f"{f(r['eta_MLML'])} {f(r['eta_QEML'])} {f(r['eta_MLQE'])} {f(r['eta_QEQE'])}  "
              f"({'' if gapd is None else f'{gapd:.1f}'} {'' if specd is None else f'{specd:.1f}'})")
    if gaps and specs:
        print(f"\nMean |Δη| from GAP swap:      {np.mean(gaps):.2f} pts")
        print(f"Mean |Δη| from SPECTRUM swap: {np.mean(specs):.2f} pts")
        print(f"gap:spectrum attribution ratio ~ {np.mean(gaps)/max(np.mean(specs),1e-9):.1f} : 1")
    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "artifacts", "attribution.json"), "w"), indent=2)
    print(f"\n{len(rows)} candidates; wrote artifacts/attribution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
