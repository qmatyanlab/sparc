#!/usr/bin/env python3
"""Select LAYERED uniaxial-dielectric candidates from the static-dielectric RL run
for DFPT verification.

Unlike select_dfpt_candidates.py (which ranks on the scalar reward only), this uses the
predicted dielectric TENSOR stored in
  {run}/deliverables_bandgap_filtered/scored_by_reward_with_bandgap.csv
(eps_xx,eps_yy,eps_zz + E3NN band gap; the E3NN optuna_bandgap model loaded via the
validated NetWrapper -- NOT the TSENN band-gap head, which floors to ln2~0.693 eV).
Ehull is merged from the candidate CSV.  A candidate must be:

  * PHYSICAL      : all eps components > 1 and < 200 (drops the negative-eps reward
                    hacks like BiIO3 -13/-13/+28 that still scored reward=1.0)
  * UNIAXIAL      : eps_xx ~= eps_yy   (in-plane isotropic)
  * LAYERED       : eps_in = (eps_xx+eps_yy)/2  >  eps_out = eps_zz  (in-plane more
                    polarisable than across the stacking axis -- the vdW-layer signature)
  * FINITE gap    : 0.5 <= Eg <= 6 eV (DFPT dielectric/phonons need an insulator)
  * STABLE        : ehull <= EHULL_MAX

Ranked by dielectric_reward (meaningful once the unphysical hacks are removed); the
conventional-cell CIF + a tensor summary are written per pick.

Example:
  .venv/bin/python scripts/select_layered_dielectric_candidates.py
  .venv/bin/python scripts/select_layered_dielectric_candidates.py --ehull-max 0.10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "exp_res/tsenn_static_dielectric_layered_uniaxial_symmcd_v3_uniform_anchor_54584243"
CAND = ROOT / "exp_res/_candidate_plots/staticdie_54584243_candidates.csv"
OUT = ROOT / "exp_res/_candidate_plots/layered_dielectric_cif"

GAP_LO, GAP_HI = 0.5, 6.0
DMIN_OK = 1.4   # A; flag structures with closer contacts as unphysical

# Open-shell (partially-filled d) transition metals: in chalcogenides these very often
# stay METALLIC even when the surrogate predicts a small gap (e.g. NbSe2/NbS2 are metals,
# NbMoSe4 = metallic-NbSe2 / semiconducting-MoSe2 layers). A DFPT dielectric needs a TRUE
# insulator, so flag these as HIGH metallicity risk; main-group + closed-shell d10 (Cu+,
# Zn2+, Cd2+, Hg2+, Ag+, Au+, Ga3+, In3+) are robustly semiconducting -> "low" risk.
OPEN_D = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Y", "Zr", "Nb", "Mo", "Tc",
          "Ru", "Rh", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Pd"}


def metallicity_risk(formula: str) -> str:
    from pymatgen.core import Composition
    bad = sorted({e.symbol for e in Composition(formula).elements} & OPEN_D)
    return "HIGH:" + ",".join(bad) if bad else "low"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ehull-max", type=float, default=0.10)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--low-risk-only", action="store_true",
                    help="drop candidates containing an open-d transition metal "
                         "(metallicity-prone); writes to a low_metal_risk/ subdir")
    args = ap.parse_args()
    out_dir = OUT / "low_metal_risk" if args.low_risk_only else OUT

    d = pd.read_csv(RUN / "deliverables_bandgap_filtered/scored_by_reward_with_bandgap.csv")
    eh = pd.read_csv(CAND)[["step", "index", "ehull"]]
    df = d.merge(eh, on=["step", "index"], how="left")

    eps = df[["eps_xx", "eps_yy", "eps_zz"]]
    df["eps_in"] = 0.5 * (df.eps_xx + df.eps_yy)
    df["eps_out"] = df.eps_zz
    df["aniso"] = df.eps_in / df.eps_out
    m = (eps.min(axis=1) > 1.0) & (eps.max(axis=1) < 200) \
        & ((df.eps_xx - df.eps_yy).abs() / df.eps_in < 0.15) \
        & (df.eps_in > 1.1 * df.eps_out) \
        & df.band_gap_ev.between(GAP_LO, GAP_HI) \
        & (df.ehull <= args.ehull_max)
    df["metal_risk"] = df.formula.map(metallicity_risk)
    if args.low_risk_only:
        m = m & (df.metal_risk == "low")
    best = (df[m].sort_values("dielectric_reward", ascending=False)
                 .drop_duplicates("formula")
                 .head(args.top_n).reset_index(drop=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, r in best.iterrows():
        atoms = ase_read(RUN / "samples" / f"step_{int(r.step):04d}_eval.extxyz", index=":")
        if not isinstance(atoms, list):
            atoms = [atoms]
        st = AseAtomsAdaptor.get_structure(atoms[int(r["index"])])
        dm = st.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        dmin = float(dm.min())
        try:
            sga = SpacegroupAnalyzer(st, symprec=0.1)
            conv = sga.get_conventional_standard_structure()
            sg_sym, sg_no = sga.get_space_group_symbol(), sga.get_space_group_number()
        except Exception:  # noqa: BLE001
            conv, sg_sym, sg_no = st, "P1", 1
        name = f"r{rank+1:02d}_{r.formula}_SG{sg_no}_step{int(r.step)}_i{int(r['index'])}.cif"
        conv.to(filename=str(out_dir / name), fmt="cif")
        rows.append({"rank": rank + 1, "formula": r.formula, "spacegroup": sg_sym, "sg_no": sg_no,
                     "metal_risk": r.metal_risk,
                     "reward": round(float(r.dielectric_reward), 3),
                     "eps_in": round(float(r.eps_in), 1), "eps_out": round(float(r.eps_out), 1),
                     "anisotropy": round(float(r.aniso), 2),
                     "band_gap_ev": round(float(r.band_gap_ev), 3),
                     "ehull": round(float(r.ehull), 4), "nsites": int(r.nsites),
                     "dmin_A": round(dmin, 2), "ok_geom": dmin >= DMIN_OK,
                     "step": int(r.step), "index": int(r["index"]), "cif": name})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"layered uniaxial dielectric candidates (ehull<={args.ehull_max}); "
          f"eps_in=in-plane (xx=yy), eps_out=along c (zz)\n")
    print(out.drop(columns=["step", "index"]).to_string(index=False))
    print(f"\nCIFs -> {out_dir}")


if __name__ == "__main__":
    main()
