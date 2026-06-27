#!/usr/bin/env python3
"""Select DFPT-verification candidates from the RL candidate CSVs.

Picks structures that are simultaneously: high reward, FINITE band gap (so a DFPT
dielectric/phonon calculation is meaningful -- metals are excluded), and predicted
LOW Ehull (stable / synthesisable).  Reads the per-run candidate CSVs written by
scripts/plot_candidates_gap_score_ehull.py (columns: step,index,formula,score,
ehull,band_gap_ev) and pulls the matching structure out of
  {run_dir}/samples/step_{step:04d}_eval.extxyz  at position `index`
(the exact source the CSV was built from), writing a conventional-cell CIF per pick.

Filters (defaults): GAP_LO <= gap <= GAP_HI, ehull <= EHULL_MAX (>= EHULL_MIN),
dedup by reduced formula keeping the highest-score occurrence, then rank by score.

Example:
  .venv/bin/python scripts/select_dfpt_candidates.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp_res"
CP = EXP / "_candidate_plots"
OUT = CP / "dfpt_candidates_cif"

# physically sane windows (surrogate outliers like gap=-3851 or ehull=114 are excluded)
GAP_LO, GAP_HI = 0.5, 6.0       # eV: finite (non-metal), not an absurd surrogate value
EHULL_MIN, EHULL_MAX = -0.05, 0.05   # eV/atom: stable (allow slightly-below-hull novel)
TOP_N = 12                      # distinct formulas to keep per run

RUNS = {
    "slme_54346667": {
        "dir": EXP / "tsenn_slme_03um_optimate_bg02_eta08_adaptive_v1_54346667",
        "score": "SLME eta (%)",
    },
    "staticdie_54584243": {
        "dir": EXP / "tsenn_static_dielectric_layered_uniaxial_symmcd_v3_uniform_anchor_54584243",
        "score": "dielectric anisotropy reward",
    },
}


def extract(run_dir: Path, step: int, index: int):
    ext = run_dir / "samples" / f"step_{step:04d}_eval.extxyz"
    atoms = ase_read(ext, index=":")
    if not isinstance(atoms, list):
        atoms = [atoms]
    return AseAtomsAdaptor.get_structure(atoms[int(index)])


def select(key: str, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(CP / f"{key}_candidates.csv")
    m = (df.band_gap_ev.between(GAP_LO, GAP_HI)
         & df.ehull.between(EHULL_MIN, EHULL_MAX))
    sub = df[m].copy()
    # dedup by formula -> keep the highest-score occurrence; record persistence
    counts = sub.groupby("formula").size().rename("n_occ")
    best = (sub.sort_values("score", ascending=False)
               .drop_duplicates("formula")
               .merge(counts, on="formula"))
    best = best.sort_values("score", ascending=False).head(TOP_N).reset_index(drop=True)

    odir = OUT / key
    odir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, r in best.iterrows():
        st = extract(cfg["dir"], int(r.step), int(r["index"]))
        try:
            sga = SpacegroupAnalyzer(st, symprec=0.1)
            conv = sga.get_conventional_standard_structure()
            sg_sym, sg_no = sga.get_space_group_symbol(), sga.get_space_group_number()
        except Exception:  # noqa: BLE001
            conv, sg_sym, sg_no = st, "P1", 1
        name = f"r{rank+1:02d}_{r.formula}_SG{sg_no}_step{int(r.step)}_i{int(r['index'])}.cif"
        conv.to(filename=str(odir / name), fmt="cif")
        rows.append({"rank": rank + 1, "formula": r.formula, "spacegroup": sg_sym,
                     "sg_no": sg_no, "score": round(float(r.score), 4),
                     "band_gap_ev": round(float(r.band_gap_ev), 3),
                     "ehull": round(float(r.ehull), 4), "n_occ": int(r.n_occ),
                     "step": int(r.step), "index": int(r["index"]), "cif": name})
    out = pd.DataFrame(rows)
    out.to_csv(odir / "_summary.csv", index=False)
    print(f"\n{'='*78}\n{key}   (score = {cfg['score']})\n{'='*78}")
    print(f"  filter: {GAP_LO}<=gap<={GAP_HI} eV, {EHULL_MIN}<=ehull<={EHULL_MAX} eV/atom"
          f"  ->  {m.sum()} rows / {sub.formula.nunique()} formulas; top {len(out)} below")
    print(out.to_string(index=False))
    print(f"  CIFs -> {odir}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for key, cfg in RUNS.items():
        select(key, cfg)


if __name__ == "__main__":
    main()
