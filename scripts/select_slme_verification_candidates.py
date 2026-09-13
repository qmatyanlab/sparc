#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
from select_good_structures import load_samples, score  # noqa: E402

# open-shell partially-filled-d TMs (metallicity-prone) — from select_layered_dielectric_candidates.py
OPEN_D = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Y", "Zr", "Nb", "Mo", "Tc",
          "Ru", "Rh", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Pd"}
TOXIC = {"Pb", "Cd", "Hg", "As", "Tl"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--gap-lo", type=float, default=1.0)
    p.add_argument("--gap-hi", type=float, default=1.8)
    p.add_argument("--eta-min", type=float, default=0.25, help="min SLME eta (fraction)")
    p.add_argument("--ehull-max", type=float, default=0.1)
    p.add_argument("--n-plausible", type=int, default=12)
    p.add_argument("--n-fluoride", type=int, default=4)
    p.add_argument("--include-formulas", nargs="*", default=[],
                   help="force-include these formulas (best-scoring instance), class=requested")
    return p.parse_args()


def elem_flags(formula: str):
    els = {str(e) for e in Composition(formula).elements}
    return ("F" in els), bool(els & OPEN_D), bool(els & TOXIC)


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    out = (args.output_dir or run / "deliverables_slme_candidates").resolve()
    (out / "cifs").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(load_samples(run))
    print(f"loaded {len(df)} finite samples from {run.name}")

    # target region: stable + solar gap window + high eta
    m = ((df["ehull_ev_per_atom"] <= args.ehull_max)
         & (df["band_gap_ev"] >= args.gap_lo) & (df["band_gap_ev"] <= args.gap_hi)
         & (df["slme_eta_frac"] >= args.eta_min))
    cand = df[m].copy()
    eta_s, bg_s, eh_s, sc = score(cand, "solar_window")
    cand["eta_score"], cand["band_gap_reward"], cand["ehull_score"], cand["selection_score"] = eta_s, bg_s, eh_s, sc
    fl = cand["formula"].map(elem_flags)
    cand["fluoride"] = fl.map(lambda t: t[0])
    cand["open_d"] = fl.map(lambda t: t[1])
    cand["toxic"] = fl.map(lambda t: t[2])
    # dedup by formula (keep highest score)
    cand = cand.sort_values("selection_score", ascending=False).drop_duplicates("formula").reset_index(drop=True)
    print(f"target region (gap {args.gap_lo}-{args.gap_hi}, eta>={args.eta_min}, ehull<={args.ehull_max}): "
          f"{len(cand)} unique formulas | fluoride {int(cand.fluoride.sum())} | open_d {int(cand.open_d.sum())} | toxic {int(cand.toxic.sum())}")

    plausible = cand[~cand["fluoride"] & ~cand["open_d"]].head(args.n_plausible).copy()
    plausible["class"] = "plausible"
    fluoride = cand[cand["fluoride"]].head(args.n_fluoride).copy()
    fluoride["class"] = "fluoride"
    sel = pd.concat([plausible, fluoride]).reset_index(drop=True)

    if args.include_formulas:
        req = {Composition(f).reduced_formula for f in args.include_formulas}
        extra = df[df["formula"].isin(req)].copy()
        if len(extra):
            e_eta, e_bg, e_eh, e_sc = score(extra, "solar_window")
            extra["eta_score"], extra["band_gap_reward"], extra["ehull_score"], extra["selection_score"] = e_eta, e_bg, e_eh, e_sc
            xf = extra["formula"].map(elem_flags)
            extra["fluoride"] = xf.map(lambda t: t[0])
            extra["open_d"] = xf.map(lambda t: t[1])
            extra["toxic"] = xf.map(lambda t: t[2])
            extra = extra.sort_values("selection_score", ascending=False).drop_duplicates("formula")
            extra = extra[~extra["formula"].isin(sel["formula"])].copy()
            extra["class"] = "requested"
            sel = pd.concat([sel, extra]).reset_index(drop=True)
        missing = req - set(df["formula"])
        if missing:
            print(f"WARNING requested formulas NOT found in samples: {sorted(missing)}")

    recs = []
    for rank, (_, r) in enumerate(sel.iterrows(), 1):
        st = AseAtomsAdaptor.get_structure(r["_atoms"])
        try:
            sga = SpacegroupAnalyzer(st, symprec=0.01)
            conv = sga.get_conventional_standard_structure()
            sg_no = int(sga.get_space_group_number())
        except Exception:  # noqa: BLE001
            conv, sg_no = st, r["spacegroup"]
        cif = f"r{rank:02d}_{r['formula']}_SG{sg_no}_step{r['step']}_i{r['index']}.cif"
        (out / "cifs" / cif).write_text(conv.to(fmt="cif"))
        recs.append({
            "rank": rank, "formula": r["formula"], "class": r["class"],
            "gen_spacegroup": r["spacegroup"], "sg_no": sg_no,
            "band_gap_ev": round(float(r["band_gap_ev"]), 3),
            "slme_eta_percent": round(float(r["slme_eta_percent"]), 2),
            "jsc": round(float(r["jsc"]), 3) if np.isfinite(r["jsc"]) else "",
            "voc": round(float(r["voc"]), 3) if np.isfinite(r["voc"]) else "",
            "ehull_ev_per_atom": round(float(r["ehull_ev_per_atom"]), 4),
            "fluoride": bool(r["fluoride"]), "open_d": bool(r["open_d"]), "toxic": bool(r["toxic"]),
            "selection_score": round(float(r["selection_score"]), 4),
            "step": int(r["step"]), "index": int(r["index"]), "cif": cif,
        })
    summ = pd.DataFrame(recs)
    summ.to_csv(out / "slme_verification_candidates_summary.csv", index=False)
    (out / "selection_meta.json").write_text(json.dumps({
        "run": str(run), "gap_window": [args.gap_lo, args.gap_hi], "eta_min": args.eta_min,
        "ehull_max": args.ehull_max, "n_target_unique": int(len(cand)),
        "n_plausible": int(len(plausible)), "n_fluoride": int(len(fluoride)),
    }, indent=2))
    print(f"\nselected {len(sel)} ({len(plausible)} plausible + {len(fluoride)} fluoride) → {out}")
    print(summ[["rank", "formula", "class", "sg_no", "band_gap_ev", "slme_eta_percent",
                "ehull_ev_per_atom", "toxic"]].to_string(index=False))


if __name__ == "__main__":
    main()
