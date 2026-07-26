#!/usr/bin/env python3
"""Select DFPT-verification candidates from the in-plane-isotropy dielectric RL run.

Target-region + metastable shortlist for `dielectric_inplane_isotropy_gapgate_mprime_newbg_b96`.
For every EVAL-pool structure it computes the combined RL reward exactly as rewards/reward.py:

    r_uni = min( scale(gap, 0.3, 0.8), scale(iso, 0.9, 1.0) )     # scale = linear, clipped [0,1]

then applies, in order, the filters:

    gap >= 0.3 eV        (past the band-gap gate; excludes metals)
    r_uni >= 0.5         (target region: gap >~ 0.55 eV AND eps_xx=eps_yy in-plane isotropic)
    positive-definite eps
    finite e_above_hull <= 0.10 eV/atom   (the run's 'stable' threshold; a finite ehull also
                                           means the structure passed the run's novel+unique filter)

No direction filter. Prints the SURVIVOR WATERFALL after each stage. Dedup is STRUCTURE-based
(StructureMatcher within reduced-formula groups) so distinct polymorphs of the same formula both
survive -- `--dedup formula` (old behaviour) and `--dedup none` are also available. Ranks
(r_uni desc, ehull asc, sampling-count desc) and exports conventional-cell CIFs + `_summary.csv`
to <run>/deliverables_dfpt_candidates/.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from plot_layered_uniaxial_samples import load_step_records  # noqa: E402
from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402


def lin(v: float, a: float, b: float) -> float:
    return float(np.clip((v - a) / (b - a + 1e-12), 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--r-uni-min", type=float, default=0.5)
    p.add_argument("--ehull-max", type=float, default=0.10, help="eV/atom")
    p.add_argument("--gap-min", type=float, default=0.3, help="eV (band-gap gate)")
    p.add_argument("--dedup", choices=["structure", "formula", "none"], default="structure")
    p.add_argument("--top", type=int, default=500, help="max candidates to export")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=None, help="pickle of computed records")
    return p.parse_args()


def load_records(run: Path, cache: Path | None):
    if cache and cache.exists():
        return pickle.load(open(cache, "rb"))
    recs = load_step_records(run, requested_step=None)
    bg_dir = run / "rewards" / "bandgap"
    gaps_by_step: dict[int, np.ndarray] = {}
    for r in recs:
        s = r["step"]
        if s not in gaps_by_step:
            f = bg_dir / f"step_{s:04d}.txt"
            gaps_by_step[s] = (np.array([float(x) for x in f.read_text().split() if x.strip()])
                               if f.exists() else np.array([]))
    out = []
    for r in recs:
        arr = gaps_by_step.get(r["step"], np.array([]))
        gap = float(arr[r["index"]]) if r["index"] < len(arr) else float("nan")
        iso = float(r["reward"])
        r_uni = min(lin(gap, 0.3, 0.8), lin(iso, 0.9, 1.0))
        c = dict(r); c.update(gap=gap, iso=iso, r_uni=r_uni)
        out.append(c)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(out, open(cache, "wb"))
    return out


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    outdir = (args.outdir or (run / "deliverables_dfpt_candidates")).resolve()
    cand = load_records(run, args.cache)

    def finite(x):
        return x is not None and np.isfinite(x)

    # ---- survivor waterfall (cumulative) ----
    stages = [
        ("valid-tensor eval structures", lambda c: True),
        ("gap >= %.2g eV (past band-gap gate)" % args.gap_min, lambda c: finite(c["gap"]) and c["gap"] >= args.gap_min),
        ("r_uni >= %.2g (target region)" % args.r_uni_min, lambda c: c["r_uni"] >= args.r_uni_min),
        ("positive-definite eps", lambda c: bool(c["positive_definite"])),
        ("finite e_hull", lambda c: finite(c["ehull"])),
        ("e_hull <= %.2g eV/atom (stable)" % args.ehull_max, lambda c: finite(c["ehull"]) and c["ehull"] <= args.ehull_max),
    ]
    surviving = cand
    print("SURVIVOR WATERFALL (cumulative):")
    print(f"  {len(surviving):5d}  {stages[0][0]}")
    for label, test in stages[1:]:
        surviving = [c for c in surviving if test(c)]
        print(f"  {len(surviving):5d}  after: {label}")
    passed = surviving

    # dedup counts for reference
    n_formula = len({c["formula"] for c in passed})
    print(f"\n  passed = {len(passed)} structures  |  {n_formula} unique reduced-formulas")

    # ---- deduplicate ----
    occ_f: dict[str, int] = defaultdict(int)
    for c in passed:
        occ_f[c["formula"]] += 1
    if args.dedup == "none":
        uniq = list(passed)
        for c in uniq:
            c["n_same"] = 1
    elif args.dedup == "formula":
        best: dict[str, dict] = {}
        for c in passed:
            f = c["formula"]
            if f not in best or (c["r_uni"], -c["ehull"]) > (best[f]["r_uni"], -best[f]["ehull"]):
                best[f] = c
        uniq = list(best.values())
        for c in uniq:
            c["n_same"] = occ_f[c["formula"]]
    else:  # structure: StructureMatcher within each formula group
        sm = StructureMatcher()
        by_formula: dict[str, list] = defaultdict(list)
        for c in passed:
            by_formula[c["formula"]].append(c)
        uniq = []
        for group in by_formula.values():
            structs = [Structure.from_str(c["cif"], fmt="cif") for c in group]
            assigned = [False] * len(group)
            for i in range(len(group)):
                if assigned[i]:
                    continue
                members = [group[i]]; assigned[i] = True
                for j in range(i + 1, len(group)):
                    if not assigned[j] and sm.fit(structs[i], structs[j]):
                        assigned[j] = True; members.append(group[j])
                rep = dict(max(members, key=lambda c: (c["r_uni"], -c["ehull"])))
                rep["n_same"] = len(members)
                uniq.append(rep)
        print(f"  structure-dedup: {len(passed)} -> {len(uniq)} distinct structures "
              f"({n_formula} distinct formulas)")

    uniq.sort(key=lambda c: (-c["r_uni"], c["ehull"], -c["n_same"]))
    if not uniq:
        print("no candidates passed"); return

    sel = uniq[:args.top]
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("r*.cif"):
        old.unlink()
    rows = []
    for rank, c in enumerate(sel, 1):
        st = Structure.from_str(c["cif"], fmt="cif")
        try:
            sga = SpacegroupAnalyzer(st, symprec=0.01)
            conv = sga.get_conventional_standard_structure()
            sg_sym, sg_no = sga.get_space_group_symbol(), sga.get_space_group_number()
        except Exception:  # noqa: BLE001
            conv, sg_sym, sg_no = st, "P1", 1
        name = f"r{rank:03d}_{c['formula']}_SG{sg_no}_step{c['step']}_i{c['index']}.cif"
        conv.to(filename=str(outdir / name), fmt="cif")
        exx, eyy, ezz = c["eps_xx"], c["eps_yy"], c["eps_zz"]
        perp = 0.5 * (exx + eyy)
        rows.append({
            "rank": rank, "formula": c["formula"], "spacegroup": sg_sym, "sg_no": sg_no,
            "r_uni": round(c["r_uni"], 4), "band_gap_ev": round(c["gap"], 3),
            "iso_scalar": round(c["iso"], 4),
            "eps_xx": round(exx, 3), "eps_yy": round(eyy, 3), "eps_zz": round(ezz, 3),
            "eps_zz_over_par": round(ezz / perp, 3) if perp else float("nan"),
            "ehull_ev_atom": round(c["ehull"], 4), "n_same": c["n_same"],
            "step": c["step"], "index": c["index"], "cif": name,
        })
    with open(outdir / "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nexported {len(sel)} CIFs + _summary.csv -> {outdir}  (dedup={args.dedup})")


if __name__ == "__main__":
    main()
