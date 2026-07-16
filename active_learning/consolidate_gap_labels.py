#!/usr/bin/env python3
"""
Phase B: consolidate PBE band-gap labels for fine-tuning the E3NN gap surrogate.

Harvests (relaxed_structure -> PBE gap) pairs from the sibling VASP/atomate2
project `dielectric_DFPT` (no re-DFT: structures + gaps are embedded):
  - local_results.json  (bulldozer batch, gap field `pbe_gap`)
  - results.json        (Mongo/FireWorks batch, gap field `bandgap`)

All labels are MP-VASP-PBE (spin-polarized, GGA(+U)) on the double-relaxed
geometry -- the same fidelity as the surrogate's training data. r2SCAN is never
in these files, so it cannot leak. Metals (gap<=gate) are kept as small/zero-gap
regression targets (teaching metal discrimination is the whole point OOD).

Split is by GENERATION FAMILY (cif_dir) so the OOD test measures generalization
to unseen generator families, not memorization.

Output: data/gap_labels_{train,val,ood_test}.json  (each a list of
{id, formula, gap, is_metal, source, family, run_type, structure(as_dict)}).

Usage:
    python consolidate_gap_labels.py [--dfpt DIR] [--val-frac 0.15]
        [--ood-frac 0.15] [--seed 0] [--no-dedup]
"""

import os
import json
import random
import argparse
from collections import Counter, defaultdict

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DFPT = "/global/cfs/cdirs/m2663/angush/sparc_dft_verifications/dielectric_DFPT"
OUT_DIR = os.path.join(HERE, "data")


def _family(rec):
    """Generation family = basename of the source cif dir (fallback: dataset/prefix)."""
    cd = rec.get("cif_dir") or ""
    if cd:
        return os.path.basename(cd.rstrip("/"))
    if rec.get("dataset"):
        return str(rec["dataset"])
    cand = rec.get("candidate", "unknown")
    return cand.split("_")[0] if "_" in cand else "unknown"


def _load(path, gap_key, source):
    if not os.path.exists(path):
        print(f"  (skip, not found: {path})")
        return []
    data = json.load(open(path))
    out = []
    for r in data:
        if r.get("status") == "FAILED" or r.get("run_error"):
            continue
        gap = r.get(gap_key)
        rs = r.get("relaxed_structure")
        if gap is None or rs is None:
            continue
        ce = r.get("computed_entry") or {}
        run_type = ((ce.get("parameters") or {}).get("run_type")) if isinstance(ce, dict) else None
        out.append({
            "id": r.get("candidate"),
            "formula": r.get("formula"),
            "gap": float(gap),
            "is_metal": bool(r.get("is_metal", float(gap) <= 0.30)),
            "source": source,
            "family": _family(r),
            "run_type": run_type or "GGA",
            "structure": rs,
        })
    print(f"  {source}: {len(out)} usable rows from {os.path.basename(path)}")
    return out


def _dedup(records):
    """Structure-match dedup within composition buckets; keep first occurrence."""
    sm = StructureMatcher(primitive_cell=True, attempt_supercell=False)
    buckets = defaultdict(list)
    for rec in records:
        s = Structure.from_dict(rec["structure"])
        key = (s.composition.reduced_formula, len(s))
        buckets[key].append((rec, s))
    kept, ndup = [], 0
    for group in buckets.values():
        uniq = []
        for rec, s in group:
            if any(sm.fit(s, u_s) for _, u_s in uniq):
                ndup += 1
                continue
            uniq.append((rec, s))
        kept.extend(rec for rec, _ in uniq)
    print(f"  dedup: removed {ndup} duplicates -> {len(kept)} unique")
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfpt", default=DEFAULT_DFPT)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--ood-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args(argv)

    print("Loading PBE gap labels from dielectric_DFPT ...")
    recs = _load(os.path.join(args.dfpt, "local_results.json"), "pbe_gap", "dfpt_local")
    recs += _load(os.path.join(args.dfpt, "results.json"), "bandgap", "dfpt_mongo")
    print(f"total raw: {len(recs)}")
    if not args.no_dedup:
        recs = _dedup(recs)

    # gap distribution sanity
    metals = sum(r["gap"] == 0.0 for r in recs)
    subgate = sum(0.0 < r["gap"] <= 0.30 for r in recs)
    gapped = sum(r["gap"] > 0.30 for r in recs)
    print(f"gap dist: metals(=0)={metals}  sub-gate(0,0.3]={subgate}  gapped(>0.3)={gapped}")

    # family-level split: whole families -> OOD test; remainder -> train/val
    fams = defaultdict(list)
    for r in recs:
        fams[r["family"]].append(r)
    fam_names = sorted(fams)
    rng = random.Random(args.seed)
    rng.shuffle(fam_names)
    n_total = len(recs)
    ood, ood_n = [], 0
    for fam in fam_names:
        if ood_n < args.ood_frac * n_total and len(fams[fam]) < 0.5 * n_total:
            ood.extend(fams[fam]); ood_n += len(fams[fam])
    ood_fams = {r["family"] for r in ood}
    rest = [r for r in recs if r["family"] not in ood_fams]
    rng.shuffle(rest)
    n_val = int(args.val_frac * len(rest))
    val, train = rest[:n_val], rest[n_val:]

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("ood_test", ood)]:
        p = os.path.join(OUT_DIR, f"gap_labels_{name}.json")
        json.dump(split, open(p, "w"))
        print(f"{name:9s}: {len(split):4d} rows "
              f"(metals {sum(r['gap']==0 for r in split)}) -> {p}")
    print("OOD families held out:", sorted(ood_fams))
    print("all families:", dict(Counter(r["family"] for r in recs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
