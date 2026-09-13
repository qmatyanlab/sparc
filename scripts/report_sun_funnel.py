#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from pymatgen.core import Structure  # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402


def _score_prop(target, minv, maxv, vals, mode="linear_target", tau=0.3):
    """Replicates rewards.reward.Reward._score_property (ascending / descending / float-tent)."""
    v = np.asarray(vals, dtype=float)
    if target == "ascending":
        return np.clip((v - minv) / (maxv - minv), 0.0, 1.0)
    if target == "descending":
        return np.clip((maxv - v) / (maxv - minv), 0.0, 1.0)
    t = float(target)
    if mode == "gaussian_target":
        return np.clip(np.exp(-0.5 * ((v - t) / tau) ** 2), 0.0, 1.0)
    diff = np.abs(v - t)                                  # linear_target (default): tent
    return np.clip((maxv - diff) / (maxv - minv), 0.0, 1.0)


def _len_pt(path: Path):
    if not path.exists():
        return None
    try:
        import torch
        d = torch.load(path, map_location="cpu")
        return len(d) if isinstance(d, list) else None
    except Exception:  # noqa: BLE001
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--n-generated", type=int, default=1000, help="batch_size x num_batches")
    p.add_argument("--solar-gap", type=float, nargs=2, default=(1.0, 1.8))
    p.add_argument("--solar-eta", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    run = a.run_dir.resolve()
    s = a.step

    # ---- reward config from hparams.yaml ----
    hp = yaml.safe_load((run / "hparams.yaml").read_text())
    rc = hp["reward"]
    reduce = str(rc.get("reduce", "mean"))
    threshold = float(rc.get("reward_threshold", 0.5))
    props = []  # (name, root_dir, target, minv, maxv, weight, mode, tau)
    for pc in rc["prop_cfg"]:
        props.append((pc["name"], pc["calculator"]["root_dir"], pc.get("target"),
                      float(pc.get("minv", 0.0)), float(pc.get("maxv", 1.0)),
                      float(pc.get("weight", 1.0)),
                      str(pc.get("reward_mode", "linear_target")), float(pc.get("tau", 0.3))))

    # ---- per-prop RAW survivor values (index-aligned to the eval set) ----
    raw = {}
    for name, root, *_ in props:
        f = run / root / f"step_{s:04d}.txt"
        raw[name] = (np.array([float(x) for x in f.read_text().split()]) if f.exists()
                     else np.array([]))
    n_scored = max((len(v) for v in raw.values()), default=0)

    # ---- reconstruct r_uni (exact rewards/reward.py logic) ----
    scaled = {}
    for name, root, target, minv, maxv, w, mode, tau in props:
        scaled[name] = _score_prop(target, minv, maxv, raw[name], mode, tau)
    if reduce == "min":
        r_uni = np.min(np.array([scaled[n] for n, *_ in props]), axis=0)
    elif reduce == "weight":
        r_uni = np.sum(np.array([scaled[n] * w for n, _r, _t, _mn, _mx, w, _m, _ta in props]), axis=0)
    else:  # mean
        r_uni = np.mean(np.array([scaled[n] for n, *_ in props]), axis=0)
    n_inrange = int((r_uni >= threshold).sum())

    # ---- SLME 'solar window' (if a band_gap + an eta prop exist) ----
    solar = None
    gapkey = next((n for n, *_ in props if n == "band_gap"), None)
    etakey = next((n for n, *_ in props if "eta" in n.lower()), None)
    if gapkey and etakey and len(raw[gapkey]) and len(raw[etakey]):
        g, e = raw[gapkey], raw[etakey]
        m = (g >= a.solar_gap[0]) & (g <= a.solar_gap[1]) & (e >= a.solar_eta)
        solar = int(m.sum())

    # ---- funnel: metrics.csv + stability.csv + valid.pt ----
    mdf = pd.read_csv(run / "metrics.csv")
    mrow = mdf.iloc[s] if s < len(mdf) else mdf.iloc[0]
    mv = lambda k, d=np.nan: (float(mrow[k]) if (k in mdf.columns and pd.notna(mrow[k])) else d)
    filter_total, filter_kept = mv("filter_total"), mv("filter_kept")
    n_valid_pt = _len_pt(run / "samples" / f"step_{s:04d}_valid.pt")
    stab_p = run / "samples" / f"step_{s:04d}_stability.csv"
    per, n_ref, n_sun = {}, None, None
    if stab_p.exists():
        stab = pd.read_csv(stab_p)
        n_ref = len(stab)
        mask = np.ones(n_ref, bool)
        for c in ("is_valid", "is_unique", "is_novel", "is_stable"):
            if c in stab.columns:
                per[c] = int(stab[c].sum())
                mask &= stab[c].astype(bool).to_numpy()
        n_sun = int(mask.sum())
    else:
        n_ref = int(filter_total) if not np.isnan(filter_total) else None
        n_sun = int(filter_kept) if not np.isnan(filter_kept) else None

    # ---- distinct materials: StructureMatcher within reduced-formula groups ----
    from ase.io import read as ase_read
    xyz = run / "samples" / f"step_{s:04d}_eval.extxyz"
    structs, formulas = [], []
    if xyz.exists():
        for atoms in ase_read(str(xyz), index=":"):
            st = AseAtomsAdaptor.get_structure(atoms)
            structs.append(st)
            formulas.append(st.composition.reduced_formula)
    groups = defaultdict(list)
    for i, f in enumerate(formulas):
        groups[f].append(i)
    sm = StructureMatcher()
    n_distinct = 0
    for f, idxs in groups.items():
        assigned = [False] * len(idxs)
        for ii in range(len(idxs)):
            if assigned[ii]:
                continue
            n_distinct += 1
            assigned[ii] = True
            for jj in range(ii + 1, len(idxs)):
                if not assigned[jj] and sm.fit(structs[idxs[ii]], structs[idxs[jj]]):
                    assigned[jj] = True
    n_distinct_formula = len(groups)

    def pct(n, d):
        return f"{100 * n / d:.1f}%" if (d and n is not None) else "n/a"

    gen = a.n_generated
    print(f"\n=== S.U.N. + target-range funnel  ({run.name}) ===")
    print(f"reward: reduce={reduce}, threshold={threshold}, props={[p[0] for p in props]}")
    print(f"generated (batch_size x num_batches)   : {gen}")
    print(f"valid (pre-relax)                      : {n_valid_pt}  ({pct(n_valid_pt, gen)})")
    print(f"e_hull-referenced                      : {n_ref}")
    if per:
        print(f"  valid/unique/novel/stable          : "
              f"{per.get('is_valid')}/{per.get('is_unique')}/{per.get('is_novel')}/{per.get('is_stable')}")
    print(f"S.U.N. survivors (S&U&N&valid)         : {n_sun}  ({pct(n_sun, gen)} of generated)")
    print(f"  scored survivors (r_uni computed)    : {n_scored}   (structures read: {len(structs)})")
    print(f"DISTINCT materials (StructureMatcher)  : {n_distinct}  ({n_distinct_formula} formulas)")
    print(f"IN target range (r_uni>={threshold})           : {n_inrange} / {n_scored}  ({pct(n_inrange, n_scored)})")
    print(f"OUTSIDE target range                   : {n_scored - n_inrange} / {n_scored}  ({pct(n_scored - n_inrange, n_scored)})")
    if solar is not None:
        print(f"[SLME] solar window (E_g {a.solar_gap[0]}-{a.solar_gap[1]} & eta>={a.solar_eta}): "
              f"{solar} / {n_scored}  ({pct(solar, n_scored)})")

    out = run / "sun_funnel_summary.csv"
    pd.DataFrame([{
        "run": run.name, "reduce": reduce, "threshold": threshold, "generated": gen,
        "valid": n_valid_pt, "e_hull_referenced": n_ref,
        "n_valid": per.get("is_valid"), "n_unique": per.get("is_unique"),
        "n_novel": per.get("is_novel"), "n_stable": per.get("is_stable"),
        "sun_survivors": n_sun, "scored": n_scored,
        "distinct_structures": n_distinct, "distinct_formulas": n_distinct_formula,
        "in_range": n_inrange, "outside_range": n_scored - n_inrange,
        "solar_window": solar,
    }]).to_csv(out, index=False)
    print(f"\n-> wrote {out}")


if __name__ == "__main__":
    main()
