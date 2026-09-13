#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from ase.io import read as ase_read  # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402
from pymatgen.io.cif import CifWriter  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402


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
    diff = np.abs(v - t)                                   # linear_target (default): tent
    return np.clip((maxv - diff) / (maxv - minv), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--select", choices=["all", "gap", "runi"], default="all",
                   help="all=every S.U.N. survivor (default); gap=in gap-window; runi=r_uni>=threshold")
    p.add_argument("--gap-window", type=float, nargs=2, default=(0.8, 1.9),
                   help="target region: band gap in [lo,hi] eV (default 0.8 1.9)")
    p.add_argument("--solar-gap", type=float, nargs=2, default=(1.0, 1.8))
    p.add_argument("--solar-eta", type=float, default=0.25)
    p.add_argument("--symprec", type=float, default=0.1, help="symprec for the REPORTED spacegroup")
    p.add_argument("--output", type=Path, default=None, help="default: <run>/dft_candidates_slme_sun")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    run = a.run_dir.resolve()
    s = a.step
    out = (a.output or (run / "dft_candidates_slme_sun")).resolve()
    if (out / "cifs").exists():
        shutil.rmtree(out / "cifs")          # avoid stale CIFs from a prior selection
    (out / "cifs").mkdir(parents=True, exist_ok=True)

    # ---- reward config ----
    hp = yaml.safe_load((run / "hparams.yaml").read_text())
    rc = hp["reward"]
    reduce = str(rc.get("reduce", "mean"))
    threshold = float(rc.get("reward_threshold", 0.5))
    props = []  # (name, root, target, minv, maxv, weight, mode, tau)
    for pc in rc["prop_cfg"]:
        props.append((pc["name"], pc["calculator"]["root_dir"], pc.get("target"),
                      float(pc.get("minv", 0.0)), float(pc.get("maxv", 1.0)),
                      float(pc.get("weight", 1.0)),
                      str(pc.get("reward_mode", "linear_target")), float(pc.get("tau", 0.3))))

    # ---- per-prop raw values (index-aligned to eval set) ----
    raw = {}
    for name, root, *_ in props:
        f = run / root / f"step_{s:04d}.txt"
        raw[name] = (np.array([float(x) for x in f.read_text().split()]) if f.exists()
                     else np.array([]))
    n = max((len(v) for v in raw.values()), default=0)

    # ---- reconstruct r_uni ----
    scaled = {name: _score_prop(t, mn, mx, raw[name], md, ta)
              for name, _r, t, mn, mx, _w, md, ta in props}
    if reduce == "min":
        r_uni = np.nanmin(np.array([scaled[p[0]] for p in props]), axis=0)
    elif reduce == "weight":
        r_uni = np.sum(np.array([scaled[p[0]] * p[5] for p in props]), axis=0)
    else:
        r_uni = np.nanmean(np.array([scaled[p[0]] for p in props]), axis=0)

    gapkey = next((p[0] for p in props if p[0] == "band_gap"), None)
    etakey = next((p[0] for p in props if "eta" in p[0].lower()), None)
    jsckey = next((p[0] for p in props if "jsc" in p[0].lower()), None)
    vockey = next((p[0] for p in props if "voc" in p[0].lower()), None)

    # ---- e_hull lookup keyed by total_energy (robust to post-relax drop) ----
    stab = pd.read_csv(run / "samples" / f"step_{s:04d}_stability.csv")
    ehmap = {round(float(r["total_energy_ev"]), 4): r for _, r in stab.iterrows()}

    # ---- structures ----
    xyz = run / "samples" / f"step_{s:04d}_eval.extxyz"
    atoms_list = ase_read(str(xyz), index=":")
    assert len(atoms_list) == n, f"eval frames {len(atoms_list)} != reward values {n}"

    glo, ghi = a.gap_window
    rows = []
    for i, atoms in enumerate(atoms_list):
        gap = float(raw[gapkey][i]) if gapkey else np.nan
        eta = float(raw[etakey][i]) if etakey else np.nan
        in_gap = bool(glo <= gap <= ghi)
        if a.select == "gap" and not in_gap:
            continue
        if a.select == "runi" and not (r_uni[i] >= threshold):   # nan-safe drop
            continue
        st = AseAtomsAdaptor.get_structure(atoms)
        try:
            sga = SpacegroupAnalyzer(st, symprec=a.symprec)
            sg_num, sg_sym = sga.get_space_group_number(), sga.get_space_group_symbol()
        except Exception:  # noqa: BLE001
            sg_num, sg_sym = 0, "P1"
        te = float(atoms.info.get("total_energy", np.nan))
        srow = ehmap.get(round(te, 4))
        eh = float(srow["energy_above_hull_ev_per_atom"]) if srow is not None else np.nan
        sceh = float(srow["self_consistent_energy_above_hull_ev_per_atom"]) if srow is not None else np.nan
        jsc = float(raw[jsckey][i]) if jsckey else np.nan
        voc = float(raw[vockey][i]) if vockey else np.nan
        solar = bool(a.solar_gap[0] <= gap <= a.solar_gap[1] and eta >= a.solar_eta)
        rows.append({
            "eval_index": i, "formula": st.composition.reduced_formula,
            "spacegroup_number": sg_num, "spacegroup_symbol": sg_sym, "nsites": len(st),
            "band_gap_eV": round(gap, 4), "eta_slme": (round(eta, 4) if np.isfinite(eta) else np.nan),
            "jsc_mA_cm2": (round(jsc, 3) if np.isfinite(jsc) else np.nan),
            "voc_V": (round(voc, 4) if np.isfinite(voc) else np.nan),
            "r_uni": (round(float(r_uni[i]), 4) if np.isfinite(r_uni[i]) else np.nan),
            "in_gap_window": in_gap, "in_solar_window": solar,
            "e_hull_ev_atom": round(eh, 4), "self_consistent_e_hull_ev_atom": round(sceh, 4),
            "total_energy_ev": round(te, 4),
            "_struct": st, "_ff": st.composition.formula.replace(" ", ""),
            "_sortkey": (-(r_uni[i] if np.isfinite(r_uni[i]) else -1.0), eh),
        })

    rows.sort(key=lambda d: d["_sortkey"])
    for rank, d in enumerate(rows, 1):
        fname = f"rank{rank:02d}_{d['_ff']}_sg{d['spacegroup_number']}.cif"
        CifWriter(d["_struct"], symprec=None).write_file(out / "cifs" / fname)
        d["cif"] = f"cifs/{fname}"
        d["rank"] = rank

    df = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_")} for d in rows])
    cols = ["rank", "cif", "formula", "spacegroup_number", "spacegroup_symbol", "nsites",
            "band_gap_eV", "eta_slme", "jsc_mA_cm2", "voc_V", "r_uni",
            "in_gap_window", "in_solar_window",
            "e_hull_ev_atom", "self_consistent_e_hull_ev_atom", "total_energy_ev", "eval_index"]
    df = df[cols]
    df.to_csv(out / "worklist.csv", index=False)

    n_gap = int(df["in_gap_window"].sum())
    n_solar = int(df["in_solar_window"].sum())
    n_stable = int((df["e_hull_ev_atom"] <= 0.1).sum())
    (out / "README.txt").write_text(
        f"SLME S.U.N. survivors -> DFT/DFPT worklist\n"
        f"source run : {run}\n"
        f"checkpoint : c2 tsenn_slme_...bgcenter13_eta08_e3nngap best_reward (step 189)\n"
        f"select     : {a.select}  ->  {len(df)} CIFs\n"
        f"target region (surrogate E3NN gap): [{glo}, {ghi}] eV  ->  {n_gap} flagged in_gap_window\n"
        f"solar window: gap [{a.solar_gap[0]},{a.solar_gap[1]}] & eta>={a.solar_eta}  ->  {n_solar}; "
        f"e_hull<=0.1 -> {n_stable}\n"
        f"cifs       : P1, exact MatterSim-relaxed coordinates (re-relax in DFT; spacegroup col = symprec={a.symprec} estimate)\n"
        f"values     : band_gap/eta/jsc/voc are E3NN/TSENN surrogate predictions (jsc/voc are weight-0, context only)\n"
        f"NOTE       : compute the true target-region yield from the DFT band gaps; the surrogate flags are guidance.\n"
    )

    print(f"exported {len(df)} CIFs (select={a.select}) -> {out}/cifs/")
    print(f"  in_gap_window[{glo},{ghi}]: {n_gap} | solar-window: {n_solar} | e_hull<=0.1: {n_stable}")
    print(f"  worklist: {out}/worklist.csv")
    print(df[["rank", "formula", "spacegroup_symbol", "band_gap_eV", "eta_slme", "r_uni",
              "in_gap_window", "e_hull_ev_atom"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
