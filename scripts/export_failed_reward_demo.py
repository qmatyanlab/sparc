#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from plot_layered_uniaxial_samples import load_step_records  # noqa: E402

GAP_MIN, GAP_MAX = 0.3, 0.8      # band-gap gate window (ascending)
DIEL_MIN, DIEL_MAX = 0.9, 1.0    # dielectric-score window (ascending)
Z_FLOOR = 0.01                   # z_anisotropy_floor: cubic-exclusion gate


def _clip(v, a, b):
    return float(np.clip((v - a) / (b - a + 1e-12), 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--per-mode", type=int, default=8,
                   help="max distinct-formula structures per rejection mode")
    p.add_argument("--ehull-max", type=float, default=0.10)
    p.add_argument("--eps-cap", type=float, default=60.0, help="drop |eps| blow-ups")
    p.add_argument("--gap-floor", type=float, default=0.02,
                   help="metallic mode: keep only finite gaps >= this (DFPT-verifiable)")
    p.add_argument("--output", type=Path, default=None,
                   help="default: <run>/failed_reward_demo")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    out = (args.output or (run / "failed_reward_demo")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    steps = sorted(int(p.stem.split("_")[1]) for p in (run / "samples").glob("step_*_eval.pt"))

    buckets: dict[str, dict[str, dict]] = {"metallic": {}, "cubic": {}, "biaxial": {}}
    for s in steps:
        try:
            recs = load_step_records(run, requested_step=s)
        except Exception:  # noqa: BLE001
            continue
        bg = run / "rewards" / "bandgap" / f"step_{s:04d}.txt"
        gaps = (np.array([float(x) for x in bg.read_text().split()])
                if bg.exists() else np.array([]))
        for r in recs:
            i = r["index"]
            gap = float(gaps[i]) if i < len(gaps) else 0.0
            xx, yy, zz = float(r["eps_xx"]), float(r["eps_yy"]), float(r["eps_zz"])
            perp = 0.5 * (xx + yy)
            az = abs(zz - perp) / (abs(zz) + abs(perp) + 1e-9)
            mm = abs(xx - yy) / (abs(xx) + abs(yy) + 1e-9)
            emax = max(abs(xx), abs(yy), abs(zz))
            eh, sg = r.get("ehull"), r.get("spacegroup")
            if not bool(r.get("positive_definite", False)):
                continue
            if eh is None or eh > args.ehull_max or emax >= args.eps_cap or emax < 3:
                continue
            z_gate = min(1.0, az / Z_FLOOR)
            iso = (1.0 - mm) * z_gate
            r_uni = min(_clip(gap, GAP_MIN, GAP_MAX), _clip(iso, DIEL_MIN, DIEL_MAX))
            rec = dict(formula=str(r["formula"]).replace(" ", ""), sg=int(sg) if sg else 0,
                       xx=xx, yy=yy, zz=zz, gap=gap, mm=mm, az=az, iso=iso, r_uni=r_uni,
                       ehull=float(eh), step=s, index=i, cif=r.get("cif"))
            # classify into exactly one mode (metallic takes priority: the gate fires first)
            if args.gap_floor <= gap < GAP_MIN:
                mode, k, keep = "metallic", "gap", (rec["gap"], False)   # closest to corner
            elif gap >= GAP_MIN and az < Z_FLOOR and mm <= 0.02:
                mode, k, keep = "cubic", "az", (rec["az"], False)        # most cubic
            elif gap >= GAP_MIN and mm >= 0.10:
                mode, k, keep = "biaxial", "mm", (rec["mm"], True)       # most biaxial
            else:
                continue
            d = buckets[mode]
            f = rec["formula"]
            if f not in d:
                d[f] = rec
            else:
                better = (rec[k] > d[f][k]) if k == "mm" else (rec[k] < d[f][k])
                if better:
                    d[f] = rec

    order = {"metallic": ("gap", False), "cubic": ("az", False), "biaxial": ("mm", True)}
    rows = []
    for mode, d in buckets.items():
        key, rev = order[mode]
        for rank, c in enumerate(sorted(d.values(), key=lambda x: x[key], reverse=rev)[:args.per_mode]):
            name = (f"{mode}_r{rank:02d}_{c['formula']}_SG{c['sg']}"
                    f"_Eg{c['gap']:.2f}_step{c['step']}_i{c['index']}.cif")
            if c["cif"]:
                (out / name).write_text(c["cif"])
            rows.append({"mode": mode, "rank": rank, "formula": c["formula"],
                         "spacegroup": c["sg"], "eps_xx": round(c["xx"], 3),
                         "eps_yy": round(c["yy"], 3), "eps_zz": round(c["zz"], 3),
                         "gap_eV": round(c["gap"], 3), "inplane_mismatch": round(c["mm"], 4),
                         "A_z": round(c["az"], 4), "iso_score": round(c["iso"], 4),
                         "r_uni": round(c["r_uni"], 3), "ehull": round(c["ehull"], 4),
                         "step": c["step"], "index": c["index"], "cif": name})

    with open(out / "_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cnt = Counter(r["mode"] for r in rows)
    print(f"-> {out}")
    for m in ("metallic", "cubic", "biaxial"):
        print(f"   {m:9s}: {cnt.get(m, 0)} CIFs")
    print(f"   _summary.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
