#!/usr/bin/env python3
"""Generate `_iso_meta.json` + `cifs/` for a Figure-4 bundle.

Figure 4's panel (b) is driven by `_iso_meta.json`, and **no generator for it existed** --
the published one was hand-edited (its editor history still shows an earlier hero set with
an orphan key left in `META`). This is that missing step.

It deliberately reads the *derived* artifacts rather than MongoDB:
`report_all.py`'s `--csv` (every DFT scalar) and `--json` (the full 3x3 tensors), plus the
run's own `_summary.csv` files for the surrogate star coordinates. That keeps it in the
sparc venv -- no `emmet.TaskDoc`, so no pymatgen-version coupling -- and means the figure
is reproducible from files a reviewer can read.

The two coordinate systems are NOT interchangeable and the paper states the distinction:
  * `STARS`  -> **surrogate** (E3NN predicted gap, r_uni). Panel (a) is the agent's belief,
                so a star must land inside the KDE built from those same two arrays.
  * `PROPS` / `FULL_EPS` -> **DFT**. Panel (b) is the verification.
Mixing them up is silent: the figure still renders, with the stars in the wrong place.

Usage:
  # reproduce the published bundle, as a regression test
  uv run python scripts/build_iso_meta.py <old_run> --verify <old_run>/figure4_bundle/_iso_meta.json

  # build for a new run
  uv run python scripts/build_iso_meta.py <run> --heroes LBL1 LBL2 LBL3 LBL4 --out <run>/figure4_bundle
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Crystal-system accents, matching the published figure's palette.
ACCENTS = ["#1f6fd0", "#b0338f", "#2a8f5a", "#c25e00"]


def formula_mathtext(f: str) -> str:
    """`RbGePO4` -> `RbGePO$_4$` (digit runs become subscripts)."""
    return re.sub(r"(\d+)", r"$_{\1}$", f).replace("$_{", "$_").replace("}$", "$") \
        if re.search(r"\d", f) else f


def hm_mathtext(sym: str) -> str:
    """pymatgen H-M symbol -> the figure's mathtext style.

    `P6_3` -> `P6$_3$`   `P-31m` -> `P$\\bar{3}$1m`   `Pbca` -> `Pbca`
    """
    s = re.sub(r"_(\d+)", r"$_\1$", sym)          # screw-axis subscript
    s = re.sub(r"-(\d)", r"$\\bar{\1}$", s)       # inversion bar
    return s


def _f(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def load_dft(run: Path) -> dict:
    """label -> DFT record, from report_all.py's csv + json."""
    d = run / "deliverables_dfpt_candidates"
    csv_p, json_p = d / "dfpt_dft_results.csv", d / "dfpt_dft_tensors.json"
    if not csv_p.exists():
        sys.exit(f"missing {csv_p}\nRun report_all.py first (step 4).")
    out = {}
    for r in csv.DictReader(open(csv_p, newline="")):
        out[r["label"]] = {
            "formula": r["formula"], "sgno": _f(r["spacegroup"]),
            "status": r["status"], "gap": _f(r["dft_gap"]),
            "ehull_ev": _f(r["dft_ehull"]), "ehull_approx": r.get("dft_ehull_approx"),
            "eps_diag": [_f(r["dft_eps0_xx"]), _f(r["dft_eps0_yy"]), _f(r["dft_eps0_zz"])],
            "tensor": None,
        }
    if json_p.exists():
        for rec in json.load(open(json_p)):
            if rec["label"] in out:
                out[rec["label"]]["tensor"] = rec.get("eps_electronic_tensor")
    return out


def load_surrogate(run: Path) -> dict:
    """label -> (surrogate gap, r_uni) for the STARS, from the run's own summaries."""
    out = {}
    c = run / "deliverables_dfpt_candidates" / "_summary.csv"
    if c.exists():
        for r in csv.DictReader(open(c, newline="")):
            out[Path(r["cif"]).stem] = (_f(r["band_gap_ev"]), _f(r["r_uni"]))
    d = run / "failed_reward_demo" / "_summary.csv"
    if d.exists():
        for r in csv.DictReader(open(d, newline="")):
            out[Path(r["cif"]).stem] = (_f(r["gap_eV"]), _f(r["r_uni"]))
    return out


def find_cif(run: Path, label: str) -> Path | None:
    for sub in ("deliverables_dfpt_candidates", "failed_reward_demo"):
        p = run / sub / f"{label}.cif"
        if p.exists():
            return p
    return None


def short_key(label: str, formula: str) -> str:
    """Hero dict key: the bare formula, as the published bundle used."""
    return formula


def spacegroup_of(cif: Path, symprec: float = 0.05):
    import warnings
    warnings.simplefilter("ignore")
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(Structure.from_file(cif), symprec=symprec)
    return sga.get_space_group_symbol(), int(sga.get_space_group_number())


def build(run: Path, heroes: list[str], reps: list | None, symprec: float) -> tuple[dict, dict]:
    dft, sur = load_dft(run), load_surrogate(run)
    meta = {"ORDER": [], "META": {}, "PROPS": {}, "FULL_EPS": {}, "STARS": [],
            "CIF_PATHS": {}}
    cifs = {}
    for i, label in enumerate(heroes):
        if label not in dft:
            sys.exit(f"{label}: not in dfpt_dft_results.csv")
        rec = dft[label]
        if rec["status"] != "dielectric" or rec["tensor"] is None:
            sys.exit(f"{label}: status={rec['status']} with no epsilon tensor -- a hero must "
                     f"have a real DFPT tensor (gapless_skipped structures have none).")
        cif = find_cif(run, label)
        if cif is None:
            sys.exit(f"{label}: no CIF found under deliverables_dfpt_candidates/ or failed_reward_demo/")
        key = short_key(label, rec["formula"])
        sym, sgno = spacegroup_of(cif, symprec)
        accent = ACCENTS[i % len(ACCENTS)]
        gap_s, runi = sur.get(label, (None, None))
        if gap_s is None:
            sys.exit(f"{label}: no surrogate (band_gap_ev, r_uni) found -- STARS would be wrong")

        meta["ORDER"].append(key)
        meta["META"][key] = {"name": formula_mathtext(rec["formula"]),
                             "formula": formula_mathtext(rec["formula"]),
                             "sg": hm_mathtext(sym), "sgno": sgno, "accent": accent}
        if reps:
            meta["META"][key]["rep"] = list(reps[i])
        meta["PROPS"][key] = {"ehull": f"{rec['ehull_ev'] * 1000:.1f}",
                              "eg": f"{rec['gap']:.2f}"}
        t = rec["tensor"]
        # `+ 0.0` normalises signed zero: rounding a tiny negative off-diagonal yields
        # `-0.0`, which is numerically identical but makes the JSON differ from the
        # published bundle and reads oddly in the table source.
        meta["FULL_EPS"][key] = [[round(float(t[a][b]), 3) + 0.0 for b in range(3)]
                                 for a in range(3)]
        meta["STARS"].append([key, gap_s, runi, accent])
        # short, stable CIF name (`RbGePO4_SG173.cif`), matching the published bundle
        # rather than the long RL label -- the label lives in `provenance` below.
        stem = f"{rec['formula']}_SG{sgno}"
        meta["CIF_PATHS"][key] = f"cifs/{stem}.cif"
        cifs[stem] = cif
        meta.setdefault("provenance", {})[key] = {"label": label,
                                                  "ehull_approx": rec.get("ehull_approx")}
    return meta, cifs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--heroes", nargs="+", default=None, help="4 DFT labels, left to right")
    ap.add_argument("--rep", nargs="+", default=None,
                    help="supercell repeat per hero, e.g. 2,2,1 2,2,1 3,3,1 1,1,1")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--symprec", type=float, default=0.05)
    ap.add_argument("--verify", type=Path, default=None,
                    help="compare against an existing _iso_meta.json instead of writing")
    args = ap.parse_args(argv)

    run = args.run_dir.resolve()
    if args.verify:
        ref = json.load(open(args.verify))
        heroes = args.heroes
        if not heroes:
            # recover the DFT labels behind the reference's short keys
            dft = load_dft(run)
            heroes = []
            for k in ref["ORDER"]:
                hits = [l for l, r in dft.items() if r["formula"] == k]
                if len(hits) != 1:
                    print(f"  {k}: {len(hits)} labels match formula -- pass --heroes explicitly")
                heroes.append(hits[0] if len(hits) == 1 else None)
            if any(h is None for h in heroes):
                return 1
            print(f"recovered hero labels: {heroes}")
        meta, _ = build(run, heroes, None, args.symprec)
        bad = []
        for sect in ("ORDER", "META", "PROPS", "FULL_EPS", "STARS", "CIF_PATHS"):
            if json.dumps(meta[sect], sort_keys=True) != json.dumps(ref[sect], sort_keys=True):
                bad.append(sect)
        print("VERIFY:", "identical to the reference" if not bad else f"differs in {bad}")
        for sect in bad:
            print(f"\n-- {sect}\n   built : {json.dumps(meta[sect], sort_keys=True)[:400]}"
                  f"\n   ref   : {json.dumps(ref[sect], sort_keys=True)[:400]}")
        return 1 if bad else 0

    if not args.heroes or len(args.heroes) != 4:
        sys.exit("--heroes needs exactly 4 DFT labels (3 uniaxial + 1 biaxial counter-example)")
    reps = [tuple(int(x) for x in r.split(",")) for r in args.rep] if args.rep else None
    meta, cifs = build(run, args.heroes, reps, args.symprec)

    out = (args.out or run / "figure4_bundle").resolve()
    (out / "cifs").mkdir(parents=True, exist_ok=True)
    for stem, src in cifs.items():
        shutil.copy(src, out / "cifs" / f"{stem}.cif")
    (out / "_iso_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}/_iso_meta.json + {len(cifs)} CIFs")
    for k in meta["ORDER"]:
        s = next(s for s in meta["STARS"] if s[0] == k)
        print(f"  {k:12s} sg={meta['META'][k]['sgno']:3d}  E_g(DFT)={meta['PROPS'][k]['eg']:>5s} eV"
              f"  E_hull={meta['PROPS'][k]['ehull']:>6s} meV/at"
              f"  star=({s[1]}, {s[2]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
