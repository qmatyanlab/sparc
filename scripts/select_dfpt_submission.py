#!/usr/bin/env python3
"""Build the DFPT submission list for a dielectric RL run, with an auditable filter.

Two things this does that a shell one-liner cannot:

1. **Drops molecular / van-der-Waals solids.** A composition made only of halogens, H, N
   or O has no metal or metalloid to carry an ionic/covalent lattice; these are
   transparent, so they score well on an in-plane-isotropy reward while being an
   Im(eps)~0 artifact rather than a dielectric material (the same family the consolidated
   DFT dataset already flags `caution_molecular_vdW`). Every drop is recorded with its
   reason, so the exclusions are on the record rather than implicit.

2. **Diffs against MongoDB**, because `dielectric_batch.py` has no resume: it dedups
   labels only *within* one invocation, so re-running a list that is already in the DB
   silently double-runs every job. On a contended cluster the campaign has to be
   submitted in chunks, which makes this mandatory, not optional.

Usage:
  uv run python scripts/select_dfpt_submission.py <run_dir> --campaign ipiso_sun
  uv run python scripts/select_dfpt_submission.py <run_dir> --campaign ipiso_sun --chunk 80
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A composition drawn only from these elements is a molecular / vdW solid here.
MOLECULAR_ELEMENTS = {"H", "N", "O", "F", "Cl", "Br", "I", "At"}
MONGO_URI = os.environ.get(
    "DFPT_MONGO_URI", "mongodb://angush:angus@127.0.0.1:12349/?authSource=atomate2_test"
)


def is_molecular(formula: str) -> bool:
    from pymatgen.core import Composition
    try:
        return {str(e) for e in Composition(formula).elements} <= MOLECULAR_ELEMENTS
    except Exception:  # noqa: BLE001 -- an unparseable formula is not a reason to drop
        return False


def labels_already_done(campaign: str, uri: str = MONGO_URI) -> set[str]:
    """Labels with an `MP GGA static` already stored under this campaign."""
    try:
        from pymongo import MongoClient
        col = MongoClient(uri, serverSelectionTimeoutMS=15000)["atomate2_test"]["outputs"]
        return {d["metadata"]["structure_label"] for d in col.find(
            {"metadata.campaign": campaign, "name": "MP GGA static"}, {"metadata": 1})}
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Could not reach MongoDB to check for already-done labels: {e}\n"
            f"The launcher has no resume, so submitting without this check risks "
            f"double-running jobs. Open the tunnel first:\n"
            f"    ssh -NfL 12349:localhost:27017 mongows2"
        ) from e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--campaign", required=True, help="candidate campaign tag (e.g. ipiso_sun)")
    ap.add_argument("--demo-campaign", default=None,
                    help="failure-demo campaign tag (default: <campaign>_failed_demo)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="emit only the first N not-yet-done candidates")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep-molecular", action="store_true")
    args = ap.parse_args(argv)

    run = args.run_dir.resolve()
    demo_campaign = args.demo_campaign or f"{args.campaign}_failed_demo"
    out = (args.out or run / "dfpt_submission").resolve()
    out.mkdir(parents=True, exist_ok=True)

    cand_dir = run / "deliverables_dfpt_candidates"
    demo_dir = run / "failed_reward_demo"
    summary = cand_dir / "_summary.csv"
    for p in (summary, demo_dir):
        if not p.exists():
            sys.exit(f"missing {p}\nRun select_isotropy_dfpt_candidates.py / "
                     f"export_failed_reward_demo.py first.")

    done_cand = labels_already_done(args.campaign)
    done_demo = labels_already_done(demo_campaign)
    print(f"already in Mongo: {len(done_cand)} under '{args.campaign}', "
          f"{len(done_demo)} under '{demo_campaign}'")

    rows, keep = [], []
    with open(summary, newline="") as fh:
        for r in csv.DictReader(fh):
            cif = cand_dir / r["cif"]
            label = cif.stem
            if not args.keep_molecular and is_molecular(r["formula"]):
                reason, sel = "molecular/vdW solid (halogen/H/N/O only)", False
            elif label in done_cand:
                reason, sel = f"already done under campaign '{args.campaign}'", False
            elif not cif.exists():
                reason, sel = "CIF missing on disk", False
            else:
                reason, sel = "selected", True
            rows.append({"set": "candidate", "label": label, "rank": r["rank"],
                         "formula": r["formula"], "spacegroup": r["spacegroup"],
                         "sg_no": r["sg_no"], "r_uni": r["r_uni"],
                         "band_gap_ev": r["band_gap_ev"], "ehull_ev_atom": r["ehull_ev_atom"],
                         "selected": sel, "reason": reason, "cif": str(cif)})
            if sel:
                keep.append(str(cif))

    demo_keep = []
    for cif in sorted(demo_dir.glob("*.cif")):
        label = cif.stem
        sel = label not in done_demo
        rows.append({"set": "failed_demo", "label": label, "rank": "", "formula": "",
                     "spacegroup": "", "sg_no": "", "r_uni": "", "band_gap_ev": "",
                     "ehull_ev_atom": "", "selected": sel,
                     "reason": "selected" if sel else
                               f"already done under campaign '{demo_campaign}'",
                     "cif": str(cif)})
        if sel:
            demo_keep.append(str(cif))

    if args.chunk:
        deferred = keep[args.chunk:]
        keep = keep[:args.chunk]
        for r in rows:
            if r["cif"] in set(deferred):
                r["selected"], r["reason"] = False, f"deferred past --chunk {args.chunk}"

    (out / "candidates.list").write_text("\n".join(keep) + ("\n" if keep else ""))
    (out / "failed_demo.list").write_text("\n".join(demo_keep) + ("\n" if demo_keep else ""))
    cols = list(rows[0])
    with open(out / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    import collections
    dropped = collections.Counter(r["reason"] for r in rows if not r["selected"])
    n_cand_rows = sum(1 for r in rows if r["set"] == "candidate")
    print(f"\ncandidates : {n_cand_rows} total -> {len(keep)} to submit")
    for reason, n in dropped.most_common():
        print(f"   dropped {n:4d}: {reason}")
    print(f"demos      : {len(demo_keep)} to submit")
    print(f"TOTAL      : {len(keep) + len(demo_keep)} structures")
    print(f"\nwrote {out}/{{candidates.list, failed_demo.list, MANIFEST.csv}}")

    missing = [p for p in keep + demo_keep if not Path(p).is_file()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} selected paths do not exist, e.g. {missing[:3]}")
    print("all selected paths exist and are readable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
