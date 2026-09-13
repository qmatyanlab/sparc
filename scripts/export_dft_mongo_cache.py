#!/usr/bin/env python3
"""STAGE 1 of the DFT dataset build: dump every VASP DFT result out of MongoDB.

This is the ONLY step that needs the atomate2 environment (pymatgen 2026.5.4 +
emmet-core, which is what the stored TaskDocs were written with) and the SSH tunnel to
the Mongo host. It writes a **plain-JSON cache** -- no pymatgen classes, no TaskDocs --
so that stage 2 (``scripts/build_dft_bundle.py``, sparc_v1 venv) can join everything
offline, with no database and no version-drift risk.

Everything that needs pymatgen-2026 happens HERE: structure deserialization, CIF
writing, and spacegroup analysis. Stage 2 only ever touches the CIF strings and the
scalars emitted below -- a Structure is never round-tripped across the venv boundary,
because site ``properties`` keys differ between pymatgen 2024 and 2026.

Two collections are read:
  outputs            -- the current campaigns (234 labels: slme / dielectric / failed_demo)
  dielectric_outputs -- the legacy NERSC pilot (87 candidates, 42 with a full `summarize`)

Usage:
  ssh -NfL 12349:localhost:27017 mongows2          # if the tunnel is down
  ~/atomate2/.venv/bin/python scripts/export_dft_mongo_cache.py
  ~/atomate2/.venv/bin/python scripts/export_dft_mongo_cache.py --out /tmp/x.json.gz --limit 5
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import os
import socket
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URI = os.environ.get(
    "DFPT_MONGO_URI", "mongodb://angush:angus@127.0.0.1:12349/?authSource=atomate2_test"
)
DEFAULT_DB = "atomate2_test"
DEFAULT_OUT = ROOT / "dft_dataset" / "_cache" / "mongo_extract.json.gz"

SCHEMA_VERSION = "1.0"

# Which campaign a label belongs to -> the dataset split.
CAMPAIGN_TO_SPLIT = {
    "cand_ehull_all": "slme",
    "cand_ehull": "slme",
    "pbe": "dielectric",
    "failed_demo": "failed_demo",
}
# Higher wins during dedup. Un-named (pilot) docs lose to every named campaign.
CAMPAIGN_RANK = {"cand_ehull_all": 3, "pbe": 3, "failed_demo": 3, "cand_ehull": 1, None: 0}

STATIC_NAMES = ("MP GGA static", "static")
RELAX_NAMES = ("MP GGA relax 2", "MP GGA relax 1", "relax")
DIELECTRIC_NAME = "dielectric"
GATE_NAME = "run_dielectric_if_gapped"


# --------------------------------------------------------------------------- helpers
def _jsonable(o):
    """datetime -> isoformat; everything else passed through (recursively)."""
    if isinstance(o, _dt.datetime):
        return o.isoformat()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _find(o, key, acc):
    """Recursively collect every value stored under ``key`` (atomate2 report_all idiom)."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                acc.append(v)
            else:
                _find(v, key, acc)
    elif isinstance(o, list):
        for v in o:
            _find(v, key, acc)
    return acc


def _first(o, key, default=None):
    vals = [v for v in _find(o, key, []) if v is not None]
    return vals[0] if vals else default


def _calc0(doc):
    """calcs_reversed[0].output -- where the per-calculation physics lives."""
    cr = ((doc or {}).get("output") or {}).get("calcs_reversed") or []
    return (cr[0].get("output") or {}) if cr else {}


def _check_tunnel(uri: str) -> None:
    if "127.0.0.1" not in uri and "localhost" not in uri:
        return
    # strip scheme first, then credentials: "mongodb://localhost:27017/" has no "@", and
    # naively splitting on "/" would leave "mongodb:" and int("") would raise.
    hostpart = uri.split("://", 1)[-1].split("@")[-1].split("/")[0]
    port = 27017
    for part in hostpart.split(","):
        host, _, p_str = part.rpartition(":")
        if host and p_str.isdigit():
            port = int(p_str)
    s = socket.socket()
    s.settimeout(3)
    try:
        s.connect(("127.0.0.1", port))
    except OSError:
        sys.exit(
            f"Cannot reach MongoDB on 127.0.0.1:{port}.\n"
            f"Open the tunnel first:\n    ssh -NfL {port}:localhost:27017 mongows2"
        )
    finally:
        s.close()


# --------------------------------------------------------------------- structure work
def _structure_payload(struct_dict, symprec=0.1):
    """pymatgen dict -> {pymatgen_dict, cif_p1, cif_symmetrized, lattice, spacegroup, ...}.

    All pymatgen-2026 work is confined here so stage 2 never deserializes a Structure.
    """
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    st = Structure.from_dict(struct_dict)
    lat = st.lattice
    out = {
        "pymatgen_dict": struct_dict,
        "cif_p1": st.to(fmt="cif"),
        "cif_symmetrized": None,
        "nsites": len(st),
        "density_g_cm3": float(st.density),
        "formula_pretty": st.composition.reduced_formula,
        "formula_full": st.composition.formula,
        "chemsys": "-".join(sorted({str(e) for e in st.composition.elements})),
        "nelements": len(st.composition.elements),
        "composition": {str(k): float(v) for k, v in st.composition.as_dict().items()},
        "lattice": {
            "a": float(lat.a), "b": float(lat.b), "c": float(lat.c),
            "alpha": float(lat.alpha), "beta": float(lat.beta), "gamma": float(lat.gamma),
            "volume_A3": float(lat.volume),
        },
        "spacegroup": None,
    }
    try:
        sga = SpacegroupAnalyzer(st, symprec=symprec)
        out["spacegroup"] = {
            "symbol": sga.get_space_group_symbol(),
            "number": int(sga.get_space_group_number()),
            "crystal_system": sga.get_crystal_system(),
            "point_group": sga.get_point_group_symbol(),
            "hall": sga.get_hall(),
            "symprec": symprec,
        }
        out["cif_symmetrized"] = st.to(fmt="cif", symprec=symprec)
    except Exception as exc:  # noqa: BLE001 -- symmetry finding is allowed to fail
        out["spacegroup"] = {"symbol": None, "number": None, "crystal_system": None,
                             "point_group": None, "hall": None, "symprec": symprec,
                             "error": str(exc)}
    return out


# ------------------------------------------------------------------------- dedup
def _pick(docs, names):
    """Choose the winning doc among ``docs`` whose name is in ``names``.

    Precedence: named target campaign > older named pilot > un-named pilot, then the
    latest ``completed_at``. Returns (kept, [dropped...]).
    """
    order = {n: i for i, n in enumerate(names)}
    cands = [d for d in docs if d.get("name") in order]
    cands = [d for d in cands if ((d.get("output") or {}).get("state") in (None, "successful"))] or cands
    if not cands:
        return None, []

    def key(d):
        camp = (d.get("metadata") or {}).get("campaign")
        ts = (d.get("output") or {}).get("completed_at") or d.get("completed_at") or ""
        return (-order[d["name"]], CAMPAIGN_RANK.get(camp, 0), str(ts), str(d.get("_id")))

    cands.sort(key=key, reverse=True)
    return cands[0], cands[1:]


def _physics(doc):
    """(energy_per_atom, bandgap) used to prove a dedup discard changed nothing."""
    o = ((doc or {}).get("output") or {}).get("output") or {}
    return o.get("energy_per_atom"), o.get("bandgap")


# How comparable a discarded doc is to the one we kept. Only ``same_flow`` pairs are
# expected to agree numerically; the other two ran a different functional or started
# from a different relaxed geometry, so a delta there is physics, not an error.
COMPARABILITY = {
    "cross_flow": "different flow (MP GGA static vs the older PBEsol pilot flow) -- deltas expected",
    "rerun_different_geometry": "same job type, earlier campaign, different input relaxation -- deltas expected",
    "same_flow": "same job type and campaign -- values must agree",
}


def _comparability(kept, dropped):
    if kept.get("name") != dropped.get("name"):
        return "cross_flow"
    kc = (kept.get("metadata") or {}).get("campaign")
    dc = (dropped.get("metadata") or {}).get("campaign")
    return "same_flow" if kc == dc else "rerun_different_geometry"


def _drop_rows(mat_id, kind, kept, dropped):
    rows = []
    kep, kbg = _physics(kept)
    for d in dropped:
        dep, dbg = _physics(d)
        de = None if (kep is None or dep is None) else abs(kep - dep)
        dg = None if (kbg is None or dbg is None) else abs(kbg - dbg)
        comp = _comparability(kept, d)
        rows.append({
            "mat_id": mat_id, "kind": kind,
            "kept_uuid": kept.get("uuid"), "dropped_uuid": d.get("uuid"),
            "kept_campaign": (kept.get("metadata") or {}).get("campaign"),
            "dropped_campaign": (d.get("metadata") or {}).get("campaign"),
            "kept_name": kept.get("name"), "dropped_name": d.get("name"),
            "delta_energy_per_atom_ev": de, "delta_bandgap_ev": dg,
            "comparability": comp,
            "reason": COMPARABILITY[comp],
        })
    return rows


# ---------------------------------------------------------------- current campaigns
def extract_outputs(col, limit=None, verbose=True):
    """`outputs` collection -> {mat_id: record}, plus the dedup log."""
    by_label = defaultdict(list)
    for d in col.find({"metadata.structure_label": {"$exists": True}}):
        by_label[d["metadata"]["structure_label"]].append(d)

    labels = sorted(by_label)
    if limit:
        labels = labels[:limit]

    records, dedup_log = {}, []
    for i, lbl in enumerate(labels, 1):
        docs = by_label[lbl]
        static, drop_s = _pick(docs, STATIC_NAMES)
        relax, drop_r = _pick(docs, RELAX_NAMES)
        diel, drop_d = _pick(docs, [DIELECTRIC_NAME])
        gate, _ = _pick(docs, [GATE_NAME])
        if static is None:
            continue

        dedup_log += _drop_rows(lbl, "static", static, drop_s)
        if diel is not None:
            dedup_log += _drop_rows(lbl, "dielectric", diel, drop_d)

        so = static["output"]
        sout = so.get("output") or {}
        scalc = _calc0(static)
        campaign = (static.get("metadata") or {}).get("campaign")

        rec = {
            "mat_id": lbl,
            "source_collection": "outputs",
            "mongo_campaign": campaign,
            "split": CAMPAIGN_TO_SPLIT.get(campaign, "unknown"),
            "structure": _structure_payload(so["structure"]),
            "energetics": {
                "energy_ev": sout.get("energy"),
                "energy_per_atom_ev": sout.get("energy_per_atom"),
                "forces": sout.get("forces"),
                "stress": sout.get("stress"),
                "state": so.get("state"),
                "volume_A3": so.get("volume"),
                "density_g_cm3": so.get("density"),
            },
            "electronic": {
                "bandgap_ev": sout.get("bandgap"),
                "direct_gap_ev": scalc.get("direct_gap"),
                "is_gap_direct": scalc.get("is_gap_direct"),
                "is_metal": scalc.get("is_metal"),
                "vbm_ev": scalc.get("vbm"),
                "cbm_ev": scalc.get("cbm"),
                "efermi_ev": scalc.get("efermi"),
            },
            "entry": so.get("entry"),
            "mongo_symmetry": so.get("symmetry"),
            "formula_pretty_vasp": so.get("formula_pretty"),
            "chemsys_vasp": so.get("chemsys"),
            "dfpt": None,
            "gate": None,
            "provenance": {
                "static_run_type": so.get("run_type"),
                "static_task_type": so.get("task_type"),
                "static_calc_type": so.get("calc_type"),
                "vasp_version": so.get("vasp_version"),
                "encut": _first(so.get("input") or {}, "ENCUT"),
                "is_hubbard": (so.get("input") or {}).get("is_hubbard"),
                "dir_name_static": so.get("dir_name"),
                "dir_name_relax": (relax or {}).get("output", {}).get("dir_name"),
                "completed_at_static": _jsonable(so.get("completed_at")),
                "doc_uuid_static": static.get("uuid"),
                "doc_uuid_relax": (relax or {}).get("uuid"),
                "doc_uuid_dielectric": None,
                "dielectric_run_type": None,
                "dir_name_dielectric": None,
                "completed_at_dielectric": None,
                "machine": (so.get("dir_name") or "").split(":")[0] or None,
            },
        }

        if gate is not None:
            go = gate.get("output")
            if isinstance(go, dict):
                rec["gate"] = {"skipped": bool(go.get("skipped")), "reason": go.get("reason"),
                               "bandgap": go.get("bandgap"), "gap_tol": go.get("gap_tol")}
            else:
                rec["gate"] = {"skipped": False, "reason": "dielectric job ran",
                               "bandgap": None, "gap_tol": None}

        if diel is not None:
            do = diel["output"]
            dcalc = _calc0(diel)
            dprops = dcalc.get("dielectric_properties") or {}
            rec["dfpt"] = {
                "eps_electronic_tensor": dcalc.get("epsilon_static"),
                "eps_ionic_tensor": dcalc.get("epsilon_ionic"),
                "eps_static_wolfe": dcalc.get("epsilon_static_wolfe"),
                "born_charges": dprops.get("born_charges") or _first(dcalc, "born"),
                "normalmode_eigenvals": dcalc.get("normalmode_eigenvals"),
                "bandgap_ev": (do.get("output") or {}).get("bandgap"),
                "energy_ev": (do.get("output") or {}).get("energy"),
            }
            rec["provenance"].update({
                "dielectric_run_type": do.get("run_type"),
                "dielectric_task_type": do.get("task_type"),
                "dielectric_calc_type": do.get("calc_type"),
                "dir_name_dielectric": do.get("dir_name"),
                "completed_at_dielectric": _jsonable(do.get("completed_at")),
                "doc_uuid_dielectric": diel.get("uuid"),
            })

        records[lbl] = rec
        if verbose and i % 25 == 0:
            print(f"  outputs: {i}/{len(labels)}", flush=True)
    return records, dedup_log


# ------------------------------------------------------------------- legacy pilot
def extract_legacy(col, limit=None, verbose=True):
    """`dielectric_outputs` -> {mat_id: record}. Complete == has a `summarize` doc."""
    by_cand = defaultdict(dict)
    for d in col.find({"metadata.candidate": {"$exists": True}}):
        by_cand[d["metadata"]["candidate"]].setdefault(d["name"], []).append(d)

    cands = sorted(by_cand)
    if limit:
        cands = cands[:limit]

    records, incomplete = {}, []
    for i, cand in enumerate(cands, 1):
        jobs = by_cand[cand]
        summ = (jobs.get("summarize") or [None])[0]
        if summ is None:
            incomplete.append({
                "mat_id": cand, "split": "legacy_pilot",
                "reason": "no `summarize` job -- relax only, never reached static/e_hull",
                "jobs_present": ",".join(sorted(jobs)),
            })
            continue

        so = summ["output"]
        ehull_doc = (jobs.get("compute_e_above_hull") or [None])[0]
        eh_alt = (ehull_doc or {}).get("output") or {}
        thermo = so.get("thermo") or {}
        diel = so.get("dielectric") or {}

        rec = {
            "mat_id": cand,
            "source_collection": "dielectric_outputs",
            "mongo_campaign": (summ.get("metadata") or {}).get("project"),
            "split": "legacy_pilot",
            "structure": _structure_payload(so["relaxed_structure"]),
            "energetics": {
                "energy_ev": so.get("static_energy"),
                "energy_per_atom_ev": eh_alt.get("energy_per_atom"),
                "forces": None, "stress": None, "state": "successful",
                "volume_A3": None, "density_g_cm3": None,
            },
            "electronic": {
                "bandgap_ev": so.get("bandgap"),
                "direct_gap_ev": None, "is_gap_direct": None,
                "is_metal": so.get("is_metal"),
                "vbm_ev": None, "cbm_ev": None, "efermi_ev": None,
            },
            "entry": None,
            "mongo_symmetry": None,
            "formula_pretty_vasp": so.get("formula"),
            "chemsys_vasp": thermo.get("chemsys"),
            "legacy_thermo": {
                "e_above_hull_ev_atom": thermo.get("e_above_hull"),
                "formation_energy_ev_atom": thermo.get("formation_energy_per_atom"),
                "is_stable": thermo.get("is_stable"),
                "decomposition": thermo.get("decomposition"),
                # the sibling job; it disagrees for 2 metallic Nb systems -> kept for the audit
                "e_above_hull_ev_atom_altjob": eh_alt.get("e_above_hull"),
                "formation_energy_ev_atom_altjob": eh_alt.get("formation_energy_per_atom"),
                "n_mp_reference_entries": eh_alt.get("n_mp_reference_entries"),
            },
            "dfpt": None,
            "gate": {"skipped": bool(so.get("is_metal")),
                     "reason": diel.get("note") or ("metallic" if so.get("is_metal") else None),
                     "bandgap": so.get("bandgap"), "gap_tol": None},
            "provenance": {
                "static_run_type": "GGA", "static_task_type": "Static",
                "static_calc_type": None, "vasp_version": None, "encut": None,
                "is_hubbard": None,
                "dir_name_static": so.get("static_dir"),
                "dir_name_relax": None,
                "completed_at_static": _jsonable(summ.get("completed_at")),
                "doc_uuid_static": summ.get("uuid"), "doc_uuid_relax": None,
                "doc_uuid_dielectric": None,
                "dielectric_run_type": None, "dir_name_dielectric": None,
                "completed_at_dielectric": None,
                "machine": (so.get("static_dir") or "").split(":")[0] or None,
            },
        }
        if diel.get("epsilon_electronic"):
            xd = (jobs.get("extract_dielectric") or [None])[0]
            xo = (xd or {}).get("output") or {}
            rec["dfpt"] = {
                "eps_electronic_tensor": diel.get("epsilon_electronic"),
                "eps_ionic_tensor": diel.get("epsilon_ionic"),
                "eps_total_tensor": diel.get("epsilon_total"),
                "eps_static_wolfe": None, "born_charges": None,
                "normalmode_eigenvals": None,
                "bandgap_ev": diel.get("bandgap"), "energy_ev": None,
            }
            rec["provenance"].update({
                "dielectric_run_type": "PBEsol",
                "dir_name_dielectric": xo.get("dielectric_dir"),
                "doc_uuid_dielectric": (xd or {}).get("uuid"),
            })
        records[cand] = rec
        if verbose and i % 25 == 0:
            print(f"  legacy: {i}/{len(cands)}", flush=True)
    return records, incomplete


# ------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=DEFAULT_URI, help="MongoDB URI (default: the db2 tunnel)")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="only the first N materials per collection")
    ap.add_argument("--skip-legacy", action="store_true")
    args = ap.parse_args(argv)

    _check_tunnel(args.uri)
    from pymongo import MongoClient

    db = MongoClient(args.uri, serverSelectionTimeoutMS=20000)[args.db]

    print(f"Reading {args.db}.outputs ...", flush=True)
    records, dedup_log = extract_outputs(db["outputs"], args.limit)
    print(f"  -> {len(records)} materials, {len(dedup_log)} duplicate docs dropped")

    incomplete = []
    if not args.skip_legacy:
        print(f"Reading {args.db}.dielectric_outputs (legacy pilot) ...", flush=True)
        legacy, incomplete = extract_legacy(db["dielectric_outputs"], args.limit)
        overlap = set(legacy) & set(records)
        if overlap:
            print(f"  !! {len(overlap)} legacy ids collide with current ids: {sorted(overlap)[:5]}")
        records.update({k: v for k, v in legacy.items() if k not in records})
        print(f"  -> {len(legacy)} complete, {len(incomplete)} relax-only (excluded)")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now().astimezone().isoformat(),
        "source": {"db": args.db, "collections": ["outputs", "dielectric_outputs"],
                   "uri_host": args.uri.split("@")[-1].split("/")[0]},
        "n_materials": len(records),
        "dedup_log": dedup_log,
        "incomplete": incomplete,
        "records": records,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(_jsonable(payload), separators=(",", ":")).encode()
    payload_sha = hashlib.sha256(blob).hexdigest()
    with gzip.open(out, "wb") as fh:
        fh.write(blob)
    (out.parent / "mongo_extract.sha256").write_text(payload_sha + "\n")

    counts = defaultdict(int)
    for r in records.values():
        counts[r["split"]] += 1
    print(f"\nWrote {out}  ({out.stat().st_size / 1e6:.1f} MB gz, sha256 {payload_sha[:12]})")
    for k in sorted(counts):
        print(f"  {k:14s} {counts[k]:4d}")
    print(f"  {'TOTAL':14s} {len(records):4d}    (+{len(incomplete)} excluded relax-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
