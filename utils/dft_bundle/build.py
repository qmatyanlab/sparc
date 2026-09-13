"""Merge every source into one record per material, then write the bundle.

``merge_records`` applies the authority rules from ``sources``: MongoDB wins for VASP
quantities, qe_ht wins for QE/SLME/hyperbolic, the RL summaries win for surrogate and
lineage.  Repeated values from a non-authoritative source are parked under
``_crosscheck`` for ``verify.py`` instead of being merged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from . import sources as S
from .ids import build_alias_map, parse_label, resolve, sanitize
from .schema import (
    EHULL_MAX, GAP_WINDOW, MAGNETIC_TM, MOLECULAR_VDW_FORMULAS, SCHEMA_VERSION,
    SLME_MIN, STATUS_BLOCKS, FLAT_FIELDS, Status, flatten_record, new_record,
)

_ROUND = 6

# |lambda| below this counts as a near-zero mode: the three acoustic modes at Gamma,
# plus any genuinely soft mode (molecular crystals such as Cl2/F2 have many).
_IMAG_TOL = 1.0


def _r(x, n=_ROUND):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, n)
    return x


def _finite_tensor(t):
    """True only if every component of a 3x3 is a finite number."""
    if not t:
        return False
    try:
        return all(v is not None and math.isfinite(float(v)) for row in t for v in row)
    except (TypeError, ValueError):
        return False


def _grids_agree(a, b, tol=1e-6):
    """Two energy grids are interchangeable: same length and same sample points.

    ``a`` arrives as a numpy array from the raw .dat reader, so length is tested
    explicitly rather than by truthiness.
    """
    if a is None or b is None or len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def _eig3(t):
    """Symmetric 3x3 -> sorted eigenvalues. None when the tensor is not finite."""
    if not _finite_tensor(t):
        return None
    import numpy as np

    a = np.array(t, dtype=float)
    a = (a + a.T) / 2.0
    try:
        return sorted(float(v) for v in np.linalg.eigvalsh(a))
    except np.linalg.LinAlgError:
        return None


def _iso(t):
    if not _finite_tensor(t):
        return None
    return float((t[0][0] + t[1][1] + t[2][2]) / 3.0)


# --------------------------------------------------------------------------- merging
def merge_records(cache_path, qe_root=S.DEFAULT_QE_ROOT, exp_res=S.DEFAULT_EXP_RES,
                  repo_root=None, verbose=True):
    """-> (records, extras) where extras carries the dedup log, exclusions and metadata."""
    mongo, dedup_log, incomplete, meta = S.load_mongo_cache(cache_path)
    alias = build_alias_map(mongo)                      # asserts injectivity

    slme_ehull = S.load_slme_ehull(qe_root)
    dfpt_res = S.load_dfpt_results(exp_res)
    qe_slme = S.load_qe_database(Path(qe_root) / "database")
    qe_dfpt = S.load_qe_database(Path(qe_root) / "database_dfpt")
    rl_slme = S.load_rl_worklist_slme(exp_res)
    rl_diel = S.load_rl_summary_dielectric(exp_res)
    rl_demo = S.load_rl_summary_failed_demo(exp_res)
    legacy_cifs = S.load_legacy_as_generated(repo_root)
    runs, resolve_stability, _ = S.load_rl_stability(exp_res)

    # dfpt_dft_results.csv ids are canonical already, but its 104th row (the
    # blue-phosphorene reference) lives in the legacy collection under another name.
    dfpt_res_by_canon = {}
    for lbl, v in dfpt_res.items():
        canon = lbl if lbl in mongo else resolve(sanitize(lbl), alias)
        dfpt_res_by_canon[canon or lbl] = v

    records = {}
    for mat_id, m in mongo.items():
        rec = new_record(mat_id)
        split = m["split"]
        lbl = parse_label(mat_id)
        st = m["structure"]

        # ---------------------------------------------------------------- identity
        rec["identity"].update({
            "mat_id_fs": sanitize(mat_id),
            "split": split,
            "mongo_campaign": m.get("mongo_campaign"),
            "formula_pretty": st.get("formula_pretty") or m.get("formula_pretty_vasp"),
            "formula_full": st.get("formula_full"),
            "chemsys": st.get("chemsys") or m.get("chemsys_vasp"),
            "nsites": st.get("nsites"),
            "nelements": st.get("nelements"),
            "composition": st.get("composition"),
            "group": "candidate",
            **lbl,
        })

        # ---------------------------------------------------------------- lineage
        prov, rlp = m["provenance"], S.rl_provenance(split)
        rec["lineage"].update({
            "rl_run_name": rlp["rl_run_name"], "rl_checkpoint": rlp["rl_checkpoint"],
            "rl_sample_mode": rlp["rl_sample_mode"], "rl_generator": rlp["rl_generator"],
            "rl_step": lbl["label_step"], "rl_eval_index": lbl["label_index"],
            "rl_cif_as_generated": None,
            "static_run_type": prov.get("static_run_type"),
            "static_task_type": prov.get("static_task_type"),
            "dielectric_run_type": prov.get("dielectric_run_type"),
            "dielectric_task_type": prov.get("dielectric_task_type"),
            "vasp_version": prov.get("vasp_version"), "encut": prov.get("encut"),
            "is_hubbard": prov.get("is_hubbard"), "machine": prov.get("machine"),
            "dir_name_static": prov.get("dir_name_static"),
            "dir_name_dielectric": prov.get("dir_name_dielectric"),
            "completed_at_static": prov.get("completed_at_static"),
            "completed_at_dielectric": prov.get("completed_at_dielectric"),
            "doc_uuid_static": prov.get("doc_uuid_static"),
            "doc_uuid_relax": prov.get("doc_uuid_relax"),
            "doc_uuid_dielectric": prov.get("doc_uuid_dielectric"),
            "source_collection": m.get("source_collection"),
        })

        # ------------------------------------------------------- vasp_structure
        sg = st.get("spacegroup") or {}
        lat = st.get("lattice") or {}
        rec["vasp_structure"].update({
            "status": Status.OK,
            "status_reason": "VASP MP-GGA relaxed geometry from the MP GGA static task",
            "spacegroup_symbol": sg.get("symbol"), "spacegroup_number": sg.get("number"),
            "crystal_system": sg.get("crystal_system"), "point_group": sg.get("point_group"),
            "hall": sg.get("hall"), "symprec": sg.get("symprec"),
            **{k: _r(v) for k, v in lat.items()},
            "density_g_cm3": _r(st.get("density_g_cm3")),
            "nsites": st.get("nsites"),
            "structure_dict": st.get("pymatgen_dict"),
            "cif_p1_text": st.get("cif_p1"),
            "cif_symmetrized_text": st.get("cif_symmetrized"),
            "geometry": "VASP MP-PBE relaxed",
        })

        # ------------------------------------------------------ vasp_energetics
        en = m["energetics"]
        rec["vasp_energetics"].update({
            "status": Status.OK, "status_reason": "MP GGA static",
            "energy_ev": _r(en.get("energy_ev")),
            "energy_per_atom_ev": _r(en.get("energy_per_atom_ev")),
            "forces": _r(en.get("forces")), "stress": _r(en.get("stress")),
            "state": en.get("state"),
            "e_hull_ev_atom": None, "formation_energy_ev_atom": None,
            "e_hull_source": None, "e_hull_approx": False,
        })
        f = en.get("forces")
        if f:
            rec["vasp_energetics"]["max_force_ev_ang"] = _r(
                max(math.sqrt(sum(c * c for c in v)) for v in f))

        # ------------------------------------------------------ vasp_electronic
        el = m["electronic"]
        rec["vasp_electronic"].update({
            "status": Status.OK, "status_reason": "MP GGA static",
            **{k: _r(v) if isinstance(v, float) else v for k, v in el.items()},
        })

        # ------------------------------------------------------------ vasp_dfpt
        d = m.get("dfpt") or {}
        te, ti = d.get("eps_electronic_tensor"), d.get("eps_ionic_tensor")
        if te:
            tt = d.get("eps_total_tensor")
            if tt is None and ti:
                tt = [[te[i][j] + ti[i][j] for j in range(3)] for i in range(3)]
            eig = _eig3(te)
            nm = d.get("normalmode_eigenvals") or []
            finite = _finite_tensor(te)
            rec["vasp_dfpt"].update({
                "status": Status.OK if finite else Status.FAILED,
                "status_reason": (
                    "VASP DFPT (LEPSILON, IBRION=8), PBEsol" if finite else
                    "VASP DFPT returned a non-finite dielectric tensor (NaN components) -- "
                    "the linear response did not converge for this near-metallic cell"),
                "eps_electronic_tensor": _r(te), "eps_ionic_tensor": _r(ti),
                "eps_total_tensor": _r(tt),
                "eps_electronic_xx": _r(te[0][0]), "eps_electronic_yy": _r(te[1][1]),
                "eps_electronic_zz": _r(te[2][2]), "eps_electronic_iso": _r(_iso(te)),
                "eps_ionic_iso": _r(_iso(ti)), "eps_total_iso": _r(_iso(tt)),
                "eps_electronic_eig_min": _r(eig[0]) if eig else None,
                "eps_electronic_eig_max": _r(eig[-1]) if eig else None,
                "eps_anisotropy": _r(eig[-1] / eig[0]) if eig and eig[0] else None,
                "born_charges": _r(d.get("born_charges"), 5),
                "normalmode_eigenvals": _r(nm, 4) or None,
                # VASP's DFPT normal-mode eigenvalues are Hessian eigenvalues, where
                # omega^2 ~ -lambda: NEGATIVE lambda is a real (stable) mode and POSITIVE
                # lambda is imaginary. The three acoustic modes sit at |lambda| ~ 0, so a
                # tolerance is required -- counting `lambda < 0` would call every stable
                # crystal dynamically unstable.
                "n_imaginary_modes": (sum(1 for v in nm if v > _IMAG_TOL) if nm else None),
                "n_near_zero_modes": (sum(1 for v in nm if abs(v) <= _IMAG_TOL) if nm else None),
                "bandgap_ev_pbesol": _r(d.get("bandgap_ev")),
            })
        else:
            gate = m.get("gate") or {}
            rec["vasp_dfpt"].update({
                "status": Status.NOT_APPLICABLE,
                "status_reason": (
                    "DFPT dielectric was not part of the SLME campaign"
                    if split == "slme" else
                    (gate.get("reason") or "band gap ~0 -- the gapped-gate skipped DFPT")),
            })

        records[mat_id] = rec

    if verbose:
        print(f"  merged {len(records)} Mongo records")

    _attach_ehull(records, mongo, slme_ehull, dfpt_res_by_canon)
    _attach_qe(records, alias, qe_slme, qe_dfpt)
    _attach_rl(records, alias, rl_slme, rl_diel, rl_demo, dfpt_res_by_canon,
               legacy_cifs, runs, resolve_stability)
    _derive_flags(records)

    extras = {"dedup_log": dedup_log, "incomplete": incomplete, "cache_meta": meta,
              "alias": alias}
    return records, extras


# ------------------------------------------------------------------------ attachment
def _attach_ehull(records, mongo, slme_ehull, dfpt_res):
    """E_hull comes from whichever published computation covers that split."""
    for mat_id, rec in records.items():
        e = rec["vasp_energetics"]
        split = rec["identity"]["split"]

        if split == "slme" and mat_id in slme_ehull:
            v = slme_ehull[mat_id]
            e.update({k: v[k] for k in
                      ("e_hull_ev_atom", "formation_energy_ev_atom", "e_hull_source")})
            rec.setdefault("_crosscheck", {})["slme_ehull"] = v["_crosscheck"]

        elif mat_id in dfpt_res:
            v = dfpt_res[mat_id]
            e.update({"e_hull_ev_atom": v["e_hull_ev_atom"],
                      "e_hull_approx": v["e_hull_approx"],
                      "e_hull_source": v["e_hull_source"]})
            if v.get("group"):
                rec["identity"]["group"] = v["group"]
            if v.get("failure_mode"):
                rec["identity"]["failure_mode"] = v["failure_mode"]
            rec.setdefault("_crosscheck", {})["dfpt_results"] = v["_crosscheck"]
            rec.setdefault("_ml_from_report", {}).update(v.get("_ml") or {})

        elif split == "legacy_pilot":
            t = (mongo[mat_id].get("legacy_thermo") or {})
            e.update({
                "e_hull_ev_atom": t.get("e_above_hull_ev_atom"),
                "formation_energy_ev_atom": t.get("formation_energy_ev_atom"),
                "e_hull_source": "legacy `summarize.thermo` "
                                 "(MaterialsProject2020Compatibility + MP hull, 2026-06/07, NERSC)",
                "is_stable_mp": t.get("is_stable"),
                "decomposition": t.get("decomposition"),
            })
            if mongo[mat_id]["energetics"].get("energy_per_atom_ev") is None:
                npa = rec["identity"]["nsites"]
                tot = e.get("energy_ev")
                if npa and tot is not None:
                    e["energy_per_atom_ev"] = _r(tot / npa)
            rec.setdefault("_crosscheck", {})["legacy_ehull_altjob"] = {
                "e_above_hull_ev_atom": t.get("e_above_hull_ev_atom_altjob"),
                "formation_energy_ev_atom": t.get("formation_energy_ev_atom_altjob"),
            }
            rec["identity"]["group"] = "legacy_pilot"

        if e.get("e_hull_ev_atom") is None:
            e["status"] = Status.PARTIAL
            e["status_reason"] = "MP GGA static energy present, but no published e_hull found"
        elif e.get("e_hull_approx"):
            e["status"] = Status.PARTIAL
            e["status_reason"] = ("MP GGA static + published e_hull; e_hull is APPROXIMATE "
                                  "(W_sv substituted for MP's W_pv POTCAR)")
        else:
            e["status_reason"] = "MP GGA static energy + published MP-referenced e_hull"


def _attach_qe(records, alias, qe_slme, qe_dfpt):
    """QE optical / SLME / hyperbolic. qe_ht is authoritative for all three."""
    for db in (qe_slme, qe_dfpt):
        for key, doc in db.items():
            mat_id = resolve(key, alias)
            if mat_id is None or mat_id not in records:
                continue                       # Si/GaAs/MgO benchmarks -- not campaign materials
            rec = records[mat_id]
            b = S.qe_blocks(doc)

            rec["qe_optical"].update(b["qe_optical"])
            status = b["qe_optical"]["status"]
            rec["qe_optical"]["status_reason"] = {
                "done": "QE PBE + SG15 ONCV, epsilon.x IPA on the VASP-relaxed cell",
                "metallic": "QE found no gap, so epsilon.x was not run",
                # keep this a one-line summary: the full stderr lives in `qe_optical.error`
                "failed": "QE run did not complete -- see the retained `error` string",
            }.get(status, "unknown QE status")

            if status == "done" and b["slme"]["eta"] is not None:
                rec["slme"].update(b["slme"])
                rec["slme"]["status"] = Status.OK
                rec["slme"]["status_reason"] = (
                    "pymatgen SLME at 0.3 um / 300 K; eps1 reconstructed from eps2 by "
                    "Kramers-Kronig (parity with the RL reward calculator)")
            elif status == "done":
                gap = rec["qe_optical"].get("indirect_gap_ev")
                rec["slme"]["status"] = Status.NOT_APPLICABLE
                rec["slme"]["status_reason"] = (
                    f"QE indirect gap is negative ({gap:.3f} eV -- semimetallic band overlap), "
                    "so the SLME integral is undefined"
                    if gap is not None and gap < 0 else
                    "spectrum present but SLME was not evaluated")
            else:
                rec["slme"]["status"] = Status.NOT_APPLICABLE
                rec["slme"]["status_reason"] = f"no absorption spectrum (QE status: {status})"

            if b["hyperbolic"]:
                rec["hyperbolic"].update(b["hyperbolic"])
                rec["hyperbolic"]["status"] = Status.OK
                rec["hyperbolic"]["status_reason"] = "interband IPA hyperbolicity screen"
            else:
                rec["hyperbolic"]["status"] = Status.NOT_APPLICABLE
                rec["hyperbolic"]["status_reason"] = (
                    "not screened: the cell-frame diagonal only equals the principal axes "
                    "for ortho/tet/trig/hex cells"
                    if status == "done" else f"no spectrum to screen (QE status: {status})")

            if b["ml_vs_dft"]:
                rec["qe_optical"]["ml_vs_dft"] = b["ml_vs_dft"]
            if b["ml_dielectric"]:
                rec["_ml_dielectric"] = b["ml_dielectric"]
            if b["surrogate"]:
                rec.setdefault("_surrogate_from_qe", {}).update(b["surrogate"])
            if b["dielectric_spectrum"]:
                rec["_qe_spectrum_avg"] = b["dielectric_spectrum"]
            rec.setdefault("_crosscheck", {})["qe_ht"] = b["_crosscheck"]
            if b["ehull"]:
                rec.setdefault("_crosscheck", {})["qe_ht_ehull"] = b["ehull"]

    for rec in records.values():
        if rec["qe_optical"].get("status") in (None, Status.MISSING):
            legacy = rec["identity"]["split"] == "legacy_pilot"
            rec["qe_optical"]["status"] = Status.NOT_APPLICABLE
            rec["qe_optical"]["status_reason"] = (
                "the QE optical pipeline post-dates the legacy NERSC pilot" if legacy else
                "VASP found no band gap, so the material was never submitted to QE")
            for blk, why in (("slme", "no QE optical run"),
                             ("hyperbolic", "no QE optical run")):
                rec[blk]["status"] = Status.NOT_APPLICABLE
                rec[blk]["status_reason"] = why


def _attach_rl(records, alias, rl_slme, rl_diel, rl_demo, dfpt_res, legacy_cifs,
               runs, resolve_stability):
    """Surrogate predictions, as-generated CIFs and the RL S.U.N. flags."""
    for db in (rl_slme, rl_diel, rl_demo):
        for key, payload in db.items():
            mat_id = resolve(key, alias)
            if mat_id is None or mat_id not in records:
                continue
            rec = records[mat_id]
            rec["rl_surrogate"].update(payload["surrogate"])
            for k, v in payload["lineage"].items():
                if v is not None or k not in rec["lineage"]:
                    rec["lineage"][k] = v
            for k, v in payload["identity"].items():
                if v is not None:
                    rec["identity"][k] = v
            # never let a later source blank a path an earlier one already resolved
            if payload["_as_generated_cif"] or not rec["lineage"].get("rl_cif_as_generated"):
                rec["lineage"]["rl_cif_as_generated"] = payload["_as_generated_cif"]

    for mat_id, rec in records.items():
        sur = rec["rl_surrogate"]

        # fall back to what the QE database / the DFT report carry
        for extra in (rec.pop("_surrogate_from_qe", {}), rec.pop("_ml_from_report", {})):
            for k, v in extra.items():
                if sur.get(k) is None and v is not None:
                    sur[k] = v
        if not sur.get("surrogate_target"):
            sur["surrogate_target"] = S.rl_provenance(rec["identity"]["split"])["rl_surrogate_target"]

        # legacy as-generated CIFs, matched on (formula, spacegroup number)
        if rec["identity"]["split"] == "legacy_pilot" and not rec["lineage"].get("rl_cif_as_generated"):
            sgno = rec["identity"].get("label_spacegroup")
            fml = S.formula_from_label(mat_id) or _legacy_formula(mat_id)
            for (cf, csg), path in legacy_cifs.items():
                if csg == sgno and _same_composition(cf, fml):
                    rec["lineage"]["rl_cif_as_generated"] = path
                    break

        # S.U.N. flags, verified against the reduced formula
        run = runs.get(rec["identity"]["split"])
        step, idx = rec["lineage"].get("rl_step"), rec["lineage"].get("rl_eval_index")
        if run and step is not None:
            sur.update(resolve_stability(run, step, idx, S.formula_from_label(mat_id),
                                         sur.get("rl_total_energy_ev")))
        else:
            sur.setdefault("stability_join_method", "no_rl_index")

        # `sur` is pre-populated by new_record() with status/status_reason, and
        # surrogate_target/stability_join_method are always filled in above -- none of
        # them counts as evidence that a surrogate record actually exists.
        _bookkeeping = {"status", "status_reason", "surrogate_target",
                        "stability_join_method", "expected_null_rule"}
        has_any = any(v is not None for k, v in sur.items() if k not in _bookkeeping)
        if not has_any:
            sur["status"] = Status.NOT_APPLICABLE
            sur["status_reason"] = "no surrogate/lineage record survives for this material"
        elif not rec["lineage"].get("rl_cif_as_generated"):
            sur["status"] = Status.PARTIAL
            sur["status_reason"] = ("surrogate values present, but the as-generated structure "
                                    "was not retained (legacy NERSC inputs)")
        elif sur.get("stability_join_method") in ("ambiguous", "no_stability_file", "no_rl_index"):
            sur["status"] = Status.PARTIAL
            sur["status_reason"] = (
                "surrogate values and as-generated structure present; the S.U.N. flags could "
                f"not be pinned to a unique RL sample ({sur['stability_join_method']}) and are "
                "left null rather than guessed")
        else:
            sur["status"] = Status.OK
            sur["status_reason"] = "RL surrogate prediction + as-generated structure + S.U.N. flags"


def _legacy_formula(mat_id):
    """``02_I_SG138_gap0.65_...`` / ``c1_Mg(TeS)2_SG187`` -> the formula token."""
    m = re.match(r"^(?:c\d+|\d+)_(?P<f>.+?)_SG\d+", mat_id)
    return m.group("f") if m else None


def _same_composition(a, b):
    if not a or not b:
        return False
    if a == b:
        return True
    from pymatgen.core import Composition
    try:
        ca, cb = Composition(a), Composition(b)
    except Exception:  # noqa: BLE001
        return False
    return (ca.reduced_formula == cb.reduced_formula
            or {str(e) for e in ca.elements} == {str(e) for e in cb.elements})


# ------------------------------------------------------------------- derived flags
def _derive_flags(records):
    for mat_id, rec in records.items():
        gap = rec["vasp_electronic"].get("bandgap_ev")
        eh = rec["vasp_energetics"].get("e_hull_ev_atom")
        eta = rec["slme"].get("eta")
        qe_gap = rec["qe_optical"].get("indirect_gap_ev")
        elements = set((rec["identity"].get("composition") or {}).keys())
        formula = rec["identity"].get("formula_pretty")

        in_win = None if gap is None else (GAP_WINDOW[0] <= gap <= GAP_WINDOW[1])
        # The published SLME campaign yield was quoted against the QE PBE gap, so keep a
        # QE-gap flavour of the same flag alongside the VASP-gap one. VASP is the headline
        # gap (QE is nspin=1 with no +U), but both are carried so the paper number is
        # reproducible from this table.
        in_win_qe = None if qe_gap is None else (GAP_WINDOW[0] <= qe_gap <= GAP_WINDOW[1])
        stable = None if eh is None else (eh <= EHULL_MAX)
        slme_ok = None if eta is None else (eta >= SLME_MIN)

        rec["flags"].update({
            "has_vasp_static": rec["vasp_energetics"]["status"] in (Status.OK, Status.PARTIAL),
            "has_vasp_dfpt": rec["vasp_dfpt"]["status"] == Status.OK,
            "has_qe_optics": rec["qe_optical"].get("status") == "done",
            "has_spectrum": False,                      # set once the spectra are written
            "has_rl_cif": bool(rec["lineage"].get("rl_cif_as_generated")),
            "is_gapless": None if gap is None else gap < 1e-6,
            "gap_in_window": in_win,
            "stable": stable,
            "slme_ok": slme_ok,
            "hits_solar_target": None if None in (in_win, stable, slme_ok)
                                 else bool(in_win and stable and slme_ok),
            "gap_in_window_qe": in_win_qe,
            "hits_solar_target_qe": None if None in (in_win_qe, stable, slme_ok)
                                    else bool(in_win_qe and stable and slme_ok),
            "qe_vs_vasp_gap_delta_ev": None if (gap is None or qe_gap is None)
                                       else _r(qe_gap - gap),
            # QE is nspin=1 with no +U, so it goes metallic on magnetic TM compounds that
            # VASP finds gapped. That is a method disagreement, not missing data.
            "qe_vasp_gap_disagreement": bool(
                rec["qe_optical"].get("status") == "metallic" and gap is not None and gap > 0.1),
            "caution_nspin1_magnetic_tm": bool(elements & MAGNETIC_TM),
            "caution_molecular_vdw": formula in MOLECULAR_VDW_FORMULAS,
        })

        expected = [b for b in STATUS_BLOCKS
                    if rec[b].get("status") in (Status.NOT_APPLICABLE, Status.PARTIAL)]
        rec["flags"]["expected_nulls"] = expected
        missing = [b for b in STATUS_BLOCKS if rec[b].get("status") == Status.MISSING]
        rec["flags"]["missing_blocks"] = missing

        if missing:
            cls = "incomplete"
        elif rec["vasp_dfpt"]["status"] == Status.OK and rec["qe_optical"].get("status") == "done":
            cls = "complete_dfpt_and_optical"
        elif rec["vasp_dfpt"]["status"] == Status.OK:
            cls = "complete_dfpt"
        elif rec["qe_optical"].get("status") == "done":
            cls = "complete_optical"
        else:
            cls = "complete_structure_energetics_only"
        rec["flags"]["completeness_class"] = cls


# --------------------------------------------------------------------- bundle writing
def write_bundle(records, extras, out_dir, qe_root=S.DEFAULT_QE_ROOT, write_spectra=True,
                 verbose=True):
    """Materialise the bundle. Returns a stats dict for the audit + dataset card."""
    out = Path(out_dir)
    for sub in ("structures/relaxed", "structures/as_generated", "tensors/dfpt", "audit"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    stats = {"n_records": len(records), "n_cif_relaxed": 0, "n_cif_as_generated": 0,
             "n_tensors": 0, "n_spectra": 0, "spectra_rows": 0, "n_ml_spectra": 0}

    # ---- structures + tensors ---------------------------------------------------
    for mat_id, rec in sorted(records.items()):
        fs = rec["identity"]["mat_id_fs"]
        cif = rec["vasp_structure"].pop("cif_p1_text", None)
        rec["vasp_structure"].pop("cif_symmetrized_text", None)
        if cif:
            (out / "structures/relaxed" / f"{fs}.cif").write_text(cif)
            rec["vasp_structure"]["cif_relaxed"] = f"structures/relaxed/{fs}.cif"
            stats["n_cif_relaxed"] += 1

        # idempotent: after a first pass ``rl_cif_as_generated`` holds the bundle-relative
        # path, so always re-resolve from the recorded absolute source.
        src = rec["lineage"].get("rl_cif_as_generated_source") or rec["lineage"].get("rl_cif_as_generated")
        if src and Path(src).is_absolute() and Path(src).exists():
            dst = out / "structures/as_generated" / f"{fs}.cif"
            dst.write_text(Path(src).read_text())
            rec["lineage"]["rl_cif_as_generated_source"] = src
            rec["lineage"]["rl_cif_as_generated"] = f"structures/as_generated/{fs}.cif"
            stats["n_cif_as_generated"] += 1
        else:
            rec["lineage"]["rl_cif_as_generated"] = None

        if rec["vasp_dfpt"].get("eps_electronic_tensor"):
            payload = {k: rec["vasp_dfpt"].get(k) for k in (
                "eps_electronic_tensor", "eps_ionic_tensor", "eps_total_tensor",
                "eps_electronic_eig_min", "eps_electronic_eig_max", "eps_anisotropy",
                "born_charges", "normalmode_eigenvals", "n_imaginary_modes",
                "status_reason")}
            payload["mat_id"] = mat_id
            payload["functional"] = rec["lineage"].get("dielectric_run_type")
            (out / "tensors/dfpt" / f"{fs}.json").write_text(json.dumps(payload, indent=1))
            rec["vasp_dfpt"]["tensor_file"] = f"tensors/dfpt/{fs}.json"
            stats["n_tensors"] += 1

    # ---- spectra ----------------------------------------------------------------
    if write_spectra:
        stats.update(_write_spectra(records, out, qe_root, verbose))

    # ---- flat table + nested records --------------------------------------------
    rows = [flatten_record(r) for _, r in sorted(records.items())]
    cols = [c for c, _ in FLAT_FIELDS]
    with open(out / "materials.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    with open(out / "materials.jsonl", "w") as fh:
        for _, rec in sorted(records.items()):
            slim = {k: v for k, v in rec.items() if not k.startswith("_")}
            fh.write(json.dumps(slim, separators=(",", ":")) + "\n")

    _write_csv(out / "audit/dedup_log.csv", extras["dedup_log"])
    _write_csv(out / "audit/excluded.csv", extras["incomplete"])
    if verbose:
        print(f"  wrote materials.csv ({len(rows)} rows x {len(cols)} cols) + materials.jsonl")
    return stats


def _write_spectra(records, out, qe_root, verbose=True):
    """Long-format eps1/eps2 per axis (from the raw .dat) and the ML surrogate spectra."""
    (out / "spectra").mkdir(parents=True, exist_ok=True)
    n_done = n_rows = n_ml = 0

    with open(out / "spectra/qe_optical.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mat_id", "energy_eV", "eps1_x", "eps1_y", "eps1_z",
                    "eps2_x", "eps2_y", "eps2_z", "eps2_avg"])
        for mat_id, rec in sorted(records.items()):
            if rec["qe_optical"].get("status") != "done":
                continue
            fs = rec["identity"]["mat_id_fs"]
            t = S.load_qe_spectrum(fs, qe_root)
            _spec = rec.pop("_qe_spectrum_avg", {}) or {}
            avg = _spec.get("eps2_avg") or []
            rec["_qe_spectrum_energies"] = _spec.get("energy_ev") or []
            if t is None:
                rec["qe_optical"]["spectrum_path"] = None
                rec["qe_optical"]["spectrum_status"] = "missing: no epsr/epsi found on disk"
                continue
            E, e1, e2 = t["energy_ev"], t["eps1_xyz"], t["eps2_xyz"]
            # eps2_avg comes from the qe_ht JSON, E from the raw .dat. They are the same
            # 0-20 eV / 2001-point grid by construction, but pairing them by index would
            # silently mis-label every energy if that ever stopped being true.
            if avg and not _grids_agree(E, rec["_qe_spectrum_energies"]):
                avg = []
                rec["qe_optical"]["eps2_avg_status"] = (
                    "dropped: the stored eps2_avg grid does not match the raw epsr/epsi grid")
            for i in range(len(E)):
                w.writerow([mat_id, round(float(E[i]), 4),
                            *[_r(float(e1[i][j]), 6) for j in range(3)],
                            *[_r(float(e2[i][j]), 6) for j in range(3)],
                            _r(float(avg[i]), 6) if i < len(avg) else None])
                n_rows += 1
            # eps1 at the lowest finite-valued energy point == the IPA eps_inf
            for j, ax in enumerate("xyz"):
                v = next((float(e1[i][j]) for i in range(len(E))
                          if not math.isnan(float(e1[i][j]))), None)
                rec["qe_optical"][f"eps1_zero_freq_{ax}"] = _r(v)
            rec["qe_optical"]["spectrum_path"] = "spectra/qe_optical.csv"
            rec["qe_optical"]["spectrum_status"] = "ok"
            rec["flags"]["has_spectrum"] = True
            n_done += 1
            if verbose and n_done % 25 == 0:
                print(f"    spectra: {n_done} materials", flush=True)

    with open(out / "spectra/ml_dielectric.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mat_id", "energy_eV", "ml_eps2"])
        for mat_id, rec in sorted(records.items()):
            ml = rec.pop("_ml_dielectric", None)
            if not ml or not ml.get("energy_ev"):
                continue
            for e, v in zip(ml["energy_ev"], ml["eps2_avg"]):
                w.writerow([mat_id, round(float(e), 4), _r(float(v), 6)])
            n_ml += 1

    if verbose:
        print(f"  wrote spectra/qe_optical.csv ({n_done} materials, {n_rows} rows) "
              f"+ spectra/ml_dielectric.csv ({n_ml} materials)")
    return {"n_spectra": n_done, "spectra_rows": n_rows, "n_ml_spectra": n_ml}


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_manifest(out_dir):
    """MANIFEST.csv: path, bytes, sha256 for every shipped file."""
    out = Path(out_dir)
    rows = []
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.csv" or "_cache" in p.parts:
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": h})
    _write_csv(out / "MANIFEST.csv", rows)
    return rows
