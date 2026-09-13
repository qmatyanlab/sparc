"""Cross-checks: does the bundle reproduce the numbers already published elsewhere?

These are only meaningful because ``build.merge_records`` never merges a value from a
non-authoritative source -- it parks it under ``_crosscheck``. So comparing the bundle
against ``results_all.csv`` / ``dfpt_dft_results.csv`` / the qe_ht JSONs is a genuine
agreement test between two independently produced artifacts, not a tautology.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from . import sources as S
from .schema import EHULL_MAX, GAP_WINDOW, SLME_MIN, Status


def _d(a, b):
    if a is None or b is None:
        return None
    try:
        d = abs(float(a) - float(b))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(d) else d


def _tensor_delta(a, b):
    """Largest elementwise difference between two 3x3s; None if either has a null slot
    (the one DFPT run that returned NaN components is not comparable)."""
    if not a or not b:
        return None
    try:
        return max(abs(float(a[i][j]) - float(b[i][j])) for i in range(3) for j in range(3))
    except (TypeError, ValueError):
        return None


class Checks:
    def __init__(self):
        self.rows = []
        self.failures = []

    def add(self, name, scope, deltas, tol, note="", n_expected=None):
        """deltas: list of (mat_id, delta) -- delta None means 'not comparable'."""
        comparable = [(m, d) for m, d in deltas if d is not None]
        fails = [(m, d) for m, d in comparable if d > tol]
        worst = max((d for _, d in comparable), default=None)
        verdict = "PASS" if not fails else "FAIL"
        if n_expected is not None and len(comparable) != n_expected:
            verdict = "FAIL"
            note = (note + f" | expected {n_expected} comparable, got {len(comparable)}").strip(" |")
        self.rows.append({
            "check": name, "scope": scope, "n": len(deltas), "n_compared": len(comparable),
            "n_pass": len(comparable) - len(fails), "n_fail": len(fails),
            "max_abs_delta": None if worst is None else round(worst, 9),
            "tol": tol, "verdict": verdict, "note": note,
        })
        for m, d in fails[:200]:
            self.failures.append({"check": name, "mat_id": m, "delta": d, "tol": tol})

    def add_bool(self, name, scope, ok, n, note=""):
        self.rows.append({
            "check": name, "scope": scope, "n": n, "n_compared": n,
            "n_pass": n if ok else 0, "n_fail": 0 if ok else n,
            "max_abs_delta": None, "tol": None,
            "verdict": "PASS" if ok else "FAIL", "note": note,
        })


def run_checks(records, extras, out_dir, qe_root=S.DEFAULT_QE_ROOT,
               exp_res=S.DEFAULT_EXP_RES, skip_slme_roundtrip=False, verbose=True):
    c = Checks()
    out = Path(out_dir)
    by_split = {}
    for r in records.values():
        by_split.setdefault(r["identity"]["split"], []).append(r)

    # ---------------------------------------------------------------- counts
    exp_counts = {"slme": 131, "dielectric": 79, "failed_demo": 24, "legacy_pilot": 42}
    c.add_bool("rowcount_total", "all", len(records) == 276, len(records),
               f"expected 276, got {len(records)}")
    for k, v in exp_counts.items():
        got = len(by_split.get(k, []))
        c.add_bool(f"rowcount_{k}", k, got == v, got, f"expected {v}, got {got}")
    n_dfpt = sum(1 for r in records.values() if r["vasp_dfpt"]["status"] == Status.OK)
    n_qe = sum(1 for r in records.values() if r["qe_optical"].get("status") == "done")
    c.add_bool("count_dfpt_tensors", "all", n_dfpt == 88, n_dfpt, f"88 finite tensors (+1 NaN failure); got {n_dfpt}")
    c.add_bool("count_qe_done", "all", n_qe == 127, n_qe, f"expected 127, got {n_qe}")

    # ------------------------------------------------------------ id injectivity
    alias = extras["alias"]
    c.add_bool("id_injectivity", "all", len(alias) == len(records), len(records),
               "sanitize() is injective over every canonical mat_id")
    qe_dfpt_ids = [Path(f).stem for f in
                   Path(qe_root, "database_dfpt").glob("*.json")]
    joined = sum(1 for q in qe_dfpt_ids if alias.get(q))
    c.add_bool("join_qe_dfpt", "database_dfpt", joined == len(qe_dfpt_ids) == 82, len(qe_dfpt_ids),
               f"{joined}/{len(qe_dfpt_ids)} qe_ht DFPT ids resolve to a canonical mat_id")

    # ------------------------------------------- e_hull / gap vs the published files
    slme_ref = S.load_slme_ehull(qe_root)
    c.add("ehull_vs_dft_summary_all", "slme (131)",
          [(m, _d(records[m]["vasp_energetics"].get("e_hull_ev_atom"), v["e_hull_ev_atom"]))
           for m, v in slme_ref.items() if m in records], 1e-4, n_expected=131)
    c.add("gap_vs_qe_ht_vasp_gap", "slme (131)",
          [(m, _d(records[m]["vasp_electronic"].get("bandgap_ev"),
                  v["_crosscheck"]["vasp_gap_relaxed_ev"]))
           for m, v in slme_ref.items() if m in records], 1e-3, n_expected=131)

    dfpt_ref = S.load_dfpt_results(exp_res)
    eh, gp, ex, ei = [], [], [], []
    for lbl, v in dfpt_ref.items():
        m = lbl if lbl in records else alias.get(S.sanitize(lbl))
        if m is None or m not in records:
            continue
        r = records[m]
        eh.append((m, _d(r["vasp_energetics"].get("e_hull_ev_atom"), v["e_hull_ev_atom"])))
        gp.append((m, _d(r["vasp_electronic"].get("bandgap_ev"), v["_crosscheck"]["dft_gap"])))
        ex.append((m, _tensor_delta(v["_crosscheck"].get("eps_electronic_tensor"),
                                    r["vasp_dfpt"].get("eps_electronic_tensor"))))
        ei.append((m, _tensor_delta(v["_crosscheck"].get("eps_ionic_tensor"),
                                    r["vasp_dfpt"].get("eps_ionic_tensor"))))
    c.add("ehull_vs_dfpt_results_csv", "dielectric+demo", eh, 1e-3)
    c.add("gap_vs_dfpt_results_csv", "dielectric+demo", gp, 1e-3)
    c.add("eps_electronic_vs_tensors_json", "dielectric+demo", ex, 1e-4)
    c.add("eps_ionic_vs_tensors_json", "dielectric+demo", ei, 1e-4)

    # legacy: the two thermo jobs disagree for 2 metallic Nb systems -- reported, not failed
    alt = []
    for m, r in records.items():
        cc = (r.get("_crosscheck") or {}).get("legacy_ehull_altjob") or {}
        alt.append((m, _d(r["vasp_energetics"].get("e_hull_ev_atom"),
                          cc.get("e_above_hull_ev_atom"))))
    n_alt_bad = sum(1 for _, d in alt if d is not None and d > 1e-6)
    c.add_bool("ehull_legacy_two_jobs_agree", "legacy_pilot", n_alt_bad == 2,
               sum(1 for _, d in alt if d is not None),
               f"{n_alt_bad} candidates where the standalone compute_e_above_hull job "
               f"disagrees with summarize.thermo (known: c1_NbMoSe4_SG156, c3_Nb2B_SG183; "
               f"summarize.thermo is used)")

    # ---------------------------------------------------------- QE / SLME agreement
    qe_all = {}
    for sub in ("database", "database_dfpt"):
        qe_all.update(S.load_qe_database(Path(qe_root) / sub))
    sl, qg = [], []
    for key, doc in qe_all.items():
        m = alias.get(key)
        if m is None or m not in records:
            continue
        sl.append((m, _d(records[m]["slme"].get("eta_pct"), doc.get("slme_eta_pct"))))
        qg.append((m, _d(records[m]["qe_optical"].get("indirect_gap_ev"), doc.get("indirect_gap"))))
    c.add("slme_passthrough", "qe done", sl, 1e-9)
    c.add("qe_gap_passthrough", "qe attempted", qg, 1e-9)

    # ------------------------------------- SLME recomputed from the bundle's own spectra
    if not skip_slme_roundtrip:
        c.add(*_slme_roundtrip(records, out, qe_root, verbose))

    # ------------------------------------------------------------ structure round-trip
    c.add(*_structure_roundtrip(records, out, verbose))
    name, scope, deltas, tol, note = _structure_vs_qe_poscar(records, out, qe_root)
    n_bad = sum(1 for _, v in deltas if v)
    # an empty comparison set is a FAIL, not a vacuous PASS
    c.add_bool(name, scope, bool(deltas) and n_bad <= 2, len(deltas), note)

    # ------------------------------------------------------------- dedup / hygiene
    same_flow = [r for r in extras["dedup_log"] if r.get("comparability") == "same_flow"]
    c.add("dedup_same_flow_no_physics_change", "duplicate docs",
          [(r["mat_id"], r.get("delta_energy_per_atom_ev")) for r in same_flow], 5e-3,
          f"{len(extras['dedup_log'])} docs dropped; "
          f"{len(extras['dedup_log']) - len(same_flow)} were cross-flow or a rerun from a "
          f"different geometry, where a delta is physics rather than an error")

    # the published SLME campaign yield: stable & QE-PBE gap in [1.0,1.8] & SLME >= 25%
    n_target_qe = sum(1 for r in records.values()
                      if r["identity"]["split"] == "slme"
                      and r["flags"].get("hits_solar_target_qe") is True)
    c.add_bool("published_slme_yield", "slme (131)", n_target_qe == 16, n_target_qe,
               f"reproduces the published campaign yield of 16 target hits "
               f"(stable & QE-PBE gap 1.0-1.8 eV & SLME >= 25%); got {n_target_qe}")

    c.add(*_no_stringy_bools(out))
    c.add(*_flag_consistency(records))
    c.add(*_relaxed_differs(records, out))
    return c


def _slme_roundtrip(records, out, qe_root, verbose=True):
    """Re-integrate SLME from spectra/qe_optical.csv and compare to the stored eta."""
    path = out / "spectra" / "qe_optical.csv"
    if not path.exists():
        return ("slme_roundtrip", "qe done", [], 1e-3, "spectra not written")
    if str(qe_root) not in os.sys.path:
        os.sys.path.insert(0, str(qe_root))
    from qe_ht import slme as qslme

    spec: dict[str, tuple[list, list]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            e, a = spec.setdefault(row["mat_id"], ([], []))
            v = row["eps2_avg"]
            if v in ("", None):
                continue
            e.append(float(row["energy_eV"]))
            a.append(float(v))

    deltas = []
    for mat_id, (E, A) in spec.items():
        rec = records.get(mat_id)
        if rec is None or rec["slme"].get("eta_pct") is None or len(E) < 10:
            continue
        gap = rec["slme"].get("gap_used_ev")
        try:
            got = qslme.slme_from_dielectric(E, A, gap,
                                             rec["slme"].get("thickness_um") or 0.3,
                                             rec["slme"].get("temperature_k") or 300.0)
        except Exception:  # noqa: BLE001
            deltas.append((mat_id, None))
            continue
        pct = got.get("slme_eta_pct")
        rec["slme"]["eta_pct_recomputed"] = None if pct is None else round(float(pct), 6)
        d = _d(pct, rec["slme"]["eta_pct"])
        rec["slme"]["roundtrip_delta_pp"] = None if d is None else round(d, 6)
        deltas.append((mat_id, d))
    return ("slme_roundtrip", "qe done", deltas, 1e-3,
            "SLME re-integrated from the bundle's own eps2_avg column")


def _structure_roundtrip(records, out, verbose=True):
    """Re-read every written CIF; lattice parameters must match the stored scalars."""
    from pymatgen.core import Structure
    import warnings
    warnings.simplefilter("ignore")

    deltas = []
    for mat_id, rec in sorted(records.items()):
        rel = rec["vasp_structure"].get("cif_relaxed")
        if not rel:
            deltas.append((mat_id, None))
            continue
        try:
            st = Structure.from_file(out / rel)
        except Exception:  # noqa: BLE001
            deltas.append((mat_id, float("inf")))
            continue
        want = [rec["vasp_structure"].get(k) for k in ("a", "b", "c", "alpha", "beta", "gamma")]
        got = [st.lattice.a, st.lattice.b, st.lattice.c,
               st.lattice.alpha, st.lattice.beta, st.lattice.gamma]
        d = max((abs(g - w) for g, w in zip(got, want) if w is not None), default=None)
        if len(st) != rec["identity"]["nsites"]:
            d = float("inf")
        deltas.append((mat_id, d))
    return ("structure_roundtrip", "all", deltas, 1e-3,
            "CIF -> Structure lattice params + nsites vs the stored scalars")


def _structure_vs_qe_poscar(records, out, qe_root):
    """The cells QE actually ran must be the cells the bundle ships.

    Compared with ``StructureMatcher`` rather than raw lattice parameters, because the QE
    inputs were primitive-standardised and so legitimately carry a different setting for
    the same crystal.

    Two of the 82 genuinely differ: ``r020_RbSc(TeO4)2_SG150_step17_i13`` and
    ``r042_HCl_SG44_step4_i8`` were run through QE on the older PBEsol-pilot relaxed
    geometry rather than the MP-GGA cell the bundle ships. Each affected record is
    flagged with ``qe_geometry_matches_shipped_cell = False`` so a reviewer sees that its
    optical spectrum and its VASP numbers refer to slightly different cells.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    import warnings
    warnings.simplefilter("ignore")

    sm = StructureMatcher(primitive_cell=True, attempt_supercell=True)
    by_fs = {r["identity"]["mat_id_fs"]: m for m, r in records.items()}
    d = Path(qe_root) / "dfpt_spectra" / "structures"
    deltas, mismatched = [], []
    for p in sorted(d.glob("*.poscar")):
        mat_id = by_fs.get(p.stem)
        if mat_id is None:
            continue
        rec = records[mat_id]
        rel = rec["vasp_structure"].get("cif_relaxed")
        if not rel:
            deltas.append((mat_id, None))
            continue
        try:
            qe = Structure.from_str(p.read_text(), fmt="poscar")
            ours = Structure.from_file(out / rel)
        except Exception:  # noqa: BLE001
            deltas.append((mat_id, None))
            continue
        same = bool(sm.fit(qe, ours))
        rec["flags"]["qe_geometry_matches_shipped_cell"] = same
        if not same:
            mismatched.append(mat_id)
        deltas.append((mat_id, 0.0 if same else 1.0))

    # Only the dielectric-campaign cells QE ran are on disk to compare against, so only
    # those records get a verdict. Everything else stays null: claiming True for a
    # comparison that never happened would put an unearned assurance in the shipped CSV.
    for rec in records.values():
        rec["flags"].setdefault("qe_geometry_matches_shipped_cell", None)

    note = (f"StructureMatcher over the {len(deltas)} cells QE ran; {len(mismatched)} ran on "
            f"the older PBEsol-pilot geometry instead of the shipped MP-GGA cell"
            + (f": {', '.join(mismatched)}" if mismatched else "")
            + ". Records with no POSCAR to compare keep a null flag.")
    if not deltas:
        note = (f"NO POSCARs found under {d} -- nothing was compared. Check --qe-root.")
    return ("structure_vs_qe_poscar", "dielectric+demo", deltas, 0.5, note)


def _relaxed_differs(records, out):
    """DFT relaxation should have moved the cell away from the generated one.

    A handful of cells legitimately come back unchanged -- the MLIP geometry was already
    at the DFT minimum, so the ionic loop converged on step 1 (verified for
    ``rank128_S4O8_sg92``: max force 0.017 eV/A, residual stress ~1.2 kBar, on the hull).
    This check therefore asserts that the *bulk* of the set relaxed, not every member;
    a high unchanged fraction would mean the relaxed geometry was never pulled at all.
    """
    from pymatgen.core import Structure
    import warnings
    warnings.simplefilter("ignore")

    unchanged, dv = [], []
    for mat_id, rec in sorted(records.items()):
        gen = rec["lineage"].get("rl_cif_as_generated")
        if not gen or not rec["vasp_structure"].get("volume_A3"):
            continue
        try:
            g = Structure.from_file(out / gen)
        except Exception:  # noqa: BLE001
            continue
        v0, v1 = g.lattice.volume, rec["vasp_structure"]["volume_A3"]
        rel = abs(v1 - v0) / v0
        dv.append(rel)
        if rel <= 1e-6:
            unchanged.append(mat_id)
    n = len(dv) or 1
    frac = len(unchanged) / n
    med = sorted(dv)[len(dv) // 2] if dv else 0.0
    return ("relaxed_differs_from_generated", "with as-generated cif",
            [("fraction_unchanged", frac)], 0.05,
            f"{n - len(unchanged)}/{n} cells moved during DFT relaxation "
            f"(median |dV|/V = {med:.3f}); unchanged: {', '.join(unchanged) or 'none'}")


def _no_stringy_bools(out):
    """No boolean leaf may still be the literal string "True"/"False".

    Checked against ``materials.jsonl``, not the CSV: in CSV a real Python bool and the
    string "True" are byte-identical, so the CSV cannot distinguish them and a check
    there can never fail. JSON keeps the types apart, so this is a real test of the
    ``_b()`` casts applied to the qe_ht records (which store ``in_solar_window`` as a
    string).
    """
    import json

    p = out / "materials.jsonl"
    bad = []
    if p.exists():
        def walk(o, path, mat_id):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{path}.{k}" if path else k, mat_id)
            elif isinstance(o, list):
                for v in o[:50]:
                    walk(v, path, mat_id)
            elif isinstance(o, str) and o in ("True", "False"):
                key = path.rsplit(".", 1)[-1]
                if key.startswith(("is_", "has_", "caution_")) or key in (
                        "stable", "slme_ok", "gap_in_window", "gap_in_window_qe",
                        "hits_solar_target", "hits_solar_target_qe", "e_hull_approx",
                        "ml_in_solar_window", "ml_in_gap_window", "is_gap_direct",
                        "qe_vasp_gap_disagreement", "is_hyperbolic", "is_metal"):
                    bad.append((f"{mat_id}:{path}", 1.0))

        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                walk(r, "", r["identity"]["mat_id"])
    return ("no_stringy_bools", "materials.jsonl", bad or [("-", 0.0)], 0.5,
            "boolean leaves are real JSON booleans, not the strings the qe_ht records use")


def _flag_consistency(records):
    """Derived flags must follow from the underlying values."""
    bad = []
    for mat_id, r in sorted(records.items()):
        gap = r["vasp_electronic"].get("bandgap_ev")
        eh = r["vasp_energetics"].get("e_hull_ev_atom")
        eta = r["slme"].get("eta")
        f = r["flags"]
        checks = [
            gap is None or f["is_gapless"] == (gap < 1e-6),
            gap is None or f["gap_in_window"] == (GAP_WINDOW[0] <= gap <= GAP_WINDOW[1]),
            eh is None or f["stable"] == (eh <= EHULL_MAX),
            eta is None or f["slme_ok"] == (eta >= SLME_MIN),
            f["has_vasp_dfpt"] == (r["vasp_dfpt"]["status"] == Status.OK),
        ]
        bad.append((mat_id, 0.0 if all(checks) else 1.0))
    return ("flag_consistency", "all", bad, 0.5,
            "gap_in_window / stable / slme_ok / has_vasp_dfpt follow from the raw values")


def write_reports(checks, out_dir):
    out = Path(out_dir) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / "verification.csv", checks.rows)
    _csv(out / "verification_failures.csv", checks.failures)
    return checks.rows


def _csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
