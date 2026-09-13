"""Loaders: one per upstream source, each returning ``{mat_id: partial record}``.

Authority is explicit and never negotiated at merge time:

    * MongoDB (via the stage-1 cache)  -- authoritative for every VASP DFT quantity
    * ~/qe_ht                          -- authoritative for QE optical / SLME / hyperbolic
    * the RL run dirs                  -- authoritative for surrogate values and lineage

Where a non-authoritative source repeats a quantity (e.g. ``dfpt_dft_results.csv`` also
carries ``dft_gap``) the value is NOT merged -- it is stashed under ``_crosscheck`` so
``verify.py`` can compare it against the authoritative one. That is what keeps the
verification from being a tautology.
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import os
import re
import sys
from pathlib import Path

from .ids import parse_label, resolve, sanitize

HOME = Path(os.path.expanduser("~"))
DEFAULT_QE_ROOT = Path(os.environ.get("QE_HT_ROOT", HOME / "qe_ht"))
DEFAULT_EXP_RES = Path(os.environ.get("SPARC_EXP_RES", Path(__file__).resolve().parents[2] / "exp_res"))

DIELECTRIC_RUN = "dielectric_inplane_isotropy_gapgate_mprime_newbg_b96"
SLME_RUN = "slme_bgcenter13_best_reward_sample1000"

# Where the raw epsilon.x output lives: the SLME campaign roots first, then the DFPT ones.
QE_WORK_ROOTS = [
    "work_all_0", "work_all_1", "work_all_2",
    "work_dfpt_0", "work_dfpt_1", "work_dfpt_2", "work_dfpt_3", "work_dfpt_4", "work_dfpt_5",
    "work_relaxed", "work",
]

# Provenance of the RL checkpoints each split came from (from the run READMEs / RUNS.md).
RL_PROVENANCE = {
    "slme": {
        "rl_run_name": SLME_RUN,
        "rl_checkpoint": "tsenn_slme_03um_optimate_bgcenter13_eta08_e3nngap (c2) best_reward, step 189",
        "rl_sample_mode": "best_reward 1000-sample draw (single step)",
        "rl_generator": "SymmCD (M-prime fine-tuned)",
        "rl_surrogate_target": "slme",
    },
    "dielectric": {
        "rl_run_name": DIELECTRIC_RUN,
        "rl_checkpoint": "fom_inplane_isotropy_gapgate, 120-step run",
        "rl_sample_mode": "per-step RL evaluation set",
        "rl_generator": "SymmCD (M-prime fine-tuned)",
        "rl_surrogate_target": "dielectric_inplane_isotropy",
    },
    "failed_demo": {
        "rl_run_name": DIELECTRIC_RUN,
        "rl_checkpoint": "fom_inplane_isotropy_gapgate, 120-step run",
        "rl_sample_mode": "deliberately selected reward-failure counter-examples",
        "rl_generator": "SymmCD (M-prime fine-tuned)",
        "rl_surrogate_target": "dielectric_inplane_isotropy",
    },
    "legacy_pilot": {
        "rl_run_name": "sparc_v0 dielectric pilot (NERSC)",
        "rl_checkpoint": None,
        "rl_sample_mode": "early DFPT screen (mixed hand-picked, screened and RL candidates)",
        "rl_generator": None,
        "rl_surrogate_target": "dielectric_inplane_isotropy",
    },
}


# The RL-side inputs live under exp_res/, which is gitignored -- on a fresh clone they
# are simply absent, and a silent {} from a loader would produce a quietly degraded
# bundle instead of an error. check_rl_sources() names exactly what is missing.
REQUIRED_RL_SOURCES = [
    ("SLME worklist", f"{SLME_RUN}/dft_candidates_slme_sun/worklist.csv"),
    ("SLME as-generated CIFs", f"{SLME_RUN}/dft_candidates_slme_sun/cifs"),
    ("dielectric summary", f"{DIELECTRIC_RUN}/deliverables_dfpt_candidates/_summary.csv"),
    ("dielectric DFT report", f"{DIELECTRIC_RUN}/deliverables_dfpt_candidates/dfpt_dft_results.csv"),
    ("failure-demo summary", f"{DIELECTRIC_RUN}/failed_reward_demo/_summary.csv"),
]


def check_rl_sources(exp_res=DEFAULT_EXP_RES) -> list[str]:
    """-> list of human-readable descriptions of the RL inputs that are missing."""
    return [f"{label}: {Path(exp_res) / rel}"
            for label, rel in REQUIRED_RL_SOURCES if not (Path(exp_res) / rel).exists()]


def rl_provenance(split: str) -> dict:
    """RL provenance for a split, tolerant of a split we have no lineage story for.

    ``export_dft_mongo_cache.py`` maps an unrecognised ``metadata.campaign`` to
    ``"unknown"`` rather than dropping the material, so this must not raise -- a new
    campaign should surface as an audit finding, not a KeyError halfway through a build.
    """
    return RL_PROVENANCE.get(split, {
        "rl_run_name": None, "rl_checkpoint": None,
        "rl_sample_mode": None, "rl_generator": None,
        "rl_surrogate_target": None,
    })


# ----------------------------------------------------------------------------- helpers
def _f(x):
    """Anything -> float or None (handles '', 'nan', 'None', already-float)."""
    if x is None:
        return None
    if isinstance(x, bool):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else v


def _b(x):
    """Anything -> bool or None.  Handles the literal 'True'/'False' STRINGS that the
    qe_ht JSONs store for `surrogate.in_solar_window`."""
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "t"):
        return True
    if s in ("false", "0", "no", "f"):
        return False
    return None


def _i(x):
    v = _f(x)
    return None if v is None else int(round(v))


def _read_csv(path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------------ 1+2. Mongo
def load_mongo_cache(path):
    """Stage-1 cache -> (records, dedup_log, incomplete, meta). Authoritative VASP DFT."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Mongo cache not found: {path}\nRun stage 1 first:\n"
            f"    ~/atomate2/.venv/bin/python scripts/export_dft_mongo_cache.py"
        )
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        d = json.load(fh)
    meta = {k: v for k, v in d.items() if k not in ("records", "dedup_log", "incomplete")}
    return d["records"], d.get("dedup_log", []), d.get("incomplete", []), meta


# ------------------------------------------------------------------- 3. SLME e_hull
def load_slme_ehull(qe_root=DEFAULT_QE_ROOT):
    """qe_ht/vasp_ehull/dft_summary_all.json -> the published e_hull for the 131."""
    p = Path(qe_root) / "vasp_ehull" / "dft_summary_all.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out = {}
    for mat_id, v in raw.items():
        out[mat_id] = {
            "e_hull_ev_atom": _f(v.get("dft_e_hull_ev_atom")),
            "formation_energy_ev_atom": _f(v.get("dft_ef_ev_atom")),
            "e_hull_source": "qe_ht/vasp_ehull/dft_summary_all.json "
                             "(MaterialsProject2020Compatibility + MP GGA/GGA+U hull, 2026-07-31)",
            "_crosscheck": {"vasp_gap_relaxed_ev": _f(v.get("vasp_gap_relaxed_ev")),
                            "formula_mp": v.get("formula_mp")},
        }
    return out


# --------------------------------------------------- 4. dielectric campaign DFT results
def load_dfpt_results(exp_res=DEFAULT_EXP_RES):
    """dfpt_dft_results.csv (+ _tensors.json) -> e_hull, group, failure_mode + cross-checks."""
    base = Path(exp_res) / DIELECTRIC_RUN / "deliverables_dfpt_candidates"
    csv_p, json_p = base / "dfpt_dft_results.csv", base / "dfpt_dft_tensors.json"
    if not csv_p.exists():
        return {}
    tensors = {}
    if json_p.exists():
        for rec in json.loads(json_p.read_text()):
            tensors[rec["label"]] = rec

    out = {}
    for row in _read_csv(csv_p):
        lbl = row["label"]
        t = tensors.get(lbl, {})
        out[lbl] = {
            "e_hull_ev_atom": _f(row.get("dft_ehull")),
            "e_hull_approx": _b(row.get("dft_ehull_approx")) or False,
            "e_hull_source": "deliverables_dfpt_candidates/dfpt_dft_results.csv "
                             "(MaterialsProject2020Compatibility + MP GGA/GGA+U hull, 2026-08-02)",
            "group": row.get("group") or None,
            "failure_mode": (row.get("failure_mode") or "") or None,
            "dfpt_report_status": row.get("status"),
            "_crosscheck": {
                "dft_gap": _f(row.get("dft_gap")),
                "dft_eps0_xx": _f(row.get("dft_eps0_xx")),
                "dft_eps0_yy": _f(row.get("dft_eps0_yy")),
                "dft_eps0_zz": _f(row.get("dft_eps0_zz")),
                "dft_eps0_iso": _f(row.get("dft_eps0_iso")),
                "dft_eps_ionic_iso": _f(row.get("dft_eps_ionic_iso")),
                "dft_eps_total_iso": _f(row.get("dft_eps_total_iso")),
                "eps_electronic_tensor": t.get("eps_electronic_tensor"),
                "eps_ionic_tensor": t.get("eps_ionic_tensor"),
            },
            "_ml": {
                "ml_band_gap_ev": _f(row.get("ml_gap")),
                "ml_eps_xx": _f(row.get("ml_eps_xx")),
                "ml_eps_yy": _f(row.get("ml_eps_yy")),
                "ml_eps_zz": _f(row.get("ml_eps_zz")),
                "ml_eps_iso": _f(row.get("ml_eps_iso")),
                "mlip_ehull_ev_atom": _f(row.get("ml_ehull")),
            },
        }
    return out


# ------------------------------------------------------------------ 5+6. qe_ht databases
def load_qe_database(db_dir):
    """qe_ht/database{,_dfpt}/*.json -> {file stem (sanitized id): parsed record}."""
    out = {}
    for f in sorted(glob.glob(str(Path(db_dir) / "*.json"))):
        out[Path(f).stem] = json.loads(Path(f).read_text())
    return out


def qe_blocks(d):
    """A qe_ht JSON record -> the (qe_optical, slme, hyperbolic, surrogate, ml) blocks."""
    diel = d.get("dielectric") or {}
    e2 = diel.get("eps2_avg") or []
    ev = diel.get("energy_ev") or []
    peak = peak_ev = None
    if e2:
        j = max(range(len(e2)), key=lambda i: e2[i])
        peak, peak_ev = _f(e2[j]), _f(ev[j]) if j < len(ev) else None

    qe = {
        "status": d.get("status"),
        "error": d.get("error"),
        "indirect_gap_ev": _f(d.get("indirect_gap")),
        "direct_gap_ev": _f(d.get("direct_gap")),
        "vbm_ev": _f(d.get("vbm")),
        "cbm_ev": _f(d.get("cbm")),
        "is_metal": _b(d.get("is_metal")),
        "nelec": _f(d.get("nelec")),
        "natoms": _i(d.get("natoms")),
        "eps2_peak": peak,
        "eps2_peak_ev": peak_ev,
        "n_energy_points": len(ev) or None,
        "provenance": d.get("provenance") or {},
    }

    sp = d.get("slme_params") or {}
    sl = {
        "eta": _f(d.get("slme_eta")),
        "eta_pct": _f(d.get("slme_eta_pct")),
        "gap_used_ev": _f(sp.get("gap_ev")),
        "thickness_um": _f(sp.get("thickness_um")),
        "temperature_k": _f(sp.get("temperature_k")),
        "broadening_ev": _f(sp.get("broadening_ev")),
    }

    h = d.get("hyperbolic") or {}
    bw = h.get("best_window") or {}
    hyp = {
        "is_hyperbolic": _b(h.get("is_hyperbolic")),
        "score": _f(h.get("score")),
        "n_windows": _i(h.get("n_windows")),
        "n_clean_windows": _i(h.get("n_clean_windows")),
        "clean_bandwidth_ev": _f(h.get("clean_bandwidth_eV")),
        "total_bandwidth_ev": _f(h.get("total_bandwidth_eV")),
        "best_type": bw.get("type"),
        "best_fom": _f(bw.get("max_fom")),
        "best_band": bw.get("band"),
        "best_lo_ev": _f(bw.get("E_lo")),
        "best_hi_ev": _f(bw.get("E_hi")),
        "windows": h.get("windows"),
        "eps0_crossings_xyz": h.get("eps0_crossings_xyz"),
        "fom_min": _f(h.get("fom_min")),
        "note": h.get("note"),
    } if h else {}

    sur = d.get("surrogate") or {}
    mlip = d.get("e_hull_mlip") or {}
    surrogate = {
        "ml_band_gap_ev": _f(sur.get("band_gap_eV")),
        "ml_eta_slme": _f(sur.get("eta_slme")),
        # stored as the literal STRING "True"/"False" in the qe_ht JSONs
        "ml_in_solar_window": _b(sur.get("in_solar_window")),
        "mlip_ehull_ev_atom": _f(mlip.get("e_hull_ev_atom")),
        "mlip_ehull_self_consistent_ev_atom": _f(mlip.get("self_consistent_e_hull_ev_atom")),
    } if (sur or mlip) else {}

    dft_eh = d.get("dft_ehull") or {}
    ehull = {
        "e_hull_ev_atom": _f(dft_eh.get("e_hull_ev_atom")),
        "formation_energy_ev_atom": _f(dft_eh.get("formation_energy_ev_atom")),
        "e_hull_source": dft_eh.get("source"),
    } if dft_eh else {}

    mlv = d.get("ml_vs_dft") or {}
    ml_diel = d.get("ml_dielectric") or {}
    return {
        "qe_optical": qe, "slme": sl, "hyperbolic": hyp, "surrogate": surrogate,
        "ehull": ehull,
        "ml_vs_dft": {
            "ml_eps2_peak": _f(mlv.get("ml_eps2_peak")),
            "ml_eps2_peak_ev": _f(mlv.get("ml_eps2_peak_eV")),
            "eps2_similarity": _f(mlv.get("eps2_SC")),
        } if mlv else {},
        "ml_dielectric": {"energy_ev": ml_diel.get("energy_ev"),
                          "eps2_avg": ml_diel.get("eps2_avg"),
                          "source": ml_diel.get("source")} if ml_diel else {},
        "_crosscheck": {
            "vasp_gap_relaxed_ev": _f(d.get("vasp_gap_relaxed_ev")),
            "structure_summary": d.get("structure_summary"),
            "structure_relaxed": d.get("structure_relaxed"),
        },
        "dielectric_spectrum": {"energy_ev": ev, "eps2_avg": e2,
                                "eps2_xyz": diel.get("eps2_xyz")} if ev else {},
    }


# ------------------------------------------------------------------- 7. raw QE spectra
def load_qe_spectrum(mat_id_fs, qe_root=DEFAULT_QE_ROOT):
    """Per-axis eps1/eps2 straight off ``epsr_pwscf.dat`` / ``epsi_pwscf.dat``.

    Delegates to ``qe_ht.hyperbolic.load_eps_tensor`` so the Fortran ``****`` overflow
    near omega->0 becomes NaN (never 0.0, which would fabricate an eps_inf).
    """
    qe_root = str(qe_root)
    if qe_root not in sys.path:
        sys.path.insert(0, qe_root)
    from qe_ht import hyperbolic  # noqa: PLC0415 -- optional external dependency

    try:
        return hyperbolic.load_eps_tensor(mat_id_fs, QE_WORK_ROOTS)
    except Exception:  # noqa: BLE001 -- a missing/corrupt scratch dir is not fatal
        return None


# ------------------------------------------------------------------ 8. SLME RL worklist
def load_rl_worklist_slme(exp_res=DEFAULT_EXP_RES):
    """dft_candidates_slme_sun/{worklist.csv, cifs/} -> surrogate values + as-generated CIF."""
    base = Path(exp_res) / SLME_RUN / "dft_candidates_slme_sun"
    p = base / "worklist.csv"
    if not p.exists():
        return {}
    out = {}
    for row in _read_csv(p):
        cif = row["cif"]                       # "cifs/rank01_As4P4Se8_sg14.cif"
        mat_id = Path(cif).stem
        src = base / cif
        out[mat_id] = {
            "surrogate": {
                "surrogate_target": "slme",
                "ml_band_gap_ev": _f(row.get("band_gap_eV")),
                "ml_eta_slme": _f(row.get("eta_slme")),
                "ml_jsc_ma_cm2": _f(row.get("jsc_mA_cm2")),
                "ml_voc_v": _f(row.get("voc_V")),
                "reward_r_uni": _f(row.get("r_uni")),
                "ml_in_gap_window": _b(row.get("in_gap_window")),
                "ml_in_solar_window": _b(row.get("in_solar_window")),
                "mlip_ehull_ev_atom": _f(row.get("e_hull_ev_atom")),
                "mlip_ehull_self_consistent_ev_atom": _f(row.get("self_consistent_e_hull_ev_atom")),
                "rl_total_energy_ev": _f(row.get("total_energy_ev")),
            },
            "lineage": {
                "rl_eval_index": _i(row.get("eval_index")),
                "rl_step": 0,
                "rl_spacegroup_as_generated_symbol": row.get("spacegroup_symbol") or None,
                "rl_spacegroup_as_generated_number": _i(row.get("spacegroup_number")),
            },
            "identity": {"label_rank": _i(row.get("rank"))},
            "_as_generated_cif": str(src) if src.exists() else None,
        }
    return out


# ------------------------------------------- 9+10. dielectric + failure-demo RL summaries
def _dfpt_summary_dir(exp_res, which):
    base = Path(exp_res) / DIELECTRIC_RUN
    return base / ("deliverables_dfpt_candidates" if which == "candidates" else "failed_reward_demo")


def load_rl_summary_dielectric(exp_res=DEFAULT_EXP_RES):
    """deliverables_dfpt_candidates/_summary.csv -> in-plane-isotropy surrogate values."""
    d = _dfpt_summary_dir(exp_res, "candidates")
    p = d / "_summary.csv"
    if not p.exists():
        return {}
    out = {}
    for row in _read_csv(p):
        mat_id = Path(row["cif"]).stem
        src = d / row["cif"]
        out[mat_id] = {
            "surrogate": {
                "surrogate_target": "dielectric_inplane_isotropy",
                "ml_band_gap_ev": _f(row.get("band_gap_ev")),
                "reward_r_uni": _f(row.get("r_uni")),
                "ml_eps_xx": _f(row.get("eps_xx")),
                "ml_eps_yy": _f(row.get("eps_yy")),
                "ml_eps_zz": _f(row.get("eps_zz")),
                "ml_eps_iso": _f(row.get("iso_scalar")),
                "ml_eps_zz_over_par": _f(row.get("eps_zz_over_par")),
                "mlip_ehull_ev_atom": _f(row.get("ehull_ev_atom")),
                "n_same_structure": _i(row.get("n_same")),
            },
            "lineage": {"rl_step": _i(row.get("step")), "rl_eval_index": _i(row.get("index")),
                        "rl_spacegroup_as_generated_symbol": row.get("spacegroup") or None,
                        "rl_spacegroup_as_generated_number": _i(row.get("sg_no"))},
            "identity": {"label_rank": _i(row.get("rank"))},
            "_as_generated_cif": str(src) if src.exists() else None,
        }
    return out


def load_rl_summary_failed_demo(exp_res=DEFAULT_EXP_RES):
    """failed_reward_demo/_summary.csv -> the counter-example surrogate values + mode."""
    d = _dfpt_summary_dir(exp_res, "demo")
    p = d / "_summary.csv"
    if not p.exists():
        return {}
    out = {}
    for row in _read_csv(p):
        mat_id = Path(row["cif"]).stem
        src = d / row["cif"]
        out[mat_id] = {
            "surrogate": {
                "surrogate_target": "dielectric_inplane_isotropy",
                "ml_band_gap_ev": _f(row.get("gap_eV")),
                "reward_r_uni": _f(row.get("r_uni")),
                "ml_eps_xx": _f(row.get("eps_xx")),
                "ml_eps_yy": _f(row.get("eps_yy")),
                "ml_eps_zz": _f(row.get("eps_zz")),
                "ml_eps_iso": _f(row.get("iso_score")),
                "ml_inplane_mismatch": _f(row.get("inplane_mismatch")),
                "mlip_ehull_ev_atom": _f(row.get("ehull")),
            },
            "lineage": {"rl_step": _i(row.get("step")), "rl_eval_index": _i(row.get("index")),
                        "rl_spacegroup_as_generated_number": _i(row.get("spacegroup"))},
            "identity": {"label_rank": _i(row.get("rank")), "failure_mode": row.get("mode") or None,
                         "group": "failed_reward_demo"},
            "_as_generated_cif": str(src) if src.exists() else None,
        }
    return out


def load_legacy_as_generated(repo_root=None):
    """sparc_v1/cifs/ -> the handful of legacy as-generated CIFs that survived.

    Matched on ``(reduced formula, spacegroup number)`` parsed out of both filenames,
    because the legacy candidate ids and these CIF names use different conventions.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    out = {}
    for f in sorted(glob.glob(str(root / "cifs" / "*.cif"))):
        stem = Path(f).stem                       # e.g. "SnS_SG144"
        m = re.match(r"^(?P<formula>.+?)_SG(?P<sg>\d+)$", stem)
        if m:
            out[(m.group("formula"), int(m.group("sg")))] = str(Path(f))
    return out


# ------------------------------------------------------------------ 11. RL S.U.N. flags
def load_rl_stability(exp_res=DEFAULT_EXP_RES):
    """``samples/step_XXXX_stability.csv`` -> is_valid / is_unique / is_novel / is_stable.

    Positional alignment between the RL eval set and the stability table is only
    partially reliable (``kept_index`` is off by a few on some steps), so every match is
    **verified against the reduced formula** and the method that succeeded is recorded:

        total_energy     a unique stability row has that RL total energy (preferred)
        kept_positional  the eval index counts kept rows in file order
        kept_index       the explicit kept_index column agrees
        raw_index        the pre-filter index column agrees
        unique_formula   exactly one kept row has that formula
        ambiguous        several equally good candidates -> flags left NULL

    Never guesses: an ambiguous material gets ``stability_join_method="ambiguous"`` and
    null S.U.N. flags rather than a plausible-looking wrong answer.
    """
    from pymatgen.core import Composition

    def norm(f):
        try:
            return Composition(str(f)).reduced_formula
        except Exception:  # noqa: BLE001
            return None

    runs = {"dielectric": DIELECTRIC_RUN, "failed_demo": DIELECTRIC_RUN, "slme": SLME_RUN}
    cache: dict[str, object] = {}
    out: dict[str, dict] = {}

    def _table(run, step):
        p = Path(exp_res) / run / "samples" / f"step_{step:04d}_stability.csv"
        if str(p) not in cache:
            cache[str(p)] = _read_csv(p) if p.exists() else None
        return cache[str(p)]

    def _flags(row, method):
        return {
            "is_valid": _b(row.get("is_valid")), "is_unique": _b(row.get("is_unique")),
            "is_novel": _b(row.get("is_novel")), "is_stable": _b(row.get("is_stable")),
            "rl_energy_above_hull_ev_atom": _f(row.get("energy_above_hull_ev_per_atom")),
            "rl_self_consistent_ehull_ev_atom":
                _f(row.get("self_consistent_energy_above_hull_ev_per_atom")),
            "rl_total_energy_ev": _f(row.get("total_energy_ev")),
            "stability_join_method": method,
        }

    def resolve_one(run, step, idx, formula, total_energy_ev=None):
        rows = _table(run, step)
        if rows is None:
            return {"stability_join_method": "no_stability_file"}
        if idx is None and total_energy_ev is None:
            return {"stability_join_method": "no_rl_index"}
        kept = [r for r in rows if str(r.get("kept")).strip().lower() == "true"]
        want = norm(formula) if formula else None

        # Preferred key: the RL total energy, which is what export_slme_inrange_dft.py
        # itself used to join the eval frames to the stability table.  The eval-frame
        # order does NOT track the kept-row order, so positional matching is unreliable
        # wherever a total energy is available.
        if total_energy_ev is not None:
            tgt = round(float(total_energy_ev), 4)
            hits = [r for r in rows if _f(r.get("total_energy_ev")) is not None
                    and round(_f(r["total_energy_ev"]), 4) == tgt]
            if want is not None:
                narrowed = [r for r in hits if norm(r["reduced_formula"]) == want]
                hits = narrowed or hits
            if len(hits) == 1:
                return _flags(hits[0], "total_energy")

        if idx is None:
            return {"stability_join_method": "ambiguous"}

        if idx < len(kept) and (want is None or norm(kept[idx]["reduced_formula"]) == want):
            return _flags(kept[idx], "kept_positional")
        for label, pool, col in (("kept_index", kept, "kept_index"), ("raw_index", rows, "index")):
            hits = [r for r in pool if _i(r.get(col)) == idx]
            if len(hits) == 1 and (want is None or norm(hits[0]["reduced_formula"]) == want):
                return _flags(hits[0], label)
        if want is not None:
            hits = [r for r in kept if norm(r["reduced_formula"]) == want]
            if len(hits) == 1:
                return _flags(hits[0], "unique_formula")
        return {"stability_join_method": "ambiguous"}

    return runs, resolve_one, out


def formula_from_label(mat_id):
    """``r001_OsO4_SG173_step96_i15`` -> ``OsO4``; None when the family has no formula field."""
    m = re.match(r"^(?:[a-z]+_)?r\d+_(?P<f>.+?)_SG\d+", mat_id)
    if m:
        return m.group("f")
    m = re.match(r"^rank\d+_(?P<f>.+?)_sg\d+$", mat_id)
    return m.group("f") if m else None
