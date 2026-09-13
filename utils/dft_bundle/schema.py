"""Record schema for the consolidated DFT dataset.

One record per material. The record is organised into **blocks**, and each block carries
its own ``status`` + ``status_reason`` rather than tagging every leaf -- so a ``null``
never has to be guessed at:

    ok              ran, values present
    partial         ran, but some leaves are legitimately absent
    not_applicable  by design never in scope (no DFPT for the SLME campaign; no optics
                    for a metal)
    not_run         in scope but never launched -- should be zero; the audit flags any
    failed          launched and errored (the QE ``nscf``/``c_bands`` failures)
    missing         should exist but the artifact was not found -- a REAL gap

``EXPECTED_NULLS`` declares every legitimate null as data, with the count we expect.
The audit asserts expected == observed, so a change in the upstream data surfaces as a
rule mismatch instead of a silent hole.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0"

BLOCKS = (
    "identity", "lineage", "rl_surrogate", "vasp_structure", "vasp_energetics",
    "vasp_electronic", "vasp_dfpt", "qe_optical", "slme", "hyperbolic", "flags",
)

# Blocks that carry a status/status_reason pair (identity and flags are always present).
STATUS_BLOCKS = (
    "rl_surrogate", "vasp_structure", "vasp_energetics", "vasp_electronic",
    "vasp_dfpt", "qe_optical", "slme", "hyperbolic",
)


class Status:
    OK = "ok"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    NOT_RUN = "not_run"
    FAILED = "failed"
    MISSING = "missing"


# Physics thresholds, kept identical to qe_ht/qe_ht/yield_report.py so the bundle
# reproduces the published yield numbers.
GAP_WINDOW = (1.0, 1.8)      # eV, the solar target window
EHULL_MAX = 0.1              # eV/atom, "DFT-stable"
SLME_MIN = 0.25              # fraction, 25%

# QE runs nspin=1 with no +U, so it is unreliable for magnetic transition metals.
MAGNETIC_TM = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"}
# Molecular van-der-Waals solids: transparent, so a "hyperbolic" hit there is an
# Im(eps)~0 artifact rather than a material.
MOLECULAR_VDW_FORMULAS = {"F2", "Cl2", "Br2", "I2", "HCl", "HF", "HBr", "HI",
                          "ICl3", "PF3", "SiF4", "SiCl4", "SiI4", "N2", "O2", "H2"}

# --------------------------------------------------------------------------------------
# Every legitimate null, declared up front.  ``rule`` is evaluated over a merged record.
# --------------------------------------------------------------------------------------
EXPECTED_NULLS = [
    # (block, the status the rule explains, a predicate over the merged record, expected count)
    {
        "id": "dfpt_not_in_slme_campaign", "block": "vasp_dfpt",
        "status": Status.NOT_APPLICABLE, "expected_n": 131,
        "reason": "DFPT dielectric was never part of the SLME solar campaign",
        "rule": lambda r: r["identity"]["split"] == "slme",
    },
    {
        "id": "dfpt_skipped_gapless", "block": "vasp_dfpt",
        "status": Status.NOT_APPLICABLE, "expected_n": 21,
        "reason": "VASP found no band gap, so the run_dielectric_if_gapped gate skipped DFPT",
        "rule": lambda r: r["identity"]["split"] in ("dielectric", "failed_demo"),
    },
    {
        "id": "dfpt_legacy_metallic", "block": "vasp_dfpt",
        "status": Status.NOT_APPLICABLE, "expected_n": 35,
        "reason": "legacy pilot: metallic, so the DFPT leg never ran",
        "rule": lambda r: r["identity"]["split"] == "legacy_pilot",
    },
    {
        "id": "dfpt_nonfinite_tensor", "block": "vasp_dfpt",
        "status": Status.FAILED, "expected_n": 1,
        "reason": "DFPT linear response returned NaN components for a near-metallic cell",
        "rule": lambda r: True,
    },
    {
        "id": "qe_post_dates_legacy", "block": "qe_optical",
        "status": Status.NOT_APPLICABLE, "expected_n": 42,
        "reason": "the QE optical pipeline post-dates the legacy NERSC pilot",
        "rule": lambda r: r["identity"]["split"] == "legacy_pilot",
    },
    {
        "id": "qe_not_submitted_gapless", "block": "qe_optical",
        "status": Status.NOT_APPLICABLE, "expected_n": 21,
        "reason": "VASP found no band gap, so the material was never submitted to QE",
        "rule": lambda r: r["identity"]["split"] in ("dielectric", "failed_demo"),
    },
    {
        "id": "qe_metallic", "block": "qe_optical", "status": "metallic", "expected_n": 78,
        "reason": "QE found no gap, so epsilon.x was not run (see caution_nspin1_magnetic_TM)",
        "rule": lambda r: True,
    },
    {
        "id": "qe_failed", "block": "qe_optical", "status": Status.FAILED, "expected_n": 8,
        "reason": "QE nscf did not converge (c_bands); the error string is retained",
        "rule": lambda r: True,
    },
    {
        "id": "slme_no_spectrum", "block": "slme",
        "status": Status.NOT_APPLICABLE, "expected_n": 149,
        "reason": "no QE absorption spectrum for this material, so no SLME",
        "rule": lambda r: r["qe_optical"].get("status") != "done",
    },
    {
        "id": "slme_negative_gap", "block": "slme",
        "status": Status.NOT_APPLICABLE, "expected_n": 2,
        "reason": "QE indirect gap is negative (semimetallic overlap) -- SLME undefined",
        "rule": lambda r: r["qe_optical"].get("status") == "done",
    },
    {
        "id": "hyperbolic_out_of_scope", "block": "hyperbolic",
        "status": Status.NOT_APPLICABLE, "expected_n": 220,
        "reason": ("no spectrum, or the cell is tri/mono/cubic where the cell-frame "
                   "diagonal is not the principal-axis dielectric tensor"),
        "rule": lambda r: True,
    },
    {
        "id": "rl_no_surrogate_record_legacy", "block": "rl_surrogate",
        "status": Status.NOT_APPLICABLE, "expected_n": 38,
        "reason": ("legacy NERSC pilot: no surrogate worklist survives, and the "
                   "as-generated structure was not carried over to this machine either"),
        "rule": lambda r: (r["identity"]["split"] == "legacy_pilot"
                           and not r["lineage"].get("rl_cif_as_generated")),
    },
    {
        "id": "rl_no_surrogate_record_legacy_with_cif", "block": "rl_surrogate",
        "status": Status.NOT_APPLICABLE, "expected_n": 4,
        "reason": ("legacy NERSC pilot: the as-generated structure survives in sparc_v1/cifs/, "
                   "but no surrogate worklist does, and the candidate id encodes no RL "
                   "step/index to trace back to"),
        "rule": lambda r: r["identity"]["split"] == "legacy_pilot",
    },
    {
        "id": "rl_sun_flags_ambiguous", "block": "rl_surrogate",
        "status": Status.PARTIAL, "expected_n": 6,
        "reason": ("several RL samples in that step share the formula, energy and site count, "
                   "so the S.U.N. flags are left null rather than guessed"),
        "rule": lambda r: r["rl_surrogate"].get("stability_join_method") == "ambiguous",
    },
    {
        "id": "ehull_approximate_W", "block": "vasp_energetics",
        "status": Status.PARTIAL, "expected_n": 4,
        "reason": "W_sv substituted for MP's W_pv POTCAR, so e_hull is approximate",
        "rule": lambda r: True,
    },
]

# Field-level absences: a block is `ok` overall, but one optional field was never
# recorded upstream. Declared here so they read as known limits rather than holes.
FIELD_ABSENCES = [
    {
        "id": "formation_energy_dielectric_campaign",
        "block": "vasp_energetics", "field": "formation_energy_ev_atom",
        "expected_n": 103,
        "reason": ("the dielectric-campaign e_hull report (dfpt_dft_results.csv) recorded "
                   "E_above_hull but not the formation energy; recovering it would need a "
                   "fresh Materials Project elemental-reference query, which this build "
                   "deliberately does not do (e_hull is reused verbatim, not re-derived)"),
        "rule": lambda r: r["identity"]["split"] in ("dielectric", "failed_demo"),
    },
]

# Any block landing on one of these has a REAL gap and must be reported, not explained.
GAP_STATUSES = (Status.MISSING, Status.NOT_RUN)


def new_record(mat_id: str) -> dict:
    """An empty record with every block present and every status pre-set to `missing`."""
    rec: dict = {b: {} for b in BLOCKS}
    rec["identity"]["mat_id"] = mat_id
    for b in STATUS_BLOCKS:
        rec[b]["status"] = Status.MISSING
        rec[b]["status_reason"] = "not populated by any source"
    return rec


# --------------------------------------------------------------------------------------
# Flattening: which nested paths become columns of materials.csv.  Anything not listed
# here (structures, tensors, spectra, Born charges) stays in materials.jsonl / sidecars.
# --------------------------------------------------------------------------------------
FLAT_FIELDS: list[tuple[str, str]] = [
    # (column name, dotted path into the record)
    ("mat_id", "identity.mat_id"),
    ("mat_id_fs", "identity.mat_id_fs"),
    ("split", "identity.split"),
    ("formula", "identity.formula_pretty"),
    ("formula_full", "identity.formula_full"),
    ("chemsys", "identity.chemsys"),
    ("nsites", "identity.nsites"),
    ("nelements", "identity.nelements"),
    ("group", "identity.group"),
    ("failure_mode", "identity.failure_mode"),
    ("mongo_campaign", "identity.mongo_campaign"),

    ("spacegroup_symbol", "vasp_structure.spacegroup_symbol"),
    ("spacegroup_number", "vasp_structure.spacegroup_number"),
    ("crystal_system", "vasp_structure.crystal_system"),
    ("a_A", "vasp_structure.a"),
    ("b_A", "vasp_structure.b"),
    ("c_A", "vasp_structure.c"),
    ("alpha_deg", "vasp_structure.alpha"),
    ("beta_deg", "vasp_structure.beta"),
    ("gamma_deg", "vasp_structure.gamma"),
    ("volume_A3", "vasp_structure.volume_A3"),
    ("density_g_cm3", "vasp_structure.density_g_cm3"),
    ("cif_relaxed", "vasp_structure.cif_relaxed"),
    ("cif_as_generated", "lineage.rl_cif_as_generated"),

    ("energy_eV", "vasp_energetics.energy_ev"),
    ("energy_per_atom_eV", "vasp_energetics.energy_per_atom_ev"),
    ("e_hull_eV_atom", "vasp_energetics.e_hull_ev_atom"),
    ("e_hull_approx", "vasp_energetics.e_hull_approx"),
    ("formation_energy_eV_atom", "vasp_energetics.formation_energy_ev_atom"),
    ("e_hull_source", "vasp_energetics.e_hull_source"),

    ("band_gap_eV", "vasp_electronic.bandgap_ev"),
    ("direct_gap_eV", "vasp_electronic.direct_gap_ev"),
    ("is_gap_direct", "vasp_electronic.is_gap_direct"),
    ("is_metal_vasp", "vasp_electronic.is_metal"),
    ("vbm_eV", "vasp_electronic.vbm_ev"),
    ("cbm_eV", "vasp_electronic.cbm_ev"),

    ("dfpt_status", "vasp_dfpt.status"),
    ("eps_electronic_iso", "vasp_dfpt.eps_electronic_iso"),
    ("eps_electronic_xx", "vasp_dfpt.eps_electronic_xx"),
    ("eps_electronic_yy", "vasp_dfpt.eps_electronic_yy"),
    ("eps_electronic_zz", "vasp_dfpt.eps_electronic_zz"),
    ("eps_ionic_iso", "vasp_dfpt.eps_ionic_iso"),
    ("eps_total_iso", "vasp_dfpt.eps_total_iso"),
    ("eps_electronic_eig_min", "vasp_dfpt.eps_electronic_eig_min"),
    ("eps_electronic_eig_max", "vasp_dfpt.eps_electronic_eig_max"),
    ("eps_anisotropy", "vasp_dfpt.eps_anisotropy"),
    ("n_imaginary_modes", "vasp_dfpt.n_imaginary_modes"),
    ("n_near_zero_modes", "vasp_dfpt.n_near_zero_modes"),
    ("dfpt_tensor_file", "vasp_dfpt.tensor_file"),

    ("qe_status", "qe_optical.status"),
    ("qe_indirect_gap_eV", "qe_optical.indirect_gap_ev"),
    ("qe_direct_gap_eV", "qe_optical.direct_gap_ev"),
    ("qe_is_metal", "qe_optical.is_metal"),
    ("qe_nelec", "qe_optical.nelec"),
    ("qe_eps2_peak", "qe_optical.eps2_peak"),
    ("qe_eps2_peak_eV", "qe_optical.eps2_peak_ev"),
    ("qe_eps1_zero_freq_x", "qe_optical.eps1_zero_freq_x"),
    ("qe_eps1_zero_freq_y", "qe_optical.eps1_zero_freq_y"),
    ("qe_eps1_zero_freq_z", "qe_optical.eps1_zero_freq_z"),
    ("qe_error", "qe_optical.error"),

    ("slme_eta", "slme.eta"),
    ("slme_eta_pct", "slme.eta_pct"),
    ("slme_gap_used_eV", "slme.gap_used_ev"),
    ("slme_thickness_um", "slme.thickness_um"),
    ("slme_temperature_K", "slme.temperature_k"),
    ("slme_eta_pct_recomputed", "slme.eta_pct_recomputed"),
    ("slme_roundtrip_delta_pp", "slme.roundtrip_delta_pp"),

    ("is_hyperbolic", "hyperbolic.is_hyperbolic"),
    ("hyperbolic_score", "hyperbolic.score"),
    ("hyperbolic_n_clean_windows", "hyperbolic.n_clean_windows"),
    ("hyperbolic_clean_bandwidth_eV", "hyperbolic.clean_bandwidth_ev"),
    ("hyperbolic_best_type", "hyperbolic.best_type"),
    ("hyperbolic_best_fom", "hyperbolic.best_fom"),
    ("hyperbolic_best_band", "hyperbolic.best_band"),
    ("hyperbolic_status", "hyperbolic.status"),

    ("rl_run_name", "lineage.rl_run_name"),
    ("rl_checkpoint", "lineage.rl_checkpoint"),
    ("rl_generator", "lineage.rl_generator"),
    ("rl_step", "lineage.rl_step"),
    ("rl_eval_index", "lineage.rl_eval_index"),
    ("rl_rank", "identity.label_rank"),
    ("surrogate_target", "rl_surrogate.surrogate_target"),
    ("surrogate_band_gap_eV", "rl_surrogate.ml_band_gap_ev"),
    ("surrogate_eta_slme", "rl_surrogate.ml_eta_slme"),
    ("surrogate_r_uni", "rl_surrogate.reward_r_uni"),
    ("surrogate_eps_xx", "rl_surrogate.ml_eps_xx"),
    ("surrogate_eps_yy", "rl_surrogate.ml_eps_yy"),
    ("surrogate_eps_zz", "rl_surrogate.ml_eps_zz"),
    ("surrogate_eps_iso", "rl_surrogate.ml_eps_iso"),
    ("surrogate_in_solar_window", "rl_surrogate.ml_in_solar_window"),
    ("mlip_e_hull_eV_atom", "rl_surrogate.mlip_ehull_ev_atom"),
    ("rl_is_stable", "rl_surrogate.is_stable"),
    ("rl_stability_join", "rl_surrogate.stability_join_method"),

    ("static_run_type", "lineage.static_run_type"),
    ("dielectric_run_type", "lineage.dielectric_run_type"),
    ("vasp_version", "lineage.vasp_version"),
    ("encut_eV", "lineage.encut"),
    ("machine", "lineage.machine"),
    ("qe_functional", "lineage.qe_functional"),
    ("qe_pseudo", "lineage.qe_pseudo"),
    ("qe_version", "lineage.qe_version"),

    ("gap_in_window", "flags.gap_in_window"),
    ("dft_stable", "flags.stable"),
    ("slme_ok", "flags.slme_ok"),
    ("hits_solar_target", "flags.hits_solar_target"),
    ("gap_in_window_qe", "flags.gap_in_window_qe"),
    ("hits_solar_target_qe", "flags.hits_solar_target_qe"),
    ("qe_vs_vasp_gap_delta_eV", "flags.qe_vs_vasp_gap_delta_ev"),
    ("qe_vasp_gap_disagreement", "flags.qe_vasp_gap_disagreement"),
    ("caution_nspin1_magnetic_TM", "flags.caution_nspin1_magnetic_tm"),
    ("caution_molecular_vdW", "flags.caution_molecular_vdw"),
    ("qe_geometry_matches_shipped_cell", "flags.qe_geometry_matches_shipped_cell"),
    ("completeness_class", "flags.completeness_class"),
    ("expected_nulls", "flags.expected_nulls"),

    ("status_vasp_structure", "vasp_structure.status"),
    ("status_vasp_energetics", "vasp_energetics.status"),
    ("status_vasp_electronic", "vasp_electronic.status"),
    ("status_vasp_dfpt", "vasp_dfpt.status"),
    ("status_qe_optical", "qe_optical.status"),
    ("status_slme", "slme.status"),
    ("status_hyperbolic", "hyperbolic.status"),
    ("status_rl_surrogate", "rl_surrogate.status"),
]

# Keys never written into materials.csv (arrays / large blobs live in the JSONL + sidecars).
HEAVY_KEYS = {
    "structure_dict", "cif_p1_text", "cif_symmetrized_text", "eps_electronic_tensor",
    "eps_ionic_tensor", "eps_total_tensor", "born_charges", "normalmode_eigenvals",
    "windows", "eps0_crossings_xyz", "forces", "stress", "composition", "entry",
}


def dig(rec: dict, path: str):
    """``dig(rec, "vasp_dfpt.eps_ionic_iso")`` -> value or None."""
    cur = rec
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def flatten_record(rec: dict) -> dict:
    """Record -> one flat row of materials.csv, in FLAT_FIELDS order.

    Multi-line strings (the retained QE stderr) are collapsed to a single line: the CSV
    is quoted correctly either way, but embedded newlines trip naive parsers and the
    HuggingFace viewer. The untouched original stays in ``materials.jsonl``.
    """
    row = {}
    for col, path in FLAT_FIELDS:
        v = dig(rec, path)
        if isinstance(v, (list, tuple)):
            v = ";".join(str(x) for x in v)
        elif isinstance(v, str) and ("\n" in v or "\r" in v):
            v = " ".join(v.split())
        row[col] = v
    return row
