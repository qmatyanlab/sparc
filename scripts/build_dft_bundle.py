#!/usr/bin/env python3
"""STAGE 2 of the DFT dataset build: join every source into one reviewable bundle.

Consolidates the DFT evidence that is currently spread across ``~/atomate2`` (MongoDB),
``~/qe_ht`` and the ``exp_res/`` RL runs into a single self-describing dataset in which
every material we ran DFT on carries its full paired set -- **structure, E_hull, band
gap, and properties** (DFPT dielectric tensor and/or QE optical spectrum + SLME) -- with
explicit provenance and explicitly *reasoned* nulls.

Runs entirely offline against the stage-1 cache: no MongoDB, no network, no dependency
that is not already in the pinned environment.

Usage:
  # stage 1 first (needs the Mongo tunnel + the atomate2 env):
  ~/atomate2/.venv/bin/python scripts/export_dft_mongo_cache.py

  uv run python scripts/build_dft_bundle.py                  # full build + audit + verify
  uv run python scripts/build_dft_bundle.py --skip-spectra   # fast, no 25 MB spectra pass
  uv run python scripts/build_dft_bundle.py --strict         # non-zero exit on any gap
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.dft_bundle import audit, build, card, sources as S, verify  # noqa: E402
from utils.dft_bundle.schema import (  # noqa: E402
    EHULL_MAX, GAP_WINDOW, SLME_MIN, SCHEMA_VERSION,
)

DEFAULT_OUT = ROOT / "dft_dataset"
DEFAULT_CACHE = DEFAULT_OUT / "_cache" / "mongo_extract.json.gz"


def _git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="stage-1 Mongo extract")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="bundle directory")
    ap.add_argument("--qe-root", default=str(S.DEFAULT_QE_ROOT))
    ap.add_argument("--exp-res", default=str(S.DEFAULT_EXP_RES))
    ap.add_argument("--skip-spectra", action="store_true",
                    help="do not write spectra/qe_optical.csv (much faster)")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if the audit finds a real gap or a check FAILs")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    qe_root, exp_res = Path(args.qe_root), Path(args.exp_res)

    for label, p in (("qe_ht", qe_root), ("exp_res", exp_res)):
        if not p.exists():
            sys.exit(f"{label} root not found: {p}\n"
                     f"Pass --{label.replace('_', '-')} or set the matching env var. "
                     f"exp_res/ is gitignored -- on a fresh clone fetch the run bundles with "
                     f"scripts/download_paper_runs.py first.")

    missing = S.check_rl_sources(exp_res)
    if missing:
        print("WARNING: RL lineage inputs are missing -- the surrogate/lineage blocks will be\n"
              "         incomplete and the audit will flag them as UNDECLARED:")
        for m in missing:
            print(f"           {m}")
        print("         exp_res/ is gitignored; fetch the run bundles with\n"
              "         scripts/download_paper_runs.py, or point --exp-res at them.\n")

    print(f"Building the consolidated DFT dataset -> {out}")
    print(f"  cache   : {args.cache}")
    print(f"  qe_ht   : {qe_root}")
    print(f"  exp_res : {exp_res}\n")

    records, extras = build.merge_records(args.cache, qe_root, exp_res, ROOT)
    stats = build.write_bundle(records, extras, out, qe_root,
                               write_spectra=not args.skip_spectra)

    # ---- audit ------------------------------------------------------------------
    matrix, reconciliation, gaps = audit.classify(records)
    build._write_csv(out / "audit/completeness.csv", matrix)
    build._write_csv(out / "audit/expected_nulls.csv", reconciliation)

    # ---- verification -----------------------------------------------------------
    checks = None
    if not args.skip_verify:
        print("\nRunning cross-checks against the previously published artifacts ...")
        checks = verify.run_checks(records, extras, out, qe_root, exp_res,
                                   skip_slme_roundtrip=args.skip_spectra)
        verify.write_reports(checks, out)
        # the round-trip fills slme.eta_pct_recomputed, so re-emit the tables
        build.write_bundle(records, extras, out, qe_root, write_spectra=False, verbose=False)
        build._write_csv(out / "audit/completeness.csv", matrix)

    # ---- provenance + manifest --------------------------------------------------
    prov = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_materials": len(records),
        "stats": stats,
        "sources": {
            "mongo_cache": {"path": str(Path(args.cache).resolve()),
                            **{k: v for k, v in extras["cache_meta"].items()}},
            "qe_ht_root": str(qe_root.resolve()),
            "exp_res_root": str(exp_res.resolve()),
            "sparc_v1_git_sha": _git_sha(ROOT),
        },
        "thresholds": {"gap_window_ev": list(GAP_WINDOW), "e_hull_max_ev_atom": EHULL_MAX,
                       "slme_min_fraction": SLME_MIN},
        "e_hull_note": (
            "E_hull values are the PUBLISHED ones, reused verbatim rather than re-queried: "
            "MaterialsProject2020Compatibility against the Materials Project GGA/GGA+U hull, "
            "computed 2026-07-31 (SLME split), 2026-08-02 (dielectric/failure-demo splits) and "
            "2026-06/07 on NERSC (legacy split). The MP hull drifts over time, so these "
            "reproduce the published numbers, not a future MP snapshot. Each record carries its "
            "own e_hull_source."),
        "functionals": {
            "structure_energy_gap": "MP GGA (PBE(+U)), PAW-PBE, ENCUT 520 eV, VASP 6.4.2",
            "dfpt_dielectric": "PBEsol, LEPSILON + IBRION=8, VASP 6.4.2",
            "qe_optical": "PBE + SG15 ONCV v1.2, QE 7.5, epsilon.x IPA, nspin=1, no +U",
        },
        "hf_repo_id": __import__("utils.assets", fromlist=["x"]).dft_dataset_repo_id(),
        "hf_repo_type": "dataset",
        "hf_private": True,
        "verification": None if checks is None else
            {r["check"]: r["verdict"] for r in checks.rows},
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))

    # ---- dataset card + the viewer, so the bundle stands alone after a snapshot download
    card.write_card(records, extras, stats, out, prov)
    viewer = ROOT / "scripts" / "view_dft_bundle.py"
    if viewer.exists():
        (out / "view_dataset.py").write_text(viewer.read_text())

    ok = audit.summarize(records, matrix, reconciliation, gaps, extras)

    if checks is not None:
        print("\nCross-checks vs the previously published artifacts:")
        print(f"  {'check':38s} {'scope':22s} {'n':>4s} {'fail':>5s} {'max|d|':>12s}  verdict")
        for r in checks.rows:
            md = "" if r["max_abs_delta"] is None else f"{r['max_abs_delta']:.3e}"
            print(f"  {r['check']:38s} {r['scope']:22s} {r['n_compared']:4d} "
                  f"{r['n_fail']:5d} {md:>12s}  {r['verdict']}")
        nfail = sum(1 for r in checks.rows if r["verdict"] == "FAIL")
        print(f"\n  {len(checks.rows) - nfail}/{len(checks.rows)} checks PASS")
        ok = ok and nfail == 0

    manifest = build.write_manifest(out)
    total = sum(m["bytes"] for m in manifest)
    print(f"\nBundle: {len(manifest)} files, {total / 1e6:.1f} MB at {out}")
    print("Next:   uv run python scripts/view_dft_bundle.py summary")
    print("        uv run python scripts/upload_dft_bundle.py --dry-run")

    return 0 if (ok or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
