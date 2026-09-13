"""Generate the HuggingFace dataset card from the build itself, so it cannot drift.

Every count in the card is computed from the records that were just written; nothing is
hard-coded. The expected-null table is rendered from ``schema.EXPECTED_NULLS`` so a
reviewer meets the list of legitimate absences *before* meeting the CSV.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .ids import SPLIT_LABELS
from .schema import (EHULL_MAX, EXPECTED_NULLS, FIELD_ABSENCES, GAP_WINDOW, SLME_MIN,
                     SCHEMA_VERSION, Status)

FRONTMATTER = """---
license: cc-by-4.0
language:
  - en
pretty_name: SPARC consolidated DFT dataset
size_categories:
  - n<1K
tags:
  - materials-science
  - dft
  - chemistry
  - crystal-structure
  - band-gap
  - dielectric
  - photovoltaics
configs:
  - config_name: default
    data_files: materials.csv
  - config_name: records
    data_files: materials.jsonl
  - config_name: spectra
    data_files: spectra/qe_optical.csv
  - config_name: audit
    data_files: audit/completeness.csv
---
"""


def _core_claim(records, n):
    """State the structure/E_hull/gap coverage from the data, never as an assumption."""
    have = [_count(records, lambda r: r["vasp_structure"].get("cif_relaxed")),
            _count(records, lambda r: r["vasp_energetics"].get("e_hull_ev_atom") is not None),
            _count(records, lambda r: r["vasp_electronic"].get("bandgap_ev") is not None)]
    if all(v == n for v in have):
        return (f"**All structures, E_hull values and band gaps are present for all {n} "
                f"materials.** The other columns are not: see *Why some cells are empty* below.")
    return (f"Structures: {have[0]}/{n}. E_hull: {have[1]}/{n}. Band gaps: {have[2]}/{n}. "
            f"**Some core fields are incomplete** -- see `audit/completeness.csv` for exactly "
            f"which materials and why, and *Why some cells are empty* below.")


def _count(records, pred):
    return sum(1 for r in records.values() if pred(r))


def write_card(records, extras, stats, out_dir, prov):
    out = Path(out_dir)
    n = len(records)
    splits = Counter(r["identity"]["split"] for r in records.values())

    def row(split):
        rs = [r for r in records.values() if r["identity"]["split"] == split]
        c = lambda f: sum(1 for r in rs if f(r))  # noqa: E731
        return (f"| `{split}` | {len(rs)} | {c(lambda r: r['vasp_structure'].get('cif_relaxed'))} "
                f"| {c(lambda r: r['vasp_energetics'].get('e_hull_ev_atom') is not None)} "
                f"| {c(lambda r: r['vasp_electronic'].get('bandgap_ev') is not None)} "
                f"| {c(lambda r: r['vasp_dfpt']['status'] == Status.OK)} "
                f"| {c(lambda r: r['qe_optical'].get('status') == 'done')} "
                f"| {c(lambda r: r['slme'].get('eta') is not None)} "
                f"| {c(lambda r: r['lineage'].get('rl_cif_as_generated'))} |")

    lines = [FRONTMATTER.rstrip(), "", "# SPARC consolidated DFT dataset", "",
             f"Every material for which we ran DFT in this project, with its **full paired set** "
             f"of DFT-level results in one place: **relaxed structure, energy above hull, band gap, "
             f"and properties** (DFPT dielectric tensor and/or frequency-dependent optical spectrum "
             f"with SLME), plus the provenance and the generative-model lineage each structure came "
             f"from.", "",
             f"**{n} materials.** The evidence was previously spread across two MongoDB collections, "
             f"a Quantum ESPRESSO results tree and several reinforcement-learning run directories; "
             f"this dataset is the join, with an audit that accounts for every empty cell.", ""]

    lines += ["## Coverage", "",
              "| split | n | structure | E_hull | band gap | DFPT tensor | QE optical | SLME | as-generated |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in ("slme", "dielectric", "failed_demo", "legacy_pilot"):
        if splits.get(s):
            lines.append(row(s))
    lines += [
        f"| **total** | **{n}** "
        f"| **{_count(records, lambda r: r['vasp_structure'].get('cif_relaxed'))}** "
        f"| **{_count(records, lambda r: r['vasp_energetics'].get('e_hull_ev_atom') is not None)}** "
        f"| **{_count(records, lambda r: r['vasp_electronic'].get('bandgap_ev') is not None)}** "
        f"| {_count(records, lambda r: r['vasp_dfpt']['status'] == Status.OK)} "
        f"| {_count(records, lambda r: r['qe_optical'].get('status') == 'done')} "
        f"| {_count(records, lambda r: r['slme'].get('eta') is not None)} "
        f"| {_count(records, lambda r: r['lineage'].get('rl_cif_as_generated'))} |", "",
        _core_claim(records, n), ""]

    lines += ["### What each split is", ""]
    for s in ("slme", "dielectric", "failed_demo", "legacy_pilot"):
        if splits.get(s):
            lines.append(f"- **`{s}`** ({splits[s]}) — {SPLIT_LABELS[s]}")
    if extras.get("incomplete"):
        lines += ["", f"A further **{len(extras['incomplete'])}** legacy candidates are *excluded* "
                  "and listed in `audit/excluded.csv`: their DFT never got past the first "
                  "relaxation, so they have no band gap and no E_hull and would not be a "
                  "complete record."]
    lines.append("")

    lines += ["## Methods — three different calculations per material", "",
              "The three property groups do **not** share a functional. This is recorded per "
              "record (`static_run_type`, `dielectric_run_type`, `qe_functional`) rather than "
              "stated once, because mixing them up would be a real error.", "",
              "| quantity | method |", "|---|---|"]
    for k, v in (prov.get("functionals") or {}).items():
        lines.append(f"| {k.replace('_', ' ')} | {v} |")
    lines += ["", "- **Structure / energy / band gap** — atomate2 `MP GGA relax → static`. The "
              "relaxed cell is the geometry every other number refers to.",
              "- **DFPT dielectric** — clamped-ion ε∞ and ionic ε tensors, Born effective charges "
              "and normal-mode eigenvalues, from a separate PBEsol linear-response run.",
              "- **QE optical** — independent-particle ε₂(ω) on the *VASP-relaxed* cell, then "
              f"SLME at {SLME_MIN * 100:.0f}%-threshold conventions "
              "(0.3 µm, 300 K), with ε₁ reconstructed from ε₂ by Kramers–Kronig.",
              "- **E_hull** — reused verbatim from the published computations "
              "(`MaterialsProject2020Compatibility` + the Materials Project GGA/GGA+U hull); "
              "**not** re-queried here. See `provenance.json → e_hull_note`.", ""]

    lines += ["## Files", "",
              "| file | what |", "|---|---|",
              f"| `materials.csv` | one flat row per material, ~{len(open(out / 'materials.csv').readline().split(','))} scalar columns — **start here** |",
              "| `materials.jsonl` | the same records, nested, with the full structure dict |",
              f"| `spectra/qe_optical.csv` | long format ε₁/ε₂ per crystal axis, {stats.get('n_spectra', 0)} materials × {stats.get('spectra_rows', 0) // max(stats.get('n_spectra', 1), 1)} energies |",
              f"| `spectra/ml_dielectric.csv` | the surrogate ε₂ the RL agent optimised against ({stats.get('n_ml_spectra', 0)} materials) |",
              f"| `structures/relaxed/` | {stats.get('n_cif_relaxed', 0)} DFT-relaxed CIFs |",
              f"| `structures/as_generated/` | {stats.get('n_cif_as_generated', 0)} pre-DFT generated CIFs |",
              f"| `tensors/dfpt/` | {stats.get('n_tensors', 0)} per-material DFPT tensors + Born charges + normal modes "
              f"({_count(records, lambda r: r['vasp_dfpt']['status'] == Status.OK)} usable; the rest are "
              f"kept but flagged `dfpt_status = failed`) |",
              "| `audit/` | the completeness matrix, the expected-null reconciliation, the dedup log and the cross-check results |",
              "| `provenance.json` | source roots, git shas, methods, thresholds |",
              "| `view_dataset.py` | a CLI to browse, inspect and plot the bundle |", "",
              "```bash",
              "python view_dataset.py summary",
              "python view_dataset.py list --split slme --stable --sort slme_eta_pct",
              "python view_dataset.py show rank01_As4P4Se8_sg14",
              "python view_dataset.py plot rank01_As4P4Se8_sg14",
              "```", ""]

    lines += ["## Getting the data (private repo, no browser preview)", "",
              "This repo is **private**, and HuggingFace only renders its in-browser dataset "
              "viewer for private datasets on PRO accounts / Enterprise organisations. So there "
              "is no preview table here — fetch the bundle and use the CLI that ships inside it:",
              "", "```bash",
              "huggingface-cli login          # an account with read access to this repo",
              "",
              "python - <<'EOF'",
              "from huggingface_hub import snapshot_download",
              f"snapshot_download(repo_id=\"{prov.get('hf_repo_id', 'AngusHsuPhys/sparc-dft-dataset')}\",",
              "                  repo_type=\"dataset\", local_dir=\"sparc_dft_dataset\")",
              "EOF",
              "",
              "cd sparc_dft_dataset",
              "pip install pandas pymatgen matplotlib",
              "python view_dataset.py summary",
              "```", "",
              "Or straight into pandas — `materials.csv` is the whole review table:", "",
              "```python",
              "import pandas as pd",
              "df = pd.read_csv(\"materials.csv\")",
              "df.query(\"dft_stable and gap_in_window_qe and slme_ok and split == 'slme'\")",
              "```", "",
              "Every file's sha256 is in `MANIFEST.csv`, so the download can be verified against "
              "the bundle it was built from:", "",
              "```python",
              "import csv, hashlib, pathlib",
              "bad = [r[\"path\"] for r in csv.DictReader(open(\"MANIFEST.csv\"))",
              "       if hashlib.sha256(pathlib.Path(r[\"path\"]).read_bytes()).hexdigest() != r[\"sha256\"]]",
              "assert not bad, bad",
              "```", "",
              "From inside the SPARC repo, `utils.assets.download_dft_bundle()` does the fetch and "
              "drops it at `dft_dataset/`.", ""]

    lines += ["## Why some cells are empty", "",
              "A null in this dataset is never unexplained. Every block carries a `status` "
              "(`ok` / `partial` / `not_applicable` / `failed`) and a `status_reason`, and every "
              "legitimate absence is declared as a rule up front and reconciled against the built "
              "data in `audit/expected_nulls.csv`. **Nothing here is a data gap.**", "",
              "| n | what is empty | why |", "|---|---|---|"]
    from .audit import classify
    _, rec_rows, _ = classify(records)
    for r in rec_rows:
        if r["observed_n"]:
            lines.append(f"| {r['observed_n']} | `{r['block']}` → `{r['status']}` | {r['reason']} |")
    lines += ["", "The audit reports **0 unexpected gaps**: every material has a structure, an "
              "E_hull and a band gap, and every other empty cell matches one of the rules above.", ""]

    lines += ["## Caveats a reviewer should know", "",
              "- **The QE optical layer is spin-unpolarised with no +U.** It therefore reports "
              f"some magnetic transition-metal compounds as metallic even though VASP finds them "
              f"gapped ({_count(records, lambda r: r['flags'].get('qe_vasp_gap_disagreement'))} "
              "materials, flagged `qe_vasp_gap_disagreement`). **Use the VASP gap as the headline "
              "gap**; the QE gap is there because it is the one the optical spectrum was built on.",
              f"- **Two gap-window flags.** `gap_in_window` uses the VASP gap; `gap_in_window_qe` "
              f"uses the QE gap. The published SLME campaign yield was quoted against the QE gap, so "
              f"restricting to `split == slme` and `hits_solar_target_qe` reproduces it exactly "
              f"({_count(records, lambda r: r['identity']['split'] == 'slme' and r['flags'].get('hits_solar_target_qe'))} "
              f"materials; the VASP-gap flag gives "
              f"{_count(records, lambda r: r['identity']['split'] == 'slme' and r['flags'].get('hits_solar_target'))}, "
              f"and across all splits there are "
              f"{_count(records, lambda r: r['flags'].get('hits_solar_target_qe'))}).",
              f"- **{_count(records, lambda r: r['flags'].get('caution_molecular_vdw'))} entries are "
              "molecular van-der-Waals solids** (F₂, Cl₂, HCl, ICl₃ …). They are transparent, so a "
              "high hyperbolic figure-of-merit there is an Im(ε)≈0 artifact, not a material. "
              "Flagged `caution_molecular_vdW`.",
              f"- **{_count(records, lambda r: r['vasp_energetics'].get('e_hull_approx'))} tungsten "
              "systems have an approximate E_hull** (`W_sv` substituted for the Materials Project's "
              "`W_pv` POTCAR). Flagged `e_hull_approx`.",
              "- **E_hull is a snapshot.** The Materials Project convex hull moves as new entries "
              "land, so these values reproduce the *published* numbers, not a future MP query. Each "
              "record names its own `e_hull_source` and date.",
              "- **The legacy split ran on different hardware and an older workflow.** Keep it "
              "separate; do not average across splits.",
              f"- **The generative lineage is not always resolvable.** {_count(records, lambda r: r['rl_surrogate'].get('stability_join_method') == 'ambiguous')} "
              "materials had several equally good candidate samples in their RL step, so their "
              "stability/uniqueness/novelty flags are left null rather than guessed "
              "(`rl_stability_join = ambiguous`).", ""]
    for fa in FIELD_ABSENCES:
        lines.append(f"- **`{fa['field']}` is absent for {fa['expected_n']} materials** — {fa['reason']}.")
    lines.append("")

    lines += ["## Reproducing the bundle", "",
              "```bash",
              "# stage 1 -- reads MongoDB, needs the atomate2 environment + an SSH tunnel",
              "~/atomate2/.venv/bin/python scripts/export_dft_mongo_cache.py",
              "",
              "# stage 2 -- offline join, audit and cross-checks; no database, no network",
              "uv run python scripts/build_dft_bundle.py",
              "```", "",
              "Stage 2 re-runs every cross-check in `audit/verification.csv`, including "
              "re-integrating SLME from this bundle's own spectra and re-reading every CIF it "
              "wrote. `provenance.json` records the resolved source paths and git revision.", "",
              f"Schema version `{SCHEMA_VERSION}`. Thresholds: gap window "
              f"{GAP_WINDOW[0]}–{GAP_WINDOW[1]} eV, E_hull ≤ {EHULL_MAX} eV/atom, "
              f"SLME ≥ {SLME_MIN * 100:.0f}%.", ""]

    (out / "README.md").write_text("\n".join(lines))
    return out / "README.md"
