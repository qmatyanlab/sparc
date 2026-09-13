"""Completeness audit: prove that every material carries the full DFT-level set.

The point of this module is that a ``null`` must never be ambiguous. Every legitimate
absence is pre-declared in ``schema.EXPECTED_NULLS`` with the count we expect to see, so:

  * an expected null shows up as ``na:<rule id>`` in the matrix and is reconciled;
  * an *unexpected* null lands on ``missing`` and is reported as a real gap;
  * and a change upstream shows up as a rule COUNT MISMATCH rather than a silent hole.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .schema import (EXPECTED_NULLS, FIELD_ABSENCES, GAP_STATUSES, STATUS_BLOCKS,
                     Status)

# Scalars that must be present for a material to count as "DFT-complete".
CRITICAL_FIELDS = [
    ("structure", "vasp_structure", "cif_relaxed"),
    ("band_gap", "vasp_electronic", "bandgap_ev"),
    ("energy", "vasp_energetics", "energy_per_atom_ev"),
    ("e_hull", "vasp_energetics", "e_hull_ev_atom"),
    ("spacegroup", "vasp_structure", "spacegroup_number"),
]


def classify(records):
    """-> (matrix rows, rule reconciliation rows, real-gap rows)."""
    # 1. assign every non-ok block status to the first declared rule that matches
    assigned: dict[str, list[str]] = defaultdict(list)      # rule id -> [mat_id]
    unexplained: list[dict] = []

    for mat_id, rec in sorted(records.items()):
        for block in STATUS_BLOCKS:
            st = rec[block].get("status")
            if st in (Status.OK, "done"):
                continue
            rule = next((r for r in EXPECTED_NULLS
                         if r["block"] == block and r["status"] == st and r["rule"](rec)), None)
            if rule is not None:
                assigned[rule["id"]].append(mat_id)
                rec[block]["expected_null_rule"] = rule["id"]
            else:
                unexplained.append({
                    "mat_id": mat_id, "block": block, "status": st,
                    "status_reason": rec[block].get("status_reason"),
                    "severity": "GAP" if st in GAP_STATUSES else "UNDECLARED",
                })

    # 2. reconcile declared vs observed
    reconciliation = [{
        "rule_id": r["id"], "block": r["block"], "status": r["status"],
        "expected_n": r["expected_n"], "observed_n": len(assigned[r["id"]]),
        "match": len(assigned[r["id"]]) == r["expected_n"],
        "reason": r["reason"],
    } for r in EXPECTED_NULLS]

    # 3. per-material presence matrix
    matrix = []
    for mat_id, rec in sorted(records.items()):
        row = {"mat_id": mat_id, "split": rec["identity"]["split"],
               "formula": rec["identity"].get("formula_pretty")}
        n_missing = n_na = 0
        for block in STATUS_BLOCKS:
            st = rec[block].get("status")
            if st in (Status.OK, "done"):
                cell = "ok"
            elif st in GAP_STATUSES:
                cell = f"missing:{st}"
                n_missing += 1
            else:
                cell = f"{st}:{rec[block].get('expected_null_rule', 'undeclared')}"
                n_na += 1
            row[block] = cell
        for name, block, field in CRITICAL_FIELDS:
            present = rec[block].get(field) is not None
            row[f"has_{name}"] = present
            if not present:
                n_missing += 1
        row["n_missing"] = n_missing
        row["n_expected_null"] = n_na
        row["completeness_class"] = rec["flags"]["completeness_class"]
        matrix.append(row)

    # 4. field-level absences inside otherwise-ok blocks
    for fa in FIELD_ABSENCES:
        observed = [m for m, rec in records.items()
                    if rec[fa["block"]].get(fa["field"]) is None and fa["rule"](rec)]
        undeclared = [m for m, rec in records.items()
                      if rec[fa["block"]].get(fa["field"]) is None and not fa["rule"](rec)]
        reconciliation.append({
            "rule_id": fa["id"], "block": f'{fa["block"]}.{fa["field"]}',
            "status": "field_absent", "expected_n": fa["expected_n"],
            "observed_n": len(observed), "match": len(observed) == fa["expected_n"]
            and not undeclared, "reason": fa["reason"],
        })

    # every non-ok block that no rule explains: real gaps AND undeclared statuses.
    # summarize() separates them; both need surfacing, neither is filtered here.
    return matrix, reconciliation, unexplained


def summarize(records, matrix, reconciliation, gaps, extras=None, out=print):
    """Print the human verdict. Returns True when nothing unexpected was found."""
    n = len(records)
    by_split = Counter(r["identity"]["split"] for r in records.values())

    out("")
    out("=" * 78)
    out("COMPLETENESS AUDIT")
    out("=" * 78)

    out("\nPer split:")
    out(f"  {'split':16s} {'n':>4s} {'struct':>7s} {'gap':>5s} {'e_hull':>7s} "
        f"{'DFPT':>5s} {'QE':>4s} {'SLME':>5s} {'as-gen':>7s}")
    for split in sorted(by_split):
        rs = [r for r in records.values() if r["identity"]["split"] == split]
        out(f"  {split:16s} {len(rs):4d} "
            f"{sum(1 for r in rs if r['vasp_structure'].get('cif_relaxed')):7d} "
            f"{sum(1 for r in rs if r['vasp_electronic'].get('bandgap_ev') is not None):5d} "
            f"{sum(1 for r in rs if r['vasp_energetics'].get('e_hull_ev_atom') is not None):7d} "
            f"{sum(1 for r in rs if r['vasp_dfpt'].get('status') == Status.OK):5d} "
            f"{sum(1 for r in rs if r['qe_optical'].get('status') == 'done'):4d} "
            f"{sum(1 for r in rs if r['slme'].get('eta') is not None):5d} "
            f"{sum(1 for r in rs if r['lineage'].get('rl_cif_as_generated')):7d}")
    tot = lambda f: sum(1 for r in records.values() if f(r))  # noqa: E731
    out(f"  {'TOTAL':16s} {n:4d} "
        f"{tot(lambda r: r['vasp_structure'].get('cif_relaxed')):7d} "
        f"{tot(lambda r: r['vasp_electronic'].get('bandgap_ev') is not None):5d} "
        f"{tot(lambda r: r['vasp_energetics'].get('e_hull_ev_atom') is not None):7d} "
        f"{tot(lambda r: r['vasp_dfpt'].get('status') == Status.OK):5d} "
        f"{tot(lambda r: r['qe_optical'].get('status') == 'done'):4d} "
        f"{tot(lambda r: r['slme'].get('eta') is not None):5d} "
        f"{tot(lambda r: r['lineage'].get('rl_cif_as_generated')):7d}")

    out("\nExpected nulls (declared up front, then reconciled against what was built):")
    out(f"  {'rule':30s} {'block':17s} {'status':16s} {'exp':>4s} {'obs':>4s}  ok")
    bad_rules = []
    for r in reconciliation:
        mark = "OK " if r["match"] else "!! "
        if not r["match"]:
            bad_rules.append(r)
        out(f"  {r['rule_id']:30s} {r['block']:17s} {r['status']:16s} "
            f"{r['expected_n']:4d} {r['observed_n']:4d}  {mark}")

    hard = [g for g in gaps if g["severity"] == "GAP"]
    soft = [g for g in gaps if g["severity"] == "UNDECLARED"]
    if hard:
        out(f"\nREAL GAPS -- {len(hard)} block(s) with no data and no explanation:")
        for g in hard[:40]:
            out(f"  {g['mat_id']:48s} {g['block']:17s} {g['status']}")
    if soft:
        out(f"\nUNDECLARED non-ok blocks -- {len(soft)} (present but not covered by a rule):")
        for g in soft[:40]:
            out(f"  {g['mat_id']:48s} {g['block']:17s} {g['status']}  {g['status_reason'][:60]}")

    crit = [m for m in matrix if m["n_missing"]]
    if crit:
        out(f"\nMaterials missing a critical DFT field: {len(crit)}")
        for m in crit[:20]:
            out(f"  {m['mat_id']:48s} n_missing={m['n_missing']}")

    if extras and extras.get("incomplete"):
        out(f"\nExcluded (listed in audit/excluded.csv): {len(extras['incomplete'])} legacy "
            f"materials whose DFT never got past the first relaxation "
            f"(no static -> no gap, no e_hull).")

    n_na = sum(m["n_expected_null"] for m in matrix)
    ok = not hard and not crit and not bad_rules
    out("")
    if ok:
        out(f"ALL {n} MATERIALS COMPLETE -- 0 unexpected gaps "
            f"({n_na} expected nulls, all reconciled to a declared rule)"
            + (f"; {len(soft)} undeclared" if soft else ""))
    else:
        out(f"INCOMPLETE: {len(hard)} real gaps, {len(crit)} materials missing a critical "
            f"field, {len(bad_rules)} rule count mismatches")
    out("=" * 78)
    return ok
