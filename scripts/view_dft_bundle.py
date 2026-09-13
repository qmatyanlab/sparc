#!/usr/bin/env python3
"""Inspect the consolidated DFT dataset bundle: browse, show, plot, summarise.

Standalone -- needs only pandas + pymatgen + matplotlib, and reads the bundle by
relative path, so it also works after a bare ``snapshot_download`` of the HF dataset
(a copy ships inside the bundle as ``view_dataset.py``).

Usage:
  uv run python scripts/view_dft_bundle.py summary
  uv run python scripts/view_dft_bundle.py list --split slme --stable --gapped --sort slme_eta_pct
  uv run python scripts/view_dft_bundle.py show rank01_As4P4Se8_sg14
  uv run python scripts/view_dft_bundle.py plot rank01_As4P4Se8_sg14 -o /tmp/x.png
  uv run python scripts/view_dft_bundle.py structure r001_OsO4_SG173_step96_i15 --fmt cif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
if not (BUNDLE / "materials.csv").exists():                 # running from scripts/
    BUNDLE = BUNDLE.parent / "dft_dataset"

# Categorical hues, fixed order, never cycled: x, y, z crystal axes.
# Validated (light surface #fcfcfb, all pairs): CVD dE 12.6 deutan, normal-vision dE 22.3,
# contrast >= 3:1 -- all six checks PASS.
AXIS_COLORS = {"x": "#0A7BBF", "y": "#D55E00", "z": "#762A83"}
DFT_COLOR, ML_COLOR = "#0A7BBF", "#D55E00"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#dcdcd8"


def _load(bundle):
    import pandas as pd
    df = pd.read_csv(bundle / "materials.csv", low_memory=False)
    return df


def _record(bundle, mat_id):
    with open(bundle / "materials.jsonl") as fh:
        for line in fh:
            if f'"mat_id":"{mat_id}"' in line or f'"mat_id": "{mat_id}"' in line:
                r = json.loads(line)
                if r["identity"]["mat_id"] == mat_id:
                    return r
    return None


def _resolve(bundle, key):
    """Accept the canonical id, the filename-safe id, or a unique prefix."""
    import pandas as pd
    df = _load(bundle)
    for col in ("mat_id", "mat_id_fs"):
        hit = df[df[col] == key]
        if len(hit) == 1:
            return hit.iloc[0]["mat_id"]
    hit = df[df["mat_id"].str.startswith(key)]
    if len(hit) == 1:
        return hit.iloc[0]["mat_id"]
    if len(hit) > 1:
        sys.exit(f"'{key}' is ambiguous: {', '.join(hit['mat_id'].head(8))} ...")
    sys.exit(f"no material matches '{key}'. Try: view_dft_bundle.py list")


# ------------------------------------------------------------------------------ list
def cmd_list(args):
    import pandas as pd
    df = _load(args.bundle)
    if args.split:
        df = df[df["split"] == args.split]
    if args.stable:
        df = df[df["dft_stable"] == True]           # noqa: E712
    if args.gapped:
        df = df[df["band_gap_eV"] > 0.01]
    if args.dfpt:
        df = df[df["dfpt_status"] == "ok"]
    if args.sort:
        df = df.sort_values(args.sort, ascending=args.ascending, na_position="last")
    cols = ["mat_id", "split", "formula", "spacegroup_symbol", "band_gap_eV",
            "e_hull_eV_atom", "eps_total_iso", "slme_eta_pct", "completeness_class"]
    with pd.option_context("display.width", 200, "display.max_rows", None,
                           "display.max_colwidth", 46):
        print(df[cols].head(args.limit).to_string(index=False))
    print(f"\n{len(df)} materials shown (of {len(_load(args.bundle))} in the bundle)")
    return 0


# ------------------------------------------------------------------------------ show
def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def cmd_show(args):
    mat_id = _resolve(args.bundle, args.mat_id)
    r = _record(args.bundle, mat_id)
    if r is None:
        sys.exit(f"{mat_id} not found in materials.jsonl")

    ident, lin, vs, ve, vel = (r["identity"], r["lineage"], r["vasp_structure"],
                               r["vasp_energetics"], r["vasp_electronic"])
    print("=" * 78)
    print(f"{mat_id}")
    print(f"{ident.get('formula_pretty')}   {vs.get('spacegroup_symbol')} "
          f"(#{vs.get('spacegroup_number')}, {vs.get('crystal_system')})   "
          f"{ident.get('nsites')} sites   split={ident['split']}")
    print("=" * 78)

    print("\nSTRUCTURE           (VASP MP-GGA relaxed -- the geometry every number below refers to)")
    print(f"  a,b,c      {_fmt(vs.get('a'))}, {_fmt(vs.get('b'))}, {_fmt(vs.get('c'))} A")
    print(f"  alpha,beta,gamma  {_fmt(vs.get('alpha'),3)}, {_fmt(vs.get('beta'),3)}, "
          f"{_fmt(vs.get('gamma'),3)} deg")
    print(f"  volume     {_fmt(vs.get('volume_A3'),3)} A^3    density {_fmt(vs.get('density_g_cm3'),3)} g/cm3")
    print(f"  cif        {vs.get('cif_relaxed')}")
    if lin.get("rl_cif_as_generated"):
        print(f"  as-generated {lin['rl_cif_as_generated']}")

    print("\nENERGETICS + GAP    (MP GGA / PBE(+U), ENCUT 520 eV, VASP {})".format(
        lin.get("vasp_version") or "?"))
    print(f"  energy     {_fmt(ve.get('energy_ev'))} eV   ({_fmt(ve.get('energy_per_atom_ev'))} eV/atom)")
    print(f"  E_hull     {_fmt(ve.get('e_hull_ev_atom'))} eV/atom"
          + ("  [APPROXIMATE: W_sv POTCAR substitution]" if ve.get("e_hull_approx") else ""))
    print(f"  E_form     {_fmt(ve.get('formation_energy_ev_atom'))} eV/atom")
    print(f"  source     {ve.get('e_hull_source')}")
    print(f"  band gap   {_fmt(vel.get('bandgap_ev'))} eV   direct {_fmt(vel.get('direct_gap_ev'))} eV "
          f"({'direct' if vel.get('is_gap_direct') else 'indirect'})")

    _block(r, "vasp_dfpt", "DFPT DIELECTRIC", _show_dfpt,
           f"PBEsol, LEPSILON + IBRION=8")
    _block(r, "qe_optical", "QE OPTICAL", _show_qe,
           "PBE + SG15 ONCV, QE 7.5, epsilon.x IPA, nspin=1, no +U")
    _block(r, "slme", "SLME", _show_slme, "pymatgen SLME, 0.3 um, 300 K")
    _block(r, "hyperbolic", "HYPERBOLIC SCREEN", _show_hyp, "interband IPA")
    _block(r, "rl_surrogate", "RL LINEAGE + SURROGATE", _show_rl, "")

    f = r["flags"]
    print("\nFLAGS")
    for k in ("gap_in_window", "stable", "slme_ok", "hits_solar_target",
              "qe_vasp_gap_disagreement", "caution_nspin1_magnetic_tm",
              "caution_molecular_vdw", "qe_geometry_matches_shipped_cell"):
        if f.get(k) is not None:
            print(f"  {k:34s} {f[k]}")
    print(f"  {'completeness_class':34s} {f.get('completeness_class')}")
    if f.get("expected_nulls"):
        print(f"\n  Blocks legitimately empty for this material: {', '.join(f['expected_nulls'])}")
        print("  (each is explained in its section above -- none of them is a data gap)")
    print("=" * 78)
    return 0


def _block(r, key, title, fn, method):
    b = r[key]
    status = b.get("status")
    head = f"\n{title:<19s} ({method})" if method else f"\n{title}"
    print(head)
    if status not in ("ok", "done"):
        print(f"  [{status}] {b.get('status_reason')}")
        if status == "failed" and b.get("error"):
            err = " ".join(str(b["error"]).split())
            print(f"  error: {err[:220]}{'...' if len(err) > 220 else ''}")
        return
    fn(b, r)


def _show_dfpt(b, r):
    for name, key in (("eps_electronic", "eps_electronic_tensor"),
                      ("eps_ionic", "eps_ionic_tensor"),
                      ("eps_total", "eps_total_tensor")):
        t = b.get(key)
        if not t:
            continue
        print(f"  {name}")
        for row in t:
            print("      [" + "  ".join(f"{v:9.4f}" if v is not None else "     None"
                                        for v in row) + "]")
    print(f"  eigenvalues (electronic)  min {_fmt(b.get('eps_electronic_eig_min'))}  "
          f"max {_fmt(b.get('eps_electronic_eig_max'))}  anisotropy {_fmt(b.get('eps_anisotropy'),3)}")
    if b.get("n_imaginary_modes") is not None:
        print(f"  imaginary phonon modes    {b['n_imaginary_modes']}")
    print(f"  tensor file               {b.get('tensor_file')}")


def _show_qe(b, r):
    print(f"  indirect gap  {_fmt(b.get('indirect_gap_ev'))} eV   direct {_fmt(b.get('direct_gap_ev'))} eV")
    print(f"  eps2 peak     {_fmt(b.get('eps2_peak'),3)} at {_fmt(b.get('eps2_peak_ev'),3)} eV")
    e1 = [b.get(f"eps1_zero_freq_{a}") for a in "xyz"]
    if any(v is not None for v in e1):
        print(f"  eps1(w->0)    x {_fmt(e1[0],3)}  y {_fmt(e1[1],3)}  z {_fmt(e1[2],3)}")
    print(f"  spectrum      {b.get('spectrum_path')} ({b.get('n_energy_points')} points)")
    dgap = r["flags"].get("qe_vs_vasp_gap_delta_ev")
    if dgap is not None:
        print(f"  QE - VASP gap {dgap:+.4f} eV")


def _show_slme(b, r):
    print(f"  eta           {_fmt(b.get('eta_pct'),3)} %   at {_fmt(b.get('thickness_um'),2)} um, "
          f"{_fmt(b.get('temperature_k'),0)} K")
    print(f"  gap used      {_fmt(b.get('gap_used_ev'))} eV")
    if b.get("eta_pct_recomputed") is not None:
        print(f"  re-integrated {_fmt(b['eta_pct_recomputed'],3)} % from the shipped spectrum "
              f"(delta {_fmt(b.get('roundtrip_delta_pp'),6)} pp)")


def _show_hyp(b, r):
    print(f"  hyperbolic    {b.get('is_hyperbolic')}   score {_fmt(b.get('score'),3)}   "
          f"clean windows {b.get('n_clean_windows')}/{b.get('n_windows')}")
    if b.get("best_lo_ev") is not None:
        print(f"  best window   {_fmt(b.get('best_lo_ev'),3)}-{_fmt(b.get('best_hi_ev'),3)} eV  "
              f"Type {b.get('best_type')}  FOM {_fmt(b.get('best_fom'),2)}  ({b.get('best_band')})")
    if r["flags"].get("caution_molecular_vdw"):
        print("  CAUTION: molecular van-der-Waals solid -- a 'hit' here is an Im(eps)~0 artifact,"
              "\n           not a usable hyperbolic material")


def _show_rl(b, r):
    lin = r["lineage"]
    print(f"  run           {lin.get('rl_run_name')}")
    if lin.get("rl_checkpoint"):
        print(f"  checkpoint    {lin['rl_checkpoint']}")
    print(f"  step / index  {lin.get('rl_step')} / {lin.get('rl_eval_index')}   "
          f"target={b.get('surrogate_target')}")
    for k in ("ml_band_gap_ev", "ml_eta_slme", "ml_eps_iso", "reward_r_uni",
              "mlip_ehull_ev_atom"):
        if b.get(k) is not None:
            print(f"  {k:13s} {_fmt(b[k])}")
    print(f"  S.U.N. flags  valid={b.get('is_valid')} unique={b.get('is_unique')} "
          f"novel={b.get('is_novel')} stable={b.get('is_stable')}  "
          f"[join: {b.get('stability_join_method')}]")


def _degenerate(spec, prefix, tol=1e-6):
    """{axis: [other axes it is numerically identical to]} -- symmetry-equivalent axes."""
    import numpy as np
    cols = {a: spec[f"{prefix}_{a}"].to_numpy(dtype=float) for a in "xyz"}
    out = {}
    for a in "xyz":
        same = [b for b in "xyz" if b != a
                and np.nanmax(np.abs(cols[a] - cols[b])) <= tol]
        if same:
            out[a] = same
    return out


# ------------------------------------------------------------------------------ plot
def cmd_plot(args):
    """Three panels, one measure per axis -- never two y-scales on one frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    mat_id = _resolve(args.bundle, args.mat_id)
    r = _record(args.bundle, mat_id)
    if r is None:
        sys.exit(f"{mat_id} not found in materials.jsonl")
    if r["qe_optical"].get("status") != "done":
        sys.exit(f"{mat_id} has no optical spectrum to plot "
                 f"({r['qe_optical'].get('status')}: {r['qe_optical'].get('status_reason')})")

    spec = pd.read_csv(args.bundle / "spectra/qe_optical.csv")
    spec = spec[spec["mat_id"] == mat_id]
    if spec.empty:
        sys.exit(f"no rows for {mat_id} in spectra/qe_optical.csv")
    E = spec["energy_eV"].to_numpy()
    gap = r["qe_optical"].get("indirect_gap_ev")

    ml = None
    mlp = args.bundle / "spectra/ml_dielectric.csv"
    if mlp.exists():
        m = pd.read_csv(mlp)
        m = m[m["mat_id"] == mat_id]
        ml = m if not m.empty else None

    n = 3 if ml is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.5), constrained_layout=True)

    def dress(ax, ylab, title):
        ax.set_xlabel("Photon energy (eV)", color=INK)
        ax.set_ylabel(ylab, color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        if gap is not None and gap > 0:
            ax.axvline(gap, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)

    # In a hexagonal/tetragonal cell the in-plane axes are symmetry-degenerate, so two
    # curves land exactly on top of each other. Draw widest-first with stepped widths so
    # a hidden series still reads as a halo, and say so in the legend.
    def draw_axes(ax, prefix, sub):
        deg = _degenerate(spec, prefix)
        for a, lw in (("z", 3.0), ("y", 2.0), ("x", 1.1)):
            lab = f"$\\epsilon_{sub}^{{{a}{a}}}$"
            if deg.get(a):
                lab += r" $\equiv$ " + "".join(f"${x}{x}$" for x in deg[a])
            ax.plot(E, spec[f"eps{sub}_{a}"], color=AXIS_COLORS[a], linewidth=lw,
                    label=lab, solid_capstyle="round")
        ax.legend(loc="upper right", fontsize=8, labelcolor=INK)

    # (a) real part per crystal axis -- the sign structure is the point, so mark zero
    ax = axes[0]
    draw_axes(ax, "eps1", 1)
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    dress(ax, r"Re $\epsilon$", "(a)  Real permittivity, per crystal axis")
    if gap:
        # label it as the QE gap: the spectra are QE, while the header quotes the VASP gap
        ax.annotate(f"QE gap {gap:.2f} eV", xy=(gap, ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", fontsize=7, color=MUTED, va="bottom")

    # (b) imaginary part per crystal axis -- absorption
    ax = axes[1]
    draw_axes(ax, "eps2", 2)
    dress(ax, r"Im $\epsilon$", "(b)  Absorptive part, per crystal axis")

    # (c) DFT vs the surrogate the RL agent optimised against
    if ml is not None:
        ax = axes[2]
        ax.plot(E, spec["eps2_avg"], color=DFT_COLOR, linewidth=2, label="DFT (QE, IPA)")
        ax.plot(ml["energy_eV"], ml["ml_eps2"], color=ML_COLOR, linewidth=2,
                linestyle=(0, (5, 2)), label="TSENN surrogate")
        sc = (r["qe_optical"].get("ml_vs_dft") or {}).get("eps2_similarity")
        dress(ax, r"Im $\epsilon$ (trace/3)",
              "(c)  DFT vs surrogate" + (f"   cos-sim {sc:.3f}" if sc else ""))
        ax.legend(loc="upper right", fontsize=8, labelcolor=INK)

    eta = r["slme"].get("eta_pct")
    sub = (f"{mat_id}   {r['identity'].get('formula_pretty')}   "
           f"{r['vasp_structure'].get('spacegroup_symbol')}   "
           f"E$_g$(VASP) {_fmt(r['vasp_electronic'].get('bandgap_ev'), 2)} eV   "
           f"E$_{{hull}}$ {_fmt(r['vasp_energetics'].get('e_hull_ev_atom'), 3)} eV/atom"
           + (f"   SLME {eta:.1f} %" if eta is not None else ""))
    fig.suptitle(sub, fontsize=9, color=MUTED, y=1.04)

    out = Path(args.out) if args.out else Path(f"{r['identity']['mat_id_fs']}_spectra.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------- summary
def cmd_summary(args):
    import pandas as pd
    b = args.bundle
    df = _load(b)
    prov = json.loads((b / "provenance.json").read_text()) if (b / "provenance.json").exists() else {}

    print("=" * 78)
    print("CONSOLIDATED DFT DATASET")
    print("=" * 78)
    print(f"  built     {prov.get('built_at', '?')}")
    print(f"  materials {len(df)}")
    for k, v in (prov.get("functionals") or {}).items():
        print(f"  {k:24s} {v}")

    print("\nPer split")
    g = df.groupby("split")
    tbl = pd.DataFrame({
        "n": g.size(),
        "structure": g["cif_relaxed"].count(),
        "E_hull": g["e_hull_eV_atom"].count(),
        "band_gap": g["band_gap_eV"].count(),
        "DFPT": g["dfpt_status"].apply(lambda s: (s == "ok").sum()),
        "QE_optical": g["qe_status"].apply(lambda s: (s == "done").sum()),
        "SLME": g["slme_eta_pct"].count(),
        "as_generated": g["cif_as_generated"].count(),
    })
    tbl.loc["TOTAL"] = tbl.sum()
    print(tbl.to_string())

    print("\nYield funnel  (gap 1.0-1.8 eV, E_hull <= 0.1 eV/atom, SLME >= 25%)")
    for split in list(df["split"].unique()) + ["ALL"]:
        d = df if split == "ALL" else df[df["split"] == split]
        print(f"  {split:14s} n={len(d):4d}  stable={int((d['dft_stable'] == True).sum()):4d}"   # noqa: E712
              f"  in-gap-window={int((d['gap_in_window'] == True).sum()):4d}"                     # noqa: E712
              f"  SLME>=25%={int((d['slme_ok'] == True).sum()):4d}"                               # noqa: E712
              f"  target={int((d['hits_solar_target'] == True).sum()):4d}")                       # noqa: E712

    print("\nCompleteness class")
    for k, v in df["completeness_class"].value_counts().items():
        print(f"  {k:36s} {v:4d}")

    ep = b / "audit/expected_nulls.csv"
    if ep.exists():
        e = pd.read_csv(ep)
        bad = e[~e["match"].astype(bool)]
        print(f"\nExpected nulls: {int(e['observed_n'].sum())} across {len(e)} declared rules, "
              f"{'all reconciled' if bad.empty else f'{len(bad)} MISMATCH'}")
        for _, row in e.iterrows():
            print(f"  {row['rule_id']:32s} {row['block']:17s} {row['status']:16s} "
                  f"{row['observed_n']:4d}  {'ok' if row['match'] else 'MISMATCH'}")

    vp = b / "audit/verification.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        nfail = int((v["verdict"] == "FAIL").sum())
        print(f"\nCross-checks vs the previously published artifacts: "
              f"{len(v) - nfail}/{len(v)} PASS")
        for _, row in v[v["verdict"] == "FAIL"].iterrows():
            print(f"  FAIL {row['check']}: {row['note']}")

    ex = b / "audit/excluded.csv"
    if ex.exists() and ex.stat().st_size:
        print(f"\nExcluded: {len(pd.read_csv(ex))} materials whose DFT never got past the "
              f"first relaxation (see audit/excluded.csv)")

    cautions = [("QE metallic but VASP gapped", "qe_vasp_gap_disagreement"),
                ("magnetic TM (QE nspin=1, no +U)", "caution_nspin1_magnetic_TM"),
                ("molecular vdW solid", "caution_molecular_vdW"),
                ("approximate E_hull (W_sv)", "e_hull_approx")]
    print("\nCautions")
    for label, col in cautions:
        if col in df:
            print(f"  {label:38s} {int((df[col] == True).sum()):4d}")           # noqa: E712
    print("=" * 78)
    return 0


# ------------------------------------------------------------------------- structure
def cmd_structure(args):
    mat_id = _resolve(args.bundle, args.mat_id)
    r = _record(args.bundle, mat_id)
    if r is None:
        sys.exit(f"{mat_id} not found in materials.jsonl")
    rel = (r["lineage"].get("rl_cif_as_generated") if args.as_generated
           else r["vasp_structure"].get("cif_relaxed"))
    if not rel:
        sys.exit(f"{mat_id} has no {'as-generated' if args.as_generated else 'relaxed'} structure")
    text = (args.bundle / rel).read_text()
    if args.fmt == "cif":
        print(text)
    else:
        import warnings
        warnings.simplefilter("ignore")
        from pymatgen.core import Structure
        print(Structure.from_str(text, fmt="cif").to(fmt=args.fmt))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="browse the review table")
    p.add_argument("--split", choices=["slme", "dielectric", "failed_demo", "legacy_pilot"])
    p.add_argument("--stable", action="store_true", help="E_hull <= 0.1 eV/atom")
    p.add_argument("--gapped", action="store_true", help="VASP band gap > 0.01 eV")
    p.add_argument("--dfpt", action="store_true", help="has a DFPT dielectric tensor")
    p.add_argument("--sort", default=None)
    p.add_argument("--ascending", action="store_true")
    p.add_argument("--limit", type=int, default=60)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="everything known about one material")
    p.add_argument("mat_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("plot", help="optical spectra + DFT-vs-surrogate overlay")
    p.add_argument("mat_id")
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("summary", help="counts, yield funnel, audit + verification status")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("structure", help="dump a geometry")
    p.add_argument("mat_id")
    p.add_argument("--fmt", default="cif", choices=["cif", "poscar", "json"])
    p.add_argument("--as-generated", action="store_true")
    p.set_defaults(func=cmd_structure)

    args = ap.parse_args(argv)
    if not (args.bundle / "materials.csv").exists():
        sys.exit(f"no bundle at {args.bundle}\nBuild it first: "
                 f"uv run python scripts/build_dft_bundle.py")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
