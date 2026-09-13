#!/usr/bin/env python3
"""Reward-surface figure for the two paper runs (two panels).

Port of `sparc_v0/scripts/plot_reward_shaping.py`, with one substantive correction and one
structural change.

**Correction.** The sparc_v0 version's panel (a) documented
`fom_layered_uniaxial_gapgate_min` (run 933): quality ramp 0->0.20, gap gate **0.5->1.2 eV**.
That reward was superseded by the in-plane-isotropy FoM and its config no longer exists in
this repo. The current dielectric runs -- including the strict-S.U.N. rerun
`dielectric_inplane_isotropy_gapgate_mprime_newbg_b96_sun` -- use
`fom_inplane_isotropy_gapgate`: isotropy ramp **0.9->1.0**, gap gate **0.3->0.8 eV**.

**Structural change.** Every threshold is now read LIVE from `configs/reward/*.yaml` instead
of being hardcoded here and restated in a docstring. That is what let the figure go stale in
the first place; now editing a reward config moves the figure with it.

  (a) Dielectric  `fom_inplane_isotropy_gapgate`  -- reduce=min, a GATE:
        r = min( ramp(iso), ramp(E_g) ).  Above the gap ceiling the reward stops depending
        on E_g at all, so the contours turn horizontal -- that is the gate signature.
  (b) SLME        `tsenn_slme_optimate_bgcenter13_eta08_e3nngap` -- reduce=weight, a TARGET:
        r = w_g * tent(E_g; 1.3) + w_eta * ramp(eta).  The tent peaks at the Shockley-Queisser
        optimum and hard-zeros outside [0.3, 2.3] eV.

Panel (b) overlays the true Shockley-Queisser ceiling eta_SQ(E_g) from detailed balance on the
same ASTM G173 AM1.5G spectrum + 300 K blackbody the reward itself uses (peak 33.7% @ 1.34 eV).

Usage:
  uv run python scripts/plot_reward_shaping.py
  uv run python scripts/plot_reward_shaping.py -o exp_res/_abrepr/reward_shaping.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _register_liberation_sans() -> None:
    """Make `publication.mplstyle`'s Liberation Sans actually resolve.

    The fonts are installed (`/usr/share/fonts/liberation-sans/`) but matplotlib's font
    manager does not scan that directory here, so it silently falls back to DejaVu Sans --
    which would give this figure a different typeface from the published
    `sparc_v0/exp_res/_candidate_plots/reward_shaping.pdf` (that one embeds LiberationSans).
    Registering the TTFs explicitly keeps the typeface consistent.
    """
    import glob
    import matplotlib.font_manager as fm

    if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist):
        return
    for path in sorted(glob.glob("/usr/share/fonts/liberation-sans/LiberationSans-*.ttf")):
        try:
            fm.fontManager.addfont(path)
        except Exception:  # noqa: BLE001 -- fall back to DejaVu rather than fail the figure
            pass


_register_liberation_sans()
_style = ROOT / "publication.mplstyle"
if _style.exists():
    plt.style.use(str(_style))

Q, KB, H, C = 1.602176634e-19, 1.380649e-23, 6.62607015e-34, 2.99792458e8
HC_EV_NM = 1239.8419843320026
AM15 = ROOT / "rewards/calculators/tsenn_slme/data/ASTMG173.csv"


def _load_am15(path):
    arr = np.genfromtxt(path, delimiter=",", skip_header=2)
    wl, gt = arr[:, 0], arr[:, 2]
    E = HC_EV_NM / wl
    irr_ev = gt * (HC_EV_NM / E**2)
    phi = irr_ev / (E * Q)
    o = np.argsort(E)
    return E[o], phi[o], irr_ev[o]


def _phi_bb(E, T=300.0):
    x = np.clip(E * Q / (KB * T), 1e-12, 700.0)
    return (2.0 * np.pi) / (H**3 * C**2) * (E * Q) ** 2 / np.expm1(x) * Q


def sq_eta(Egs):
    E, phi, irr = _load_am15(AM15)
    Pin = np.trapz(irr, E)
    pb = _phi_bb(E, 300.0)
    out = []
    for Eg in Egs:
        m = E >= Eg
        if m.sum() < 2:
            out.append(0.0)
            continue
        Jsc = Q * np.trapz(phi[m], E[m])
        J0 = Q * np.trapz(pb[m], E[m])
        V = np.linspace(0.0, float(Eg), 500)
        P = V * (Jsc - J0 * np.expm1(Q * V / (KB * 300.0)))
        out.append(max(0.0, float(P.max())) / Pin)
    return np.array(out)


def ramp(x, minv, maxv):
    return np.clip((x - minv) / (maxv - minv), 0.0, 1.0)


def read_reward(name: str) -> dict:
    c = yaml.safe_load(open(ROOT / "configs" / "reward" / f"{name}.yaml"))
    return {"reduce": c.get("reduce"),
            "props": {p["name"]: p for p in c.get("prop_cfg", [])}}


def build_runs() -> list[dict]:
    """Assemble the two panel specs from the live reward configs."""
    d = read_reward("fom_inplane_isotropy_gapgate")
    assert d["reduce"] == "min", f"panel (a) expects reduce=min, got {d['reduce']}"
    dg = d["props"]["band_gap"]
    di = d["props"]["tsenn_static_dielectric_inplane_isotropy"]

    s = read_reward("tsenn_slme_optimate_bgcenter13_eta08_e3nngap")
    assert s["reduce"] == "weight", f"panel (b) expects reduce=weight, got {s['reduce']}"
    sg = s["props"]["band_gap"]
    se = s["props"]["tsenn_slme_optimate_eta"]
    assert not isinstance(sg["target"], str), "panel (b) gap term must be a float target (tent)"

    return [
        dict(letter="(a)", kind="min",
             # `r_q` is the paper's generic symbol for the quality term -- here it is the
             # in-plane-isotropy reward. Keep the published notation: do NOT rename it to
             # r_iso just because the underlying term changed.
             formula=r"$r_{\mathrm{uni}}=\min\,(r_{q},\ r_{g,\mathrm{uni}})$",
             plabel=r"$r_{q}$", pscale=1.0, pmin=0.0, pmax=1.0, pxmax=1.0,
             gmin=float(dg["minv"]), gmax=float(dg["maxv"]), gxmax=4.0,
             # both ends of the gate: onset (reward 0 below) and ceiling (saturation)
             guides=[(float(dg["minv"]), 0.7), (float(dg["maxv"]), 0.7)], sq=False,
             formula_pos=(0.97, 0.06, "right", "bottom"),
             iso_minv=float(di["minv"]), iso_maxv=float(di["maxv"])),
        dict(letter="(b)", kind="weighted_tent",
             formula=r"$r_{\mathrm{SLME}}=w_{\eta}\,r_{\eta}+w_{g}\,r_{g,\mathrm{SLME}}$",
             plabel=r"SLME  $\eta$ (%)", pscale=100.0,
             pmin=float(se["minv"]), pmax=float(se["maxv"]), pxmax=0.40,
             tent_target=float(sg["target"]), tent_half=float(sg["maxv"]),
             w_gap=float(sg["weight"]), w_eta=float(se["weight"]),
             gxmax=4.0, guides=[], sq=True,
             formula_pos=(0.97, 0.95, "right", "top")),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path,
                    default=ROOT / "exp_res" / "_abrepr" / "reward_shaping.pdf")
    args = ap.parse_args(argv)

    runs = build_runs()
    fig = plt.figure(figsize=(9.6, 4.2))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.28,
                  left=0.07, right=0.9, top=0.93, bottom=0.15, figure=fig)
    im = None
    for i, run in enumerate(runs):
        ax = fig.add_subplot(gs[0, i])
        s = run["pscale"]
        gx = np.linspace(0, run["gxmax"], 320)
        py = np.linspace(0, run["pxmax"] * s, 320)
        GX, PY = np.meshgrid(gx, py)
        p_ramp = ramp(PY / s, run["pmin"], run["pmax"])
        if run["kind"] == "min":
            Z = np.minimum(p_ramp, ramp(GX, run["gmin"], run["gmax"]))
        else:
            tent = np.clip(1.0 - np.abs(GX - run["tent_target"]) / run["tent_half"], 0.0, 1.0)
            Z = run["w_gap"] * tent + run["w_eta"] * p_ramp
        im = ax.imshow(Z, origin="lower", extent=[0, run["gxmax"], 0, run["pxmax"] * s],
                       aspect="auto", cmap="viridis", vmin=0, vmax=1, zorder=0)
        for gx_line, a in run["guides"]:
            ax.axvline(gx_line, color="w", ls=":", lw=1.0, alpha=a)
        ax.set_xlabel(r"Band gap (eV)")
        ax.set_ylabel(run["plabel"])
        ax.set_xticks(np.linspace(0, run["gxmax"], 5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.annotate(run["letter"], xy=(0, 1), xycoords="axes fraction", xytext=(-38, 6),
                    textcoords="offset points", ha="left", va="bottom", fontsize=13,
                    fontweight="bold", annotation_clip=False)
        fx, fy, fha, fva = run["formula_pos"]
        ax.text(fx, fy, run["formula"], transform=ax.transAxes, ha=fha, va=fva,
                fontsize=10, color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.35))
        ax.margins(0)
        if run["sq"]:
            egs = np.linspace(0.25, run["gxmax"], 140)
            eta = sq_eta(egs) * s
            ax.plot(egs, eta, color="white", ls="--", lw=1.8, zorder=4)
            ipk = int(np.argmax(eta))
            ax.annotate("SQ limit", (egs[ipk], eta[ipk]), xytext=(6, 8),
                        textcoords="offset points", fontsize=8.5, color="white")

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("reward")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    a, b = runs
    print(f"(a) dielectric  reduce=min     gap gate {a['gmin']}->{a['gmax']} eV, "
          f"isotropy ramp {a['iso_minv']}->{a['iso_maxv']}")
    print(f"(b) SLME        reduce=weight  tent@{b['tent_target']} eV (half {b['tent_half']}, "
          f"zero outside {b['tent_target']-b['tent_half']}-{b['tent_target']+b['tent_half']}), "
          f"w_g={b['w_gap']}  eta ramp {b['pmin']}->{b['pmax']} w={b['w_eta']}")
    print(f"SQ peak: {sq_eta(np.array([1.34]))[0]*100:.1f}% @ 1.34 eV")
    print(f"wrote {args.output.with_suffix('.pdf')} / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
