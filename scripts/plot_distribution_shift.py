#!/usr/bin/env python3
"""Story A: prior -> finetuned distribution shift (surrogate-over-estimation-robust).

Overlays the eta (SLME efficiency) distribution of an EARLY/near-prior sample and a
STEERED (best-reward) sample, both scored by the SAME surrogate, and annotates mean eta
and the eta>0.25 / eta>0.30 hit rates. Because both sides use the same (biased) ruler,
the *relative* rightward shift is robust to the surrogate's absolute over-estimation.

Example:
  python scripts/plot_distribution_shift.py \
      --early   exp_res/<..._loop0009_sample_relaxed_...> \
      --steered exp_res/<..._best_reward_sample_relaxed_...> \
      --output  exp_res/<..._best_reward_sample_relaxed_...>/deliverables/distribution_shift.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# Group publication style (repo root). Apply if present so the saved figures
# match the rest of the paper (fonts, tick direction, 600-dpi vector export).
_STYLE = Path(__file__).resolve().parent.parent / "publication.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))

ETA_MID, ETA_HI = 0.25, 0.30
GOOD_BG_MIN, GOOD_BG_MAX = 1.0, 1.5
ETA_PLOT_MAX = 0.36
TARGET_ETA_MIN = ETA_MID  # lower eta bound of the 2D target region (--target-eta-min)
SQ_EXT_PATH = (
    Path(__file__).resolve().parents[1]
    / "rewards/calculators/tsenn_slme/data/SMARTS_295_Linux/Examples/"
    / "Example 6-USSA_084/smarts295.ext.txt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--early", type=Path, required=True, help="early/near-prior run dir")
    p.add_argument("--steered", type=Path, required=True, help="steered (best) run dir")
    p.add_argument("--prior", type=Path, default=None, help="optional true-prior run dir")
    p.add_argument("--early-steps", type=str, default=None, help="restrict early to step range 'lo:hi' (e.g. 0:10)")
    p.add_argument("--steered-steps", type=str, default=None, help="restrict steered to step range 'lo:hi' (e.g. 86:96)")
    p.add_argument("--prior-steps", type=str, default=None, help="restrict prior to step range 'lo:hi'")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--scatter-output",
        type=Path,
        default=None,
        help="band_gap-vs-eta density scatter (default: <output>_bg_eta.png)",
    )
    p.add_argument(
        "--kde",
        action="store_true",
        help="plot smooth KDE densities instead of histograms for the 1D eta shift",
    )
    p.add_argument(
        "--kde-bw",
        type=float,
        default=None,
        help="KDE bandwidth (gaussian_kde bw_method); default uses Scott's rule",
    )
    p.add_argument(
        "--target-eta-min",
        type=float,
        default=ETA_MID,
        help="lower eta bound of the 2D target region (default 0.25; try 0.30)",
    )
    return p.parse_args()


def detect_eta_dir(run_dir: Path) -> Path:
    rewards = run_dir / "rewards"
    for name in ("tsenn_slme_optimate_eta", "tsenn_slme_eta"):
        if (rewards / name).is_dir():
            return rewards / name
    cands = [d for d in rewards.iterdir() if d.is_dir() and d.name.endswith("_eta")]
    if not cands:
        raise FileNotFoundError(f"No *_eta reward dir under {rewards}")
    return cands[0]


def parse_range(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    lo, hi = spec.split(":")
    return int(lo), int(hi)


def step_of(path: Path) -> int:
    return int(path.stem.split("_")[1])


def in_range(path: Path, rng: tuple[int, int] | None) -> bool:
    if rng is None:
        return True
    s = step_of(path)
    return rng[0] <= s < rng[1]


def load_eta(run_dir: Path, steps: tuple[int, int] | None = None) -> np.ndarray:
    eta_dir = detect_eta_dir(run_dir.resolve())
    vals: list[float] = []
    for f in sorted(eta_dir.glob("step_*.txt")):
        if not in_range(f, steps):
            continue
        for line in f.read_text().splitlines():
            try:
                vals.append(float(line.strip()))
            except ValueError:
                pass
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def load_bg_eta(run_dir: Path, steps: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample (band_gap, eta) paired row-wise per step."""
    run_dir = run_dir.resolve()
    eta_dir = detect_eta_dir(run_dir)
    bg_dir = run_dir / "rewards" / "bandgap"
    bgs: list[float] = []
    etas: list[float] = []
    for ef in sorted(eta_dir.glob("step_*.txt")):
        if not in_range(ef, steps):
            continue
        bf = bg_dir / ef.name
        if not bf.exists():
            continue
        e = [float(x) if x.strip() else float("nan") for x in ef.read_text().splitlines() if x.strip()]
        b = [float(x) if x.strip() else float("nan") for x in bf.read_text().splitlines() if x.strip()]
        n = min(len(e), len(b))
        bgs.extend(b[:n])
        etas.extend(e[:n])
    bg = np.asarray(bgs, dtype=float)
    eta = np.asarray(etas, dtype=float)
    m = np.isfinite(bg) & np.isfinite(eta)
    return bg[m], eta[m]


def _read_smarts_ext(path: Path, irradiance_col: str):
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        i_w = header.index("Wvlgth")
        i_i = header.index(irradiance_col)
        wl_nm, i_lambda = [], []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            wl_nm.append(float(parts[i_w]))
            i_lambda.append(float(parts[i_i]))

    hc_ev_nm = 1239.8419843320026
    q = 1.602176634e-19
    wl_nm_arr = np.asarray(wl_nm, dtype=np.float64)
    i_lambda_arr = np.asarray(i_lambda, dtype=np.float64)
    e_ev = hc_ev_nm / wl_nm_arr
    dlam_de_abs = hc_ev_nm / (e_ev * e_ev)
    i_e = i_lambda_arr * dlam_de_abs
    phi_e = i_e / (e_ev * q)
    order = np.argsort(e_ev)
    return e_ev[order], i_e[order], phi_e[order]


def _cumtrapz_from_right(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    areas = 0.5 * (y[:-1] + y[1:]) * (x[1:] - x[:-1])
    out = np.zeros_like(x, dtype=np.float64)
    out[:-1] = np.cumsum(areas[::-1])[::-1]
    return out


def _phi_bb_eV(omega_eV: np.ndarray, temperature_k: float) -> np.ndarray:
    q = 1.602176634e-19
    h = 6.62607015e-34
    c = 299792458.0
    k_b = 1.380649e-23
    e_j = omega_eV * q
    x = np.clip(e_j / (k_b * temperature_k), 1e-12, 700.0)
    denom = np.expm1(x)
    phi_per_j = (2.0 * math.pi) / (h**3 * c**2) * (e_j * e_j) / denom
    return phi_per_j * q


def _sq_eta_curve(
    omega_eV: np.ndarray,
    i_e: np.ndarray,
    phi_sun_eV: np.ndarray,
    eg_grid: np.ndarray,
    temperature_k: float = 300.0,
) -> np.ndarray:
    pin = float(np.trapz(i_e, omega_eV))
    if not (math.isfinite(pin) and pin > 0.0):
        return np.full_like(eg_grid, np.nan)
    phi_bb = _phi_bb_eV(omega_eV, temperature_k)
    sun_cum = _cumtrapz_from_right(omega_eV, phi_sun_eV)
    bb_cum = _cumtrapz_from_right(omega_eV, phi_bb)
    q = 1.602176634e-19
    k_b = 1.380649e-23
    vt = (k_b * temperature_k) / q
    jsc = q * np.interp(eg_grid, omega_eV, sun_cum, left=np.nan, right=0.0)
    j0 = q * np.interp(eg_grid, omega_eV, bb_cum, left=np.nan, right=0.0)
    eta = np.full_like(eg_grid, np.nan)
    for i, (jsc_i, j0_i) in enumerate(zip(jsc, j0)):
        if not (math.isfinite(float(jsc_i)) and math.isfinite(float(j0_i))):
            continue
        if jsc_i <= 0.0 or j0_i <= 0.0:
            continue
        voc = vt * math.log(float(jsc_i / j0_i) + 1.0)
        if not (math.isfinite(voc) and voc > 0.0):
            continue
        voltage = np.linspace(0.0, voc, 256)
        current = jsc_i - j0_i * (np.exp(voltage / vt) - 1.0)
        pmax = float(np.max(voltage * current))
        if math.isfinite(pmax) and pmax > 0.0:
            eta[i] = pmax / pin
    return eta


def _sq_curve(eg_grid: np.ndarray):
    """SQ-limit eta over a band-gap grid, or None if the spectrum file is absent."""
    if not SQ_EXT_PATH.exists():
        return None
    omega, i_e, phi_sun = _read_smarts_ext(SQ_EXT_PATH, "Global_tilted_irradiance")
    return _sq_eta_curve(omega, i_e, phi_sun, eg_grid)


def _plot_sq_limit(ax, eg_max: float = 3.0) -> None:
    eg_grid = np.linspace(0.5, eg_max, 400)
    eta_sq = _sq_curve(eg_grid)
    if eta_sq is None:
        return
    ax.plot(eg_grid, eta_sq, color="black", ls="--", lw=1.2, alpha=0.5, zorder=1.5)


def _annotate_good_region(ax, label_loc: str = "peak", fontsize: float = 8.5,
                          label_dx: float = 0.0) -> None:
    """Shade the achievable target region: eta > TARGET_ETA_MIN AND below the SQ
    limit. The band-gap extent is set by where the SQ curve crosses the threshold
    (a dome under SQ), NOT a fixed band-gap window.

    label_loc: "peak" (centered at the SQ peak, default) or "left" (at the dome's
    left crossing, to clear an upper-right legend)."""
    grid = np.linspace(0.05, 3.0, 800)
    sq = _sq_curve(grid)
    if sq is None:
        return
    top = np.minimum(sq, ETA_PLOT_MAX)
    mask = top > TARGET_ETA_MIN
    ax.fill_between(grid, TARGET_ETA_MIN, top, where=mask,
                    facecolor="#fb9a99", edgecolor="none", alpha=0.15, zorder=0.5)
    if mask.any():
        if label_loc == "left":
            cx, ha = grid[mask][0], "left"
        else:
            cx, ha = grid[mask][int(np.argmax(top[mask]))], "center"  # SQ peak
        ax.text(cx + label_dx, ETA_PLOT_MAX - 0.006, "Target region",
                ha=ha, va="top", fontsize=fontsize, color="#b30000", zorder=4)


def plot_bg_eta_density(series, output: Path) -> None:
    """band_gap-vs-eta scatter per series, shared axes AND shared density color scale."""
    from scipy.stats import gaussian_kde

    # Pass 1: compute KDE density per series so panels share one color normalization.
    computed = []
    zmax = 0.0
    for name, bg, eta, _color in series:
        if bg.size >= 5:
            xy = np.vstack([bg, eta])
            try:
                z = gaussian_kde(xy)(xy)
            except Exception:  # noqa: BLE001
                z = np.ones(bg.size)
        else:
            z = np.ones(bg.size)
        computed.append((name, bg, eta, z))
        if z.size:
            zmax = max(zmax, float(z.max()))

    # SQ curve on a fine grid to test membership in the (capped) target region
    sq_grid_x = np.linspace(0.0, 3.0, 600)
    sq_grid_y = _sq_curve(sq_grid_x)

    def _frac_in_target(bg, eta):
        if bg.size == 0:
            return 0.0
        cap = (np.interp(bg, sq_grid_x, sq_grid_y) if sq_grid_y is not None
               else np.full_like(bg, ETA_PLOT_MAX))
        # in region = above the eta threshold AND below the SQ limit
        # (band-gap extent emerges from where SQ exceeds the threshold)
        inside = (eta > TARGET_ETA_MIN) & (eta <= cap)
        return 100.0 * inside.mean()

    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n + 0.8, 4.8), sharex=True, sharey=True, squeeze=False)
    sc = None
    for ax, (name, bg, eta, z) in zip(axes[0], computed):
        _annotate_good_region(ax)
        order = z.argsort()
        sc = ax.scatter(
            bg[order], eta[order], c=z[order], s=14, cmap="viridis",
            vmin=0.0, vmax=zmax, edgecolors="none", zorder=2,
        )
        _plot_sq_limit(ax)
        ax.set_xlabel("Band gap (eV)")
        ax.set_xlim(0, 3.0)
        ax.set_ylim(0, ETA_PLOT_MAX)
        ax.set_title(name, fontsize=10)
        # Per-panel legend: sample count + target-region stat + SQ-limit key, inside each box.
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#4c4c4c",
                       markeredgecolor="none", markersize=5, label=f"N = {bg.size}"),
                Patch(facecolor="#fb9a99", edgecolor="#d7301f", alpha=0.15,
                      label=f"In target region: {_frac_in_target(bg, eta):.0f}%"),
                Line2D([0], [0], color="black", ls="--", lw=1.25, label="SQ limit"),
            ],
            loc="upper right", frameon=False, fontsize=9, handlelength=1.2,
            handletextpad=0.5, borderaxespad=0.4,
        )
    axes[0][0].set_ylabel("SLME $\\eta$ (%)")
    if sc is not None:
        fig.colorbar(sc, ax=axes[0].tolist(), label="KDE density", fraction=0.046, pad=0.02)
    # fig.suptitle("Prior $\\to$ fine-tuned shift in (band gap, $\\eta$) space", y=1.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)  # dpi/bbox come from publication.mplstyle (600 dpi, tight)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {output}")


def stat_label(name: str, eta: np.ndarray) -> str:
    return (
        f"{name} (N={eta.size}): "
        f"mean {eta.mean():.3f}, "
        f"$\\eta{{>}}{ETA_MID:g}$ {100*(eta>ETA_MID).mean():.0f}%, "
        f"$\\eta{{>}}{ETA_HI:g}$ {100*(eta>ETA_HI).mean():.0f}%"
    )


def main() -> None:
    args = parse_args()
    global TARGET_ETA_MIN
    TARGET_ETA_MIN = float(args.target_eta_min)
    er = parse_range(args.early_steps)
    sr = parse_range(args.steered_steps)
    pr = parse_range(args.prior_steps)
    series = []
    if args.prior is not None:
        series.append(("Prior", load_eta(args.prior, pr), "#9CA3AF"))
    series.append(("Early", load_eta(args.early, er), "#2563EB"))
    series.append(("Steered (best)", load_eta(args.steered, sr), "#DC2626"))

    bins = np.linspace(0.0, 0.36, 37)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if args.kde:
        from scipy.stats import gaussian_kde
        grid = np.linspace(0.0, 0.36, 400)
        for name, eta, color in series:
            if eta.size < 2 or np.allclose(eta, eta[0]):
                continue
            kde = gaussian_kde(eta, bw_method=args.kde_bw)
            dens = kde(grid)
            ax.fill_between(grid, dens, color=color, alpha=0.30,
                            label=stat_label(name, eta))
            ax.plot(grid, dens, color=color, lw=2.2)
            ax.axvline(eta.mean(), color=color, ls=":", lw=1.5)
    else:
        for name, eta, color in series:
            ax.hist(eta, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                    color=color, label=stat_label(name, eta))
            ax.hist(eta, bins=bins, density=True, histtype="step", color=color, lw=2)
            ax.axvline(eta.mean(), color=color, ls=":", lw=1.5)
    for thr in (ETA_MID, ETA_HI):
        ax.axvline(thr, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("TSENN/OptiMate SLME $\\eta$")
    ax.set_ylabel("density")
    ax.set_title("Prior $\\to$ fine-tuned distribution shift (same surrogate ruler)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)  # dpi/bbox from publication.mplstyle
    plt.close(fig)
    print(f"Wrote {args.output}")
    for name, eta, _ in series:
        print(f"  {stat_label(name, eta)}")

    # 2D band_gap-vs-eta density scatter
    series_2d = []
    if args.prior is not None:
        bg, eta = load_bg_eta(args.prior, pr)
        series_2d.append(("Prior", bg, eta, "#9CA3AF"))
    bg, eta = load_bg_eta(args.early, er)
    series_2d.append(("Early", bg, eta, "#2563EB"))
    bg, eta = load_bg_eta(args.steered, sr)
    series_2d.append(("Steered (best)", bg, eta, "#DC2626"))
    scatter_out = args.scatter_output or args.output.with_name(
        args.output.stem + "_bg_eta" + args.output.suffix
    )
    plot_bg_eta_density(series_2d, scatter_out)


if __name__ == "__main__":
    main()
