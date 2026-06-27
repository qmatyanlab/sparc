#!/usr/bin/env python3

from __future__ import annotations

import math
import re
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase.io import read as ase_read
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from omegaconf import OmegaConf

from rewards.calculators.tsenn_slme.calc import (
    TSENNSLME,
    _absorption_coef_um_inv,
    _kk_eps1_from_eps2,
)


SQ_EXT_PATH = Path(
    "rewards/calculators/tsenn_slme/data/SMARTS_295_Linux/Examples/Example 6-USSA_084/smarts295.ext.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_run_config(exp_dir: Path):
    config_path = exp_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    return OmegaConf.load(require_path(exp_dir / "hparams.yaml", "run config"))


def _reward_subdir_name(prop_cfg: Any) -> str:
    calculator = getattr(prop_cfg, "calculator", None)
    root_dir = getattr(calculator, "root_dir", None)
    if root_dir is not None:
        return Path(str(root_dir)).name
    return str(prop_cfg.name)


def _resolve_reward_dirs(exp_dir: Path, cfg: Any) -> tuple[Path, Path]:
    rewards_dir = require_path(exp_dir / "rewards", "rewards directory")

    band_gap_cfg = None
    eta_cfg = None
    for prop_cfg in cfg.reward.prop_cfg:
        name = str(prop_cfg.name)
        if name == "band_gap":
            band_gap_cfg = prop_cfg
        if name.endswith("_eta"):
            eta_cfg = prop_cfg

    if band_gap_cfg is None:
        raise RuntimeError("Run config does not define a band_gap reward")
    if eta_cfg is None:
        raise RuntimeError("Run config does not define a TSENN eta reward")

    bg_dir = require_path(
        rewards_dir / _reward_subdir_name(band_gap_cfg), "band gap reward directory"
    )
    eta_dir = require_path(
        rewards_dir / _reward_subdir_name(eta_cfg), "TSENN eta reward directory"
    )
    return bg_dir, eta_dir


def _eta_percent_label() -> str:
    return r"Predicted $\eta$ (\%)"


def _best_step_from_metadata(exp_dir: Path) -> int | None:
    candidates = [
        exp_dir / "models" / "best_region.json",
        exp_dir / "models" / "best_reward.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        step = data.get("step")
        if isinstance(step, int):
            return step
    return None


def _configure_latex_style() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": shutil.which("latex") is not None,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 220,
        }
    )


def _disable_grid(ax: Axes) -> None:
    ax.grid(False)


def _faint_ygrid(ax: Axes) -> None:
    ax.grid(False)
    ax.grid(axis="y", alpha=0.12, linewidth=0.6)


def _read_step(path: Path):
    vals: list[float] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                vals.append(float("nan"))
    return np.array(vals, dtype=np.float64)


def _discover_steps(bg_dir: Path, eta_dir: Path) -> list[int]:
    bg_steps = {int(path.stem.split("_")[-1]) for path in bg_dir.glob("step_*.txt")}
    eta_steps = {int(path.stem.split("_")[-1]) for path in eta_dir.glob("step_*.txt")}
    steps = sorted(bg_steps & eta_steps)
    if not steps:
        raise RuntimeError(
            f"No overlapping TSENN/band-gap reward steps found under {bg_dir.parent}"
        )
    return steps


def _build_step_df(bg_dir: Path, eta_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for step in _discover_steps(bg_dir, eta_dir):
        tag = f"step_{step:04d}.txt"
        bg_path = bg_dir / tag
        eta_path = eta_dir / tag
        if not bg_path.exists() or not eta_path.exists():
            continue
        bg = _read_step(bg_path)
        eta = _read_step(eta_path)
        n = min(len(bg), len(eta))
        for i in range(n):
            rows.append(
                {
                    "rl_step": step,
                    "band_gap_ev": bg[i],
                    "eta_frac": eta[i],
                    "eta_percent": eta[i] * 100.0,
                }
            )
    return pd.DataFrame(rows)


def _load_step_records(exp_dir: Path, cfg: Any) -> pd.DataFrame:
    samples_dir = require_path(exp_dir / "samples", "samples directory")
    bg_dir, eta_dir = _resolve_reward_dirs(exp_dir, cfg)
    rows: list[dict[str, object]] = []
    eval_pt_paths = sorted(samples_dir.glob("step_*_eval.pt"))
    if eval_pt_paths:
        step_sources = [(int(path.stem.split("_")[1]), True) for path in eval_pt_paths]
    else:
        step_sources = [
            (int(path.stem.split("_")[1]), False)
            for path in sorted(samples_dir.glob("step_*.extxyz"))
            if "_eval" not in path.stem and "_valid" not in path.stem
        ]
    for step, has_eval_payload in step_sources:
        extxyz_path = (
            samples_dir / f"step_{step:04d}_eval.extxyz"
            if has_eval_payload
            else samples_dir / f"step_{step:04d}.extxyz"
        )
        bg_path = bg_dir / f"step_{step:04d}.txt"
        eta_path = eta_dir / f"step_{step:04d}.txt"
        if not extxyz_path.exists() or not bg_path.exists() or not eta_path.exists():
            continue
        atoms_list = ase_read(extxyz_path, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        structures = [
            AseAtomsAdaptor.get_structure(cast(Any, atom)) for atom in atoms_list
        ]
        bg = _read_step(bg_path)
        eta = _read_step(eta_path)
        if has_eval_payload:
            payload = torch.load(
                samples_dir / f"step_{step:04d}_eval.pt", map_location="cpu"
            )
            if not isinstance(payload, list):
                continue
            n = min(len(payload), len(structures), len(bg), len(eta))
        else:
            n = min(len(structures), len(bg), len(eta))
        for i in range(n):
            rows.append(
                {
                    "step": step,
                    "index": i,
                    "band_gap": float(bg[i]),
                    "tsenn_slme_eta": float(eta[i]),
                    "formula": structures[i].composition.reduced_formula,
                    "structure": structures[i],
                    "cif": structures[i].to(fmt="cif"),
                }
            )
    if not rows:
        raise RuntimeError(f"No TSENN eval records found under {exp_dir}")
    return pd.DataFrame(rows)


def _ascending_reward(values, minv: float, maxv: float):
    return np.clip((values - minv) / (maxv - minv), 0.0, 1.0)


def _read_smarts_ext(path: str, irradiance_col: str):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        i_w = header.index("Wvlgth")
        i_I = header.index(irradiance_col)
        wl_nm, i_lambda = [], []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            wl_nm.append(float(parts[i_w]))
            i_lambda.append(float(parts[i_I]))

    hc_eV_nm = 1239.8419843320026
    q = 1.602176634e-19
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    i_lambda = np.asarray(i_lambda, dtype=np.float64)
    E_eV = hc_eV_nm / wl_nm
    dlam_dE_abs = hc_eV_nm / (E_eV * E_eV)
    i_e = i_lambda * dlam_dE_abs
    phi_E = i_e / (E_eV * q)
    order = np.argsort(E_eV)
    return E_eV[order], i_e[order], phi_E[order]


def _cumtrapz_from_right(x, y):
    areas = 0.5 * (y[:-1] + y[1:]) * (x[1:] - x[:-1])
    out = np.zeros_like(x, dtype=np.float64)
    out[:-1] = np.cumsum(areas[::-1])[::-1]
    return out


def _phi_bb_eV(omega_eV, T: float):
    q = 1.602176634e-19
    h = 6.62607015e-34
    c = 299792458.0
    kB = 1.380649e-23
    E_J = omega_eV * q
    x = np.clip(E_J / (kB * T), 1e-12, 700.0)
    denom = np.expm1(x)
    phi_per_J = (2.0 * math.pi) / (h**3 * c**2) * (E_J * E_J) / denom
    return phi_per_J * q


def _sq_eta_curve(omega_eV, i_e, phi_sun_eV, Eg_grid, T: float):
    Pin = float(np.trapz(i_e, omega_eV))
    if not (math.isfinite(Pin) and Pin > 0.0):
        return np.full_like(Eg_grid, np.nan)
    phi_bb = _phi_bb_eV(omega_eV, T=T)
    sun_cum = _cumtrapz_from_right(omega_eV, phi_sun_eV)
    bb_cum = _cumtrapz_from_right(omega_eV, phi_bb)
    q = 1.602176634e-19
    kB = 1.380649e-23
    Vt = (kB * T) / q
    Jsc = q * np.interp(Eg_grid, omega_eV, sun_cum, left=np.nan, right=0.0)
    J0 = q * np.interp(Eg_grid, omega_eV, bb_cum, left=np.nan, right=0.0)
    eta = np.full_like(Eg_grid, np.nan)
    for i in range(len(Eg_grid)):
        jsc, j0 = float(Jsc[i]), float(J0[i])
        if not (math.isfinite(jsc) and math.isfinite(j0)):
            continue
        if jsc <= 0.0 or j0 <= 0.0:
            continue
        voc = Vt * math.log(jsc / j0 + 1.0)
        if not (math.isfinite(voc) and voc > 0.0):
            continue
        V = np.linspace(0.0, voc, 256)
        J = jsc - j0 * (np.exp(V / Vt) - 1.0)
        P = V * J
        pmax = float(np.max(P))
        if math.isfinite(pmax) and pmax > 0.0:
            eta[i] = pmax / Pin
    return eta


def _overlay_sq(ax: Axes) -> None:
    if not SQ_EXT_PATH.exists():
        return
    omega, ie, phi_sun = _read_smarts_ext(str(SQ_EXT_PATH), "Global_tilted_irradiance")
    Eg_grid = np.linspace(0.05, 5.0, 400)
    eta_sq = _sq_eta_curve(omega, ie, phi_sun, Eg_grid, T=300.0)
    eta_pct = 100.0 * eta_sq
    ax.plot(Eg_grid, eta_pct, color="black", lw=1.4, label="SQ limit")
    if np.isfinite(eta_pct).any():
        j = int(np.nanargmax(eta_pct))
        ax.scatter([Eg_grid[j]], [eta_pct[j]], color="black", s=18, zorder=3)


def _density_scatter(ax: Axes, x, y):
    from scipy.stats import gaussian_kde

    vals = np.vstack([x, y])
    kde = gaussian_kde(vals)
    z = kde(vals)
    order = np.argsort(z)
    sc = ax.scatter(
        x[order],
        y[order],
        c=z[order],
        s=10,
        cmap="viridis",
        alpha=0.9,
        edgecolors="none",
    )
    return sc


def _term_reward(values, target, minv, maxv):
    """Mirror rewards/reward.py: ascending -> linear_scaling; float target ->
    centered linear_target (maxv-|v-target|)/(maxv-minv)."""
    if str(target).strip().lower() == "ascending":
        return _ascending_reward(values, minv=minv, maxv=maxv)
    t = float(target)
    return np.clip((maxv - np.abs(values - t)) / (maxv - minv), 0.0, 1.0)


def _bg_eta_terms(cfg):
    bg = eta = None
    for p in cfg.reward.prop_cfg:
        n = str(p.name)
        if n == "band_gap":
            bg = p
        if n.endswith("_eta"):
            eta = p
    return bg, eta


def plot_reward_function_landscape(out_dir: Path, cfg: Any) -> None:
    bg_cfg, eta_cfg = _bg_eta_terms(cfg)
    bt = bg_cfg.target
    bmin, bmax, bw = float(bg_cfg.minv), float(bg_cfg.maxv), float(bg_cfg.weight)
    emin, emax, ew = float(eta_cfg.minv), float(eta_cfg.maxv), float(eta_cfg.weight)
    bg_ascending = str(bt).strip().lower() == "ascending"
    bg_kind = "ascending" if bg_ascending else f"centered@{float(bt):g}"

    eg = np.linspace(-0.2, 5.0, 600)
    eta = np.linspace(-0.02, 0.45, 600)

    r_bg = _term_reward(eg, bt, bmin, bmax)
    r_eta = _ascending_reward(eta, minv=emin, maxv=emax)

    fig, axes = plt.subplots(
        1, 3, figsize=(13.8, 4.1), gridspec_kw={"width_ratios": [1, 1, 1.3]}
    )

    ax = axes[0]
    ax.plot(eg, r_bg, color="#2563EB", lw=2.0)
    for xv in ([bmin, bmax] if bg_ascending else [float(bt)]):
        ax.axvline(xv, color="gray", ls=":", lw=0.9)
    ax.set_xlabel(r"Band gap $E_g$ (eV)")
    ax.set_ylabel(r"Scaled reward $r_{E_g}$")
    ax.set_title(f"Band gap {bg_kind} ($w={bw:g}$, min={bmin:g}, max={bmax:g})")
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-0.05, 1.08)
    _faint_ygrid(ax)

    ax = axes[1]
    ax.plot(eta * 100.0, r_eta, color="#059669", lw=2.0)
    ax.axvline(emin * 100.0, color="gray", ls=":", lw=0.9)
    ax.axvline(emax * 100.0, color="gray", ls=":", lw=0.9)
    ax.set_xlabel(r"SLME efficiency $\eta$ (\%)")
    ax.set_ylabel(r"Scaled reward $r_{\eta}$")
    ax.set_title(f"SLME $\\eta$ ascending ($w={ew:g}$, min={emin*100:g}, max={emax*100:g}\\%)")
    ax.set_xlim(-2, 45)
    ax.set_ylim(-0.05, 1.08)
    _faint_ygrid(ax)

    ax = axes[2]
    eg_2d = np.linspace(-0.2, 5.0, 360)
    eta_2d = np.linspace(-0.02, 0.45, 360)
    EG, ETA = np.meshgrid(eg_2d, eta_2d)
    R_total = bw * _term_reward(EG, bt, bmin, bmax) + ew * _ascending_reward(
        ETA, minv=emin, maxv=emax
    )
    im = ax.pcolormesh(
        EG,
        ETA * 100.0,
        R_total,
        cmap="inferno",
        shading="gouraud",
        vmin=0,
        vmax=float(bw + ew),
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"Total reward $R$")
    ax.set_xlabel(r"Band gap $E_g$ (eV)")
    ax.set_ylabel(r"SLME efficiency $\eta$ (\%)")
    ax.set_title(f"$R = {bw:g}\\,r_{{E_g}} + {ew:g}\\,r_{{\\eta}}$")

    cs = ax.contour(
        EG,
        ETA * 100.0,
        R_total,
        levels=[0.25, 0.5, 0.75, 0.9],
        colors="white",
        linewidths=0.7,
        alpha=0.5,
    )
    ax.clabel(cs, fmt="%.2f", fontsize=7, colors="white")

    # SQ limit overlay
    if SQ_EXT_PATH.exists():
        omega, ie, phi_sun = _read_smarts_ext(
            str(SQ_EXT_PATH), "Global_tilted_irradiance"
        )
        Eg_grid = np.linspace(0.05, 5.0, 400)
        eta_sq = _sq_eta_curve(omega, ie, phi_sun, Eg_grid, T=300.0)
        eta_pct = 100.0 * eta_sq
        ax.plot(Eg_grid, eta_pct, color="black", lw=1.8, label="SQ limit")
        if np.isfinite(eta_pct).any():
            j = int(np.nanargmax(eta_pct))
            ax.scatter([Eg_grid[j]], [eta_pct[j]], color="black", s=22, zorder=5)
        ax.legend(loc="lower right", frameon=True, framealpha=0.8, fontsize=8)

    fig.tight_layout()
    out = out_dir / "reward_function_landscape.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_density_first_last(df: pd.DataFrame, out_dir: Path) -> None:
    steps = sorted(df["rl_step"].unique().tolist())
    first10 = steps[:10]
    last10 = steps[-10:]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)

    for ax, step_set, title in [
        (axes[0], first10, "First 10 steps"),
        (axes[1], last10, "Last 10 steps"),
    ]:
        sub = df[df["rl_step"].isin(step_set)]
        mask = np.isfinite(sub["band_gap_ev"]) & np.isfinite(sub["eta_percent"])
        x = sub.loc[mask, "band_gap_ev"].to_numpy(dtype=float)
        y = sub.loc[mask, "eta_percent"].to_numpy(dtype=float)

        if len(x) > 10:
            sc = _density_scatter(ax, x, y)
            cb = fig.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label("Point density")
            cb.locator = MaxNLocator(nbins=5)
            cb.formatter = FormatStrFormatter("%.3f")
            cb.update_ticks()

        _overlay_sq(ax)
        ax.set_xlabel("Band Gap (eV)")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 35)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(alpha=0.25)
        _disable_grid(ax)
        ax.set_title(f"{title} (n={len(x)})")
        ax.legend(loc="upper right", frameon=True)

    axes[0].set_ylabel(_eta_percent_label())
    fig.tight_layout()
    out = out_dir / "first10_vs_last10_density.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)


def plot_density_first_best(df: pd.DataFrame, out_dir: Path, exp_dir: Path) -> None:
    steps = sorted(df["rl_step"].unique().tolist())
    first1 = steps[:1]
    best_step = _best_step_from_metadata(exp_dir)
    if best_step is None or best_step not in steps:
        best_step = steps[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)

    for ax, step_set, title in [
        (axes[0], first1, f"Step {first1[0]}"),
        (axes[1], [best_step], f"Best step ({best_step})"),
    ]:
        sub = df[df["rl_step"].isin(step_set)]
        mask = np.isfinite(sub["band_gap_ev"]) & np.isfinite(sub["eta_percent"])
        x = sub.loc[mask, "band_gap_ev"].to_numpy(dtype=float)
        y = sub.loc[mask, "eta_percent"].to_numpy(dtype=float)

        if len(x) > 10:
            sc = _density_scatter(ax, x, y)
            cb = fig.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label("Point density")
            cb.locator = MaxNLocator(nbins=5)
            cb.formatter = FormatStrFormatter("%.3f")
            cb.update_ticks()

        _overlay_sq(ax)
        ax.set_xlabel("Band Gap (eV)")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 35)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        _disable_grid(ax)
        ax.set_title(f"{title} (n={len(x)})")
        ax.legend(loc="upper right", frameon=True)

    axes[0].set_ylabel(_eta_percent_label())
    fig.tight_layout()
    out = out_dir / "first1_vs_best_density.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)


def plot_density_all(df: pd.DataFrame, out_dir: Path) -> None:
    mask = np.isfinite(df["band_gap_ev"]) & np.isfinite(df["eta_percent"])
    x = df.loc[mask, "band_gap_ev"].to_numpy(dtype=float)
    y = df.loc[mask, "eta_percent"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    sc = _density_scatter(ax, x, y)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Point density")
    cb.locator = MaxNLocator(nbins=6)
    cb.formatter = FormatStrFormatter("%.3f")
    cb.update_ticks()

    _overlay_sq(ax)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlabel("Band Gap (eV)")
    ax.set_ylabel(_eta_percent_label())
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    _disable_grid(ax)
    fig.tight_layout()

    out = out_dir / "tsenn_slme_all_eta_vs_bandgap_density.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)


def plot_value_vs_step(
    df: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    ylabel: str,
    title: str,
    filename: str,
    color: str,
) -> None:
    steps = sorted(df["rl_step"].unique().tolist())
    medians = []
    p25s = []
    p75s = []
    means = []
    counts = []

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for step in steps:
        vals = df.loc[df["rl_step"] == step, value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            medians.append(np.nan)
            p25s.append(np.nan)
            p75s.append(np.nan)
            means.append(np.nan)
            counts.append(0)
            continue
        ax.scatter(
            np.full(len(vals), step, dtype=float),
            vals,
            s=8,
            alpha=0.12,
            color=color,
            edgecolors="none",
        )
        medians.append(float(np.median(vals)))
        p25s.append(float(np.percentile(vals, 25)))
        p75s.append(float(np.percentile(vals, 75)))
        means.append(float(np.mean(vals)))
        counts.append(len(vals))

    step_arr = np.array(steps, dtype=float)
    median_arr = np.array(medians, dtype=float)
    p25_arr = np.array(p25s, dtype=float)
    p75_arr = np.array(p75s, dtype=float)
    mean_arr = np.array(means, dtype=float)

    ax.fill_between(step_arr, p25_arr, p75_arr, color=color, alpha=0.18, label="IQR")
    ax.plot(step_arr, median_arr, color=color, lw=2.0, label="median")
    ax.plot(step_arr, mean_arr, color=color, lw=1.5, ls="--", alpha=0.9, label="mean")
    ax.set_xlabel("RL step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=250)
    plt.close(fig)


def _pick_representative_indices(df: pd.DataFrame) -> list[int]:
    eta = df["tsenn_slme_eta"].to_numpy(dtype=float)
    bg = df["band_gap"].to_numpy(dtype=float)

    finite = np.isfinite(eta) & np.isfinite(bg)
    valid_idx = np.where(finite)[0]
    if len(valid_idx) < 4:
        raise RuntimeError("Not enough valid rows to pick representative materials.")

    eta_valid = eta[valid_idx]
    bg_valid = bg[valid_idx]

    i_top = valid_idx[int(np.nanargmax(eta_valid))]
    i_low = valid_idx[int(np.nanargmin(eta_valid))]
    i_med = valid_idx[int(np.nanargmin(np.abs(eta_valid - np.nanmedian(eta_valid))))]
    i_bg13 = valid_idx[int(np.nanargmin(np.abs(bg_valid - 1.3)))]

    chosen: list[int] = []
    for i in [i_top, i_med, i_bg13, i_low]:
        if i not in chosen:
            chosen.append(i)

    if len(chosen) < 4:
        for i in valid_idx:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) == 4:
                break

    return chosen[:4]


def _latex_formula(formula: str) -> str:
    def _fmt(match: re.Match[str]) -> str:
        el = match.group(1)
        num = match.group(2)
        if num:
            return rf"\mathrm{{{el}}}_{{{num}}}"
        return rf"\mathrm{{{el}}}"

    formatted = re.sub(r"([A-Z][a-z]?)([0-9.]+)?", _fmt, formula)
    formatted = re.sub(r"\)([0-9.]+)", r")_{\1}", formatted)
    return formatted


def _load_tsenn_thickness_um(cfg) -> float:
    eta_cfg = _find_eta_prop_cfg(cfg)
    if eta_cfg is not None:
        return float(eta_cfg.calculator.thickness_um)
    return 0.1


def _find_eta_prop_cfg(cfg):
    for prop_cfg in cfg.reward.prop_cfg:
        if str(prop_cfg.name).endswith("_eta"):
            return prop_cfg
    return None


def _resolve_plot_model_path(model_path: object) -> str:
    path = str(model_path)
    path = path.replace("${hydra:runtime.cwd}", str(ROOT_DIR))
    return str(Path(path).expanduser())


def _tsenn_slme_from_run_config(cfg) -> TSENNSLME:
    eta_cfg = _find_eta_prop_cfg(cfg)
    if eta_cfg is None:
        raise RuntimeError("Run config does not define a TSENN eta reward")
    calc_cfg = OmegaConf.to_container(eta_cfg.calculator, resolve=False)
    if not isinstance(calc_cfg, dict):
        raise RuntimeError("TSENN eta calculator config is not a mapping")
    return TSENNSLME(
        root_dir="rewards/tsenn_slme_plot",
        task="eta",
        model_path=_resolve_plot_model_path(calc_cfg["model_path"]),
        device="cpu",
        batch_size=int(calc_cfg.get("batch_size", 8)),
        r_max=float(calc_cfg.get("r_max", 6.0)),
        out_dim=int(calc_cfg.get("out_dim", 300)),
        em_dim=int(calc_cfg.get("em_dim", 128)),
        lmax=int(calc_cfg.get("lmax", 2)),
        layers=int(calc_cfg.get("layers", 4)),
        mul=int(calc_cfg.get("mul", 32)),
        num_neighbors=float(calc_cfg.get("num_neighbors", 12.0)),
        scale_0e=float(calc_cfg.get("scale_0e", 1.0)),
        scale_2e=float(calc_cfg.get("scale_2e", 1.0)),
        dropout_prob=float(calc_cfg.get("dropout_prob", 0.0)),
        use_batch_norm=bool(calc_cfg.get("use_batch_norm", False)),
        output_mode=str(calc_cfg.get("output_mode", "tensor")),
        energy_min=float(calc_cfg.get("energy_min", 0.0)),
        energy_max=float(calc_cfg.get("energy_max", 30.0)),
        integration_lower_bound=str(calc_cfg.get("integration_lower_bound", "band_gap")),
        integration_energy_max=calc_cfg.get("integration_energy_max", None),
        thickness_um=float(calc_cfg.get("thickness_um", 0.5)),
        temperature_k=float(calc_cfg.get("temperature_k", 300.0)),
        radiative_fraction=float(calc_cfg.get("radiative_fraction", 1.0)),
        voltage_points=int(calc_cfg.get("voltage_points", 2000)),
        eg_mode=str(calc_cfg.get("eg_mode", "alignn")),
        fixed_eg_ev=float(calc_cfg.get("fixed_eg_ev", 1.3)),
        eta_backend=str(calc_cfg.get("eta_backend", "native")),
    )


def plot_absorption_spectra(out_dir: Path, records_df: pd.DataFrame, cfg) -> None:
    indices = _pick_representative_indices(records_df)

    structures: list[Structure] = []
    meta: list[tuple[str, float, float, int]] = []
    for idx in indices:
        row = records_df.iloc[idx]
        s = row.get("structure")
        if not isinstance(s, Structure):
            s = Structure.from_str(row["cif"], fmt="cif")
        structures.append(s)
        meta.append(
            (
                str(row["formula"]),
                float(row["band_gap"]),
                float(row["tsenn_slme_eta"]) * 100.0,
                int(row["step"]),
            )
        )

    calc = _tsenn_slme_from_run_config(cfg)

    energies_ev, eps2_iso, valid_mask = calc.tsenn.predict_epsilon2_iso(structures)
    eps2_iso = np.nan_to_num(eps2_iso, nan=0.0)
    eps1 = _kk_eps1_from_eps2(energies_ev, eps2_iso)
    alpha_um_inv = _absorption_coef_um_inv(energies_ev, eps1, eps2_iso)
    alpha_cm_inv = alpha_um_inv * 1.0e4
    run_thickness_um = _load_tsenn_thickness_um(cfg)
    thickness_styles = [
        (0.1, ":", 1.4),
        (run_thickness_um, "-", 1.8),
        (0.5, "--", 1.5),
    ]
    unique_thickness_styles = []
    seen_thicknesses: set[float] = set()
    for thickness_um, linestyle, linewidth in thickness_styles:
        key = round(float(thickness_um), 6)
        if key in seen_thicknesses:
            continue
        seen_thicknesses.add(key)
        unique_thickness_styles.append((float(thickness_um), linestyle, linewidth))
    absorptance_by_thickness = {
        thickness_um: 1.0
        - np.exp(-2.0 * np.clip(alpha_um_inv, 0.0, None) * thickness_um)
        for thickness_um, _, _ in unique_thickness_styles
    }

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for i in range(len(structures)):
        if not bool(valid_mask[i]):
            continue
        comp, bg, eta, step = meta[i]
        comp_tex = _latex_formula(comp)
        label = rf"${comp_tex}$ ($E_g={bg:.2f}\,\mathrm{{eV}},\ \eta={eta:.1f}\%$, step {step})"
        axes[0].plot(energies_ev, alpha_cm_inv[i], color=colors[i], lw=1.7, label=label)
        for thickness_um, linestyle, linewidth in unique_thickness_styles:
            axes[1].plot(
                energies_ev,
                absorptance_by_thickness[thickness_um][i],
                color=colors[i],
                lw=linewidth,
                linestyle=linestyle,
            )

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\alpha(E)\ (\mathrm{cm}^{-1})$")
    axes[0].set_title(r"TSENN-derived absorption spectra for representative materials")
    _faint_ygrid(axes[0])
    axes[0].legend(loc="center right", frameon=True)

    axes[1].set_xlabel(r"Photon energy $E$ (eV)")
    axes[1].set_ylabel(r"Absorptance $A(E)=1-e^{-2\alpha(E)L}$")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlim(0.0, 10.0)
    _faint_ygrid(axes[1])

    thickness_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=linewidth,
            linestyle=linestyle,
            label=rf"$L={thickness_um:g}\,\mu$m",
        )
        for thickness_um, linestyle, linewidth in unique_thickness_styles
    ]
    axes[1].legend(
        handles=thickness_handles,
        title=r"Thickness",
        loc="center right",
        frameon=True,
    )

    fig.tight_layout()
    out = out_dir / "tsenn_absorption_spectra_latex.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else exp_dir / "deliverables"
    )
    cfg = load_run_config(exp_dir)
    bg_dir, eta_dir = _resolve_reward_dirs(exp_dir, cfg)
    records_df = _load_step_records(exp_dir, cfg)

    _configure_latex_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _build_step_df(bg_dir, eta_dir)
    if df.empty:
        raise RuntimeError(f"No step data found under {exp_dir / 'rewards'}.")

    plot_reward_function_landscape(out_dir, cfg)
    plot_value_vs_step(
        df,
        out_dir,
        value_col="band_gap_ev",
        ylabel="Band gap (eV)",
        title="Band gap vs step",
        filename="band_gap_vs_step.png",
        color="#2563EB",
    )
    plot_value_vs_step(
        df,
        out_dir,
        value_col="eta_percent",
        ylabel=_eta_percent_label(),
        title="SLME vs step",
        filename="slme_vs_step.png",
        color="#059669",
    )
    plot_density_first_last(df, out_dir)
    plot_density_first_best(df, out_dir, exp_dir)
    plot_absorption_spectra(out_dir, records_df, cfg)
    plot_density_all(df, out_dir)

    print(str(out_dir / "reward_function_landscape.png"))
    print(str(out_dir / "band_gap_vs_step.png"))
    print(str(out_dir / "slme_vs_step.png"))
    print(str(out_dir / "first10_vs_last10_density.png"))
    print(str(out_dir / "first1_vs_best_density.png"))
    print(str(out_dir / "tsenn_absorption_spectra_latex.png"))
    print(str(out_dir / "tsenn_slme_all_eta_vs_bandgap_density.png"))


if __name__ == "__main__":
    main()
