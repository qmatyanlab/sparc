import os
import math
from dataclasses import dataclass
from typing import List

import numpy as np
from pymatgen.core.structure import Structure

from rewards.calculators.base import Calculator
from rewards.calculators.tsenn.calc import TSENN


HC_EV_NM = 1239.8419843320026
INV_HBARC_EV_UM = 5.067726


@dataclass(frozen=True)
class _Constants:
    q: float = 1.602176634e-19
    h: float = 6.62607015e-34
    hbar: float = 1.054571817e-34
    c: float = 299792458.0
    kB: float = 1.380649e-23


CONST = _Constants()


def _load_astmg173_global_tilt(path: str):
    arr = np.genfromtxt(path, delimiter=",", skip_header=2)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected ASTM G173 CSV shape: {arr.shape}")

    wavelength_nm = arr[:, 0]
    global_tilt_w_m2_nm = arr[:, 2]

    energies_ev = HC_EV_NM / wavelength_nm
    dlam_dE = HC_EV_NM / (energies_ev**2)
    irradiance_w_m2_ev = global_tilt_w_m2_nm * dlam_dE
    energies_j = energies_ev * CONST.q
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_ev = irradiance_w_m2_ev / energies_j
    phi_ev = np.nan_to_num(phi_ev, nan=0.0, posinf=0.0, neginf=0.0)

    order = np.argsort(energies_ev)
    return energies_ev[order], phi_ev[order]


def _read_smarts_ext(path: str, irradiance_col: str):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if not header:
            raise RuntimeError(f"Empty SMARTS ext file: {path}")
        try:
            i_w = header.index("Wvlgth")
        except ValueError as exc:
            raise KeyError(f"Wvlgth column not found in {path}") from exc
        try:
            i_I = header.index(irradiance_col)
        except ValueError as exc:
            raise KeyError(
                f"{irradiance_col} column not found in {path}; cols={header}"
            ) from exc

        wl_nm = []
        I_lambda = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            wl_nm.append(float(parts[i_w]))
            I_lambda.append(float(parts[i_I]))

    wl_nm_arr = np.asarray(wl_nm, dtype=float)
    I_lambda_arr = np.asarray(I_lambda, dtype=float)

    energies_ev = HC_EV_NM / wl_nm_arr
    dlam_dE_abs = HC_EV_NM / (energies_ev * energies_ev)
    I_E = I_lambda_arr * dlam_dE_abs
    energies_j = energies_ev * CONST.q
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_ev = I_E / energies_j
    phi_ev = np.nan_to_num(phi_ev, nan=0.0, posinf=0.0, neginf=0.0)

    order = np.argsort(energies_ev)
    return energies_ev[order], phi_ev[order], I_E[order]


def _trapz_weights(x):
    w = np.zeros_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def _kk_eps1_from_eps2(omega_eV, eps2):
    omega = omega_eV.astype(float).copy()
    omega[0] = max(float(omega[0]), 1e-12)
    w = _trapz_weights(omega)
    om2 = omega * omega
    denom = om2[None, :] - om2[:, None]
    np.fill_diagonal(denom, np.inf)
    M = (omega[None, :] / denom) * w[None, :]
    np.fill_diagonal(M, 0.0)
    return 1.0 + (2.0 / math.pi) * (eps2 @ M.T)


def _absorption_coef_um_inv(omega_eV, eps1, eps2):
    abs_eps = np.sqrt(eps1 * eps1 + eps2 * eps2)
    term = np.sqrt(np.maximum(abs_eps - eps1, 0.0))
    return (math.sqrt(2.0) * INV_HBARC_EV_UM) * omega_eV[None, :] * term


def _phi_bb_eV(omega_eV, temperature_k: float):
    E_J = omega_eV * CONST.q
    x = np.clip(E_J / (CONST.kB * float(temperature_k)), 1e-12, 700.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denom = np.expm1(x)
        phi_per_J = (2.0 * math.pi) / (CONST.h**3 * CONST.c**2) * (E_J * E_J) / denom
    phi_per_eV = phi_per_J * CONST.q
    phi_per_eV = np.nan_to_num(phi_per_eV, nan=0.0, posinf=0.0, neginf=0.0)
    return phi_per_eV


def _pymatgen_slme_eta_fraction(
    energies_ev: np.ndarray,
    alpha_um_inv: np.ndarray,
    gap_ev: float,
    thickness_um: float,
    temperature_k: float,
) -> float:
    if not math.isfinite(float(gap_ev)) or float(gap_ev) <= 0.0:
        return float("nan")
    try:
        from pymatgen.analysis.solar.slme import slme

        return (
            float(
                slme(
                    energies_ev,
                    alpha_um_inv * 1.0e4,
                    float(gap_ev),
                    float(gap_ev),
                    thickness=float(thickness_um) * 1.0e-6,
                    temperature=float(temperature_k),
                    absorbance_in_inverse_centimeters=True,
                    cut_off_absorbance_below_direct_allowed_gap=True,
                    plot_current_voltage=False,
                )
            )
            / 100.0
        )
    except Exception:
        return float("nan")


class TSENNSLME(Calculator):
    VALID_TASKS = [
        "eta",
        "jsc",
        "j0",
        "voc",
        "pmax",
        "psolar",
        "neg_penalty",
    ]

    def __init__(
        self,
        root_dir: str,
        task: str = "eta",
        model_path: str | None = None,
        config_path: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
        # architecture: None => read from config_path YAML `model:` block, else fallback default
        r_max: float | None = None,
        out_dim: int | None = None,
        em_dim: int | None = None,
        lmax: int | None = None,
        layers: int | None = None,
        mul: int | None = None,
        num_neighbors: float | None = None,
        scale_0e: float | None = None,
        scale_2e: float | None = None,
        dropout_prob: float | None = None,
        use_batch_norm: bool | None = None,
        output_mode: str | None = None,
        max_edges: int | None = None,
        max_edges_per_atom: float | None = 256.0,
        energy_min: float = 0.0,
        energy_max: float = 30.0,
        integration_lower_bound: str = "band_gap",
        integration_energy_max: float | None = None,
        thickness_um: float = 0.5,
        temperature_k: float = 300.0,
        radiative_fraction: float = 1.0,
        voltage_points: int = 2000,
        eg_mode: str = "e3nn",
        eg_model_path: str | None = None,
        eg_config_path: str | None = None,
        fixed_eg_ev: float = 1.3,
        solar_source: str = "smarts",
        solar_path: str | None = None,
        solar_irradiance_col: str = "Global_tilted_irradiance",
        eta_backend: str = "native",
    ) -> None:
        super().__init__(root_dir, task)
        if task not in self.VALID_TASKS:
            raise ValueError(f"Invalid task '{task}'. Valid tasks: {self.VALID_TASKS}")

        if radiative_fraction <= 0.0:
            raise ValueError("radiative_fraction must be > 0")

        self.thickness_um = float(thickness_um)
        self.temperature_k = float(temperature_k)
        self.radiative_fraction = float(radiative_fraction)
        self.voltage_points = int(voltage_points)
        self.energy_min = float(energy_min)
        self.energy_max = float(energy_max)
        self.integration_lower_bound = str(integration_lower_bound).lower()
        if self.integration_lower_bound not in {"band_gap", "energy_min"}:
            raise ValueError(
                "integration_lower_bound must be 'band_gap' or 'energy_min'"
            )
        self.integration_energy_max = (
            None if integration_energy_max is None else float(integration_energy_max)
        )
        if self.voltage_points < 50:
            raise ValueError("voltage_points must be >= 50")

        self.eg_mode = str(eg_mode)
        self.eg_model_path = eg_model_path
        self.eg_config_path = eg_config_path
        self._eg_calc = None  # lazily built, reused across steps
        self.fixed_eg_ev = float(fixed_eg_ev)
        self.eta_backend = str(eta_backend).lower()
        if self.eta_backend not in {"native", "pymatgen"}:
            raise ValueError("eta_backend must be 'native' or 'pymatgen'")

        self.solar_source = str(solar_source)
        self.solar_irradiance_col = str(solar_irradiance_col)
        if solar_path is None:
            if self.solar_source == "smarts":
                solar_path = os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    "SMARTS_295_Linux",
                    "Examples",
                    "Example 6-USSA_084",
                    "smarts295.ext.txt",
                )
            else:
                solar_path = os.path.join(
                    os.path.dirname(__file__), "data", "ASTMG173.csv"
                )
        self.solar_path = os.path.abspath(str(solar_path))
        if not os.path.exists(self.solar_path):
            raise FileNotFoundError(f"Solar spectrum not found: {self.solar_path}")

        if self.solar_source == "smarts":
            self.solar_energies_ev, self.solar_phi_ev, self.solar_I_ev = (
                _read_smarts_ext(
                    self.solar_path, irradiance_col=self.solar_irradiance_col
                )
            )
        elif self.solar_source == "astm":
            self.solar_energies_ev, self.solar_phi_ev = _load_astmg173_global_tilt(
                self.solar_path
            )
            self.solar_I_ev = None
        else:
            raise ValueError("solar_source must be 'smarts' or 'astm'")

        mcfg = self.load_model_cfg(config_path)
        self.tsenn = TSENN(
            root_dir=os.path.join(root_dir, "tsenn_cache"),
            task="tsenn_dielectric",
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            r_max=float(self.pick(mcfg, r_max, "r_max", 6.0)),
            out_dim=int(self.pick(mcfg, out_dim, "out_dim", 300)),
            em_dim=int(self.pick(mcfg, em_dim, "em_dim", 128)),
            lmax=int(self.pick(mcfg, lmax, "lmax", 2)),
            layers=int(self.pick(mcfg, layers, "layers", 4)),
            mul=int(self.pick(mcfg, mul, "mul", 32)),
            num_neighbors=float(self.pick(mcfg, num_neighbors, "num_neighbors", 12.0)),
            scale_0e=float(self.pick(mcfg, scale_0e, "scale_0e", 1.0)),
            scale_2e=float(self.pick(mcfg, scale_2e, "scale_2e", 1.0)),
            dropout_prob=float(self.pick(mcfg, dropout_prob, "dropout_prob", 0.0)),
            use_batch_norm=bool(self.pick(mcfg, use_batch_norm, "use_batch_norm", False)),
            output_mode=str(self.pick(mcfg, output_mode, "output_mode", "tensor")),
            max_edges=max_edges,
            max_edges_per_atom=max_edges_per_atom,
        )

    def compute_all_metrics(self, struc_list: List[Structure]):
        N = len(struc_list)
        energies_ev, eps2_iso, valid_mask = self.tsenn.predict_epsilon2_iso(
            struc_list,
            energy_min=self.energy_min,
            energy_max=self.energy_max,
        )

        phi_solar = np.interp(
            energies_ev,
            self.solar_energies_ev,
            self.solar_phi_ev,
            left=0.0,
            right=0.0,
        )
        if self.solar_I_ev is None:
            i_e = (energies_ev * CONST.q) * phi_solar
        else:
            i_e = np.interp(
                energies_ev,
                self.solar_energies_ev,
                self.solar_I_ev,
                left=0.0,
                right=0.0,
            )
        psolar = float(np.trapz(i_e, energies_ev))

        phi_bb = _phi_bb_eV(energies_ev, self.temperature_k)

        eg_ev = None
        if self.integration_lower_bound == "band_gap":
            if self.eg_mode == "e3nn":
                if self._eg_calc is None:
                    from rewards.calculators.e3nn_bandgap import E3NNBandGap

                    self._eg_calc = E3NNBandGap(
                        root_dir=os.path.join(self.root_dir, "_e3nn_bandgap"),
                        task="band_gap",
                        model_path=self.eg_model_path,
                        config_path=self.eg_config_path,
                        device=self.tsenn.device,
                        silent=True,
                    )
                eg_ev = self._eg_calc.calc((struc_list, ""), label="band_gap")
            elif self.eg_mode == "fixed":
                eg_ev = np.full(N, self.fixed_eg_ev, dtype=float)
            elif self.eg_mode == "none":
                eg_ev = np.zeros(N, dtype=float)
            else:
                raise ValueError(
                    "eg_mode must be 'e3nn', 'fixed', or 'none'")

        eta = np.full(N, np.nan, dtype=float)
        jsc = np.full(N, np.nan, dtype=float)
        j0 = np.full(N, np.nan, dtype=float)
        voc = np.full(N, np.nan, dtype=float)
        pmax = np.full(N, np.nan, dtype=float)
        neg_penalty = np.full(N, np.nan, dtype=float)

        beta = CONST.q / (CONST.kB * self.temperature_k)

        eps2_for_kk = np.nan_to_num(eps2_iso, nan=0.0)
        eps1_all = _kk_eps1_from_eps2(energies_ev, eps2_for_kk)
        alpha_um_inv_all = _absorption_coef_um_inv(energies_ev, eps1_all, eps2_for_kk)
        absorptance_all = 1.0 - np.exp(
            -2.0 * np.clip(alpha_um_inv_all, 0.0, None) * float(self.thickness_um)
        )
        integration_upper_bound = (
            float(energies_ev[-1])
            if self.integration_energy_max is None
            else min(float(self.integration_energy_max), float(energies_ev[-1]))
        )

        for i in range(N):
            if not valid_mask[i]:
                continue

            eps2_raw = eps2_iso[i]
            if not np.isfinite(eps2_raw).all():
                continue

            neg_mag = float(np.clip(-eps2_raw, 0.0, None).mean())
            tot_mag = float(np.abs(eps2_raw).mean() + 1e-12)
            neg_penalty[i] = float(np.clip(neg_mag / tot_mag, 0.0, 1.0))

            lower_bound_ev = self.energy_min
            if self.integration_lower_bound == "band_gap":
                Eg_i = float(eg_ev[i])
                if not math.isfinite(Eg_i) or Eg_i <= 0.0:
                    continue
                lower_bound_ev = Eg_i

            mask = (energies_ev >= lower_bound_ev) & (
                energies_ev <= integration_upper_bound
            )
            if not bool(np.any(mask)):
                continue

            absorptance = absorptance_all[i]

            jsc_i = CONST.q * float(
                np.trapz(absorptance[mask] * phi_solar[mask], energies_ev[mask])
            )
            if not np.isfinite(jsc_i) or jsc_i <= 0.0:
                continue

            j0_rad = CONST.q * float(
                np.trapz(absorptance[mask] * phi_bb[mask], energies_ev[mask])
            )
            j0_i = j0_rad / self.radiative_fraction
            if not np.isfinite(j0_i) or j0_i <= 0.0:
                continue

            voc_i = float((1.0 / beta) * np.log1p(jsc_i / j0_i))
            if not np.isfinite(voc_i) or voc_i <= 0.0:
                continue

            V = np.linspace(0.0, voc_i, self.voltage_points)
            J = jsc_i - j0_i * (np.exp(beta * V) - 1.0)
            P = J * V
            pmax_i = float(np.max(P))
            if not np.isfinite(pmax_i) or pmax_i <= 0.0 or psolar <= 0.0:
                continue

            if self.eta_backend == "pymatgen":
                backend_mask = energies_ev <= integration_upper_bound
                eta_i = _pymatgen_slme_eta_fraction(
                    energies_ev=energies_ev[backend_mask],
                    alpha_um_inv=alpha_um_inv_all[i, backend_mask],
                    gap_ev=lower_bound_ev,
                    thickness_um=self.thickness_um,
                    temperature_k=self.temperature_k,
                )
                if not np.isfinite(eta_i):
                    continue
            else:
                eta_i = pmax_i / psolar
            eta[i] = eta_i
            jsc[i] = jsc_i
            j0[i] = j0_i
            voc[i] = voc_i
            pmax[i] = pmax_i

        psolar_arr = np.full(N, psolar, dtype=float)
        return {
            "eta": eta,
            "jsc": jsc,
            "j0": j0,
            "voc": voc,
            "pmax": pmax,
            "psolar": psolar_arr,
            "neg_penalty": neg_penalty,
            "valid_mask": valid_mask,
        }

    def calc(self, *args, **kwargs):
        if args:
            samples = args[0]
            label = args[1] if len(args) > 1 else kwargs.get("label", "tmp")
        else:
            samples = kwargs["samples"]
            label = kwargs.get("label", "tmp")

        struc_list = samples[0]
        out_path = os.path.join(self.root_dir, f"{label}.txt")
        out_path = os.path.abspath(out_path)

        metrics = self.compute_all_metrics(struc_list)
        results = metrics[self.task]
        np.savetxt(out_path, results, fmt="%.6f")
        return results
