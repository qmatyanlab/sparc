import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from rewards.calculators.base import Calculator
from rewards.calculators.tsenn.calc import TSENN


class TSENNStaticDielectric(Calculator):
    VALID_SCALAR_MODES = {
        "trace_mean",
        "max_diag",
        "anisotropy",
        "component",
        "xz_ratio",
        "layered_uniaxial",
        "inplane_isotropy",
    }

    VALID_STANDARDIZE_MODES = {"none", "conventional", "refined"}

    def __init__(
        self,
        root_dir: str,
        task: str = "static_dielectric",
        model_path: str | None = None,
        config_path: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
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
        scalar_mode: str = "trace_mean",
        component_i: int = 0,
        component_j: int = 0,
        standardize_structure: str = "none",
        standardize_symprec: float = 0.01,
        inplane_penalty_power: float = 1.0,
        z_anisotropy_floor: float = 0.0,
    ) -> None:
        super().__init__(root_dir, task)
        self.root_path = Path(self.root_dir).resolve()

        self.scalar_mode = str(scalar_mode).lower()
        if self.scalar_mode not in self.VALID_SCALAR_MODES:
            raise ValueError(
                f"Invalid scalar_mode '{scalar_mode}'. "
                f"Valid modes: {sorted(self.VALID_SCALAR_MODES)}"
            )

        self.component_i = int(component_i)
        self.component_j = int(component_j)
        if self.scalar_mode == "component":
            if self.component_i not in (0, 1, 2) or self.component_j not in (0, 1, 2):
                raise ValueError("component_i and component_j must be in {0, 1, 2}")

        self.standardize_structure = str(standardize_structure).lower()
        if self.standardize_structure not in self.VALID_STANDARDIZE_MODES:
            raise ValueError(
                "standardize_structure must be one of "
                f"{sorted(self.VALID_STANDARDIZE_MODES)}"
            )
        self.standardize_symprec = float(standardize_symprec)

        self.inplane_penalty_power = float(inplane_penalty_power)
        if self.inplane_penalty_power <= 0.0:
            raise ValueError("inplane_penalty_power must be > 0.")

        # Light z-anisotropy floor for scalar_mode="inplane_isotropy": 0 disables it (pure in-plane
        # score); >0 multiplies the score by min(1, layered_anisotropy / z_anisotropy_floor) so that
        # fully-isotropic cubic (layered_anisotropy == 0) is excluded while any structure with modest
        # out-of-plane anisotropy keeps full credit.
        self.z_anisotropy_floor = float(z_anisotropy_floor)
        if self.z_anisotropy_floor < 0.0:
            raise ValueError("z_anisotropy_floor must be >= 0.")

        if model_path is None:
            raise ValueError("TSENNStaticDielectric model_path must be provided.")
        from utils.assets import resolve_path

        model_file = resolve_path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"TSENNStaticDielectric model not found: {model_path}"
            )
        self.model_path = str(model_file.resolve())

        mcfg = self.load_model_cfg(config_path)
        self.tsenn = TSENN(
            root_dir=root_dir,
            task=task,
            model_path=self.model_path,
            device=device,
            batch_size=batch_size,
            r_max=float(self.pick(mcfg, r_max, "r_max", 6.0)),
            out_dim=int(self.pick(mcfg, out_dim, "out_dim", 1)),
            em_dim=int(self.pick(mcfg, em_dim, "em_dim", 64)),
            lmax=int(self.pick(mcfg, lmax, "lmax", 2)),
            layers=int(self.pick(mcfg, layers, "layers", 2)),
            mul=int(self.pick(mcfg, mul, "mul", 32)),
            num_neighbors=float(self.pick(mcfg, num_neighbors, "num_neighbors", 59.902574690065045)),
            scale_0e=float(self.pick(mcfg, scale_0e, "scale_0e", 8.20143833581954)),
            scale_2e=float(self.pick(mcfg, scale_2e, "scale_2e", 0.6969301341467804)),
            dropout_prob=float(self.pick(mcfg, dropout_prob, "dropout_prob", 0.4)),
            use_batch_norm=bool(self.pick(mcfg, use_batch_norm, "use_batch_norm", False)),
        )
        if self.tsenn.out_dim != 1:
            raise ValueError(
                "TSENNStaticDielectric expects out_dim=1 for static-limit tensor prediction."
            )

    def predict_static_tensor(
        self,
        struc_list: List[Structure],
    ) -> tuple[np.ndarray, np.ndarray]:
        prepared_structures = []
        prepared_indices = []
        for index, structure in enumerate(struc_list):
            if self.standardize_structure == "conventional":
                try:
                    structure = SpacegroupAnalyzer(
                        structure, symprec=self.standardize_symprec
                    ).get_conventional_standard_structure()
                except Exception:
                    continue
            elif self.standardize_structure == "refined":
                try:
                    structure = SpacegroupAnalyzer(
                        structure, symprec=self.standardize_symprec
                    ).get_refined_structure()
                except Exception:
                    continue
            prepared_structures.append(structure)
            prepared_indices.append(index)

        tensors = np.full((len(struc_list), 3, 3), np.nan, dtype=float)
        valid_mask = np.zeros(len(struc_list), dtype=bool)
        if not prepared_structures:
            return tensors, valid_mask

        _, prepared_tensors, prepared_valid_mask = self.tsenn.predict_epsilon2_tensor(
            prepared_structures,
            energy_min=0.0,
            energy_max=0.0,
        )

        prepared_tensors = prepared_tensors[:, 0]
        prepared_tensors = 0.5 * (
            prepared_tensors + np.swapaxes(prepared_tensors, 1, 2)
        )
        for prepared_index, source_index in enumerate(prepared_indices):
            if not prepared_valid_mask[prepared_index]:
                continue
            tensors[source_index] = prepared_tensors[prepared_index]
            valid_mask[source_index] = True

        return tensors, valid_mask

    def _resolve_output_path(self, label: str, suffix: str) -> str:
        safe_label = str(label)
        if safe_label in {"", ".", ".."}:
            raise ValueError("label must be a non-empty filename stem")
        if Path(safe_label).name != safe_label:
            raise ValueError("label must not contain path separators")

        path = (self.root_path / f"{safe_label}{suffix}").resolve()
        if path.parent != self.root_path:
            raise ValueError("label must resolve within calculator root_dir")
        return str(path)

    def _reduce_tensor_to_scalar(
        self,
        tensors: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        results = np.full(len(tensors), np.nan, dtype=float)
        if not bool(np.any(valid_mask)):
            return results

        valid_tensors = tensors[valid_mask]

        if self.scalar_mode == "trace_mean":
            scalars = np.trace(valid_tensors, axis1=1, axis2=2) / 3.0
        elif self.scalar_mode == "max_diag":
            scalars = np.max(np.diagonal(valid_tensors, axis1=1, axis2=2), axis=1)
        elif self.scalar_mode == "anisotropy":
            iso_scalar = np.trace(valid_tensors, axis1=1, axis2=2)[:, None, None] / 3.0
            scalars = np.linalg.norm(
                valid_tensors - iso_scalar * np.eye(3)[None],
                axis=(1, 2),
            )
        elif self.scalar_mode == "xz_ratio":
            scalars = np.abs(valid_tensors[:, 0, 2]) / (
                np.abs(valid_tensors[:, 0, 0]) + np.abs(valid_tensors[:, 2, 2]) + 1e-8
            )
        elif self.scalar_mode == "layered_uniaxial":
            eps_xx = valid_tensors[:, 0, 0]
            eps_yy = valid_tensors[:, 1, 1]
            eps_zz = valid_tensors[:, 2, 2]
            eps_perp = 0.5 * (eps_xx + eps_yy)
            layered_anisotropy = np.abs(eps_zz - eps_perp) / (
                np.abs(eps_zz) + np.abs(eps_perp) + 1e-8
            )
            inplane_mismatch = np.abs(eps_xx - eps_yy) / (
                np.abs(eps_xx) + np.abs(eps_yy) + 1e-8
            )
            # In-plane isotropy (eps_xx == eps_yy) is symmetry-protected only in the
            # uniaxial crystal classes. Raising the (1 - mismatch) gate to a power > 1
            # makes biaxial (orthorhombic/monoclinic/triclinic) structures fall below
            # the reward ceiling while clean uniaxial ones still saturate, so RL can
            # discover and condense onto the uniaxial family without a hard SG whitelist.
            scalars = layered_anisotropy * (
                (1.0 - inplane_mismatch) ** self.inplane_penalty_power
            )
        elif self.scalar_mode == "inplane_isotropy":
            # Target ONLY the in-plane relation eps_xx == eps_yy (the signature of the in-plane
            # rotation groups: trigonal/tetragonal/hexagonal + cubic). No z-anisotropy term, so
            # this does NOT distinguish uniaxial (eps_zz != eps_perp) from fully isotropic cubic.
            # Returns 1.0 when eps_xx == eps_yy (in-plane isotropic), lower as they diverge.
            # In [0,1] already; use linear_scaling(minv,maxv) to set the discrimination window.
            eps_xx = valid_tensors[:, 0, 0]
            eps_yy = valid_tensors[:, 1, 1]
            eps_zz = valid_tensors[:, 2, 2]
            inplane_mismatch = np.abs(eps_xx - eps_yy) / (
                np.abs(eps_xx) + np.abs(eps_yy) + 1e-8
            )
            scalars = 1.0 - inplane_mismatch
            if self.z_anisotropy_floor > 0.0:
                # Light floor: exclude fully-isotropic cubic (layered_anisotropy == 0) without
                # re-rewarding the DEGREE of z-anisotropy. z_gate saturates to 1 once the
                # out-of-plane anisotropy reaches z_anisotropy_floor.
                eps_perp = 0.5 * (eps_xx + eps_yy)
                layered_anisotropy = np.abs(eps_zz - eps_perp) / (
                    np.abs(eps_zz) + np.abs(eps_perp) + 1e-8
                )
                z_gate = np.clip(layered_anisotropy / self.z_anisotropy_floor, 0.0, 1.0)
                scalars = scalars * z_gate
        else:
            scalars = valid_tensors[:, self.component_i, self.component_j]

        results[valid_mask] = scalars
        return results

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = "tmp",
    ):
        struc_list = samples[0]
        out_path = self._resolve_output_path(label, ".txt")
        tensor_path = self._resolve_output_path(f"{label}_tensor", ".npz")

        tensors, valid_mask = self.predict_static_tensor(struc_list)
        results = self._reduce_tensor_to_scalar(tensors, valid_mask)

        np.savetxt(out_path, results, fmt="%.6f")
        np.savez(tensor_path, tensor=tensors, valid_mask=valid_mask)
        return results
