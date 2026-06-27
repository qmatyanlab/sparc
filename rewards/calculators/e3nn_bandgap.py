"""E3NN band-gap calculator (drop-in replacement for the ALIGNN band-gap reward).

Loads the project's own E3NN band-gap regressor (e.g. optuna_bandgap_trial_2) and
predicts a scalar band gap per structure. Interface matches ALIGNN.calc so it can be
swapped into the reward config or used as the SLME integration-lower-bound source.

CRITICAL: this model MUST be run through the softplus-OFF Network
(data.dielectric.utils.utils_model_scalar.Network) -- the TSENN Network applies
_apply_softplus_to_0e to the 0e channel (positive-definiteness for the dielectric
tensor), which the band-gap weights were trained WITHOUT. Running it with softplus on
floors every prediction to softplus(0)=ln2~0.693 eV and destroys metal discrimination.
Validated against the cached test split: MAE 0.19 eV, R^2 0.91.
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import yaml
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from rewards.calculators.base import Calculator
from rewards.calculators.tsenn.calc import build_graph, load_or_build_onehot

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# softplus-OFF Network (the one the band-gap checkpoint was trained with).
# IMPORTANT: utils_model_scalar runs torch.set_default_dtype(float64) at MODULE import,
# which flips the process-global default dtype and corrupts the float32 SymmCD diffusion
# model when this calculator is loaded inside the RL pipeline (sampling crashes with
# "mat1 and mat2 must have the same dtype, Float vs Double"). E3NNBandGap sets float64
# explicitly on its own model/graphs, so it does not rely on the global default --
# capture and restore it around the import to contain the side effect.
_prev_default_dtype = torch.get_default_dtype()
from data.dielectric.utils.utils_model_scalar import Network  # noqa: E402
torch.set_default_dtype(_prev_default_dtype)

_DEFAULT_MODEL = os.path.join(
    _PROJECT_ROOT, "data/dielectric/optuna_bandgap_trial_2_gpu0_best.torch")
_DEFAULT_CONFIG = os.path.join(
    _PROJECT_ROOT, "data/dielectric/e3_band_gap_inference_config.yaml")


class _NetWrapper(Network):
    """Embedding heads + mean-pool, around the softplus-OFF scalar Network.

    Identical to scripts.plot_band_gap_splits.NetWrapper (the validated path).
    """

    def __init__(self, in_dim: int, em_dim: int, **kwargs) -> None:
        self.pool = False
        if kwargs.get("reduce_output", False):
            kwargs["reduce_output"] = False
            self.pool = True
        super().__init__(**kwargs)
        self.em_z = nn.Linear(in_dim, em_dim)
        self.em_x = nn.Linear(in_dim, em_dim)

    def forward(self, data: Data) -> torch.Tensor:
        data.z = F.relu(self.em_z(data.z))
        data.x = F.relu(self.em_x(data.x))
        output = super().forward(data)
        if self.pool:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)
        return output


class E3NNBandGap(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = "band_gap",
        model_path: str | None = None,
        config_path: str | None = None,
        device: str | None = None,
        batch_size: int = 256,
        in_dim: int = 118,
        em_dim: int | None = None,
        layers: int | None = None,
        mul: int | None = None,
        lmax: int | None = None,
        r_max: float | None = None,
        num_neighbors: float | None = None,
        clip_negative: bool = True,
        silent: bool = True,
    ) -> None:
        super().__init__(root_dir, task)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.clip_negative = bool(clip_negative)
        self.silent = bool(silent)
        self.dtype = torch.float64

        from utils.assets import resolve_path

        self.model_path = str(resolve_path(model_path or _DEFAULT_MODEL))
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"E3NN band-gap model not found: {self.model_path}")
        cfg_path = str(resolve_path(config_path or _DEFAULT_CONFIG, optional=True))
        mcfg = {}
        if os.path.exists(cfg_path):
            mcfg = (yaml.safe_load(open(cfg_path)) or {}).get("model", {}) or {}

        def pick(arg, key, default):
            return arg if arg is not None else mcfg.get(key, default)

        self.em_dim = int(pick(em_dim, "em_dim", 128))
        self.layers = int(pick(layers, "layers", 4))
        self.mul = int(pick(mul, "mul", 16))
        self.lmax = int(pick(lmax, "lmax", 2))
        self.r_max = float(pick(r_max, "r_max", 6.0))
        self.num_neighbors = float(pick(num_neighbors, "num_neighbors",
                                        55.328226741470544))

        self.model = _NetWrapper(
            in_dim=int(in_dim), em_dim=self.em_dim,
            irreps_in=f"{self.em_dim}x0e", irreps_out="1x0e",
            irreps_node_attr=f"{self.em_dim}x0e",
            layers=self.layers, mul=self.mul, lmax=self.lmax,
            max_radius=self.r_max, num_neighbors=self.num_neighbors,
            reduce_output=True, dropout_prob=0.0, use_batch_norm=False,
        ).to(self.device, dtype=self.dtype)
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state["state"])
        self.model.eval()

        (self.type_onehot, self.mass_onehot, self.dipole_onehot,
         self.radius_onehot, self.type_encoding) = load_or_build_onehot(
            self.root_dir, dtype=self.dtype)

    def predict(self, struc_list: List[Structure]) -> np.ndarray:
        out = np.full(len(struc_list), np.nan, dtype=float)
        data_list, valid = [], []
        for i, s in enumerate(struc_list):
            try:
                g = build_graph(s, self.type_onehot, self.mass_onehot,
                                self.dipole_onehot, self.radius_onehot,
                                self.type_encoding, self.r_max, dtype=self.dtype)
            except Exception:  # noqa: BLE001
                continue
            data_list.append(Data(
                pos=g.pos.double(), x=g.x.double(), z=g.z.double(),
                edge_index=g.edge_index.long(), edge_vec=g.edge_vec.double()))
            valid.append(i)
        if data_list:
            preds = []
            with torch.no_grad():
                for batch in DataLoader(data_list, batch_size=self.batch_size,
                                        shuffle=False):
                    preds.append(self.model(batch.to(self.device)).view(-1).cpu().numpy())
            for i, p in zip(valid, np.concatenate(preds)):
                out[i] = float(p)
        if self.clip_negative:
            neg = np.isfinite(out) & (out < 0.0)
            out[neg] = 0.0
        return out

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = "tmp",
    ) -> np.ndarray:
        struc_list = samples[0]
        results = self.predict(struc_list)
        out_path = os.path.abspath(os.path.join(self.root_dir, f"{label}.txt"))
        os.makedirs(self.root_dir, exist_ok=True)
        np.savetxt(out_path, results, fmt="%.6f")
        return results
