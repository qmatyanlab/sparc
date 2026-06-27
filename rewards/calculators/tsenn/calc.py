import os
import logging
from typing import Tuple, List

import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase import Atoms
from ase.neighborlist import neighbor_list
from mendeleev import element
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from e3nn import o3
from e3nn.io import CartesianTensor
import torch_scatter

from rewards.calculators.base import Calculator
from rewards.calculators.tsenn.model import Network


CRYSTAL_SYSTEMS = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]


def process_atom(atomic_number: int, default_dipole: float = 67.0):
    specie = element(atomic_number)
    dipole = specie.dipole_polarizability
    if dipole is None:
        dipole = default_dipole
    radius = specie.covalent_radius_pyykko
    return atomic_number - 1, specie.atomic_weight, dipole, radius


def build_onehot(dtype: torch.dtype = torch.float64):
    type_encoding = {}
    specie_mass = []
    specie_dipole = []
    specie_radius = []

    for atomic_number in range(1, 119):
        encoding, mass, dipole, radius = process_atom(atomic_number)
        symbol = element(atomic_number).symbol
        type_encoding[symbol] = encoding
        specie_mass.append(mass)
        specie_dipole.append(dipole)
        specie_radius.append(radius)

    type_onehot = torch.eye(len(type_encoding), dtype=dtype)
    mass_onehot = torch.diag(torch.tensor(specie_mass, dtype=dtype))
    dipole_onehot = torch.diag(torch.tensor(specie_dipole, dtype=dtype))
    radius_onehot = torch.diag(torch.tensor(specie_radius, dtype=dtype))

    return type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding


def _load_onehot_bundle(cache_dir: str, dtype: torch.dtype = torch.float64):
    bundle_path = os.path.join(cache_dir, "onehot_fp64.pt")
    if os.path.exists(bundle_path):
        payload = torch.load(bundle_path, map_location="cpu")
        return (
            payload["type_onehot"].to(dtype=dtype),
            payload["mass_onehot"].to(dtype=dtype),
            payload["dipole_onehot"].to(dtype=dtype),
            payload["radius_onehot"].to(dtype=dtype),
            payload["type_encoding"],
            bundle_path,
        )

    component_paths = {
        "type_onehot": os.path.join(cache_dir, "type_onehot.torch"),
        "mass_onehot": os.path.join(cache_dir, "mass_onehot.torch"),
        "dipole_onehot": os.path.join(cache_dir, "dipole_onehot.torch"),
        "radius_onehot": os.path.join(cache_dir, "radius_onehot.torch"),
        "type_encoding": os.path.join(cache_dir, "type_encoding.torch"),
    }
    if not all(os.path.exists(path) for path in component_paths.values()):
        return None

    return (
        torch.load(component_paths["type_onehot"], map_location="cpu").to(dtype=dtype),
        torch.load(component_paths["mass_onehot"], map_location="cpu").to(dtype=dtype),
        torch.load(component_paths["dipole_onehot"], map_location="cpu").to(
            dtype=dtype
        ),
        torch.load(component_paths["radius_onehot"], map_location="cpu").to(
            dtype=dtype
        ),
        torch.load(component_paths["type_encoding"], map_location="cpu"),
        cache_dir,
    )


def _save_onehot_bundle(
    cache_dir: str,
    type_onehot: torch.Tensor,
    mass_onehot: torch.Tensor,
    dipole_onehot: torch.Tensor,
    radius_onehot: torch.Tensor,
    type_encoding: dict,
):
    os.makedirs(cache_dir, exist_ok=True)
    bundle_path = os.path.join(cache_dir, "onehot_fp64.pt")
    torch.save(
        {
            "type_onehot": type_onehot.cpu(),
            "mass_onehot": mass_onehot.cpu(),
            "dipole_onehot": dipole_onehot.cpu(),
            "radius_onehot": radius_onehot.cpu(),
            "type_encoding": type_encoding,
        },
        bundle_path,
    )
    return bundle_path


def load_or_build_onehot(root_dir: str, dtype: torch.dtype = torch.float64):
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    shared_cache_dir = os.path.join(project_root, "rewards", "tsenn_cache")
    candidate_dirs = [
        shared_cache_dir,
        root_dir,
        os.path.join(project_root, "onehot_data"),
    ]

    seen = set()
    for cache_dir in candidate_dirs:
        if cache_dir in seen:
            continue
        seen.add(cache_dir)
        loaded = _load_onehot_bundle(cache_dir, dtype=dtype)
        if loaded is None:
            continue
        (
            type_onehot,
            mass_onehot,
            dipole_onehot,
            radius_onehot,
            type_encoding,
            source,
        ) = loaded
        if cache_dir != shared_cache_dir:
            _save_onehot_bundle(
                shared_cache_dir,
                type_onehot,
                mass_onehot,
                dipole_onehot,
                radius_onehot,
                type_encoding,
            )
        print(f"Loading TSENN onehot data from {source}")
        return type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding

    onehot = build_onehot(dtype=dtype)
    bundle_path = _save_onehot_bundle(shared_cache_dir, *onehot)
    print(f"Built TSENN onehot data and saved cache to {bundle_path}")
    return onehot


def crystal_system_onehot(structure: Structure, dtype: torch.dtype = torch.float64):
    try:
        crystal_system = (
            SpacegroupAnalyzer(structure, symprec=0.1).get_crystal_system().lower()
        )
    except Exception:
        crystal_system = None

    onehot = torch.zeros(len(CRYSTAL_SYSTEMS), dtype=dtype)
    if crystal_system in CRYSTAL_SYSTEMS:
        onehot[CRYSTAL_SYSTEMS.index(crystal_system)] = 1.0
    return onehot


def build_graph(
    structure: Structure,
    type_onehot: torch.Tensor,
    mass_onehot: torch.Tensor,
    dipole_onehot: torch.Tensor,
    radius_onehot: torch.Tensor,
    type_encoding: dict,
    r_max: float,
    dtype: torch.dtype = torch.float64,
):
    symbols = [site.specie.symbol for site in structure.sites]
    positions = torch.tensor(structure.cart_coords, dtype=dtype)
    lattice = torch.tensor(structure.lattice.matrix, dtype=dtype).unsqueeze(0)

    atoms = Atoms(
        symbols=symbols,
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )
    edge_src, edge_dst, edge_shift = neighbor_list(
        "ijS", a=atoms, cutoff=r_max, self_interaction=True
    )

    edge_src_t = torch.tensor(edge_src, dtype=torch.long)
    edge_dst_t = torch.tensor(edge_dst, dtype=torch.long)
    edge_shift_t = torch.tensor(edge_shift, dtype=dtype)
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[edge_src_t]
    edge_vec = (
        positions[edge_dst_t]
        - positions[edge_src_t]
        + torch.einsum("ni,nij->nj", edge_shift_t, lattice[edge_batch])
    )
    edge_len = edge_vec.norm(dim=1)

    indices = [type_encoding[symbol] for symbol in symbols]
    idx = torch.tensor(indices, dtype=torch.long)

    return Data(
        pos=positions,
        lattice=lattice,
        symbol=symbols,
        x_mass=mass_onehot[idx],
        x_dipole=dipole_onehot[idx],
        x_radius=radius_onehot[idx],
        z=type_onehot[idx],
        x=type_onehot[idx],
        edge_index=torch.stack([edge_src_t, edge_dst_t], dim=0),
        edge_shift=edge_shift_t,
        edge_vec=edge_vec,
        edge_len=edge_len,
        crystal_system_onehot=crystal_system_onehot(structure, dtype=dtype).unsqueeze(
            0
        ),
    )


class NetWrapper(Network):
    def __init__(self, in_dim, em_dim, **kwargs):
        self.pool = False
        if kwargs.get("reduce_output", False):
            kwargs["reduce_output"] = False
            self.pool = True
        super().__init__(**kwargs)
        self.em_z = torch.nn.Linear(in_dim, em_dim)
        self.em_x = torch.nn.Linear(in_dim, em_dim)

    def forward(self, data: Data) -> torch.Tensor:
        data.z = torch.nn.functional.relu(self.em_z(data.z))
        data.x = torch.nn.functional.relu(self.em_x(data.x))
        output = super().forward(data)
        if self.pool:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)
        return output


class TSENN(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = "tsenn_dielectric",
        model_path: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
        r_max: float = 6.0,
        out_dim: int = 300,
        em_dim: int = 128,
        lmax: int = 2,
        layers: int = 4,
        mul: int = 32,
        num_neighbors: float = 12.0,
        scale_0e: float = 1.0,
        scale_2e: float = 1.0,
        dropout_prob: float = 0.0,
        use_batch_norm: bool = False,
        output_mode: str = "tensor",
        max_edges: int | None = None,
        max_edges_per_atom: float | None = 256.0,
    ) -> None:
        super().__init__(root_dir, task)
        self.batch_size = batch_size
        self.r_max = r_max
        self.out_dim = out_dim
        self.scale_0e = scale_0e
        self.scale_2e = scale_2e
        self.dtype = torch.float64
        self.output_mode = str(output_mode).lower()
        self.max_edges = None if max_edges is None else int(max_edges)
        self.max_edges_per_atom = (
            None if max_edges_per_atom is None else float(max_edges_per_atom)
        )
        if self.output_mode not in {"tensor", "trace"}:
            raise ValueError("TSENN output_mode must be 'tensor' or 'trace'.")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_path is None:
            raise ValueError("TSENN model_path must be provided.")
        from utils.assets import resolve_path

        self.model_path = str(resolve_path(model_path))
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TSENN model not found: {self.model_path}")

        (
            self.type_onehot,
            self.mass_onehot,
            self.dipole_onehot,
            self.radius_onehot,
            self.type_encoding,
        ) = load_or_build_onehot(self.root_dir, dtype=self.dtype)

        irreps_out = f"{out_dim}x0e + {out_dim}x2e"
        if self.output_mode == "trace":
            irreps_out = f"{out_dim}x0e"

        self.model = NetWrapper(
            in_dim=118,
            em_dim=em_dim,
            irreps_in=f"{em_dim}x0e",
            irreps_out=irreps_out,
            irreps_node_attr=f"{em_dim}x0e",
            layers=layers,
            mul=mul,
            lmax=lmax,
            max_radius=r_max,
            num_neighbors=num_neighbors,
            reduce_output=True,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm,
        ).to(self.device, dtype=self.dtype)

        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state["state"])
        self.model.eval()
        self.cartesian = (
            CartesianTensor("ij=ji") if self.output_mode == "tensor" else None
        )

    def _graph_skip_reason(self, data: Data) -> str | None:
        num_edges = int(data.edge_index.shape[1])
        num_nodes = int(data.x.shape[0])
        if self.max_edges is not None and num_edges > self.max_edges:
            return f"edges={num_edges} > max_edges={self.max_edges}"
        if self.max_edges_per_atom is not None and num_nodes > 0:
            edges_per_atom = num_edges / num_nodes
            if edges_per_atom > self.max_edges_per_atom:
                return (
                    f"edges_per_atom={edges_per_atom:.1f} "
                    f"> max_edges_per_atom={self.max_edges_per_atom:g} "
                    f"(edges={num_edges}, atoms={num_nodes})"
                )
        return None

    @staticmethod
    def _log_skipped_graphs(
        skipped_graphs: list[tuple[int, str]],
        total_structures: int,
    ) -> None:
        if not skipped_graphs:
            return
        preview = "; ".join(
            f"idx={idx}: {reason}" for idx, reason in skipped_graphs[:3]
        )
        if len(skipped_graphs) > 3:
            preview += f"; ... {len(skipped_graphs) - 3} more"
        logging.warning(
            "TSENN skipped %d/%d structures with oversized radius graphs: %s",
            len(skipped_graphs),
            total_structures,
            preview,
        )

    def predict_epsilon2_iso(
        self,
        struc_list: List[Structure],
        energy_min: float = 0.0,
        energy_max: float = 30.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict isotropic epsilon_2 spectrum for each structure.

        Returns (energies, eps2_iso, valid_mask):
        - energies: (F,)
        - eps2_iso: (N, F) with NaN rows for invalid structures
        - valid_mask: (N,) bool
        """
        energies = np.linspace(float(energy_min), float(energy_max), int(self.out_dim))
        eps2_iso = np.full((len(struc_list), self.out_dim), np.nan, dtype=float)
        valid_mask = np.zeros(len(struc_list), dtype=bool)

        data_list = []
        valid_indices = []
        skipped_graphs = []
        for i, structure in enumerate(struc_list):
            try:
                data = build_graph(
                    structure,
                    self.type_onehot,
                    self.mass_onehot,
                    self.dipole_onehot,
                    self.radius_onehot,
                    self.type_encoding,
                    self.r_max,
                    dtype=self.dtype,
                )
            except Exception:
                continue
            skip_reason = self._graph_skip_reason(data)
            if skip_reason is not None:
                skipped_graphs.append((i, skip_reason))
                continue
            data_list.append(data)
            valid_indices.append(i)
        self._log_skipped_graphs(skipped_graphs, len(struc_list))

        if not data_list:
            return energies, eps2_iso, valid_mask

        loader = DataLoader(data_list, batch_size=self.batch_size)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                if self.output_mode == "trace":
                    preds.append((output * self.scale_0e).detach().cpu().numpy())
                    continue
                irreps_0e = self.model.irreps_out.count(o3.Irrep("0e"))
                irreps_2e = self.model.irreps_out.count(o3.Irrep("2e")) * 5

                output_0e = output[:, :irreps_0e] * self.scale_0e
                output_2e = (
                    output[:, irreps_0e : irreps_0e + irreps_2e]
                    .contiguous()
                    .view(-1, self.out_dim, 5)
                    * self.scale_2e
                )
                sph = torch.cat([output_0e.unsqueeze(2), output_2e], dim=2)
                cart = self.cartesian.to_cartesian(sph)  # (B, F, 3, 3)
                trace = cart.diagonal(dim1=2, dim2=3).sum(dim=2) / 3.0
                preds.append(trace.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)  # (valid_N, F)
        for row_idx, value in zip(valid_indices, preds):
            eps2_iso[row_idx] = value
            valid_mask[row_idx] = True

        return energies, eps2_iso, valid_mask

    def predict_epsilon2_tensor(
        self,
        struc_list: List[Structure],
        energy_min: float = 0.0,
        energy_max: float = 30.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict full symmetric epsilon_2 tensor spectrum for each structure.

        Returns (energies, eps2, valid_mask):
        - energies: (F,)
        - eps2: (N, F, 3, 3) with NaNs for invalid structures
        - valid_mask: (N,) bool
        """
        if self.output_mode != "tensor":
            raise ValueError(
                "TSENN output_mode='trace' does not support full tensor prediction."
            )

        energies = np.linspace(float(energy_min), float(energy_max), int(self.out_dim))
        eps2 = np.full((len(struc_list), self.out_dim, 3, 3), np.nan, dtype=float)
        valid_mask = np.zeros(len(struc_list), dtype=bool)

        data_list = []
        valid_indices = []
        skipped_graphs = []
        for i, structure in enumerate(struc_list):
            try:
                data = build_graph(
                    structure,
                    self.type_onehot,
                    self.mass_onehot,
                    self.dipole_onehot,
                    self.radius_onehot,
                    self.type_encoding,
                    self.r_max,
                    dtype=self.dtype,
                )
            except Exception:
                continue
            skip_reason = self._graph_skip_reason(data)
            if skip_reason is not None:
                skipped_graphs.append((i, skip_reason))
                continue
            data_list.append(data)
            valid_indices.append(i)
        self._log_skipped_graphs(skipped_graphs, len(struc_list))

        if not data_list:
            return energies, eps2, valid_mask

        loader = DataLoader(data_list, batch_size=self.batch_size)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                irreps_0e = self.model.irreps_out.count(o3.Irrep("0e"))
                irreps_2e = self.model.irreps_out.count(o3.Irrep("2e")) * 5

                output_0e = output[:, :irreps_0e] * self.scale_0e
                output_2e = (
                    output[:, irreps_0e : irreps_0e + irreps_2e]
                    .contiguous()
                    .view(-1, self.out_dim, 5)
                    * self.scale_2e
                )
                sph = torch.cat([output_0e.unsqueeze(2), output_2e], dim=2)
                cart = self.cartesian.to_cartesian(sph)  # (B, F, 3, 3)
                preds.append(cart.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)  # (valid_N, F, 3, 3)
        for row_idx, value in zip(valid_indices, preds):
            eps2[row_idx] = value
            valid_mask[row_idx] = True

        return energies, eps2, valid_mask

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = "tmp",
    ) -> np.ndarray[float]:
        struc_list = samples[0]
        out_path = os.path.join(self.root_dir, f"{label}.txt")
        out_path = os.path.abspath(out_path)

        energies, eps2_iso, valid_mask = self.predict_epsilon2_iso(struc_list)
        results = np.full(len(struc_list), np.nan, dtype=float)
        results[valid_mask] = eps2_iso[valid_mask].mean(axis=1)

        np.savetxt(out_path, results, fmt="%.6f")
        return results
