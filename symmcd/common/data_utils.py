from __future__ import annotations

import importlib
import numpy as np
import torch

from models.diffcsp.utils import (
    EPSILON,
    StandardScalerTorch,
    cart_to_frac_coords,
    frac_to_cart_coords,
    get_pbc_distances,
    lattice_params_to_matrix_torch,
    radius_graph_pbc,
    repeat_blocks,
)


N_SPACEGROUPS = 230
LATTICE_MAPPER = {
    "P": 0,
    "I": 1,
    "F": 2,
    "A": 3,
    "B": 4,
    "C": 5,
    "R": 6,
}

B_MATRICES = np.zeros((6, 3, 3))
B_MATRICES[0, 0, 1] = B_MATRICES[0, 1, 0] = 1
B_MATRICES[1, 0, 2] = B_MATRICES[1, 2, 0] = 1
B_MATRICES[2, 1, 2] = B_MATRICES[2, 2, 1] = 1
B_MATRICES[3, 0, 0] = 1
B_MATRICES[3, 1, 1] = -1
B_MATRICES[4, 0, 0] = B_MATRICES[4, 1, 1] = 1
B_MATRICES[4, 2, 2] = -2
B_MATRICES[5, 0, 0] = B_MATRICES[5, 1, 1] = B_MATRICES[5, 2, 2] = 1


def lattice_ks_to_matrix_torch(ks: torch.Tensor) -> torch.Tensor:
    basis = torch.tensor(B_MATRICES, device=ks.device, dtype=ks.dtype)
    symm = torch.einsum("bij,nb->nij", basis, ks)
    return torch.matrix_exp(symm)


def sg_to_ks_mask(sg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_lattices = sg.shape[0]
    ks_mask = torch.ones((n_lattices, 6), device=sg.device)
    ks_add = torch.zeros((n_lattices, 6), device=sg.device)

    monoclinic_mask = (sg >= 3) & (sg <= 15)
    ks_mask[monoclinic_mask, 0] = ks_mask[monoclinic_mask, 2] = 0

    orthorhombic_mask = (sg >= 16) & (sg <= 74)
    ks_mask[orthorhombic_mask, 0:3] = 0

    tetragonal_mask = (sg >= 75) & (sg <= 142)
    ks_mask[tetragonal_mask, 0:4] = 0

    hexagonal_mask = (sg >= 143) & (sg <= 194)
    ks_mask[hexagonal_mask, 0:4] = 0
    ks_add[hexagonal_mask, 0] = -np.log(3) / 4

    cubic_mask = (sg >= 195) & (sg <= 230)
    ks_mask[cubic_mask, 0:5] = 0

    return ks_mask, ks_add


def mask_ks(
    ks: torch.Tensor, ks_mask: torch.Tensor, ks_add: torch.Tensor
) -> torch.Tensor:
    return ks * ks_mask + ks_add


def lengths_angles_to_volume(
    lengths: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    vector_a, vector_b, vector_c = torch.unbind(lattice, dim=1)
    return torch.abs(
        torch.einsum("bi,bi->b", vector_a, torch.cross(vector_b, vector_c, dim=1))
    )


def mard(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((true - pred) / (true.abs() + EPSILON)))


def get_spacegroup_binary_repr(number: int) -> torch.Tensor:
    Group = importlib.import_module("pyxtal.symmetry").Group
    spg = Group(number)
    ss = spg.get_spg_symmetry_object()
    axis_wise_binary_repr = torch.from_numpy(
        ss.to_matrix_representation_spg().reshape(
            -1,
        )
    )
    lattice_type = LATTICE_MAPPER[spg.symbol[0]]
    lattice_type_repr = torch.zeros(7)
    lattice_type_repr[lattice_type] = 1
    return torch.cat([lattice_type_repr, axis_wise_binary_repr], dim=0)
