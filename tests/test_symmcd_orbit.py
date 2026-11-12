"""t=0 Wyckoff snap / orbit expansion of the SymmCD sampler."""

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from models.symmcd.pl_modules.discrete_diffusion_w_site_symm import (
    SITE_SYMM_AXES,
    SITE_SYMM_PGS,
    _merge_orbit_images,
    _wp_to_site_symm,
    modify_frac_coords,
    modify_frac_coords_one,
)

N_TYPES = 94


def _label(spacegroup, multiplicity):
    """Site-symmetry one-hot of the Wyckoff position with this multiplicity."""
    for wp, one_hot in _wp_to_site_symm(spacegroup).items():
        if wp.multiplicity == multiplicity:
            return torch.as_tensor(one_hot, dtype=torch.float32).reshape(1, -1)
    raise KeyError(multiplicity)


def _one_hot_type(z):
    t = torch.zeros(1, N_TYPES)
    t[0, z - 1] = 1.0
    return t


def _expand(spacegroup, coord, multiplicity, lattice, **kwargs):
    return modify_frac_coords_one(
        torch.tensor([coord], dtype=torch.float32),
        _label(spacegroup, multiplicity),
        _one_hot_type(11),
        spacegroup,
        lattice=lattice,
        **kwargs,
    )


def test_general_label_near_1a_collapses_to_one_atom_and_keeps_pm3m():
    lattice = np.eye(3) * 4.0
    coord = [0.004, -0.003, 0.002]  # converged next to 1a (0, 0, 0)
    frac, n, *_ = _expand(221, coord, 48, lattice, merge_tol=0.5)
    assert n == 1  # 48 coincident images merged into one atom
    # The centroid of the stabiliser orbit is the fixed point itself.
    assert np.allclose(((frac[0] + 0.5) % 1.0) - 0.5, 0.0, atol=1e-9)
    structure = Structure(Lattice(lattice), ["Na"], frac)
    assert SpacegroupAnalyzer(structure, symprec=0.01).get_space_group_number() == 221


def test_legacy_mode_reproduces_stacked_images():
    coord = [0.004, -0.003, 0.002]
    frac, n, *_ = _expand(221, coord, 48, lattice=None, merge_tol=0.0)
    assert n == 48
    d = frac - frac[0]
    d -= np.rint(d)
    assert np.abs(d * 4.0).max() < 0.1  # 48 copies within 0.1 A of each other


def test_mirror_site_general_label_lands_on_mirror():
    lattice = np.diag([4.0, 5.0, 6.0])
    coord = [0.3, 0.003, 0.7]  # 0.015 A off the y=0 mirror of Pm
    frac, n, *_ = _expand(6, coord, 2, lattice, merge_tol=0.5)
    assert n == 1
    assert abs(((frac[0][1] + 0.5) % 1.0) - 0.5) < 1e-12
    assert np.allclose(frac[0][[0, 2]], [0.3, 0.7], atol=1e-6)


def test_separated_images_are_not_merged():
    lattice = np.eye(3) * 5.0
    frac, n, *_ = _expand(2, [0.2, 0.3, 0.4], 2, lattice, merge_tol=0.5)
    assert n == 2


def test_merge_respects_periodic_boundary():
    lattice = np.eye(3) * 5.0
    merged, n_merged = _merge_orbit_images(
        [[0.999, 0.5, 0.5], [0.001, 0.5, 0.5]], lattice, tol=0.5
    )
    assert n_merged == 1
    assert np.allclose(merged[0], [0.0, 0.5, 0.5], atol=1e-9)


def test_cartesian_snap_matches_fractional_on_cubic_cell():
    lattice = np.eye(3) * 4.0
    coord = [0.27, 0.27, 0.01]
    a = _expand(221, coord, 24, lattice, merge_tol=0.0, cartesian_snap=True)
    b = _expand(221, coord, 24, lattice, merge_tol=0.0, cartesian_snap=False)
    assert a[1] == b[1]
    assert np.allclose(np.sort(a[0], axis=0), np.sort(b[0], axis=0))


def _traj(spacegroups, coords_per_crystal):
    frac, ss, types, num = [], [], [], []
    for sg, coords in zip(spacegroups.tolist(), coords_per_crystal):
        num.append(len(coords))
        for c in coords:
            frac.append(c)
            ss.append(_label(sg, max(w.multiplicity for w in _wp_to_site_symm(sg))))
            types.append(_one_hot_type(8))
    traj = {
        "frac_coords": torch.tensor(frac, dtype=torch.float32).reshape(-1, 3),
        "site_symm": torch.cat(ss).reshape(-1, SITE_SYMM_AXES, SITE_SYMM_PGS)
        if ss
        else torch.zeros(0, SITE_SYMM_AXES, SITE_SYMM_PGS),
        "atom_types": torch.cat(types) if types else torch.zeros(0, N_TYPES),
    }
    return traj, torch.tensor(num)


def test_empty_crystal_keeps_spacegroups_aligned():
    spacegroups = torch.tensor([2, 221, 6])
    traj, num = _traj(spacegroups, [[[0.2, 0.3, 0.4]], [], [[0.3, 0.25, 0.7]]])
    lattices = torch.eye(3).repeat(3, 1, 1) * 5.0
    out = modify_frac_coords(
        traj, spacegroups, num, lattices=lattices, merge_tol=0.5, strict=True
    )
    assert out["kept_index"].tolist() == [0, 2]
    assert spacegroups[out["kept_index"]].tolist() == [2, 6]
    assert out["num_atoms"].tolist() == [2, 2]
    assert len(out["representative"]["num_atoms"]) == 2


def test_strict_snap_drops_failing_crystal(monkeypatch):
    import models.symmcd.pl_modules.discrete_diffusion_w_site_symm as dd

    def boom(*args, **kwargs):
        raise RuntimeError("pyxtal failure")

    monkeypatch.setattr(dd, "_search_closest_wp", boom)
    lattice = np.eye(3) * 5.0
    _, n, *_ = _expand(2, [0.2, 0.3, 0.4], 2, lattice, strict=True)
    assert n == 0
    # Legacy: the unexpanded representative is kept.
    _, n_legacy, *_ = _expand(2, [0.2, 0.3, 0.4], 2, lattice, strict=False)
    assert n_legacy == 1


def test_nonfinite_lattice_falls_back_to_fractional_snap():
    lattice = np.full((3, 3), np.nan)
    frac, n, *_ = _expand(2, [0.2, 0.3, 0.4], 2, lattice, merge_tol=0.5)
    assert n == 2  # no crash; falls back to the fractional snap
