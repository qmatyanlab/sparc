import os
import pickle

import ase.io
import torch
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor


def data2atoms(data):
    atom_types = data.atom_types.long().tolist()
    frac_coords = data.frac_coords.tolist()
    cell = data.lengths.tolist()[0] + data.angles.tolist()[0]
    atoms = Atoms(
        cell=cell,
        pbc=True,
        numbers=atom_types,
        scaled_positions=frac_coords,
    )

    return atoms


def save_samples(data_list, save_dir, filename):
    atoms_list = [data2atoms(data) for data in data_list]
    # save the samples in ase.Atoms format
    pkl_path = os.path.join(save_dir, filename)
    with open(pkl_path, "wb") as f:
        pickle.dump(atoms_list, f)

    return pkl_path


def save_structures(structures, save_dir, filename):
    """Save structures to disk in a extxyz file."""
    ase_atoms = [AseAtomsAdaptor.get_atoms(x) for x in structures]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    try:
        ase.io.write(save_path, ase_atoms, format="extxyz")
        return save_path
    except IOError as e:
        print(f"Got error {e} writing the generated structures to disk.")


def _serialize_data(data):
    payload = {
        "frac_coords": data.frac_coords.detach().cpu(),
        "atom_types": data.atom_types.detach().cpu(),
        "lengths": data.lengths.detach().cpu(),
        "angles": data.angles.detach().cpu(),
        "num_atoms": data.num_atoms.detach().cpu(),
        "spacegroup": getattr(data, "spacegroup", None),
    }

    if payload["spacegroup"] is not None:
        if torch.is_tensor(payload["spacegroup"]):
            payload["spacegroup"] = int(payload["spacegroup"].detach().cpu().item())
        else:
            payload["spacegroup"] = int(payload["spacegroup"])

    return payload


def save_data_records(data_list, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    payload = [_serialize_data(data) for data in data_list]
    torch.save(payload, save_path)
    return save_path
