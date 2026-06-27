"""SymmCD model suite for SPARC.

Sampling runs in-process: the loaded CSPDiffusion model is called directly
instead of via a subprocess, enabling gradient-based finetuning.

The finetuning API (add_noise / calc_sample_loss / calc_kl_reg) is implemented
on CSPDiffusion in symmcd/pl_modules/discrete_diffusion_w_site_symm.py.
"""

import os
import shutil
from collections import defaultdict
from typing import Any, cast
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray
from omegaconf import DictConfig
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from models.diffcsp.finetune import DiffCSPDataset
from models.suite.base import ModelSuite


AVA_MODEL_NAME = Literal["symmcd"]
EPS = 1e-4 * np.random.randn(3)
POINT = np.array([0.5, 0.5, 0.5]) + EPS

_SYMMCD_MODEL_TARGETS = {
    "symmcd.pl_modules.diffusion.CSPDiffusion": "symmcd.pl_modules.diffusion",
    "symmcd.pl_modules.diffusion_w_type.CSPDiffusion": "symmcd.pl_modules.diffusion_w_type",
    "symmcd.pl_modules.diffusion_w_site_symm.CSPDiffusion": "symmcd.pl_modules.diffusion_w_site_symm",
    "symmcd.pl_modules.discrete_diffusion_w_site_symm.CSPDiffusion": "symmcd.pl_modules.discrete_diffusion_w_site_symm",
    "symmcd.pl_modules.model.CrystGNN_Supervise": "symmcd.pl_modules.model",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lattices_to_params_shape(lattices: torch.Tensor):
    """Convert 3×3 lattice matrices to lengths and angles (degrees)."""
    lengths = torch.sqrt(torch.sum(lattices**2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = torch.clamp(
            torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1)
            / (lengths[..., j] * lengths[..., k]),
            -1.0,
            1.0,
        )
    angles = torch.arccos(angles) * 180.0 / np.pi
    return lengths, angles


def _load_symmcd_model(model_path: str):
    from utils.assets import resolve_path

    path = Path(model_path).resolve()
    # Materialize the pretrained checkpoint bundle from HuggingFace if absent.
    for _rel in (
        "hparams.yaml",
        "epoch=0-step=0.ckpt",
        "lattice_scaler.pt",
        "prop_scaler.pt",
    ):
        resolve_path(path / _rel)
    GlobalHydra.instance().clear()
    with initialize_config_dir(str(path), version_base="1.1"):
        cfg = compose(config_name="hparams")

    local_data_root = path.parents[2] / "data" / path.name
    if local_data_root.exists():
        cfg.data.root_path = str(local_data_root)

    target = cfg.model._target_
    if target not in _SYMMCD_MODEL_TARGETS:
        raise NotImplementedError(f"Unknown SymmCD model target: {target}")

    module_path = _SYMMCD_MODEL_TARGETS[target]
    class_name = target.split(".")[-1]
    module = __import__(module_path, fromlist=[class_name])
    Model = getattr(module, class_name)

    ckpts = list(path.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {path}")
    non_last = [c for c in ckpts if "last" not in c.parts[-1]]
    if non_last:
        epochs = [int(c.parts[-1].split("-")[0].split("=")[1]) for c in non_last]
        ckpt = str(non_last[sorted(range(len(epochs)), key=lambda i: epochs[i])[-1]])
    else:
        ckpt = str(ckpts[0])

    model = Model.load_from_checkpoint(ckpt, strict=False)
    setattr(model, "_sparc_hparams_cfg", cfg)
    model.lattice_scaler = torch.load(path / "lattice_scaler.pt")
    model.scaler = torch.load(path / "prop_scaler.pt")
    return model, cfg


def _normalize_atom_types(atom_types: torch.Tensor) -> torch.Tensor:
    """Convert SymmCD sampled atom types to atomic numbers."""
    if atom_types.ndim == 2:
        return atom_types.argmax(dim=-1).to(torch.long) + 1
    return atom_types.to(torch.long)


def _resolve_restrict_spacegroups(
    runtime_value: Optional[List[int] | NDArray | Tuple[int, ...]],
    fallback_value: Optional[List[int]],
) -> Optional[NDArray]:
    candidate = runtime_value if runtime_value is not None else fallback_value
    if candidate is None:
        return None
    candidate_array = np.asarray(candidate, dtype=int).reshape(-1)
    if candidate_array.size == 0:
        return None
    return candidate_array


def _normalize_sg_distribution(
    distribution: NDArray,
    support_mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray:
    dist = np.asarray(distribution, dtype=float).reshape(-1).copy()
    if dist.shape != (230,):
        raise ValueError(f"Expected SG distribution of shape (230,), got {dist.shape}")
    if not np.isfinite(dist).all():
        raise ValueError("SG distribution must contain only finite values")
    if (dist < 0).any():
        raise ValueError("SG distribution cannot contain negative values")
    if support_mask is not None:
        dist[~support_mask] = 0.0
    total = float(dist.sum())
    if total <= 0.0:
        raise ValueError("SG distribution must have positive mass after masking")
    return dist / total


class _MatInventSampleDataset(Dataset):
    """MatInvent-owned SymmCD sampling dataset."""

    def __init__(
        self,
        dataset: str,
        total_num: int,
        train_ori_path: Optional[str] = None,
        sg_info_path: Optional[str] = None,
        restrict_spacegroups: Optional[NDArray] = None,
    ) -> None:
        super().__init__()
        self.total_num = total_num
        self.is_carbon = dataset == "carbon"
        self.sg_num_atoms, self.sg_dist, self.sg_number_binary_mapper = (
            self.get_sg_statistics(train_ori_path, sg_info_path)
        )

        if restrict_spacegroups is not None:
            print("Sampling ONLY from spacegroups " + str(restrict_spacegroups))
            new_sg_dist = np.zeros_like(self.sg_dist)
            new_sg_dist[restrict_spacegroups - 1] = self.sg_dist[
                restrict_spacegroups - 1
            ]
            if new_sg_dist.sum() <= 0:
                raise ValueError(
                    "Requested restrict_spacegroups are absent from the training "
                    f"distribution: {restrict_spacegroups.tolist()}"
                )
            self.sg_dist = new_sg_dist / new_sg_dist.sum()

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        del index
        spacegroup = int(np.random.choice(230, p=self.sg_dist)) + 1
        num_atom = int(
            np.random.choice(
                list(self.sg_num_atoms[spacegroup].keys()),
                p=list(self.sg_num_atoms[spacegroup].values()),
            )
        )
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
            spacegroup=spacegroup,
            sg_condition=self.sg_number_binary_mapper[spacegroup],
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data

    @staticmethod
    def get_sg_statistics(
        train_path: Optional[str] = None,
        sg_info_path: Optional[str] = None,
    ):
        from symmcd.common.data_utils import get_spacegroup_binary_repr
        from utils.assets import resolve_path

        if sg_info_path:
            sg_info_path = str(resolve_path(sg_info_path))
        if sg_info_path and os.path.exists(sg_info_path):
            print(f"Loading spacegroup statistics from {sg_info_path}")
            return torch.load(sg_info_path)
        if train_path is None:
            raise ValueError(
                "train_ori_path is required when sg_info_path is unavailable"
            )

        dataset = torch.load(str(resolve_path(train_path)))
        dataset_len = len(dataset)
        sg_counter = defaultdict(lambda: 0)
        sg_num_atoms = defaultdict(lambda: defaultdict(lambda: 0))
        sg_number_binary_mapper = {}

        for i in range(dataset_len):
            data_dict = dataset[i]
            (
                frac_coords,
                atom_types,
                lengths,
                angles,
                ks,
                edge_indices,
                to_jimages,
                num_atoms,
            ) = data_dict["graph_arrays"]
            del atom_types, lengths, angles, ks, edge_indices, to_jimages, num_atoms
            spacegroup = data_dict["spacegroup"]
            identifiers = data_dict["identifier"]

            mask = np.zeros_like(identifiers)
            for identifier in np.unique(identifiers):
                indices = np.where(identifiers == identifier)[0]
                min_index = ((frac_coords - POINT) ** 2).sum(1)[indices].argmin().item()
                mask[indices[min_index]] = 1

            frac_coords = frac_coords[mask.astype(bool)]
            num_atoms_reduced = len(frac_coords)

            sg_counter[spacegroup] += 1
            sg_number_binary_mapper[spacegroup] = get_spacegroup_binary_repr(spacegroup)
            sg_num_atoms[spacegroup][num_atoms_reduced] += 1

        sg_dist = np.array([sg_counter[i] for i in range(1, 231)], dtype=float)
        sg_dist = sg_dist / dataset_len

        for sg in sg_num_atoms:
            total = sum(sg_num_atoms[sg].values())
            sg_counts = cast(Any, sg_num_atoms[sg])
            for num_atoms in sg_num_atoms[sg]:
                sg_counts[num_atoms] = float(sg_counts[num_atoms]) / total

        if sg_info_path:
            sg_num_atoms_hashable = {
                k: {kk: vv for kk, vv in v.items()} for k, v in sg_num_atoms.items()
            }
            torch.save(
                (sg_num_atoms_hashable, sg_dist, sg_number_binary_mapper), sg_info_path
            )
        return sg_num_atoms, sg_dist, sg_number_binary_mapper


def _is_valid_structure_inputs(
    frac_coords: torch.Tensor,
    atom_types: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
) -> bool:
    if len(atom_types) == 0:
        return False
    if len(atom_types) != len(frac_coords):
        return False
    if len(atom_types) > 30:
        return False
    if not torch.isfinite(frac_coords).all():
        return False
    if not torch.isfinite(lengths).all():
        return False
    if not torch.isfinite(angles).all():
        return False
    if (atom_types < 1).any() or (atom_types > 94).any():
        return False
    if (lengths <= 0).any() or (lengths < 1).any() or (lengths > 1000).any():
        return False
    if (angles <= 0).any() or (angles >= 180).any():
        return False
    if (frac_coords < 0).any() or (frac_coords > 1).any():
        return False
    return True


def _split_generation_output(
    outputs: dict,
    input_batch,
) -> Tuple[List[Data], List[Structure]]:
    """Split a batched SymmCD sample() output into individual Data + Structure."""
    num_atoms = outputs["num_atoms"]
    frac_coords = outputs["frac_coords"]
    atom_types = outputs["atom_types"]
    lattices = outputs["lattices"]
    ks = outputs.get("ks")
    spacegroups = outputs["spacegroup"]
    site_symm = outputs.get("site_symm")

    lengths, angles = _lattices_to_params_shape(lattices.detach().cpu())

    B = len(num_atoms)
    sg_condition = getattr(input_batch, "sg_condition", None)
    if sg_condition is not None:
        try:
            sg_condition = sg_condition.detach().cpu().reshape(B, -1)
        except Exception:
            sg_condition = None

    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(num_atoms.cpu(), dim=0)]
    )

    data_list: List[Data] = []
    structure_list: List[Structure] = []

    for i in range(B):
        n = int(num_atoms[i].item())
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())

        fc = frac_coords[start:end].detach().cpu().float()
        at = _normalize_atom_types(atom_types[start:end].detach().cpu())
        lengths_t = lengths[i].detach().cpu().unsqueeze(0).float()
        angles_t = angles[i].detach().cpu().unsqueeze(0).float()

        structure = None
        if _is_valid_structure_inputs(fc, at, lengths_t[0], angles_t[0]):
            try:
                pmg_lat = Lattice.from_parameters(
                    *(lengths_t[0].tolist() + angles_t[0].tolist())
                )
                structure = Structure(
                    lattice=pmg_lat,
                    species=at.tolist(),
                    coords=fc.tolist(),
                    coords_are_cartesian=False,
                )
                if (
                    not np.isfinite(structure.lattice.matrix).all()
                    or not np.isfinite(structure.frac_coords).all()
                    or structure.volume < 0.1
                ):
                    structure = None
            except Exception:
                structure = None

        kwargs = dict(
            frac_coords=fc,
            atom_types=at,
            lengths=lengths_t,
            angles=angles_t,
            num_atoms=torch.LongTensor([n]),
            num_nodes=n,
            spacegroup=int(spacegroups[i].item()),
        )
        if ks is not None:
            kwargs["ks"] = ks[i].detach().cpu().unsqueeze(0)
        if sg_condition is not None:
            kwargs["sg_condition"] = sg_condition[i]
        if site_symm is not None:
            kwargs["site_symm"] = site_symm[start:end].detach().cpu()

        data_list.append(Data(**cast(dict[str, Any], kwargs)))
        structure_list.append(structure)

    return data_list, structure_list


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class SymmCDSampler:
    """In-process SymmCD sampler."""

    def __init__(
        self,
        model_path: str,
        dataset: str = "mp",
        step_lr: float = 1e-5,
        restrict_spacegroups: Optional[List[int]] = None,
        generation_batch_size: int = 64,
        sg_temperature: float = 1.0,
    ) -> None:
        self.model_path = model_path
        self.dataset = dataset
        self.step_lr = step_lr
        self.restrict_spacegroups = restrict_spacegroups
        self.generation_batch_size = generation_batch_size
        self.sg_temperature = sg_temperature

    def _resolve_dataset_paths(self, model) -> tuple[str, str]:
        cfg = getattr(model, "_sparc_hparams_cfg", None)
        if cfg is not None:
            return (
                cfg.data.datamodule.datasets.train.save_path,
                cfg.data.datamodule.datasets.train.sg_info_path,
            )
        return (
            model.hparams.data.datamodule.datasets.train.save_path,
            model.hparams.data.datamodule.datasets.train.sg_info_path,
        )

    def _build_sample_dataset(
        self,
        model,
        total_num: int,
        restrict_spacegroups: Optional[NDArray],
    ) -> _MatInventSampleDataset:
        train_path, sg_info_path = self._resolve_dataset_paths(model)
        sample_dataset = _MatInventSampleDataset(
            dataset=self.dataset,
            total_num=total_num,
            train_ori_path=train_path,
            sg_info_path=sg_info_path,
            restrict_spacegroups=restrict_spacegroups,
        )
        if self.sg_temperature != 1.0:
            dist = sample_dataset.sg_dist.copy()
            nonzero = dist > 0
            dist[nonzero] = dist[nonzero] ** (1.0 / self.sg_temperature)
            sample_dataset.sg_dist = dist / dist.sum()
        return sample_dataset

    def get_sampling_sg_distribution(
        self,
        model,
        restrict_spacegroups: Optional[List[int] | NDArray | Tuple[int, ...]] = None,
    ) -> NDArray:
        restrict = _resolve_restrict_spacegroups(
            restrict_spacegroups, self.restrict_spacegroups
        )
        sample_dataset = self._build_sample_dataset(
            model=model,
            total_num=1,
            restrict_spacegroups=restrict,
        )
        return sample_dataset.sg_dist.copy()

    def generate(
        self,
        model,
        batch_size: Optional[int] = None,
        num_batches: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[Data], List[Structure]]:
        if batch_size is None or num_batches is None:
            raise ValueError("batch_size and num_batches are required")

        from torch_geometric.loader import DataLoader as PyGLoader

        n_total = int(batch_size) * int(num_batches)
        gen_bs = min(self.generation_batch_size, n_total)

        restrict = _resolve_restrict_spacegroups(
            kwargs.get("restrict_spacegroups"), self.restrict_spacegroups
        )
        sample_dataset = self._build_sample_dataset(
            model=model,
            total_num=n_total,
            restrict_spacegroups=restrict,
        )
        sg_distribution = kwargs.get("sg_distribution")
        if sg_distribution is not None:
            support_mask = sample_dataset.sg_dist > 0
            sample_dataset.sg_dist = _normalize_sg_distribution(
                np.asarray(sg_distribution, dtype=float),
                support_mask=support_mask,
            )

        loader = PyGLoader(cast(Any, sample_dataset), batch_size=gen_bs)

        model.eval()
        device = next(model.parameters()).device

        all_data: List[Data] = []
        all_structures: List[Structure] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                outputs, _ = model.sample(batch, step_lr=self.step_lr)
                batch_data, batch_structures = _split_generation_output(outputs, batch)
                all_data.extend(batch_data)
                all_structures.extend(batch_structures)

        valid = [(d, s) for d, s in zip(all_data, all_structures) if s is not None]
        if valid:
            all_data = [d for d, _ in valid]
            all_structures = [cast(Structure, s) for _, s in valid]
        else:
            all_data, all_structures = [], []

        return all_data[:n_total], all_structures[:n_total]


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class SymmCDSuite(ModelSuite):
    def __init__(
        self,
        model_name: AVA_MODEL_NAME,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        model_path: Optional[str] = None,
        config_overrides: List[str] = [],
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            model_path=model_path,
            config_overrides=config_overrides,
            device=device,
            **kwargs,
        )

    def load_model(self):
        if self.model_path is None:
            raise ValueError("SymmCDSuite requires model_path")
        model, _ = _load_symmcd_model(str(self.model_path))
        return model

    def get_sampler(self) -> SymmCDSampler:
        if self.model_path is None:
            raise ValueError("SymmCDSuite requires model_path")
        restrict = list(self.sample_cfg.get("restrict_spacegroups", []))
        return SymmCDSampler(
            model_path=str(self.model_path),
            dataset=str(self.sample_cfg.get("dataset", "mp")),
            step_lr=float(self.sample_cfg.get("step_lr", 1e-5)),
            restrict_spacegroups=restrict or None,
            generation_batch_size=int(self.sample_cfg.get("generation_batch_size", 64)),
            sg_temperature=float(self.sample_cfg.get("sg_temperature", 1.0)),
        )

    def get_dataloader(self, *args, **kwargs) -> DataLoader:
        samples: List[Data] = kwargs.pop("samples", args[0] if len(args) > 0 else [])
        rewards: Optional[NDArray] = kwargs.pop(
            "rewards", args[1] if len(args) > 1 else None
        )
        batch_size: Optional[int] = kwargs.pop(
            "batch_size", args[2] if len(args) > 2 else None
        )
        shuffle: bool = kwargs.pop("shuffle", args[3] if len(args) > 3 else True)
        if batch_size is None:
            batch_size = self.finetune_cfg.batch_size
        assert batch_size is not None
        dataset = DiffCSPDataset(samples, rewards)
        return DataLoader(
            cast(Any, dataset), shuffle=shuffle, batch_size=int(batch_size)
        )

    def save_model(self, *args, **kwargs) -> None:
        model = kwargs.pop("model", args[0] if len(args) > 0 else None)
        save_dir: str = kwargs.pop("save_dir", args[1] if len(args) > 1 else None)
        if model is None or save_dir is None:
            raise ValueError("save_model requires model and save_dir")
        os.makedirs(save_dir, exist_ok=True)
        ckpt = {
            "state_dict": model.state_dict(),
            "hyper_parameters": model.hparams,
            "epoch": 0,
            "global_step": 0,
            "pytorch-lightning_version": "2.0.3",
        }
        ckpt_path = os.path.join(save_dir, "epoch=0-step=0.ckpt")
        torch.save(ckpt, ckpt_path)

        if self.model_path is not None:
            for fname in ("lattice_scaler.pt", "prop_scaler.pt", "hparams.yaml"):
                src = Path(str(self.model_path)) / fname
                if src.exists():
                    shutil.copy(src, Path(save_dir) / fname)
