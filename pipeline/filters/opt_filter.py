import gc
import inspect
import io
import contextlib
import csv
import logging
import multiprocessing as mp
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Literal, Sequence, cast
import numpy as np
from numpy.typing import NDArray
from ase.filters import OptimizableFilter
from ase.io import write
from ase.optimize.optimize import Optimizer
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import torch


from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import (
    LMDBBackedReferenceDatasetImpl,
    gzip_decompress,
)
from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.metrics.structure import structure_validity, is_smact_valid
from mattergen.evaluation.utils.lmdb_utils import lmdb_get, lmdb_read_metadata
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential
from tempfile import mkdtemp

# Singleton cache: LMDBGZSerializer opens an LMDB environment that cannot be
# opened twice in the same process.  Both OptFilter and OptEval share this ref.
_LMDB_REFERENCE = None


class _LocalLMDBBackedReferenceDatasetImpl(LMDBBackedReferenceDatasetImpl):
    def _build_num_entries_by_chemsys_reduced_formulas(
        self, lmdb_path: Path
    ) -> dict[str, dict[str, int]]:
        result: defaultdict[str, dict[str, int]] = defaultdict(dict)
        with self.env.begin() as txn:
            chemical_systems = lmdb_get(txn, "chemical_systems")
            for chemsys in chemical_systems:
                reduced_formulas = lmdb_get(txn, f"{chemsys}.reduced_formulas")
                for reduced_formula in reduced_formulas:
                    result[chemsys][reduced_formula] = lmdb_get(
                        txn, f"{chemsys}.{reduced_formula}.length"
                    )
        return {key: val for key, val in result.items()}


def _deserialize_reference_dataset(dataset_path: str | Path) -> ReferenceDataset:
    tempdir = mkdtemp()
    lmdb_path = gzip_decompress(dataset_path, tempdir)
    name = lmdb_read_metadata(lmdb_path, "name")
    return ReferenceDataset(
        name=name,
        impl=_LocalLMDBBackedReferenceDatasetImpl(lmdb_path, cleanup_dir=True),
    )


def _get_reference(reference_path=None):
    global _LMDB_REFERENCE
    if _LMDB_REFERENCE is None:
        _path = reference_path or hf_hub_download(
            repo_id="jwchen25/MatInvent",
            filename="reference_MP2020correction.gz",
        )
        _LMDB_REFERENCE = _deserialize_reference_dataset(_path)
    return _LMDB_REFERENCE


METRIC_LIST = [
    "validity",
    "novel",
    "unique",
    "stable",
    "synthesizable",
]


def get_device(device: str | None = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def _worker_init():
    warnings.filterwarnings("ignore")


def parallel_run(func, args_list, num_workers: int | None = None):
    if num_workers is None:
        num_workers = mp.cpu_count()
    with mp.Pool(processes=num_workers, initializer=_worker_init) as pool:
        if isinstance(args_list[0], tuple):
            results = pool.starmap(func, args_list)
        else:
            results = pool.map(func, args_list)

    return results


def _expand_atomic_number_ranges(
    exclude_atomic_numbers: Sequence[int | Sequence[int]] | None,
) -> set[int]:
    if exclude_atomic_numbers is None:
        return set()

    excluded: set[int] = set()
    for item in exclude_atomic_numbers:
        if isinstance(item, int):
            excluded.add(item)
            continue

        bounds = [int(x) for x in item]
        if len(bounds) == 2:
            start, end = sorted(bounds)
            excluded.update(range(start, end + 1))
        else:
            excluded.update(bounds)

    return excluded


def _format_atomic_number_ranges(atomic_numbers: set[int]) -> str:
    if not atomic_numbers:
        return "disabled"

    ranges = []
    sorted_numbers = sorted(atomic_numbers)
    start = prev = sorted_numbers[0]
    for number in sorted_numbers[1:]:
        if number == prev + 1:
            prev = number
            continue
        ranges.append((start, prev))
        start = prev = number
    ranges.append((start, prev))

    return ",".join(
        str(start) if start == end else f"{start}-{end}" for start, end in ranges
    )


def invalid_filter(
    sample_data,
    sample_struc,
    return_mask=False,
    max_atoms_per_structure: int | None = None,
    min_volume_per_atom: float | None = None,
    exclude_atomic_numbers: Sequence[int | Sequence[int]] | None = None,
    label: str = "Invalid filter",
):
    valid_comp = parallel_run(structure_validity, sample_struc)
    valid_struc = parallel_run(is_smact_valid, sample_struc)
    valid_comp = np.array(valid_comp, dtype=bool)
    valid_struc = np.array(valid_struc, dtype=bool)
    valid_cell = np.array([max(list(s.lattice.abc)) < 25 for s in sample_struc])
    if max_atoms_per_structure is None:
        valid_atoms = np.ones(len(sample_struc), dtype=bool)
    else:
        valid_atoms = np.array(
            [len(s) < max_atoms_per_structure for s in sample_struc],
            dtype=bool,
        )
    if min_volume_per_atom is None:
        valid_volume_per_atom = np.ones(len(sample_struc), dtype=bool)
    else:
        valid_volume_per_atom = np.array(
            [
                len(s) > 0 and (float(s.volume) / len(s)) >= min_volume_per_atom
                for s in sample_struc
            ],
            dtype=bool,
        )
    excluded_atomic_numbers = _expand_atomic_number_ranges(exclude_atomic_numbers)
    if excluded_atomic_numbers:
        valid_elements = np.array(
            [
                all(int(el.Z) not in excluded_atomic_numbers for el in s.composition)
                for s in sample_struc
            ],
            dtype=bool,
        )
    else:
        valid_elements = np.ones(len(sample_struc), dtype=bool)

    mask = (
        valid_comp
        & valid_struc
        & valid_cell
        & valid_atoms
        & valid_volume_per_atom
        & valid_elements
    )
    n_total = len(sample_struc)
    max_atoms_label = (
        "disabled" if max_atoms_per_structure is None else f"<{max_atoms_per_structure}"
    )
    min_volume_label = (
        "disabled" if min_volume_per_atom is None else f">={min_volume_per_atom:g}"
    )
    exclude_label = _format_atomic_number_ranges(excluded_atomic_numbers)
    logging.info(
        "%s: kept %d/%d | structure_validity %d/%d | "
        "smact_validity %d/%d | max_lattice_abc<25 %d/%d | "
        "num_atoms%s %d/%d | volume_per_atom%s %d/%d | "
        "excluded_atomic_numbers=%s %d/%d",
        label,
        int(mask.sum()),
        n_total,
        int(valid_comp.sum()),
        n_total,
        int(valid_struc.sum()),
        n_total,
        int(valid_cell.sum()),
        n_total,
        max_atoms_label,
        int(valid_atoms.sum()),
        n_total,
        min_volume_label,
        int(valid_volume_per_atom.sum()),
        n_total,
        exclude_label,
        int(valid_elements.sum()),
        n_total,
    )
    filtered_data = [x for x, m in zip(sample_data, mask) if m]
    filtered_struc = [x for x, m in zip(sample_struc, mask) if m]

    if return_mask:
        return mask

    return filtered_data, filtered_struc


def _relaxation_kwargs(cfg) -> dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        return {}
    allowed = {
        "fmax",
        "max_n_steps",
        "max_natoms_per_batch",
        "max_total_atoms_per_relax_call",
        "optimizer",
        "cell_filter",
    }
    return {key: value for key, value in cfg_dict.items() if key in allowed}


def _ensure_ase_gradient_compat() -> None:
    if hasattr(OptimizableFilter, "get_gradient"):
        pass
    else:

        def get_gradient(self):
            return -self.get_forces()

        OptimizableFilter.get_gradient = get_gradient

    converged_sig = inspect.signature(Optimizer.converged)
    gradient_param = converged_sig.parameters.get("gradient")
    if gradient_param is None or gradient_param.default is not inspect._empty:
        return

    original_converged = Optimizer.converged

    def converged(self, gradient=None):
        if gradient is None:
            gradient = self.optimizable.get_gradient()
        return original_converged(self, gradient)

    Optimizer.converged = converged


def _chunk_structures_by_total_atoms(
    structures: list[Structure],
    max_total_atoms_per_relax_call: int | None,
) -> list[list[Structure]]:
    if not structures:
        return []
    if max_total_atoms_per_relax_call is None or max_total_atoms_per_relax_call <= 0:
        return [structures]

    chunks: list[list[Structure]] = []
    current_chunk: list[Structure] = []
    current_atoms = 0

    for structure in structures:
        natoms = len(structure)
        if current_chunk and current_atoms + natoms > max_total_atoms_per_relax_call:
            chunks.append(current_chunk)
            current_chunk = [structure]
            current_atoms = natoms
            continue

        current_chunk.append(structure)
        current_atoms += natoms

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _run_batch_relaxer(
    batch_relaxer: BatchRelaxer,
    atoms_list: list,
    max_n_steps: int | None,
):
    if max_n_steps is None:
        return batch_relaxer.relax(atoms_list)

    batch_relaxer.trajectories = {}
    batch_relaxer.tqdmcounter = tqdm(total=len(atoms_list), file=sys.stdout)
    pointer = 0
    step_counts = {}
    atoms_list_ = []
    for i in range(len(atoms_list)):
        atoms_list_.append(atoms_list[i].copy())
        atoms_list_[i].info["structure_index"] = i

    while pointer < len(atoms_list) or not batch_relaxer.finished:
        while pointer < len(atoms_list) and (
            sum(len(opt.atoms) for opt in batch_relaxer.optimizer_instances)
            + len(atoms_list[pointer])
            <= batch_relaxer.max_natoms_per_batch
        ):
            batch_relaxer.insert(atoms_list_[pointer])
            structure_index = atoms_list_[pointer].info["structure_index"]
            step_counts[structure_index] = 0
            batch_relaxer.tqdmcounter.update(1)
            pointer += 1

        stepped_indices = [
            opt.atoms.info["structure_index"]
            for opt in batch_relaxer.optimizer_instances
        ]
        batch_relaxer.step_batch()
        for structure_index in stepped_indices:
            step_counts[structure_index] = step_counts.get(structure_index, 0) + 1

        capped_instances = []
        capped_count = 0
        for opt in batch_relaxer.optimizer_instances:
            structure_index = opt.atoms.info["structure_index"]
            if step_counts.get(structure_index, 0) >= max_n_steps:
                batch_relaxer.trajectories[opt.atoms.info["structure_index"]].append(
                    opt.atoms.copy()
                )
                capped_count += 1
            else:
                capped_instances.append(opt)

        if capped_count:
            print(
                f"[OptFilter] Step cap reached for {capped_count} structures "
                + f"(max_n_steps={max_n_steps})",
                flush=True,
            )

        batch_relaxer.optimizer_instances = capped_instances
        batch_relaxer.is_active_instance = [True] * len(capped_instances)
        batch_relaxer.finished = pointer >= len(atoms_list) and not capped_instances

    batch_relaxer.tqdmcounter.close()
    return batch_relaxer.trajectories


def _relax_structures_with_progress(
    structures: list[Structure],
    device: torch.device,
    potential_load_path: str,
    output_path: str | None = None,
    **kwargs,
) -> tuple[list[Structure], np.ndarray]:
    _ensure_ase_gradient_compat()

    relax_kwargs = dict(kwargs)
    optimizer = relax_kwargs.pop("optimizer", "FIRE")
    cell_filter = relax_kwargs.pop("cell_filter", "EXPCELLFILTER")
    max_n_steps = relax_kwargs.pop("max_n_steps", None)
    max_total_atoms_per_relax_call = relax_kwargs.pop(
        "max_total_atoms_per_relax_call", None
    )
    total = len(structures)
    chunks = _chunk_structures_by_total_atoms(
        structures,
        max_total_atoms_per_relax_call,
    )
    print(
        "[OptFilter] Relax config: "
        f"total={total}, optimizer={optimizer}, filter={cell_filter}, "
        f"fmax={relax_kwargs.get('fmax', 0.05)}, "
        f"max_n_steps={max_n_steps if max_n_steps is not None else 'default'}, "
        f"max_natoms_per_batch={relax_kwargs.get('max_natoms_per_batch', 512)}, "
        f"max_total_atoms_per_relax_call={max_total_atoms_per_relax_call}, "
        f"num_calls={len(chunks)}",
        flush=True,
    )

    start = time.perf_counter()
    potential = cast(Any, Potential).from_checkpoint(
        device=str(device),
        load_path=potential_load_path,
        load_training_state=False,
    )
    relaxed_atoms = []
    for chunk_idx, chunk_structures in enumerate(chunks, start=1):
        chunk_atoms = [
            cast(Any, AseAtomsAdaptor.get_atoms(s)) for s in chunk_structures
        ]
        batch_relaxer = BatchRelaxer(
            potential=potential,
            optimizer=optimizer,
            filter=cell_filter,
            **relax_kwargs,
        )
        print(
            f"[OptFilter] Relax call {chunk_idx}/{len(chunks)}: "
            f"structures={len(chunk_structures)}, "
            f"atoms={sum(len(s) for s in chunk_structures)}",
            flush=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"logm result may be inaccurate, approximate err = .*",
                category=RuntimeWarning,
                module=r"ase\.filters",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Please use atoms\.calc = calc",
                category=FutureWarning,
                module=r"mattersim\.applications\.batch_relax",
            )
            relaxation_trajectories = _run_batch_relaxer(
                batch_relaxer,
                chunk_atoms,
                max_n_steps=max_n_steps,
            )
        relaxed_atoms.extend(
            relaxation_trajectories[idx][-1] for idx in sorted(relaxation_trajectories)
        )
        del relaxation_trajectories
        del batch_relaxer
        del chunk_atoms
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_energies = np.array([a.info["total_energy"] for a in relaxed_atoms])

    if output_path:
        write(output_path, relaxed_atoms, format="extxyz")

    elapsed = time.perf_counter() - start
    rate = (len(relaxed_atoms) / elapsed) if elapsed > 0 else float("inf")
    print(
        f"[OptFilter] Relaxation finished in {elapsed:.1f}s ({rate:.2f} structures/s)",
        flush=True,
    )

    relaxed_structures = [
        AseAtomsAdaptor.get_structure(cast(Any, a)) for a in relaxed_atoms
    ]
    return relaxed_structures, total_energies


def _normalize_energies(
    energies: Sequence[float] | np.ndarray | None, count: int
) -> list[float]:
    if energies is None:
        return [float("nan")] * count
    if isinstance(energies, np.ndarray):
        return energies.astype(float).tolist()
    return [float(e) for e in energies]


class OptFilter:
    def __init__(
        self,
        metrics: List[str],
        relax: bool = True,
        silent: bool = True,
        device: str | None = None,
        reference_path: str | None = None,
        structure_matcher: Literal["ordered", "disordered"] = "disordered",
        penalize_unstable: bool = False,
        **kwargs,
    ) -> None:
        assert all(m in METRIC_LIST for m in metrics)
        self.metrics = metrics
        self.relax = relax
        # When True, structures failing ONLY the `stable` (e_hull) metric are KEPT (and
        # tagged via struc.properties["is_unstable"]=True) instead of dropped, so the RL
        # loop can penalize them as hard negatives. Off by default => legacy behavior.
        self.penalize_unstable = penalize_unstable
        self.silent = silent
        self.device = get_device(device)
        self.structure_matcher = (
            DefaultDisorderedStructureMatcher()
            if structure_matcher == "disordered"
            else DefaultOrderedStructureMatcher()
        )
        self.cfg = OmegaConf.create(kwargs)
        self.reference = _get_reference(reference_path)

    def __call__(
        self,
        data_list: list,
        structures: list[Structure],
        energies: list[float] | None = None,
        potential_load_path: str = "MatterSim-v1.0.0-5M.pth",
        save_path: str | None = None,
    ):
        n_input = len(structures)
        no_ref_strucs = structures  # keep reference before filtering for sidecar
        data_list, structures, pre_filter_mask = self.pre_filter_ehull(
            data_list, structures
        )
        n_after_ehull = len(structures)
        n_no_reference = n_input - n_after_ehull
        print(
            f"[OptFilter] Input: {n_input} → after ehull pre-filter: {n_after_ehull} ({n_no_reference} no-reference)",
            flush=True,
        )
        print(f"[OptFilter] Relaxing {n_after_ehull} structures...", flush=True)

        if self.relax and energies is None:
            relaxed_struc, relax_energies = _relax_structures_with_progress(
                structures,
                device=self.device,
                potential_load_path=potential_load_path,
                **_relaxation_kwargs(self.cfg),
            )
        else:
            relaxed_struc = structures
            relax_energies = _normalize_energies(energies, len(structures))

        if self.silent:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                (
                    mask,
                    per_metric_masks,
                    metrics,
                    evaluator,
                ) = self._compute_metrics(structures, relaxed_struc, relax_energies)
        else:
            (
                mask,
                per_metric_masks,
                metrics,
                evaluator,
            ) = self._compute_metrics(structures, relaxed_struc, relax_energies)

        # Penalize-unstable mode: keep structures that fail ONLY the stability gate so the
        # RL loop can use them as hard negatives. `mask`/CSV/metrics still report the TRUE
        # stable mask; only `keep_mask` (what we return) is relaxed, and kept-but-unstable
        # structures are tagged on their .properties (survives downstream reordering).
        keep_mask = mask
        if self.penalize_unstable and "stable" in per_metric_masks:
            stable_pm = per_metric_masks["stable"]
            non_stable_masks = [
                v for k, v in per_metric_masks.items() if k != "stable"
            ]
            keep_mask = (
                np.logical_and.reduce(non_stable_masks)
                if non_stable_masks
                else np.ones_like(mask)
            )
            for struc, keep, stable in zip(relaxed_struc, keep_mask, stable_pm):
                if keep:
                    struc.properties["is_unstable"] = bool(not stable)

        n_survived = int(mask.sum())
        breakdown = ", ".join(
            f"{k}: {int(v.sum())}/{n_after_ehull}" for k, v in per_metric_masks.items()
        )
        print(
            f"[OptFilter] Survived: {n_survived}/{n_after_ehull} ({breakdown})",
            flush=True,
        )
        if self.penalize_unstable:
            print(
                f"[OptFilter] penalize_unstable: keeping {int(keep_mask.sum())}/"
                f"{n_after_ehull} (incl. {int(keep_mask.sum()) - n_survived} unstable as "
                "hard negatives)",
                flush=True,
            )

        if save_path is not None and len(relaxed_struc) > 0:
            self._save_stability_csv(
                evaluator,
                relaxed_struc,
                relax_energies,
                mask,
                save_path,
                per_metric_masks,
            )

        # Inject filter counts/ratios into metrics so callers can log them
        for key, pmask in per_metric_masks.items():
            metrics[f"filter_{key}_ratio"] = (
                float(pmask.mean()) if n_after_ehull > 0 else float("nan")
            )
        metrics["filter_kept_ratio"] = (
            float(mask.mean()) if n_after_ehull > 0 else float("nan")
        )
        metrics["filter_total"] = n_after_ehull
        metrics["filter_kept"] = n_survived
        if self.penalize_unstable:
            metrics["unstable_kept"] = int(keep_mask.sum()) - n_survived
        metrics["frac_no_reference"] = (
            float(n_no_reference / n_input) if n_input > 0 else float("nan")
        )

        # Save sidecar file for no-reference structures (novel compositions with no MP data)
        if save_path is not None and n_no_reference > 0:
            no_ref_path = save_path.replace(".csv", "_no_reference.txt")
            no_ref_strucs_filtered = [
                s for s, m in zip(no_ref_strucs, pre_filter_mask) if not m
            ]
            with open(no_ref_path, "w") as f:
                f.write(
                    "# Structures removed by pre_filter_ehull (no MP reference data)\n"
                )
                f.write(
                    "# These are potentially novel compositions not in the MP database\n"
                )
                for i, s in enumerate(no_ref_strucs_filtered):
                    f.write(
                        f"{s.composition.reduced_formula}\t{s.composition.formula}\n"
                    )
            print(
                f"[OptFilter] Saved {n_no_reference} no-reference structure formulas → {no_ref_path}",
                flush=True,
            )

        filtered_data = [x for x, m in zip(data_list, keep_mask) if m]
        filtered_struc = [x for x, m in zip(relaxed_struc, keep_mask) if m]
        return filtered_data, filtered_struc, metrics

    def _save_stability_csv(
        self,
        evaluator,
        relaxed_struc,
        energies,
        mask,
        save_path,
        per_metric_masks=None,
    ):
        try:
            from mattergen.evaluation.metrics.energy import EnergyMetricsCapability

            has_energy = EnergyMetricsCapability in evaluator.available_capability_types
            ehull = (
                evaluator.energy_capability.energy_above_hull if has_energy else None
            )
            sc_ehull = (
                evaluator.energy_capability.self_consistent_energy_above_hull
                if has_energy
                else None
            )
            threshold = (
                evaluator.energy_capability.stability_threshold if has_energy else 0.1
            )
        except Exception:
            ehull = sc_ehull = None
            threshold = 0.1

        validity_mask = None
        if per_metric_masks is not None:
            validity_mask = per_metric_masks.get("validity")

        unique_mask = getattr(evaluator, "is_unique", None)
        novel_mask = getattr(evaluator, "is_novel", None)

        rows = []
        kept_idx = 0
        for i, (struc, kept) in enumerate(zip(relaxed_struc, mask)):
            e_total = (
                float(energies[i])
                if energies is not None and energies[i] is not None
                else float("nan")
            )
            nsites = struc.num_sites
            e_per_atom = e_total / nsites if nsites > 0 else float("nan")
            e_hull = float(ehull[i]) if ehull is not None else float("nan")
            sc_e_hull = float(sc_ehull[i]) if sc_ehull is not None else float("nan")
            is_stable = bool(e_hull <= threshold) if ehull is not None else False
            rows.append(
                {
                    "index": i,
                    "kept": kept,
                    "kept_index": kept_idx if kept else -1,
                    "reduced_formula": struc.composition.reduced_formula,
                    "nsites": nsites,
                    "total_energy_ev": e_total,
                    "energy_per_atom_ev": e_per_atom,
                    "energy_above_hull_ev_per_atom": e_hull,
                    "stability_threshold_ev_per_atom": threshold,
                    "is_valid": "" if validity_mask is None else bool(validity_mask[i]),
                    "is_unique": "" if unique_mask is None else bool(unique_mask[i]),
                    "is_novel": "" if novel_mask is None else bool(novel_mask[i]),
                    "is_stable": is_stable,
                    "self_consistent_energy_above_hull_ev_per_atom": sc_e_hull,
                }
            )
            if kept:
                kept_idx += 1

        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OptFilter] Saved stability data → {save_path}", flush=True)

    def pre_filter_ehull(
        self,
        data_list: list,
        structures: list[Structure],
    ):
        uni_element_set = set()
        for s in structures:
            uni_element_set.update([str(el) for el in s.composition.elements])
        terminal_systems = uni_element_set
        ref_set = set(self.reference.entries_by_chemsys.keys())
        missing_terminals = list(terminal_systems - ref_set)
        terminals_in_reference = terminal_systems & ref_set
        missing_energy = [
            chemsys
            for chemsys in terminals_in_reference
            if all(
                [np.isnan(e.energy) for e in self.reference.entries_by_chemsys[chemsys]]
            )
        ]

        mask = []
        bad_elements = set(missing_terminals + missing_energy)
        for struc in structures:
            struc_element_set = set([str(e) for e in struc.composition.elements])
            if len(struc_element_set & bad_elements) > 0:
                mask.append(False)
            else:
                mask.append(True)

        filtered_data = [x for x, m in zip(data_list, mask) if m]
        filtered_struc = [x for x, m in zip(structures, mask) if m]
        return filtered_data, filtered_struc, mask

    def _compute_metrics(
        self,
        original_structures: list[Structure],
        relaxed_structures: list[Structure],
        energies: Sequence[float] | np.ndarray | None,
    ):
        normalized_energies = _normalize_energies(energies, len(relaxed_structures))
        print(
            f"[OptFilter] Computing metrics for {len(relaxed_structures)} structures...",
            flush=True,
        )
        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=relaxed_structures,
            energies=normalized_energies,
            original_structures=original_structures,
            reference=self.reference,
            structure_matcher=self.structure_matcher,
        )

        metric_dict = evaluator.compute_metrics(
            metrics=evaluator.available_metrics,
        )

        mask_list = []
        per_metric_masks = {}
        if "validity" in self.metrics:
            valid_struc = parallel_run(structure_validity, relaxed_structures)
            valid_comp = parallel_run(is_smact_valid, relaxed_structures)
            valid_mask = np.array(valid_comp) & np.array(valid_struc)
            mask_list.append(valid_mask)
            per_metric_masks["validity"] = valid_mask

        if "novel" in self.metrics:
            mask_list.append(evaluator.is_novel)
            per_metric_masks["novel"] = evaluator.is_novel

        if "unique" in self.metrics:
            mask_list.append(evaluator.is_unique)
            per_metric_masks["unique"] = evaluator.is_unique

        if "stable" in self.metrics:
            mask_list.append(evaluator.is_stable)
            per_metric_masks["stable"] = evaluator.is_stable

        if "synthesizable" in self.metrics:
            pass

        mask_all = np.logical_and.reduce(mask_list)
        return (
            mask_all,
            per_metric_masks,
            metric_dict,
            evaluator,
        )

    @staticmethod
    def get_slice(data: list, mask: NDArray) -> list:
        """Filters a list of data points based on a boolean mask."""
        assert len(data) == len(mask), "Data and mask must have the same length."
        return [x for x, m in zip(data, mask) if m]


class OptEval:
    def __init__(
        self,
        relax: bool = True,
        silent: bool = True,
        device: str | None = None,
        structure_matcher: Literal["ordered", "disordered"] = "disordered",
        **kwargs,
    ) -> None:
        self.relax = relax
        self.silent = silent
        self.device = get_device(device)
        self.structure_matcher = (
            DefaultDisorderedStructureMatcher()
            if structure_matcher == "disordered"
            else DefaultOrderedStructureMatcher()
        )
        self.cfg = OmegaConf.create(kwargs)

        self.reference = _get_reference()

    def __call__(
        self,
        structures: list[Structure],
        energies: list[float] | None = None,
        potential_load_path: str = "MatterSim-v1.0.0-5M.pth",
    ):
        structures = self.pre_filter_ehull(structures)

        if self.relax and energies is None:
            relaxed_struc, relax_energies = _relax_structures_with_progress(
                structures,
                device=self.device,
                potential_load_path=potential_load_path,
                **_relaxation_kwargs(self.cfg),
            )
        else:
            relaxed_struc = structures
            relax_energies = _normalize_energies(energies, len(structures))

        if self.silent:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                metrics = self._compute_metrics(
                    structures, relaxed_struc, relax_energies
                )
        else:
            metrics = self._compute_metrics(structures, relaxed_struc, relax_energies)

        return metrics, relaxed_struc

    def pre_filter_ehull(
        self,
        structures: list[Structure],
    ):
        uni_element_set = set()
        for s in structures:
            uni_element_set.update([str(el) for el in s.composition.elements])
        terminal_systems = uni_element_set
        ref_set = set(self.reference.entries_by_chemsys.keys())
        missing_terminals = list(terminal_systems - ref_set)
        terminals_in_reference = terminal_systems & ref_set
        missing_energy = [
            chemsys
            for chemsys in terminals_in_reference
            if all(
                [np.isnan(e.energy) for e in self.reference.entries_by_chemsys[chemsys]]
            )
        ]

        mask = []
        bad_elements = set(missing_terminals + missing_energy)
        for struc in structures:
            struc_element_set = set([str(e) for e in struc.composition.elements])
            if len(struc_element_set & bad_elements) > 0:
                mask.append(False)
            else:
                mask.append(True)

        filtered_struc = [x for x, m in zip(structures, mask) if m]
        return filtered_struc

    def _compute_metrics(
        self,
        original_structures: list[Structure],
        relaxed_structures: list[Structure],
        energies: Sequence[float] | np.ndarray | None,
    ):
        normalized_energies = _normalize_energies(energies, len(relaxed_structures))
        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=relaxed_structures,
            energies=normalized_energies,
            original_structures=original_structures,
            reference=self.reference,
            structure_matcher=self.structure_matcher,
        )

        metric_dict = evaluator.compute_metrics(
            metrics=evaluator.available_metrics,
        )

        return metric_dict
