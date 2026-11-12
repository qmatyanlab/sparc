"""SymmCDSampler post-processing: clash rejection and sg_condition alignment."""

from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from models.suite.symmcd import SymmCDSampler, _split_generation_output

N_TYPES = 94


def _types(z_list):
    t = torch.zeros(len(z_list), N_TYPES)
    for i, z in enumerate(z_list):
        t[i, z - 1] = 1.0
    return t


def _outputs(frac_per_crystal, spacegroups, kept_index=None, a=5.0):
    frac = torch.tensor([c for cs in frac_per_crystal for c in cs], dtype=torch.float64)
    n = torch.tensor([len(cs) for cs in frac_per_crystal])
    out = {
        "num_atoms": n,
        "frac_coords": frac,
        "atom_types": _types([8] * len(frac)),
        "lattices": torch.eye(3).repeat(len(n), 1, 1) * a,
        "ks": torch.zeros(len(n), 6),
        "spacegroup": torch.tensor(spacegroups),
        "site_symm": torch.zeros(len(frac), 15, 13),
    }
    if kept_index is not None:
        out["kept_index"] = torch.tensor(kept_index)
    return out


OK = [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]]
CLASH = [[0.1, 0.1, 0.1], [0.14, 0.1, 0.1]]  # 0.2 A apart in a 5 A cube


def test_clash_rejected():
    batch = SimpleNamespace(num_graphs=2, sg_condition=None)
    _, structures = _split_generation_output(
        _outputs([OK, CLASH], [1, 1]), batch, min_interatomic_distance=0.5
    )
    assert structures[0] is not None and structures[1] is None
    # Disabled check keeps it (legacy behaviour).
    _, structures = _split_generation_output(
        _outputs([OK, CLASH], [1, 1]), batch, min_interatomic_distance=0.0
    )
    assert structures[1] is not None


def test_sg_condition_follows_kept_index():
    sg_condition = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    batch = SimpleNamespace(num_graphs=3, sg_condition=sg_condition.reshape(-1))
    data, _ = _split_generation_output(
        _outputs([OK, OK], [2, 6], kept_index=[0, 2]), batch
    )
    assert torch.equal(data[0].sg_condition, sg_condition[0])
    assert torch.equal(data[1].sg_condition, sg_condition[2])
    assert [d.spacegroup for d in data] == [2, 6]


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return Data(num_nodes=2, spacegroup=torch.tensor([1]), idx=torch.tensor([i]))


class _FakeModel(torch.nn.Module):
    """Every odd-numbered sample (by global draw order) clashes."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def sample(self, batch, step_lr=1e-5, orbit_merge_tol=0.5, cartesian_snap=True, strict_snap=True):
        crystals = []
        for _ in range(batch.num_graphs):
            crystals.append(CLASH if self.calls % 2 else OK)
            self.calls += 1
        return _outputs(crystals, [1] * len(crystals)), {}


def _sampler():
    sampler = SymmCDSampler(model_path="unused", generation_batch_size=4)
    sampler._build_sample_dataset = lambda model, total_num, restrict_spacegroups: _FakeDataset(total_num)
    return sampler


def test_clashing_samples_are_dropped_not_replaced():
    sampler = _sampler()
    data, structures = sampler.generate(_FakeModel(), batch_size=4, num_batches=1)
    assert len(structures) == 2 and len(data) == 2
    assert all(s is not None for s in structures)
