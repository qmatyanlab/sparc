import csv
from types import SimpleNamespace

from pipeline.filters.opt_filter import parallel_run
from pipeline.sparc import SPARC


def _square(x):
    return x * x


def test_parallel_run_empty_input():
    assert parallel_run(_square, []) == []
    assert parallel_run(_square, [1, 2, 3], num_workers=2) == [1, 4, 9]


def test_metrics_csv_columns_never_shift(tmp_path):
    host = SimpleNamespace(save_dir=str(tmp_path), _csv_extra_warned=False)
    SPARC._append_metrics_csv(host, {"step": 0, "reward mean": 0.1, "cost": 3})
    SPARC._append_metrics_csv(host, {"step": 1, "n_hard_negatives": 2, "reward mean": 0.2, "cost": 7})
    SPARC._append_metrics_csv(host, {"step": 2, "cost": 9})
    with open(tmp_path / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["step"] for r in rows] == ["0", "1", "2"]
    assert [r["cost"] for r in rows] == ["3", "7", "9"]
    assert rows[1]["reward mean"] == "0.2" and rows[2]["reward mean"] == ""


def test_save_data_records_handles_mattergen_chemgraph(tmp_path):
    import torch
    from mattergen.common.data.chemgraph import ChemGraph
    from torch_geometric.data import Data

    from pipeline.utils.save import save_data_records

    graph = ChemGraph(
        pos=torch.rand(2, 3), cell=torch.eye(3).unsqueeze(0) * 4.0,
        atomic_numbers=torch.tensor([11, 17]), num_atoms=torch.tensor([2]),
    )
    symmcd = Data(frac_coords=torch.rand(1, 3), atom_types=torch.tensor([8]),
                  lengths=torch.ones(1, 3), angles=torch.full((1, 3), 90.0),
                  num_atoms=torch.tensor([1]), spacegroup=torch.tensor([221]))
    path = save_data_records([graph, symmcd], str(tmp_path), "x.pt")
    loaded = torch.load(path)
    assert set(loaded[0]) >= {"pos", "cell", "atomic_numbers"}
    assert loaded[1]["spacegroup"] == 221 and "frac_coords" in loaded[1]
