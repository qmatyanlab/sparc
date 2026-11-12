from pathlib import Path

import numpy as np
import pytest
import yaml
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/surrogates/TSENN_static_dielectric_tensor.torch"
CARD = ROOT / "data/surrogates/TSENN_static_dielectric_tensor.yaml"

pytestmark = pytest.mark.skipif(not MODEL.is_file(), reason="static-dielectric surrogate weights not present")


def _calc(**kwargs):
    from rewards.calculators.tsenn_static_dielectric import TSENNStaticDielectric

    return TSENNStaticDielectric(root_dir=str(ROOT / "rewards/tsenn_static_dielectric_layered_uniaxial"),
                                 model_path=str(MODEL), config_path=str(CARD), device="cpu",
                                 standardize_structure="refined", **kwargs)


def _structures():
    nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rutile = Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(4.65, 2.97), ["Ti", "O"],
                                       [[0, 0, 0], [0.305, 0.305, 0]])
    return [nacl, rutile]


def test_default_recovers_eps_inf():
    """Trained without the softplus: NaCl's MP PBE eps_inf label is 2.49 and the training-time prediction 2.63."""
    t, ok = _calc().predict_static_tensor(_structures())
    assert ok.all()
    assert 2.3 < t[0, 0, 0] < 3.0
    assert t[1, 0, 0] == pytest.approx(t[1, 1, 1], rel=1e-7)  # in-plane isotropy exact (to float noise) in a uniaxial cell
    assert t[1, 2, 2] != pytest.approx(t[1, 0, 0], abs=0.1)


def test_legacy_flag_keeps_the_old_values():
    """softplus_0e=True reproduces runs made before the option existed."""
    t, ok = _calc(softplus_0e=True).predict_static_tensor(_structures())
    assert ok.all()
    assert np.diag(t[0]) == pytest.approx([4.78] * 3, abs=0.01)
    assert np.diag(t[1]) == pytest.approx([8.39, 8.39, 9.55], abs=0.01)


def test_legacy_reward_config_only_turns_the_softplus_on():
    legacy = yaml.safe_load((ROOT / "configs/reward/fom_inplane_isotropy_gapgate_legacy_softplus.yaml").read_text())
    base = yaml.safe_load((ROOT / "configs/reward/fom_inplane_isotropy_gapgate.yaml").read_text())
    calc = legacy["prop_cfg"][0]["calculator"]
    assert calc.pop("softplus_0e") is True
    assert "softplus_0e" not in base["prop_cfg"][0]["calculator"]  # base config uses the default (off)
    assert legacy == base  # nothing else differs
