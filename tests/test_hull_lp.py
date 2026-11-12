import numpy as np
import pytest
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from pipeline.filters.hull_lp import LP_MIN_ELEMENTS, LPHull, phase_diagram_or_lp

ELEMENTS = ["Li", "Na", "O", "Cl", "S"]


def _random_entries(rng, n):
    entries = [PDEntry(Composition({el: 1}), rng.normal(-2.0, 0.5)) for el in ELEMENTS]
    for _ in range(n):
        els = rng.choice(ELEMENTS, size=rng.integers(2, 5), replace=False)
        comp = Composition({el: int(rng.integers(1, 5)) for el in els})
        entries.append(PDEntry(comp, comp.num_atoms * rng.normal(-3.0, 0.6)))
    return entries


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lp_matches_phase_diagram(seed):
    rng = np.random.default_rng(seed)
    entries = _random_entries(rng, 300)
    pd, lp = PhaseDiagram(entries), LPHull(entries)
    queries = _random_entries(rng, 200)[len(ELEMENTS):]
    for q in queries:
        assert lp.get_e_above_hull(q, allow_negative=True) == pytest.approx(
            pd.get_e_above_hull(q, allow_negative=True), abs=1e-7)
    for e in entries[::10]:  # entries that are part of the hull's input (self-consistent case)
        assert lp.get_e_above_hull(e, allow_negative=True) == pytest.approx(
            pd.get_e_above_hull(e, allow_negative=True), abs=1e-7)


def test_dispatch_by_element_count():
    rng = np.random.default_rng(0)
    assert isinstance(phase_diagram_or_lp(_random_entries(rng, 50)), PhaseDiagram)
    many = [PDEntry(Composition({el: 1}), -1.0) for el in
            ["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si"][:LP_MIN_ELEMENTS]]
    assert isinstance(phase_diagram_or_lp(many), LPHull)


def test_missing_element_raises():
    lp = LPHull([PDEntry(Composition("Li"), -1.0), PDEntry(Composition("O"), -1.0)])
    with pytest.raises(ValueError):
        lp.get_e_above_hull(PDEntry(Composition("NaCl"), -5.0), allow_negative=True)
