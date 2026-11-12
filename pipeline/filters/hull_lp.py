"""Energy above hull by linear programming, for chemical systems too large for a full convex hull.

MatterGen's energy metrics build a pymatgen PhaseDiagram over every reference entry of a structure's
chemical system, i.e. an (n-1)-dimensional convex hull for n elements. For a 12-element structure with
O, S, Cl, I, Li, Ba, ... qhull did not finish, so the RL run hung.

The energy above hull of one composition c only needs the lower hull at c:
    hull(c) = min sum_i x_i e_i  over x >= 0 with sum_i x_i f_i = c,
where e_i are per-atom energies and f_i fractional compositions of the entries. This is the value
PhaseDiagram.get_e_above_hull(entry, allow_negative=True) returns (tests/test_hull_lp.py). It is used only
for systems with at least LP_MIN_ELEMENTS elements; every smaller system keeps the PhaseDiagram path.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

# Smaller systems keep the PhaseDiagram path, so their results are unchanged.
LP_MIN_ELEMENTS = 11
_TOL = 1e-10


class LPHull:
    """Drop-in for the part of PhaseDiagram the MatterGen energy metrics use (get_e_above_hull)."""

    def __init__(self, entries):
        self.entries = list(entries)
        self.elements = sorted({el for e in self.entries for el in e.composition.elements})
        self._fractions = np.array(
            [[e.composition.get_atomic_fraction(el) for e in self.entries] for el in self.elements]
        )
        self._energies = np.array([e.energy_per_atom for e in self.entries])

    def get_hull_energy_per_atom(self, composition) -> float:
        target = np.array([composition.get_atomic_fraction(el) for el in self.elements])
        if not np.isclose(target.sum(), 1.0):
            raise ValueError(f"{composition.reduced_formula} has elements outside {self.elements}")
        res = linprog(self._energies, A_eq=self._fractions, b_eq=target, bounds=(0, None), method="highs",
                      options={"primal_feasibility_tolerance": _TOL, "dual_feasibility_tolerance": _TOL})
        if not res.success:
            raise ValueError(f"No hull at {composition.reduced_formula}: {res.message}")
        return float(res.fun)

    def get_e_above_hull(self, entry, allow_negative: bool = False) -> float:
        e_above_hull = entry.energy_per_atom - self.get_hull_energy_per_atom(entry.composition)
        if e_above_hull < -1e-8 and not allow_negative:
            raise ValueError(f"{entry.composition.reduced_formula} lies below the hull ({e_above_hull:.3g} eV/atom)")
        return e_above_hull


def phase_diagram_or_lp(entries, *args, **kwargs):
    """PhaseDiagram for systems below LP_MIN_ELEMENTS elements, LPHull from there on."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    entries = list(entries)
    if len({el for e in entries for el in e.composition.elements}) >= LP_MIN_ELEMENTS:
        return LPHull(entries)
    return PhaseDiagram(entries, *args, **kwargs)


def patch_mattergen_energy_metrics():
    """Route MatterGen's energy-metric phase diagrams through phase_diagram_or_lp."""
    import mattergen.evaluation.metrics.energy as energy

    energy.PhaseDiagram = phase_diagram_or_lp
