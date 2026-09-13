"""Canonical material ids and the sanitization rule that joins every source together.

The canonical ``mat_id`` is the MongoDB ``metadata.structure_label`` (or, for the legacy
pilot, ``metadata.candidate``) **verbatim** -- it is the only key present for all 276
materials, and it is also the stem of the RL as-generated CIF.

The QE side (``qe_ht/database_dfpt/``, the ``work_dfpt_*`` scratch dirs) stores the same
materials under a filename-safe spelling in which ``.``, ``(`` and ``)`` became ``_``::

    r020_RbSc(TeO4)2_SG150_step17_i13   ->  r020_RbSc_TeO4_2_SG150_step17_i13
    biaxial_r00_TeAs_SG61_Eg0.87_...    ->  biaxial_r00_TeAs_SG61_Eg0_87_...

``sanitize`` reproduces that spelling. It is applied in ONE direction only (canonical ->
qe_ht) and the alias map is asserted injective, because two distinct canonical ids that
sanitized to the same string would silently mislabel a row.
"""
from __future__ import annotations

import re

__all__ = [
    "SPLITS", "SPLIT_LABELS", "sanitize", "build_alias_map", "parse_label", "resolve",
]

SPLITS = ("slme", "dielectric", "failed_demo", "legacy_pilot")

SPLIT_LABELS = {
    "slme": "SLME solar-absorber candidates (VASP relax+static, QE optical, no DFPT)",
    "dielectric": "In-plane-isotropy dielectric candidates (VASP relax+static+DFPT, QE optical)",
    "failed_demo": "Failure-mode counter-examples (biaxial / cubic / metallic)",
    "legacy_pilot": "Legacy NERSC dielectric pilot (older flow, no QE optical layer)",
}

_UNSAFE = re.compile(r"[^A-Za-z0-9_]")

# r001_OsO4_SG173_step96_i15  /  biaxial_r03_TeI2_SG60_Eg0.36_step87_i11
_RE_STEP_I = re.compile(r"_step(?P<step>\d+)_i(?P<index>\d+)$")
_RE_RANK = re.compile(r"^rank(?P<rank>\d+)_")
_RE_R = re.compile(r"^(?:(?P<mode>[a-z]+)_)?r(?P<rank>\d+)_")
_RE_SG = re.compile(r"_[Ss][Gg](?P<sg>\d+)(?:_|$)")
_RE_EG = re.compile(r"_Eg(?P<eg>\d+[._]\d+)(?:_|$)")


def sanitize(mat_id: str) -> str:
    """Canonical id -> the filename-safe spelling used by ``qe_ht``."""
    return _UNSAFE.sub("_", mat_id)


def build_alias_map(mat_ids) -> dict[str, str]:
    """{sanitized -> canonical}. Raises if the mapping is not injective."""
    alias: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for m in mat_ids:
        s = sanitize(m)
        if s in alias and alias[s] != m:
            collisions.append((s, alias[s], m))
        alias[s] = m
    if collisions:
        raise ValueError(
            "sanitize() is not injective over these material ids; the qe_ht join would "
            f"mislabel rows: {collisions}"
        )
    return alias


def resolve(key: str, alias: dict[str, str]) -> str | None:
    """Map any spelling of an id back to the canonical one (identity if already canonical)."""
    if key in alias.values():
        return key
    return alias.get(key) or alias.get(sanitize(key))


def parse_label(mat_id: str) -> dict:
    """Pull the lineage that the label itself encodes: rank, RL step, eval index, SG, mode.

    Returns keys with ``None`` where the family does not carry that piece.
    """
    out = {"label_rank": None, "label_step": None, "label_index": None,
           "label_spacegroup": None, "label_gap_ev": None, "failure_mode": None}

    m = _RE_STEP_I.search(mat_id)
    if m:
        out["label_step"] = int(m.group("step"))
        out["label_index"] = int(m.group("index"))

    m = _RE_RANK.match(mat_id)
    if m:
        out["label_rank"] = int(m.group("rank"))
    else:
        m = _RE_R.match(mat_id)
        if m:
            out["label_rank"] = int(m.group("rank"))
            if m.group("mode"):
                out["failure_mode"] = m.group("mode")

    m = _RE_SG.search(mat_id)
    if m:
        out["label_spacegroup"] = int(m.group("sg"))

    m = _RE_EG.search(mat_id)
    if m:
        out["label_gap_ev"] = float(m.group("eg").replace("_", "."))

    return out
