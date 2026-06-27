"""SymmCD Figure-1 decomposition for an ACTUAL sampled crystal.

Loads a real multi-element sample produced by the SymmCD model, runs
pymatgen's SpacegroupAnalyzer to recover the asymmetric unit and Wyckoff
labels, then renders the same three-panel template as the toy schematic:

    (a) Asymmetric unit       coords + species + site symm (Wyckoff)
    (b) Site symmetries       asu atoms + all symmetry images linked
    (c) Replicated unit cell  full crystal in the conventional cell

Selected sample: ErGePd2 (Pnma, SG 62), 3 asymmetric-unit sites → 16 atoms.
The structure is rendered as a 2-D projection looking down the b axis
(i.e. onto the ac plane) so the asymmetric-unit region shows up as a
shaded rectangle, just like the wallpaper-group toy figure.

Run:
    source .venv/bin/activate
    python paper_materials/symmcd_real_sample_decomp.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.vis.structure_vtk import EL_COLORS


SAMPLE_PT = Path(
    "/global/cfs/cdirs/m2663/angush/sparc/exp_res/"
    "bandgap3_symmcd_sgtemp3_test_v12/samples/step_0010_eval.pt"
)
SAMPLE_INDEX = 7  # TbPr2Ge, Pnma — balanced cell aspect (a=5.7, b=7.3, c=4.9)

# Pnma asu has dimensions (1/2, 1/4, 1) and 8 equivalent placements in the
# unit cell.  We pick whichever placement best fits the model's pymatgen-rep
# atoms as-is (anchor on the atoms it produced, then draw the box around them).
ASU_DIMS = (0.5, 0.25, 1.0)
ASU_X_OFFSETS = (0.0, 0.5)
ASU_Y_OFFSETS = (0.0, 0.25, 0.5, 0.75)
ASU_Z_OFFSETS = (0.0,)


def candidate_boxes():
    for xo in ASU_X_OFFSETS:
        for yo in ASU_Y_OFFSETS:
            for zo in ASU_Z_OFFSETS:
                yield (
                    (xo, xo + ASU_DIMS[0]),
                    (yo, yo + ASU_DIMS[1]),
                    (zo, zo + ASU_DIMS[2]),
                )


def in_box(p: np.ndarray, box, tol: float = 0.02) -> bool:
    return all(box[i][0] - tol <= p[i] <= box[i][1] + tol for i in range(3))


def infer_asu_box(raw_reps: np.ndarray, tol: float = 0.02):
    best, best_score = None, -1
    for box in candidate_boxes():
        score = sum(in_box(np.mod(p, 1.0), box, tol) for p in raw_reps)
        if score > best_score:
            best, best_score = box, score
    return best, best_score

# Project onto the ac plane (drop b -> view down +b).  The two visible axes
# are x and z in fractional coords.
PROJ_X, PROJ_Y = 0, 2
DROP_AXIS = 1
PROJ_LABELS = ("x  (a, fractional)", "z  (c, fractional)")


# ---------------------------------------------------------------------------
# Sample loading + colour helpers
# ---------------------------------------------------------------------------


def load_sample(path: Path, idx: int) -> Structure:
    samples = torch.load(path, map_location="cpu", weights_only=False)
    s = samples[idx]
    species = [Element.from_Z(int(z)).symbol for z in s["atom_types"].tolist()]
    L = s["lengths"][0].tolist()
    A = s["angles"][0].tolist()
    return Structure(Lattice.from_parameters(*L, *A), species,
                     s["frac_coords"].numpy())


def jmol(symbol: str) -> tuple[float, float, float]:
    return tuple(c / 255.0 for c in EL_COLORS["Jmol"][symbol])


# ---------------------------------------------------------------------------
# Asymmetric-unit folding
# ---------------------------------------------------------------------------


def fold_into_box(frac: np.ndarray, sym_ops, box, tol: float = 0.02
                   ) -> np.ndarray:
    centre = np.array([0.5 * (b[0] + b[1]) for b in box])
    cands: list[np.ndarray] = []
    for op in [None] + list(sym_ops):
        base = frac if op is None else op.operate(frac)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    p = base + np.array([dx, dy, dz])
                    if in_box(p, box, tol):
                        cands.append(p)
    if not cands:
        return np.mod(frac, 1.0)
    cands.sort(key=lambda p: np.linalg.norm(p - centre))
    return cands[0]


# ---------------------------------------------------------------------------
# 2-D drawing helpers (projected onto the ac plane)
# ---------------------------------------------------------------------------


def draw_unit_cell(ax) -> None:
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black",
                            lw=1.4))


def draw_asu_box(ax, box, alpha: float = 0.4, label: bool = False) -> None:
    (x0, x1), (_, _), (z0, z1) = box
    ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0,
                            facecolor="#ffd166", edgecolor="#b8860b",
                            lw=1.0, alpha=alpha, zorder=0))
    if label:
        cx = 0.5 * (x0 + x1)
        cz = 0.5 * (z0 + z1)
        # callout placed in the empty quadrant opposite the box
        tx = 0.78 if cx < 0.5 else 0.22
        tz = 0.86 if cz < 0.5 else 0.14
        ax.annotate(
            "asymmetric\nunit\n(Pnma)",
            xy=(cx, cz), xytext=(tx, tz),
            fontsize=8, ha="center", va="center",
            color="#7a5300", style="italic",
            arrowprops=dict(arrowstyle="->", color="#7a5300", lw=1.0,
                            connectionstyle="arc3,rad=-0.35",
                            shrinkA=10, shrinkB=4),
        )


def project(frac: np.ndarray) -> tuple[float, float]:
    return float(frac[PROJ_X]), float(frac[PROJ_Y])


def draw_atom(ax, frac, colour, size=180, lw=0.9, alpha=1.0,
              symbol: str | None = None, ec: str = "black") -> None:
    x, y = project(frac)
    ax.plot(x, y, "o", ms=np.sqrt(size) * 1.6, mfc=colour, mec=ec,
            mew=lw, alpha=alpha, zorder=5)
    if symbol is not None:
        ax.text(x, y, symbol, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=6)


def style_axes(ax, title: str, lattice: Lattice) -> None:
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.10, 1.10)
    a, _, c = lattice.abc
    ax.set_aspect(c / a)  # preserve real cell proportions on screen
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(PROJ_LABELS[0], fontsize=9)
    ax.set_ylabel(PROJ_LABELS[1], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", labelsize=8)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(out_dir: Path) -> None:
    structure = load_sample(SAMPLE_PT, SAMPLE_INDEX)
    spa = SpacegroupAnalyzer(structure, symprec=0.01)
    sym_struct = spa.get_symmetrized_structure()
    sg_num = spa.get_space_group_number()
    sg_sym = spa.get_space_group_symbol()
    formula = structure.composition.reduced_formula
    sg_ops = SpaceGroup.from_int_number(sg_num).symmetry_ops

    asu_indices = [grp[0] for grp in sym_struct.equivalent_indices]
    asu_sites = [sym_struct[i] for i in asu_indices]
    asu_wyckoff = list(sym_struct.wyckoff_symbols)
    site_syms = spa.get_symmetry_dataset().get("site_symmetry_symbols", None)
    if site_syms is None:
        asu_site_syms = ["?"] * len(asu_indices)
    else:
        asu_site_syms = [site_syms[i] for i in asu_indices]

    # fold each asu representative INTO the conventional Pnma asu rectangle
    asu_frac_raw = np.array([s.frac_coords for s in asu_sites])
    asu_box, n_in_box = infer_asu_box(asu_frac_raw)
    asu_frac = np.array(
        [fold_into_box(f, sg_ops, asu_box) for f in asu_frac_raw]
    )
    asu_species = [s.specie.symbol for s in asu_sites]
    print(f"asu box {asu_box} contains {n_in_box}/{len(asu_frac_raw)} raw reps "
          f"as-is")

    # full structure (already replicated by SymmCD)
    all_species = [s.specie.symbol for s in sym_struct]
    all_frac = sym_struct.frac_coords % 1.0

    # orbit index for each atom in the full cell
    orbit_of = np.zeros(len(sym_struct), dtype=int)
    for orbit_i, grp in enumerate(sym_struct.equivalent_indices):
        for j in grp:
            orbit_of[j] = orbit_i
    orbit_palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    # ----- figure ---------------------------------------------------------
    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.06, right=0.985,
                          bottom=0.16, top=0.84)

    # --- (a) Asymmetric unit ---------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    draw_unit_cell(ax_a)
    draw_asu_box(ax_a, asu_box, alpha=0.55, label=True)
    for sp, frac, wp, ss in zip(
        asu_species, asu_frac, asu_wyckoff, asu_site_syms
    ):
        draw_atom(ax_a, frac, jmol(sp), size=240, lw=1.1, symbol=sp)
        x, y = project(frac)
        label = (f"{wp}, symm: {ss}\n"
                 f"({frac[0]:.2f}, {frac[1]:.2f}, {frac[2]:.2f})")
        ax_a.annotate(
            label, xy=(x, y), xytext=(10, 10),
            textcoords="offset points", fontsize=7, color="#222",
            bbox=dict(boxstyle="round,pad=0.22",
                      fc="white", ec="#999", lw=0.5, alpha=0.92),
        )
    style_axes(ax_a, "(a) Asymmetric unit\ncoords + species + site symm",
               structure.lattice)

    # --- (b) Site symmetries: asu atoms + their symmetry images ----------
    ax_b = fig.add_subplot(gs[0, 1])
    draw_unit_cell(ax_b)
    draw_asu_box(ax_b, asu_box, alpha=0.30, label=False)
    # dashed lines from each asu rep to its images, coloured by orbit
    for orbit_i, grp in enumerate(sym_struct.equivalent_indices):
        line_col = orbit_palette[orbit_i % len(orbit_palette)]
        rep_xy = project(asu_frac[orbit_i])
        for j in grp:
            img_xy = project(all_frac[j])
            ax_b.plot([rep_xy[0], img_xy[0]], [rep_xy[1], img_xy[1]],
                       color=line_col, lw=0.9, ls=(0, (4, 2)), alpha=0.55,
                       zorder=2)
    # symmetry-image atoms (Jmol fill, orbit colour as edge ring, faded)
    for orbit_i, grp in enumerate(sym_struct.equivalent_indices):
        edge_col = orbit_palette[orbit_i % len(orbit_palette)]
        for j in grp:
            draw_atom(ax_b, all_frac[j], jmol(all_species[j]), size=70,
                      lw=1.0, alpha=0.55, ec=edge_col)
    # asu representatives: bold Jmol fill, orbit-coloured ring
    for orbit_i, frac in enumerate(asu_frac):
        edge_col = orbit_palette[orbit_i % len(orbit_palette)]
        draw_atom(ax_b, frac, jmol(asu_species[orbit_i]), size=260,
                  lw=2.0, symbol=asu_species[orbit_i], ec=edge_col)
    style_axes(
        ax_b,
        f"(b) Site symmetries\nspace group #{sg_num} ({sg_sym}) ops applied",
        structure.lattice,
    )

    # --- (c) Replicated unit cell ----------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    draw_unit_cell(ax_c)
    draw_asu_box(ax_c, asu_box, alpha=0.18, label=False)
    for sp, frac in zip(all_species, all_frac):
        draw_atom(ax_c, frac, jmol(sp), size=180, lw=0.7)
    style_axes(ax_c, f"(c) Replicated unit cell\n{len(structure)} atoms total",
               structure.lattice)

    # --- legend ----------------------------------------------------------
    species_unique = sorted(set(all_species))
    handles = [
        Line2D([0], [0], marker="o", linestyle="", mfc=jmol(sp), mec="black",
               mew=0.8, ms=10, label=sp)
        for sp in species_unique
    ]
    handles += [
        Rectangle((0, 0), 1, 1, facecolor="#ffd166", edgecolor="#b8860b",
                  label="asymmetric unit"),
        Line2D([0], [0], color="#666", lw=1.2, ls=(0, (4, 2)),
               label="symmetry image link (panel b)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"SymmCD sample decomposition — {formula}  (SG #{sg_num} {sg_sym}) "
        f"— view down b axis",
        fontsize=12, y=0.96,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "symmcd_real_sample_decomp.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    # metadata CSV
    csv_path = out_dir / "symmcd_real_sample_decomp_metadata.csv"
    with csv_path.open("w") as f:
        f.write("key,value\n")
        f.write(f"sample_pt,{SAMPLE_PT}\n")
        f.write(f"sample_index,{SAMPLE_INDEX}\n")
        f.write(f"formula,{formula}\n")
        f.write(f"spacegroup_number,{sg_num}\n")
        f.write(f"spacegroup_symbol,{sg_sym}\n")
        f.write(f"n_atoms_total,{len(structure)}\n")
        f.write(f"n_asu_sites,{len(asu_sites)}\n")
        f.write(f"lengths,\"{structure.lattice.abc}\"\n")
        f.write(f"angles,\"{structure.lattice.angles}\"\n")
        for k, (sp, raw, folded, wp, ss) in enumerate(zip(
            asu_species, asu_frac_raw, asu_frac, asu_wyckoff, asu_site_syms
        )):
            f.write(f"asu_site_{k},\"{sp} {wp} symm={ss} "
                    f"raw=({raw[0]:.4f},{raw[1]:.4f},{raw[2]:.4f}) "
                    f"folded=({folded[0]:.4f},{folded[1]:.4f},{folded[2]:.4f})\"\n")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    make_figure(Path(__file__).resolve().parent)
