"""SPARC Figure-1 decomposition for an actual sample, 3-D rendering.

Sister script to ``SPARC_real_sample_decomp.py`` (which does the 2-D
projection down the b axis).  This one keeps the full 3-D view.

    (a) Asymmetric unit       coords + species + site symm (Wyckoff)
    (b) Site symmetries       asu atoms + all symmetry images linked
    (c) Replicated unit cell  full crystal in the conventional cell

Selected sample: TbPr2Ge (Pnma, SG 62), 3 asymmetric-unit sites → 16 atoms,
loaded from the model's eval split.

Differences vs. the early 3-D version:
  * fractional-coord axes (0..1) — cell shape preserved by box aspect
  * grid + panes removed
  * conventional Pnma asymmetric unit shaded as a translucent yellow cuboid
  * asu representatives are folded into the conventional asu

Run:
    source .venv/bin/activate
    python paper_materials/SPARC_real_sample_decomp_3d.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.vis.structure_vtk import EL_COLORS
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from matplotlib.patches import Circle

SAMPLE_PT = Path(
    "/global/cfs/cdirs/m2663/angush/sparc/exp_res/"
    "bandgap3_symmcd_sgtemp3_test_v12/samples/step_0010_eval.pt"
)
SAMPLE_INDEX = 7  # TbPr2Ge, Pnma

# Pnma asu has dimensions (1/2, 1/4, 1) and 8 equivalent placements in the
# unit cell.  Rather than forcing every pymatgen-rep into the ITA-conventional
# box at the origin, we pick whichever placement best fits the reps as-is,
# anchoring on the atoms the model produced.
ASU_DIMS = (0.5, 0.25, 1.0)
ASU_X_OFFSETS = (0.0, 0.5)
ASU_Y_OFFSETS = (0.0, 0.25, 0.5, 0.75)
ASU_Z_OFFSETS = (0.0,)


def format_chemical_formula(formula):
    if not formula:
        return formula
    result = ''
    i = 0
    while i < len(formula):
        char = formula[i]
        if char.isalpha():
            element = char
            i += 1
            if i < len(formula) and formula[i].islower():
                element += formula[i]
                i += 1
            result += element
            number = ''
            while i < len(formula) and formula[i].isdigit():
                number += formula[i]
                i += 1
            if number:
                result += f'_{{{number}}}'
            if i < len(formula) and formula[i] == '^':
                i += 1
                superscript = ''
                while i < len(formula) and (formula[i].isdigit() or formula[i] in ['+', '-']):
                    superscript += formula[i]
                    i += 1
                if superscript:
                    result += f'^{{{superscript}}}'
        else:
            result += char
            i += 1
    return result
    
def add_ab_corner_indicator(fig):
    # ax_dir = fig.add_axes([0.03, 0.06, 0.14, 0.16])
    ax_dir = fig.add_axes([0.00, 0.09, 0.18, 0.18])
    ax_dir.set_xlim(0, 1)
    ax_dir.set_ylim(0, 1)
    ax_dir.set_aspect("equal", adjustable="box")
    ax_dir.axis("off")

    origin = (0.18, 0.18)

    # a-axis arrow
    ax_dir.annotate(
        "", xy=(0.82, 0.18), xytext=origin,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black")
    )
    ax_dir.text(0.86, 0.18, "a", fontsize=16, ha="left", va="center")

    # b-axis arrow
    ax_dir.annotate(
        "", xy=(0.18, 0.82), xytext=origin,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black")
    )
    ax_dir.text(0.18, 0.88, "b", fontsize=16, ha="center", va="bottom")

    # c out of plane: circle with dot
    c_center = (0.18, 0.18)
    ax_dir.add_patch(Circle(c_center, 0.08, fill=False, ec="black", lw=1.0))
    ax_dir.plot(c_center[0], c_center[1], marker="o", ms=5, color="black")
    ax_dir.text(c_center[0] + 0.10, c_center[1] - 0.05, "c", fontsize=16,
                ha="left", va="top")
    
class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        self.xyz = xyz
        super().__init__(text, xy=(0, 0), *args, **kwargs)

    def draw(self, renderer):
        x, y, z = self.xyz
        x2, y2, _ = proj3d.proj_transform(x, y, z, self.axes.get_proj())
        self.xy = (x2, y2)
        super().draw(renderer)


def annotate3D(ax, text, xyz, xytext=(20, 20), **kwargs):
    ann = Annotation3D(
        text,
        xyz=xyz,
        xytext=xytext,
        textcoords="offset points",
        **kwargs,
    )
    ax.add_artist(ann)
    return ann

def add_ab_corner_legend(fig):
    # [left, bottom, width, height] in figure coordinates
    ax_dir = fig.add_axes([0.02, 0.05, 0.10, 0.10])
    ax_dir.set_xlim(0, 1)
    ax_dir.set_ylim(0, 1)
    ax_dir.axis("off")

    origin = (0.22, 0.22)

    # → a
    ax_dir.annotate(
        "", xy=(0.88, 0.22), xytext=origin,
        arrowprops=dict(arrowstyle="->", lw=1.6, color="black")
    )
    ax_dir.text(0.91, 0.22, "a", fontsize=11, fontweight="bold",
                ha="left", va="center")

    # ↑ b
    ax_dir.annotate(
        "", xy=(0.22, 0.88), xytext=origin,
        arrowprops=dict(arrowstyle="->", lw=1.6, color="black")
    )
    ax_dir.text(0.22, 0.91, "b", fontsize=11, fontweight="bold",
                ha="center", va="bottom")

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
    """Pick the asu placement that contains the most raw reps as-is."""
    best, best_score = None, -1
    for box in candidate_boxes():
        score = sum(in_box(np.mod(p, 1.0), box, tol) for p in raw_reps)
        if score > best_score:
            best, best_score = box, score
    return best, best_score


def fold_into_box(frac: np.ndarray, sym_ops, box, tol: float = 0.02
                   ) -> np.ndarray:
    """Find a symmetry-equivalent position of `frac` inside `box`."""
    centre = np.array([0.5 * (b[0] + b[1]) for b in box])
    cands: list[np.ndarray] = []
    # try lattice translations of the raw point first (no symmetry op),
    # then symmetry images
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
# Sample loading
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
# 3-D drawing helpers (axes are fractional coords, 0..1)
# ---------------------------------------------------------------------------


def draw_unit_cell(ax, lw: float = 1.2, colour: str = "black",
                   alpha: float = 0.7) -> None:
    v = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (2, 4), (2, 6),
        (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
    ]
    segs = [[v[i], v[j]] for i, j in edges]
    ax.add_collection3d(Line3DCollection(segs, colors=colour, lw=lw, linestyles="-.",
                                          alpha=alpha))


def draw_asu_box(ax, box, face="#ffd166", edge="#b8860b",
                 alpha: float = 0.1) -> None:
    (x0, x1), (y0, y1), (z0, z1) = box
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[0], v[3], v[7], v[4]],
    ]
    pc = Poly3DCollection(faces, facecolors=face, edgecolors=edge,
                           lw=0.8, alpha=alpha)
    ax.add_collection3d(pc)


def scatter(ax, frac: np.ndarray, colours, sizes=140, lw=0.7, alpha=1.0,
             ec="black") -> None:
    ax.scatter(frac[:, 0], frac[:, 1], frac[:, 2], c=colours, s=sizes,
               edgecolors=ec, linewidths=lw, alpha=alpha, depthshade=False)


def style_3d(ax, lattice: Lattice, title: str) -> None:
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(-0.05, 1.05)
    a, b, c = lattice.abc
    try:
        ax.set_box_aspect((a, b, c))
    except Exception:
        pass
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    # ax.set_zticks([0.0, 0.5, 1.0])
    ax.set_zticks([])  # z tick labels often overlap, so hide them
    # ax.set_xlabel("x (frac)", fontsize=9, labelpad=-2)
    # ax.set_ylabel("y (frac)", fontsize=9, labelpad=-2)
    # ax.set_zlabel("z (frac)", fontsize=9, labelpad=-4)
    # ax.tick_params(axis="both", labelsize=7, pad=-1)
    ax.set_title(title, fontsize=11, pad=4)
    ax.view_init(elev=20, azim=-62)
    # remove grid + panes
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)


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

    asu_frac_raw = np.array([s.frac_coords for s in asu_sites])
    asu_box, n_in_box = infer_asu_box(asu_frac_raw)
    asu_frac = np.array(
        [fold_into_box(f, sg_ops, asu_box) for f in asu_frac_raw]
    )
    asu_species = [s.specie.symbol for s in asu_sites]
    print(f"asu box {asu_box} contains {n_in_box}/{len(asu_frac_raw)} raw reps "
          f"as-is")

    all_species = [s.specie.symbol for s in sym_struct]
    all_frac = sym_struct.frac_coords % 1.0

    orbit_palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig = plt.figure(figsize=(13.5, 6.8))
    gs = fig.add_gridspec(1, 3, wspace=0.00, left=0.02, right=0.98,
                          bottom=0.12, top=0.86)

    # --- (a) Asymmetric unit ---------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    draw_unit_cell(ax_a)
    draw_asu_box(ax_a, asu_box, alpha=0.1)
    asu_colours = [jmol(sp) for sp in asu_species]
    scatter(ax_a, asu_frac, asu_colours, sizes=240, lw=1.0)
    # manual screen-space offsets so labels sit outside the asymmetric unit
    label_specs = {
        ("Tb", "4c"): dict(xytext=(-45, 35)),
        ("Pr", "8d"): dict(xytext=(5, 55)),
        ("Ge", "4c"): dict(xytext=(-25, -30)),
    }
    for frac, sp, wp, ss in zip(asu_frac, asu_species, asu_wyckoff,
                                asu_site_syms):
        label = (f"{sp} {wp}\n"
                f"({frac[0]:.2f}, {frac[1]:.2f}, {frac[2]:.2f})\n"
                f"symm: {ss}")
        spec = label_specs.get((sp, wp), dict(xytext=(30, 30)))

        annotate3D(
            ax_a,
            label,
            xyz=(frac[0], frac[1], frac[2]),
            xytext=spec["xytext"],
            fontsize=7.5,
            ha="center",
            va="center",
            multialignment="center",
            color="#222",
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="white",
                ec="#999",
                lw=0.5,
                alpha=0.95,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#555",
                lw=0.9,
                shrinkA=2,
                shrinkB=4,
            ),
            annotation_clip=False,
        )
    # --- label-format legend for panel (a) -------------------------------
    label_legend_text = (
        "Atom label format:\n"
        "Species Wyckoff\n"
        "(x, y, z) fractional\n"
        "Site symmetry"
    )

    fig.text(
        0.95, 0.5,
        label_legend_text,
        transform=ax_a.transAxes,
        fontsize=8.0,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="#999",
            linewidth=0.6,
            alpha=0.92,
        ),
    )
    style_3d(ax_a, structure.lattice,
              f"(a) Asymmetric unit\n{len(asu_sites)} unique sites")

    # --- (b) Site symmetries ---------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    draw_unit_cell(ax_b)
    draw_asu_box(ax_b, asu_box, alpha=0.1)
    # links from each asu rep to its symmetry images
    seg_list = []
    seg_colours = []
    for orbit_i, grp in enumerate(sym_struct.equivalent_indices):
        col = orbit_palette[orbit_i % len(orbit_palette)]
        rep = asu_frac[orbit_i]
        for j in grp:
            img = all_frac[j]
            seg_list.append([rep, img])
            seg_colours.append(col)
    ax_b.add_collection3d(Line3DCollection(
        seg_list, colors=seg_colours, lw=1.0, alpha=0.55,
        linestyles=(0, (4, 2))))
    # image atoms (Jmol fill, orbit-coloured edge ring, faded)
    for orbit_i, grp in enumerate(sym_struct.equivalent_indices):
        edge_col = orbit_palette[orbit_i % len(orbit_palette)]
        img_frac = np.array([all_frac[j] for j in grp])
        img_cols = [jmol(all_species[j]) for j in grp]
        scatter(ax_b, img_frac, img_cols, sizes=80, lw=1.1, alpha=0.6,
                ec=edge_col)
    # asu reps highlighted
    for orbit_i, frac in enumerate(asu_frac):
        edge_col = orbit_palette[orbit_i % len(orbit_palette)]
        scatter(ax_b, frac[None, :], [jmol(asu_species[orbit_i])],
                 sizes=260, lw=2.0, ec=edge_col)
    style_3d(ax_b, structure.lattice,
              f"(b) Site symmetries\nspace group #{sg_num} ({sg_sym}) "
              f"ops applied")

    # --- (c) Replicated unit cell ----------------------------------------
    ax_c = fig.add_subplot(gs[0, 2], projection="3d")
    draw_unit_cell(ax_c)
    draw_asu_box(ax_c, asu_box, alpha=0.1)
    all_colours = [jmol(sp) for sp in all_species]
    scatter(ax_c, all_frac, all_colours, sizes=180, lw=0.7)
    style_3d(ax_c, structure.lattice,
              f"(c) Replicated unit cell\n{len(structure)} atoms total")
    # --- Set same camera/view angle for all panels ------------------------
    for ax in [ax_a, ax_b, ax_c]:
        ax.view_init(elev=90, azim=270)

    # --- legend ----------------------------------------------------------
    species_unique = sorted(set(all_species))
    handles = [
        Line2D([0], [0], marker="o", linestyle="", mfc=jmol(sp), mec="black",
               mew=0.8, ms=10, label=sp)
        for sp in species_unique
    ]
    handles += [
        Patch(facecolor="#ffd166", edgecolor="#b8860b",
              label="asymmetric unit"),
        Line2D([0], [0], color="#666", lw=1.2, ls=(0, (4, 2)),
               label="symmetry image link (panel b)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.1))

    fig.suptitle(
        rf"SPARC sample decomposition — ${format_chemical_formula(formula)}$ "
        rf"(SG #{sg_num} {sg_sym})",
        fontsize=16, y=0.96,
    )
    add_ab_corner_indicator(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sparc_real_sample_decomp_3d.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    csv_path = out_dir / "sparc_real_sample_decomp_3d_metadata.csv"
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
