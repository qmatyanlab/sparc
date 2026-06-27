"""SymmCD Figure 1 style schematic.

Reproduces the pedagogical decomposition of a crystal into:
  (1) an asymmetric unit (smallest region that tiles the cell under symmetry),
  (2) site symmetries (which symmetry ops fix each atom),
  (3) fractional coordinates of asymmetric-unit atoms,
  (4) atomic species (colour-coded).

A 2D wallpaper group p4mm cell is drawn with three fake species.  The
asymmetric unit is the triangle (0,0)-(1/2,0)-(1/2,1/2).  Three unique atoms
populate it: one on the 4mm axis (corner), one on a diagonal mirror, one in
general position.  Three panels show the asymmetric unit, the symmetry
operators, and the fully replicated unit cell.

Run:
    source .venv/bin/activate
    python paper_materials/symmcd_figure1_schematic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle


# ---------------------------------------------------------------------------
# Crystal data: wallpaper group p4mm with three fake species in the asym unit
# ---------------------------------------------------------------------------

# Asymmetric unit triangle: corners (0,0), (1/2, 0), (1/2, 1/2)
ASU_VERTS = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])

# (label, fractional x, fractional y, site symmetry, colour)
ASU_ATOMS = [
    ("X", 0.001, 0.001, "4mm", "#d62728"),  # 4-fold + mirrors
    ("Y", 0.32, 0.32, "..m", "#1f77b4"),  # diagonal mirror plane
    ("Z", 0.40, 0.15, "1",   "#2ca02c"),  # general position
]

# Generators of p4mm (8 operations: identity, 3 rotations, 4 mirrors).  Each
# is a 2x2 matrix acting on fractional coords (the cell is square so these are
# orthogonal).
P4MM_OPS = [
    np.array([[ 1, 0], [ 0, 1]]),   # E
    np.array([[ 0,-1], [ 1, 0]]),   # C4
    np.array([[-1, 0], [ 0,-1]]),   # C2
    np.array([[ 0, 1], [-1, 0]]),   # C4^-1
    np.array([[-1, 0], [ 0, 1]]),   # mx
    np.array([[ 1, 0], [ 0,-1]]),   # my
    np.array([[ 0, 1], [ 1, 0]]),   # md (y=x)
    np.array([[ 0,-1], [-1, 0]]),   # md (y=-x)
]


def replicate(x: float, y: float) -> list[tuple[float, float]]:
    """Apply all p4mm ops + lattice translations into the [0,1)^2 cell."""
    pts: list[tuple[float, float]] = []
    v = np.array([x, y])
    for op in P4MM_OPS:
        p = op @ v
        # wrap into [0, 1)
        p = np.mod(p, 1.0)
        pts.append((float(p[0]), float(p[1])))
    # de-duplicate (atoms on special positions land on themselves)
    uniq: list[tuple[float, float]] = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in uniq):
            uniq.append(p)
    return uniq


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_unit_cell(ax: plt.Axes) -> None:
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", lw=1.4))


def draw_asu(ax: plt.Axes, alpha: float = 0.18, label: bool = True) -> None:
    poly = Polygon(ASU_VERTS, closed=True, facecolor="#ffd166",
                   edgecolor="#b8860b", lw=1.2, alpha=alpha, zorder=0)
    ax.add_patch(poly)
    if label:
        ax.annotate(
            "asymmetric\nunit",
            xy=(0.25, 0.25), xytext=(0.5, 0.8),
            fontsize=8, ha="center", va="center",
            color="#7a5300", style="italic",
            arrowprops=dict(arrowstyle="->", color="#7a5300", lw=1.0,
                            connectionstyle="arc3,rad=0.45",
                            shrinkA=10, shrinkB=4),
        )


def draw_symmetry_ops(ax: plt.Axes) -> None:
    """Mirror lines + rotation centres for p4mm."""
    mirror_kw = dict(color="#8b008b", lw=1.0, ls=(0, (4, 2)), alpha=0.75)
    # horizontal & vertical mirrors at 0, 1/2
    for c in (0.0, 0.5, 1.0):
        ax.plot([0, 1], [c, c], **mirror_kw)
        ax.plot([c, c], [0, 1], **mirror_kw)
    # diagonal mirrors
    ax.plot([0, 1], [0, 1], **mirror_kw)
    ax.plot([0, 1], [1, 0], **mirror_kw)

    # 4-fold rotation centres (corners + centre): filled squares
    for cx, cy in [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]:
        ax.plot(cx, cy, marker="s", ms=8, mfc="#8b008b", mec="black",
                mew=0.8, zorder=4)
    # 2-fold rotation centres (edge midpoints): filled ellipses
    for cx, cy in [(0.5, 0.0), (0.0, 0.5), (1.0, 0.5), (0.5, 1.0)]:
        ax.plot(cx, cy, marker="o", ms=7, mfc="#8b008b", mec="black",
                mew=0.8, zorder=4)


def draw_atoms(ax: plt.Axes, atoms: list[tuple[str, float, float, str, str]],
               only_asu: bool = False, label_coords: bool = False,
               label_sym: bool = False) -> None:
    for label, x, y, sym, colour in atoms:
        if only_asu:
            ax.plot(x, y, "o", ms=14, mfc=colour, mec="black", mew=1.0,
                    zorder=5)
            ax.text(x, y, label, ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=6)
            if label_coords or label_sym:
                lines = []
                if label_coords:
                    lines.append(f"({x:.2f}, {y:.2f})")
                if label_sym:
                    lines.append(f"symm: {sym}")
                ax.annotate(
                    "\n".join(lines), xy=(x, y),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7, color="#222",
                    bbox=dict(boxstyle="round,pad=0.18",
                              fc="white", ec="#999", lw=0.5, alpha=0.92),
                )
        else:
            for px, py in replicate(x, y):
                ax.plot(px, py, "o", ms=11, mfc=colour, mec="black",
                        mew=0.8, zorder=5)


def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, 1.12)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("x (fractional)")
    ax.set_ylabel("y (fractional)")
    ax.set_title(title, fontsize=11)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(out_path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 5.0))
    gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.06, right=0.985,
                          bottom=0.13, top=0.86)

    # --- (a) Asymmetric unit + coords + species + site symmetries ----------
    ax_a = fig.add_subplot(gs[0, 0])
    draw_unit_cell(ax_a)
    draw_asu(ax_a, alpha=0.45)
    draw_atoms(ax_a, ASU_ATOMS, only_asu=True, label_coords=True,
               label_sym=True)
    style_axes(ax_a, "(a) Asymmetric unit\ncoords + species + site symm")

    # --- (b) Symmetry operations of the space group ------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    draw_unit_cell(ax_b)
    draw_asu(ax_b, alpha=0.25, label=False)
    draw_symmetry_ops(ax_b)
    draw_atoms(ax_b, ASU_ATOMS, only_asu=True)
    style_axes(ax_b, "(b) Site symmetries\nmirror lines + rotation centres")

    # --- (c) Full crystal: asymmetric unit replicated by symmetry ----------
    ax_c = fig.add_subplot(gs[0, 2])
    draw_unit_cell(ax_c)
    draw_asu(ax_c, alpha=0.18, label=False)
    draw_atoms(ax_c, ASU_ATOMS, only_asu=False)
    style_axes(ax_c, "(c) Replicated unit cell\n(asym unit $\\times$ space-group ops)")

    # arrows between panels: anchor to right edge of each source axis so they
    # survive bbox_inches="tight"
    for src in (ax_a, ax_b):
        src.annotate("", xy=(1.30, 0.5), xytext=(1.13, 0.5),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="black",
                                     lw=1.6, mutation_scale=20),
                     annotation_clip=False)

    # legend (species + symmetry markers)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", mfc=c, mec="black",
               mew=0.8, ms=9, label=f"species {l}")
        for (l, _, _, _, c) in ASU_ATOMS
    ]
    legend_handles += [
        Line2D([0], [0], marker="s", linestyle="", mfc="#8b008b", mec="black",
               mew=0.8, ms=9, label="4-fold rotation"),
        Line2D([0], [0], marker="o", linestyle="", mfc="#8b008b", mec="black",
               mew=0.8, ms=8, label="2-fold rotation"),
        Line2D([0], [0], color="#8b008b", lw=1.2, ls=(0, (4, 2)),
               label="mirror plane"),
        mpatches.Patch(facecolor="#ffd166", edgecolor="#b8860b",
                       label="asymmetric unit"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("SymmCD decomposition of a crystal "
                 "(2D toy: wallpaper group p4mm, fake species X / Y / Z)",
                 fontsize=12, y=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    make_figure(here / "symmcd_figure1_schematic.png")
