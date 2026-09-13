#!/usr/bin/env python3
"""
Regeneration script for Supplementary Figure: DFT-relaxed structures and DFPT
static dielectric tensors of first-principles-verified candidates.

Outputs
    sup_dfpt_verified_structures.png   (300 dpi raster)
    sup_dfpt_verified_structures.pdf   (vector)

Inputs (all read from disk; nothing about the figure is hardcoded except the
per-cell projection choices, which are cached search results -- see _PROJ)
    <BASE>/_summary.csv                       79 accepted candidates
    <FAIL>/_summary.csv                       24 rejected (negative control)
    <BASE>/*.cif, <FAIL>/*.cif                relaxed structures
    <REWARDS>/step_XXXX_tensor.npz            full 3x3 DFPT tensors
    sup_dfpt_table.csv                        gallery panel assignment (cells 1-9)
"""
import os, re, ast
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Circle
from matplotlib.offsetbox import DrawingArea, TextArea, HPacker, AnchoredOffsetbox
from ase.io import read
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from ase.build import make_supercell
import spglib

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = "/Users/angus/Downloads/dielectric_inplane_isotropy_gapgate_mprime_newbg_b96"
BASE    = f"{ROOT}/deliverables_dfpt_candidates"     # accepted candidates
FAIL    = f"{ROOT}/failed_reward_demo"               # rejected (negative control)
REWARDS = f"{ROOT}/rewards/tsenn_static_dielectric_layered_uniaxial"
TABLE   = "sup_dfpt_table.csv"                       # gallery panel assignment
OUT     = "sup_dfpt_verified_structures"

import re, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Circle
from matplotlib.offsetbox import DrawingArea, TextArea, HPacker, AnchoredOffsetbox
from matplotlib.lines import Line2D
from ase.io import read
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.data.colors import jmol_colors
from ase.build import make_supercell
import spglib

META_GREY = "#888888"

def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)

apply_figure_style(sizes=(8,7,6))
mpl.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','axes.linewidth':0.8})

# ── house helpers ─────────────────────────────────────────────────────────────
def col_of(Z): return jmol_colors[int(Z)]
def colors_for(nums): return np.array([col_of(z) for z in nums])
def bond_col_of(Z): return col_of(Z)

def bonds_from_cutoff(pos, nums, scale=1.25):
    n=len(pos); bl=[]; r=covalent_radii[nums]
    for i in range(n):
        d=np.linalg.norm(pos-pos[i],axis=1)
        for j in range(i+1,n):
            if 0.1<d[j]<scale*(r[i]+r[j]): bl.append((i,j))
    return bl

def cell_edges(atoms):
    C=np.array(atoms.cell); corners=[]
    for i in (0,1):
        for j in (0,1):
            for k in (0,1): corners.append(i*C[0]+j*C[1]+k*C[2])
    corners=np.array(corners); idx=lambda i,j,k:i*4+j*2+k; edges=[]
    for i in (0,1):
        for j in (0,1): edges.append((idx(i,j,0),idx(i,j,1)))
    for i in (0,1):
        for k in (0,1): edges.append((idx(i,0,k),idx(i,1,k)))
    for j in (0,1):
        for k in (0,1): edges.append((idx(0,j,k),idx(1,j,k)))
    return corners, edges

def glossy(ax,cx,cy,r,color,z=2):
    ax.add_patch(Circle((cx,cy),r,facecolor=color,edgecolor='#222',lw=0.6,zorder=z))
    core=np.clip(np.array(color)*1.22+0.06,0,1)
    ax.add_patch(Circle((cx-0.28*r,cy+0.28*r),0.60*r,facecolor=core,edgecolor='none',zorder=z+0.1,alpha=0.5))
    ax.add_patch(Circle((cx-0.34*r,cy+0.34*r),0.18*r,facecolor='white',edgecolor='none',zorder=z+0.2,alpha=0.9))

def view_basis(atoms,view='a',up='c'):
    C=np.array(atoms.cell); vmap={'a':C[0],'b':C[1],'c':C[2]}
    w=vmap[view]/np.linalg.norm(vmap[view])
    upv=vmap[up]-np.dot(vmap[up],w)*w; v=upv/np.linalg.norm(upv)
    u=np.cross(v,w); u=u/np.linalg.norm(u)
    return u,v,w

def rot2d(x,y,deg):
    t=np.radians(deg); return np.cos(t)*x-np.sin(t)*y, np.sin(t)*x+np.cos(t)*y

def render_proj4(ax,atoms,rep,view='b',up='c',rscale=0.42,bond_scale=1.25,lw=2.0,show_cell=True,inplane=0.0,dim=False):
    ax.set_facecolor('white')
    sc=atoms*rep; pos=sc.get_positions(); nums=sc.get_atomic_numbers()
    center=pos.mean(0); pos=pos-center; bl=bonds_from_cutoff(pos,nums,bond_scale)
    u,v,w=view_basis(atoms,view,up); x=pos@u; y=pos@v; z=pos@w
    x,y=rot2d(x,y,inplane)
    cols=colors_for(nums); bcols=np.array([bond_col_of(zz) for zz in nums])
    if dim:
        g=cols.mean(1,keepdims=True); cols=0.45*cols+0.55*np.repeat(g,3,1)*0+0.55*0.78
        bcols=0.45*bcols+0.55*0.78
    rad=covalent_radii[nums]*rscale; zspan=np.ptp(z)+1e-9; zmin=z.min()
    if show_cell:
        corners,edges=cell_edges(atoms); off=(np.array(rep)//2)@np.array(atoms.cell); cc=corners+off-center
        cx=cc@u; cy=cc@v; cx,cy=rot2d(cx,cy,inplane)
        for (i,j) in edges:
            ax.plot([cx[i],cx[j]],[cy[i],cy[j]],color='#555',lw=0.9,alpha=0.55,zorder=0.5)
    draw=[('bond',(z[i]+z[j])/2,i,j) for (i,j) in bl]+[('atom',z[i],i,None) for i in range(len(pos))]
    draw.sort(key=lambda t:t[1])
    for typ,zz,i,j in draw:
        if typ=='bond':
            mid=[(x[i]+x[j])/2,(y[i]+y[j])/2]
            ax.plot([x[i],mid[0]],[y[i],mid[1]],color=bcols[i],lw=lw,solid_capstyle='round',zorder=1)
            ax.plot([mid[0],x[j]],[mid[1],y[j]],color=bcols[j],lw=lw,solid_capstyle='round',zorder=1)
        else:
            depth=0.62+0.38*(z[i]-zmin)/zspan
            glossy(ax,x[i],y[i],rad[i],np.clip(np.array(cols[i])*depth,0,1))
    ax.set_aspect('equal'); ax.axis('off'); m=rad.max()+0.5
    ax.set_xlim(x.min()-m,x.max()+m); ax.set_ylim(y.min()-m,y.max()+m)

def hm_symbol(sym):
    toks=re.findall(r'[A-Za-z/]+|-\d|\d_\d|\d',sym); out=[]
    for t in toks:
        if re.match(r'[A-Za-z/]+$',t): out.append(r'\mathrm{%s}'%t)
        elif re.match(r'-\d$',t):      out.append(r'\overline{%s}'%t[1])
        elif re.match(r'\d_\d$',t):    out.append(r'%s_{%s}'%(t[0],t[2]))
        else:                          out.append(t)
    return '$'+''.join(out)+'$'

def fmt_formula(f): return re.sub(r'(\d+)',r'$_{\1}$',f)


# ── data loading ──────────────────────────────────────────────────────────────
df = pd.read_csv(f"{BASE}/_summary.csv")
fd = pd.read_csv(f"{FAIL}/_summary.csv").set_index("cif")

mism = lambda x, y: np.abs(x - y) / (x + y)
Azf  = lambda xx, yy, zz: np.abs(zz - (xx + yy) / 2) / (zz + (xx + yy) / 2)
df["mism"] = mism(df.eps_xx, df.eps_yy)
df["Az"]   = Azf(df.eps_xx, df.eps_yy, df.eps_zz)

def sysname(n):
    for hi, nm in [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
                   (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"),
                   (230, "cubic")]:
        if n <= hi:
            return nm
df["system"] = [sysname(n) for n in df.sg_no]

def sg_of(path, symprec=0.01):
    """Space group from spglib at the pipeline tolerance, not from the CSV."""
    a = read(path)
    ds = spglib.get_symmetry_dataset(
        (a.cell[:], a.get_scaled_positions(), a.get_atomic_numbers()), symprec=symprec)
    return ds.international, ds.number, a

SYS_COL = {"hexagonal": "#1f6fd0", "trigonal": "#b0338f", "tetragonal": "#2a8f5a",
           "orthorhombic": "#c25e00", "monoclinic": "#7a5cc7", "triclinic": "#8a8f98"}
MODE_STY = {"cubic": ("#d1495b", "fully isotropic"),
            "biaxial": ("#8c6d1f", "biaxial"),
            "metallic": ("#3f7d7a", "metallic")}

# Gallery membership and panel order come from the supplementary table.
tab = pd.read_csv(TABLE)
gal_rows = tab[(tab.set == "accepted") & (tab.gallery_panel.notna())].sort_values("gallery_panel")
PICKS = list(zip(gal_rows.cif, gal_rows.gallery_panel.astype(int)))

di = df.set_index("cif")
GAL = []
for cif, k in PICKS:
    r = di.loc[cif]
    s, n, _ = sg_of(f"{BASE}/{cif}")
    GAL.append(dict(cif=cif, idx=k, formula=r.formula, sg=s, sgno=int(n), system=r.system,
                    exx=r.eps_xx, eyy=r.eps_yy, ezz=r.eps_zz,
                    gap=r.band_gap_ev, eh=r.ehull_ev_atom * 1000,
                    Az=r.Az, mism=r["mism"], accent=SYS_COL[r.system]))

# Negative control: one structure per failure mode.
#   Tl3SnSe4  cubic     -> fully isotropic, A_z below the 0.01 floor
#   TeAs      Pbca      -> biaxial, in-plane mismatch 0.310 > 0.1
#   KTe       P4/mmm    -> passes both tensor criteria, fails the 0.3 eV gap floor
REJ_CIF = ["cubic_r00_Tl3SnSe4_SG217_Eg1.07_step2_i19.cif",
           "biaxial_r00_TeAs_SG61_Eg0.87_step23_i13.cif",
           "metallic_r07_KTe_SG131_Eg0.10_step29_i20.cif"]
REJ = []
for cif in REJ_CIF:
    r = fd.loc[cif]
    s, n, _ = sg_of(f"{FAIL}/{cif}")
    REJ.append(dict(cif=cif, formula=r.formula, sg=s, sgno=int(n), mode=r["mode"],
                    system=sysname(int(n)),
                    exx=r.eps_xx, eyy=r.eps_yy, ezz=r.eps_zz,
                    gap=r.gap_eV, eh=r.ehull * 1000,          # CSV is eV/atom -> meV/atom
                    Az=r.A_z, mism=r.inplane_mismatch,
                    accent=MODE_STY[r["mode"]][0]))

# Off-diagonal components are symmetry-forbidden except in the low-symmetry settings.
_RULE = {"hexagonal": "diagonal", "trigonal": "diagonal", "tetragonal": "diagonal",
         "orthorhombic": "diagonal", "cubic": "isotropic",
         "monoclinic": "one off-diag pair allowed", "triclinic": "all off-diag allowed"}
for g in GAL + REJ:
    g["offdiag_allowed"] = _RULE[g["system"]] not in ("diagonal", "isotropic")


def hex_ortho_transform(atoms, tol=2.0):
    """Recast a hexagonal/trigonal cell in its equivalent orthogonal (non-primitive)
    lattice so the tiled render reads as a rectangle rather than a rhombus."""
    a_len, b_len, c_len, alpha, beta, gamma = atoms.cell.cellpar()
    if (abs(gamma - 120) < tol and abs(a_len - b_len) < tol * 0.05 * a_len + 1e-2
            and abs(alpha - 90) < tol and abs(beta - 90) < tol):
        return make_supercell(atoms, np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]]))
    return None


# ── cached projection choices ─────────────────────────────────────────────────
# (formula, sg_no) -> view axis, up axis, supercell repeat, hex->ortho recast.
# These are the settled outputs of a per-cell legibility search (all 6 axis/up
# combinations were rendered and compared); cached so the figure regenerates
# identically without re-running it.
_PROJ = {
    ("OsO4",     173): dict(view="c", up="a", rep=(3, 2, 1), ortho_hex=True),
    ("TiPO3",    136): dict(view="c", up="a", rep=(2, 2, 1), ortho_hex=False),
    ("Cl2",      166): dict(view="c", up="b", rep=(4, 2, 1), ortho_hex=True),
    ("LiZrTeO6", 163): dict(view="c", up="b", rep=(2, 1, 1), ortho_hex=True),
    ("RbLiTeO4", 173): dict(view="c", up="b", rep=(2, 1, 1), ortho_hex=True),
    ("ZrTeO3",   194): dict(view="c", up="b", rep=(4, 2, 1), ortho_hex=True),
    ("Cl2",       29): dict(view="c", up="b", rep=(2, 2, 1), ortho_hex=False),
    ("TeO2",      19): dict(view="b", up="a", rep=(1, 1, 1), ortho_hex=False),
    ("CsNaWO4",    2): dict(view="c", up="a", rep=(1, 1, 1), ortho_hex=False),
    ("Tl3SnSe4", 217): dict(view="c", up="b", rep=(1, 1, 1), ortho_hex=False),
    ("TeAs",      61): dict(view="a", up="b", rep=(1, 1, 1), ortho_hex=False),
    ("KTe",      123): dict(view="c", up="b", rep=(5, 5, 1), ortho_hex=False),
}
for g in GAL + REJ:
    g.update(_PROJ[(g["formula"], g["sgno"])])


# ── full DFPT tensors ─────────────────────────────────────────────────────────
def full_tensor(step, index):
    """Symmetrised full 3x3 tensor from the reward archive at (step, index)."""
    d = np.load(f"{REWARDS}/step_{int(step):04d}_tensor.npz")
    t = d["tensor"][int(index)]
    return (t + t.T) / 2

for g in GAL:
    r = di.loc[g["cif"]]
    g["tensor"] = full_tensor(r.step, r["index"])
for g in REJ:
    r = fd.loc[g["cif"]]
    g["tensor"] = full_tensor(r.step, r["index"])

# The archive tensor is the source of truth; assert its diagonal against the CSV.
for g in GAL + REJ:
    assert np.allclose(np.diag(g["tensor"]), [g["exx"], g["eyy"], g["ezz"]], atol=0.01), \
        f"tensor diagonal disagrees with summary for {g['formula']}"


# ── drawing ───────────────────────────────────────────────────────────────────
from matplotlib.offsetbox import DrawingArea, TextArea, HPacker, AnchoredOffsetbox
from matplotlib.patches import Circle

def chip_row2(ax, zs, fs=7.5, dim=False, axis_label=None):
    """Legend row: optional axis-view glyph (e.g. r'$\\odot\\,c$') first, then
    element chips. All rows anchor flush-left at the same x across panels."""
    ax.axis('off'); r=5.2; boxes=[]
    if axis_label is not None:
        boxes.append(TextArea(axis_label, textprops=dict(fontsize=fs+2.0, color='#222')))
    for Z in zs:
        col=np.array(col_of(Z))
        if dim: col=0.45*col+0.55*0.78
        da=DrawingArea(2*r+2,2*r+2,0,0)
        da.add_artist(Circle((r+1,r+1),r,facecolor=col,edgecolor='#222',lw=0.7))
        core=np.clip(col*1.22+0.06,0,1)
        da.add_artist(Circle((r+1-0.30*r,r+1+0.30*r),0.42*r,facecolor=core,edgecolor='none',alpha=0.5))
        da.add_artist(Circle((r+1-0.40*r,r+1+0.40*r),0.16*r,facecolor='white',edgecolor='none',alpha=0.9))
        txt=TextArea(chemical_symbols[Z],textprops=dict(fontsize=fs,color='#111'))
        boxes.append(HPacker(children=[da,txt],align='center',pad=0,sep=2))
    row=HPacker(children=boxes,align='center',pad=0,sep=7)
    ab=AnchoredOffsetbox(loc='center left',child=row,frameon=False,pad=0,
                         bbox_to_anchor=(0.06,0.5),bbox_transform=ax.transAxes)
    ax.add_artist(ab)


from ase.data import covalent_radii as _cr

def content_aspect(atoms, rep, view, up, rscale=0.60):
    """Projected width/height of what render_proj4 will draw (same math, no drawing)."""
    sc = atoms*rep; pos = sc.get_positions(); nums = sc.get_atomic_numbers()
    pos = pos - pos.mean(0)
    u,v,w = view_basis(atoms, view, up)
    x = pos@u; y = pos@v
    rad = _cr[nums]*rscale; m = rad.max()+0.5
    W = (x.max()+m)-(x.min()-m); H = (y.max()+m)-(y.min()-m)
    return W/H


def eps_matrix3x3(ax, tensor, x, y, fs=6.0, col='#111', w=0.72, h=0.86, lw=0.85,
                  zero_tol=1e-6, col_frac=0.32, label_gap=0.055, anchor_bottom=None):
    """3x3 dielectric tensor bracket display.

    Geometry is corrected so the bracket renders visually square on the page
    regardless of the host axes' aspect ratio. If `anchor_bottom` (in figure
    fraction) is given, the bracket centre is placed that fixed distance above
    the axes' bottom edge instead of at the supplied `y` fraction -- this keeps
    the bracket-to-text spacing identical across panel (a) and the taller
    panel (b) rows, which otherwise open an empty band beneath the bracket.
    """
    fig = ax.figure
    bbox = ax.get_position()
    figw, figh = fig.get_size_inches()
    ax_aspect = (bbox.width*figw) / (bbox.height*figh)
    h_corr = h * ax_aspect

    if anchor_bottom is not None:
        y = anchor_bottom / bbox.height

    T = tensor
    ax.text(x-w/2-label_gap,y,r'$\varepsilon_0=$',transform=ax.transAxes,fontsize=fs+0.5,
            ha='right',va='center',color=col,zorder=25)
    x0,x1=x-w/2,x+w/2; y0,y1=y-h_corr/2,y+h_corr/2; tk=0.030
    for xb,s in ((x0,+1),(x1,-1)):
        ax.plot([xb,xb],[y0,y1],color=col,lw=lw,transform=ax.transAxes,zorder=22,clip_on=False)
        ax.plot([xb,xb+s*tk],[y1,y1],color=col,lw=lw,transform=ax.transAxes,zorder=22,clip_on=False)
        ax.plot([xb,xb+s*tk],[y0,y0],color=col,lw=lw,transform=ax.transAxes,zorder=22,clip_on=False)
    cols=[x-w*col_frac, x, x+w*col_frac]; rows=[y+h_corr*0.30, y, y-h_corr*0.30]
    for i,ry in enumerate(rows):
        for j,cx in enumerate(cols):
            v = T[i,j]
            if abs(v) < zero_tol:
                txt = '0'
            else:
                rv = round(v,2)
                txt = '0.00' if rv == 0 else ('%.2f'%rv)
            c = col if i==j else '#666'
            fsz = fs if i==j else fs-0.4
            ax.text(cx,ry,txt,transform=ax.transAxes,fontsize=fsz,fontweight='normal',
                     ha='center',va='center',color=c,zorder=25,
                     path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])


def rejected_group_frame(fig, rej_axes, pad=0.010):
    """Single neutral dashed frame around all rejected cells -- replaces the
    per-cell colour-coded dashed boxes (cell_frame(), pre-v6)."""
    b1 = rej_axes[0][0].get_position(); b2 = rej_axes[-1][1].get_position()
    x0 = b1.x0-pad; x1 = b2.x1+pad
    y0 = min(ax.get_position().y0 for _,ax in rej_axes)-pad*1.3
    y1 = max(at.get_position().y1 for at,_ in rej_axes)+pad*0.55
    fig.patches.append(Rectangle((x0,y0),x1-x0,y1-y0,transform=fig.transFigure,
        facecolor='none',edgecolor='#666',lw=1.0,linestyle=(0,(3.2,2.2)),zorder=0.2))
    return x0,y0,x1,y1


BRACKET_ANCHOR = 0.0395   # figure fraction: bracket centre above axes bottom (panel-a value)
TEXT_ANCHOR    = 0.0245   # figure fraction: Eg/Ehull text centre below axes top

def draw_cell_v7(fig, ss, g, rejected=False):
    inner = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=ss,
                height_ratios=[0.135,1.0,0.155], hspace=0.03)
    axt = fig.add_subplot(inner[0]); axt.set_axis_off()
    axc = fig.add_subplot(inner[2]); axc.set_axis_off()
    main = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=inner[1],
                width_ratios=[0.98,0.80], wspace=0.02)
    axr = fig.add_subplot(main[0]); axr.set_axis_off()
    right = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=main[1],
                height_ratios=[0.62,0.38], hspace=0.10)
    axe = fig.add_subplot(right[0]); axe.set_axis_off()
    axp = fig.add_subplot(right[1]); axp.set_axis_off()

    atoms = read((FAIL if rejected else BASE)+"/"+g["cif"])
    if g.get("ortho_hex"):
        atoms = hex_ortho_transform(atoms)

    num = r'$\bf{%d}$  '%g["idx"] if not rejected else ''
    axt.text(0.0,0.5, num+fmt_formula(g["formula"])+'  '+hm_symbol(g["sg"])+' ('+str(g["sgno"])+')',
             transform=axt.transAxes, fontsize=7.6, ha='left', va='center', color='#222')

    render_proj4(axr, atoms, g["rep"], view=g["view"], up=g["up"], rscale=0.60, lw=2.3)
    if rejected:
        axr.set_facecolor(g["accent"]); axr.patch.set_alpha(0.045)
    u,v,w = view_basis(atoms, g["view"], g["up"]); C = np.array(atoms.cell)
    vec = C[{'a':0,'b':1,'c':2}[g["view"]]]; vn = vec/np.linalg.norm(vec)
    symb = r'\odot' if (vn@w)>=0 else r'\otimes'
    axis_lbl = r'$%s\,%s$'%(symb, g["view"])

    # Anchor bracket and text at fixed figure-space offsets so the vertical gap
    # between them is identical in panel (a) and the taller panel (b) rows.
    eps_matrix3x3(axe, g["tensor"], 0.60, 0.50, fs=6.0, w=0.72, h=0.86,
                  anchor_bottom=BRACKET_ANCHOR)

    info = (r'$E_g$ = %.2f eV'%g["gap"]+'\n'+r'$E_\mathrm{hull}$ = %+.0f meV/atom'%g["eh"])
    pb = axp.get_position()
    y_txt = 1.0 - (TEXT_ANCHOR / pb.height)
    axp.text(0.5, y_txt, info, transform=axp.transAxes, fontsize=6.2, ha='center', va='center',
              color='#333', linespacing=1.55)

    zs = sorted(set(atoms.get_atomic_numbers()), key=lambda z:-covalent_radii[z])
    chip_row2(axc, zs, fs=6.5, axis_label=axis_lbl)
    return axr,(axt,axc)


# ── figure assembly ───────────────────────────────────────────────────────────
# B_RATIO sizes panel (b) to what its renders need (they are width-limited by
# equal aspect); FIG_H is solved so panel (a)'s cell height is unchanged.
B_RATIO = 0.70
FIG_H   = 8.684

def build(path, dpi=None, b_ratio=B_RATIO, figh=FIG_H):
    fig = plt.figure(figsize=(7.6, figh))
    outer = gridspec.GridSpec(2, 1, height_ratios=[2.62, b_ratio], hspace=0.10,
                              left=0.045, right=0.978, top=0.975, bottom=0.040)
    gsa = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[0], hspace=0.30, wspace=0.10)
    tops = []
    for i, g in enumerate(GAL):
        axr, (axt, axc) = draw_cell_v7(fig, gsa[i // 3, i % 3], g)
        tops.append(axt)
    gsc = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.10)
    rej_axes = []
    for i, g in enumerate(REJ):
        axr, (axt, axc) = draw_cell_v7(fig, gsc[i], g, rejected=True)
        rej_axes.append((axt, axc))
    fig.canvas.draw()
    x0, y0, x1, y1 = rejected_group_frame(fig, rej_axes)
    ba = tops[0].get_position()
    fig.text(0.008, ba.y1 + 0.006, 'a', fontsize=10.0, fontweight='bold', ha='left', va='bottom')
    fig.text(x0 - 0.006, y1 + 0.010, 'b', fontsize=10.0, fontweight='bold', ha='left', va='bottom')
    fig.savefig(path, facecolor='white', **({'dpi': dpi} if dpi else {}))
    return fig


if __name__ == "__main__":
    apply_figure_style()
    f = build(f"{OUT}.png", dpi=300); plt.close(f)
    f = build(f"{OUT}.pdf");          plt.close(f)
    print(f"wrote {OUT}.png ({os.path.getsize(OUT + '.png')} B), "
          f"{OUT}.pdf ({os.path.getsize(OUT + '.pdf')} B)")
