#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch, Circle as MCircle
from matplotlib.offsetbox import DrawingArea, TextArea, HPacker, AnchoredOffsetbox
from ase.io import read
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors

plt.style.use('./publication.mplstyle')

# --------------------------------------------------------------------------
# Per-crystal metadata (space groups from spglib at symprec = 0.05 A)
# --------------------------------------------------------------------------
meta = {
    'P': dict(name='Blue phosphorene', formula='P$_6$', sg='P3$_1$21', sgno=152, accent='#1f6fd0'),
    'CdHgI': dict(name='Cd(HgI$_3$)$_2$', formula='Cd(HgI$_3$)$_2$', sg='P$\\bar{3}$1m', sgno=162, accent='#b0338f'),
    'SnS': dict(name='SnS', formula='Sn$_3$S$_3$', sg='P3$_1$', sgno=144, accent='#5a8f2a'),
    'Se': dict(name='Monoclinic Se', formula='Se$_6$', sg='P2$_1$', sgno=4, accent='#c25e00'),
}

order = ['P', 'CdHgI', 'SnS', 'Se']

# DFT: E_hull (meV/atom, above MP GGA/GGA+U hull), E_g (PBE, eV)
props = {
    'P': dict(ehull='46.5', eg='1.83'),
    'CdHgI': dict(ehull='108.9', eg='1.47'),
    'SnS': dict(ehull='175.6', eg='0.52'),
    'Se': dict(ehull='26.6', eg='1.26'),
}

# Static (relaxed-ion) epsilon_0 = electronic + ionic 3x3 tensors from DFPT
full_eps = {
    'P': np.array([[8.80, -0.01, -0.01], [-0.01, 8.79, 0.00], [-0.01, 0.00, 2.54]]),
    'CdHgI': np.array([[26.78, 0.00, 0.00], [0.00, 26.77, 0.00], [0.00, 0.00, 4.72]]),
    'SnS': np.array([[15.54, 0.00, 0.00], [0.00, 15.54, 0.00], [0.00, 0.00, 8.24]]),
    'Se': np.array([[4.75, 0.00, 0.35], [0.00, 6.04, 0.00], [0.35, 0.00, 4.33]]),
}

# --------------------------------------------------------------------------
# Load structures and derive the element set per panel
# --------------------------------------------------------------------------
cif_paths = {
    'P':     'cifs/P_SG144.cif',
    'CdHgI': 'cifs/CdHgI_SG162.cif',
    'SnS':   'cifs/SnS_SG144.cif',
    'Se':    'cifs/Se_SG4.cif',
}
atoms_cache = {k: read(p) for k, p in cif_paths.items()}

panel_elems = {}
for k in order:
    zs = []
    for Z in atoms_cache[k].get_atomic_numbers():
        Z = int(Z)
        if Z not in zs:
            zs.append(Z)
    panel_elems[k] = zs


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def fmt(v):
    if abs(v)<0.005: return "0"
    return f"{v:.2f}"


def build_cell(atoms, rep): return atoms*rep


def rot_matrix(elev,azim):
    e=np.radians(elev); a=np.radians(azim)
    Rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(e),-np.sin(e)],[0,np.sin(e),np.cos(e)]])
    return Rx@Rz


def bonds_from_cutoff(pos,nums,scale=1.25):
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
    return corners,edges


def view_basis(atoms,view='a',up='c'):
    C=np.array(atoms.cell); vmap={'a':C[0],'b':C[1],'c':C[2]}
    w=vmap[view]/np.linalg.norm(vmap[view])
    upv=vmap[up]-np.dot(vmap[up],w)*w; v=upv/np.linalg.norm(upv)
    u=np.cross(v,w); u=u/np.linalg.norm(u); return u,v,w


def glossy_sphere_w(ax,cx,cy,r,color,z=2):
    ax.add_patch(Circle((cx,cy),r,facecolor=color,edgecolor='#222',lw=0.6,zorder=z))
    core=np.clip(np.array(color)*1.22+0.06,0,1)
    ax.add_patch(Circle((cx-0.28*r,cy+0.28*r),0.60*r,facecolor=core,edgecolor='none',zorder=z+0.1,alpha=0.5))
    ax.add_patch(Circle((cx-0.34*r,cy+0.34*r),0.18*r,facecolor='white',edgecolor='none',zorder=z+0.2,alpha=0.9))


def render_w(ax,atoms,rep,elev,azim,rscale=0.42,bond_scale=1.25,lw=2.6,show_cell=True):
    ax.set_facecolor('white')
    sc=build_cell(atoms,rep); pos=sc.get_positions(); nums=sc.get_atomic_numbers()
    center=pos.mean(0); pos=pos-center; R=rot_matrix(elev,azim); P=pos@R.T
    x,y,z=P[:,0],P[:,1],P[:,2]; bl=bonds_from_cutoff(pos,nums,bond_scale)
    cols=jmol_colors[nums]; rad=covalent_radii[nums]*rscale; zspan=np.ptp(z)+1e-9; zmin=z.min()
    if show_cell:
        corners,edges=cell_edges(atoms); off=(np.array(rep)//2)@np.array(atoms.cell); cc=(corners+off-center)@R.T
        for (i,j) in edges: ax.plot([cc[i,0],cc[j,0]],[cc[i,1],cc[j,1]],color='#555',lw=0.8,alpha=0.5,zorder=0.5)
    draw=[('bond',(z[i]+z[j])/2,i,j) for (i,j) in bl]+[('atom',z[i],i,None) for i in range(len(pos))]; draw.sort(key=lambda t:t[1])
    for typ,zz,i,j in draw:
        if typ=='bond':
            mid=[(x[i]+x[j])/2,(y[i]+y[j])/2]
            ax.plot([x[i],mid[0]],[y[i],mid[1]],color=cols[i],lw=lw,solid_capstyle='round',zorder=1)
            ax.plot([mid[0],x[j]],[mid[1],y[j]],color=cols[j],lw=lw,solid_capstyle='round',zorder=1)
        else:
            depth=0.62+0.38*(z[i]-zmin)/zspan
            glossy_sphere_w(ax,x[i],y[i],rad[i],np.clip(np.array(cols[i])*depth,0,1))
    ax.set_aspect('equal'); ax.axis('off'); m=rad.max()+0.6
    ax.set_xlim(x.min()-m,x.max()+m); ax.set_ylim(y.min()-m,y.max()+m)


def render_along(ax,atoms,rep,view='a',up='c',rscale=0.42,bond_scale=1.25,lw=2.6,drop_dangling=False,ycrop=None,show_cell=False):
    ax.set_facecolor('white')
    sc=build_cell(atoms,rep); pos=sc.get_positions(); nums=sc.get_atomic_numbers()
    center=pos.mean(0); pos=pos-center; bl=bonds_from_cutoff(pos,nums,bond_scale)
    if drop_dangling:
        bonded=set()
        for i,j in bl: bonded.add(i); bonded.add(j)
        keep=np.array(sorted(bonded)); remap={o:n for n,o in enumerate(keep)}
        pos=pos[keep]; nums=nums[keep]; bl=[(remap[i],remap[j]) for i,j in bl]
    u,v,w=view_basis(atoms,view,up); x=pos@u; y=pos@v; z=pos@w
    cols=jmol_colors[nums]; rad=covalent_radii[nums]*rscale; zspan=np.ptp(z)+1e-9; zmin=z.min()
    if show_cell:
        # draw the FULL supercell outer box, centered on the atom cloud (same `center`
        # subtraction as the atoms) so the frame tracks the structure instead of floating
        corners,edges=cell_edges(sc); cc=corners-center
        cx=cc@u; cy=cc@v
        for (i,j) in edges: ax.plot([cx[i],cx[j]],[cy[i],cy[j]],color='#555',lw=0.8,alpha=0.45,zorder=0.5)
    draw=[('bond',(z[i]+z[j])/2,i,j) for (i,j) in bl]+[('atom',z[i],i,None) for i in range(len(pos))]; draw.sort(key=lambda t:t[1])
    for typ,zz,i,j in draw:
        if typ=='bond':
            mid=[(x[i]+x[j])/2,(y[i]+y[j])/2]
            ax.plot([x[i],mid[0]],[y[i],mid[1]],color=cols[i],lw=lw,solid_capstyle='round',zorder=1)
            ax.plot([mid[0],x[j]],[mid[1],y[j]],color=cols[j],lw=lw,solid_capstyle='round',zorder=1)
        else:
            depth=0.62+0.38*(z[i]-zmin)/zspan
            glossy_sphere_w(ax,x[i],y[i],rad[i],np.clip(np.array(cols[i])*depth,0,1))
    ax.set_aspect('equal'); ax.axis('off'); m=rad.max()+0.5
    xlo,xhi=x.min()-m,x.max()+m; ylo,yhi=y.min()-m,y.max()+m
    if show_cell:
        # ensure the full supercell frame is inside the view (small pad past corners)
        pad=0.4
        xlo=min(xlo,cx.min()-pad); xhi=max(xhi,cx.max()+pad)
        ylo=min(ylo,cy.min()-pad); yhi=max(yhi,cy.max()+pad)
    if ycrop is not None:
        yr=yhi-ylo; yhi=ylo+ycrop[1]*yr; ylo=ylo+ycrop[0]*yr
    ax.set_xlim(xlo,xhi); ax.set_ylim(ylo,yhi)


def draw_chip_row(ax, zs, fs=15):
    ax.axis('off')
    r=6.5  # circle radius in points -> guaranteed round
    boxes=[]
    for Z in zs:
        col=np.array(jmol_colors[Z])
        da=DrawingArea(2*r+2,2*r+2,0,0)
        da.add_artist(MCircle((r+1,r+1),r,facecolor=col,edgecolor='#222',lw=0.7))
        core=np.clip(col*1.22+0.06,0,1)
        da.add_artist(MCircle((r+1-0.30*r,r+1+0.30*r),0.42*r,facecolor=core,edgecolor='none',alpha=0.5))
        da.add_artist(MCircle((r+1-0.40*r,r+1+0.40*r),0.16*r,facecolor='white',edgecolor='none',alpha=0.9))
        txt=TextArea(chemical_symbols[Z],textprops=dict(fontsize=fs,color='#111'))
        boxes.append(HPacker(children=[da,txt],align='center',pad=0,sep=3))
    row=HPacker(children=boxes,align='center',pad=0,sep=15)
    ab=AnchoredOffsetbox(loc='center',child=row,frameon=False,pad=0,
                         bbox_to_anchor=(0.5,0.5),bbox_transform=ax.transAxes)
    ax.add_artist(ab)


def draw_tensor(ax, cx, cy, M, hw, hh, color='#222', fs=8.0):
    """3x3 matrix with brackets, centered at (cx,cy) in axes coords."""
    dx=hw*0.62; dy=hh*0.62
    xs=[cx-dx,cx,cx+dx]; ys=[cy+dy,cy,cy-dy]
    for r in range(3):
        for c in range(3):
            ax.text(xs[c],ys[r],fmt(M[r,c]),ha='center',va='center',
                    fontsize=fs,color=color,zorder=3,transform=ax.transAxes)
    bx=hw*1.02; ty=hh*1.0; tick=hw*0.14
    for sgn in (-1,1):
        xb=cx+sgn*bx
        ax.plot([xb,xb],[cy-ty,cy+ty],color=color,lw=0.9,transform=ax.transAxes,zorder=3)
        ax.plot([xb,xb-sgn*tick],[cy+ty,cy+ty],color=color,lw=0.9,transform=ax.transAxes,zorder=3)
        ax.plot([xb,xb-sgn*tick],[cy-ty,cy-ty],color=color,lw=0.9,transform=ax.transAxes,zorder=3)


def draw_table(ax, order, meta, props, full_eps):
    ax.set_facecolor('white'); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ncol=len(order); label_w=0.20; col_w=(1-label_w)/ncol
    # row layout: header + 3 scalar rows + 1 tall tensor row
    header_h=0.16; eps_h=0.34; scalar_h=(1-header_h-eps_h)/3
    # headers
    for c,k in enumerate(order):
        x0=label_w+c*col_w; ac=meta[k]['accent']
        ax.add_patch(FancyBboxPatch((x0+0.006,1-header_h+0.010),col_w-0.012,header_h-0.020,
            boxstyle="round,pad=0,rounding_size=0.02",facecolor=ac,edgecolor='none',transform=ax.transAxes,zorder=1))
        ax.text(x0+col_w/2,1-header_h*0.5,meta[k]['formula'],ha='center',va='center',color='white',fontsize=17,fontweight='bold',zorder=2)
    scalar_rows=[("Space group",lambda k:f"{meta[k]['sg']} ({meta[k]['sgno']})"),
                 (r"$E_\mathrm{hull}$ (meV/atom)",lambda k:props[k]['ehull']),
                 (r"$E_\mathrm{g}$ (eV)",lambda k:props[k]['eg'])]
    for r,(label,fn) in enumerate(scalar_rows):
        y0=1-header_h-(r+1)*scalar_h
        if r%2==0:
            ax.add_patch(FancyBboxPatch((0.004,y0+0.004),0.992,scalar_h-0.008,
                boxstyle="round,pad=0,rounding_size=0.01",facecolor='#f2f2f4',edgecolor='none',transform=ax.transAxes,zorder=0.5))
        ax.text(label_w-0.02,y0+scalar_h/2,label,ha='right',va='center',fontsize=13,color='#111',zorder=2)
        for c,k in enumerate(order):
            ax.text(label_w+c*col_w+col_w/2,y0+scalar_h/2,str(fn(k)),ha='center',va='center',fontsize=13,color='#222',zorder=2)
    # tensor row
    y0=0.0
    ax.add_patch(FancyBboxPatch((0.004,y0+0.004),0.992,eps_h-0.008,
        boxstyle="round,pad=0,rounding_size=0.01",facecolor='#f2f2f4',edgecolor='none',transform=ax.transAxes,zorder=0.5))
    ax.text(label_w-0.02,eps_h/2,r"$\varepsilon_0$",ha='right',va='center',fontsize=15,color='#111',zorder=2)
    for c,k in enumerate(order):
        cx=(label_w+c*col_w+col_w/2); cy=eps_h/2
        draw_tensor(ax,cx,cy,full_eps[k],hw=col_w*0.36,hh=eps_h*0.38,color='#111',fs=12.5)
    ax.plot([0,1],[1-header_h,1-header_h],color='#333',lw=1.0,transform=ax.transAxes,zorder=3)


def build_figure(save="crystals_nc_figure.png"):
    hero={"P":(5,5,1),"CdHgI":(4,4,1),"SnS":(5,5,1)}
    side_final={
     "P":dict(view='a',up='c',rep=(1,5,1),drop=True,ycrop=(0.0,0.68)),
     "CdHgI":dict(view='a',up='c',rep=(1,4,2),drop=True,ycrop=(0.44,0.98)),
     "SnS":dict(view='b',up='c',rep=(6,1,1),drop=True,ycrop=(0.60,1.0)),
     "Se":dict(view='a',up='c',rep=(1,4,2),drop=True)}
    letters=['(a)','(b)','(c)','(d)']
    fig=plt.figure(figsize=(15,10.6))
    gs=gridspec.GridSpec(3,4,figure=fig,height_ratios=[1.0,0.92,0.70],
            hspace=0.06,wspace=0.03,left=0.015,right=0.985,top=0.950,bottom=0.045)
    # fixed row-0 cell geometry (independent of aspect shrink)
    cell0=[gs[0,c].get_position(fig) for c in range(4)]
    for c,k in enumerate(order):
        ax=fig.add_subplot(gs[0,c]); m=meta[k]
        if k=="Se":
            render_along(ax,atoms_cache["Se"],(3,3,3),view='c',up='b',drop_dangling=True)
            # zoom out: pad limits symmetrically so the helices sit smaller with margin
            zx=0.28*(ax.get_xlim()[1]-ax.get_xlim()[0]); zy=0.28*(ax.get_ylim()[1]-ax.get_ylim()[0])
            ax.set_xlim(ax.get_xlim()[0]-zx,ax.get_xlim()[1]+zx)
            ax.set_ylim(ax.get_ylim()[0]-zy,ax.get_ylim()[1]+zy)
        else: render_w(ax,atoms_cache[k],hero[k],12,18,show_cell=True)
        ax.set_anchor('S')  # bottom-align heroes on the shared chip baseline (closes the gap)
        ax.text(0.5,1.015,m['name'],transform=ax.transAxes,ha='center',va='bottom',fontsize=15,fontweight='bold',color=m['accent'])
    # shared chip row centered in the visual gap between hero structures and layer profiles
    ys=cell0[0].y0-0.080
    for c,k in enumerate(order):
        strip=fig.add_axes([cell0[c].x0, ys, cell0[c].width, 0.028])
        draw_chip_row(strip, panel_elems[k])
    # letter labels at a fixed figure height (top of row 0), so (a)(b)(c) match (d)
    ly=cell0[0].y1-0.006
    for c in range(4):
        fig.text(cell0[c].x0+0.006, ly, letters[c], ha='left', va='top',
                 fontsize=16, fontweight='bold', color='#111')
    for c,k in enumerate(order):
        ax=fig.add_subplot(gs[1,c]); sf=side_final[k]
        render_along(ax,atoms_cache[k],sf['rep'],view=sf['view'],up=sf['up'],drop_dangling=sf.get('drop',False),ycrop=sf.get('ycrop'))
        proj={'a':'down $a$','b':'down $b$','c':'down $c$'}[sf['view']]
        ax.text(0.5,-0.02,f"layer profile — {proj}",transform=ax.transAxes,ha='center',va='top',fontsize=11,color='#444')
    axt=fig.add_subplot(gs[2,:]); draw_table(axt,order,meta,props,full_eps)
    fig.text(0.5,0.022,
        "Space groups from spglib (symprec = 0.05 Å).  "
        r"$E_\mathrm{hull}$: energy above the MP GGA/GGA+U convex hull;  $E_\mathrm{g}$: PBE band gap;  "
        r"$\varepsilon_0$: static (relaxed-ion) dielectric tensor, electronic + ionic, from DFPT.",
        ha='center',va='bottom',fontsize=9.5,color='#666',style='italic')
    fig.savefig(save,dpi=200,facecolor='white'); plt.close(fig)



# --------------------------------------------------------------------------
if __name__ == '__main__':
    build_figure('crystals_nc_figure.png')
    print('wrote crystals_nc_figure.png')
