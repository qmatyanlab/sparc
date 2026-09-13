#!/usr/bin/env python
"""Nature-Communications Figure 4 (in-plane-isotropy run).
(a) RL design-space drift KDE contour (aggregate over all loops) with early->late
    median-shift arrow + numbered, accent-colored stars for the 4 DFPT-verified crystals.
(b) 4-crystal composite: hero (down c) + layer profile (down a) + property table.
Panel (a) rendering mirrors plot_fig4_drift.py --aggregate exactly (KDE bw=0.55,
gridn=200, reds_trunc, target rectangle gt=0.3/st=0.5). Data from drift_iso_data.npz.
"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import make_crystal_figure as mcf
from ase.io import read

plt.style.use('./publication.mplstyle')

M = json.load(open('_iso_meta.json'))
ORDER = M['ORDER']; META = M['META']; PROPS = M['PROPS']
FULL_EPS = {k: np.array(v) for k, v in M['FULL_EPS'].items()}
STARS = M['STARS']; CIF_PATHS = M['CIF_PATHS']
ATOMS = {k: read(p) for k, p in CIF_PATHS.items()}
PANEL_ELEMS = {}
for k in ORDER:
    zs = []
    for Z in ATOMS[k].get_atomic_numbers():
        Z = int(Z)
        if Z not in zs: zs.append(Z)
    PANEL_ELEMS[k] = zs

# ---------------------------------------------------------------- panel (a)
def draw_drift(ax, npz='drift_iso_data.npz', n=10, gt=0.3, st=0.5, gap_max=4.0):
    d = np.load(npz)
    gap, reward, step = d['gap'], d['reward'], d['step']
    gate_min = float(d['gate_min'])
    steps = np.unique(step)
    nn = min(n, max(1, len(steps)//2))
    em = np.isin(step, steps[:nn]); lm = np.isin(step, steps[-nn:])
    reds_trunc = LinearSegmentedColormap.from_list('rt', plt.cm.Reds(np.linspace(0.35,1.0,256)))
    x_hi = min(gap_max, float(np.percentile(gap,99))+0.3)
    xlim, ylim = (-0.1, x_hi), (0.0, 1.03)
    gn = 200
    xs = np.linspace(*xlim, gn); ys = np.linspace(*ylim, gn)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = gaussian_kde(np.vstack([gap, reward]), bw_method=0.55)(
        np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    Zn = ZZ/ZZ.max() if ZZ.max() > 0 else ZZ
    levels = [0.15,0.3,0.45,0.6,0.75,0.9]
    cs = ax.contour(XX, YY, Zn, levels=levels, cmap=reds_trunc, linewidths=1.5, zorder=3)
    # HORIZONTAL colorbar built from the contour LINES, as an inset in the sparse
    # right-middle region (clear of the dense contours, stars, and legend box)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width="34%", height="3.2%", loc='center',
                     bbox_to_anchor=(0.24, -0.15, 1, 1), bbox_transform=ax.transAxes,
                     borderpad=0)
    cb = ax.figure.colorbar(cs, cax=cax, ticks=levels, orientation='horizontal')
    cb.set_label('relative KDE density', fontsize=11.5)
    cb.ax.tick_params(labelsize=10)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')
    # target rectangle
    ax.plot([gt,gt],[st,ylim[1]], color='0.3', ls='--', lw=2.0, zorder=4)
    ax.plot([gt,xlim[1]],[st,st], color='0.3', ls='--', lw=2.0, zorder=4)
    # early -> late median shift arrow
    pA = (float(np.median(gap[em])), float(np.median(reward[em])))
    pB = (float(np.median(gap[lm])), float(np.median(reward[lm])))
    pAd = (pA[0], pA[1] + 0.022)   # lift drawn first-10 dot + arrow tail off the zero crop
    ax.annotate('', xy=pB, xytext=pAd,
        arrowprops=dict(arrowstyle='-|>', color='0.1', lw=2.0, mutation_scale=20,
                        shrinkA=5, shrinkB=6), zorder=6)
    ax.scatter([pAd[0]],[pAd[1]], s=95, fc='#3b3b3b', ec='k', lw=1, zorder=7)
    ax.scatter([pB[0]],[pB[1]], s=95, fc='#2ca25f', ec='k', lw=1, zorder=7)
    # accent-colored numbered stars for the verified crystals
    for i,(lab,gx,gy,ac) in enumerate(STARS, 1):
        gyp = gy + 0.022 if gy < 0.05 else gy   # lift floor marker so it clears the zero crop
        ax.scatter([gx],[gyp], marker='o', s=360, facecolor=ac, edgecolor='white',
                   linewidths=1.6, zorder=8)
        ax.annotate(str(i), (gx,gyp), xytext=(0,0), textcoords='offset points',
                    fontsize=12.5, fontweight='bold', color='white', zorder=9,
                    ha='center', va='center')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xlabel('Predicted band gap (eV)', fontsize=16)
    ax.set_ylabel(r'Reward $r_\mathrm{uni}$', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    # legend: all six keys as identical colored disks (numbered for the 4 crystals,
    # plain for the two medians) via one handler -> every row shares the same baseline.
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Circle
    from matplotlib.text import Text as _Text
    class DiskKey:
        def __init__(self, color, num=None): self.color=color; self.num=num; self._lab=''
        def get_label(self): return self._lab
        def set_label(self, s): self._lab = s
    class HandlerDisk(HandlerBase):
        def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
            cx, cy = xd + w/2.0, yd + h/2.0 - h*0.40
            rad = min(w, h)*0.42
            arts = [Circle((cx, cy), rad, facecolor=orig.color, edgecolor='white',
                           linewidth=1.4, transform=trans, zorder=4)]
            if orig.num is not None:
                arts.append(_Text(cx, cy, str(orig.num), ha='center', va='center',
                                  fontsize=8.5, fontweight='bold', color='white',
                                  transform=trans, zorder=5))
            return arts
    keys   = [DiskKey('#3b3b3b'), DiskKey('#2ca25f')]
    labels = ['First 10 (median)', 'Last 10 (median)']
    for i,(lab,gx,gy,ac) in enumerate(STARS, 1):
        m = META[lab]
        keys.append(DiskKey(ac, i)); labels.append(f"{m['name']} ({m['sgno']})")
    hmap = {k: HandlerDisk() for k in keys}
    ax.legend(keys, labels, handler_map=hmap, loc='lower right',
              fontsize=11, frameon=True, framealpha=0.92,
              handleheight=1.4, handlelength=1.4, labelspacing=0.65, borderpad=0.7)
    return pA, pB

# ---------------------------------------------------------------- build
def build(save='figure4_iso.png'):
    fig = plt.figure(figsize=(18.5, 9.4))
    B_TOP, B_BOT = 0.905, 0.085
    # panel (a) left
    axA = fig.add_axes([0.055, 0.155, 0.340, 0.70])
    draw_drift(axA)
    fig.text(0.028, 0.955, '(a)', ha='left', va='top', fontsize=20, fontweight='bold', color='#111')

    # panel (b) right: 2x4 grid (hero-down-c / property table)
    gb = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.0,0.62],
        hspace=0.20, wspace=0.03, left=0.435, right=0.992, top=B_TOP, bottom=B_BOT)
    cell0 = [gb[0,c].get_position(fig) for c in range(4)]
    cell1 = [gb[1,c].get_position(fig) for c in range(4)]

    # hero row: down c (shows in-plane isotropy)
    HERO_REP = {'ZrTeO2':(3,3,1),'ZrTi2O6':(2,2,1),'P':(3,3,1),'KCuAs2S3':(1,1,1)}
    title_y = cell0[0].y1 + 0.012   # common baseline for all four names
    for c,k in enumerate(ORDER):
        ax = fig.add_subplot(gb[0,c]); m = META[k]
        mcf.render_along(ax, ATOMS[k], HERO_REP[k], view='c', up='b',
                         drop_dangling=False, show_cell=True)
        ax.set_anchor('S')
        fig.text(cell0[c].x0 + 0.5*cell0[c].width, title_y, m['name'],
                 ha='center', va='bottom', fontsize=17, fontweight='bold', color=m['accent'])

    # (b) label
    fig.text(0.408, 0.955, '(b)', ha='left', va='top', fontsize=20, fontweight='bold', color='#111')

    # chip legend centered in the gap between hero-row bottom and table top
    ch = 0.024
    gap_mid = 0.5*(cell1[0].y1 + cell0[0].y0)
    ys = gap_mid - ch/2
    for c,k in enumerate(ORDER):
        strip = fig.add_axes([cell0[c].x0, ys, cell0[c].width, ch])
        mcf.draw_chip_row(strip, PANEL_ELEMS[k])

    # table
    axt = fig.add_subplot(gb[1,:])
    mcf.draw_table(axt, ORDER, META, PROPS, FULL_EPS)

    fig.savefig(save, dpi=200, facecolor='white')
    import os
    fig.savefig(os.path.splitext(save)[0]+'.pdf', facecolor='white')
    plt.close(fig)
    print('wrote', save)

if __name__ == '__main__':
    build('figure4_iso.png')
