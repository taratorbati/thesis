# =============================================================================
# scripts/visualization/plot_ch3_dem_heatmap.py
#
# Figure: ch3_dem_heatmap  (Figure 3.1 in the thesis)
#
# Caption (from ch3_system_description.tex):
#   "Three dimensional surface plot of the study site, resampled to the
#    13×10 agent grid. High ground (warm colors) forms ridges in the
#    northwestern portion of the field; the valley floor (cool colors)
#    drains toward the southeastern corner."
#
# Data source: gilan_farm.tif  (USGS SRTM 30 m, clipped and resampled)
#   Loaded via src/terrain.py::load_dem()  — the locked source of truth.
#   Shape: (10, 13), int32, elevation range 70–181 m (111 m relief).
#
# Output:
#   figures/ch3_dem_heatmap.pdf   ← include in LaTeX
#   figures/ch3_dem_heatmap.png   ← quick preview
#
# Usage (run from the repository root):
#   python -m scripts.visualization.plot_ch3_dem_heatmap
#
# Dependencies: matplotlib, numpy, Pillow (already in requirements.txt)
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.terrain import load_dem, normalize_elevation

# ── Paths ─────────────────────────────────────────────────────────────────────
DEM_PATH     = PROJECT_ROOT / 'gilan_farm.tif'
FIGURES_DIR  = PROJECT_ROOT / 'figures'
OUT_PDF      = FIGURES_DIR  / 'ch3_dem_heatmap.pdf'
OUT_PNG      = FIGURES_DIR  / 'ch3_dem_heatmap.png'

# ── Plot style ────────────────────────────────────────────────────────────────
# Match the existing ch3_climatology.png style used in the chapter.
plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'font.size'      : 9,
    'axes.titlesize' : 10,
    'axes.labelsize' : 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi'     : 150,
    'savefig.dpi'    : 300,
    'pdf.fonttype'   : 42,   # embed fonts — required for most thesis templates
    'ps.fonttype'    : 42,
})

# Colormap: warm (high elevation) → cool (low elevation), per the caption.
# 'terrain_r' reverses the default terrain map so warm = high, cool = low.
CMAP = 'RdYlGn_r'


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load real DEM ─────────────────────────────────────────────────────────
    elevation = load_dem(DEM_PATH).astype(float)   # shape (10, 13)
    nrows, ncols = elevation.shape                  # 10, 13
    gamma = normalize_elevation(elevation)          # [0, 1], per Eq. 3.1

    norm   = Normalize(vmin=elevation.min(), vmax=elevation.max())
    sm     = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    # Grid coordinates
    col_idx = np.arange(ncols)   # 0..12  →  West to East
    row_idx = np.arange(nrows)   # 0..9   →  North to South (row 0 = North)
    C, R = np.meshgrid(col_idx, row_idx)

    # ── Figure layout: 2D heatmap (left) + 3D surface (right) ────────────────
    fig = plt.figure(figsize=(11, 4.6))
    fig.patch.set_facecolor('white')

    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    # ── Panel A: 2D heatmap with agent grid + contours ────────────────────────
    im = ax2d.imshow(
        elevation,
        cmap=CMAP, norm=norm,
        origin='upper',                           # row 0 at top = North
        extent=[-0.5, ncols - 0.5,
                nrows - 0.5, -0.5],
        interpolation='bilinear',
    )

    # Agent grid lines (thin white)
    for i in range(ncols + 1):
        ax2d.axvline(i - 0.5, color='white', lw=0.25, alpha=0.5)
    for j in range(nrows + 1):
        ax2d.axhline(j - 0.5, color='white', lw=0.25, alpha=0.5)

    # Contour lines (elevation isolines)
    cs = ax2d.contour(
        C, R, elevation,
        levels=7, colors='k', linewidths=0.45, alpha=0.6,
    )
    ax2d.clabel(cs, fmt='%d m', fontsize=6.5, inline=True)

    # Axis labels (col/row indices, 1-based for readability)
    ax2d.set_xticks(range(ncols))
    ax2d.set_xticklabels(range(1, ncols + 1), fontsize=6)
    ax2d.set_yticks(range(nrows))
    ax2d.set_yticklabels(range(1, nrows + 1), fontsize=6)
    ax2d.set_xlabel('Column  (West → East)', fontsize=8)
    ax2d.set_ylabel('Row  (North → South)',  fontsize=8)
    ax2d.set_title('(A)  Top view — agent grid', fontsize=9, pad=4)
    ax2d.tick_params(length=2)

    # Annotate NW ridge and SE valley
    nw_row, nw_col = np.unravel_index(np.argmax(elevation), elevation.shape)
    se_row, se_col = np.unravel_index(np.argmin(elevation), elevation.shape)

    ax2d.annotate(
        f'NW ridge\n({int(elevation.max())} m)',
        xy=(nw_col, nw_row),
        xytext=(nw_col + 2.2, nw_row + 1.5),
        fontsize=7, color='saddlebrown', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='saddlebrown', lw=0.8),
    )
    ax2d.annotate(
        f'SE valley\n({int(elevation.min())} m)',
        xy=(se_col, se_row),
        xytext=(se_col - 3.5, se_row - 1.5),
        fontsize=7, color='steelblue', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.8),
    )

    cbar2d = fig.colorbar(sm, ax=ax2d, fraction=0.046, pad=0.04)
    cbar2d.set_label('Elevation (m)', fontsize=8)
    cbar2d.ax.tick_params(labelsize=7)

    # ── Panel B: 3D surface plot ──────────────────────────────────────────────
    # Flip rows so row 0 (North) appears at the back of the 3D view
    elev_3d = elevation[::-1, :]
    C3, R3  = np.meshgrid(col_idx, row_idx[::-1])

    face_colors = matplotlib.colormaps[CMAP](norm(elev_3d))

    ax3d.plot_surface(
        C3, R3, elev_3d,
        facecolors=face_colors,
        linewidth=0, antialiased=True,
        alpha=0.92, shade=True,
    )

    ax3d.set_xlabel('Column (W→E)', fontsize=7, labelpad=2)
    ax3d.set_ylabel('Row (N→S)',    fontsize=7, labelpad=2)
    ax3d.set_zlabel('Elevation (m)', fontsize=7, labelpad=2)
    ax3d.set_title('(B)  3D surface', fontsize=9, pad=2)
    ax3d.tick_params(labelsize=6, pad=1)
    ax3d.view_init(elev=28, azim=-55)

    #cbar3d = fig.colorbar(sm, ax=ax3d, fraction=0.032, pad=0.04, shrink=0.72)
    #cbar3d.set_label('Elevation (m)', fontsize=7)
    #cbar3d.ax.tick_params(labelsize=6)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        f'Gilan study site — $13 \\times 10$ agent grid'
        f'  ({int(elevation.min())}–{int(elevation.max())} m, '
        f'{int(elevation.max() - elevation.min())} m relief)',
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight', format='pdf')
    fig.savefig(OUT_PNG, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f'Saved: {OUT_PDF}')
    print(f'Saved: {OUT_PNG}')
    print(f'DEM range: {int(elevation.min())}–{int(elevation.max())} m  '
          f'(span = {int(elevation.max() - elevation.min())} m)')


if __name__ == '__main__':
    main()
