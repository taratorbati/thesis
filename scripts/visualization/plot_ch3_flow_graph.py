# =============================================================================
# scripts/visualization/plot_ch3_flow_graph.py
#
# Figure: ch3_flow_graph  (Figure 3.2 in the thesis)
#
# Caption (from ch3_system_description.tex):
#   "Directed flow graph overlaid on the normalized elevation map.
#    Arrows indicate the direction of lateral surface runoff between
#    agents. Steep regions exhibit dense, parallel flow lines; the flat
#    valley floor shows sparser, more diffuse connections."
#
# Data source: gilan_farm.tif  — loaded by src/terrain.py::load_terrain().
#   build_directed_graph() implements the padded-DEM D8 routing described
#   in §3.1.2: edges go from agent n to all Moore-neighborhood agents m
#   with gamma(m) < gamma(n), with off-farm ghost cells absorbing boundary
#   runoff so that Nr[n] > 0 for every internal agent.
#
# Layout: single panel.
#   Background  : normalized elevation heatmap (gamma in [0,1], per Eq. 3.1)
#   Arrows      : one arrow per directed internal edge, from src centroid
#                 toward dst centroid, scaled to cell size
#   Arrow color : gray, semi-transparent (so elevation shading stays visible)
#   Nr colorbar : Nr (total downhill neighbor count) annotated per agent
#                 as small text to show the routing density pattern
#
# Output:
#   figures/ch3_flow_graph.pdf   ← include in LaTeX
#   figures/ch3_flow_graph.png   ← quick preview
#
# Usage (run from the repository root):
#   python -m scripts.visualization.plot_ch3_flow_graph
#
# Dependencies: matplotlib, numpy, Pillow (already in requirements.txt)
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.terrain import load_terrain

# ── Paths ─────────────────────────────────────────────────────────────────────
DEM_PATH    = PROJECT_ROOT / 'gilan_farm.tif'
FIGURES_DIR = PROJECT_ROOT / 'figures'
OUT_PDF     = FIGURES_DIR  / 'ch3_flow_graph.pdf'
OUT_PNG     = FIGURES_DIR  / 'ch3_flow_graph.png'

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'font.size'      : 9,
    'axes.titlesize' : 10,
    'axes.labelsize' : 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi'     : 150,
    'savefig.dpi'    : 300,
    'pdf.fonttype'   : 42,
    'ps.fonttype'    : 42,
})

# Colormap for elevation background: warm (high) → cool (low)
CMAP_ELEV   = 'RdYlGn_r'
# Arrow properties
ARROW_COLOR = '0.25'     # dark gray
ARROW_ALPHA = 0.55
ARROW_WIDTH = 0.012      # relative to axes (FancyArrowPatch scale)
ARROW_HS    = 0.18       # arrowhead size in data units (fraction of cell)


def _agent_center(agent_idx, ncols):
    """Return (col, row) center of agent in data (grid) coordinates."""
    row = agent_idx // ncols
    col = agent_idx  % ncols
    return float(col), float(row)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load DEM and build directed graph ─────────────────────────────────────
    # load_terrain() is the single locked function that produces the same
    # graph used by the ABM and the MPC symbolic dynamics.
    terrain = load_terrain(DEM_PATH)

    elevation  = terrain['elevation_2d'].astype(float)  # (10, 13)
    gamma_flat = terrain['gamma_flat']                   # (130,)
    sends_to   = terrain['sends_to']                     # {n: [m, ...]}
    Nr         = terrain['Nr']                           # {n: total_lower_count}
    nrows      = terrain['rows']                         # 10
    ncols      = terrain['cols']                         # 13
    N          = terrain['N']                            # 130

    gamma_2d   = gamma_flat.reshape(nrows, ncols)
    Nr_2d      = np.array([Nr[n] for n in range(N)], dtype=float).reshape(nrows, ncols)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')

    # Background: normalized elevation heatmap
    norm_elev = Normalize(vmin=0.0, vmax=1.0)
    ax.imshow(
        gamma_2d,
        cmap=CMAP_ELEV, norm=norm_elev,
        origin='upper',
        extent=[-0.5, ncols - 0.5, nrows - 0.5, -0.5],
        interpolation='bilinear', alpha=0.85,
        zorder=0,
    )

    # Thin agent grid lines
    for i in range(ncols + 1):
        ax.axvline(i - 0.5, color='white', lw=0.2, alpha=0.4, zorder=1)
    for j in range(nrows + 1):
        ax.axhline(j - 0.5, color='white', lw=0.2, alpha=0.4, zorder=1)

    # Directed edges (arrows)
    # Each edge (n → m) is drawn as an arrow from cell-center n to cell-center m,
    # displaced slightly toward m (0.35 of the distance) so that the arrows
    # remain legible when multiple edges leave the same cell.
    REACH = 0.38   # arrow tip stops at 38% of the way to dst center

    for src, dsts in sends_to.items():
        sx, sy = _agent_center(src, ncols)
        for dst in dsts:
            dx, dy = _agent_center(dst, ncols)
            # Midpoint displacement: start at 12%, end at (12% + REACH)
            START = 0.12
            x0 = sx + START * (dx - sx)
            y0 = sy + START * (dy - sy)
            x1 = sx + (START + REACH) * (dx - sx)
            y1 = sy + (START + REACH) * (dy - sy)

            ax.annotate(
                '',
                xy     =(x1, y1),
                xytext =(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color=ARROW_COLOR,
                    lw=0.7, alpha=ARROW_ALPHA,
                    mutation_scale=7,
                ),
                zorder=3,
            )

    # Annotate Nr per cell (small text showing total downhill neighbor count)
    for r in range(nrows):
        for c in range(ncols):
            n   = r * ncols + c
            nr  = Nr[n]
            # White text on dark cells, dark text on light cells
            bg  = gamma_2d[r, c]
            txt_color = 'white' if bg > 0.55 else 'black'
            ax.text(
                c, r, str(nr),
                ha='center', va='center',
                fontsize=5.5, color=txt_color, alpha=0.85,
                fontweight='bold', zorder=4,
            )

    # Colorbar for elevation
    sm = cm.ScalarMappable(cmap=CMAP_ELEV, norm=norm_elev)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label('Normalised elevation  $\\gamma^{(n)}$', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Axis formatting
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(range(1, ncols + 1), fontsize=6.5)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(range(1, nrows + 1), fontsize=6.5)
    ax.set_xlabel('Column  (West → East)', fontsize=8)
    ax.set_ylabel('Row  (North → South)',  fontsize=8)
    ax.tick_params(length=2)

    # Summary statistics as subtitle
    n_internal_edges = sum(len(v) for v in sends_to.values())
    n_sinks = sum(1 for n in range(N) if Nr[n] == 0)
    ax.set_title(
        f'Directed D8 flow graph — {N} agents, '
        f'{n_internal_edges} internal edges, '
        f'{n_sinks} sink agents  '
        f'(numbers show $N_r^{{(n)}}$, total downhill neighbours)',
        fontsize=8.5, pad=5,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight', format='pdf')
    fig.savefig(OUT_PNG, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f'Saved: {OUT_PDF}')
    print(f'Saved: {OUT_PNG}')
    print(f'Graph: {N} agents, {n_internal_edges} internal edges, '
          f'{n_sinks} internal sinks')
    print(f'Nr range: {int(Nr_2d.min())}–{int(Nr_2d.max())}')


if __name__ == '__main__':
    main()
