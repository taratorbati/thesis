# =============================================================================
# scripts/visualization/plot_ch3_dem_interactive.py
#
# Interactive version of Figure 3.1 using Plotly.
# Provides 3D rotation, panning, zooming, and hover data for the agent grid.
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.terrain import load_dem

# ── Paths ─────────────────────────────────────────────────────────────────────
DEM_PATH = PROJECT_ROOT / 'gilan_farm.tif'

def main():
    # ── Load real DEM ─────────────────────────────────────────────────────────
    # shape (10, 13) -> row 0 = North, column 0 = West
    elevation = load_dem(DEM_PATH).astype(float)   
    nrows, ncols = elevation.shape                  

    # Convert colormap name format from Matplotlib to Plotly compatible
    # Matplotlib's 'RdYlGn_r' is equivalent to Plotly's 'RdYlGn' (since Plotly 
    # handles the scale mapping cleanly, we map low to cool/green and high to warm/red)
    plotly_cmap = 'RdYlGn' 

    # 1-based indexing for the axis tick displays to match thesis text
    col_coords = np.arange(1, ncols + 1)
    row_coords = np.arange(1, nrows + 1)

    # ── Create Subplots ───────────────────────────────────────────────────────
    # Left: 2D Heatmap, Right: 3D Surface
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'scene'}]],
        subplot_titles=('(A) Top view — agent grid', '(B) 3D surface'),
        horizontal_spacing=0.1
    )

    # ── Panel A: 2D Heatmap + Contours ────────────────────────────────────────
    # Custom hover text template
# Change ':.1s' to ':.1f' (for 1 decimal place) or ':.0f' (for integers)
    hover_text = [[f"Row: {r}<br>Col: {c}<br>Elev: {elevation[r-1, c-1]:.1f} m" 
                   for c in col_coords] for r in row_coords]

    fig.add_trace(
        go.Heatmap(
            z=elevation,
            x=col_coords,
            y=row_coords,
            colorscale=plotly_cmap,
            reversescale=True,  # Keeps warm colors on high ridges
            showscale=True,
            colorbar=dict(title="Elevation (m)", x=0.45, len=0.8),
            text=hover_text,
            hoverinfo='text',
            zsmooth='best' # Emulates the 'bilinear' interpolation
        ),
        row=1, col=1
    )

    # Overlay Contour lines
    fig.add_trace(
        # Change this:
    # line=dict(width=1, opacity=0.6),
    
    go.Contour(
            z=elevation,
            x=col_coords,
            y=row_coords,
            colorscale=[[0, 'black'], [1, 'black']], # Solid dark contours
            showscale=False,
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
            line=dict(width=1, color='rgba(0, 0, 0, 0.6)'),  # Pure black with 60% opacity
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # ── Panel B: 3D Surface Plot ──────────────────────────────────────────────
    fig.add_trace(
        go.Surface(
            z=elevation,
            x=col_coords,
            y=row_coords,
            colorscale=plotly_cmap,
            reversescale=True,
            showscale=False, # Shared with the heatmap colorbar
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Elevation: %{z} m<extra></extra>"
        ),
        row=1, col=2
    )

    # ── Layout & Aesthetics ───────────────────────────────────────────────────
    # Update 2D axes behavior
    fig.update_xaxes(title_text="Column (West → East)", tickmode='linear', row=1, col=1)
    fig.update_yaxes(title_text="Row (North → South)", tickmode='linear', autorange="reversed", row=1, col=1)

    # Update 3D scene behavior
    fig.update_scenes(
        xaxis=dict(title='Column (W→E)', tickmode='linear'),
        yaxis=dict(title='Row (N→S)', tickmode='linear', autorange="reversed"), # Maintain North at top/back
        zaxis=dict(title='Elevation (m)'),
        camera=dict(
            eye=dict(x=-1.5, y=-1.5, z=1.2) # Matches an intuitive perspective view
        ),
        row=1, col=2
    )

    # Title setup mimicking the thesis caption details
    fig.update_layout(
        title_text=f"Gilan Study Site — 13×10 Agent Grid ({int(elevation.min())}–{int(elevation.max())} m)",
        title_x=0.5,
        font=dict(family="DejaVu Sans", size=11),
        width=1100,
        height=550,
        margin=dict(t=80, b=50, l=50, r=50),
        template="plotly_white"
    )

    # Open in browser local port automatically
    fig.show()

if __name__ == '__main__':
    main()