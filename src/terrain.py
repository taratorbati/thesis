# =============================================================================
# src/terrain.py
# Single source of truth for DEM loading and directed-graph construction.
#
# v2.0: Fractional routing boundary condition via DEM padding.
#
# Problem (v1): The 10×13 grid was a closed basin. Edge agents could only
# route water to internal neighbors, causing 3 sink agents to accumulate
# all runoff from the entire field — the "bathtub effect."
#
# Fix: Pad the DEM by 1 cell on each edge, extrapolating the slope outward.
# Compute the full D8 neighbor count using the padded grid, but only list
# internal (farm) agents in sends_to. Nr counts ALL lower neighbors
# (internal + external), so when the ABM divides phi2 by Nr, the fraction
# routed to off-farm cells is implicitly removed from the mass balance.
#
# Example: An edge agent with 4 lower neighbors (2 internal, 2 external)
# routes phi2/4 to each of the 2 internal neighbors. The remaining phi2/2
# is off-farm drainage — it exits the system without any explicit code in
# the ABM or CasADi dynamics.
#
# This is standard D8 fractional flow routing with absorbing boundary
# conditions, widely used in computational hydrology (O'Callaghan & Mark,
# 1984; Tarboton, 1997).
# =============================================================================

from pathlib import Path

import numpy as np
from PIL import Image


def load_dem(filepath):
    """Load a digital elevation model from a GeoTIFF as a 2D numpy array.

    Uses PIL (no rasterio dependency).

    Parameters
    ----------
    filepath : str or Path
        Path to the GeoTIFF (e.g. 'gilan_farm.tif').

    Returns
    -------
    np.ndarray
        2D array of elevations, shape (rows, cols).
    """
    return np.array(Image.open(filepath))


def normalize_elevation(elevation_2d):
    """Min-max normalize elevation to [0, 1].

    Parameters
    ----------
    elevation_2d : np.ndarray
        Raw elevation, shape (rows, cols).

    Returns
    -------
    np.ndarray
        Normalized elevation, same shape, values in [0, 1].
    """
    e_min = elevation_2d.min()
    e_max = elevation_2d.max()
    if e_max == e_min:
        return np.zeros_like(elevation_2d, dtype=float)
    return (elevation_2d - e_min) / (e_max - e_min)


def _pad_dem(elevation_2d):
    """Pad the DEM by 1 cell on each edge, extrapolating the slope.

    For each edge cell, the padded cell continues the slope from the
    nearest interior neighbor outward. Corner pads are the average of
    their two adjacent edge pads.

    Parameters
    ----------
    elevation_2d : np.ndarray
        Raw elevation, shape (rows, cols).

    Returns
    -------
    np.ndarray
        Padded elevation, shape (rows+2, cols+2).
    """
    rows, cols = elevation_2d.shape
    padded = np.zeros((rows + 2, cols + 2), dtype=float)
    padded[1:-1, 1:-1] = elevation_2d

    # Top row: continue slope outward from row 0
    for c in range(cols):
        slope = elevation_2d[0, c] - elevation_2d[min(1, rows - 1), c]
        padded[0, c + 1] = elevation_2d[0, c] + slope

    # Bottom row: continue slope outward from last row
    for c in range(cols):
        slope = elevation_2d[-1, c] - elevation_2d[max(-2, -rows), c]
        padded[-1, c + 1] = elevation_2d[-1, c] + slope

    # Left column: continue slope outward from col 0
    for r in range(rows):
        slope = elevation_2d[r, 0] - elevation_2d[r, min(1, cols - 1)]
        padded[r + 1, 0] = elevation_2d[r, 0] + slope

    # Right column: continue slope outward from last col
    for r in range(rows):
        slope = elevation_2d[r, -1] - elevation_2d[r, max(-2, -cols)]
        padded[r + 1, -1] = elevation_2d[r, -1] + slope

    # Corners: average of adjacent edge pads
    padded[0, 0] = (padded[0, 1] + padded[1, 0]) / 2
    padded[0, -1] = (padded[0, -2] + padded[1, -1]) / 2
    padded[-1, 0] = (padded[-1, 1] + padded[-2, 0]) / 2
    padded[-1, -1] = (padded[-1, -2] + padded[-2, -1]) / 2

    return padded


def build_directed_graph(elevation_2d):
    """Build the directed water-flow graph from an elevation map.

    Uses DEM padding to compute fractional routing at boundaries.
    Each cell sends water to its 8-connected (Moore neighborhood) neighbors
    that have strictly lower elevation. Agents are indexed in row-major
    order: agent n at position (row, col) has n = row * n_cols + col.

    Nr[n] counts ALL lower neighbors (internal + external/padded), so that
    phi2[n] / Nr[n] distributes runoff across the true number of downhill
    paths. sends_to[n] only lists internal neighbors, so the fraction
    routed to off-farm (padded) cells exits the system automatically.

    Parameters
    ----------
    elevation_2d : np.ndarray
        Raw elevation, shape (rows, cols).

    Returns
    -------
    dict
        {
            'gamma_flat':         np.ndarray, shape (N,), normalized elevation
            'sends_to':           dict[int, list[int]], agent -> internal
                                  downhill neighbors
            'Nr':                 dict[int, int], agent -> count of ALL
                                  downhill neighbors (internal + external)
            'Nr_internal':        dict[int, int], agent -> count of internal
                                  downhill neighbors only
            'topological_order':  np.ndarray, shape (N,), agents sorted
                                  high-to-low
            'rows':               int, grid rows
            'cols':               int, grid cols
            'N':                  int, total number of agents
            'elevation_flat':     np.ndarray, shape (N,), raw elevations
        }
    """
    rows, cols = elevation_2d.shape
    N = rows * cols

    # Normalize on the original (unpadded) grid for consistent gamma values
    gamma_2d = normalize_elevation(elevation_2d)

    # Pad the DEM to compute boundary routing
    padded = _pad_dem(elevation_2d)
    # Normalize the padded DEM for comparison (using padded min/max)
    padded_norm = normalize_elevation(padded)

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    sends_to = {}
    Nr = {}
    Nr_internal = {}

    for ri in range(rows):
        for ci in range(cols):
            n = ri * cols + ci
            # Position in padded grid
            pri, pci = ri + 1, ci + 1
            my_elev = padded_norm[pri, pci]

            internal_lower = []
            total_lower = 0

            for dr, dc in directions:
                nr2, nc2 = pri + dr, pci + dc
                # All neighbors in padded grid are valid (no bounds check needed)
                if padded_norm[nr2, nc2] < my_elev:
                    total_lower += 1
                    # Is this neighbor inside the real grid?
                    real_r, real_c = nr2 - 1, nc2 - 1
                    if 0 <= real_r < rows and 0 <= real_c < cols:
                        m = real_r * cols + real_c
                        internal_lower.append(m)

            sends_to[n] = internal_lower
            Nr[n] = total_lower
            Nr_internal[n] = len(internal_lower)

    elevation_flat = elevation_2d.flatten().astype(float)
    topological_order = np.argsort(-elevation_flat, kind='stable')

    return {
        'gamma_flat':        gamma_2d.flatten(),
        'sends_to':          sends_to,
        'Nr':                Nr,
        'Nr_internal':       Nr_internal,
        'topological_order': topological_order,
        'rows':              rows,
        'cols':              cols,
        'N':                 N,
        'elevation_flat':    elevation_flat,
    }


def get_sink_agents(graph):
    """Return list of agents with no downhill neighbors (Nr = 0).

    With fractional routing, true sinks (agents with no lower neighbors
    even in the padded DEM) should be rare or nonexistent on sloped terrain.

    Parameters
    ----------
    graph : dict
        Output of build_directed_graph.

    Returns
    -------
    list[int]
        Sink agent indices.
    """
    return [n for n in range(graph['N']) if graph['Nr'][n] == 0]


def get_hilltop_agents(graph, k=10):
    """Return the k highest-elevation agents.

    Parameters
    ----------
    graph : dict
        Output of build_directed_graph.
    k : int
        Number of hilltop agents to return.

    Returns
    -------
    list[int]
        Agent indices, ordered from highest to lowest elevation.
    """
    return graph['topological_order'][:k].tolist()


def load_terrain(filepath):
    """Convenience function: load DEM and build the full graph in one call.

    Parameters
    ----------
    filepath : str or Path
        Path to the GeoTIFF.

    Returns
    -------
    dict
        Same as build_directed_graph, plus:
            'elevation_2d': np.ndarray, raw 2D elevation map
    """
    elevation_2d = load_dem(filepath)
    graph = build_directed_graph(elevation_2d)
    graph['elevation_2d'] = elevation_2d
    return graph
