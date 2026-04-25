# =============================================================================
# src/terrain.py
# Single source of truth for DEM loading and directed-graph construction.
# Replaces the duplicated build_directed_graph in cross_validate_gwetroot.py,
# run_comparison.py, and run_comparison_2.py.
# =============================================================================

from pathlib import Path

import numpy as np
from PIL import Image


def load_dem(filepath):
    """Load a digital elevation model from a GeoTIFF as a 2D numpy array.

    Uses PIL (no rasterio dependency). Matches the existing approach in
    cross_validate_gwetroot.py and run_comparison.py.

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


def build_directed_graph(elevation_2d):
    """Build the directed water-flow graph from an elevation map.

    Each cell sends water to its 8-connected (Moore neighborhood) neighbors
    that have strictly lower normalized elevation. Agents are indexed in
    row-major order: agent n at position (row, col) has n = row * n_cols + col.

    Parameters
    ----------
    elevation_2d : np.ndarray
        Raw elevation, shape (rows, cols).

    Returns
    -------
    dict
        {
            'gamma_flat':         np.ndarray, shape (N,), normalized elevation
            'sends_to':           dict[int, list[int]], agent -> downhill neighbors
            'Nr':                 dict[int, int], agent -> count of downhill neighbors
            'topological_order':  np.ndarray, shape (N,), agents sorted high-to-low
            'rows':               int, grid rows
            'cols':               int, grid cols
            'N':                  int, total number of agents
            'elevation_flat':     np.ndarray, shape (N,), raw elevations
        }

    Notes
    -----
    Agents with Nr=0 are sink agents (they have no lower neighbors). They
    receive runoff but cannot send any.

    The topological_order is a stable sort by descending elevation, which
    guarantees that for the cascade routing mode in CropSoilABM, every
    sender is processed before any of its receivers within the same day.
    """
    rows, cols = elevation_2d.shape
    gamma_2d = normalize_elevation(elevation_2d)

    sends_to = {}
    Nr = {}

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for ri in range(rows):
        for ci in range(cols):
            n = ri * cols + ci
            lower = []
            for dr, dc in directions:
                nr2, nc2 = ri + dr, ci + dc
                if 0 <= nr2 < rows and 0 <= nc2 < cols:
                    if gamma_2d[nr2, nc2] < gamma_2d[ri, ci]:
                        m = nr2 * cols + nc2
                        lower.append(m)
            sends_to[n] = lower
            Nr[n] = len(lower)

    elevation_flat = elevation_2d.flatten()
    topological_order = np.argsort(-elevation_flat, kind='stable')

    return {
        'gamma_flat':        gamma_2d.flatten(),
        'sends_to':          sends_to,
        'Nr':                Nr,
        'topological_order': topological_order,
        'rows':              rows,
        'cols':              cols,
        'N':                 rows * cols,
        'elevation_flat':    elevation_flat,
    }


def get_sink_agents(graph):
    """Return list of agents with no downhill neighbors (Nr = 0).

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
