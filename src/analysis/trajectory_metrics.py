# =============================================================================
# src/analysis_lib/trajectory_metrics.py
# Dependency-light helpers for post-hoc analysis of saved run trajectories.
#
# These read the parquet/JSON pairs written by src/runner.run_season and
# compute derived control-quality metrics that are NOT stored in the JSON
# sidecar (notably mean|Delta u|, the temporal control-effort / "pulsing"
# metric reported in the thesis comparison table).
#
# Used by:
#   - scripts/analysis/ema_pareto.py        (future-work item #5, EMA smoothing)
#   - (reusable for item #1 robustness sweeps and any later trajectory study)
#
# WHY a separate module: the |Delta u| definition must be IDENTICAL everywhere
# it is reported, otherwise Pareto curves and the headline table disagree.  It
# is defined once, here, and validated against the thesis value (seed-0 9-cell
# mean = 2.35) in the module self-test at the bottom.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Trajectory pivot ─────────────────────────────────────────────────────────

def load_action_matrix(parquet_path) -> np.ndarray:
    """Load the applied irrigation trajectory as a dense (T, N) matrix.

    The runner writes trajectories in LONG format with one row per
    (day, agent).  This pivots the ``u`` column to a (season_days, N_agents)
    array of the *applied* per-agent action in mm/day (i.e. after the runner's
    actuator clip and budget enforcement), which is the quantity the thesis
    control-effort metric is computed on.

    Parameters
    ----------
    parquet_path : str or Path
        Path to a run parquet written by ``src.runner.run_season``.

    Returns
    -------
    np.ndarray, shape (T, N)
        Applied action, day-major.
    """
    df = pd.read_parquet(parquet_path)
    if 'u' not in df.columns:
        raise KeyError(f"{parquet_path}: no 'u' column (cols={list(df.columns)})")
    U = df.pivot(index='day', columns='agent', values='u').to_numpy()
    return np.asarray(U, dtype=float)


# ── Control-effort / pulsing metric ─────────────────────────────────────────

def mean_abs_delta_u(U: np.ndarray) -> float:
    """Mean absolute temporal first-difference of the action (the "pulsing").

    Definition (matches the thesis comparison table, column "mean |Delta u|"):

        mean_{n} mean_{t>=1} | u[t, n] - u[t-1, n] |

    i.e. the average over agents AND days of the day-to-day change in the
    applied per-agent irrigation depth, in mm/day.  A perfectly steady
    controller scores 0; the more the controller pulses on/off, the larger the
    value.  MPC (smooth) scores ~0.97; TD3 v2.21c scores ~2.50 (3-seed mean).

    Parameters
    ----------
    U : np.ndarray, shape (T, N)
        Applied action matrix from :func:`load_action_matrix`.

    Returns
    -------
    float
        Mean |Delta u| in mm/day.  Returns 0.0 for T < 2.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError(f"expected 2-D (T, N) action matrix, got shape {U.shape}")
    if U.shape[0] < 2:
        return 0.0
    dU = np.abs(np.diff(U, axis=0))   # (T-1, N)
    return float(dU.mean())


def mean_abs_delta_u_from_parquet(parquet_path) -> float:
    """Convenience wrapper: load a run parquet and return its mean |Delta u|."""
    return mean_abs_delta_u(load_action_matrix(parquet_path))


# ── JSON-sidecar metric access ───────────────────────────────────────────────

def read_final_metrics(json_path) -> dict:
    """Read the ``final_metrics`` dict from a run's JSON sidecar (utf-8)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return meta.get('final_metrics', {})


# ── MPC reference table (for %MPC normalisation) ─────────────────────────────

def build_mpc_reference(
    runs_dir,
    horizon: int = 8,
    crop: str = 'rice',
) -> dict:
    """Build a per-cell MPC reference-yield lookup for %MPC normalisation.

    Scans ``runs_dir`` for the perfect-forecast MPC sidecars
    (``mpc_perfect_<scenario>_<crop>_<budget>pct_Hp<horizon>.json``) and
    returns ``{(scenario, budget_pct): yield_kg_ha}``.  The thesis normalises
    RL yield against MPC Hp8 (perfect) per cell, so the default horizon is 8.

    Parameters
    ----------
    runs_dir : str or Path
        Directory containing the MPC run sidecars (usually results/runs/).
    horizon : int
        MPC prediction horizon to use as the reference (default 8).
    crop : str
        Crop tag in the filename (default 'rice').

    Returns
    -------
    dict
        {(scenario:str, budget_pct:int): yield_kg_ha:float}
    """
    runs_dir = Path(runs_dir)
    ref: dict = {}
    pattern = f"mpc_perfect_*_{crop}_*pct_Hp{horizon}.json"
    for jp in sorted(runs_dir.glob(pattern)):
        # filename: mpc_perfect_<scenario>_<crop>_<budget>pct_Hp<H>.json
        stem = jp.stem  # drop .json
        parts = stem.split('_')
        # parts = ['mpc','perfect',<scenario>,<crop>,'<budget>pct','Hp<H>']
        try:
            scenario = parts[2]
            budget_pct = int(parts[4].replace('pct', ''))
        except (IndexError, ValueError):
            continue
        fm = read_final_metrics(jp)
        y = fm.get('yield_kg_ha')
        if y is not None:
            ref[(scenario, budget_pct)] = float(y)
    return ref


def pct_of_mpc(
    yield_kg_ha: float,
    scenario: str,
    budget_pct: int,
    mpc_ref: dict,
) -> Optional[float]:
    """Return 100 * yield / MPC_reference for one cell, or None if no ref."""
    ref = mpc_ref.get((scenario, int(budget_pct)))
    if ref is None or ref <= 0:
        return None
    return 100.0 * float(yield_kg_ha) / ref


# ── Module self-test (validates the |Delta u| definition on real data) ───────

if __name__ == '__main__':
    import glob

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    seed0_dir = PROJECT_ROOT / 'results' / 'runs' / 'td3_v221c_seed0_best_model'
    cells = sorted(glob.glob(str(seed0_dir / 'sac_perfect_det_*_seed0.parquet')))
    if not cells:
        print("No v2.21c seed-0 parquet cells found; skipping self-test.")
    else:
        vals = []
        for c in cells:
            v = mean_abs_delta_u_from_parquet(c)
            vals.append(v)
            name = Path(c).stem.replace('sac_perfect_det_', '').replace('_seed0', '')
            print(f"  {name:28s} mean|du| = {v:.3f}")
        mean9 = float(np.mean(vals))
        print(f"\n  seed-0 9-cell mean |du| = {mean9:.3f}  "
              f"(thesis reports seed-0 = 2.35)")
        assert abs(mean9 - 2.35) < 0.05, "metric drifted from thesis definition!"
        print("  OK: matches the thesis control-effort definition.")
