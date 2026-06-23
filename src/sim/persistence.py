# =============================================================================
# src/persistence.py
# Parquet I/O and skip-if-exists logic for experiment results.
#
# Each run produces a single .parquet file (long format: one row per
# day-agent pair) and a sidecar .json file with metadata.
#
# Long-format schema:
#     day, agent, x1, x2, x3, x4, x5, u, rainfall, et0, budget_remaining
#
# Metadata (sidecar JSON):
#     scenario, year, crop, controller, budget_total, run_seed,
#     solve_times (list, optional), final_metrics (dict), config_snapshot,
#     wallclock_seconds, completed_at (ISO timestamp)
# =============================================================================

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ── Path helpers ──────────────────────────────────────────────────────────────

def ensure_dir(path):
    """Create directory and any missing parents. No error if it already exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def metadata_path_for(filepath):
    """Return the sidecar JSON path that pairs with a .parquet file."""
    p = Path(filepath)
    return p.with_suffix('.json')


def should_skip(filepath, force=False):
    """True if the parquet file exists and we are not forcing recomputation.

    Used by experiment scripts to skip already-completed runs.

    Parameters
    ----------
    filepath : str or Path
        Path to the expected output .parquet file.
    force : bool
        If True, always return False (do not skip).

    Returns
    -------
    bool
    """
    if force:
        return False
    return Path(filepath).exists()


# ── Saving runs ───────────────────────────────────────────────────────────────

def save_run(filepath, trajectory, metadata):
    """Save a season trajectory to a parquet file with a JSON metadata sidecar.

    Parameters
    ----------
    filepath : str or Path
        Output path. The .parquet extension is added if missing. A matching
        .json file is written alongside.
    trajectory : dict
        Required keys:
            'x1', 'x2', 'x3', 'x4', 'x5' : np.ndarray of shape (T, N)
            'u'                          : np.ndarray of shape (T, N)
            'rainfall'                   : np.ndarray of shape (T,)
            'et0'                        : np.ndarray of shape (T,)
            'budget_remaining'           : np.ndarray of shape (T,)
    metadata : dict
        Free-form dict; must be JSON-serializable. Recommended keys:
            scenario, year, crop, controller, budget_total, seed,
            solve_times (list[float]), final_metrics (dict),
            config_snapshot (dict), wallclock_seconds (float)

    Notes
    -----
    The 'completed_at' ISO timestamp is added automatically.
    """
    filepath = Path(filepath)
    if filepath.suffix != '.parquet':
        filepath = filepath.with_suffix('.parquet')
    ensure_dir(filepath.parent)

    df = trajectory_to_long_df(trajectory)
    df.to_parquet(filepath, engine='pyarrow', compression='zstd', index=False)

    meta_path = metadata_path_for(filepath)
    full_metadata = dict(metadata)
    full_metadata['completed_at'] = datetime.now().isoformat(timespec='seconds')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(full_metadata, f, indent=2, default=_json_default)


def _json_default(obj):
    """JSON serializer for numpy types and Path objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def trajectory_to_long_df(trajectory):
    """Convert a (T, N) trajectory dict to a long-format DataFrame.

    See save_run() for the expected keys in trajectory.

    Returns
    -------
    pd.DataFrame
        Columns: day, agent, x1, x2, x3, x4, x5, u, rainfall, et0,
                 budget_remaining
        Length: T * N rows.
    """
    x1 = np.asarray(trajectory['x1'])
    T, N = x1.shape

    days_grid, agents_grid = np.meshgrid(
        np.arange(T), np.arange(N), indexing='ij'
    )

    state_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'u']
    data = {
        'day':   days_grid.ravel(),
        'agent': agents_grid.ravel(),
    }
    for col in state_cols:
        arr = np.asarray(trajectory[col])
        if arr.shape != (T, N):
            raise ValueError(
                f"trajectory['{col}'] has shape {arr.shape}, expected ({T}, {N})"
            )
        data[col] = arr.ravel()

    # Per-day scalars broadcast across agents
    for col in ('rainfall', 'et0', 'budget_remaining'):
        arr = np.asarray(trajectory[col])
        if arr.shape != (T,):
            raise ValueError(
                f"trajectory['{col}'] has shape {arr.shape}, expected ({T},)"
            )
        data[col] = np.repeat(arr, N)

    return pd.DataFrame(data)


# ── Loading runs ──────────────────────────────────────────────────────────────

def load_run(filepath):
    """Load a saved run.

    Parameters
    ----------
    filepath : str or Path
        Path to the .parquet file (with or without extension).

    Returns
    -------
    df : pd.DataFrame
        Long-format trajectory.
    metadata : dict
        Contents of the sidecar JSON.
    """
    filepath = Path(filepath)
    if filepath.suffix != '.parquet':
        filepath = filepath.with_suffix('.parquet')

    df = pd.read_parquet(filepath, engine='pyarrow')

    meta_path = metadata_path_for(filepath)
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return df, metadata


def long_df_to_trajectory(df):
    """Inverse of trajectory_to_long_df: reshape long DataFrame back to (T, N).

    Returns
    -------
    dict
        Same keys as the input to save_run.
    """
    T = int(df['day'].max()) + 1
    N = int(df['agent'].max()) + 1

    df = df.sort_values(['day', 'agent']).reset_index(drop=True)

    state_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'u']
    out = {}
    for col in state_cols:
        out[col] = df[col].to_numpy().reshape(T, N)

    # Per-day scalars: take first agent's value for each day
    daily = df[df['agent'] == 0].sort_values('day')
    for col in ('rainfall', 'et0', 'budget_remaining'):
        out[col] = daily[col].to_numpy()

    return out


# ── Lightweight checkpointing for long MPC runs ──────────────────────────────

def save_partial(filepath, trajectory_partial, day_completed, metadata):
    """Save a mid-run checkpoint for crash recovery.

    Writes to {filepath.stem}_partial.parquet so the final filepath is reserved
    for completed runs.

    Parameters
    ----------
    filepath : str or Path
        Final destination path of the completed run.
    trajectory_partial : dict
        Same keys as a full trajectory, but arrays are sized (day_completed+1, N).
    day_completed : int
        Last fully completed day (inclusive).
    metadata : dict
        Same as for save_run, plus this function adds 'last_completed_day'.
    """
    filepath = Path(filepath)
    partial_path = filepath.with_name(filepath.stem + '_partial.parquet')
    ensure_dir(partial_path.parent)

    df = trajectory_to_long_df(trajectory_partial)
    df.to_parquet(partial_path, engine='pyarrow', compression='zstd', index=False)

    meta_path = partial_path.with_suffix('.json')
    full_metadata = dict(metadata)
    full_metadata['last_completed_day'] = int(day_completed)
    full_metadata['completed_at'] = None  # not yet complete
    full_metadata['checkpointed_at'] = datetime.now().isoformat(timespec='seconds')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(full_metadata, f, indent=2, default=_json_default)


def load_partial(filepath):
    """Load a partial run checkpoint, if it exists.

    Parameters
    ----------
    filepath : str or Path
        Final destination path of the completed run (not the partial file).

    Returns
    -------
    (df, metadata) tuple if partial exists, else None.
    """
    filepath = Path(filepath)
    partial_path = filepath.with_name(filepath.stem + '_partial.parquet')
    if not partial_path.exists():
        return None
    return load_run(partial_path)


def discard_partial(filepath):
    """Remove the partial checkpoint files for a given run.

    Called once a run completes successfully.
    """
    filepath = Path(filepath)
    partial_path = filepath.with_name(filepath.stem + '_partial.parquet')
    partial_meta = partial_path.with_suffix('.json')
    if partial_path.exists():
        partial_path.unlink()
    if partial_meta.exists():
        partial_meta.unlink()
