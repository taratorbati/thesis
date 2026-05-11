# =============================================================================
# src/precompute.py
# Cache deterministic, climate-only quantities used by the MPC.
#
# These quantities depend only on (scenario, crop), not on irrigation or
# state. By computing once per scenario-crop pair and caching to disk, the
# MPC's IPOPT symbolic graph is smaller and its per-iteration cost is lower.
#
# What is precomputed (per (scenario, crop)):
#
#   h1[k]    — daily thermal increment      = max(T_mean[k] - theta7, 0)
#   x2[k]    — cumulative thermal time      = x2_init + cumsum(h1)
#   h2[k]    — heat stress factor           ∈ [0, 1]  (1 = no stress)
#   h7[k]    — low-temp stress factor       ∈ {0, 1}  (1 if T_mean > theta7)
#   g_base[k]— growth function ASSUMING x3=0 throughout
#              In branch 1 (x2 <= theta18/2) g_base IS exact since g depends
#              only on x2. In branch 2 g_base is the no-stress baseline; the
#              true g may be lower if x3 has accumulated.
#   Kc_ET[k] — crop atmospheric demand      = Kc * ET0[k]      (mm/day)
#
# What is NOT precomputed (depends on x or u):
#   - x3 (maturity stress accumulator)
#   - h3 (drought stress, depends on transpiration)
#   - h6 (waterlogging stress, depends on x1)
#   - phi1, phi2, phi3, x1, x4, x5
#
# Cache file format: NumPy .npz with a metadata sidecar (.json).
# Cache key: 'precomputed_{scenario}_{crop}.npz' under results/precomputed/.
#
# Usage:
#     from src.precompute import get_precomputed
#     pre = get_precomputed('dry', 'rice')   # loads from cache, or computes & saves
#     pre.h1                                 # np.ndarray, shape (n_days,)
#     pre.x2                                 # np.ndarray, shape (n_days,)
#
# For arbitrary climate dicts (used by IrrigationEnv to compute precomputed
# quantities on-the-fly for the sampled training year, avoiding the previous
# Markov leak where all training years used the dry-year cached precomputed):
#
#     from src.precompute import compute_precomputed_from_climate
#     pre = compute_precomputed_from_climate(climate_dict, 'rice')
# =============================================================================

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from soil_data import get_crop
from climate_data import load_cleaned_data, extract_scenario_by_name


# Default cache directory — relative to project root by convention.
DEFAULT_CACHE_DIR = Path('results/precomputed')


@dataclass
class Precomputed:
    """Container for the climate-only precomputed arrays.

    All arrays are 1-D of length n_days = crop['season_days'].
    """
    scenario: str
    crop_name: str
    n_days: int
    h1: np.ndarray
    x2: np.ndarray
    h2: np.ndarray
    h7: np.ndarray
    g_base: np.ndarray
    Kc_ET: np.ndarray

    def __post_init__(self):
        # Defensive: convert lists to ndarrays if loaded from JSON-like sources
        for name in ('h1', 'x2', 'h2', 'h7', 'g_base', 'Kc_ET'):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                setattr(self, name, np.asarray(arr, dtype=float))

    def __repr__(self):
        return (
            f"Precomputed(scenario={self.scenario!r}, crop={self.crop_name!r}, "
            f"n_days={self.n_days}, "
            f"x2[final]={self.x2[-1]:.0f} GDD, "
            f"Kc_ET total={self.Kc_ET.sum():.0f} mm)"
        )


# ── Math (climate-only) ──────────────────────────────────────────────────────

def _compute_h1(temp_mean, theta7):
    """Daily thermal increment: max(T_mean - theta7, 0)."""
    return np.maximum(temp_mean - theta7, 0.0)


def _compute_x2(h1, x2_init):
    """Cumulative thermal time, starting from x2_init."""
    return x2_init + np.cumsum(h1)


def _compute_h2(temp_max, theta9, theta10):
    """Heat stress factor.

    h2 = 1                                 if T_max <= theta9
       = 1 - (T_max - theta9)/(theta10 - theta9)  if theta9 < T_max <= theta10
       = 0                                 if T_max > theta10
    """
    h2 = np.where(
        temp_max <= theta9,
        1.0,
        np.where(
            temp_max <= theta10,
            1.0 - (temp_max - theta9) / max(theta10 - theta9, 1e-9),
            0.0,
        ),
    )
    return np.clip(h2, 0.0, 1.0)


def _compute_h7(temp_mean, theta7):
    """Low-temperature stress (binary): 1 if T_mean > theta7 else 0."""
    return (temp_mean > theta7).astype(float)


def _compute_g_baseline(x2, theta18, theta19, theta20):
    """Growth function value assuming x3 = 0 throughout the season.

    g(x2)  for x2 <= theta18/2:  theta19 / (1 + exp(-0.01*(x2 - theta20)))
    g(x2)  for x2  > theta18/2:  theta19 / (1 + exp( 0.01*(x2 + 0    - theta18)))
                                 (i.e. x3 set to zero in the senescence branch)

    In the first branch this is exact (g depends only on x2).
    In the second branch this is the no-stress upper bound on g; actual g may
    be lower if x3 has accumulated by then.
    """
    half = theta18 / 2.0
    g = np.where(
        x2 <= half,
        theta19 / (1.0 + np.exp(-0.01 * (x2 - theta20))),
        theta19 / (1.0 + np.exp(0.01 * (x2 - theta18))),
    )
    return g


def _compute_kc_et(et0, kc):
    """Crop-specific atmospheric demand: Kc * ET0."""
    return kc * et0


# ── Public API ────────────────────────────────────────────────────────────────

def compute_precomputed_from_climate(climate, crop_name, scenario_tag='custom'):
    """Compute precomputed bundle from an arbitrary climate dict.

    This is the year-agnostic core; takes whatever climate the caller has
    in hand and produces the matching precomputed quantities. Used by
    IrrigationEnv during training to keep precomputed in sync with the
    sampled training year's climate.

    Parameters
    ----------
    climate : dict
        Must contain keys 'temp_mean', 'temp_max', 'ET' (each at least
        crop['season_days'] long).
    crop_name : str
        'rice' or 'tobacco'.
    scenario_tag : str
        Label stored in the returned Precomputed.scenario field. Default
        'custom'. For named-scenario callers, pass the scenario name.

    Returns
    -------
    Precomputed
    """
    crop = get_crop(crop_name)
    n_days = crop['season_days']

    if len(climate['temp_mean']) < n_days:
        raise ValueError(
            f"Climate data has {len(climate['temp_mean'])} days but crop "
            f"{crop_name!r} expects {n_days}."
        )

    temp_mean = np.asarray(climate['temp_mean'][:n_days], dtype=float)
    temp_max  = np.asarray(climate['temp_max'][:n_days],  dtype=float)
    et0       = np.asarray(climate['ET'][:n_days],        dtype=float)

    h1 = _compute_h1(temp_mean, crop['theta7'])
    x2 = _compute_x2(h1, crop.get('x2_init', 0.0))
    h2 = _compute_h2(temp_max, crop['theta9'], crop['theta10'])
    h7 = _compute_h7(temp_mean, crop['theta7'])
    g_base = _compute_g_baseline(x2, crop['theta18'], crop['theta19'], crop['theta20'])
    Kc_ET = _compute_kc_et(et0, crop.get('Kc', 1.0))

    return Precomputed(
        scenario=scenario_tag,
        crop_name=crop_name,
        n_days=n_days,
        h1=h1, x2=x2, h2=h2, h7=h7,
        g_base=g_base, Kc_ET=Kc_ET,
    )


def compute_precomputed(scenario, crop_name, df=None):
    """Compute the precomputed bundle for (scenario, crop). Does NOT cache.

    Thin wrapper that resolves a named scenario to a climate dict and
    delegates to compute_precomputed_from_climate.

    Parameters
    ----------
    scenario : str
        'dry', 'moderate', or 'wet'.
    crop_name : str
        'rice' or 'tobacco'.
    df : pd.DataFrame or None
        Cleaned climate dataframe. If None, loads from disk.

    Returns
    -------
    Precomputed
    """
    crop = get_crop(crop_name)
    if df is None:
        df = load_cleaned_data()
    climate = extract_scenario_by_name(df, scenario, crop)
    return compute_precomputed_from_climate(climate, crop_name, scenario_tag=scenario)


def cache_path(scenario, crop_name, cache_dir=DEFAULT_CACHE_DIR):
    """Where the cached .npz lives for this scenario/crop pair."""
    cache_dir = Path(cache_dir)
    return cache_dir / f"precomputed_{scenario}_{crop_name}.npz"


def metadata_path(scenario, crop_name, cache_dir=DEFAULT_CACHE_DIR):
    return cache_path(scenario, crop_name, cache_dir).with_suffix('.json')


def save_precomputed(pre, cache_dir=DEFAULT_CACHE_DIR):
    """Save a Precomputed bundle to .npz + .json sidecar."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_path(pre.scenario, pre.crop_name, cache_dir)
    np.savez_compressed(
        npz_path,
        h1=pre.h1, x2=pre.x2, h2=pre.h2, h7=pre.h7,
        g_base=pre.g_base, Kc_ET=pre.Kc_ET,
    )

    meta = {
        'scenario': pre.scenario,
        'crop_name': pre.crop_name,
        'n_days': pre.n_days,
        'arrays': ['h1', 'x2', 'h2', 'h7', 'g_base', 'Kc_ET'],
        'final_x2': float(pre.x2[-1]),
        'total_Kc_ET_mm': float(pre.Kc_ET.sum()),
        'mean_h2': float(pre.h2.mean()),
        'mean_h7': float(pre.h7.mean()),
    }
    with open(metadata_path(pre.scenario, pre.crop_name, cache_dir), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def load_precomputed(scenario, crop_name, cache_dir=DEFAULT_CACHE_DIR):
    """Load a Precomputed bundle from disk. Returns None if not cached."""
    npz_path = cache_path(scenario, crop_name, cache_dir)
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    return Precomputed(
        scenario=scenario,
        crop_name=crop_name,
        n_days=int(data['h1'].shape[0]),
        h1=data['h1'],
        x2=data['x2'],
        h2=data['h2'],
        h7=data['h7'],
        g_base=data['g_base'],
        Kc_ET=data['Kc_ET'],
    )


def get_precomputed(scenario, crop_name, cache_dir=DEFAULT_CACHE_DIR, force=False):
    """Return the Precomputed bundle, computing and caching it if needed.

    Parameters
    ----------
    scenario : str
    crop_name : str
    cache_dir : Path
    force : bool
        If True, recompute even if a cached version exists.

    Returns
    -------
    Precomputed
    """
    if not force:
        cached = load_precomputed(scenario, crop_name, cache_dir)
        if cached is not None:
            return cached

    pre = compute_precomputed(scenario, crop_name)
    save_precomputed(pre, cache_dir)
    return pre
