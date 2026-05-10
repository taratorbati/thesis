# =============================================================================
# climate_data.py
# Load cleaned climate data and extract crop-season scenarios.
# Reads from climate_apr_oct_cleaned.csv (produced by preprocess.py).
# Uses Penman-Monteith ET0.
#
# Scenario split (thesis, Chapter 4):
#   SCENARIO_YEARS — three named evaluation scenarios, held out from SAC
#                    training, used for both MPC and SAC final evaluation:
#                      'dry'      (2022) — 39.7 mm, in-distribution
#                      'moderate' (2018) — 108.8 mm, upper training edge
#                      'wet'      (2024) — 176.8 mm, OOD extreme
#   TRAINING_YEARS — 23 years (2000-2025 minus eval years), sampled
#                    uniformly at each SAC episode reset.
#   EVAL_YEARS    — frozenset of the three eval year integers, used by
#                   gym_env.py to exclude them from the training pool.
#
# NOTE: The scenario key 'moderate' corresponds to year 2018.
# Earlier versions of this file used 'moderate_wet' — this was renamed to
# 'moderate' for consistency with the precomputed cache files
# (precomputed_moderate_rice.npz) and all analysis/plotting scripts.
# =============================================================================

import pandas as pd
import numpy as np


DATA_CSV = 'results/preprocessing/climate_apr_oct_cleaned.csv'

# ── Evaluation scenarios (held out — never seen during SAC training) ─────────
# Keys must match the precomputed cache filenames: precomputed_<key>_rice.npz
SCENARIO_YEARS = {
    'dry':      2022,   # P35 — 39.7 mm, in-distribution
    'moderate': 2018,   # P80 — 108.8 mm, upper edge of training distribution
    'wet':      2024,   # P96 — 176.8 mm, OOD extreme (3 R20 events)
}

# ── Training years (23 years, sampled uniformly per SAC episode) ──────────────
EVAL_YEARS = frozenset(SCENARIO_YEARS.values())   # {2018, 2022, 2024}
TRAINING_YEARS = sorted(
    y for y in range(2000, 2026) if y not in EVAL_YEARS
)
# = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,
#    2010,2011,2012,2013,2014,2015,2016,2017,2019,2020,
#    2021,2023,2025]  (23 years)


def load_cleaned_data(filepath=DATA_CSV):
    """Load the cleaned April-October climate CSV produced by preprocess.py."""
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df


def extract_scenario(df, year, crop, start_doy=None, n_days=None):
    """Extract crop-season climate data for a given year and crop."""
    if start_doy is None:
        start_doy = crop['season_start_doy']
    if n_days is None:
        n_days = crop['season_days']

    df_year = df[df['YEAR'] == year].copy()
    df_season = df_year[df_year['DOY'] >= start_doy].head(n_days)
    df_season = df_season.reset_index(drop=True)

    if len(df_season) < n_days:
        print(f"WARNING: Only {len(df_season)} days found for year {year}, "
              f"expected {n_days}")

    return {
        'rainfall':  df_season['PRECTOTCORR'].values,
        'temp_mean': df_season['T2M'].values,
        'temp_max':  df_season['T2M_MAX'].values,
        'temp_min':  df_season['T2M_MIN'].values,
        'radiation': df_season['ALLSKY_SFC_SW_DWN'].values,
        'ET':        df_season['ET0_penman_monteith'].values,
        'humidity':  df_season['RH2M'].values,
        'wind':      df_season['WS2M'].values,
        'gwetroot':  df_season['GWETROOT'].values,
        'gwettop':   df_season['GWETTOP'].values,
    }


def extract_scenario_by_name(df, scenario_name, crop, **kwargs):
    """Extract a named evaluation scenario ('dry', 'moderate', or 'wet').

    Parameters
    ----------
    scenario_name : str
        One of 'dry' (2022), 'moderate' (2018), or 'wet' (2024).
    """
    key = scenario_name.lower().strip()
    if key not in SCENARIO_YEARS:
        available = ', '.join(sorted(SCENARIO_YEARS.keys()))
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {available}"
        )
    return extract_scenario(df, SCENARIO_YEARS[key], crop, **kwargs)


if __name__ == '__main__':
    from soil_data import get_crop
    crop = get_crop('rice')
    df = load_cleaned_data()
    print(f"Training years ({len(TRAINING_YEARS)}): {TRAINING_YEARS}")
    print(f"Eval scenarios: {SCENARIO_YEARS}")
    for name, year in SCENARIO_YEARS.items():
        c = extract_scenario(df, year, crop)
        print(f"  {name} ({year}): {c['rainfall'].sum():.1f} mm")
