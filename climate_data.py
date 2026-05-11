# =============================================================================
# climate_data.py
# Load cleaned climate data and extract crop-season scenarios.
# Reads from climate_apr_oct_cleaned.csv (produced by preprocess.py).
# Uses Penman-Monteith ET0.
#
# Year split (thesis, Chapter 4) — 20 train / 3 dev / 3 test:
#   SCENARIO_YEARS (TEST, 3 years) — named evaluation scenarios used only
#                   for the final headline thesis comparison. Touched once
#                   at the end of training:
#                     'dry'      (2022) — 39.7 mm, in-distribution
#                     'moderate' (2018) — 108.8 mm, upper training edge
#                     'wet'      (2024) — 176.8 mm, OOD extreme
#   DEV_YEARS      (DEV, 3 years) — stratified by rainfall tercile,
#                   used by EvalCallback during training for best_model
#                   selection and learning-curve monitoring:
#                     2002 — 27.1 mm, low-rainfall tercile
#                     2016 — 77.0 mm, mid-rainfall tercile
#                     2023 — 88.4 mm, high-rainfall tercile (non-extreme)
#   TRAINING_YEARS (TRAIN, 20 years) — all remaining years 2000-2025.
#                   Sampled uniformly at each SAC episode reset().
#
# NOTE on scenario key naming: the test scenarios are keyed 'dry',
# 'moderate', 'wet' (not 'moderate_wet') to match the precomputed cache
# filenames (precomputed_<key>_rice.npz) and all analysis/plotting scripts.
# =============================================================================

import pandas as pd
import numpy as np


DATA_CSV = 'results/preprocessing/climate_apr_oct_cleaned.csv'

# ── Test scenarios (held out, touched only for final thesis comparison) ──────
# Keys must match the precomputed cache filenames: precomputed_<key>_rice.npz
SCENARIO_YEARS = {
    'dry':      2022,   # P35 — 39.7 mm, in-distribution
    'moderate': 2018,   # P80 — 108.8 mm, upper edge of training distribution
    'wet':      2024,   # P96 — 176.8 mm, OOD extreme (3 R20 events)
}

# ── Test years (3) — final-evaluation only ───────────────────────────────────
# Frozenset of integer years for fast membership tests in gym_env.py etc.
EVAL_YEARS = frozenset(SCENARIO_YEARS.values())   # {2018, 2022, 2024}

# ── Dev years (3) — used by EvalCallback during training ─────────────────────
# Chosen to span the training distribution by rainfall tercile so that the
# in-training eval reward is a meaningful generalization signal:
#   - 2002 (27.1 mm) — low-rainfall tercile
#   - 2016 (77.0 mm) — mid-rainfall tercile
#   - 2023 (88.4 mm) — high-rainfall tercile (non-extreme)
# Each dev year is at least 20 mm seasonal rainfall away from any test year
# to avoid leaking near-duplicate climate into best_model selection.
DEV_YEARS = (2002, 2016, 2023)
DEV_YEARS_SET = frozenset(DEV_YEARS)

# ── Training years (20) — sampled uniformly per SAC episode ──────────────────
# All years 2000-2025 excluding eval (test) years AND dev years.
TRAINING_YEARS = tuple(sorted(
    y for y in range(2000, 2026)
    if y not in EVAL_YEARS and y not in DEV_YEARS_SET
))
# = (2000,2001,2003,2004,2005,2006,2007,2008,2009,2010,
#    2011,2012,2013,2014,2015,2017,2019,2020,2021,2025)
# (20 years)


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
    """Extract a named test scenario ('dry', 'moderate', or 'wet').

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
    print(f"Dev years ({len(DEV_YEARS)}): {DEV_YEARS}")
    print(f"Test scenarios: {SCENARIO_YEARS}")
    print()
    for label, years in [('TRAIN', TRAINING_YEARS),
                         ('DEV',   DEV_YEARS),
                         ('TEST',  tuple(SCENARIO_YEARS.values()))]:
        print(f"{label}:")
        for y in years:
            c = extract_scenario(df, y, crop)
            print(f"  {y}: {c['rainfall'].sum():.1f} mm")
