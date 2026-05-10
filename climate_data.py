# =============================================================================
# climate_data.py
# Load cleaned climate data and extract crop-season scenarios.
# Reads from climate_apr_oct_cleaned.csv (produced by preprocess.py).
# Uses Penman-Monteith ET0.
#
# Year split (v2.3):
#   EVAL_YEARS    — three held-out years, never seen during SAC training:
#                     2022 (39.7mm)  — dry, near training mean
#                     2018 (108.8mm) — moderate-wet, upper edge of training dist
#                     2024 (176.8mm) — extreme wet, out-of-distribution
#                   These are also the years used for all MPC nominal runs,
#                   giving a consistent 9-cell evaluation grid (3 years × 3
#                   budgets) across all controllers.
#   TRAINING_YEARS — all remaining years 2000-2025 (23 years). Sampled
#                   uniformly at each episode reset during SAC training.
# =============================================================================

import pandas as pd
import numpy as np


DATA_CSV = 'results/preprocessing/climate_apr_oct_cleaned.csv'

# Canonical scenario names → eval year mapping
SCENARIO_YEARS = {
    'dry':          2022,   # P35 — 39.7 mm rice-season rainfall
    'moderate_wet': 2018,   # P80 — 108.8 mm, upper edge of training distribution
    'wet':          2024,   # P96 — 176.8 mm rice-season rainfall (3 R20 events)
}

# Held-out eval years. Never used during SAC training.
EVAL_YEARS = frozenset(SCENARIO_YEARS.values())   # {2018, 2022, 2024}

# Training years: all 2000-2025 except the three eval years (23 total).
# 2025 confirmed present in dataset (93-day season, 46.5mm rainfall).
TRAINING_YEARS = tuple(
    y for y in range(2000, 2026) if y not in EVAL_YEARS
)


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
    """Extract a named scenario ('dry', 'moderate_wet', 'wet') for a crop."""
    key = scenario_name.lower().strip()
    if key not in SCENARIO_YEARS:
        available = ', '.join(sorted(SCENARIO_YEARS.keys()))
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. Available: {available}"
        )
    return extract_scenario(df, SCENARIO_YEARS[key], crop, **kwargs)


if __name__ == '__main__':
    from soil_data import get_crop
    crop = get_crop('rice')
    df = load_cleaned_data()
    print(f"Training years ({len(TRAINING_YEARS)}): {TRAINING_YEARS}")
    print(f"Eval years: {sorted(EVAL_YEARS)}")
    print()
    for name, year in SCENARIO_YEARS.items():
        clim = extract_scenario(df, year, crop)
        print(f"{name} ({year}): {clim['rainfall'].sum():.1f}mm rainfall")
