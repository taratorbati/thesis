# =============================================================================
# climate_data.py
# Load cleaned climate data and extract crop-season scenarios.
# Reads from climate_apr_oct_cleaned.csv (produced by preprocess.py).
# Uses Penman-Monteith ET0 and season windows from soil_data.py.
# =============================================================================

import pandas as pd
import numpy as np
from soil_data import theta as crop_params


DATA_CSV = 'results/preprocessing/climate_apr_oct_cleaned.csv'

SCENARIO_YEARS = {
    'dry':      2022,
    'moderate': 2020,
    'wet':      2024,
}


def load_cleaned_data(filepath=DATA_CSV):
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df


def extract_scenario(df, year, start_doy=None, n_days=None):
    """
    Extract crop season climate data for a given year.
    Uses season window from soil_data.py if not specified.
    Returns dict of daily arrays for the ABM.
    """
    if start_doy is None:
        start_doy = crop_params['season_start_doy']
    if n_days is None:
        n_days = crop_params['season_days']

    df_year = df[df['YEAR'] == year].copy()
    df_season = df_year[df_year['DOY'] >= start_doy].head(n_days)
    df_season = df_season.reset_index(drop=True)

    if len(df_season) < n_days:
        print(f"WARNING: Only {len(df_season)} days found for year {year}, "
              f"expected {n_days}")

    climate = {
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
    return climate


# ── Load data and extract scenarios ───────────────────────────────────────────

df = load_cleaned_data()

climate_dry      = extract_scenario(df, SCENARIO_YEARS['dry'])
climate_moderate = extract_scenario(df, SCENARIO_YEARS['moderate'])
climate_wet      = extract_scenario(df, SCENARIO_YEARS['wet'])


if __name__ == '__main__':
    crop_name = crop_params.get('name', 'Unknown')
    print(f"Crop: {crop_name}")
    print(f"Season: DOY {crop_params['season_start_doy']} to "
          f"{crop_params['season_end_doy']} ({crop_params['season_days']} days)")
    print()

    for label, clim in [('Dry (2022)', climate_dry),
                         ('Moderate (2020)', climate_moderate),
                         ('Wet (2024)', climate_wet)]:
        print(f"{label}:")
        print(f"  Rainfall total: {clim['rainfall'].sum():.1f} mm")
        print(f"  Mean temp:      {clim['temp_mean'].mean():.1f} °C")
        print(f"  Mean ET0 (PM):  {clim['ET'].mean():.2f} mm/day")
        print(f"  Mean radiation: {clim['radiation'].mean():.1f} MJ/m²/day")
        print()
