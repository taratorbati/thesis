# =============================================================================
# climate_data.py
# Load cleaned climate data and extract crop-season scenarios.
# Reads from climate_apr_oct_cleaned.csv (produced by preprocess.py).
# Uses Penman-Monteith ET0.
#
# extract_scenario() takes the crop dict as an explicit parameter rather
# than importing it at module load time. This means the same Python process
# can load rice and tobacco scenarios without conflicting state.
#
# Scenarios: thesis uses two contrasting scenarios chosen from the 25-year
# climatological record. The intermediate-rainfall year 2020 (P38) is omitted
# because it is climatologically too close to the dry year 2022 (P35) to
# produce meaningfully different controller behaviour. See thesis Section
# 3.3 (Climatology and Scenario Selection) for the empirical justification.
# =============================================================================

import pandas as pd
import numpy as np


DATA_CSV = 'results/preprocessing/climate_apr_oct_cleaned.csv'

SCENARIO_YEARS = {
    'dry': 2022,   # P35 — 39.7 mm rice-season rainfall
    'wet': 2024,   # P96 — 176.8 mm rice-season rainfall (3 R20 events)
}


def load_cleaned_data(filepath=DATA_CSV):
    """Load the cleaned April-October climate CSV produced by preprocess.py.

    Parameters
    ----------
    filepath : str or Path
        Default: 'results/preprocessing/climate_apr_oct_cleaned.csv'.

    Returns
    -------
    pd.DataFrame
        Climate dataframe with DATE, YEAR, DOY, MONTH, and weather columns
        (T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, WS2M, PS, ALLSKY_SFC_SW_DWN,
        GWETROOT, GWETTOP, ET0_hargreaves, ET0_penman_monteith).
    """
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df


def extract_scenario(df, year, crop, start_doy=None, n_days=None):
    """Extract crop-season climate data for a given year and crop.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_cleaned_data().
    year : int
        Calendar year (e.g. 2022).
    crop : dict
        Crop parameter dict (from soil_data.RICE, soil_data.TOBACCO,
        or soil_data.get_crop(name)). Must contain 'season_start_doy'
        and 'season_days'.
    start_doy : int, optional
        Override the crop's season_start_doy.
    n_days : int, optional
        Override the crop's season_days.

    Returns
    -------
    dict
        Daily arrays of length n_days, keyed by:
            'rainfall', 'temp_mean', 'temp_max', 'temp_min',
            'radiation', 'ET', 'humidity', 'wind',
            'gwetroot', 'gwettop'
    """
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


def extract_scenario_by_name(df, scenario_name, crop, **kwargs):
    """Extract a named scenario ('dry' or 'wet') for a crop.

    Convenience wrapper around extract_scenario() that maps scenario names
    to their canonical years.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_cleaned_data().
    scenario_name : str
        One of 'dry' (2022) or 'wet' (2024).
    crop : dict
        Crop parameter dict.
    **kwargs
        Forwarded to extract_scenario.

    Returns
    -------
    dict
    """
    key = scenario_name.lower().strip()
    if key not in SCENARIO_YEARS:
        available = ', '.join(sorted(SCENARIO_YEARS.keys()))
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. Available: {available}"
        )
    return extract_scenario(df, SCENARIO_YEARS[key], crop, **kwargs)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from soil_data import get_crop

    crop = get_crop('rice')
    df = load_cleaned_data()

    print(f"Crop: {crop['name']}")
    print(f"Season: DOY {crop['season_start_doy']} to "
          f"{crop['season_end_doy']} ({crop['season_days']} days)")
    print()

    for scenario_name in ('dry', 'wet'):
        clim = extract_scenario_by_name(df, scenario_name, crop)
        year = SCENARIO_YEARS[scenario_name]
        print(f"{scenario_name.capitalize()} ({year}):")
        print(f"  Rainfall total: {clim['rainfall'].sum():.1f} mm")
        print(f"  Mean temp:      {clim['temp_mean'].mean():.1f} °C")
        print(f"  Mean ET0 (PM):  {clim['ET'].mean():.2f} mm/day")
        print(f"  Mean radiation: {clim['radiation'].mean():.1f} MJ/m²/day")
        print()
