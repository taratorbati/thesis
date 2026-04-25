import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_climate_data(filepath):
    df = pd.read_csv(filepath, skiprows=14)

    # Replace -999 with NaN
    df = df.replace(-999, np.nan)

    # Fix the 2 missing days using column means
    for col in ['T2M', 'T2M_MAX', 'PRECTOTCORR', 'GWETROOT', 'GWETTOP']:
        df[col] = df[col].fillna(df[col].mean())

    # Fix missing solar radiation using monthly average
    df['MONTH'] = pd.to_datetime(
        df['YEAR'].astype(str) +
        df['DOY'].astype(str).str.zfill(3),
        format='%Y%j').dt.month
    monthly_solar = df.groupby('MONTH')['ALLSKY_SFC_SW_DWN'].transform('mean')
    df['ALLSKY_SFC_SW_DWN'] = df['ALLSKY_SFC_SW_DWN'].fillna(monthly_solar)

    return df


def compute_ET0(temp_mean, temp_max, radiation):
    """
    Hargreaves equation (FAO Paper 56, Eq. 52).
    Used when only temperature and radiation are available.
    """
    T_min = 2 * temp_mean - temp_max  # estimate minimum temperature
    ET0 = 0.0023 * (temp_mean + 17.8) * (temp_max - T_min)**0.5 * radiation
    return np.clip(ET0, 0, None)


def extract_scenario(df, year, start_doy=91, n_days=120):
    """
    Extract 120-day crop season.
    start_doy=91 corresponds to April 1st — 
    typical wheat sowing start for this region.
    """
    df_year = df[df['YEAR'] == year].copy()
    df_season = df_year[df_year['DOY'] >= start_doy].head(n_days)
    df_season = df_season.reset_index(drop=True)

    climate = {
        'rainfall':  df_season['PRECTOTCORR'].values,
        'temp_mean': df_season['T2M'].values,
        'temp_max':  df_season['T2M_MAX'].values,
        'radiation': df_season['ALLSKY_SFC_SW_DWN'].values,
        'ET':        compute_ET0(
            df_season['T2M'].values,
            df_season['T2M_MAX'].values,
            df_season['ALLSKY_SFC_SW_DWN'].values),
        'gwetroot':  df_season['GWETROOT'].values,
        'gwettop':   df_season['GWETTOP'].values,
    }
    return climate


# --- Load and extract ---
df = load_climate_data(
    'POWER_Point_Daily_20200101_20260412_038d30N_048d85E_UTC.csv'
)

climate_dry = extract_scenario(df, year=2021)  # dry scenario
climate_wet = extract_scenario(df, year=2022)  # wet scenario


if __name__ == '__main__':

    # --- Verification plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Climate Scenarios — Dry (2021) vs Wet (2022)', fontsize=13)

    axes[0, 0].bar(range(120), climate_dry['rainfall'],
                   color='orange', alpha=0.7, label='2021 dry')
    axes[0, 0].bar(range(120), climate_wet['rainfall'],
                   color='blue', alpha=0.5, label='2022 wet')
    axes[0, 0].set_title(
        f"Daily Rainfall — dry:{climate_dry['rainfall'].sum():.0f}mm "
        f"wet:{climate_wet['rainfall'].sum():.0f}mm")
    axes[0, 0].legend()

    axes[0, 1].plot(climate_dry['temp_mean'], color='orange', label='2021 dry')
    axes[0, 1].plot(climate_wet['temp_mean'], color='blue',   label='2022 wet')
    axes[0, 1].set_title('Mean Temperature (°C)')
    axes[0, 1].legend()

    axes[1, 0].plot(climate_dry['ET'], color='orange', label='2021 dry')
    axes[1, 0].plot(climate_wet['ET'], color='blue',   label='2022 wet')
    axes[1, 0].set_title('Reference ET0 (mm/day)')
    axes[1, 0].legend()

    axes[1, 1].plot(climate_dry['radiation'], color='orange', label='2021 dry')
    axes[1, 1].plot(climate_wet['radiation'], color='blue',   label='2022 wet')
    axes[1, 1].set_title('Solar Radiation (MJ/m²/day)')
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel('Day of crop season (from April 1)')

    plt.tight_layout()
    plt.show()


# Print summary
print(f"Dry (2021): rainfall={climate_dry['rainfall'].sum():.1f}mm, "
      f"mean_temp={climate_dry['temp_mean'].mean():.1f}°C, "
      f"mean_ET={climate_dry['ET'].mean():.1f}mm/day")
print(f"Wet (2022): rainfall={climate_wet['rainfall'].sum():.1f}mm, "
      f"mean_temp={climate_wet['temp_mean'].mean():.1f}°C, "
      f"mean_ET={climate_wet['ET'].mean():.1f}mm/day")
