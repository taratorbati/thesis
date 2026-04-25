# =============================================================================
# preprocessing.py
# Complete data preprocessing, analysis, and visualization for thesis.
# Reads 25 years of NASA POWER data (2000–2025), cleans anomalies,
# computes ET0 (Hargreaves + Penman-Monteith), determines season window,
# calculates water budgets, and produces all figures for Chapter 3.
#
# Study site: Gilan Province, Iran (38.298°N, 48.847°E)
# Crop: Hashemi rice (Kc=1.15, base temp=10°C, θ18=1300 GDD)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

CSV_PATH = 'POWER_Point_Daily_20000101_20260417_038d30N_048d85E_LST.csv'
OUTPUT_DIR = Path('results/preprocessing')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Site parameters
LAT = 38.298          # degrees North
LON = 48.847          # degrees East
ELEVATION = 130.0     # meters (field average, not MERRA-2 grid average of 720m)

# Crop parameters (Hashemi rice, Gilan)
Kc = 1.15             # FAO season-average crop coefficient
BASE_TEMP = 10.0      # °C, base temperature for thermal time
THETA18 = 1300        # °C·days, thermal time to maturity (Sadidi Shal et al. 2021)
NURSERY_GDD = 140     # °C·days accumulated in nursery (~14 days at ~20°C)

# Season parameters
SEASON_START_DOY = 135   # May 14 (based on 25-year transplanting analysis)
SEASON_DAYS = 120        # simulation window (maturity at ~day 90-105, margin to 120)

# Water budget
SCARCITY_LEVELS = [0.50, 0.70, 0.80]

# Scenario years
YEAR_DRY = 2020
YEAR_MODERATE = 2023
YEAR_WET = 2024

# Rainfall anomaly threshold (mm/day) — values above this are flagged
RAIN_ANOMALY_THRESHOLD = 100.0

# Plot style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
})
# =============================================================================
# SECTION 1: Data Loading and Cleaning
# =============================================================================

def load_raw_data(filepath):
    """Load NASA POWER CSV with auto-detected header."""
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if '-END HEADER-' in line:
                skip = i + 1
                break

    df = pd.read_csv(filepath, skiprows=skip)
    df = df.replace(-999, np.nan)

    # Create proper date column
    df['DATE'] = pd.to_datetime(
        df['YEAR'].astype(int).astype(str) +
        df['DOY'].astype(int).astype(str).str.zfill(3),
        format='%Y%j'
    )
    df['MONTH'] = df['DATE'].dt.month
    df['DAY'] = df['DATE'].dt.day

    # Exclude 2026 (incomplete)
    df = df[df['YEAR'] <= 2025].copy()

    return df


def detect_anomalies(df):
    """Detect missing values and anomalous data points."""
    report = []
    report.append("=" * 70)
    report.append("DATA QUALITY REPORT")
    report.append("=" * 70)

    # 1. Missing values
    report.append("\n1. MISSING VALUES (NaN / -999):")
    for col in df.columns:
        if col in ['DATE', 'MONTH', 'DAY']:
            continue
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            years_affected = df[df[col].isna()]['YEAR'].unique()
            report.append(f"   {col:<25}: {n_miss} missing in years {sorted(years_affected)}")

    # 2. Rainfall anomalies (extreme single-day events)
    rain_extreme = df[df['PRECTOTCORR'] > RAIN_ANOMALY_THRESHOLD].copy()
    report.append(f"\n2. EXTREME RAINFALL (>{RAIN_ANOMALY_THRESHOLD} mm/day):")
    if len(rain_extreme) > 0:
        for _, row in rain_extreme.iterrows():
            report.append(
                f"   {row['DATE'].date()}: {row['PRECTOTCORR']:.1f} mm "
                f"(DOY {int(row['DOY'])}, Year {int(row['YEAR'])})"
            )
    else:
        report.append("   None found.")

    # 3. Temperature consistency (Tmin should be <= Tmean <= Tmax)
    temp_issues = df[
        (df['T2M_MIN'] > df['T2M']) | (df['T2M'] > df['T2M_MAX'])
    ]
    report.append(f"\n3. TEMPERATURE CONSISTENCY (Tmin > Tmean or Tmean > Tmax):")
    report.append(f"   {len(temp_issues)} days with inconsistent temperatures")

    # 4. Physical range checks
    report.append("\n4. PHYSICAL RANGE CHECKS:")
    checks = [
        ('T2M', -30, 50, '°C'),
        ('T2M_MAX', -25, 55, '°C'),
        ('T2M_MIN', -35, 45, '°C'),
        ('RH2M', 0, 100, '%'),
        ('WS2M', 0, 20, 'm/s'),
        ('ALLSKY_SFC_SW_DWN', 0, 40, 'MJ/m²/day'),
        ('PRECTOTCORR', 0, 500, 'mm/day'),
    ]
    for col, lo, hi, unit in checks:
        out = df[(df[col] < lo) | (df[col] > hi)]
        if len(out) > 0:
            report.append(f"   {col}: {len(out)} values outside [{lo}, {hi}] {unit}")

    return '\n'.join(report), rain_extreme.index.tolist()


def clean_data(df, anomaly_indices):
    """Replace missing values and anomalies with local averages."""
    df_clean = df.copy()

    # Replace rainfall anomalies with DOY-window average from other years
    for idx in anomaly_indices:
        doy = int(df_clean.loc[idx, 'DOY'])
        year = int(df_clean.loc[idx, 'YEAR'])
        # Average rainfall on this DOY across all other years (±3 day window)
        mask = (df_clean['DOY'].between(doy - 3, doy + 3)) & (df_clean['YEAR'] != year)
        replacement = df_clean.loc[mask, 'PRECTOTCORR'].mean()
        original = df_clean.loc[idx, 'PRECTOTCORR']
        df_clean.loc[idx, 'PRECTOTCORR'] = replacement
        print(f"  Replaced rain on {df_clean.loc[idx, 'DATE'].date()}: "
              f"{original:.1f} → {replacement:.1f} mm")

    # Fill remaining NaN with DOY-window average (±7 day window from other years)
    cols_to_fill = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'PS',
                    'WS2M', 'T2MDEW', 'GWETROOT', 'GWETTOP', 'ALLSKY_SFC_SW_DWN']

    for col in cols_to_fill:
        n_before = df_clean[col].isna().sum()
        if n_before > 0:
            # Use monthly average as fallback
            monthly_avg = df_clean.groupby('MONTH')[col].transform('mean')
            df_clean[col] = df_clean[col].fillna(monthly_avg)
            n_after = df_clean[col].isna().sum()
            if n_before > n_after:
                print(f"  Filled {n_before - n_after} missing values in {col}")

    return df_clean


# =============================================================================
# SECTION 2: ET0 Calculations
# =============================================================================

def hargreaves_et0(tmean, tmax, tmin, rad):
    """Hargreaves-Samani ET0 estimate (FAO Paper 56, Eq. 52)."""
    tdiff = np.maximum(tmax - tmin, 0)
    et0 = 0.0023 * (tmean + 17.8) * tdiff**0.5 * rad
    return np.clip(et0, 0, 20)


def penman_monteith_et0(tmean, tmax, tmin, rh, ws, rad, pressure, doy, lat=LAT):
    """
    FAO-56 Penman-Monteith reference evapotranspiration.

    Parameters
    ----------
    tmean, tmax, tmin : array — temperatures (°C)
    rh : array — relative humidity (%)
    ws : array — wind speed at 2m (m/s)
    rad : array — incoming shortwave radiation (MJ/m²/day)
    pressure : array — surface pressure (kPa)
    doy : array — day of year
    lat : float — latitude (degrees)

    Returns
    -------
    et0 : array — reference ET (mm/day)
    """
    # --- Vapor pressures ---
    es_max = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es_min = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_max + es_min) / 2
    # Actual vapor pressure: FAO recommends ea from Tmin when only mean RH
    ea = (rh / 100.0) * es_min

    # --- Slope of saturation vapor pressure curve ---
    delta = 4098 * 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3)) / (tmean + 237.3)**2

    # --- Psychrometric constant ---
    gamma = 0.000665 * pressure

    # --- Net radiation (FAO-56 procedure) ---
    # Net shortwave
    Rns = (1 - 0.23) * rad  # albedo = 0.23

    # Extraterrestrial radiation for Rso estimate
    lat_rad = lat * np.pi / 180
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    sdec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws_angle = np.arccos(-np.tan(lat_rad) * np.tan(sdec))
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws_angle * np.sin(lat_rad) * np.sin(sdec) +
        np.cos(lat_rad) * np.cos(sdec) * np.sin(ws_angle)
    )
    # Clear-sky radiation
    Rso = (0.75 + 2e-5 * ELEVATION) * Ra
    Rso = np.maximum(Rso, 0.1)

    # Net longwave (FAO-56 Eq. 39)
    sigma = 4.903e-9  # Stefan-Boltzmann (MJ/m²/day/K⁴)
    Rnl = sigma * ((tmax + 273.16)**4 + (tmin + 273.16)**4) / 2 * \
          (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.001))) * \
          (1.35 * np.minimum(rad / Rso, 1.0) - 0.35)

    Rn = Rns - Rnl
    G = 0  # soil heat flux ≈ 0 for daily time step

    # --- FAO-56 Penman-Monteith equation ---
    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (tmean + 273)) * ws * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * ws)

    et0 = numerator / denominator
    return np.clip(et0, 0, 15)


def compute_et0(df):
    """Add both ET0 estimates to dataframe."""
    df = df.copy()
    df['ET0_hargreaves'] = hargreaves_et0(
        df['T2M'].values, df['T2M_MAX'].values,
        df['T2M_MIN'].values, df['ALLSKY_SFC_SW_DWN'].values
    )
    df['ET0_penman_monteith'] = penman_monteith_et0(
        df['T2M'].values, df['T2M_MAX'].values,
        df['T2M_MIN'].values, df['RH2M'].values,
        df['WS2M'].values, df['ALLSKY_SFC_SW_DWN'].values,
        df['PS'].values, df['DOY'].values
    )
    return df


# =============================================================================
# SECTION 3: Season Extraction
# =============================================================================

def extract_rice_season(df, year, start_doy=SEASON_START_DOY, n_days=SEASON_DAYS):
    """Extract one rice season for a given year."""
    dy = df[(df['YEAR'] == year) & (df['DOY'] >= start_doy)].head(n_days)
    if len(dy) < n_days * 0.8:
        return None
    return dy.reset_index(drop=True)


def extract_all_seasons(df, years=None):
    """Extract rice seasons for all specified years."""
    if years is None:
        years = range(2000, 2026)
    seasons = {}
    for y in years:
        s = extract_rice_season(df, y)
        if s is not None:
            seasons[y] = s
    return seasons


# =============================================================================
# SECTION 4: Climatological Averages (April–October)
# =============================================================================

def compute_daily_climatology(df, start_month=4, end_month=10):
    """
    Compute 25-year daily average for all variables
    from April through October.
    """
    mask = df['MONTH'].between(start_month, end_month)
    df_period = df[mask].copy()

    # Group by DOY and compute mean
    clim = df_period.groupby('DOY').agg({
        'T2M': 'mean',
        'T2M_MAX': 'mean',
        'T2M_MIN': 'mean',
        'PRECTOTCORR': 'mean',
        'RH2M': 'mean',
        'WS2M': 'mean',
        'PS': 'mean',
        'ALLSKY_SFC_SW_DWN': 'mean',
        'GWETROOT': 'mean',
        'GWETTOP': 'mean',
        'ET0_hargreaves': 'mean',
        'ET0_penman_monteith': 'mean',
    }).reset_index()

    # Convert DOY to approximate date for plotting (use 2024 as reference, leap year)
    clim['DATE'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(clim['DOY'] - 1, unit='D')

    return clim


# =============================================================================
# SECTION 5: Water Budget
# =============================================================================

def compute_water_budgets(seasons, method='B', avg_rain=None):
    """
    Compute water budget for each season.
    Method A: uses actual rainfall per year.
    Method B: uses long-term average rainfall (fixed budget).
    """
    results = []

    if avg_rain is None and method == 'B':
        # Compute from all seasons, first 100 days
        all_rain = [s.head(100)['PRECTOTCORR'].sum() for s in seasons.values()]
        avg_rain = np.mean(all_rain)

    for year, s in sorted(seasons.items()):
        s100 = s.head(100)
        rain = s100['PRECTOTCORR'].sum()
        et0_pm = s100['ET0_penman_monteith'].mean()
        crop_demand = et0_pm * Kc * 100

        if method == 'A':
            full_need = max(crop_demand - rain, 0)
        else:
            full_need_avg = max(et0_pm * Kc * 100 - avg_rain, 0)
            full_need = full_need_avg  # same basis for all years

        results.append({
            'year': year,
            'rain_100d': rain,
            'et0_pm_avg': et0_pm,
            'crop_demand': crop_demand,
            'full_need': full_need,
            'budget_80': 0.80 * full_need,
            'budget_70': 0.70 * full_need,
            'budget_50': 0.50 * full_need,
        })

    return pd.DataFrame(results), avg_rain


# =============================================================================
# SECTION 6: Thermal Time Analysis
# =============================================================================

def thermal_time_analysis(seasons):
    """Compute thermal time accumulation for each season."""
    results = []
    for year, s in sorted(seasons.items()):
        daily_gdd = (s['T2M'].values - BASE_TEMP).clip(min=0)
        cumgdd = np.cumsum(daily_gdd) + NURSERY_GDD

        gdd_100 = cumgdd[99] if len(cumgdd) >= 100 else cumgdd[-1]
        gdd_120 = cumgdd[min(119, len(cumgdd) - 1)]

        d_mat = int(np.argmax(cumgdd >= THETA18) + 1) if np.any(cumgdd >= THETA18) else None

        results.append({
            'year': year,
            'gdd_100d': gdd_100,
            'gdd_120d': gdd_120,
            'days_to_maturity': d_mat,
            'cumgdd': cumgdd.copy(),
        })

    return results


# =============================================================================
# SECTION 7: Transplanting Date Analysis
# =============================================================================

def transplanting_analysis(df, threshold=18.0, window=7):
    """Find first date when rolling mean temperature exceeds threshold."""
    results = []
    for year in range(2000, 2026):
        dy = df[df['YEAR'] == year].sort_values('DOY').copy()
        dy['T_roll'] = dy['T2M'].rolling(window, min_periods=5).mean()
        warm = dy[dy['T_roll'] >= threshold]
        if len(warm) > 0:
            first = warm.iloc[0]
            results.append({
                'year': year,
                'doy': int(first['DOY']),
                'date': first['DATE'],
                'temp_7d': first['T_roll'],
            })
    return pd.DataFrame(results)


# =============================================================================
# SECTION 8: Plotting Functions
# =============================================================================

def plot_daily_climatology(clim):
    """Plot 1: Daily average temperature, rainfall, radiation (Apr–Oct)."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle('25-Year Daily Climatology — Gilan Province (38.3°N, 48.8°E)\n'
                 'April–October, 2000–2025', fontsize=14)

    # Rice season shading
    rice_start = pd.to_datetime(f'2024-01-01') + pd.to_timedelta(SEASON_START_DOY - 1, unit='D')
    rice_end = rice_start + pd.to_timedelta(SEASON_DAYS, unit='D')

    for ax in axes:
        ax.axvspan(rice_start, rice_end, alpha=0.08, color='green', label='Rice season')

    # Temperature
    ax = axes[0]
    ax.fill_between(clim['DATE'], clim['T2M_MIN'], clim['T2M_MAX'],
                    alpha=0.25, color='red', label='Tmin–Tmax range')
    ax.plot(clim['DATE'], clim['T2M'], 'r-', linewidth=1.5, label='Tmean')
    ax.plot(clim['DATE'], clim['T2M_MAX'], 'r--', linewidth=0.8, alpha=0.5)
    ax.plot(clim['DATE'], clim['T2M_MIN'], 'b--', linewidth=0.8, alpha=0.5)
    ax.axhline(BASE_TEMP, color='gray', linestyle=':', linewidth=0.8,
               label=f'Rice base temp ({BASE_TEMP}°C)')
    ax.axhline(18, color='orange', linestyle=':', linewidth=0.8,
               label='Transplanting threshold (18°C)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Daily Average Temperature')
    ax.legend(loc='upper left', fontsize=7)

    # Rainfall
    ax = axes[1]
    ax.bar(clim['DATE'], clim['PRECTOTCORR'], width=1.5, color='steelblue', alpha=0.7)
    ax.set_ylabel('Rainfall (mm/day)')
    ax.set_title('Daily Average Rainfall')

    # Radiation
    ax = axes[2]
    ax.plot(clim['DATE'], clim['ALLSKY_SFC_SW_DWN'], color='orange', linewidth=1.2)
    ax.fill_between(clim['DATE'], 0, clim['ALLSKY_SFC_SW_DWN'],
                    alpha=0.2, color='orange')
    ax.set_ylabel('Radiation (MJ/m²/day)')
    ax.set_title('Daily Average Solar Radiation')

    # ET0 comparison
    ax = axes[3]
    ax.plot(clim['DATE'], clim['ET0_penman_monteith'], color='blue',
            linewidth=1.5, label='Penman-Monteith')
    ax.plot(clim['DATE'], clim['ET0_hargreaves'], color='red',
            linewidth=1.2, linestyle='--', label='Hargreaves')
    ax.set_ylabel('ET₀ (mm/day)')
    ax.set_title('Reference Evapotranspiration — Penman-Monteith vs Hargreaves')
    ax.legend()

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_daily_climatology.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_daily_climatology.png")


def plot_et0_comparison(seasons):
    """Plot 2: Hargreaves vs PM scatter + monthly comparison."""
    all_data = pd.concat(seasons.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ET₀ Comparison: Hargreaves vs Penman-Monteith\n'
                 'Rice Season (May 14 – Sep 10), 26 years, n=3120 days', fontsize=13)

    # Scatter
    ax = axes[0]
    ax.scatter(all_data['ET0_penman_monteith'], all_data['ET0_hargreaves'],
               s=3, alpha=0.15, color='steelblue')
    lims = [0, 14]
    ax.plot(lims, lims, 'k--', linewidth=0.8, label='1:1 line')
    # Regression line
    mask = all_data[['ET0_penman_monteith', 'ET0_hargreaves']].notna().all(axis=1)
    z = np.polyfit(all_data.loc[mask, 'ET0_penman_monteith'],
                   all_data.loc[mask, 'ET0_hargreaves'], 1)
    ax.plot(lims, [z[0] * x + z[1] for x in lims], 'r-', linewidth=1,
            label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')
    corr = all_data[['ET0_penman_monteith', 'ET0_hargreaves']].corr().iloc[0, 1]
    ax.text(0.05, 0.92, f'r = {corr:.3f}\nHarg/PM = {z[0]:.2f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Penman-Monteith ET₀ (mm/day)')
    ax.set_ylabel('Hargreaves ET₀ (mm/day)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_title('Daily Scatter Plot')

    # Monthly boxplot comparison
    ax = axes[1]
    months_data = []
    for m in [5, 6, 7, 8, 9]:
        mdata = all_data[all_data['MONTH'] == m]
        months_data.append({
            'month': m,
            'harg_mean': mdata['ET0_hargreaves'].mean(),
            'pm_mean': mdata['ET0_penman_monteith'].mean(),
            'harg_std': mdata['ET0_hargreaves'].std(),
            'pm_std': mdata['ET0_penman_monteith'].std(),
        })
    md = pd.DataFrame(months_data)
    month_labels = ['May', 'Jun', 'Jul', 'Aug', 'Sep']
    x = np.arange(len(month_labels))
    w = 0.35
    ax.bar(x - w / 2, md['pm_mean'], w, yerr=md['pm_std'], capsize=3,
           label='Penman-Monteith', color='steelblue', alpha=0.8)
    ax.bar(x + w / 2, md['harg_mean'], w, yerr=md['harg_std'], capsize=3,
           label='Hargreaves', color='indianred', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel('ET₀ (mm/day)')
    ax.set_title('Monthly Averages ± Std Dev')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_et0_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_et0_comparison.png")


def plot_rainfall_analysis(seasons, avg_rain_25yr):
    """Plot 3: Rainfall distribution across years + seasonal pattern."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Rainfall Analysis — Rice Season (100 days from May 14)', fontsize=13)

    years = sorted(seasons.keys())
    rain_totals = [seasons[y].head(100)['PRECTOTCORR'].sum() for y in years]

    # Bar chart of annual totals
    ax = axes[0]
    colors = ['steelblue' if y < 2020 else 'darkblue' for y in years]
    bars = ax.bar(range(len(years)), rain_totals, color=colors, alpha=0.7)
    ax.axhline(avg_rain_25yr, color='red', linestyle='--', linewidth=1.5,
               label=f'25-year average: {avg_rain_25yr:.1f} mm')
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=90, fontsize=7)
    ax.set_ylabel('Total Rainfall (mm)')
    ax.set_title('Seasonal Rainfall by Year')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Highlight scenario years
    for i, y in enumerate(years):
        if y == YEAR_DRY:
            bars[i].set_color('red')
            bars[i].set_alpha(0.9)
        elif y == YEAR_WET:
            bars[i].set_color('green')
            bars[i].set_alpha(0.9)
        elif y == YEAR_MODERATE:
            bars[i].set_color('orange')
            bars[i].set_alpha(0.9)

    # Average daily rainfall pattern
    ax = axes[1]
    all_data = pd.concat([s.head(100) for s in seasons.values()])
    all_data['season_day'] = all_data.groupby(all_data.index // 100).cumcount()
    # Recompute season day properly
    daily_avg = []
    for d in range(100):
        day_rain = [seasons[y].iloc[d]['PRECTOTCORR'] for y in years if len(seasons[y]) > d]
        daily_avg.append(np.mean(day_rain))

    ax.bar(range(100), daily_avg, color='steelblue', alpha=0.7, width=1)
    ax.set_xlabel('Day of Rice Season')
    ax.set_ylabel('Average Daily Rainfall (mm)')
    ax.set_title('Seasonal Rainfall Pattern (25-year average)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_rainfall_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_rainfall_analysis.png")


def plot_thermal_time(tt_results, scenario_years=None):
    """Plot 4: Thermal time accumulation curves."""
    if scenario_years is None:
        scenario_years = [YEAR_DRY, YEAR_MODERATE, YEAR_WET]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f'Thermal Time Accumulation — Hashemi Rice\n'
                 f'Base temp = {BASE_TEMP}°C, Nursery GDD = {NURSERY_GDD}, '
                 f'θ₁₈ = {THETA18}°C·days', fontsize=13)

    # Plot all years in gray
    for r in tt_results:
        if r['year'] not in scenario_years:
            ax.plot(range(1, len(r['cumgdd']) + 1), r['cumgdd'],
                    color='lightgray', linewidth=0.5, alpha=0.6)

    # Plot scenario years with color
    colors = {'dry': 'red', 'moderate': 'orange', 'wet': 'blue'}
    labels = {YEAR_DRY: f'{YEAR_DRY} (dry)', YEAR_MODERATE: f'{YEAR_MODERATE} (moderate)',
              YEAR_WET: f'{YEAR_WET} (wet)'}
    color_map = {YEAR_DRY: 'red', YEAR_MODERATE: 'orange', YEAR_WET: 'blue'}

    for r in tt_results:
        if r['year'] in scenario_years:
            ax.plot(range(1, len(r['cumgdd']) + 1), r['cumgdd'],
                    color=color_map.get(r['year'], 'black'), linewidth=2,
                    label=labels.get(r['year'], str(r['year'])))

    # Maturity line
    ax.axhline(THETA18, color='darkgreen', linestyle='--', linewidth=1.5,
               label=f'Maturity (θ₁₈ = {THETA18}°C·days)')
    ax.axhline(THETA18 / 2, color='green', linestyle=':', linewidth=1,
               label=f'Mid-season (θ₁₈/2 = {THETA18 / 2:.0f}°C·days)')

    ax.set_xlabel('Day of Field Season')
    ax.set_ylabel('Cumulative Thermal Time (°C·days)')
    ax.set_xlim(0, 125)
    ax.set_ylim(0, max(r['cumgdd'][-1] for r in tt_results) * 1.05)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_thermal_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_thermal_time.png")


def plot_scenario_comparison(seasons):
    """Plot 5: Climate comparison of the three scenario years."""
    scenario_years = [YEAR_DRY, YEAR_MODERATE, YEAR_WET]
    colors = {YEAR_DRY: 'red', YEAR_MODERATE: 'orange', YEAR_WET: 'blue'}
    labels = {YEAR_DRY: f'{YEAR_DRY} (dry)', YEAR_MODERATE: f'{YEAR_MODERATE} (moderate)',
              YEAR_WET: f'{YEAR_WET} (wet)'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Climate Scenarios — Dry vs Moderate vs Wet', fontsize=14)

    for year in scenario_years:
        s = seasons[year]
        days = range(len(s))
        c = colors[year]
        l = labels[year]

        # Rainfall
        axes[0, 0].bar([d + 0.25 * (scenario_years.index(year) - 1) for d in days],
                       s['PRECTOTCORR'].values, width=0.25, color=c, alpha=0.6, label=l)
        # Temperature
        axes[0, 1].plot(days, s['T2M'].values, color=c, linewidth=1, label=l)
        # ET0 PM
        axes[1, 0].plot(days, s['ET0_penman_monteith'].values, color=c, linewidth=1, label=l)
        # GWETROOT
        axes[1, 1].plot(days, s['GWETROOT'].values, color=c, linewidth=1, label=l)

    axes[0, 0].set_title('Daily Rainfall')
    axes[0, 0].set_ylabel('mm/day')
    axes[0, 1].set_title('Mean Temperature')
    axes[0, 1].set_ylabel('°C')
    axes[1, 0].set_title('Reference ET₀ (Penman-Monteith)')
    axes[1, 0].set_ylabel('mm/day')
    axes[1, 1].set_title('NASA GWETROOT (satellite soil moisture)')
    axes[1, 1].set_ylabel('Wetness fraction (0–1)')

    for ax in axes.flat:
        ax.set_xlabel('Day of Rice Season')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_scenario_comparison.png")


def plot_gwetroot_cross_validation(seasons):
    """Plot 6: GWETROOT seasonal pattern for cross-validation reference."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('NASA GWETROOT — Independent Soil Moisture Reference\n'
                 'For cross-validation against ABM simulated x₁', fontsize=13)

    # All years overlay
    ax = axes[0]
    for year, s in sorted(seasons.items()):
        c = 'lightgray'
        lw = 0.5
        if year == YEAR_DRY:
            c, lw = 'red', 2
        elif year == YEAR_MODERATE:
            c, lw = 'orange', 2
        elif year == YEAR_WET:
            c, lw = 'blue', 2
        ax.plot(range(len(s)), s['GWETROOT'].values, color=c, linewidth=lw,
                alpha=0.7 if c != 'lightgray' else 0.4)

    # Add manual legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label=f'{YEAR_DRY} (dry)'),
        Line2D([0], [0], color='orange', linewidth=2, label=f'{YEAR_MODERATE} (moderate)'),
        Line2D([0], [0], color='blue', linewidth=2, label=f'{YEAR_WET} (wet)'),
        Line2D([0], [0], color='lightgray', linewidth=1, label='Other years'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.set_xlabel('Day of Rice Season')
    ax.set_ylabel('GWETROOT (0–1)')
    ax.set_title('Root Zone Soil Wetness — All Years')
    ax.grid(alpha=0.3)

    # Convert to mm equivalent
    ax = axes[1]
    ROOT_DEPTH = 400  # mm
    for year, s in sorted(seasons.items()):
        x1_equiv = s['GWETROOT'].values * ROOT_DEPTH
        c = 'lightgray'
        lw = 0.5
        if year == YEAR_DRY:
            c, lw = 'red', 2
        elif year == YEAR_MODERATE:
            c, lw = 'orange', 2
        elif year == YEAR_WET:
            c, lw = 'blue', 2
        ax.plot(range(len(s)), x1_equiv, color=c, linewidth=lw,
                alpha=0.7 if c != 'lightgray' else 0.4)

    # Reference lines
    FC = 0.35 * ROOT_DEPTH  # 140 mm
    WP = 0.15 * ROOT_DEPTH  # 60 mm
    STRESS = FC - 0.20 * (FC - WP)  # 124 mm
    ax.axhline(FC, color='red', linestyle='--', linewidth=1, label=f'Field Capacity ({FC:.0f}mm)')
    ax.axhline(STRESS, color='purple', linestyle=':', linewidth=1,
               label=f'Stress Threshold ({STRESS:.0f}mm)')
    ax.axhline(WP, color='orange', linestyle='--', linewidth=1, label=f'Wilting Point ({WP:.0f}mm)')

    ax.legend(handles=legend_elements + [
        Line2D([0], [0], color='red', linestyle='--', label=f'FC ({FC:.0f}mm)'),
        Line2D([0], [0], color='purple', linestyle=':', label=f'Stress ({STRESS:.0f}mm)'),
        Line2D([0], [0], color='orange', linestyle='--', label=f'WP ({WP:.0f}mm)'),
    ], fontsize=7, loc='upper right')
    ax.set_xlabel('Day of Rice Season')
    ax.set_ylabel('Equivalent Soil Water (mm)')
    ax.set_title(f'GWETROOT × {ROOT_DEPTH}mm vs. ABM Reference Lines')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_gwetroot_crossval.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_gwetroot_crossval.png")


def plot_humidity_wind(clim):
    """Plot 7: Humidity and wind — why Hargreaves fails here."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Humidity and Wind Speed — Why Hargreaves Overestimates in Gilan\n'
                 'These variables are invisible to Hargreaves but used by Penman-Monteith',
                 fontsize=13)

    rice_start = pd.to_datetime('2024-01-01') + pd.to_timedelta(SEASON_START_DOY - 1, unit='D')
    rice_end = rice_start + pd.to_timedelta(SEASON_DAYS, unit='D')

    for ax in axes:
        ax.axvspan(rice_start, rice_end, alpha=0.08, color='green')

    ax = axes[0]
    ax.plot(clim['DATE'], clim['RH2M'], color='teal', linewidth=1.5)
    ax.fill_between(clim['DATE'], 0, clim['RH2M'], alpha=0.15, color='teal')
    ax.axhline(65, color='red', linestyle=':', linewidth=0.8,
               label='Rice season avg (~65%)')
    ax.set_ylabel('Relative Humidity (%)')
    ax.set_title('Daily Average Relative Humidity')
    ax.legend()
    ax.set_ylim(50, 90)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(clim['DATE'], clim['WS2M'], color='navy', linewidth=1.5)
    ax.fill_between(clim['DATE'], 0, clim['WS2M'], alpha=0.15, color='navy')
    ax.axhline(1.4, color='red', linestyle=':', linewidth=0.8,
               label='Rice season avg (~1.4 m/s)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Daily Average Wind Speed at 2m')
    ax.legend()
    ax.grid(alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_humidity_wind.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_humidity_wind.png")


# =============================================================================
# SECTION 9: Summary Report
# =============================================================================

def print_summary(df, seasons, tt_results, budget_df, avg_rain, transplant_df):
    """Print comprehensive summary of all analysis results."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE PREPROCESSING SUMMARY")
    print("=" * 80)

    print(f"\nDataset: {len(df)} days, {df['YEAR'].min():.0f}–{df['YEAR'].max():.0f}")
    print(f"Location: {LAT}°N, {LON}°E (Gilan Province, Iran)")
    print(f"Crop: Hashemi rice (Kc={Kc}, base={BASE_TEMP}°C, θ₁₈={THETA18}°C·days)")

    # Climate averages
    all_rice = pd.concat(seasons.values())
    print(f"\n── Rice Season Climate (DOY {SEASON_START_DOY}–{SEASON_START_DOY + SEASON_DAYS}, "
          f"n={len(seasons)} years) ──")
    print(f"  Temperature:  Tmin={all_rice['T2M_MIN'].mean():.1f}, "
          f"Tmean={all_rice['T2M'].mean():.1f}, "
          f"Tmax={all_rice['T2M_MAX'].mean():.1f} °C")
    print(f"  Humidity:     {all_rice['RH2M'].mean():.1f}%")
    print(f"  Wind:         {all_rice['WS2M'].mean():.2f} m/s")
    print(f"  Radiation:    {all_rice['ALLSKY_SFC_SW_DWN'].mean():.1f} MJ/m²/day")

    # ET0
    print(f"\n── ET₀ Comparison ──")
    print(f"  Hargreaves:      {all_rice['ET0_hargreaves'].mean():.2f} mm/day")
    print(f"  Penman-Monteith: {all_rice['ET0_penman_monteith'].mean():.2f} mm/day")
    ratio = all_rice['ET0_hargreaves'].mean() / all_rice['ET0_penman_monteith'].mean()
    print(f"  Overestimation:  {(ratio - 1) * 100:.0f}%")
    corr = all_rice[['ET0_hargreaves', 'ET0_penman_monteith']].corr().iloc[0, 1]
    print(f"  Correlation:     r = {corr:.3f}")

    # Transplanting
    print(f"\n── Transplanting Analysis ──")
    print(f"  Avg DOY: {transplant_df['doy'].mean():.0f} "
          f"(≈ {pd.Timestamp(2024, 1, 1) + pd.Timedelta(days=int(transplant_df['doy'].mean()) - 1):%B %d})")
    print(f"  Range: DOY {transplant_df['doy'].min()}–{transplant_df['doy'].max()}")
    print(f"  Chosen: DOY {SEASON_START_DOY} (May 14)")

    # Thermal time
    print(f"\n── Thermal Time ──")
    mat_days = [r['days_to_maturity'] for r in tt_results if r['days_to_maturity'] is not None]
    print(f"  Days to maturity: {np.min(mat_days):.0f}–{np.max(mat_days):.0f} "
          f"(avg {np.mean(mat_days):.0f})")
    print(f"  All years reach θ₁₈={THETA18} within 120 days: "
          f"{'YES' if len(mat_days) == len(tt_results) else 'NO'}")

    # Water budget
    print(f"\n── Water Budget (Method B, 25-year avg rain = {avg_rain:.1f} mm) ──")
    avg_et0 = np.mean([seasons[y].head(100)['ET0_penman_monteith'].mean()
                       for y in seasons])
    demand = avg_et0 * Kc * 100
    need = max(demand - avg_rain, 0)
    print(f"  Avg PM-ET₀:       {avg_et0:.2f} mm/day")
    print(f"  Crop demand:      {demand:.1f} mm (100 days)")
    print(f"  Full irrig need:  {need:.1f} mm/agent")
    for s in SCARCITY_LEVELS:
        print(f"  Budget at {s * 100:.0f}%:    {s * need:.1f} mm/agent "
              f"({s * need * 130:.0f} mm field total)")

    # Scenarios
    print(f"\n── Scenario Years ──")
    for year, label in [(YEAR_DRY, 'DRY'), (YEAR_MODERATE, 'MODERATE'), (YEAR_WET, 'WET')]:
        rain = seasons[year].head(100)['PRECTOTCORR'].sum()
        et0 = seasons[year].head(100)['ET0_penman_monteith'].mean()
        print(f"  {year} ({label}): rain={rain:.1f}mm, ET0_PM={et0:.2f}mm/d")

    # Parameters to update
    print(f"\n── Parameters for Code Update ──")
    print(f"  θ₁₈ = {THETA18}  (was 1800)")
    print(f"  θ₂₀ = {int(THETA18 / 1800 * 600)}  (was 600, proportionally scaled)")
    print(f"  x₂_init = {NURSERY_GDD}  (was 0)")
    print(f"  Season start = DOY {SEASON_START_DOY}  (was 91)")
    print(f"  ET₀ method = Penman-Monteith  (was Hargreaves)")
    print(f"  Budget rain = {avg_rain:.1f} mm (25-year avg)  (was per-year actual)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':

    print("=" * 70)
    print("THESIS PREPROCESSING PIPELINE")
    print("Gilan Province Rice Irrigation Study")
    print("=" * 70)

    # Step 1: Load
    print("\n[1/8] Loading NASA POWER data...")
    df = load_raw_data(CSV_PATH)
    print(f"  Loaded {len(df)} rows, {df['YEAR'].min():.0f}–{df['YEAR'].max():.0f}")
    print(f"  Variables: {[c for c in df.columns if c not in ['DATE', 'MONTH', 'DAY']]}")

    # Step 2: Detect anomalies
    print("\n[2/8] Detecting anomalies...")
    report, anomaly_idx = detect_anomalies(df)
    print(report)

    # Step 3: Clean
    print("\n[3/8] Cleaning data...")
    df = clean_data(df, anomaly_idx)

    # Step 4: Compute ET0
    print("\n[4/8] Computing ET₀ (Hargreaves + Penman-Monteith)...")
    df = compute_et0(df)
    print(f"  Hargreaves mean:      {df['ET0_hargreaves'].mean():.2f} mm/day (full year)")
    print(f"  Penman-Monteith mean: {df['ET0_penman_monteith'].mean():.2f} mm/day (full year)")

    # Step 5: Extract seasons
    print("\n[5/8] Extracting rice seasons...")
    seasons = extract_all_seasons(df)
    print(f"  Extracted {len(seasons)} seasons ({min(seasons.keys())}–{max(seasons.keys())})")

    # Step 6: Analyses
    print("\n[6/8] Running analyses...")

    # Transplanting
    transplant_df = transplanting_analysis(df)

    # Thermal time
    tt_results = thermal_time_analysis(seasons)

    # Water budget
    budget_df, avg_rain = compute_water_budgets(seasons, method='B')

    # Climatology
    clim = compute_daily_climatology(df)

    # Step 7: Generate plots
    print("\n[7/8] Generating figures...")
    plot_daily_climatology(clim)
    plot_et0_comparison(seasons)
    plot_rainfall_analysis(seasons, avg_rain)
    plot_thermal_time(tt_results)
    plot_scenario_comparison(seasons)
    plot_gwetroot_cross_validation(seasons)
    plot_humidity_wind(clim)

    # Step 8: Summary
    print("\n[8/8] Summary report...")
    print_summary(df, seasons, tt_results, budget_df, avg_rain, transplant_df)

    # Save cleaned data for use by other scripts
    df.to_csv(OUTPUT_DIR / 'climate_data_cleaned.csv', index=False)
    print(f"\n  Cleaned data saved to: {OUTPUT_DIR / 'climate_data_cleaned.csv'}")
    print(f"  All figures saved to:  {OUTPUT_DIR}/")
    print("\nDone.")
