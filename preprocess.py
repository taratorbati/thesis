# =============================================================================
# preprocess.py
# Load NASA POWER data (2000–2025), filter Apr–Oct, clean anomalies,
# compute ET0 (Hargreaves + Penman-Monteith), save cleaned CSV.
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_CSV = 'POWER_Point_Daily_20000101_20260417_038d30N_048d85E_LST.csv'
OUTPUT_DIR = Path('results/preprocessing')
OUTPUT_CSV = OUTPUT_DIR / 'climate_apr_oct_cleaned.csv'
REPORT_TXT = OUTPUT_DIR / 'cleaning_report.txt'

LAT = 38.298
ELEVATION = 130.0  # meters (field average)

RAIN_ANOMALY_THRESHOLD = 100.0  # mm/day — flag values above this
YEAR_START = 2000
YEAR_END = 2025
MONTH_START = 4   # April
MONTH_END = 10    # October

# =============================================================================
# Step 1: Load raw data
# =============================================================================


def load_raw(filepath):
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if '-END HEADER-' in line:
                skip = i + 1
                break

    df = pd.read_csv(filepath, skiprows=skip)
    df = df.replace(-999, np.nan)
    df['DATE'] = pd.to_datetime(
        df['YEAR'].astype(int).astype(str) +
        df['DOY'].astype(int).astype(str).str.zfill(3),
        format='%Y%j'
    )
    df['MONTH'] = df['DATE'].dt.month
    return df


# =============================================================================
# Step 2: Filter years and months
# =============================================================================

def filter_period(df):
    mask = (
        (df['YEAR'] >= YEAR_START) &
        (df['YEAR'] <= YEAR_END) &
        (df['MONTH'] >= MONTH_START) &
        (df['MONTH'] <= MONTH_END)
    )
    return df[mask].copy().reset_index(drop=True)


# =============================================================================
# Step 3: Detect and fix anomalies and missing data
# =============================================================================

def clean_data(df):
    report = []
    report.append("=" * 60)
    report.append("DATA CLEANING REPORT")
    report.append(f"Source: {RAW_CSV}")
    report.append(
        f"Period: {YEAR_START}–{YEAR_END}, months {MONTH_START}–{MONTH_END}")
    report.append("=" * 60)

    # --- Missing values ---
    report.append("\nMISSING VALUES (before cleaning):")
    weather_cols = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M',
                    'PS', 'WS2M', 'T2MDEW', 'GWETROOT', 'GWETTOP',
                    'ALLSKY_SFC_SW_DWN']
    total_missing = 0
    for col in weather_cols:
        n = df[col].isna().sum()
        if n > 0:
            report.append(f"  {col:<25}: {n} missing")
            total_missing += n
    if total_missing == 0:
        report.append("  None")

    # --- Rainfall anomalies ---
    rain_anomalies = df[df['PRECTOTCORR'] > RAIN_ANOMALY_THRESHOLD]
    report.append(f"\nRAINFALL ANOMALIES (>{RAIN_ANOMALY_THRESHOLD} mm/day):")
    if len(rain_anomalies) > 0:
        for idx, row in rain_anomalies.iterrows():
            doy = int(row['DOY'])
            year = int(row['YEAR'])
            original = row['PRECTOTCORR']
            # Replace with average of same DOY ±3 days from other years
            mask = (df['DOY'].between(doy - 3, doy + 3)) & (df['YEAR'] != year)
            replacement = df.loc[mask, 'PRECTOTCORR'].mean()
            df.loc[idx, 'PRECTOTCORR'] = replacement
            report.append(
                f"  {row['DATE'].date()}: {original:.1f} → {replacement:.1f} mm")
    else:
        report.append("  None")

    # --- Temperature consistency ---
    temp_issues = df[(df['T2M_MIN'] > df['T2M']) | (df['T2M'] > df['T2M_MAX'])]
    report.append(
        f"\nTEMPERATURE INCONSISTENCIES (Tmin > Tmean or Tmean > Tmax):")
    report.append(f"  {len(temp_issues)} days")

    # --- Fill missing values with monthly average ---
    report.append(f"\nFILLING MISSING VALUES (monthly average):")
    for col in weather_cols:
        n_before = df[col].isna().sum()
        if n_before > 0:
            monthly_avg = df.groupby('MONTH')[col].transform('mean')
            df[col] = df[col].fillna(monthly_avg)
            n_after = df[col].isna().sum()
            report.append(f"  {col}: filled {n_before - n_after} values")

    # --- Summary ---
    report.append(f"\nSUMMARY:")
    report.append(f"  Total rows loaded (full dataset): see terminal")
    report.append(
        f"  Rows after filtering (Apr–Oct, {YEAR_START}–{YEAR_END}): {len(df)}")
    report.append(f"  Rainfall anomalies replaced: {len(rain_anomalies)}")
    report.append(f"  Missing values filled: {total_missing}")
    report.append(f"  Remaining NaN: {df[weather_cols].isna().sum().sum()}")

    return df, '\n'.join(report)


# =============================================================================
# Step 4: Compute ET0
# =============================================================================

# Extraterrestrial radiation
def compute_ra(doy):
    lat_rad = LAT * np.pi / 180
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    sdec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws_angle = np.arccos(
        np.clip(-np.tan(lat_rad) * np.tan(sdec), -1, 1)
    )
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws_angle * np.sin(lat_rad) * np.sin(sdec) +
        np.cos(lat_rad) * np.cos(sdec) * np.sin(ws_angle)
    )
    return Ra


def hargreaves_et0(tmean, tmax, tmin, Ra):
    tdiff = np.maximum(tmax - tmin, 0)
    # 0.408 converts Ra from MJ/m2/day to mm/day equivalent evaporation
    et0 = 0.0023 * (tmean + 17.8) * tdiff**0.5 * (0.408 * Ra)
    return np.clip(et0, 0, 20)


def penman_monteith_et0(tmean, tmax, tmin, rh, ws, rad, pressure, Ra):
    # Saturation vapor pressure
    es_max = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es_min = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_max + es_min) / 2

    # Actual vapor pressure
    ea = (rh / 100.0) * es_min

    # Slope of saturation vapor pressure curve
    delta = 4098 * 0.6108 * \
        np.exp(17.27 * tmean / (tmean + 237.3)) / (tmean + 237.3)**2

    # Psychrometric constant
    gamma = 0.000665 * pressure

    # Net shortwave radiation
    Rns = (1 - 0.23) * rad

    # Clear-sky radiation
    Rso = np.maximum((0.75 + 2e-5 * ELEVATION) * Ra, 0.1)

    # Net longwave radiation
    sigma = 4.903e-9
    Rnl = sigma * ((tmax + 273.16)**4 + (tmin + 273.16)**4) / 2 * \
        (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.001))) * \
        (1.35 * np.minimum(rad / Rso, 1.0) - 0.35)

    Rn = Rns - Rnl
    G = 0

    # FAO-56 equation
    num = 0.408 * delta * (Rn - G) + gamma * \
        (900 / (tmean + 273)) * ws * (es - ea)
    den = delta + gamma * (1 + 0.34 * ws)

    return np.clip(num / den, 0, 15)


def compute_et0(df):
    df = df.copy()
    Ra = compute_ra(df['DOY'].values)

    df['ET0_hargreaves'] = hargreaves_et0(
        df['T2M'].values, df['T2M_MAX'].values,
        df['T2M_MIN'].values, Ra
    )
    df['ET0_penman_monteith'] = penman_monteith_et0(
        df['T2M'].values, df['T2M_MAX'].values,
        df['T2M_MIN'].values, df['RH2M'].values,
        df['WS2M'].values, df['ALLSKY_SFC_SW_DWN'].values,
        df['PS'].values, Ra
    )
    return df


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading raw data...")
    df_raw = load_raw(RAW_CSV)
    n_raw = len(df_raw)

    # Filter
    print("Filtering Apr–Oct, 2000–2025...")
    df = filter_period(df_raw)

    # Clean
    print("Cleaning anomalies and missing data...")
    df, report = clean_data(df)

    # ET0
    print("Computing ET0...")
    df = compute_et0(df)

    # Save
    df.to_csv(OUTPUT_CSV, index=False)

    # Add load info to report
    report = report.replace(
        "Total rows loaded (full dataset): see terminal",
        f"Total rows loaded (full dataset): {n_raw}"
    )

    # Print and save report
    print("\n" + report)

    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {REPORT_TXT}")
    print(f"Rows in cleaned file: {len(df)}")
    print(f"Columns: {list(df.columns)}")
