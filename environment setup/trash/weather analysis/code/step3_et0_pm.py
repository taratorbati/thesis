"""
STEP 3: CORRECTED ET₀ WITH ACTUAL T2M_MIN + FAO-56 PENMAN-MONTEITH
====================================================================
Using newly downloaded NASA POWER data with all parameters.
"""

import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# LOAD NEW DATA
# ═══════════════════════════════════════════════════════════════

df = pd.read_csv(
    '/mnt/user-data/uploads/POWER_Point_Daily_20200101_20260415_038d30N_048d85E_LST.csv',
    skiprows=19  # new file has 19 header rows (line 19 = -END HEADER-)
)
df.columns = df.columns.str.strip()
df.replace(-999.0, np.nan, inplace=True)

df['DATE'] = pd.to_datetime(
    df['YEAR'].astype(int).astype(str) + df['DOY'].astype(int).astype(str).str.zfill(3),
    format='%Y%j'
)
df['MONTH'] = df['DATE'].dt.month

print("=" * 70)
print("NEW DATASET OVERVIEW")
print("=" * 70)
print(f"Records: {len(df)}")
print(f"Date range: {df['DATE'].min().strftime('%Y-%m-%d')} to {df['DATE'].max().strftime('%Y-%m-%d')}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing values:")
for col in ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'GWETROOT', 'GWETTOP',
            'WS2M', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'PS', 'T2MDEW']:
    n = df[col].isna().sum()
    if n > 0:
        first_miss = df[df[col].isna()]['DATE'].iloc[0].strftime('%Y-%m-%d')
        print(f"  {col:>20}: {n:>4} missing (from {first_miss})")

# ═══════════════════════════════════════════════════════════════
# COMPARE ACTUAL vs DERIVED T2M_MIN
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ACTUAL T2M_MIN vs DERIVED (2×Tmean − Tmax)")
print("=" * 70)

valid = df[df['T2M_MIN'].notna()].copy()
valid['T2M_MIN_derived'] = 2 * valid['T2M'] - valid['T2M_MAX']
diff = valid['T2M_MIN'] - valid['T2M_MIN_derived']

print(f"\nComparison across {len(valid)} days with valid data:")
print(f"  Actual T2M_MIN mean:   {valid['T2M_MIN'].mean():.2f}°C")
print(f"  Derived T2M_MIN mean:  {valid['T2M_MIN_derived'].mean():.2f}°C")
print(f"  Mean difference:       {diff.mean():.2f}°C (actual − derived)")
print(f"  Std of difference:     {diff.std():.2f}°C")
print(f"  Max |difference|:      {diff.abs().max():.2f}°C")
print(f"  RMSE:                  {np.sqrt((diff**2).mean()):.2f}°C")
print(f"  Correlation:           {valid['T2M_MIN'].corr(valid['T2M_MIN_derived']):.4f}")

# Season-specific comparison
for year in [2024, 2025]:
    mask = (valid['YEAR'] == year) & (valid['DOY'] >= 135) & (valid['DOY'] <= 254)
    s = valid[mask]
    d = s['T2M_MIN'] - s['T2M_MIN_derived']
    print(f"\n  {year} growing season (May 15–Sep 11):")
    print(f"    Actual Tmin mean: {s['T2M_MIN'].mean():.2f}°C vs Derived: {s['T2M_MIN_derived'].mean():.2f}°C")
    print(f"    Mean diff: {d.mean():.2f}°C, RMSE: {np.sqrt((d**2).mean()):.2f}°C")
    print(f"    Actual Tmin range: {s['T2M_MIN'].min():.1f} to {s['T2M_MIN'].max():.1f}°C")
    print(f"    Derived Tmin range: {s['T2M_MIN_derived'].min():.1f} to {s['T2M_MIN_derived'].max():.1f}°C")

# ═══════════════════════════════════════════════════════════════
# IMPUTATION (only for 2026 missing data — not used in analysis)
# ═══════════════════════════════════════════════════════════════

for col in ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'GWETROOT', 'GWETTOP',
            'WS2M', 'RH2M', 'PS', 'T2MDEW']:
    df[col] = df[col].fillna(df.groupby('DOY')[col].transform('mean'))
df['ALLSKY_SFC_SW_DWN'] = df['ALLSKY_SFC_SW_DWN'].fillna(
    df.groupby('MONTH')['ALLSKY_SFC_SW_DWN'].transform('mean')
)

# ═══════════════════════════════════════════════════════════════
# ET₀ CALCULATIONS — THREE METHODS
# ═══════════════════════════════════════════════════════════════

lat_rad = 38.298 * math.pi / 180
lat_deg = 38.298
elevation = 719.72  # from MERRA-2 header

print("\n" + "=" * 70)
print("ET₀ CALCULATION: THREE METHODS COMPARED")
print("=" * 70)

# ── Helper functions ──

def compute_Ra(doy, lat_rad):
    """Extraterrestrial radiation in MJ/m²/day"""
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.4093 * math.sin(2 * math.pi * doy / 365 - 1.39)
    cos_arg = max(-1, min(1, -math.tan(lat_rad) * math.tan(delta)))
    ws = math.acos(cos_arg)
    Ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(delta) +
        math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )
    return Ra

def saturation_vp(T):
    """Saturation vapour pressure (kPa) from temperature (°C) — FAO Eq. 11"""
    return 0.6108 * np.exp((17.27 * T) / (T + 237.3))

def slope_svp(T):
    """Slope of saturation vapour pressure curve (kPa/°C) — FAO Eq. 13"""
    return 4098 * saturation_vp(T) / (T + 237.3)**2

# ── Compute for each day ──

df['Ra_MJ'] = df['DOY'].apply(lambda d: compute_Ra(int(d), lat_rad))
df['Ra_mm'] = df['Ra_MJ'] / 2.45

# Actual temperature range (with real T2M_MIN!)
df['Trange_actual'] = (df['T2M_MAX'] - df['T2M_MIN']).clip(lower=0)
# Old derived range for comparison
df['T2M_MIN_derived'] = 2 * df['T2M'] - df['T2M_MAX']
df['Trange_derived'] = (df['T2M_MAX'] - df['T2M_MIN_derived']).clip(lower=0)

# ── METHOD 1: Hargreaves with DERIVED Tmin (your old method, but Ra corrected) ──
df['ET0_harg_derived'] = 0.0023 * (df['T2M'] + 17.8) * np.sqrt(df['Trange_derived']) * df['Ra_mm']

# ── METHOD 2: Hargreaves with ACTUAL T2M_MIN ──
df['ET0_harg_actual'] = 0.0023 * (df['T2M'] + 17.8) * np.sqrt(df['Trange_actual']) * df['Ra_mm']

# ── METHOD 3: FAO-56 Penman-Monteith ──
# Reference: FAO Irrigation & Drainage Paper 56, Eq. 6
# ET0 = [0.408 Δ(Rn-G) + γ(900/(T+273))u2(es-ea)] / [Δ + γ(1 + 0.34 u2)]

# Step 1: Atmospheric pressure and psychrometric constant
df['P_kPa'] = df['PS']  # actual surface pressure from MERRA-2 (already kPa)
df['gamma'] = 0.000665 * df['P_kPa']  # psychrometric constant (kPa/°C)

# Step 2: Saturation and actual vapour pressure
df['es'] = (saturation_vp(df['T2M_MAX']) + saturation_vp(df['T2M_MIN'])) / 2
# Actual vapor pressure from dew point (most reliable method — FAO Eq. 14)
df['ea'] = saturation_vp(df['T2MDEW'])
# Cross-check with RH method
df['ea_rh'] = df['es'] * df['RH2M'] / 100
df['vpd'] = (df['es'] - df['ea']).clip(lower=0)

# Step 3: Slope of saturation vapour pressure curve
df['Delta'] = slope_svp(df['T2M'])

# Step 4: Net radiation
# Rns = (1 - albedo) × Rs; albedo = 0.23 for grass reference
df['Rns'] = (1 - 0.23) * df['ALLSKY_SFC_SW_DWN']

# Rnl (net outgoing longwave) — FAO Eq. 39
sigma = 4.903e-9  # Stefan-Boltzmann (MJ/m²/day/K⁴)
df['Rnl'] = sigma * ((df['T2M_MAX'] + 273.16)**4 + (df['T2M_MIN'] + 273.16)**4) / 2 * \
            (0.34 - 0.14 * np.sqrt(df['ea'])) * \
            (1.35 * df['ALLSKY_SFC_SW_DWN'] / (0.75 * df['Ra_MJ']) - 0.35)
# Clamp Rnl to reasonable range (clear-sky ratio can go >1 in some edge cases)
df['Rso'] = (0.75 + 2e-5 * elevation) * df['Ra_MJ']
ratio = (df['ALLSKY_SFC_SW_DWN'] / df['Rso']).clip(0.3, 1.0)
df['Rnl'] = sigma * ((df['T2M_MAX'] + 273.16)**4 + (df['T2M_MIN'] + 273.16)**4) / 2 * \
            (0.34 - 0.14 * np.sqrt(df['ea'])) * \
            (1.35 * ratio - 0.35)

df['Rn'] = df['Rns'] - df['Rnl']

# Step 5: Soil heat flux (G ≈ 0 for daily calculation — FAO recommendation)
G = 0

# Step 6: Penman-Monteith equation
df['ET0_PM'] = (
    (0.408 * df['Delta'] * (df['Rn'] - G) +
     df['gamma'] * (900 / (df['T2M'] + 273)) * df['WS2M'] * df['vpd'])
    /
    (df['Delta'] + df['gamma'] * (1 + 0.34 * df['WS2M']))
)
df['ET0_PM'] = df['ET0_PM'].clip(lower=0)

# ═══════════════════════════════════════════════════════════════
# RESULTS COMPARISON
# ═══════════════════════════════════════════════════════════════

print(f"\nSite: 38.298°N, 48.847°E, elevation {elevation:.0f}m")
print(f"Actual surface pressure (mean): {df['PS'].mean():.1f} kPa")
print(f"Mean wind speed (2m): {df['WS2M'].mean():.2f} m/s")
print(f"Mean RH: {df['RH2M'].mean():.1f}%")

# Compare methods across season windows
DOY_S, DOY_E = 135, 254  # May 15 – Sep 11

print(f"\n{'─'*75}")
print(f"COMPARISON: May 15 – Sep 11 season (DOY {DOY_S}–{DOY_E})")
print(f"{'─'*75}")
print(f"\n{'Year':>6} {'Harg(derived)':>14} {'Harg(actual)':>14} {'Penman-Mont':>14} {'PM/Harg ratio':>14}")
print(f"{'':>6} {'mm/season':>14} {'mm/season':>14} {'mm/season':>14} {'':>14}")

for year in range(2020, 2026):
    mask = (df['YEAR'] == year) & (df['DOY'] >= DOY_S) & (df['DOY'] <= DOY_E)
    s = df[mask]
    if len(s) < 100:
        continue
    h_d = s['ET0_harg_derived'].sum()
    h_a = s['ET0_harg_actual'].sum()
    pm = s['ET0_PM'].sum()
    ratio_val = pm / h_a if h_a > 0 else 0
    print(f"{year:>6} {h_d:>14.1f} {h_a:>14.1f} {pm:>14.1f} {ratio_val:>14.2f}")

# Detailed comparison for scenario years
print(f"\n{'─'*75}")
print(f"DETAILED SCENARIO COMPARISON (May 15 – Sep 11)")
print(f"{'─'*75}")

for year in [2024, 2025]:
    mask = (df['YEAR'] == year) & (df['DOY'] >= DOY_S) & (df['DOY'] <= DOY_E)
    s = df[mask]
    print(f"\n  {year} ({'WET' if year == 2024 else 'DRY'} scenario):")
    print(f"  {'Method':<25} {'Daily mean':>12} {'Season total':>14} {'Kc=1.15 ETc':>14}")
    for method, col in [('Hargreaves (derived)',  'ET0_harg_derived'),
                         ('Hargreaves (actual)',   'ET0_harg_actual'),
                         ('Penman-Monteith (FAO)', 'ET0_PM')]:
        daily = s[col].mean()
        total = s[col].sum()
        etc = total * 1.15
        print(f"  {method:<25} {daily:>10.2f} mm/d {total:>12.1f} mm {etc:>12.1f} mm")
    
    precip = s['PRECTOTCORR'].sum()
    print(f"\n  Precipitation:           {precip:.1f} mm")
    print(f"  Water deficit (P - ETc):")
    for method, col in [('Hargreaves (actual)', 'ET0_harg_actual'),
                         ('Penman-Monteith',     'ET0_PM')]:
        etc = s[col].sum() * 1.15
        print(f"    {method:<25} {precip - etc:>10.1f} mm")

# Daily method comparison stats
print(f"\n{'─'*75}")
print(f"DAILY COMPARISON STATISTICS (all complete years, 2020–2025)")
print(f"{'─'*75}")
complete = df[(df['YEAR'] <= 2025) & (df['ET0_PM'].notna()) & (df['ET0_harg_actual'].notna())]
diff_pm_ha = complete['ET0_PM'] - complete['ET0_harg_actual']
print(f"  PM vs Hargreaves(actual): mean diff = {diff_pm_ha.mean():.3f} mm/d")
print(f"  PM vs Hargreaves(actual): RMSE = {np.sqrt((diff_pm_ha**2).mean()):.3f} mm/d")
print(f"  Correlation: r = {complete['ET0_PM'].corr(complete['ET0_harg_actual']):.4f}")
print(f"  PM mean: {complete['ET0_PM'].mean():.3f} mm/d, Harg mean: {complete['ET0_harg_actual'].mean():.3f} mm/d")

# Check vapor pressure deficit
print(f"\n{'─'*75}")
print(f"CLIMATE PARAMETERS (growing season means, 2020–2025)")
print(f"{'─'*75}")
for year in range(2020, 2026):
    mask = (df['YEAR'] == year) & (df['DOY'] >= DOY_S) & (df['DOY'] <= DOY_E)
    s = df[mask]
    if len(s) < 100:
        continue
    print(f"  {year}: Tmean={s['T2M'].mean():.1f}°C, Tmin={s['T2M_MIN'].mean():.1f}°C, "
          f"RH={s['RH2M'].mean():.0f}%, WS={s['WS2M'].mean():.1f}m/s, "
          f"VPD={s['vpd'].mean():.2f}kPa, Rs={s['ALLSKY_SFC_SW_DWN'].mean():.1f}MJ/m²/d")

# ═══════════════════════════════════════════════════════════════
# UPDATED RICE SUITABILITY WITH ACTUAL T2M_MIN
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"UPDATED RICE TEMPERATURE SUITABILITY (with actual T2M_MIN)")
print(f"{'='*70}")

for year in [2024, 2025]:
    mask = (df['YEAR'] == year) & (df['DOY'] >= 135) & (df['DOY'] <= 254)
    s = df[mask].copy().reset_index(drop=True)
    s['day'] = range(len(s))
    
    print(f"\n  {year} ({'WET' if year == 2024 else 'DRY'}):")
    print(f"    Actual Tmin: mean={s['T2M_MIN'].mean():.1f}°C, "
          f"min={s['T2M_MIN'].min():.1f}°C, max={s['T2M_MIN'].max():.1f}°C")
    print(f"    Days Tmin < 10°C: {(s['T2M_MIN'] < 10).sum()}")
    print(f"    Days Tmin < 15°C: {(s['T2M_MIN'] < 15).sum()}")
    print(f"    Days Tmax > 35°C: {(s['T2M_MAX'] > 35).sum()}")
    print(f"    GDD (base 10°C):  {(s['T2M'] - 10).clip(lower=0).sum():.0f} °C-days")
    
    # Cold events
    cold = s[s['T2M_MIN'] < 10]
    if len(cold) > 0:
        print(f"    Cold nights (<10°C):")
        for _, r in cold.iterrows():
            print(f"      Day {int(r['day'])} ({r['DATE'].strftime('%b %d')}): "
                  f"Tmin={r['T2M_MIN']:.1f}°C")

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
    'axes.titlesize': 11, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
})

C_HARG = '#e67e22'
C_PM = '#2980b9'
C_DRY = '#c0392b'
C_WET = '#2980b9'

# ── Figure 12: ET0 method comparison ──
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle('ET₀ Method Comparison: Hargreaves vs FAO-56 Penman-Monteith\n'
             'May 15 – Sep 11 (DOY 135–254) · 38.3°N, 48.8°E, 700 m',
             fontsize=11, fontweight='bold', y=0.99)

# (a) Daily ET0 time series — 2024
ax = axes[0, 0]
for year, color, label in [(2024, C_WET, '2024 (wet)'), (2025, C_DRY, '2025 (dry)')]:
    mask = (df['YEAR'] == year) & (df['DOY'] >= DOY_S) & (df['DOY'] <= DOY_E)
    s = df[mask].copy().reset_index(drop=True)
    s['day'] = range(len(s))
    pm7 = s['ET0_PM'].rolling(7, center=True).mean()
    ha7 = s['ET0_harg_actual'].rolling(7, center=True).mean()
    lw = 1.5
    ax.plot(s['day'], pm7, color=color, linewidth=lw, label=f'PM {year}')
    ax.plot(s['day'], ha7, color=color, linewidth=lw, linestyle='--', alpha=0.6, label=f'Harg {year}')

ax.set_ylabel('ET₀ (mm/day, 7-day avg)')
ax.set_xlabel('Day in season')
ax.set_title('(a) Daily ET₀ comparison', loc='left', fontsize=10)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7)

# (b) Scatter: PM vs Hargreaves
ax = axes[0, 1]
season = df[(df['YEAR'] <= 2025) & (df['DOY'] >= DOY_S) & (df['DOY'] <= DOY_E)]
ax.scatter(season['ET0_harg_actual'], season['ET0_PM'], s=4, alpha=0.3, color='#2c3e50')
lim = [0, 8]
ax.plot(lim, lim, 'k--', linewidth=0.5, alpha=0.5, label='1:1 line')
# Regression line
from numpy.polynomial.polynomial import polyfit
b, m = polyfit(season['ET0_harg_actual'].dropna(), season['ET0_PM'].dropna(), 1)
x_fit = np.linspace(0, 8, 100)
ax.plot(x_fit, b + m * x_fit, color='#e74c3c', linewidth=1, alpha=0.7,
        label=f'Fit: PM = {m:.2f}×Harg + {b:.2f}')
ax.set_xlabel('Hargreaves ET₀ (mm/day)')
ax.set_ylabel('Penman-Monteith ET₀ (mm/day)')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.set_title('(b) Method agreement (growing seasons)', loc='left', fontsize=10)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7)

# (c) Actual vs Derived Tmin
ax = axes[1, 0]
ax.scatter(valid['T2M_MIN_derived'], valid['T2M_MIN'], s=2, alpha=0.15, color='#2c3e50')
lim2 = [-15, 25]
ax.plot(lim2, lim2, 'k--', linewidth=0.5, alpha=0.5, label='1:1 line')
ax.set_xlabel('Derived Tmin (2×Tmean − Tmax) (°C)')
ax.set_ylabel('Actual T2M_MIN (°C)')
ax.set_xlim(lim2)
ax.set_ylim(lim2)
ax.set_aspect('equal')
ax.set_title('(c) T2M_MIN: actual vs derived', loc='left', fontsize=10)
rmse = np.sqrt(((valid['T2M_MIN'] - valid['T2M_MIN_derived'])**2).mean())
ax.annotate(f'RMSE = {rmse:.2f}°C\nr = {valid["T2M_MIN"].corr(valid["T2M_MIN_derived"]):.3f}',
            xy=(0.05, 0.85), xycoords='axes fraction', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(loc='lower right', framealpha=0.9, fontsize=7)

# (d) Seasonal totals bar chart
ax = axes[1, 1]
years = list(range(2020, 2026))
et0_harg = []
et0_pm = []
for year in years:
    mask = (df['YEAR'] == year) & (df['DOY'] >= DOY_S) & (df['DOY'] <= DOY_E)
    s = df[mask]
    if len(s) < 100:
        et0_harg.append(0)
        et0_pm.append(0)
        continue
    et0_harg.append(s['ET0_harg_actual'].sum())
    et0_pm.append(s['ET0_PM'].sum())

x = np.arange(len(years))
ax.bar(x - 0.17, et0_harg, 0.34, color=C_HARG, alpha=0.8, label='Hargreaves')
ax.bar(x + 0.17, et0_pm, 0.34, color=C_PM, alpha=0.8, label='Penman-Monteith')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_ylabel('Seasonal ET₀ (mm)')
ax.set_title('(d) Seasonal ET₀ totals', loc='left', fontsize=10)
ax.legend(framealpha=0.9)

for i, year in enumerate(years):
    if year == 2024:
        ax.annotate('WET', xy=(i, max(et0_harg[i], et0_pm[i])+10), ha='center', 
                    fontsize=7, color=C_WET, fontweight='bold')
    if year == 2025:
        ax.annotate('DRY', xy=(i, max(et0_harg[i], et0_pm[i])+10), ha='center', 
                    fontsize=7, color=C_DRY, fontweight='bold')

for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('/home/claude/fig12_et0_comparison.png', facecolor='white')
fig.savefig('/home/claude/fig12_et0_comparison.pdf', facecolor='white')
plt.close()
print("\nFigure 12 saved: ET0 method comparison")

# ── Figure 13: Updated scenario comparison with PM ET0 ──
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
fig.suptitle('Final Climate Scenarios: 2025 (Dry) vs 2024 (Wet)\n'
             'Penman-Monteith ET₀ · May 15 – Sep 11 · 38.3°N, 48.8°E, 700 m',
             fontsize=11, fontweight='bold', y=0.98)

def get_season(year):
    mask = (df['YEAR'] == year) & (df['DOY'] >= DOY_S) & (df['DOY'] <= DOY_E)
    s = df[mask].copy().reset_index(drop=True)
    s['day'] = range(len(s))
    s['cum_precip'] = s['PRECTOTCORR'].cumsum()
    s['cum_et0'] = s['ET0_PM'].cumsum()
    s['cum_etc'] = s['cum_et0'] * 1.15
    return s

s25 = get_season(2025)
s24 = get_season(2024)

# (a) Daily precipitation
ax = axes[0, 0]
ax.bar(s25['day'], s25['PRECTOTCORR'], width=1, color=C_DRY, alpha=0.7, label='2025 (dry)')
ax.bar(s24['day'], -s24['PRECTOTCORR'], width=1, color=C_WET, alpha=0.7, label='2024 (wet)')
ax.set_ylabel('Precipitation (mm/day)')
ax.set_xlabel('Day in season')
ax.set_title('(a) Daily precipitation', loc='left', fontsize=10)
ax.legend(loc='lower left', framealpha=0.9)
ax.axhline(0, color='k', linewidth=0.3)

# (b) Cumulative P vs ETc (using PM)
ax = axes[0, 1]
ax.plot(s25['day'], s25['cum_precip'], color=C_DRY, label='P (2025)')
ax.plot(s25['day'], s25['cum_etc'], color=C_DRY, linestyle='--', label='ETc (2025)')
ax.plot(s24['day'], s24['cum_precip'], color=C_WET, label='P (2024)')
ax.plot(s24['day'], s24['cum_etc'], color=C_WET, linestyle='--', label='ETc (2024)')
ax.fill_between(s25['day'], s25['cum_precip'], s25['cum_etc'], alpha=0.1, color=C_DRY)
ax.fill_between(s24['day'], s24['cum_precip'], s24['cum_etc'], alpha=0.1, color=C_WET)
ax.set_ylabel('Cumulative (mm)')
ax.set_xlabel('Day in season')
ax.set_title('(b) Cumulative P vs ETc (Penman-Monteith)', loc='left', fontsize=10)
ax.legend(loc='upper left', framealpha=0.9, ncol=2, fontsize=7)

# (c) Temperature with actual Tmin
ax = axes[1, 0]
ax.fill_between(s25['day'], s25['T2M_MIN'], s25['T2M_MAX'], alpha=0.1, color=C_DRY)
ax.fill_between(s24['day'], s24['T2M_MIN'], s24['T2M_MAX'], alpha=0.1, color=C_WET)
ax.plot(s25['day'], s25['T2M'], color=C_DRY, label='Tmean 2025')
ax.plot(s24['day'], s24['T2M'], color=C_WET, label='Tmean 2024')
ax.plot(s25['day'], s25['T2M_MIN'], color=C_DRY, linewidth=0.5, linestyle=':', alpha=0.5)
ax.plot(s24['day'], s24['T2M_MIN'], color=C_WET, linewidth=0.5, linestyle=':', alpha=0.5)
ax.axhline(15, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_ylabel('Temperature (°C)')
ax.set_xlabel('Day in season')
ax.set_title('(c) Temperature (actual T2M_MIN)', loc='left', fontsize=10)
ax.legend(loc='lower right', framealpha=0.9)

# (d) Soil wetness + precipitation overlay
ax = axes[1, 1]
ax2 = ax.twinx()
ax2.bar(s25['day'], s25['PRECTOTCORR'], width=1, color=C_DRY, alpha=0.15)
ax2.bar(s24['day'], s24['PRECTOTCORR'], width=1, color=C_WET, alpha=0.15)
ax2.set_ylabel('Precipitation (mm/day)', color='gray', alpha=0.5)
ax2.tick_params(axis='y', colors='gray')

ax.plot(s25['day'], s25['GWETROOT'], color=C_DRY, label='GWETROOT 2025')
ax.plot(s24['day'], s24['GWETROOT'], color=C_WET, label='GWETROOT 2024')
ax.set_ylabel('Soil wetness (0–1)')
ax.set_xlabel('Day in season')
ax.set_title('(d) Soil wetness + precipitation', loc='left', fontsize=10)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0.2, 0.8)

for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('/home/claude/fig13_final_scenarios.png', facecolor='white')
fig.savefig('/home/claude/fig13_final_scenarios.pdf', facecolor='white')
plt.close()
print("Figure 13 saved: Final scenario comparison with PM ET0")

# ── Summary table for thesis ──
print(f"\n{'='*70}")
print(f"THESIS-READY SUMMARY TABLE")
print(f"{'='*70}")
print(f"""
Table X: Climate scenario summary for irrigation control study
Site: 38.298°N, 48.847°E, elevation 720 m, Gilan Province, Iran
Season: May 15 – September 11 (DOY 135–254, 120 days)
ET₀ method: FAO-56 Penman-Monteith
Crop: Rice (Kc = 1.15 season average)
""")
print(f"{'Parameter':<35} {'2025 (Dry)':>12} {'2024 (Wet)':>12} {'Units':>8}")
print(f"{'─'*70}")
for year in [2025, 2024]:
    mask = (df['YEAR'] == year) & (df['DOY'] >= DOY_S) & (df['DOY'] <= DOY_E)
    s = df[mask]
    vals = {
        'Mean temperature': (f"{s['T2M'].mean():.1f}", '°C'),
        'Mean Tmax': (f"{s['T2M_MAX'].mean():.1f}", '°C'),
        'Mean Tmin': (f"{s['T2M_MIN'].mean():.1f}", '°C'),
        'Total precipitation': (f"{s['PRECTOTCORR'].sum():.1f}", 'mm'),
        'Rainy days (>1mm)': (f"{(s['PRECTOTCORR'] > 1).sum()}", 'days'),
        'Max daily precipitation': (f"{s['PRECTOTCORR'].max():.1f}", 'mm'),
        'ET₀ (Penman-Monteith)': (f"{s['ET0_PM'].sum():.1f}", 'mm'),
        'ETc (Kc=1.15)': (f"{s['ET0_PM'].sum()*1.15:.1f}", 'mm'),
        'Water deficit (P−ETc)': (f"{s['PRECTOTCORR'].sum() - s['ET0_PM'].sum()*1.15:.1f}", 'mm'),
        'ET₀ (Hargreaves)': (f"{s['ET0_harg_actual'].sum():.1f}", 'mm'),
        'Mean RH': (f"{s['RH2M'].mean():.0f}", '%'),
        'Mean wind speed (2m)': (f"{s['WS2M'].mean():.1f}", 'm/s'),
        'Mean solar radiation': (f"{s['ALLSKY_SFC_SW_DWN'].mean():.1f}", 'MJ/m²/d'),
        'Mean GWETROOT': (f"{s['GWETROOT'].mean():.3f}", '–'),
    }
    if year == 2025:
        stored = vals
    else:
        for param, (v2024, unit) in vals.items():
            v2025 = stored[param][0]
            print(f"  {param:<35} {v2025:>10} {v2024:>10} {unit:>8}")

print(f"\n✓ Analysis complete. All figures saved.")
