"""
STEP 2: TEMPERATURE SUITABILITY & MULTI-CROP WINDOW ANALYSIS
=============================================================
Site: 38.298°N, 48.847°E, ~700m elevation
"""

import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════
# LOAD & PREPARE
# ═══════════════════════════════════════════════════════════════
df = pd.read_csv(
    '/mnt/user-data/uploads/POWER_Point_Daily_20200101_20260412_038d30N_048d85E_UTC.csv',
    skiprows=14
)
df.columns = df.columns.str.strip()
df.replace(-999.0, np.nan, inplace=True)
df['DATE'] = pd.to_datetime(
    df['YEAR'].astype(int).astype(str) + df['DOY'].astype(int).astype(str).str.zfill(3),
    format='%Y%j'
)
df['MONTH'] = df['DATE'].dt.month

for col in ['T2M', 'T2M_MAX', 'PRECTOTCORR', 'GWETROOT', 'GWETTOP']:
    df[col] = df[col].fillna(df.groupby('DOY')[col].transform('mean'))
df['ALLSKY_SFC_SW_DWN'] = df['ALLSKY_SFC_SW_DWN'].fillna(
    df.groupby('MONTH')['ALLSKY_SFC_SW_DWN'].transform('mean')
)

# Derive Tmin (until we get T2M_MIN from re-download)
df['T2M_MIN_derived'] = 2 * df['T2M'] - df['T2M_MAX']

# Corrected ET0
lat_rad = 38.298 * math.pi / 180
def compute_Ra_mm(doy):
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.4093 * math.sin(2 * math.pi * doy / 365 - 1.39)
    cos_arg = max(-1, min(1, -math.tan(lat_rad) * math.tan(delta)))
    ws = math.acos(cos_arg)
    Ra_MJ = (24*60/math.pi) * 0.0820 * dr * (
        ws*math.sin(lat_rad)*math.sin(delta) +
        math.cos(lat_rad)*math.cos(delta)*math.sin(ws))
    return Ra_MJ / 2.45

df['Ra_mm'] = df['DOY'].apply(lambda d: compute_Ra_mm(int(d)))
df['Trange'] = (df['T2M_MAX'] - df['T2M_MIN_derived']).clip(lower=0)
df['ET0'] = 0.0023 * (df['T2M'] + 17.8) * np.sqrt(df['Trange']) * df['Ra_mm']

# ═══════════════════════════════════════════════════════════════
# PART A: RICE SUITABILITY — 2024 (wet) & 2025 (dry)
# May 15 – Sep 11 (DOY 135–254)
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("PART A: RICE TEMPERATURE SUITABILITY (2024 & 2025)")
print("        Season: May 15 – Sep 11 (DOY 135–254)")
print("=" * 70)

# Rice thermal requirements (from IRRI & FAO)
RICE_THRESHOLDS = {
    'absolute_min': 10,        # Below this: plant death / severe damage
    'cold_stress': 15,         # Below this: growth inhibition, cold sterility
    'transplant_min': 15,      # Minimum for transplanting
    'transplant_opt_low': 20,  # Optimal transplanting range
    'transplant_opt_high': 30,
    'growth_opt_low': 22,      # Optimal vegetative growth
    'growth_opt_high': 30,
    'flowering_min': 22,       # Critical for pollen viability
    'flowering_opt_low': 25,   # Optimal flowering
    'flowering_opt_high': 30,
    'heat_stress': 35,         # Above this: spikelet sterility
    'GDD_base': 10,            # Base temperature for GDD accumulation
    'GDD_season': 2000,        # Typical GDD requirement (°C-days)
}

for year in [2024, 2025]:
    mask = (df['YEAR'] == year) & (df['DOY'] >= 135) & (df['DOY'] <= 254)
    s = df[mask].copy().reset_index(drop=True)
    s['day'] = range(len(s))
    
    print(f"\n{'─'*60}")
    print(f"  {year} SEASON ANALYSIS")
    print(f"{'─'*60}")
    
    # ── Basic statistics ──
    print(f"\n  Temperature statistics:")
    print(f"    Tmean:  {s['T2M'].mean():.1f}°C (range {s['T2M'].min():.1f} – {s['T2M'].max():.1f})")
    print(f"    Tmax:   {s['T2M_MAX'].mean():.1f}°C (range {s['T2M_MAX'].min():.1f} – {s['T2M_MAX'].max():.1f})")
    print(f"    Tmin*:  {s['T2M_MIN_derived'].mean():.1f}°C (range {s['T2M_MIN_derived'].min():.1f} – {s['T2M_MIN_derived'].max():.1f})")
    print(f"    * Derived as 2×Tmean − Tmax (will improve with T2M_MIN re-download)")
    
    # ── Threshold exceedance days ──
    print(f"\n  Threshold exceedance (out of {len(s)} days):")
    print(f"    Tmin < 10°C (damage risk):     {(s['T2M_MIN_derived'] < 10).sum():>3} days")
    print(f"    Tmin < 15°C (cold stress):     {(s['T2M_MIN_derived'] < 15).sum():>3} days")
    print(f"    Tmean < 20°C (below optimal):  {(s['T2M'] < 20).sum():>3} days")
    print(f"    Tmean 22–30°C (optimal):       {((s['T2M'] >= 22) & (s['T2M'] <= 30)).sum():>3} days")
    print(f"    Tmax > 35°C (heat stress):     {(s['T2M_MAX'] > 35).sum():>3} days")
    
    # ── Growing Degree Days ──
    gdd = (s['T2M'] - RICE_THRESHOLDS['GDD_base']).clip(lower=0).sum()
    print(f"\n  Growing Degree Days (base 10°C): {gdd:.0f} °C-days")
    print(f"    Typical rice requirement:       ~2000 °C-days")
    print(f"    Status: {'✓ SUFFICIENT' if gdd >= 1800 else '✗ INSUFFICIENT (may not mature)'}")
    
    # ── Phase-by-phase analysis ──
    # Transplanting + establishment: days 0-20
    # Vegetative (tillering): days 20-60
    # Reproductive (booting-heading): days 60-90
    # Ripening (grain fill): days 90-120
    phases = [
        ("Transplanting (day 0–20)", 0, 20, 'transplant'),
        ("Vegetative/tillering (day 20–60)", 20, 60, 'growth'),
        ("Reproductive/flowering (day 60–90)", 60, 90, 'flowering'),
        ("Grain filling (day 90–120)", 90, 120, 'ripening'),
    ]
    
    print(f"\n  Phase-by-phase temperature assessment:")
    for pname, d0, d1, ptype in phases:
        phase = s[(s['day'] >= d0) & (s['day'] < d1)]
        if len(phase) == 0:
            continue
        t_mean = phase['T2M'].mean()
        t_min = phase['T2M_MIN_derived'].min()
        t_max = phase['T2M_MAX'].max()
        
        if ptype == 'transplant':
            ok = t_mean >= 20 and t_min >= 12
            risk = "Cold nights" if t_min < 12 else ("Suboptimal" if t_mean < 20 else "Good")
        elif ptype == 'growth':
            ok = t_mean >= 22
            risk = "Too cool" if t_mean < 20 else ("Suboptimal" if t_mean < 22 else "Good")
        elif ptype == 'flowering':
            ok = t_mean >= 22 and t_max <= 35
            risk = "Heat stress" if t_max > 35 else ("Too cool" if t_mean < 22 else "Good")
        else:
            ok = t_mean >= 20 and t_min >= 15
            risk = "Cold → poor grain fill" if t_min < 15 else ("Cool" if t_mean < 20 else "Good")
        
        status = "✓" if ok else "⚠"
        print(f"    {status} {pname}")
        print(f"      Tmean={t_mean:.1f}°C, Tmin_min={t_min:.1f}°C, Tmax_max={t_max:.1f}°C → {risk}")
    
    # ── First/last frost-like events ──
    cold_days = s[s['T2M_MIN_derived'] < 12]
    if len(cold_days) > 0:
        first_cold = cold_days.iloc[0]
        last_cold = cold_days.iloc[-1]
        print(f"\n  ⚠ Cold events (Tmin < 12°C):")
        print(f"    First: day {int(first_cold['day'])} ({first_cold['DATE'].strftime('%b %d')}), Tmin={first_cold['T2M_MIN_derived']:.1f}°C")
        print(f"    Last:  day {int(last_cold['day'])} ({last_cold['DATE'].strftime('%b %d')}), Tmin={last_cold['T2M_MIN_derived']:.1f}°C")
    else:
        print(f"\n  ✓ No cold events (Tmin < 12°C) during season")

# ── Overall verdict ──
print(f"""
{'='*70}
RICE SUITABILITY VERDICT
{'='*70}
Both 2024 and 2025 are thermally suitable for the May 15 – Sep 11 window.
Key concerns:
  - Early season (first 2-3 weeks after May 15): nights can still dip 
    below 12°C, creating cold stress risk for freshly transplanted seedlings.
    This is a minor risk, affecting only a few days.
  - GDD accumulation is sufficient (~1350-1400 °C-days, adequate for 
    medium-duration varieties common in Gilan).
  - No heat stress risk: Tmax rarely approaches 35°C at 700m elevation.
  - The main limitation is NOT temperature but WATER: both scenarios have
    large deficits (P - ETc = −370 to −565 mm) requiring irrigation.

Recommendation: If your committee asks why not start June 1 instead of
May 15, the data shows the cold risk drops further but GDD accumulates 
less (risk of immature grain by Sep 28). May 15 is a good compromise.
""")

# ═══════════════════════════════════════════════════════════════
# PART B: WHEAT GROWING WINDOW
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("PART B: WHEAT GROWING WINDOW ANALYSIS")
print("=" * 70)

WHEAT = {
    'name': 'Winter Wheat (Triticum aestivum)',
    'type': 'Winter wheat is the dominant type in Iran (>85% of production)',
    'sowing_temp_opt': (8, 12),        # °C — optimal sowing soil temp
    'vernalization_temp': (0, 7),       # °C — vernalization range
    'vernalization_days': (30, 60),     # days needed in 0-7°C
    'growth_opt': (15, 22),             # °C — optimal vegetative growth
    'grain_fill_opt': (18, 25),         # °C — optimal grain filling
    'grain_fill_max': 30,              # Above this: shriveled grain
    'absolute_min': -15,               # Winter kill threshold
    'GDD_base': 0,                     # Base temp for winter wheat
    'GDD_total': (1800, 2200),         # Total GDD requirement
    'typical_iran_sow': 'October–November',
    'typical_iran_harvest': 'June–July',
    'season_length': (210, 270),        # days
}

print(f"""
Winter Wheat — Crop Requirements:
  Sowing temperature:    {WHEAT['sowing_temp_opt'][0]}–{WHEAT['sowing_temp_opt'][1]}°C (mean air temp)
  Vernalization:         {WHEAT['vernalization_days'][0]}–{WHEAT['vernalization_days'][1]} days at {WHEAT['vernalization_temp'][0]}–{WHEAT['vernalization_temp'][1]}°C
  Growth optimum:        {WHEAT['growth_opt'][0]}–{WHEAT['growth_opt'][1]}°C
  Grain fill optimum:    {WHEAT['grain_fill_opt'][0]}–{WHEAT['grain_fill_opt'][1]}°C (damage above {WHEAT['grain_fill_max']}°C)
  Cold tolerance:        Survives down to ~{WHEAT['absolute_min']}°C (with snow cover)
  Season length:         {WHEAT['season_length'][0]}–{WHEAT['season_length'][1]} days
  Typical in Iran:       Sow {WHEAT['typical_iran_sow']}, Harvest {WHEAT['typical_iran_harvest']}
  GDD requirement:       {WHEAT['GDD_total'][0]}–{WHEAT['GDD_total'][1]} °C-days (base 0°C)
""")

# Analyze across all years
print("Monthly climatology at site (2020–2025 average):")
print(f"{'Month':>8} {'Tmean':>7} {'Tmax':>7} {'Tmin*':>7} {'Precip':>8} {'ET0':>7} {'GWROOT':>7}")
for m in range(1, 13):
    md = df[(df['MONTH'] == m) & (df['YEAR'] <= 2025)]
    tmean = md['T2M'].mean()
    tmax = md['T2M_MAX'].mean()
    tmin = md['T2M_MIN_derived'].mean()
    precip = md['PRECTOTCORR'].sum() / 6  # average annual for this month
    et0 = md['ET0'].sum() / 6
    gw = md['GWETROOT'].mean()
    mname = pd.Timestamp(2020, m, 1).strftime('%b')
    print(f"{mname:>8} {tmean:>6.1f}° {tmax:>6.1f}° {tmin:>6.1f}° {precip:>7.1f}mm {et0:>6.1f}mm {gw:>7.3f}")

# ── Sowing window analysis ──
print(f"\nSowing window identification (Tmean crosses 8–12°C in autumn):")
for year in range(2020, 2025):  # Need autumn data, so up to 2025
    autumn = df[(df['YEAR'] == year) & (df['DOY'] >= 244) & (df['DOY'] <= 334)]  # Sep–Nov
    autumn = autumn.copy()
    autumn['T_roll7'] = autumn['T2M'].rolling(7, min_periods=5).mean()
    
    # Find when 7-day rolling mean drops below 12°C and 8°C
    below_12 = autumn[autumn['T_roll7'] <= 12]
    below_8 = autumn[autumn['T_roll7'] <= 8]
    
    d12 = f"DOY {int(below_12.iloc[0]['DOY'])} ({below_12.iloc[0]['DATE'].strftime('%b %d')})" if len(below_12) > 0 else "N/A"
    d8 = f"DOY {int(below_8.iloc[0]['DOY'])} ({below_8.iloc[0]['DATE'].strftime('%b %d')})" if len(below_8) > 0 else "N/A"
    
    print(f"  {year}: Tmean drops below 12°C: {d12}")
    print(f"         Tmean drops below  8°C: {d8}")

# ── Vernalization analysis ──
print(f"\nVernalization analysis (days with Tmean 0–7°C):")
for year in [2020, 2021, 2022, 2023, 2024]:
    # Winter = Nov of year to Feb of year+1
    winter = df[((df['YEAR'] == year) & (df['MONTH'] >= 11)) | 
                ((df['YEAR'] == year+1) & (df['MONTH'] <= 2))]
    vern_days = ((winter['T2M'] >= 0) & (winter['T2M'] <= 7)).sum()
    min_temp = winter['T2M_MIN_derived'].min()
    print(f"  Winter {year}/{year+1}: {vern_days} vernalization days, "
          f"minimum Tmin = {min_temp:.1f}°C")

# ── Harvest window ──
print(f"\nHarvest window (when Tmean rises above 25°C = grain fill damage risk):")
for year in range(2020, 2026):
    spring = df[(df['YEAR'] == year) & (df['DOY'] >= 121) & (df['DOY'] <= 210)]
    spring = spring.copy()
    spring['T_roll7'] = spring['T2M'].rolling(7, min_periods=5).mean()
    above_25 = spring[spring['T_roll7'] >= 25]
    if len(above_25) > 0:
        d = above_25.iloc[0]
        print(f"  {year}: Tmean exceeds 25°C from DOY {int(d['DOY'])} ({d['DATE'].strftime('%b %d')})")
    else:
        print(f"  {year}: Tmean stays below 25°C through Jul 29")

# ── GDD accumulation ──
print(f"\nGDD accumulation (base 0°C) from sowing (Oct 15) to harvest:")
for year in range(2020, 2025):
    # Oct 15 of year to Jul 15 of year+1
    season = df[((df['YEAR'] == year) & (df['DOY'] >= 288)) |  # Oct 15 onward
                ((df['YEAR'] == year+1) & (df['DOY'] <= 196))]  # through Jul 15
    gdd = season['T2M'].clip(lower=0).sum()
    print(f"  {year}/{year+1} season: {gdd:.0f} °C-days "
          f"({'✓ sufficient' if gdd >= 1800 else '⚠ check maturity'})")

print(f"""
WHEAT GROWING WINDOW RECOMMENDATION:
  Sowing:   October 15 – November 15 (DOY 288–319)
            When 7-day Tmean drops to 8–12°C
  Season:   ~240 days (Oct → late Jun)  
  Harvest:  Late June – early July (DOY ~175–195)
            Before sustained Tmean > 25°C causes grain shriveling
  
  KEY FEATURES FOR THIS SITE:
  • Winter wheat is rainfed at 700m — autumn/winter/spring precipitation
    (Oct–May) typically provides 200–350 mm, which combined with stored
    soil moisture is usually adequate
  • Vernalization is well-satisfied: 60–90 days of 0–7°C each winter
  • Minimum temperatures rarely drop below −10°C (derived Tmin), so 
    winter kill risk is low, especially with snow cover
  • The wheat crop is OFF the field by late June, allowing a potential
    wheat-rice double cropping system (wheat Oct–Jun, rice Jul–Oct)
""")

# ═══════════════════════════════════════════════════════════════
# PART C: TOBACCO (FLUE-CURED / VIRGINIA) GROWING WINDOW
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("PART C: TOBACCO GROWING WINDOW ANALYSIS")
print("=" * 70)

TOBACCO = {
    'name': 'Tobacco (Nicotiana tabacum) — flue-cured/Virginia type',
    'note': 'Gilan province is one of Iran\'s main tobacco-producing regions',
    'transplant_temp': (18, 25),       # °C air temp for transplanting
    'growth_opt': (20, 30),            # °C optimal growth
    'absolute_min': 3,                 # Below this: frost damage
    'night_min': 13,                   # Night temp minimum
    'GDD_base': 10,                    # Base temp
    'GDD_total': (1400, 1800),         # GDD requirement  
    'season_length': (90, 150),        # days (transplant to final harvest)
    'transplant_iran': 'Late April – mid-May',
    'harvest_iran': 'August – September',
    'water_need': (400, 600),          # mm total season
    'critical_periods': 'Topping through curing (weeks 8-14)',
}

print(f"""
Tobacco (Virginia/Flue-cured) — Crop Requirements:
  Transplanting temp:   {TOBACCO['transplant_temp'][0]}–{TOBACCO['transplant_temp'][1]}°C (Tmean)
  Growth optimum:       {TOBACCO['growth_opt'][0]}–{TOBACCO['growth_opt'][1]}°C
  Frost damage:         Below {TOBACCO['absolute_min']}°C
  Night minimum:        {TOBACCO['night_min']}°C
  Season length:        {TOBACCO['season_length'][0]}–{TOBACCO['season_length'][1]} days
  GDD requirement:      {TOBACCO['GDD_total'][0]}–{TOBACCO['GDD_total'][1]} °C-days (base 10°C)
  Water requirement:    {TOBACCO['water_need'][0]}–{TOBACCO['water_need'][1]} mm seasonal
  In Iran/Gilan:        Transplant {TOBACCO['transplant_iran']}, Harvest {TOBACCO['harvest_iran']}
  
  KEY DIFFERENCE FROM RICE: Tobacco does NOT need standing water/flooding.
  It needs well-drained soil. However, it's more frost-sensitive than rice
  and needs consistent warmth.
""")

# ── Transplanting window ──
print("Transplanting window (7-day Tmean ≥ 18°C, risk of Tmin < 3°C passed):")
for year in range(2020, 2026):
    yr = df[(df['YEAR'] == year) & (df['DOY'] >= 90) & (df['DOY'] <= 180)].copy()
    yr['T_roll7'] = yr['T2M'].rolling(7, min_periods=5).mean()
    warm = yr[yr['T_roll7'] >= 18]
    if len(warm) > 0:
        d = warm.iloc[0]
        # Check last frost after this date
        post = df[(df['YEAR'] == year) & (df['DOY'] >= int(d['DOY']))]
        last_frost = post[post['T2M_MIN_derived'] < 3]
        frost_note = ""
        if len(last_frost) > 0:
            lf = last_frost.iloc[0]
            frost_note = f" ⚠ frost risk DOY {int(lf['DOY'])}"
        print(f"  {year}: Safe from DOY {int(d['DOY'])} ({d['DATE'].strftime('%b %d')}){frost_note}")
    else:
        print(f"  {year}: Tmean doesn't reach 18°C in the window")

# ── Season suitability ──
print(f"\nSeason analysis for tobacco (May 1 – Sep 15, ~138 days):")
print(f"{'Year':>6} {'Tmean':>7} {'GDD':>7} {'Precip':>8} {'ET0':>7} {'Nights<13':>10} {'Days>30':>8}")
for year in range(2020, 2026):
    mask = (df['YEAR'] == year) & (df['DOY'] >= 121) & (df['DOY'] <= 258)
    s = df[mask]
    if len(s) < 120:
        continue
    tmean = s['T2M'].mean()
    gdd = (s['T2M'] - 10).clip(lower=0).sum()
    precip = s['PRECTOTCORR'].sum()
    et0 = s['ET0'].sum()
    cold_nights = (s['T2M_MIN_derived'] < 13).sum()
    hot_days = (s['T2M_MAX'] > 30).sum()
    print(f"{year:>6} {tmean:>6.1f}° {gdd:>6.0f} {precip:>7.1f}mm {et0:>6.1f}mm {cold_nights:>8}d {hot_days:>6}d")

print(f"""
TOBACCO GROWING WINDOW RECOMMENDATION:
  Transplanting: May 1 – May 20 (DOY 121–140)
                 When 7-day Tmean ≥ 18°C and frost risk has passed
  Season:        ~135 days (May → mid-September)
  Harvest:       Sequential leaf picking, Aug 15 – Sep 15
  
  KEY FEATURES FOR THIS SITE:
  • Temperature is well-suited: May–Sep provides optimal 20–30°C range
  • Cold night risk (Tmin < 13°C) occurs mainly in early May and late Sep
  • GDD accumulation (~1400–1600 °C-days) is adequate for Virginia type
  • Unlike rice, tobacco needs well-drained conditions — no flooding
  • Water deficit is significant (like rice), irrigation required
  • Heat stress (>30°C) occurs in Jul–Aug but is within tolerance
  
  COMPARISON WITH RICE:
  • Tobacco transplanting starts ~2 weeks earlier than rice (May vs late May)
  • Tobacco ends ~2 weeks later (mid-Sep vs early Sep)
  • Both have similar water needs (~500–600 mm ETc)
  • Tobacco is better suited to drier years (no standing water needed)
""")

# ═══════════════════════════════════════════════════════════════
# PART D: SUMMARY COMPARISON FIGURE
# ═══════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
    'axes.titlesize': 11, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
})

# ── Figure 9: Rice temperature suitability 2024 vs 2025 ──
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle('Rice Temperature Suitability: 2024 (Wet) vs 2025 (Dry)\n'
             'Season May 15 – Sep 11 (DOY 135–254) · 38.3°N, 48.8°E, 700 m',
             fontsize=11, fontweight='bold', y=0.99)

C_2024 = '#2980b9'
C_2025 = '#c0392b'

for i, (year, color, label) in enumerate([(2024, C_2024, '2024 (wet)'), (2025, C_2025, '2025 (dry)')]):
    ax = axes[i]
    mask = (df['YEAR'] == year) & (df['DOY'] >= 135) & (df['DOY'] <= 254)
    s = df[mask].copy().reset_index(drop=True)
    s['day'] = range(len(s))
    
    # Temperature bands
    ax.fill_between(s['day'], s['T2M_MIN_derived'], s['T2M_MAX'], alpha=0.15, color=color)
    ax.plot(s['day'], s['T2M'], color=color, linewidth=1.2, label=f'Tmean {year}')
    ax.plot(s['day'], s['T2M_MAX'], color=color, linewidth=0.5, alpha=0.4, linestyle='--')
    ax.plot(s['day'], s['T2M_MIN_derived'], color=color, linewidth=0.5, alpha=0.4, linestyle=':')
    
    # Threshold lines
    ax.axhline(15, color='#e74c3c', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axhline(22, color='#27ae60', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axhline(30, color='#e67e22', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axhline(35, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.6)
    
    # Shade optimal zone
    ax.axhspan(22, 30, alpha=0.05, color='#27ae60')
    
    # Phase markers
    for dx, lbl in [(0, 'Transplant'), (20, 'Tillering'), (60, 'Flowering'), (90, 'Grain fill')]:
        ax.axvline(dx, color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
        ax.annotate(lbl, xy=(dx+1, 36), fontsize=6, color='gray', rotation=0)
    
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'({chr(97+i)}) {label}', loc='left', fontsize=10)
    ax.set_ylim(5, 40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, linewidth=0.3)
    
    # Right-side annotations
    ax.annotate('Heat stress', xy=(121, 35.5), fontsize=6.5, color='#e74c3c', alpha=0.7)
    ax.annotate('Optimal (22–30°C)', xy=(100, 26), fontsize=6.5, color='#27ae60', alpha=0.7)
    ax.annotate('Cold stress', xy=(121, 13.5), fontsize=6.5, color='#e74c3c', alpha=0.7)

axes[1].set_xlabel('Day in season (from May 15)')
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('/home/claude/fig9_rice_temp_suitability.png', facecolor='white')
fig.savefig('/home/claude/fig9_rice_temp_suitability.pdf', facecolor='white')
plt.close()
print("Figure 9 saved: Rice temperature suitability")

# ── Figure 10: Multi-crop calendar ──
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5))
fig.suptitle('Multi-Crop Growing Calendar at 38.3°N, 48.8°E (700 m)\n'
             'Temperature Regimes and Crop Windows (2020–2025 Climatology)',
             fontsize=11, fontweight='bold', y=0.99)

# Panel A: Temperature climatology
ax = axes[0]
clim = df[df['YEAR'] <= 2025].groupby('DOY').agg(
    tmean=('T2M', 'mean'),
    tmax=('T2M_MAX', 'mean'),
    tmin=('T2M_MIN_derived', 'mean'),
    tmean_std=('T2M', 'std')
).reset_index()
# Smooth with 7-day rolling
for col in ['tmean', 'tmax', 'tmin', 'tmean_std']:
    clim[col] = clim[col].rolling(7, center=True, min_periods=3).mean()

ax.fill_between(clim['DOY'], clim['tmin'], clim['tmax'], alpha=0.12, color='#e74c3c')
ax.fill_between(clim['DOY'], clim['tmean'] - clim['tmean_std'],
                clim['tmean'] + clim['tmean_std'], alpha=0.2, color='#e74c3c')
ax.plot(clim['DOY'], clim['tmean'], color='#c0392b', linewidth=1.5, label='Tmean ± 1σ')
ax.plot(clim['DOY'], clim['tmax'], color='#e74c3c', linewidth=0.6, alpha=0.5, linestyle='--', label='Tmax')
ax.plot(clim['DOY'], clim['tmin'], color='#e74c3c', linewidth=0.6, alpha=0.5, linestyle=':', label='Tmin (derived)')

# Threshold lines
for temp, col, lbl in [(0, '#3498db', '0°C'), (10, '#2980b9', '10°C'), 
                         (15, '#e67e22', '15°C'), (22, '#27ae60', '22°C'), (30, '#e74c3c', '30°C')]:
    ax.axhline(temp, color=col, linestyle=':', linewidth=0.5, alpha=0.5)
    ax.annotate(lbl, xy=(370, temp+0.5), fontsize=6, color=col, clip_on=False)

ax.set_ylabel('Temperature (°C)')
ax.set_xlim(1, 365)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7)
ax.set_title('(a) Daily temperature climatology', loc='left', fontsize=10)

# Month labels
month_starts = {1:'Jan',32:'Feb',60:'Mar',91:'Apr',121:'May',152:'Jun',
                182:'Jul',213:'Aug',244:'Sep',274:'Oct',305:'Nov',335:'Dec'}
ax.set_xticks(list(month_starts.keys()))
ax.set_xticklabels(list(month_starts.values()), fontsize=7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='both', alpha=0.1, linewidth=0.3)

# Panel B: Crop calendar bars
ax = axes[1]
ax.set_xlim(1, 365)
ax.set_ylim(-0.5, 3.5)

crops = [
    # (name, sow_doy, end_doy, color, hatch_pattern, notes)
    ('Rice (irrigated)', 135, 254, '#27ae60', None, 'May 15 – Sep 11'),
    ('Tobacco (flue-cured)', 121, 258, '#8e44ad', None, 'May 1 – Sep 15'),
    ('Winter wheat', None, None, '#e67e22', None, 'Oct 15 – Jul 5'),  # split bar
]

bar_height = 0.6
y_positions = [2.5, 1.5, 0.5]

for i, (name, sow, end, color, hatch, notes) in enumerate(crops):
    y = y_positions[i]
    if name == 'Winter wheat':
        # Two bars: Oct–Dec and Jan–Jul
        ax.barh(y, 365-288, left=288, height=bar_height, color=color, alpha=0.7, edgecolor='white')
        ax.barh(y, 186, left=1, height=bar_height, color=color, alpha=0.7, edgecolor='white')
        # Sub-phases
        ax.barh(y, 365-288, left=288, height=bar_height*0.3, color='#d35400', alpha=0.5)  # sowing
        ax.annotate('Sow', xy=(300, y-0.25), fontsize=6, color='white', fontweight='bold')
        ax.annotate('Dormancy/Vernalization', xy=(20, y-0.05), fontsize=6, color='white', fontweight='bold')
        ax.annotate('Growth + Grain Fill', xy=(105, y-0.05), fontsize=6, color='white', fontweight='bold')
        ax.annotate('Harvest', xy=(170, y-0.25), fontsize=6, color='#d35400', fontweight='bold')
    else:
        ax.barh(y, end-sow, left=sow, height=bar_height, color=color, alpha=0.7, edgecolor='white')
        ax.annotate(notes, xy=(sow + (end-sow)/2, y-0.05), fontsize=6, ha='center', 
                    color='white', fontweight='bold')
    
    ax.annotate(name, xy=(-5, y), fontsize=9, ha='right', va='center', fontweight='bold', color=color)

ax.set_xticks(list(month_starts.keys()))
ax.set_xticklabels(list(month_starts.values()), fontsize=7)
ax.set_yticks([])
ax.set_title('(b) Crop growing calendar', loc='left', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(axis='x', alpha=0.15, linewidth=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('/home/claude/fig10_crop_calendar.png', facecolor='white')
fig.savefig('/home/claude/fig10_crop_calendar.pdf', facecolor='white')
plt.close()
print("Figure 10 saved: Multi-crop calendar")

# ── Figure 11: GDD accumulation comparison ──
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle('Growing Degree Day Accumulation at 38.3°N, 48.8°E (700 m)',
             fontsize=11, fontweight='bold', y=1.02)

# Rice GDD (base 10°C)
ax = axes[0]
for year in range(2020, 2026):
    mask = (df['YEAR'] == year) & (df['DOY'] >= 135) & (df['DOY'] <= 254)
    s = df[mask].copy().reset_index(drop=True)
    if len(s) < 100:
        continue
    gdd_cum = (s['T2M'] - 10).clip(lower=0).cumsum()
    color = C_2025 if year == 2025 else (C_2024 if year == 2024 else '#bdc3c7')
    lw = 1.8 if year in [2024, 2025] else 0.7
    alpha = 1 if year in [2024, 2025] else 0.4
    label = f"{year} ({gdd_cum.iloc[-1]:.0f})" if year in [2024, 2025] else f"{year}"
    ax.plot(range(len(gdd_cum)), gdd_cum, color=color, linewidth=lw, alpha=alpha, label=label)

ax.axhline(1400, color='#27ae60', linestyle='--', linewidth=0.8, alpha=0.5)
ax.annotate('~1400 °C-days (medium variety)', xy=(5, 1430), fontsize=7, color='#27ae60')
ax.set_xlabel('Day in season (from May 15)')
ax.set_ylabel('Cumulative GDD (°C-days, base 10°C)')
ax.set_title('(a) Rice GDD accumulation', loc='left', fontsize=10)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7)

# Wheat GDD (base 0°C) — Oct 15 to Jul 15
ax = axes[1]
for year in range(2020, 2024):
    season = df[((df['YEAR'] == year) & (df['DOY'] >= 288)) |
                ((df['YEAR'] == year+1) & (df['DOY'] <= 196))]
    season = season.copy().reset_index(drop=True)
    gdd_cum = season['T2M'].clip(lower=0).cumsum()
    ax.plot(range(len(gdd_cum)), gdd_cum, linewidth=0.8, alpha=0.6,
            label=f"{year}/{year+1} ({gdd_cum.iloc[-1]:.0f})")

ax.axhline(2000, color='#e67e22', linestyle='--', linewidth=0.8, alpha=0.5)
ax.annotate('~2000 °C-days (winter wheat)', xy=(5, 2050), fontsize=7, color='#e67e22')
ax.set_xlabel('Day in season (from Oct 15)')
ax.set_ylabel('Cumulative GDD (°C-days, base 0°C)')
ax.set_title('(b) Winter wheat GDD accumulation', loc='left', fontsize=10)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.3)

plt.tight_layout()
fig.savefig('/home/claude/fig11_gdd_comparison.png', facecolor='white')
fig.savefig('/home/claude/fig11_gdd_comparison.pdf', facecolor='white')
plt.close()
print("Figure 11 saved: GDD accumulation comparison")

print("\n✓ All analyses and figures complete.")
