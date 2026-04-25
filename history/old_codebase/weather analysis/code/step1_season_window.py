"""
STEP 1: SEASON WINDOW ANALYSIS
================================
Thesis Action Item: Shift growing season from Apr 1 (DOY 91) to May 15 (DOY 135)
This script provides the evidence and justification for your Methods chapter.

Site: 38.298°N, 48.847°E, ~700m elevation, Gilan/Ardabil border, Iran
Crop: Irrigated rice, 120-day season, Kc = 1.15
"""

import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# PART A: LOAD AND PREPARE DATA
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

# Imputation: same-DOY climatological mean (better than column mean)
for col in ['T2M', 'T2M_MAX', 'PRECTOTCORR', 'GWETROOT', 'GWETTOP']:
    df[col] = df[col].fillna(df.groupby('DOY')[col].transform('mean'))
df['ALLSKY_SFC_SW_DWN'] = df['ALLSKY_SFC_SW_DWN'].fillna(
    df.groupby('MONTH')['ALLSKY_SFC_SW_DWN'].transform('mean')
)

# Corrected ET0 (Ra in mm/day)
lat_rad = 38.298 * math.pi / 180

def compute_Ra_mm(doy):
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.4093 * math.sin(2 * math.pi * doy / 365 - 1.39)
    cos_arg = max(-1, min(1, -math.tan(lat_rad) * math.tan(delta)))
    ws = math.acos(cos_arg)
    Ra_MJ = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(delta) +
        math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )
    return Ra_MJ / 2.45  # ← CRITICAL: convert to mm/day

df['Ra_mm'] = df['DOY'].apply(lambda d: compute_Ra_mm(int(d)))
df['T2M_MIN_derived'] = 2 * df['T2M'] - df['T2M_MAX']
df['Trange'] = (df['T2M_MAX'] - df['T2M_MIN_derived']).clip(lower=0)
df['ET0'] = 0.0023 * (df['T2M'] + 17.8) * np.sqrt(df['Trange']) * df['Ra_mm']

print("=" * 70)
print("STEP 1A: TEMPERATURE THRESHOLD ANALYSIS")
print("        (Evidence for shifting season start date)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# PART B: TEMPERATURE THRESHOLD ANALYSIS
# Why April 1 is too early — evidence for your Methods chapter
# ═══════════════════════════════════════════════════════════════

print("""
Rice transplanting requires:
  - Water temperature ≥ 15°C (absolute minimum, risk of cold sterility)
  - Air temperature ≥ 20°C (optimal transplanting range: 20–30°C)
  - Stable warm conditions (no cold snaps below 12°C after transplanting)

At your 700m elevation, air temperature is a reasonable proxy for paddy
water temperature since shallow standing water equilibrates quickly.
""")

# Year-by-year threshold crossing dates
print(f"{'Year':>6}  {'First Tmean≥15°C':>18}  {'First Tmean≥20°C':>18}  {'Apr mean T':>11}")
print("-" * 60)
for year in range(2020, 2026):
    yr = df[df['YEAR'] == year].copy()
    spring = yr[yr['DOY'] >= 60].copy()
    spring['T_roll7'] = spring['T2M'].rolling(7, min_periods=5).mean()
    
    # April mean temperature
    apr_mean = yr[(yr['MONTH'] == 4)]['T2M'].mean()
    
    # Find threshold crossings
    t15 = spring[spring['T_roll7'] >= 15]
    t20 = spring[spring['T_roll7'] >= 20]
    
    d15 = f"DOY {int(t15.iloc[0]['DOY']):>3} ({t15.iloc[0]['DATE'].strftime('%b %d')})" if len(t15) > 0 else "Never"
    d20 = f"DOY {int(t20.iloc[0]['DOY']):>3} ({t20.iloc[0]['DATE'].strftime('%b %d')})" if len(t20) > 0 else "Never"
    
    print(f"{year:>6}  {d15:>18}  {d20:>18}  {apr_mean:>9.1f}°C")

# Count days below 12°C in different windows
print(f"\nDays with Tmean < 12°C (cold stress risk for transplanted rice):")
print(f"{'Year':>6}  {'Apr 1-30':>10}  {'May 1-15':>10}  {'May 15-31':>10}  {'Jun 1-30':>10}")
print("-" * 55)
for year in range(2020, 2026):
    yr = df[df['YEAR'] == year]
    apr = ((yr['MONTH'] == 4) & (yr['T2M'] < 12)).sum()
    may_early = ((yr['MONTH'] == 5) & (yr['DOY'] <= 135) & (yr['T2M'] < 12)).sum()
    may_late = ((yr['MONTH'] == 5) & (yr['DOY'] > 135) & (yr['T2M'] < 12)).sum()
    jun = ((yr['MONTH'] == 6) & (yr['T2M'] < 12)).sum()
    print(f"{year:>6}  {apr:>8} d  {may_early:>8} d  {may_late:>8} d  {jun:>8} d")

print("""
CONCLUSION FOR METHODS CHAPTER:
  April has an average temperature of ~13°C at this site, with frequent
  days below 12°C (cold stress threshold). The 7-day rolling mean does not
  consistently exceed 15°C until late April–early May, and does not reach
  20°C (optimal transplanting) until mid-May to early June.
  
  A May 15 start date (DOY 135) ensures:
    ✓ Mean temperature consistently above 15°C
    ✓ Minimal cold stress risk after transplanting  
    ✓ Alignment with actual Gilan rice transplanting practice (late May–June)
    ✓ The 120-day season ends Sep 11 (DOY 254), before autumn cold onset
""")

# ═══════════════════════════════════════════════════════════════
# PART C: COMPARE ALL CANDIDATE SEASON WINDOWS
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 1B: SEASON WINDOW COMPARISON")
print("=" * 70)

windows = {
    'Apr 1 – Jul 29':   (91, 210),   # Your current window
    'May 1 – Aug 28':   (121, 240),  # Alternative 1
    'May 15 – Sep 11':  (135, 254),  # Recommended
    'Jun 1 – Sep 28':   (152, 271),  # Late transplanting
}

for wname, (doy_s, doy_e) in windows.items():
    print(f"\n── {wname} (DOY {doy_s}–{doy_e}) ──")
    print(f"{'Year':>6} {'Precip':>8} {'ET0':>7} {'ETc':>7} {'P-ETc':>8} {'MaxDayP':>8} {'Tmean':>6} {'RainDays':>9}")
    
    yearly = []
    for year in range(2020, 2026):
        mask = (df['YEAR'] == year) & (df['DOY'] >= doy_s) & (df['DOY'] <= doy_e)
        s = df[mask]
        if len(s) < 100:
            continue
        p = s['PRECTOTCORR'].sum()
        et0 = s['ET0'].sum()
        etc = et0 * 1.15
        max_p = s['PRECTOTCORR'].max()
        tmean = s['T2M'].mean()
        rain_d = (s['PRECTOTCORR'] > 1).sum()
        deficit = p - etc
        yearly.append({
            'year': year, 'precip': p, 'et0': et0, 'etc': etc,
            'deficit': deficit, 'max_p': max_p, 'tmean': tmean, 'rain_d': rain_d
        })
        print(f"{year:>6} {p:>8.1f} {et0:>7.1f} {etc:>7.1f} {deficit:>8.1f} {max_p:>8.1f} {tmean:>6.1f} {rain_d:>7}d")
    
    precips = [y['precip'] for y in yearly]
    years = [y['year'] for y in yearly]
    driest_idx = np.argmin(precips)
    wettest_idx = np.argmax(precips)
    print(f"  → Driest:  {years[driest_idx]} ({precips[driest_idx]:.1f} mm)")
    print(f"  → Wettest: {years[wettest_idx]} ({precips[wettest_idx]:.1f} mm)")
    
    # Coefficient of variation
    cv = np.std(precips) / np.mean(precips) * 100
    print(f"  → Mean: {np.mean(precips):.1f} mm, CV: {cv:.1f}%")

# ═══════════════════════════════════════════════════════════════
# PART D: RECOMMENDED WINDOW — DETAILED YEAR SELECTION
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 1C: YEAR SELECTION FOR MAY 15 – SEP 11 WINDOW")
print("=" * 70)

DOY_START = 135  # May 15
DOY_END = 254    # Sep 11

records = []
for year in range(2020, 2026):
    mask = (df['YEAR'] == year) & (df['DOY'] >= DOY_START) & (df['DOY'] <= DOY_END)
    s = df[mask].copy()
    if len(s) < 100:
        continue
    
    p_total = s['PRECTOTCORR'].sum()
    et0_total = s['ET0'].sum()
    etc_total = et0_total * 1.15
    max_daily_p = s['PRECTOTCORR'].max()
    max_p_date = s.loc[s['PRECTOTCORR'].idxmax(), 'DATE']
    tmean = s['T2M'].mean()
    rain_days = (s['PRECTOTCORR'] > 1).sum()
    dry_spells = 0  # longest consecutive run of <1mm days
    current_run = 0
    for p in s['PRECTOTCORR']:
        if p < 1:
            current_run += 1
            dry_spells = max(dry_spells, current_run)
        else:
            current_run = 0
    
    gwetroot_mean = s['GWETROOT'].mean()
    gwettop_mean = s['GWETTOP'].mean()
    
    records.append({
        'year': year, 'precip': p_total, 'et0': et0_total, 'etc': etc_total,
        'deficit': p_total - etc_total, 'max_p': max_daily_p,
        'max_p_date': max_p_date, 'tmean': tmean, 'rain_days': rain_days,
        'max_dry_spell': dry_spells, 'gwetroot': gwetroot_mean, 'gwettop': gwettop_mean
    })

print(f"\nDetailed season statistics (May 15 – Sep 11, 120 days):\n")
print(f"{'Year':>6} {'Precip':>8} {'ETc':>7} {'Deficit':>8} {'MaxP':>7} {'RainD':>6} {'DrySpell':>9} {'Tmean':>6} {'GWROOT':>7}")
for r in records:
    print(f"{r['year']:>6} {r['precip']:>8.1f} {r['etc']:>7.1f} {r['deficit']:>8.1f} "
          f"{r['max_p']:>7.1f} {r['rain_days']:>5}d {r['max_dry_spell']:>7}d {r['tmean']:>6.1f} {r['gwetroot']:>7.3f}")

precips = [r['precip'] for r in records]
sorted_years = sorted(records, key=lambda x: x['precip'])

print(f"\nRanking by total precipitation (driest → wettest):")
for i, r in enumerate(sorted_years):
    marker = " ← DRY SCENARIO" if i == 0 else (" ← WET SCENARIO" if i == len(sorted_years)-1 else "")
    print(f"  {i+1}. {r['year']}: {r['precip']:.1f} mm{marker}")

dry = sorted_years[0]
wet = sorted_years[-1]

print(f"""
═══════════════════════════════════════════════════════════════
RECOMMENDED SCENARIO SELECTION (May 15 – Sep 11 window):

  DRY SCENARIO: {dry['year']}
    Seasonal precipitation:    {dry['precip']:.1f} mm
    Seasonal ETc (Kc=1.15):   {dry['etc']:.1f} mm
    Water deficit (P - ETc):   {dry['deficit']:.1f} mm
    Max daily rainfall:        {dry['max_p']:.1f} mm
    Rainy days (>1mm):         {dry['rain_days']}
    Longest dry spell:         {dry['max_dry_spell']} days
    Mean temperature:          {dry['tmean']:.1f}°C
    Mean root zone wetness:    {dry['gwetroot']:.3f}

  WET SCENARIO: {wet['year']}
    Seasonal precipitation:    {wet['precip']:.1f} mm
    Seasonal ETc (Kc=1.15):   {wet['etc']:.1f} mm  
    Water deficit (P - ETc):   {wet['deficit']:.1f} mm
    Max daily rainfall:        {wet['max_p']:.1f} mm
    Rainy days (>1mm):         {wet['rain_days']}
    Longest dry spell:         {wet['max_dry_spell']} days
    Mean temperature:          {wet['tmean']:.1f}°C
    Mean root zone wetness:    {wet['gwetroot']:.3f}

  KEY ADVANTAGE: Neither scenario depends on the questionable
  371.8 mm extreme event (which falls on April 7, DOY 97,
  outside this season window).
═══════════════════════════════════════════════════════════════
""")

# ═══════════════════════════════════════════════════════════════
# PART E: GENERATE UPDATED FIGURES
# ═══════════════════════════════════════════════════════════════

# Colors
C_DRY = '#c0392b'
C_WET = '#2980b9'
C_GRAY = '#95a5a6'
YEAR_COLORS = {2020: '#95a5a6', 2021: '#e67e22', 2022: '#8e44ad', 
               2023: '#34495e', 2024: '#f39c12', 2025: '#1abc9c'}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
    'axes.titlesize': 11, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
})

def get_season(year, doy_s=DOY_START, doy_e=DOY_END):
    mask = (df['YEAR'] == year) & (df['DOY'] >= doy_s) & (df['DOY'] <= doy_e)
    s = df[mask].copy().reset_index(drop=True)
    s['day'] = range(len(s))
    s['cum_precip'] = s['PRECTOTCORR'].cumsum()
    s['cum_et0'] = s['ET0'].cumsum()
    s['cum_etc'] = s['cum_et0'] * 1.15
    return s

dry_year = dry['year']
wet_year = wet['year']

s_dry = get_season(dry_year)
s_wet = get_season(wet_year)

# ── Figure 6: Updated scenario comparison ──
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
fig.suptitle(f'Updated Climate Scenarios: Dry ({dry_year}) vs Wet ({wet_year})\n'
             f'Rice Season May 15 – Sep 11 (DOY 135–254) · 38.3°N, 48.8°E, 700 m a.s.l.',
             fontsize=11, fontweight='bold', y=0.98)

# (a) Daily precipitation — mirrored
ax = axes[0, 0]
ax.bar(s_dry['day'], s_dry['PRECTOTCORR'], width=1, color=C_DRY, alpha=0.7, label=f'{dry_year} (dry)')
ax.bar(s_wet['day'], -s_wet['PRECTOTCORR'], width=1, color=C_WET, alpha=0.7, label=f'{wet_year} (wet)')
ax.set_ylabel('Precipitation (mm/day)')
ax.set_xlabel('Day in season')
ax.set_title('(a) Daily precipitation', loc='left', fontsize=10)
ax.legend(loc='lower left', framealpha=0.9)
ax.axhline(0, color='k', linewidth=0.3)

# (b) Cumulative P vs ETc
ax = axes[0, 1]
ax.plot(s_dry['day'], s_dry['cum_precip'], color=C_DRY, label=f'P ({dry_year})')
ax.plot(s_dry['day'], s_dry['cum_etc'], color=C_DRY, linestyle='--', label=f'ETc ({dry_year})')
ax.plot(s_wet['day'], s_wet['cum_precip'], color=C_WET, label=f'P ({wet_year})')
ax.plot(s_wet['day'], s_wet['cum_etc'], color=C_WET, linestyle='--', label=f'ETc ({wet_year})')
ax.fill_between(s_dry['day'], s_dry['cum_precip'], s_dry['cum_etc'], alpha=0.1, color=C_DRY)
ax.fill_between(s_wet['day'], s_wet['cum_precip'], s_wet['cum_etc'], alpha=0.1, color=C_WET)
ax.set_ylabel('Cumulative (mm)')
ax.set_xlabel('Day in season')
ax.set_title('(b) Cumulative precipitation vs crop ET', loc='left', fontsize=10)
ax.legend(loc='upper left', framealpha=0.9, ncol=2, fontsize=7)

# (c) Temperature
ax = axes[1, 0]
ax.plot(s_dry['day'], s_dry['T2M'], color=C_DRY, label=f'Tmean {dry_year}', alpha=0.8)
ax.plot(s_wet['day'], s_wet['T2M'], color=C_WET, label=f'Tmean {wet_year}', alpha=0.8)
ax.fill_between(s_dry['day'], s_dry['T2M_MIN_derived'], s_dry['T2M_MAX'], alpha=0.08, color=C_DRY)
ax.fill_between(s_wet['day'], s_wet['T2M_MIN_derived'], s_wet['T2M_MAX'], alpha=0.08, color=C_WET)
ax.axhline(15, color=C_GRAY, linestyle=':', linewidth=0.8, label='15°C min.')
ax.set_ylabel('Temperature (°C)')
ax.set_xlabel('Day in season')
ax.set_title('(c) Temperature regime', loc='left', fontsize=10)
ax.legend(loc='lower right', framealpha=0.9)

# (d) Soil wetness
ax = axes[1, 1]
ax.plot(s_dry['day'], s_dry['GWETROOT'], color=C_DRY, label=f'Root zone {dry_year}')
ax.plot(s_wet['day'], s_wet['GWETROOT'], color=C_WET, label=f'Root zone {wet_year}')
ax.plot(s_dry['day'], s_dry['GWETTOP'], color=C_DRY, linestyle=':', alpha=0.6, label=f'Surface {dry_year}')
ax.plot(s_wet['day'], s_wet['GWETTOP'], color=C_WET, linestyle=':', alpha=0.6, label=f'Surface {wet_year}')
ax.set_ylabel('Soil wetness index (0–1)')
ax.set_xlabel('Day in season')
ax.set_title('(d) MERRA-2 soil wetness', loc='left', fontsize=10)
ax.legend(loc='upper right', framealpha=0.9, ncol=2, fontsize=7)
ax.set_ylim(0.2, 1.0)

for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('/home/claude/fig6_updated_scenarios.png', facecolor='white')
fig.savefig('/home/claude/fig6_updated_scenarios.pdf', facecolor='white')
plt.close()
print("Figure 6 saved: Updated scenario comparison")

# ── Figure 7: All-years for new window ──
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle(f'Seasonal Precipitation (2020–2025)\n'
             f'May 15 – Sep 11 (DOY 135–254)',
             fontsize=11, fontweight='bold', y=1.02)

ax = axes[0]
for year in range(2020, 2026):
    s = get_season(year)
    is_selected = year in [dry_year, wet_year]
    lw = 2 if is_selected else 0.8
    alpha = 1 if is_selected else 0.4
    color = C_DRY if year == dry_year else (C_WET if year == wet_year else YEAR_COLORS.get(year, C_GRAY))
    label_suffix = ' (DRY)' if year == dry_year else (' (WET)' if year == wet_year else '')
    ax.plot(s['day'], s['cum_precip'], color=color, linewidth=lw, alpha=alpha,
            label=f"{year}{label_suffix} ({s['cum_precip'].iloc[-1]:.0f} mm)")
ax.set_ylabel('Cumulative precipitation (mm)')
ax.set_xlabel('Day in season')
ax.set_title('(a) Cumulative precipitation', loc='left', fontsize=10)
ax.legend(loc='upper left', framealpha=0.9, fontsize=7)

ax = axes[1]
years_list = []
precips_list = []
deficits_list = []
colors_list = []
for year in range(2020, 2026):
    s = get_season(year)
    p = s['PRECTOTCORR'].sum()
    etc = s['ET0'].sum() * 1.15
    years_list.append(year)
    precips_list.append(p)
    deficits_list.append(p - etc)
    colors_list.append(C_DRY if year == dry_year else (C_WET if year == wet_year else C_GRAY))

x = np.arange(len(years_list))
ax.bar(x - 0.17, precips_list, 0.34, color=colors_list, alpha=0.8, label='Precipitation')
ax.bar(x + 0.17, deficits_list, 0.34, color=[c if c != C_GRAY else '#bdc3c7' for c in colors_list],
       alpha=0.6, label='P − ETc')
ax.set_xticks(x)
ax.set_xticklabels(years_list)
ax.set_ylabel('mm')
ax.axhline(0, color='k', linewidth=0.3)
ax.set_title('(b) Seasonal water balance', loc='left', fontsize=10)
ax.legend(framealpha=0.9, fontsize=7)

for i, year in enumerate(years_list):
    if year == dry_year:
        ax.annotate('DRY', xy=(i-0.17, precips_list[i]+8), ha='center', fontsize=7, color=C_DRY, fontweight='bold')
    if year == wet_year:
        ax.annotate('WET', xy=(i-0.17, precips_list[i]+8), ha='center', fontsize=7, color=C_WET, fontweight='bold')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.3)

plt.tight_layout()
fig.savefig('/home/claude/fig7_allyears_new_window.png', facecolor='white')
fig.savefig('/home/claude/fig7_allyears_new_window.pdf', facecolor='white')
plt.close()
print("Figure 7 saved: All-years comparison for new window")

# ── Figure 8: Season window justification (temperature thresholds) ──
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
fig.suptitle('Temperature Threshold Analysis for Season Start Date\n'
             '7-Day Rolling Mean Temperature (2020–2025)',
             fontsize=11, fontweight='bold', y=1.02)

for year in range(2020, 2026):
    yr = df[(df['YEAR'] == year) & (df['DOY'] >= 60) & (df['DOY'] <= 270)].copy()
    yr['T_roll7'] = yr['T2M'].rolling(7, min_periods=5).mean()
    color = YEAR_COLORS.get(year, C_GRAY)
    ax.plot(yr['DOY'], yr['T_roll7'], color=color, alpha=0.5, linewidth=0.8, label=str(year))

ax.axhline(15, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
ax.annotate('15°C — transplanting minimum', xy=(65, 15.5), fontsize=8, color='#e74c3c')
ax.axhline(20, color='#e67e22', linestyle='--', linewidth=1, alpha=0.7)
ax.annotate('20°C — optimal transplanting', xy=(65, 20.5), fontsize=8, color='#e67e22')

# Shade current vs proposed windows
ax.axvspan(91, 91, color=C_DRY, alpha=0.8, linewidth=2)
ax.axvspan(135, 135, color=C_WET, alpha=0.8, linewidth=2)
ax.annotate('Current start\n(Apr 1, DOY 91)', xy=(91, 5), fontsize=7, color=C_DRY,
            ha='center', fontweight='bold')
ax.annotate('Proposed start\n(May 15, DOY 135)', xy=(135, 5), fontsize=7, color=C_WET,
            ha='center', fontweight='bold')

ax.axvline(91, color=C_DRY, linestyle='-', linewidth=1.5, alpha=0.5)
ax.axvline(135, color=C_WET, linestyle='-', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Day of Year')
ax.set_ylabel('7-Day Rolling Mean Temperature (°C)')
ax.legend(loc='lower right', framealpha=0.9, ncol=3, fontsize=7)
ax.set_xlim(60, 270)
ax.set_ylim(0, 30)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, linewidth=0.3)

# Add month labels on x-axis
import matplotlib.ticker as mticker
month_starts = {60: 'Mar', 91: 'Apr', 121: 'May', 152: 'Jun', 182: 'Jul', 213: 'Aug', 244: 'Sep'}
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(list(month_starts.keys()))
ax2.set_xticklabels(list(month_starts.values()), fontsize=8)
ax2.tick_params(length=0)

plt.tight_layout()
fig.savefig('/home/claude/fig8_temp_thresholds.png', facecolor='white')
fig.savefig('/home/claude/fig8_temp_thresholds.pdf', facecolor='white')
plt.close()
print("Figure 8 saved: Temperature threshold justification")

print("\n✓ All outputs generated. See summary below.\n")
