# =============================================================================
# plot_et0.py
# ET0 comparison: Hargreaves vs Penman-Monteith
# Left: 25-year daily average seasonal pattern
# Right: scatter plot with 1:1 line and correlation
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_CSV = Path('results/preprocessing/climate_apr_oct_cleaned.csv')
OUTPUT = Path('results/preprocessing/fig_et0.png')

df = pd.read_csv(DATA_CSV)
df['DATE'] = pd.to_datetime(df['DATE'])

clim = df.groupby('DOY').agg({
    'ET0_hargreaves': 'mean',
    'ET0_penman_monteith': 'mean',
}).reset_index()
clim['DATE'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(clim['DOY'] - 1, unit='D')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Reference ET₀: Hargreaves vs Penman-Monteith\n'
             'April–October, 2000–2025', fontsize=13)

# Left: seasonal pattern
ax = axes[0]
ax.plot(clim['DATE'], clim['ET0_penman_monteith'], color='blue',
        linewidth=1.5, label='Penman-Monteith')
ax.plot(clim['DATE'], clim['ET0_hargreaves'], color='red',
        linewidth=1.2, linestyle='--', label='Hargreaves')
ax.set_ylabel('ET₀ (mm/day)')
ax.set_title('Daily Average Seasonal Pattern')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.grid(axis='y', alpha=0.3)

# Right: scatter
ax = axes[1]
mask = df[['ET0_hargreaves', 'ET0_penman_monteith']].notna().all(axis=1)
x = df.loc[mask, 'ET0_penman_monteith']
y = df.loc[mask, 'ET0_hargreaves']

ax.scatter(x, y, s=2, alpha=0.08, color='steelblue')

lims = [0, 14]
ax.plot(lims, lims, 'k--', linewidth=0.8, label='1:1 line')

z = np.polyfit(x, y, 1)
ax.plot(lims, [z[0] * v + z[1] for v in lims], 'r-', linewidth=1,
        label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')

corr = x.corr(y)
ratio = y.mean() / x.mean()
ax.text(0.05, 0.90, f'r = {corr:.3f}\nHarg/PM = {ratio:.2f}\nn = {len(x)}',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Penman-Monteith ET₀ (mm/day)')
ax.set_ylabel('Hargreaves ET₀ (mm/day)')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.set_title('Daily Scatter — All Data Points')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT}")
