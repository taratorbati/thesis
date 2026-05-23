# =============================================================================
# plot_rainfall.py
# 25-year daily average rainfall (April–October)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_CSV = Path('results/preprocessing/climate_apr_oct_cleaned.csv')
OUTPUT_FULL = Path('results/preprocessing/fig_rainfall.png')
OUTPUT_MAY = Path('results/preprocessing/fig_rainfall_may.png')

df = pd.read_csv(DATA_CSV)
df['DATE'] = pd.to_datetime(df['DATE'])
clim = df.groupby('DOY')['PRECTOTCORR'].mean().reset_index()
clim['DATE'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(clim['DOY'] - 1, unit='D')

# --- Plot 1: April–October ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(clim['DATE'], clim['PRECTOTCORR'], width=1.2, color='steelblue', alpha=0.7)
ax.set_ylabel('Rainfall (mm/day)')
ax.set_title('25-Year Daily Average Rainfall — Gilan\nApril–October, 2000–2025')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FULL, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_FULL}")

# --- Plot 2: May only ---
may = clim[clim['DATE'].dt.month == 5].copy()
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(may['DATE'], may['PRECTOTCORR'], width=1.2, color='steelblue', alpha=0.7)
ax.set_ylabel('Rainfall (mm/day)')
ax.set_title('25-Year Daily Average Rainfall — May\nGilan (38.3°N, 48.8°E), 2000–2025')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_MAY, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_MAY}")