# =============================================================================
# plot_temperature.py
# 25-year daily average: Tmin, Tmean, Tmax (April–October)
# Reads from cleaned CSV produced by preprocess.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_CSV = Path('results/preprocessing/climate_apr_oct_cleaned.csv')
OUTPUT_FULL = Path('results/preprocessing/fig_temperature.png')
OUTPUT_MAY = Path('results/preprocessing/fig_temperature_may.png')

df = pd.read_csv(DATA_CSV)
df['DATE'] = pd.to_datetime(df['DATE'])

# 25-year daily average grouped by DOY
clim = df.groupby('DOY').agg({
    'T2M': 'mean', 'T2M_MAX': 'mean', 'T2M_MIN': 'mean'
}).reset_index()
clim['DATE'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(clim['DOY'] - 1, unit='D')

# --- Plot 1: April–October ---
fig, ax = plt.subplots(figsize=(14, 6))

ax.fill_between(clim['DATE'], clim['T2M_MIN'], clim['T2M_MAX'],
                alpha=0.2, color='red', label='Tmin – Tmax range')
ax.plot(clim['DATE'], clim['T2M'], 'r-', linewidth=1.5, label='Tmean')
ax.plot(clim['DATE'], clim['T2M_MAX'], 'r--', linewidth=0.7, alpha=0.5, label='Tmax')
ax.plot(clim['DATE'], clim['T2M_MIN'], 'b--', linewidth=0.7, alpha=0.5, label='Tmin')

ax.set_ylabel('Temperature (°C)')
ax.set_title('25-Year Daily Average Temperature — Gilan (38.3°N, 48.8°E)\nApril–October, 2000–2025')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FULL, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_FULL}")

# --- Plot 2: May only ---
may = clim[clim['DATE'].dt.month == 5].copy()

fig, ax = plt.subplots(figsize=(14, 6))

ax.fill_between(may['DATE'], may['T2M_MIN'], may['T2M_MAX'],
                alpha=0.2, color='red', label='Tmin – Tmax range')
ax.plot(may['DATE'], may['T2M'], 'r-', linewidth=1.5, label='Tmean')
ax.plot(may['DATE'], may['T2M_MAX'], 'r--', linewidth=0.7, alpha=0.5, label='Tmax')
ax.plot(may['DATE'], may['T2M_MIN'], 'b--', linewidth=0.7, alpha=0.5, label='Tmin')

ax.axhline(18, color='orange', linestyle=':', linewidth=1, label='Transplanting threshold (18°C)')
ax.axhline(12, color='blue', linestyle=':', linewidth=1, label='Tmin threshold (12°C)')

ax.set_ylabel('Temperature (°C)')
ax.set_title('25-Year Daily Average Temperature — May\nGilan (38.3°N, 48.8°E), 2000–2025')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_MAY, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_MAY}")