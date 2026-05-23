# =============================================================================
# plot_wind.py
# 25-year daily average wind speed at 2m (April–October)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_CSV = Path('results/preprocessing/climate_apr_oct_cleaned.csv')
OUTPUT = Path('results/preprocessing/fig_wind.png')

df = pd.read_csv(DATA_CSV)
clim = df.groupby('DOY')['WS2M'].mean().reset_index()
clim['DATE'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(clim['DOY'] - 1, unit='D')

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(clim['DATE'], clim['WS2M'], color='navy', linewidth=1.2)
ax.fill_between(clim['DATE'], 0, clim['WS2M'], alpha=0.15, color='navy')
ax.set_ylabel('Wind Speed (m/s)')
ax.set_title('25-Year Daily Average Wind Speed at 2m — Gilan\nApril–October, 2000–2025')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT}")
