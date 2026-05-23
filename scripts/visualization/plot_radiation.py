# =============================================================================
# plot_radiation.py
# 25-year daily average solar radiation (April–October)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_CSV = Path('results/preprocessing/climate_apr_oct_cleaned.csv')
OUTPUT = Path('results/preprocessing/fig_radiation.png')

df = pd.read_csv(DATA_CSV)
clim = df.groupby('DOY')['ALLSKY_SFC_SW_DWN'].mean().reset_index()
clim['DATE'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(clim['DOY'] - 1, unit='D')

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(clim['DATE'], clim['ALLSKY_SFC_SW_DWN'], color='orange', linewidth=1.2)
ax.fill_between(clim['DATE'], 0, clim['ALLSKY_SFC_SW_DWN'], alpha=0.2, color='orange')
ax.set_ylabel('Solar Radiation (MJ/m²/day)')
ax.set_title('25-Year Daily Average Solar Radiation — Gilan\nApril–October, 2000–2025')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT}")
