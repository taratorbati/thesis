# =============================================================================
# plot_pressure.py
# 25-year daily average surface pressure (April–October)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_CSV = Path('results/preprocessing/climate_apr_oct_cleaned.csv')
OUTPUT = Path('results/preprocessing/fig_pressure.png')

df = pd.read_csv(DATA_CSV)
clim = df.groupby('DOY')['PS'].mean().reset_index()
clim['DATE'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(clim['DOY'] - 1, unit='D')

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(clim['DATE'], clim['PS'], color='purple', linewidth=1.2)
ax.set_ylabel('Surface Pressure (kPa)')
ax.set_title('25-Year Daily Average Surface Pressure — Gilan\nApril–October, 2000–2025')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT}")
