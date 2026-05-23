# =============================================================================
# plot_soil_moisture.py
# 25-year daily average: GWETROOT and GWETTOP (April–October)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_CSV = Path('results/preprocessing/climate_apr_oct_cleaned.csv')
OUTPUT = Path('results/preprocessing/fig_soil_moisture.png')

df = pd.read_csv(DATA_CSV)
clim = df.groupby('DOY').agg({
    'GWETROOT': 'mean', 'GWETTOP': 'mean'
}).reset_index()
clim['DATE'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(clim['DOY'] - 1, unit='D')

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(clim['DATE'], clim['GWETROOT'], color='saddlebrown', linewidth=1.5,
        label='GWETROOT (root zone, ~100 cm)')
ax.plot(clim['DATE'], clim['GWETTOP'], color='sandybrown', linewidth=1.2,
        linestyle='--', label='GWETTOP (surface, ~5 cm)')
ax.fill_between(clim['DATE'], clim['GWETTOP'], clim['GWETROOT'],
                alpha=0.1, color='brown')

ax.set_ylabel('Soil Wetness Fraction (0–1)')
ax.set_title('25-Year Daily Average Soil Moisture — Gilan\n'
             'NASA MERRA-2 Reanalysis, April–October, 2000–2025')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_ylim(0.2, 0.9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT}")
