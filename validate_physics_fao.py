import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from soil_data import RICE as crop_params
from climate_data import extract_scenario, load_cleaned_data
from abm import CropSoilABM

OUTPUT_DIR = Path('results/validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Setup Fake "Flat" Topography for Unit Test ──────────────────────────────
# We don't need the real Gilan map for a pure physics test. We just need agents.
N = 10
gamma_flat = np.zeros(N)  # perfectly flat
sends_to = {i: [] for i in range(N)}  # no water flowing anywhere
Nr = {i: 0 for i in range(N)}
elevation = np.zeros(N)

# ── Configuration ─────────────────────────────────────────────────────────────
df = load_cleaned_data()
n_days = crop_params['season_days']
root_depth = crop_params['theta5']
FC = crop_params['theta6'] * root_depth
WP = crop_params['theta2'] * root_depth

# We will just test the Dry Year (2022) to keep it simple
climate = extract_scenario(df, 2022, crop_params)

# ── 1. Run Your ABM (The Code to be Tested) ──────────────────────────────────
model = CropSoilABM(
    gamma_flat=gamma_flat,
    sends_to=sends_to,
    Nr=Nr,
    theta=crop_params,
    N=N,
    runoff_mode='simple',  # Gravity turned OFF for the unit test
    elevation=elevation,
)
state = model.reset()

abm_soil_moisture = []

for day in range(n_days):
    daily_climate = {
        'rainfall':  climate['rainfall'][day],
        'temp_mean': climate['temp_mean'][day],
        'temp_max':  climate['temp_max'][day],
        'radiation': climate['radiation'][day],
        'ET':        climate['ET'][day],
    }
    u = np.zeros(N)  # No irrigation
    state = model.step(u, daily_climate)
    # x1 is your root zone moisture
    abm_soil_moisture.append(state['x1'].mean())

# ── 2. Run Theoretical FAO-56 (The Benchmark) ────────────────────────────────
fao56_soil_moisture = []
# Assume initial soil moisture starts exactly where your ABM starts it
current_moisture = FC * 0.8  # Or whatever your model.reset() defaults to

for day in range(n_days):
    rain = climate['rainfall'][day]
    et = climate['ET'][day]

    # FAO-56 simple daily bucket math
    current_moisture = current_moisture + rain - et

    # Physics constraints: Cannot hold more than Saturation/FC, cannot drop below Wilting Point
    if current_moisture > FC:
        current_moisture = FC
    if current_moisture < WP:
        current_moisture = WP

    fao56_soil_moisture.append(current_moisture)

# ── Plot the Results ──────────────────────────────────────────────────────────
days = np.arange(n_days)
plt.figure(figsize=(10, 5))

# If your math is perfect, the blue dashed line will sit exactly on top of the red line
plt.plot(days, fao56_soil_moisture, 'r-', linewidth=3,
         label='Theoretical Standard (FAO-56)', alpha=0.7)
plt.plot(days, abm_soil_moisture, 'b--',
         linewidth=2, label='Your ABM Engine (x1)')

plt.axhline(FC, color='green', linestyle=':', label='Field Capacity')
plt.axhline(WP, color='red', linestyle=':', label='Wilting Point')

plt.title('Physics Unit Test: ABM Core vs Theoretical Standard (Un-irrigated)')
plt.ylabel('Soil Moisture (mm)')
plt.xlabel('Day of Season')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'physics_unit_test.png', dpi=150)
print(f"Saved: {OUTPUT_DIR / 'physics_unit_test.png'}")
