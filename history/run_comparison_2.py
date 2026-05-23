# =============================================================================
# run_comparison_2.py
# Run simulation for three runoff modes (none, simple, cascade)
# across three irrigation scenarios (0, 5, 8 mm/day) using the wet scenario.
#
# (Refactored: now uses src.terrain.load_terrain instead of an inline
# build_directed_graph implementation. Behavior is unchanged.)
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from abm import CropSoilABM
from climate_data import extract_scenario, load_cleaned_data
from soil_data import RICE as crop_params
from src.terrain import load_terrain

sys.path.insert(0, '.')


# ── Load DEM and build topology ───────────────────────────────────────────────

terrain = load_terrain('gilan_farm.tif')
gamma_flat = terrain['gamma_flat']
sends_to = terrain['sends_to']
Nr = terrain['Nr']
elevation = terrain['elevation_flat']
N = terrain['N']

OUTPUT_DIR = Path('results/comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Setup ─────────────────────────────────────────────────────────────────────

YEAR = 2024  # Wet scenario

df = load_cleaned_data()
climate = extract_scenario(df, YEAR, crop_params)
n_days = crop_params['season_days']

irrigations = [0.0, 5.0, 8.0]
modes = ['none', 'simple', 'cascade']
results = {}

# Identify sink and hilltop agents globally
sink_agents = [n for n in range(N) if Nr[n] == 0]
hilltop_agents = sorted(range(N), key=lambda n: -elevation[n])[:10]

# ── Run simulations ───────────────────────────────────────────────────────────

for irr in irrigations:
    results[irr] = {}
    print(f"\n--- Running Scenarios for {irr} mm Daily Irrigation ---")

    for mode in modes:
        print(f"  -> runoff_mode = '{mode}'")

        model = CropSoilABM(
            gamma_flat=gamma_flat,
            sends_to=sends_to,
            Nr=Nr,
            theta=crop_params,
            N=N,
            runoff_mode=mode,
            elevation=elevation,
        )

        state = model.reset()

        history = {
            'x4_mean': [],
            'x1_sinks': [],
            'x5_sinks': [],
        }

        for day in range(n_days):
            daily_climate = {
                'rainfall':  climate['rainfall'][day],
                'temp_mean': climate['temp_mean'][day],
                'temp_max':  climate['temp_max'][day],
                'radiation': climate['radiation'][day],
                'ET':        climate['ET'][day],
            }

            u = np.full(N, irr)  # Apply daily irrigation level
            state = model.step(u, daily_climate)

            history['x4_mean'].append(state['x4'].mean())
            history['x1_sinks'].append(state['x1'][sink_agents].mean())
            history['x5_sinks'].append(state['x5'][sink_agents].mean())

        # Convert to arrays and store final states
        for key in history:
            history[key] = np.array(history[key])

        history['x4_final'] = state['x4'].copy()
        history['x5_final'] = state['x5'].copy()

        results[irr][mode] = history

# ── Plot 1: High-Resolution Independent Scale Grid (3x3) ────────────────────

fig, axes = plt.subplots(3, 3, figsize=(22, 18), sharex=True, sharey=False)

fig.suptitle(f'Topographical Impact Analysis: {YEAR} Wet Scenario',
             fontsize=30, fontweight='normal', y=0.9)

metrics = [
    {'key': 'x1_sinks', 'label': 'Sink Agent Moisture ($x_1$)', 'unit': 'mm'},
    {'key': 'x5_sinks', 'label': 'Sink Agent Ponding ($x_5$)', 'unit': 'mm'},
    {'key': 'x4_mean',
        'label': 'Field Mean Biomass ($x_4$)', 'unit': '$g/m^2$'}
]

days = np.arange(n_days)
colors = {'none': '#95a5a6', 'simple': '#d35400', 'cascade': '#2980b9'}

for row_idx, metric in enumerate(metrics):
    for col_idx, irr in enumerate(irrigations):
        ax = axes[row_idx, col_idx]

        local_max = 0
        local_min = 100
        for mode in modes:
            data = results[irr][mode][metric['key']]
            local_max = max(local_max, np.max(data))
            local_min = min(local_min, np.min(data))

            lw = 2.0
            ax.plot(days, data, label=f'Mode: {mode}',
                    color=colors[mode],
                    linewidth=lw,
                    zorder=2)

        ax.set_ylim(local_min * 0.95, max(local_max * 1.1, 10))

        if row_idx == 0:
            ax.set_title(
                f'Irrigation: {irr} mm/day', fontsize=20, fontweight='normal', pad=10, y=1)

        if col_idx == 0:
            ax.set_ylabel(f"{metric['label']}\n[{metric['unit']}]",
                          fontsize=20, fontweight='normal', labelpad=10)

        if row_idx == 2:
            ax.set_xlabel('Day of Season', fontsize=20, fontweight='normal')

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)

        if row_idx == 0 and col_idx == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                      ncol=3, fontsize=20, frameon=True, shadow=True, borderpad=1)


plt.tight_layout(rect=[0, 0.03, 1, 0.85])
plt.subplots_adjust(wspace=0.20, hspace=0.25)
fig.suptitle(f'Topographical Impact Analysis: {YEAR} Wet Scenario',
             fontsize=36, fontweight='normal', y=0.84)

plt.savefig(OUTPUT_DIR / 'fig_runoff_comparison_grid.png',
            dpi=300, bbox_inches='tight')

print(f"Saved: {OUTPUT_DIR / 'fig_runoff_comparison_grid.png'}")

plt.close()
