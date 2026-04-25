# =============================================================================
# run_comparison.py
# Run no-irrigation simulation for three runoff modes (none, simple, cascade)
# using the wet scenario (2024). Compare soil water, biomass, ponding, and yield.
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
from soil_data import theta as crop_params
from src.terrain import load_terrain

sys.path.insert(0, '.')


# ── Load DEM and build topology ───────────────────────────────────────────────

terrain = load_terrain('gilan_farm.tif')
gamma_flat = terrain['gamma_flat']
sends_to = terrain['sends_to']
Nr = terrain['Nr']
elevation = terrain['elevation_flat']
N = terrain['N']
rows = terrain['rows']
cols = terrain['cols']

OUTPUT_DIR = Path('results/comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Setup ─────────────────────────────────────────────────────────────────────

YEAR = 2022  # wet scenario — has actual surface runoff events

df = load_cleaned_data()
climate = extract_scenario(df, YEAR)
n_days = crop_params['season_days']

modes = ['none', 'simple', 'cascade']
results = {}

# ── Run simulations ───────────────────────────────────────────────────────────

for mode in modes:
    print(f"Running: runoff_mode = '{mode}'...")

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
        'x1_mean': [], 'x1_min': [], 'x1_max': [],
        'x4_mean': [],
        'x1_sinks': [],   # average root zone water of sink agents
        'x1_hilltop': [],  # average root zone water of hilltop agents
        'x5_sinks': [],   # average surface ponding of sink agents
    }

    # Identify sink and hilltop agents
    sink_agents = [n for n in range(N) if Nr[n] == 0]
    hilltop_agents = sorted(range(N), key=lambda n: -elevation[n])[:10]

    for day in range(n_days):
        daily_climate = {
            'rainfall':  climate['rainfall'][day],
            'temp_mean': climate['temp_mean'][day],
            'temp_max':  climate['temp_max'][day],
            'radiation': climate['radiation'][day],
            'ET':        climate['ET'][day],
        }

        # u = np.zeros(N)  # no irrigation
        daily_irrigation = 0.0
        u = np.full(N, daily_irrigation)
        state = model.step(u, daily_climate)

        history['x1_mean'].append(state['x1'].mean())
        history['x1_min'].append(state['x1'].min())
        history['x1_max'].append(state['x1'].max())
        history['x4_mean'].append(state['x4'].mean())
        history['x1_sinks'].append(state['x1'][sink_agents].mean())
        history['x1_hilltop'].append(state['x1'][hilltop_agents].mean())
        history['x5_sinks'].append(state['x5'][sink_agents].mean())

    # Convert to arrays
    for key in history:
        history[key] = np.array(history[key])

    # Final state
    history['x4_final'] = state['x4'].copy()
    history['x1_final'] = state['x1'].copy()
    history['x5_final'] = state['x5'].copy()

    results[mode] = history

# ── Print summary ─────────────────────────────────────────────────────────────

HI = crop_params['HI']
print(f"\n{'=' * 85}")
print(f"NO-IRRIGATION COMPARISON — Rice, {YEAR} (Wet Scenario)")
print(f"{'=' * 85}")
print(f"\n{'Mode':<12} {'x4 mean':>10} {'Yield':>10} {'x1 final':>10} "
      f"{'x1 sinks':>10} {'x1 hills':>10} {'x5 sinks':>12}")
print(f"{'':>12} {'(g/m²)':>10} {'(kg/ha)':>10} {'(mm)':>10} "
      f"{'(mm)':>10} {'(mm)':>10} {'(flood mm)':>12}")
print("-" * 85)

for mode in modes:
    r = results[mode]
    x4_avg = r['x4_final'].mean()
    yield_kg = x4_avg * HI * 10
    x1_avg = r['x1_final'].mean()
    x1_sinks = r['x1_sinks'][-1]
    x1_hills = r['x1_hilltop'][-1]
    x5_sinks = r['x5_sinks'][-1]
    print(f"{mode:<12} {x4_avg:>10.1f} {yield_kg:>10.0f} {x1_avg:>10.1f} "
          f"{x1_sinks:>10.1f} {x1_hills:>10.1f} {x5_sinks:>12.1f}")

# ── Plot 1: Soil water comparison ─────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
fig.suptitle(f'Over-Irrigation Comparison ({daily_irrigation}mm daily irrigation) — Rice, {YEAR} (Wet Scenario)\n'
             f'Three runoff redistribution modes', fontsize=13)
days = np.arange(n_days)

# Panel 1: Mean soil water
ax = axes[0]
for mode in modes:
    ax.plot(days, results[mode]['x1_mean'], label=f'{mode}', linewidth=1.5)
ax.axhline(crop_params['theta6'] * crop_params['theta5'], color='green',
           linestyle='--', linewidth=0.8, label='FC')
ax.axhline(((1 - crop_params['p']) * crop_params['theta6'] + crop_params['p'] * crop_params['theta2']) * crop_params['theta5'], color='blue',
           linestyle='--', linewidth=0.8, label='Stress Threshold')
ax.axhline(crop_params['theta2'] * crop_params['theta5'], color='red',
           linestyle='--', linewidth=0.8, label='WP')
ax.set_ylabel('Soil water x₁ (mm)')
ax.set_title('Field-average root zone soil water')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Panel 2: Sink agents (Root Zone)
ax = axes[1]
for mode in modes:
    ax.plot(days, results[mode]['x1_sinks'], label=f'{mode}', linewidth=1.5)
ax.axhline(crop_params['theta_sat'] * crop_params['theta5'], color='blue',
           linestyle='--', linewidth=0.8, label='Saturation Cap')
ax.axhline(crop_params['theta6'] * crop_params['theta5'], color='green',
           linestyle='--', linewidth=0.8, label='FC')
ax.axhline(((1 - crop_params['p']) * crop_params['theta6'] + crop_params['p'] * crop_params['theta2']) * crop_params['theta5'], color='red',
           linestyle='--', linewidth=0.8, label='Stress Threshold')
ax.set_ylabel('Soil water x₁ (mm)')
ax.set_title('Sink agents (Nr=0) — Root Zone Moisture')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Panel 3: Sink agents (Surface Ponding)
ax = axes[2]
for mode in modes:
    ax.plot(days, results[mode]['x5_sinks'], label=f'{mode}', linewidth=1.5)
ax.set_ylabel('Ponding x₅ (mm)')
ax.set_title('Sink agents (Nr=0) — Standing Floodwater on Surface')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Panel 4: Biomass
ax = axes[3]
for mode in modes:
    ax.plot(days, results[mode]['x4_mean'], label=f'{mode}', linewidth=1.5)
ax.set_ylabel('Biomass x₄ (g/m²)')
ax.set_xlabel('Day of crop season')
ax.set_title('Field-average biomass accumulation')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_runoff_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'fig_runoff_comparison.png'}")

# ── Plot 2: Spatial difference at harvest ─────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f'Final surface ponding x₅ at harvest — spatial distribution', fontsize=13)

for i, mode in enumerate(modes):
    ax = axes[i]
    x5_2d = results[mode]['x5_final'].reshape(rows, cols)
    im = ax.imshow(x5_2d, cmap='Blues', origin='upper',
                   vmin=0, vmax=max(1.0, np.max(x5_2d)))
    ax.set_title(f'{mode}')
    plt.colorbar(im, ax=ax, label='x₅ (mm)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_runoff_spatial.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'fig_runoff_spatial.png'}")
