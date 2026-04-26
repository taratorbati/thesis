# =============================================================================
# cross_validate_gwetroot.py
# Compare ABM simulated x1 (no irrigation, simple runoff) against
# NASA GWETROOT × root_depth for the three scenario years.
#
# (Refactored: now uses src.terrain.load_terrain instead of an inline
# build_directed_graph implementation. Behavior is unchanged.)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from soil_data import RICE as crop_params
from climate_data import extract_scenario, load_cleaned_data
from abm import CropSoilABM
from src.terrain import load_terrain

OUTPUT_DIR = Path('results/crossval')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load DEM and build topology ──────────────────────────────────────────────

terrain = load_terrain('gilan_farm.tif')
gamma_flat = terrain['gamma_flat']
sends_to = terrain['sends_to']
Nr = terrain['Nr']
elevation = terrain['elevation_flat']
N = terrain['N']

# ── Configuration ─────────────────────────────────────────────────────────────

df = load_cleaned_data()
n_days = crop_params['season_days']
root_depth = crop_params['theta5']
FC = crop_params['theta6'] * root_depth
WP = crop_params['theta2'] * root_depth

scenarios = {
    'Dry (2022)':      2022,
    'Moderate (2020)':  2020,
    'Wet (2024)':      2024,
}

# ── Run simulations and compare ──────────────────────────────────────────────

results = {}

for label, year in scenarios.items():
    climate = extract_scenario(df, year, crop_params)

    model = CropSoilABM(
        gamma_flat=gamma_flat,
        sends_to=sends_to,
        Nr=Nr,
        theta=crop_params,
        N=N,
        runoff_mode='cascade',
        elevation=elevation,
    )
    state = model.reset()

    x1_daily = []
    gwetroot_frac = []

    for day in range(n_days):
        daily_climate = {
            'rainfall':  climate['rainfall'][day],
            'temp_mean': climate['temp_mean'][day],
            'temp_max':  climate['temp_max'][day],
            'radiation': climate['radiation'][day],
            'ET':        climate['ET'][day],
        }
        u = np.zeros(N)
        state = model.step(u, daily_climate)

        x1_daily.append(state['x1'].mean())
        gwetroot_frac.append(climate['gwetroot'][day])

    x1_daily = np.array(x1_daily)
    gwetroot_frac = np.array(gwetroot_frac)

    # Assuming crop_params has a porosity/saturation parameter (e.g., 'theta_sat' or similar)
    # If not, a standard approximation for clay/loam rice soil is FC / 0.5 to FC / 0.7
    SATURATION = crop_params.get(
        'theta_sat', crop_params['theta6'] * 1.4) * root_depth

    # Normalize ABM x1 to fraction of TOTAL available pore space (0 = WP, 1 = SATURATION)
    x1_norm = np.clip((x1_daily - WP) / (SATURATION - WP), 0, None)

    # Normalize ABM x1 to fraction of available water (0 = WP, 1 = FC)
    # x1_norm = np.clip((x1_daily - WP) / (FC - WP), 0, None)

    # Correlation on normalized series
    r_val, p_val = stats.pearsonr(x1_norm, gwetroot_frac)

    results[label] = {
        'x1_mm': x1_daily,
        'x1_norm': x1_norm,
        'gwetroot_frac': gwetroot_frac,
        'r': r_val,
        'p': p_val,
        'year': year,
        'rainfall': climate['rainfall'],
    }

# ── Print report ──────────────────────────────────────────────────────────────

lines = []
lines.append("=" * 70)
lines.append("GWETROOT CROSS-VALIDATION — No irrigation, simple runoff")
lines.append(f"Crop: {crop_params['name']}")
lines.append(f"Root depth: {root_depth} mm")
lines.append(f"FC: {FC:.0f} mm, WP: {WP:.0f} mm")
lines.append("=" * 70)

for label, r in results.items():
    lines.append(f"\n{label}:")
    lines.append(f"  Pearson r:        {r['r']:.3f}")
    lines.append(f"  p-value:          {r['p']:.2e}")
    lines.append(
        f"  ABM x1:           {r['x1_mm'].mean():.1f} mm (range: {r['x1_mm'].min():.1f}--{r['x1_mm'].max():.1f})")
    lines.append(f"  ABM normalized:   {r['x1_norm'].mean():.3f} (0=WP, 1=FC)")
    lines.append(
        f"  GWETROOT:         {r['gwetroot_frac'].mean():.3f} (NASA fraction)")
    lines.append(f"  Rainfall:         {r['rainfall'].sum():.1f} mm total")

report = '\n'.join(lines)
print(report)

with open(OUTPUT_DIR / 'gwetroot_crossval_report_pearson.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# ── Plot 1: Time series comparison ────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Soil Moisture Cross-Validation: ABM vs NASA GWETROOT\n'
             'Normalized comparison — No irrigation, simple runoff', fontsize=13)

for i, (label, r) in enumerate(results.items()):
    ax = axes[i]
    days = np.arange(n_days)

    ax.plot(days, r['x1_norm'], color='blue', linewidth=1.5,
            label='ABM (x₁−WP)/(FC−WP)')
    ax.plot(days, r['gwetroot_frac'], color='brown', linewidth=1.5,
            linestyle='--', label='NASA GWETROOT')

    ax.axhline(1.0, color='green', linestyle=':', linewidth=0.8, label='FC')
    ax.axhline(0.0, color='red', linestyle=':', linewidth=0.8, label='WP')

    # Rain bars on secondary axis
    ax2 = ax.twinx()
    ax2.bar(days, r['rainfall'], color='lightblue', alpha=0.4, width=1)
    ax2.set_ylim(0, max(r['rainfall'].max() * 3, 10))
    ax2.invert_yaxis()
    ax2.set_ylabel('Rain (mm)', color='lightblue')

    ax.set_ylabel('Soil wetness fraction')
    ax.set_title(f"{label} — r = {r['r']:.3f} (p = {r['p']:.2e})")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

axes[-1].set_xlabel('Day of crop season')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_gwetroot_timeseries_linear.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'fig_gwetroot_timeseries_linear.png'}")

# ── Plot 2: Scatter plots ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('ABM vs GWETROOT — Normalized Scatter Comparison', fontsize=13)

for i, (label, r) in enumerate(results.items()):
    ax = axes[i]
    ax.scatter(r['gwetroot_frac'], r['x1_norm'],
               s=10, alpha=0.6, color='steelblue')

    # 1:1 line
    lims = [0, max(r['gwetroot_frac'].max(), r['x1_norm'].max()) + 0.1]
    ax.plot(lims, lims, 'k--', linewidth=0.8, label='1:1')

    # Regression line
    z = np.polyfit(r['gwetroot_frac'], r['x1_norm'], 1)
    ax.plot(lims, [z[0]*v + z[1] for v in lims], 'r-', linewidth=1,
            label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

    ax.text(0.05, 0.90, f"r = {r['r']:.3f}", transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('NASA GWETROOT (fraction)')
    ax.set_ylabel('ABM normalized (0=WP, 1=FC)')
    ax.set_title(label)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_gwetroot_scatter_linear.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'fig_gwetroot_scatter_linear.png'}")
