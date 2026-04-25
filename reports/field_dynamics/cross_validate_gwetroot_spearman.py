# =============================================================================
# cross_validate_gwetroot_spearman.py
# Compare ABM simulated x1 (no irrigation, CASCADE runoff) against
# NASA GWETROOT. Includes Spearman Rank Correlation and Quadratic Curve fitting.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy import stats

from soil_data import RICE as crop_params
from climate_data import extract_scenario, load_cleaned_data
from abm import CropSoilABM

OUTPUT_DIR = Path('results/crossval')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load DEM and build topology ──────────────────────────────────────────────

elevation_matrix = np.array(Image.open('gilan_farm.tif'))
rows, cols = elevation_matrix.shape


def build_directed_graph(elev):
    r, c = elev.shape
    gamma = (elev - elev.min()) / (elev.max() - elev.min())
    sends_to = {}
    Nr = {}
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]
    for ri in range(r):
        for ci in range(c):
            n = ri * c + ci
            lower = []
            for dr, dc in directions:
                nr2, nc2 = ri + dr, ci + dc
                if 0 <= nr2 < r and 0 <= nc2 < c:
                    m = nr2 * c + nc2
                    if gamma[nr2, nc2] < gamma[ri, ci]:
                        lower.append(m)
            sends_to[n] = lower
            Nr[n] = len(lower)
    return gamma.flatten(), sends_to, Nr


gamma_flat, sends_to, Nr = build_directed_graph(elevation_matrix)
elevation = elevation_matrix.flatten()
N = len(gamma_flat)

# ── Configuration ─────────────────────────────────────────────────────────────

df = load_cleaned_data()
n_days = crop_params['season_days']
root_depth = crop_params['theta5']
FC = crop_params['theta6'] * root_depth
WP = crop_params['theta2'] * root_depth

# Optional: If you have a saturation/porosity param, use it here instead of FC
# Keeping your original baseline for now, but recommended to use actual Saturation
SATURATION = FC

scenarios = {
    'Dry (2022)':      2022,
    'Moderate (2020)':  2020,
    'Wet (2024)':      2024,
}

# ── Run simulations and compare ──────────────────────────────────────────────

results = {}

for label, year in scenarios.items():
    climate = extract_scenario(df, year)

    model = CropSoilABM(
        gamma_flat=gamma_flat,
        sends_to=sends_to,
        Nr=Nr,
        theta=crop_params,
        N=N,
        runoff_mode='cascade',  # UPDATED: Validating the true DAG topography
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

        # Un-irrigated baseline (Abandoned Farm scenario)
        u = np.zeros(N)
        state = model.step(u, daily_climate)

        # Spatial upscaling: average all 130 micro-topography agents to match the NASA macro-pixel
        x1_daily.append(state['x1'].mean())
        gwetroot_frac.append(climate['gwetroot'][day])

    x1_daily = np.array(x1_daily)
    gwetroot_frac = np.array(gwetroot_frac)

    # Normalize ABM x1 to fraction of available water (0 = WP, 1 = FC/SAT)
    x1_norm = np.clip((x1_daily - WP) / (SATURATION - WP), 0, None)

    # Calculate both Pearson (linear) and Spearman (monotonic/curved)
    pearson_r, pearson_p = stats.pearsonr(x1_norm, gwetroot_frac)
    spearman_r, spearman_p = stats.spearmanr(x1_norm, gwetroot_frac)

    results[label] = {
        'x1_mm': x1_daily,
        'x1_norm': x1_norm,
        'gwetroot_frac': gwetroot_frac,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'year': year,
        'rainfall': climate['rainfall'],
    }

# ── Print report ──────────────────────────────────────────────────────────────

lines = []
lines.append("=" * 70)
lines.append("GWETROOT CROSS-VALIDATION — No irrigation, CASCADE runoff")
lines.append(f"Crop: {crop_params['name']}")
lines.append(f"Root depth: {root_depth} mm")
lines.append(f"FC: {FC:.0f} mm, WP: {WP:.0f} mm")
lines.append("=" * 70)

for label, r in results.items():
    lines.append(f"\n{label}:")
    lines.append(
        f"  Spearman rank ρ:  {r['spearman_r']:.3f} (p = {r['spearman_p']:.2e})")
    lines.append(
        f"  Pearson linear r: {r['pearson_r']:.3f} (p = {r['pearson_p']:.2e})")
    lines.append(
        f"  ABM x1:           {r['x1_mm'].mean():.1f} mm (range: {r['x1_mm'].min():.1f}--{r['x1_mm'].max():.1f})")
    lines.append(f"  ABM normalized:   {r['x1_norm'].mean():.3f} (0=WP, 1=FC)")
    lines.append(
        f"  GWETROOT:         {r['gwetroot_frac'].mean():.3f} (NASA fraction)")
    lines.append(f"  Rainfall:         {r['rainfall'].sum():.1f} mm total")

report = '\n'.join(lines)
print(report)

with open(OUTPUT_DIR / 'gwetroot_crossval_spearman_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# ── Plot 2: Scatter plots with Quadratic Fit ──────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'ABM vs NASA GWETROOT — 2nd-Order Scatter Comparison (Cascade Mode)', fontsize=14)

for i, (label, r) in enumerate(results.items()):
    ax = axes[i]
    ax.scatter(r['gwetroot_frac'], r['x1_norm'],
               s=15, alpha=0.6, color='steelblue')

    # 1:1 line for reference
    lims = [0, max(r['gwetroot_frac'].max(), r['x1_norm'].max()) + 0.1]
    ax.plot(lims, lims, 'k--', linewidth=0.8, label='1:1')

    # UPDATED: 2nd-Order (Quadratic) Regression Curve
    z = np.polyfit(r['gwetroot_frac'], r['x1_norm'], 2)
    x_plot = np.linspace(lims[0], lims[1], 100)
    y_plot = z[0]*x_plot**2 + z[1]*x_plot + z[2]

    ax.plot(x_plot, y_plot, 'r-', linewidth=1.5,
            label=f'y = {z[0]:.2f}x² + {z[1]:.2f}x + {z[2]:.2f}')

    # Text box displaying both correlation metrics
    stats_text = (f"Spearman ρ = {r['spearman_r']:.3f}\n"
                  f"Pearson r = {r['pearson_r']:.3f}")

    ax.text(0.05, 0.85, stats_text, transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('NASA GWETROOT (fraction)')
    ax.set_ylabel('ABM normalized (0=WP, 1=FC)')
    ax.set_title(label)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_gwetroot_scatter_quadratic.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'fig_gwetroot_scatter_quadratic.png'}")
