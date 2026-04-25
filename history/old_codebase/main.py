# =============================================================================
# main.py
# Master script: ABM validation + MPC baseline for thesis irrigation study.
# Field: Gilan province, Iran | 130 agents | 6 hectares
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

from thesis_env import gamma_flat, sends_to, Nr, rows, cols
from soil_data import theta
from climate_data import climate_dry, climate_wet
from abm import CropSoilABM
from mpc import run_mpc_season, compute_water_budget

N = rows * cols  # 130 agents

# =============================================================================
# SECTION 1 — ABM Validation (fixed irrigation schedule)
# Purpose: confirm ABM equations produce physically correct behavior
# Reference: Lopez-Jimenez et al. (2024) Figure 7 baseline
# =============================================================================
print("=" * 60)
print("SECTION 1: ABM Validation")
print("=" * 60)

abm_val = CropSoilABM(gamma_flat, sends_to, Nr, theta, N)
abm_val.reset()
x1_val, x4_val = [], []

# Calculate the strict budget for the specific crop and climate
W_total, W_per_agent, full_need = compute_water_budget(
    climate_dry, theta, N, scarcity=0.50)

# Calculate the fixed open-loop schedule
interval = 4  # 8
num_events = 120 / interval
fixed_irrigation_amount = W_per_agent / num_events

print(
    f"Fixed schedule applies {fixed_irrigation_amount:.1f} mm every {interval} days.")

for day in range(120):
    today = {k: v[day] for k, v in climate_dry.items()
             if k not in ['gwetroot', 'gwettop']}

    # Fixed schedule: irrigate every 8 days (mirrors paper baseline)
    # Apply the perfectly distributed fraction of the budget
    u = np.full(
        N, fixed_irrigation_amount) if day % interval == 0 else np.zeros(N)
    # u = np.full(N, 15) if day % interval == 0 else np.zeros(N)

    state = abm_val.step(u, today)
    x1_val.append(state['x1'].mean())
    x4_val.append(state['x4'].mean())

'''
irrigation_days  = {8, 16, 24, 32, 52, 75, 100, 115}
amount_per_event = W_total / (N * len(irrigation_days))

for day in range(120):
    today = {k: v[day] for k, v in climate_dry.items()
             if k not in ['gwetroot', 'gwettop']}
    u = np.full(N, amount_per_event) if day in irrigation_days \
        else np.zeros(N)
    state = abm_val.step(u, today)
    x1_val.append(state['x1'].mean())
    x4_val.append(state['x4'].mean())

'''

fc = theta['theta6'] * theta['theta5']   # 140mm
wp = theta['theta2'] * theta['theta5']   # 60mm
raw = fc - theta['p'] * (fc - wp)         # stress threshold

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
ax1.plot(x1_val, color='blue', label='Soil water')
ax1.axhline(fc,  color='red',    linestyle='--',
            label=f'Field capacity ({fc:.0f}mm)')
ax1.axhline(wp,  color='orange', linestyle='--',
            label=f'Wilting point ({wp:.0f}mm)')
ax1.axhline(raw, color='purple', linestyle=':',
            label=f'Stress threshold ({raw:.0f}mm)')
ax1.set_ylabel('Mean Soil Water (mm)')
ax1.set_title(f'ABM Validation — Soil Water ({theta["name"]})')
ax1.legend()
ax2.plot(x4_val, color='green')
ax2.set_ylabel('Mean Biomass')
ax2.set_xlabel('Day')
ax2.set_title('ABM Validation — Biomass Growth (S-curve expected)')
plt.tight_layout()
plt.savefig('results/abm_validation.png', dpi=150)
plt.show()

print(f"Fixed-schedule biomass:  {x4_val[-1]:.2f}")
print(f"Mean soil water:         {np.mean(x1_val):.1f} mm")

# =============================================================================
# SECTION 2 — No-irrigation baseline
# Purpose: establish lower bound for biomass without any irrigation
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: No-Irrigation Baseline")
print("=" * 60)

abm_noirr = CropSoilABM(gamma_flat, sends_to, Nr, theta, N)
abm_noirr.reset()
x4_noirr = []

for day in range(120):
    today = {k: v[day] for k, v in climate_dry.items()
             if k not in ['gwetroot', 'gwettop']}
    state = abm_noirr.step(np.zeros(N), today)
    x4_noirr.append(state['x4'].mean())

print(f"No-irrigation biomass:   {x4_noirr[-1]:.2f}")

# =============================================================================
# SECTION 3 — Water budget calculation
# Purpose: compute agronomically correct budget reflecting Iranian scarcity
# Reference: FAO Paper 56 (Kc method) + Afrasiabikia et al. (2026)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: Water Budget")
print("=" * 60)

SCARCITY = 0.70   # 70% of full crop requirement — moderate Iranian scarcity

W_total, W_per_agent, full_need = compute_water_budget(
    climate_dry, theta, N, scarcity=SCARCITY
)


print(f"Crop:                    {theta['name']}")
print(f"Kc (season average):     {theta['Kc']}")
print(f"ET0 mean:                {np.mean(climate_dry['ET']):.2f} mm/day")
print(
    f"Seasonal ET demand:      {np.mean(climate_dry['ET'])*theta['Kc']*120:.0f} mm/agent")
print(f"Seasonal rainfall:       {climate_dry['rainfall'].sum():.0f} mm/agent")
print(f"Full irrigation need:    {full_need:.0f} mm/agent")
print(f"Scarcity factor:         {SCARCITY*100:.0f}%")
print(
    f"Budget per agent:        {W_per_agent:.0f} mm ({W_per_agent/120:.1f} mm/day avg)")
print(f"Total field budget:      {W_total:.0f} mm")

# =============================================================================
# SECTION 4 — MPC runs: 2 scenarios × 2 horizons = 4 total runs
# =============================================================================

scenarios = {
    'dry': climate_dry,
    'wet': climate_wet,
}
horizons = [8, 14]

# --- No-irrigation baselines for both scenarios ---
print("\n" + "=" * 60)
print("No-Irrigation Baselines")
print("=" * 60)

noirr_biomass = {}
for scenario_name, climate in scenarios.items():
    abm_base = CropSoilABM(gamma_flat, sends_to, Nr, theta, N)
    abm_base.reset()
    x4_base = []
    for day in range(120):
        today = {k: v[day] for k, v in climate.items()
                 if k not in ['gwetroot', 'gwettop']}
        state = abm_base.step(np.zeros(N), today)
        x4_base.append(state['x4'].mean())
    noirr_biomass[scenario_name] = x4_base[-1]
    print(f"  {scenario_name}: {x4_base[-1]:.2f}")

# --- Run all 4 MPC combinations ---
results = {}

for scenario_name, climate in scenarios.items():

    # Compute budget from this scenario's actual climate
    W_total, W_per_agent, full_need = compute_water_budget(
        climate, theta, N, scarcity=0.70
    )
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name.upper()} | "
          f"Budget: {W_total:.0f}mm total ({W_per_agent:.0f}mm/agent)")
    print(f"{'='*60}")

    for Hp in horizons:
        key = f"{scenario_name}_Hp{Hp}"
        print(f"\nRunning MPC — {scenario_name} scenario, Hp={Hp}...")

        abm_mpc = CropSoilABM(gamma_flat, sends_to, Nr, theta, N)
        results[key] = run_mpc_season(
            abm_mpc, climate, W_total,
            Hp=Hp, lam=0.05, UB=15.0
        )

# =============================================================================
# SECTION 5 — Summary table
# =============================================================================
print(f"\n{'='*75}")
print(f"{'Scenario':<10} {'Hp':>4} {'MPC biomass':>12} "
      f"{'No-irr':>10} {'Increase':>10} "
      f"{'Avg solve':>11} {'Max solve':>11}")
print(f"{'='*75}")

for scenario_name in scenarios:
    for Hp in horizons:
        key = f"{scenario_name}_Hp{Hp}"
        r = results[key]
        noirr = noirr_biomass[scenario_name]
        inc = (r['x4'][-1] - noirr) / max(noirr, 1) * 100
        print(f"{scenario_name:<10} {Hp:>4} "
              f"{r['x4'][-1]:>12.2f} "
              f"{noirr:>10.2f} "
              f"{inc:>9.1f}% "
              f"{np.mean(r['time']):>10.1f}s "
              f"{np.max(r['time']):>10.1f}s")

# =============================================================================
# SECTION 6 — Plots for all 4 runs
# =============================================================================
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle(f'MPC Results — {theta["name"]} — All Scenarios',
             fontsize=14)

fc = theta['theta6'] * theta['theta5']
wp = theta['theta2'] * theta['theta5']
raw = fc - theta['p'] * (fc - wp)

for i, scenario_name in enumerate(scenarios):
    climate = scenarios[scenario_name]
    noirr = noirr_biomass[scenario_name]

    # No-irrigation biomass curve for this scenario
    abm_base = CropSoilABM(gamma_flat, sends_to, Nr, theta, N)
    abm_base.reset()
    x4_noirr_curve = []
    for day in range(120):
        today = {k: v[day] for k, v in climate.items()
                 if k not in ['gwetroot', 'gwettop']}
        state = abm_base.step(np.zeros(N), today)
        x4_noirr_curve.append(state['x4'].mean())

    for j, Hp in enumerate(horizons):
        key = f"{scenario_name}_Hp{Hp}"
        r = results[key]
        row = i * 2 + j  # rows: dry_Hp8, dry_Hp14, wet_Hp8, wet_Hp14

        # Soil water
        axes[row, 0].plot(r['x1'], color='blue')
        axes[row, 0].axhline(fc,  color='red',    linestyle='--',
                             label=f'FC ({fc:.0f}mm)')
        axes[row, 0].axhline(wp,  color='orange', linestyle='--',
                             label=f'WP ({wp:.0f}mm)')
        axes[row, 0].axhline(raw, color='purple', linestyle=':',
                             label=f'Stress ({raw:.0f}mm)')
        axes[row, 0].set_ylabel('Soil Water (mm)')
        axes[row, 0].set_title(
            f'{scenario_name.upper()} Hp={Hp} — Soil Water')
        axes[row, 0].legend(fontsize=7)

        # Biomass
        axes[row, 1].plot(r['x4'],        color='green', label='MPC')
        axes[row, 1].plot(x4_noirr_curve, color='gray',
                          linestyle='--',  label='No irrigation')
        inc = (r['x4'][-1] - noirr) / max(noirr, 1) * 100
        axes[row, 1].set_title(
            f'{scenario_name.upper()} Hp={Hp} — '
            f'Biomass (+{inc:.0f}%)')
        axes[row, 1].set_ylabel('Mean Biomass')
        axes[row, 1].legend(fontsize=7)

        # Irrigation schedule
        axes[row, 2].bar(range(len(r['u'])), r['u'],
                         color='steelblue', alpha=0.8)
        axes[row, 2].set_title(
            f'{scenario_name.upper()} Hp={Hp} — '
            f'Irrigation ({sum(r["u"]):.0f}mm)')
        axes[row, 2].set_ylabel('Field Irrigation (mm/day)')
        axes[row, 2].set_xlabel('Day')

plt.tight_layout()
plt.savefig('results/mpc_all_scenarios.png', dpi=150)
plt.show()
