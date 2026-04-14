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

print("\n" + "=" * 60)
print("SECTION 4: MPC — Dry Scenario (2021)")
print("=" * 60)

abm_mpc = CropSoilABM(gamma_flat, sends_to, Nr, theta, N)

mpc_dry = run_mpc_season(
    abm_mpc, climate_dry, W_total,
    Hp=8,       # prediction horizon: 8 days (matches paper)
    lam=0.05,   # water cost: meaningful but not dominant
    UB=15.0     # max irrigation per agent per day (mm)
                # UB > ET0_mean (6.5mm) so controller can overcome losses
)

print(f"\n{'='*60}")
print(f"MPC Results — {theta['name']} — Dry Scenario (2021)")
print(f"{'='*60}")
print(f"  Final biomass (MPC):      {mpc_dry['x4'][-1]:.2f}")
print(f"  Final biomass (no irr):   {x4_noirr[-1]:.2f}")
print(f"  Biomass increase:         "
      f"{(mpc_dry['x4'][-1]-x4_noirr[-1])/max(x4_noirr[-1], 1)*100:.1f}%")
print(f"  Total irrigation used:    {sum(mpc_dry['u']):.1f} mm")
print(f"  Budget used:              "
      f"{sum(mpc_dry['u'])/W_total*100:.1f}% of {W_total:.0f}mm")
print(f"  Avg solve time per day:   {np.mean(mpc_dry['time']):.2f} s")
print(f"  Max solve time:           {np.max(mpc_dry['time']):.2f} s")
print(f"  Days above stress thresh: "
      f"{sum(x >= raw for x in mpc_dry['x1'])}/120")

# =============================================================================
# SECTION 5 — Plot MPC results
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 11))
fig.suptitle(f'MPC Results — {theta["name"]} — Dry Scenario (2021)',
             fontsize=13)

axes[0].plot(mpc_dry['x1'], color='blue', label='MPC soil water')
axes[0].axhline(fc,  color='red',    linestyle='--',
                label=f'Field capacity ({fc:.0f}mm)')
axes[0].axhline(wp,  color='orange', linestyle='--',
                label=f'Wilting point ({wp:.0f}mm)')
axes[0].axhline(raw, color='purple', linestyle=':',
                label=f'Stress threshold ({raw:.0f}mm)')
axes[0].set_ylabel('Mean Soil Water (mm)')
axes[0].set_title('Soil Water Content')
axes[0].legend(fontsize=8)

axes[1].plot(mpc_dry['x4'], color='green', label='MPC')
axes[1].plot(x4_noirr,      color='gray',  linestyle='--',
             label='No irrigation')
axes[1].set_ylabel('Mean Biomass')
axes[1].set_title('Biomass Growth')
axes[1].legend()

axes[2].bar(range(len(mpc_dry['u'])), mpc_dry['u'],
            color='steelblue', alpha=0.8)
axes[2].set_ylabel('Field Irrigation (mm/day)')
axes[2].set_xlabel('Day')
axes[2].set_title(
    f"Irrigation Schedule — "
    f"Used: {sum(mpc_dry['u']):.0f}mm / Budget: {W_total:.0f}mm "
    f"({sum(mpc_dry['u'])/W_total*100:.0f}%)")

plt.tight_layout()
plt.savefig('results/mpc_dry.png', dpi=150)
plt.show()
