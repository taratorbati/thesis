# =============================================================================
# mpc.py
# Model Predictive Control for crop irrigation.
# Baseline: Lopez-Jimenez et al. (2024), Control Engineering Practice.
# =============================================================================

import numpy as np
import copy
import time
from scipy.optimize import minimize


def compute_water_budget(climate, theta, N, scarcity=0.50):
    """
    Compute a physically meaningful seasonal water budget.

    Uses FAO crop water requirement method:
        Irrigation need = (ET0 * Kc * days) - rainfall
    Then applies a scarcity factor reflecting Iranian water allocation.

    Parameters
    ----------
    climate  : climate dict with 'ET' and 'rainfall' arrays
    theta    : crop/soil parameter dict (must contain 'Kc')
    N        : number of agents
    scarcity : fraction of full irrigation need to allow (0–1)
                0.50 = moderate scarcity (documented in Iranian literature)

    Returns
    -------
    W_total       : total field water budget (mm)
    W_per_agent   : budget per agent (mm)
    full_need     : full irrigation requirement per agent (mm)
    """
    n_days = len(climate['ET'])
    ET0_mean = float(np.mean(climate['ET']))
    Kc = theta['Kc']
    rainfall_total = float(np.sum(climate['rainfall']))

    # Full crop water demand minus rainfall already provided
    full_need = max(ET0_mean * Kc * n_days - rainfall_total, 0.0)

    W_per_agent = scarcity * full_need
    W_total = W_per_agent * N

    return W_total, W_per_agent, full_need


def run_mpc(abm, climate_forecast, W_remaining,
            Hp=8, lam=0.05, UB=15.0):
    """
    Solve one MPC step and return today's irrigation action.

    Parameters
    ----------
    abm              : CropSoilABM instance at current state
    climate_forecast : list of Hp dicts, one per forecast day
    W_remaining      : total water budget still available (mm)
    Hp               : prediction horizon (days)
    lam              : water cost weight — higher = more water-conservative
    UB               : max irrigation per agent per day (mm)

    Returns
    -------
    u_today : irrigation array of shape (N,) for today only
    """
    N = abm.N
    fc_total = abm.theta['theta6'] * abm.theta['theta5']  # field capacity mm
    wp_total = abm.theta['theta2'] * abm.theta['theta5']  # wilting point mm
    p = abm.theta['p']                              # FAO depletion fraction

    # FAO RAW stress threshold: stress begins below this soil water level
    # stress_threshold = FC - p * (FC - WP)
    stress_threshold = fc_total - p * (fc_total - wp_total)

    def cost(u_flat):
        u = u_flat.reshape(Hp, N)
        abm_sim = copy.deepcopy(abm)

        total_biomass = 0.0
        total_water = 0.0
        total_stress = 0.0

        for k in range(Hp):
            u_k = np.clip(u[k], 0, UB)
            state = abm_sim.step(u_k, climate_forecast[k])

            total_biomass += state['x4'].mean()
            total_water += u_k.mean()          # FIXED: mean not sum

            deficit = np.maximum(stress_threshold - state['x1'], 0)
            total_stress += deficit.mean()

        return -total_biomass + lam * total_water + 0.1 * total_stress

    # Effective upper bound: never allow more than remaining budget per agent
    ub_effective = min(UB, W_remaining / max(N, 1))
    if ub_effective < 0.01:
        return np.zeros(N)

    u0 = np.full(Hp * N, 2.0)              # clean warm start
    bounds = [(0.0, UB)] * (Hp * N)            # UB directly, not ub_effective

    result = minimize(
        cost, u0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-4,
                 'disp': False, 'eps': 1e-2}   # eps fixes gradient noise
    )

    # Take only first day (receding horizon principle)
    u_today = np.clip(result.x[:N], 0.0, UB)

    # Hard budget enforcement: never overspend
    total_today = u_today.sum()
    if total_today > W_remaining:
        u_today = u_today * (W_remaining / total_today)

    return u_today


def run_mpc_season(abm, climate, W_total, Hp=8, lam=0.05, UB=15.0):
    """
    Run MPC for a complete crop season.

    Parameters
    ----------
    abm     : CropSoilABM instance
    climate : climate dict with arrays of length n_days
    W_total : total seasonal water budget (mm)
    Hp      : prediction horizon (days)
    lam     : water cost weight
    UB      : max irrigation per agent per day (mm)

    Returns
    -------
    dict with keys: x1, x4, u, time (all lists of length n_days)
    """
    n_days = len(climate['rainfall'])
    W_remaining = W_total

    x1_history = []
    x4_history = []
    u_history = []
    time_history = []

    abm.reset()

    for day in range(n_days):

        # Build Hp-day forecast (clamp to last available day)
        forecast = []
        for k in range(Hp):
            idx = min(day + k, n_days - 1)
            forecast.append({
                'rainfall':  float(climate['rainfall'][idx]),
                'ET':        float(climate['ET'][idx]),
                'temp_mean': float(climate['temp_mean'][idx]),
                'temp_max':  float(climate['temp_max'][idx]),
                'radiation': float(climate['radiation'][idx]),
            })

        # Solve MPC
        t_start = time.time()
        u_today = run_mpc(abm, forecast, W_remaining,
                          Hp=Hp, lam=lam, UB=UB)
        t_end = time.time()

        # Apply to real ABM
        today = {k: v[day] for k, v in climate.items()
                 if k not in ['gwetroot', 'gwettop']}
        state = abm.step(u_today, today)

        # Update budget
        W_remaining = max(W_remaining - u_today.sum(), 0.0)

        # Record
        x1_history.append(float(state['x1'].mean()))
        x4_history.append(float(state['x4'].mean()))
        u_history.append(float(u_today.sum()))
        time_history.append(t_end - t_start)

        if day % 20 == 0:
            print(f"Day {day:3d} | "
                  f"biomass={state['x4'].mean():8.1f} | "
                  f"soil_water={state['x1'].mean():6.1f}mm | "
                  f"irrigation={u_today.sum():6.1f}mm | "
                  f"budget_left={W_remaining:7.0f}mm | "
                  f"solve={t_end-t_start:.1f}s")

    return {'x1': x1_history, 'x4': x4_history,
            'u': u_history,   'time': time_history}
