#!/usr/bin/env python3
# =============================================================================
# scripts/analysis/compute_correlations.py
#
# Computes pooled closed-loop correlation diagnostics for ALL SAC evaluation
# results.  Re-runs each saved scenario through the trained SAC model and
# collects per-step, per-agent action and state arrays.
#
# Fixes two problems with the notebook Cell 6 diagnostic:
#   1. Scenario: always wet/100% (the interesting cell).  This script runs
#      ALL 9 scenario×budget cells so you can see the pattern everywhere.
#   2. Correlation: POOLED across all (day × agent) pairs (N = 93 × 130 = 12 090),
#      not a time-series of field means (N = 93).  The pooled version has
#      sufficient statistical power to distinguish signal from noise.
#
# What each correlation means:
#   corr(u, x1)   POOLED  — does the policy reduce irrigation when soil is wet?
#                            MPC target: < −0.50.  v2.7 was near zero.
#   corr(u, rain) DAILY   — does the policy back off when it rains?
#                            MPC target: < −0.30.  v2.6 was +0.38.
#   corr(u, elev) SPATIAL — does the policy irrigate more at higher elevations?
#                            MPC target: > +0.50.  v2.6 was +0.05.
#
# Usage:
#   # All v2.7 seed 0 results:
#   python -m scripts.analysis.compute_correlations \
#       --model results/rl/sac_seed0_v27_*/best_model/best_model.zip \
#       --scenario all --budget all --out results/correlations_v27_seed0.csv
#
#   # Specific cell:
#   python -m scripts.analysis.compute_correlations \
#       --model results/rl/sac_seed0_v27_*/best_model/best_model.zip \
#       --scenario wet --budget 100 --out results/correlations_v27_seed0.csv
#
#   # v2.8 seed 0:
#   python -m scripts.analysis.compute_correlations \
#       --model results/rl/sac_v28_seed0/best_model/best_model.zip \
#       --scenario all --budget all --out results/correlations_v28_seed0.csv
#
# Output CSV columns:
#   scenario, budget_pct, corr_u_x1_pooled, corr_u_rain_daily,
#   corr_u_elev_spatial, waterlog_days_per_agent, water_used_mm,
#   yield_kg_ha, N_pooled
#
# Requirements: stable-baselines3, numpy, pandas, gymnasium
# The script imports from the repo root — run from the repo root.
# =============================================================================

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

# Make sure the repo root is on sys.path when running as a module
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCENARIOS = ['dry', 'moderate', 'wet']
BUDGETS   = [100, 85, 70]

SCENARIO_YEAR = {'dry': 2022, 'moderate': 2018, 'wet': 2024}
FULL_SEASON_NEED_MM = 484.0
UB_MM = 12.0

# Observation layout of the loaded checkpoint (set in main() from load_policy);
# the analysis env must match it so model.predict receives the right obs width.
_DEDUPE = True


def _make_env(scenario: str, budget_pct: int):
    """Create a fixed-mode IrrigationEnv forced to the given scenario/budget."""
    from src.rl.gym_env import IrrigationEnv, FULL_SEASON_NEED_MM as FSN
    from src.model.climate_data import load_cleaned_data, extract_scenario
    from src.model.soil_data import get_crop
    from src.sim.precompute import get_precomputed

    env = IrrigationEnv(randomize=False, dedupe_today_weather=_DEDUPE)
    # The env defaults to dry/100 on reset; patch it post-construction.
    # We store the overrides and apply them after each reset.
    env._target_year    = SCENARIO_YEAR[scenario]
    env._target_budget  = FSN * (budget_pct / 100.0)
    env._target_scenario = scenario

    _df   = load_cleaned_data()
    _crop = get_crop('rice')
    env._climate_override  = extract_scenario(_df, SCENARIO_YEAR[scenario], _crop)
    env._precomp_override  = get_precomputed(scenario, 'rice')
    env._crop_ref = _crop
    return env


def _apply_overrides(env):
    """Force the internal year/budget/climate after reset()."""
    env._year      = env._target_year
    env._budget_mm = env._target_budget
    env._climate   = env._climate_override
    env._precomp   = env._precomp_override


def run_episode(model, scenario: str, budget_pct: int, terrain) -> dict:
    """Run one deterministic episode and return correlation diagnostics.

    Parameters
    ----------
    model : loaded SB3 SAC model
    scenario : 'dry' | 'moderate' | 'wet'
    budget_pct : 100 | 85 | 70
    terrain : dict from load_terrain()

    Returns
    -------
    dict with all correlation metrics and summary scalars.
    """
    from src.rl.gym_env import _FC_MM, _WP_MM

    env_fn   = lambda: _make_env(scenario, budget_pct)
    vec_env  = DummyVecEnv([env_fn])
    inner    = vec_env.envs[0]

    obs = vec_env.reset()
    _apply_overrides(inner)
    # Re-build the initial obs with the overridden state
    obs = np.array([inner._build_obs()])

    # Determine obs layout from model's obs_dim
    obs_dim = obs.shape[1]
    if obs_dim == 1227:
        n_feat, block_end = 9, 1170
    elif obs_dim == 1097:
        n_feat, block_end = 8, 1040
    else:  # v2.6
        n_feat, block_end = 5, 650

    # Static per-agent elevation (agent-major, slot 4)
    elev_static = terrain['gamma_flat'].astype(np.float32)  # (130,)

    # Collect per-step arrays
    all_u       = []   # (130,) irrigations in mm/day
    all_x1      = []   # (130,) soil moisture in mm (decoded from obs)
    all_rain    = []   # float rain_today
    ep_len      = 0
    done        = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        u_agents  = action[0] * UB_MM            # (130,) in mm/day

        raw_obs    = obs[0]                        # (obs_dim,)
        agent_grid = raw_obs[:block_end].reshape(130, n_feat)

        # Feature 0 is x1_norm = (x1 - WP)/(FC - WP)
        # Invert to get x1 in mm
        x1_norm = agent_grid[:, 0]
        x1_mm   = x1_norm * (_FC_MM - _WP_MM) + _WP_MM

        # Rain is scalar block index 4 (first 4 scalar positions are
        # day_frac, budget_frac, budget_total_norm, burn_rate)
        rain_idx = block_end + 4
        rain     = float(raw_obs[rain_idx])

        all_u.append(u_agents)
        all_x1.append(x1_mm)
        all_rain.append(rain)

        obs, _, done_arr, info = vec_env.step(action)
        _apply_overrides(inner)
        ep_len += 1
        done    = done_arr[0]

    T   = ep_len
    N   = 130
    u   = np.array(all_u,    dtype=np.float64)   # (T, N)
    x1  = np.array(all_x1,   dtype=np.float64)   # (T, N)
    rain = np.array(all_rain, dtype=np.float64)   # (T,)

    # ── corr(u, x1) pooled across all (day, agent) pairs ────────────────────
    u_pool  = u.flatten()    # (T*N,)
    x1_pool = x1.flatten()   # (T*N,)
    corr_u_x1 = float(np.corrcoef(u_pool, x1_pool)[0, 1])

    # ── corr(u, rain) on field-mean daily time series ────────────────────────
    u_daily = u.mean(axis=1)   # (T,)
    corr_u_rain = float(np.corrcoef(u_daily, rain)[0, 1])

    # ── corr(u, elev) on per-agent season mean ───────────────────────────────
    u_agent_mean = u.mean(axis=0)   # (N,)
    corr_u_elev  = float(np.corrcoef(u_agent_mean, elev_static)[0, 1])

    # ── waterlog days per agent ───────────────────────────────────────────────
    waterlog_days = float((x1 > _FC_MM).sum(axis=0).mean())   # per-agent mean

    # ── budget and yield ─────────────────────────────────────────────────────
    water_used = float(u.mean(axis=1).sum())   # field-mean daily × days
    # Yield from last info
    yield_kg_ha = float(info[0].get('yield_kg_ha', np.nan))

    return {
        'scenario':                scenario,
        'budget_pct':              budget_pct,
        'corr_u_x1_pooled':        round(corr_u_x1,   4),
        'corr_u_rain_daily':       round(corr_u_rain,  4),
        'corr_u_elev_spatial':     round(corr_u_elev,  4),
        'waterlog_days_per_agent': round(waterlog_days, 2),
        'water_used_mm':           round(water_used,    2),
        'yield_kg_ha':             round(yield_kg_ha,   1),
        'N_pooled':                T * N,
        'ep_len':                  T,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute pooled correlation diagnostics for a trained SAC model.'
    )
    parser.add_argument(
        '--model', required=True,
        help='Glob pattern or exact path to best_model.zip. '
             'Example: results/rl/sac_seed0_v27_*/best_model/best_model.zip',
    )
    parser.add_argument(
        '--scenario', default='all',
        choices=['dry', 'moderate', 'wet', 'all'],
        help='Climate scenario to evaluate (default: all).',
    )
    parser.add_argument(
        '--budget', default='all',
        choices=['100', '85', '70', 'all'],
        help='Budget percentage to evaluate (default: all).',
    )
    parser.add_argument(
        '--out', default=None,
        help='Output CSV path. Default: results/correlations_<model_stem>.csv',
    )
    args = parser.parse_args()

    # ── resolve model path ───────────────────────────────────────────────────
    model_paths = sorted(glob.glob(str(args.model)))
    if not model_paths:
        # Try appending .zip
        model_paths = sorted(glob.glob(str(args.model) + '.zip'))
    if not model_paths:
        print(f"ERROR: No model found matching '{args.model}'")
        sys.exit(1)
    model_path = Path(model_paths[-1])
    print(f"Model: {model_path}")

    # ── load model ───────────────────────────────────────────────────────────
    from src.rl.runner import load_policy
    model, arch_label, dedupe = load_policy(model_path, device='cpu')
    globals()['_DEDUPE'] = dedupe   # env obs layout must match the checkpoint
    print(f"Architecture: {arch_label}")

    # ── load terrain (for elevation correlation) ─────────────────────────────
    from src.model.terrain import load_terrain
    terrain = load_terrain('data/gilan_farm.tif')

    # ── build cell list ───────────────────────────────────────────────────────
    scenarios = SCENARIOS if args.scenario == 'all' else [args.scenario]
    budgets   = BUDGETS   if args.budget   == 'all' else [int(args.budget)]
    cells     = [(s, b) for s in scenarios for b in budgets]

    # ── run cells ────────────────────────────────────────────────────────────
    rows = []
    for scenario, budget_pct in cells:
        label = f"{scenario}/{budget_pct}%"
        print(f"\n  Running {label} ...", end='', flush=True)
        try:
            row = run_episode(model, scenario, budget_pct, terrain)
            rows.append(row)
            print(f"  corr(u,x1)={row['corr_u_x1_pooled']:+.3f}  "
                  f"corr(u,rain)={row['corr_u_rain_daily']:+.3f}  "
                  f"waterlog={row['waterlog_days_per_agent']:.0f}d  "
                  f"yield={row['yield_kg_ha']:.0f} kg/ha")
        except Exception as e:
            print(f"  FAILED: {e}")

    if not rows:
        print("No results collected.")
        sys.exit(1)

    # ── save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_path = args.out or f"results/correlations_{model_path.parents[1].name}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {out_path}")

    # ── print summary table ───────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"{'Cell':<18}  {'corr(u,x1)':>10}  {'corr(u,rain)':>12}  "
          f"{'corr(u,elev)':>12}  {'waterlog_d':>10}  {'yield':>8}")
    print("=" * 90)
    for _, r in df.iterrows():
        cell = f"{r.scenario}/{r.budget_pct}%"
        print(f"{cell:<18}  {r.corr_u_x1_pooled:>+10.3f}  "
              f"{r.corr_u_rain_daily:>+12.3f}  "
              f"{r.corr_u_elev_spatial:>+12.3f}  "
              f"{r.waterlog_days_per_agent:>10.1f}  "
              f"{r.yield_kg_ha:>8.0f}")
    print()
    print("MPC Hp=3 reference targets (from thesis):")
    print("  corr(u, x1)   ~  -0.58 (wet/100%)   [SAC target: < 0]")
    print("  corr(u, rain) ~  -0.31 (wet/100%)   [SAC target: < 0]")
    print("  corr(u, elev) ~  +0.65 (dry/100%)   [SAC target: > +0.5]")
    print("  waterlog days ~   20.2 (wet/100%)   [SAC target: < 60]")


if __name__ == '__main__':
    main()
