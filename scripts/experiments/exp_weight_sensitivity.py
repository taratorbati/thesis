# =============================================================================
# scripts/experiments/exp_weight_sensitivity.py
# Weight sensitivity analysis for the MPC cost function.
#
# Runs 7 weight configurations on dry/100% with Hp=8:
#   - α2 sweep (water cost): {0.0001, 0.01, 0.03} — based on Iranian water prices
#     0.0001 = agricultural subsidized price
#     0.01   = domestic water price (nominal)
#     0.03   = industrial water price
#   - α3 sweep (drought penalty): {0.03, 0.3} — ±1 order of magnitude
#   - α4 sweep (ponding penalty): {0.17, 1.5} — ±½ order of magnitude
#
# Each sweep varies one weight while keeping others at nominal values
# (α1=1.0, α2=0.01, α3=0.1, α4=0.5, α5=0.005).
#
# The nominal run (α2=0.01, α3=0.1, α4=0.5) is already in the MPC grid,
# so only 6 additional runs are needed (7 total including nominal).
#
# Usage:
#   python -m scripts.experiments.exp_weight_sensitivity
#   python -m scripts.experiments.exp_weight_sensitivity --force
#
# References:
#   - α2 values: Iran Water Resources Management Company (2023) tariff data
#   - Sensitivity methodology: Saltelli et al. (2004) "Sensitivity Analysis
#     in Practice"
# =============================================================================

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from climate_data import load_cleaned_data, extract_scenario_by_name, SCENARIO_YEARS
from soil_data import get_crop
from src.mpc.controller import MPCController
from src.runner import run_season
from src.terrain import load_terrain

DEM_PATH = PROJECT_ROOT / 'gilan_farm.tif'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'runs'

# Nominal weights (from ARCHITECTURE.md)
NOMINAL = {'alpha1': 1.0, 'alpha2': 0.01, 'alpha3': 0.1, 'alpha4': 0.5, 'alpha5': 0.005}

# Weight configurations to test
WEIGHT_CONFIGS = [
    # α2 sweep (water cost)
    {'name': 'a2_0.0001', 'alpha2': 0.0001},
    {'name': 'a2_0.01',   'alpha2': 0.01},    # nominal (will skip if exists)
    {'name': 'a2_0.03',   'alpha2': 0.03},
    # α3 sweep (drought penalty)
    {'name': 'a3_0.03',   'alpha3': 0.03},
    {'name': 'a3_0.3',    'alpha3': 0.3},
    # α4 sweep (ponding penalty)
    {'name': 'a4_0.17',   'alpha4': 0.17},
    {'name': 'a4_1.5',    'alpha4': 1.5},
]


def main():
    parser = argparse.ArgumentParser(description='Weight sensitivity analysis.')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed scenario: dry/100%/Hp8/perfect
    scenario = 'dry'
    budget_pct = 100
    Hp = 8

    crop = get_crop('rice')
    terrain = load_terrain(str(DEM_PATH))
    df_climate = load_cleaned_data()
    climate = extract_scenario_by_name(df_climate, scenario, crop)
    climate['year'] = SCENARIO_YEARS[scenario]

    budget_total = 484.0 * (budget_pct / 100.0)

    print(f"Weight sensitivity analysis: {scenario}/{budget_pct}% Hp={Hp}")
    print(f"Nominal weights: {NOMINAL}")
    print(f"Configurations to test: {len(WEIGHT_CONFIGS)}\n")

    for config in WEIGHT_CONFIGS:
        name = config['name']

        # Build weight dict: nominal + override
        weights = dict(NOMINAL)
        for k, v in config.items():
            if k != 'name':
                weights[k] = v

        output_filename = f"mpc_perfect_{scenario}_rice_{budget_pct}pct_Hp{Hp}_wsens_{name}.parquet"
        output_path = OUTPUT_DIR / output_filename

        if not args.quiet:
            override_str = ', '.join(f'{k}={v}' for k, v in config.items() if k != 'name')
            print(f"Config: {name} ({override_str})")

        controller = MPCController(
            Hp=Hp,
            weights=weights,
            use_smooth=True,
            forecast_mode='perfect',
            verbose=not args.quiet,
        )

        run_season(
            controller=controller,
            terrain=terrain,
            crop=crop,
            climate=climate,
            budget_total=budget_total,
            output_path=output_path,
            scenario_name=scenario,
            seed=0,
            force=args.force,
            verbose=not args.quiet,
        )


if __name__ == '__main__':
    main()
