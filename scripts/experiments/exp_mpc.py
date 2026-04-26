# =============================================================================
# scripts/experiments/exp_mpc.py
# Run MPC (CasADi + IPOPT) for a given scenario, budget, and horizon.
#
# Usage:
#   python -m scripts.experiments.exp_mpc --scenario dry --budget 100 --horizon 8
#   python -m scripts.experiments.exp_mpc --scenario dry --budget 100 --horizon 8 --forecast noisy --noise-seed 42
#   python -m scripts.experiments.exp_mpc --scenario all --budget all --horizon 8
#
# Prerequisites:
#   - preprocess.py has been run (climate CSV exists)
#   - scripts.preprocess.03_precompute_thermal has been run (cached thermal arrays)
#   - pip install casadi
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


SCENARIOS_ALL = list(SCENARIO_YEARS.keys())
BUDGET_LEVELS = {100: 1.00, 85: 0.85, 70: 0.70}

CROP_FULL_BUDGET_MM = {
    'rice':    484.0,
    'tobacco': 389.0,
}

DEM_PATH = PROJECT_ROOT / 'gilan_farm.tif'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'runs'


def main():
    parser = argparse.ArgumentParser(description='Run MPC irrigation controller.')
    parser.add_argument('--scenario', choices=SCENARIOS_ALL + ['all'],
                        default='dry',
                        help="Scenario name or 'all'. Default: 'dry'.")
    parser.add_argument('--crop', default='rice',
                        help="Crop name. Default: 'rice'.")
    parser.add_argument('--budget',
                        choices=[str(b) for b in BUDGET_LEVELS] + ['all'],
                        default='100',
                        help="Budget percentage or 'all'. Default: '100'.")
    parser.add_argument('--horizon', type=int, default=8,
                        help='Prediction horizon Hp. Default: 8.')
    parser.add_argument('--forecast', choices=['perfect', 'noisy'],
                        default='perfect',
                        help="Forecast mode. Default: 'perfect'.")
    parser.add_argument('--noise-seed', type=int, default=None,
                        help='Seed for noisy forecast. Default: None.')
    parser.add_argument('--alpha2', type=float, default=None,
                        help='Override alpha2 weight. Default: use ARCHITECTURE.md value.')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable smooth approximations (use CasADi native fmax/fmin).')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files.')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios_to_run = SCENARIOS_ALL if args.scenario == 'all' else [args.scenario]
    budget_pcts = list(BUDGET_LEVELS.keys()) if args.budget == 'all' else [int(args.budget)]

    crop = get_crop(args.crop)
    if args.crop.lower() not in CROP_FULL_BUDGET_MM:
        raise SystemExit(f"No full-budget value defined for crop '{args.crop}'.")
    full_need_mm = CROP_FULL_BUDGET_MM[args.crop.lower()]

    terrain = load_terrain(str(DEM_PATH))
    df_climate = load_cleaned_data()

    # Build weights override
    weights = None
    if args.alpha2 is not None:
        weights = {'alpha2': args.alpha2}

    if not args.quiet:
        print(f"MPC Configuration:")
        print(f"  Crop: {crop['name']} ({crop['season_days']} days)")
        print(f"  Field: {terrain['N']} agents")
        print(f"  Horizon: Hp={args.horizon}")
        print(f"  Forecast: {args.forecast}")
        print(f"  Smooth: {not args.no_smooth}")
        if weights:
            print(f"  Weight overrides: {weights}")
        print()

    for scenario in scenarios_to_run:
        year = SCENARIO_YEARS[scenario]
        climate = extract_scenario_by_name(df_climate, scenario, crop)
        climate['year'] = year

        for budget_pct in budget_pcts:
            multiplier = BUDGET_LEVELS[budget_pct]
            budget_total = full_need_mm * multiplier

            # Build output filename
            parts = [f"mpc_{args.forecast}", scenario, args.crop,
                     f"{budget_pct}pct", f"Hp{args.horizon}"]
            if args.noise_seed is not None:
                parts.append(f"seed{args.noise_seed}")
            if args.alpha2 is not None:
                parts.append(f"a2_{args.alpha2}")
            output_filename = '_'.join(parts) + '.parquet'
            output_path = OUTPUT_DIR / output_filename

            # Create controller (NLP is built inside reset(), called by run_season)
            controller = MPCController(
                Hp=args.horizon,
                weights=weights,
                use_smooth=not args.no_smooth,
                forecast_mode=args.forecast,
                noise_sigma=0.15,
                noise_seed=args.noise_seed,
                verbose=not args.quiet,
            )

            if not args.quiet:
                print(f"Scenario: {scenario}/{budget_pct}%  "
                      f"rainfall={climate['rainfall'].sum():.1f}mm  "
                      f"budget={budget_total:.1f}mm")

            run_season(
                controller=controller,
                terrain=terrain,
                crop=crop,
                climate=climate,
                budget_total=budget_total,
                output_path=output_path,
                scenario_name=scenario,
                seed=args.noise_seed or 0,
                force=args.force,
                verbose=not args.quiet,
            )


if __name__ == '__main__':
    main()
