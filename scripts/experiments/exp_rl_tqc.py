# =============================================================================
# scripts/experiments/exp_rl_tqc.py    v2.10.0
#
# Evaluation script for TQC v2.10 checkpoints (E2 and later).
# Mirrors scripts/experiments/exp_rl.py --mode eval but loads via TQCRLController
# instead of RLController, since SAC.load cannot load a TQC checkpoint.
#
# The output parquet schema and filenames are IDENTICAL to exp_rl.py so
# downstream analysis scripts (comparison plots, thesis tables) work without
# modification.  Only the algorithm prefix in the filename changes:
#   exp_rl.py    : sac_perfect_det_dry_rice_100pct_seed0.parquet
#   exp_rl_tqc   : tqc_perfect_det_dry_rice_100pct_seed0.parquet
#
# Usage:
#   # Perfect forecast (primary thesis numbers):
#   python -m scripts.experiments.exp_rl_tqc \
#       --model results/rl/sac_v210_e2_seed0/best_model/best_model.zip \
#       --scenario all --budget all --forecast perfect
#
#   # Noisy forecast (Chapter 5 robustness analysis):
#   python -m scripts.experiments.exp_rl_tqc \
#       --model results/rl/sac_v210_e2_seed0/best_model/best_model.zip \
#       --scenario all --budget all --forecast noisy --noise-seed 42
#
#   # Single cell:
#   python -m scripts.experiments.exp_rl_tqc \
#       --model .../best_model.zip --scenario wet --budget 100
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from climate_data import SCENARIO_YEARS
from soil_data import get_crop
from src.terrain import load_terrain

SCENARIOS_ALL       = list(SCENARIO_YEARS.keys())            # ['dry', 'moderate', 'wet']
BUDGET_LEVELS       = {100: 1.00, 85: 0.85, 70: 0.70}
CROP_FULL_BUDGET_MM = {'rice': 484.0}

DEM_PATH         = PROJECT_ROOT / 'gilan_farm.tif'
RUNS_OUTPUT_DIR  = PROJECT_ROOT / 'results' / 'runs'


def run_evaluation(args):
    """Evaluate a trained TQC model on the 9 held-out (scenario x budget) cells.

    Identical control flow to scripts/experiments/exp_rl.py run_evaluation,
    except:
      - imports TQCRLController instead of RLController
      - output filename starts with 'tqc_' instead of 'sac_'
    All other behaviour (output schema, noise generation, forcing logic)
    matches the existing eval pipeline so downstream parquet readers
    keep working.
    """
    from src.rl.runner_tqc import TQCRLController
    from src.runner import run_season
    from climate_data import load_cleaned_data, extract_scenario_by_name

    if args.model is None:
        raise SystemExit("--model is required")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    crop         = get_crop('rice')
    terrain      = load_terrain(str(DEM_PATH))
    df_climate   = load_cleaned_data()
    full_need_mm = CROP_FULL_BUDGET_MM['rice']

    scenarios   = SCENARIOS_ALL if args.scenario == 'all' else [args.scenario]
    budget_pcts = list(BUDGET_LEVELS.keys()) if args.budget == 'all' else [int(args.budget)]

    # Extract seed number from model path for output filename.
    # Convention: .../sac_v210_e2_seed{N}/best_model/best_model.zip
    seed_str = '0'
    for part in model_path.parts:
        if 'seed' in part:
            seed_str = part.split('seed')[-1].split('_')[0]
            break

    forecast_mode = args.forecast
    noise_seed    = args.noise_seed

    RUNS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTQC Evaluation config:")
    print(f"  Model:         {model_path}")
    print(f"  Scenarios:     {scenarios}")
    print(f"  Budgets:       {budget_pcts}")
    print(f"  Forecast mode: {forecast_mode}")
    if forecast_mode == 'noisy':
        print(f"  Noise seed:    {noise_seed}")
        print(f"  Noise sigma:   {args.noise_sigma}")
        print(f"  Noise rho:     {args.noise_rho}")
    print()

    for scenario in scenarios:
        climate = extract_scenario_by_name(df_climate, scenario, crop)
        climate['year'] = SCENARIO_YEARS[scenario]

        for budget_pct in budget_pcts:
            budget_total = full_need_mm * BUDGET_LEVELS[budget_pct]

            policy_tag = 'det' if args.deterministic else 'stoch'
            if forecast_mode == 'noisy':
                noise_tag = f'_ns{noise_seed}' if noise_seed is not None else '_noisy'
                fc_tag    = f'noisy{noise_tag}'
            else:
                fc_tag = 'perfect'

            # Filename matches exp_rl.py convention with 'tqc_' prefix.
            output_filename = (
                f"tqc_{fc_tag}_{policy_tag}"
                f"_{scenario}_rice_{budget_pct}pct_seed{seed_str}.parquet"
            )
            output_path = RUNS_OUTPUT_DIR / output_filename

            if output_path.exists() and not args.force:
                print(f"  Skipping {output_filename} "
                      f"(already exists; use --force to overwrite)")
                continue

            controller = TQCRLController(
                model_path=str(model_path),
                deterministic=args.deterministic,
                forecast_mode=forecast_mode,
                noise_sigma=args.noise_sigma,
                noise_rho=args.noise_rho,
                noise_seed=noise_seed,
                verbose=True,
            )

            print(f"Evaluating: {scenario}/{budget_pct}%  "
                  f"forecast={forecast_mode}  "
                  f"(model: {model_path.name})")

            run_season(
                controller=controller,
                terrain=terrain,
                crop=crop,
                climate=climate,
                budget_total=budget_total,
                output_path=output_path,
                scenario_name=scenario,
                seed=int(seed_str),
                force=args.force,
                verbose=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained TQC v2.10 irrigation agent on the 9-cell test grid.'
    )

    parser.add_argument('--model', required=True,
                        help='Path to trained TQC model .zip')
    parser.add_argument('--scenario',
                        choices=SCENARIOS_ALL + ['all'], default='all',
                        help='Eval scenario or "all".')
    parser.add_argument('--budget',
                        choices=[str(b) for b in BUDGET_LEVELS] + ['all'],
                        default='all',
                        help='Budget level or "all".')

    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic policy (default).')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy.')

    parser.add_argument(
        '--forecast', choices=['perfect', 'noisy'], default='perfect',
        help=("Forecast mode for evaluation. 'perfect' (default) is the "
              "primary thesis number; 'noisy' is Chapter 5 robustness.")
    )
    parser.add_argument(
        '--noise-seed', type=int, default=42,
        help='RNG seed for NoisyForecast. Default 42 (matches exp_mpc.py).'
    )
    parser.add_argument(
        '--noise-sigma', type=float, default=0.15,
        help='Base noise sigma for NoisyForecast. Default 0.15.'
    )
    parser.add_argument(
        '--noise-rho', type=float, default=0.6,
        help='AR(1) persistence for NoisyForecast. Default 0.6.'
    )

    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files.')

    args = parser.parse_args()
    if args.stochastic:
        args.deterministic = False

    run_evaluation(args)


if __name__ == '__main__':
    main()
