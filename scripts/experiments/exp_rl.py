# =============================================================================
# scripts/experiments/exp_rl.py
# Train and/or evaluate SAC agents for the irrigation task.
#
# Usage:
#   # Smoke test (validates env + CTDE policy, ~2 min on GPU):
#   python -m scripts.experiments.exp_rl --mode train --seed 0 --timesteps 10000
#
#   # Full training, seed 0 (run 5 notebooks in parallel with seeds 0-4):
#   python -m scripts.experiments.exp_rl --mode train --seed 0 --timesteps 500000
#
#   # Resume an interrupted session:
#   python -m scripts.experiments.exp_rl --mode train --seed 0 --timesteps 500000 \
#       --resume results/rl/sac_general_seed0/checkpoints/sac_general_seed0_300000_steps.zip
#
#   # Evaluate trained model on all 9 held-out cells:
#   python -m scripts.experiments.exp_rl --mode eval \
#       --model results/rl/sac_general_seed0/best_model/best_model.zip \
#       --scenario all --budget all
#
# Training design: one general policy trained on 23 years x U(70-100%) budget.
# Scenarios/budgets are NOT passed during training — the env randomizes them.
# --scenario and --budget args are only used in eval mode.
#
# Seeds: 5 seeds (0-4) for conference-paper-standard statistical robustness.
# =============================================================================

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from climate_data import SCENARIO_YEARS
from soil_data import get_crop
from src.terrain import load_terrain

SCENARIOS_ALL = list(SCENARIO_YEARS.keys())   # ['dry', 'moderate', 'wet']
BUDGET_LEVELS = {100: 1.00, 85: 0.85, 70: 0.70}
CROP_FULL_BUDGET_MM = {'rice': 484.0}

# 5 seeds for statistical robustness (conference-paper standard).
# Run each seed in a separate Kaggle GPU notebook simultaneously.
SEEDS = [0, 1, 2, 3, 4]

DEM_PATH       = PROJECT_ROOT / 'gilan_farm.tif'
RL_OUTPUT_DIR  = PROJECT_ROOT / 'results' / 'rl'
RUNS_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'runs'


def run_training(args):
    """Train one general SAC policy (year+budget randomized per episode)."""
    from src.rl.train import train_sac

    seeds = SEEDS if args.seed == 'all' else [int(args.seed)]

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training: general policy  seed={seed}")
        print(f"{'='*60}\n")

        # NOTE: scenario and budget are NOT passed — the env samples them
        # randomly each episode from TRAINING_YEARS and U(70%, 100%).
        train_sac(
            total_timesteps=args.timesteps,
            seed=seed,
            output_dir=str(RL_OUTPUT_DIR),
            dem_path=str(DEM_PATH),
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
            resume_path=args.resume,
            verbose=1,
        )


def run_evaluation(args):
    """Evaluate a trained model on the 9 held-out (scenario x budget) cells."""
    from src.rl.runner import RLController
    from src.runner import run_season
    from climate_data import load_cleaned_data, extract_scenario_by_name

    if args.model is None:
        raise SystemExit("--model is required for eval mode")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    crop    = get_crop('rice')
    terrain = load_terrain(str(DEM_PATH))
    df_climate  = load_cleaned_data()
    full_need_mm = CROP_FULL_BUDGET_MM['rice']

    scenarios   = SCENARIOS_ALL if args.scenario == 'all' else [args.scenario]
    budget_pcts = list(BUDGET_LEVELS.keys()) if args.budget == 'all' else [int(args.budget)]

    # Extract seed from model path
    seed_str = '0'
    for part in model_path.parts:
        if 'seed' in part:
            seed_str = part.split('seed')[-1].split('_')[0]
            break

    for scenario in scenarios:
        climate = extract_scenario_by_name(df_climate, scenario, crop)
        climate['year'] = SCENARIO_YEARS[scenario]

        for budget_pct in budget_pcts:
            budget_total = full_need_mm * BUDGET_LEVELS[budget_pct]

            output_filename = (
                f"sac_{'det' if args.deterministic else 'stoch'}"
                f"_{scenario}_rice_{budget_pct}pct_seed{seed_str}.parquet"
            )
            output_path = RUNS_OUTPUT_DIR / output_filename

            controller = RLController(
                model_path=str(model_path),
                deterministic=args.deterministic,
                verbose=True,
            )

            print(f"\nEvaluating: {scenario}/{budget_pct}% (model: {model_path.name})")

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
        description='Train or evaluate SAC irrigation agent.'
    )
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--scenario',
                        choices=SCENARIOS_ALL + ['all'], default='all',
                        help='Eval scenario (eval mode only). Ignored during training.')
    parser.add_argument('--budget',
                        choices=[str(b) for b in BUDGET_LEVELS] + ['all'],
                        default='all',
                        help='Budget level (eval mode only). Ignored during training.')
    parser.add_argument('--seed', default='0',
                        help="Seed (int) or 'all' for SEEDS=[0,1,2,3,4].")
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--checkpoint-freq', type=int, default=50_000)
    parser.add_argument('--eval-freq',       type=int, default=25_000)
    parser.add_argument('--resume', default=None,
                        help='Path to .zip checkpoint to resume from.')
    parser.add_argument('--model', default=None,
                        help='Path to trained model .zip (eval mode).')
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.stochastic:
        args.deterministic = False

    if args.mode == 'train':
        run_training(args)
    else:
        run_evaluation(args)


if __name__ == '__main__':
    main()
