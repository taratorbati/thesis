# =============================================================================
# scripts/experiments/exp_fixed_schedule.py
# Run the fixed-schedule (linear-decay 19-event) baseline.
#
# Usage:
#   python -m scripts.experiments.exp_fixed_schedule --scenario dry --crop rice --budget 100
#   python -m scripts.experiments.exp_fixed_schedule --scenario all --crop rice --budget all
#
# Budget levels (as percent of full irrigation need):
#   100 — full irrigation need (rice: 484 mm, tobacco: 389 mm)
#    85 — moderate scarcity
#    70 — severe scarcity
# =============================================================================

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from climate_data import load_cleaned_data, extract_scenario_by_name, SCENARIO_YEARS
from soil_data import get_crop
from src.controllers.fixed_schedule import FixedScheduleController
from src.runner import run_season
from src.terrain import load_terrain


SCENARIOS_ALL = list(SCENARIO_YEARS.keys())
BUDGET_LEVELS = {100: 1.00, 85: 0.85, 70: 0.70}

# Full seasonal irrigation need per crop, in mm field-averaged.
# Derived from 25-year climatology in ARCHITECTURE.md Appendix B:
#   rice:    ETc = 1.15 × 5.02 × 93 = 537 mm; rainfall ≈ 53 → need ≈ 484 mm
#   tobacco: ETc ≈ 463 mm; rainfall ≈ 75 → need ≈ 389 mm
CROP_FULL_BUDGET_MM = {
    'rice':    484.0,
    'tobacco': 389.0,
}

DEM_PATH = PROJECT_ROOT / 'gilan_farm.tif'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'runs'


def main():
    parser = argparse.ArgumentParser(description='Run fixed-schedule baseline.')
    parser.add_argument('--scenario', choices=SCENARIOS_ALL + ['all'],
                        default='all')
    parser.add_argument('--crop', default='rice',
                        help="Crop name. Default: 'rice'.")
    parser.add_argument('--budget',
                        choices=[str(b) for b in BUDGET_LEVELS] + ['all'],
                        default='all',
                        help="Budget percentage (100, 85, 70) or 'all'.")
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios_to_run = SCENARIOS_ALL if args.scenario == 'all' else [args.scenario]
    if args.budget == 'all':
        budget_pcts = list(BUDGET_LEVELS.keys())
    else:
        budget_pcts = [int(args.budget)]

    crop = get_crop(args.crop)
    if args.crop.lower() not in CROP_FULL_BUDGET_MM:
        raise SystemExit(
            f"No full-budget value defined for crop '{args.crop}'. "
            f"Add it to CROP_FULL_BUDGET_MM in this script."
        )
    full_need_mm = CROP_FULL_BUDGET_MM[args.crop.lower()]

    terrain = load_terrain(str(DEM_PATH))
    df_climate = load_cleaned_data()

    print(f"Crop: {crop['name']} ({crop['season_days']} days)")
    print(f"Field: {terrain['N']} agents")
    print(f"Full irrigation need: {full_need_mm:.0f} mm field-averaged")
    print()

    for scenario in scenarios_to_run:
        year = SCENARIO_YEARS[scenario]
        climate = extract_scenario_by_name(df_climate, scenario, crop)
        climate['year'] = year

        for budget_pct in budget_pcts:
            multiplier = BUDGET_LEVELS[budget_pct]
            budget_total = full_need_mm * multiplier

            output_filename = (
                f"fixed_schedule_{scenario}_{args.crop}_{budget_pct}pct.parquet"
            )
            output_path = OUTPUT_DIR / output_filename

            controller = FixedScheduleController()

            print(f"{scenario}/{budget_pct}%: rainfall {climate['rainfall'].sum():.1f} mm, "
                  f"budget {budget_total:.1f} mm")

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
