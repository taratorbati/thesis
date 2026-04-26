# =============================================================================
# scripts/experiments/exp_no_irrigation.py
# Run the no-irrigation baseline for a given scenario and crop.
#
# Usage:
#   python -m scripts.experiments.exp_no_irrigation --scenario dry --crop rice
#   python -m scripts.experiments.exp_no_irrigation --scenario all --crop rice
# =============================================================================

import argparse
import sys
from pathlib import Path

# Make the project root importable when running as `python -m scripts.experiments...`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from climate_data import load_cleaned_data, extract_scenario_by_name, SCENARIO_YEARS
from soil_data import get_crop
from src.controllers.no_irrigation import NoIrrigationController
from src.runner import run_season
from src.terrain import load_terrain


SCENARIOS_ALL = list(SCENARIO_YEARS.keys())  # ['dry', 'moderate', 'wet']
DEM_PATH = PROJECT_ROOT / 'gilan_farm.tif'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'runs'


def main():
    parser = argparse.ArgumentParser(description='Run no-irrigation baseline.')
    parser.add_argument('--scenario', choices=SCENARIOS_ALL + ['all'],
                        default='all',
                        help="Scenario name or 'all'. Default: 'all'.")
    parser.add_argument('--crop', default='rice',
                        help="Crop name. Default: 'rice'.")
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files.')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-run output.')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios_to_run = SCENARIOS_ALL if args.scenario == 'all' else [args.scenario]

    crop = get_crop(args.crop)
    terrain = load_terrain(str(DEM_PATH))
    df_climate = load_cleaned_data()

    print(f"Crop: {crop['name']} ({crop['season_days']} days)")
    print(f"Field: {terrain['N']} agents, {terrain['rows']}×{terrain['cols']} grid")
    print(f"Sinks: {sum(1 for n in range(terrain['N']) if terrain['Nr'][n] == 0)} agents")
    print()

    for scenario in scenarios_to_run:
        year = SCENARIO_YEARS[scenario]
        climate = extract_scenario_by_name(df_climate, scenario, crop)
        climate['year'] = year

        output_filename = f"no_irrigation_{scenario}_{args.crop}.parquet"
        output_path = OUTPUT_DIR / output_filename

        controller = NoIrrigationController()

        print(f"Scenario: {scenario} ({year}), rainfall total: "
              f"{climate['rainfall'].sum():.1f} mm")

        # Budget is irrelevant for no-irrigation, but we pass 0 for cleanliness.
        run_season(
            controller=controller,
            terrain=terrain,
            crop=crop,
            climate=climate,
            budget_total=0.0,
            output_path=output_path,
            scenario_name=scenario,
            seed=0,
            force=args.force,
            verbose=not args.quiet,
        )


if __name__ == '__main__':
    main()
