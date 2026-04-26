# =============================================================================
# scripts/preprocess/03_precompute_thermal.py
# Populate the climate-only precomputation cache for the MPC.
#
# These quantities are deterministic functions of the climate scenario and
# the crop's thermal parameters. Computing them once and caching avoids
# embedding the operations in the MPC's IPOPT symbolic graph.
#
# Cached arrays per (scenario, crop):
#   h1, x2, h2, h7, g_base, Kc_ET
#
# Usage:
#   python -m scripts.preprocess.03_precompute_thermal
#   python -m scripts.preprocess.03_precompute_thermal --crop tobacco
#   python -m scripts.preprocess.03_precompute_thermal --scenario dry --force
#
# Output:
#   results/precomputed/precomputed_{scenario}_{crop}.npz
#   results/precomputed/precomputed_{scenario}_{crop}.json
# =============================================================================

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from climate_data import SCENARIO_YEARS
from src.precompute import compute_precomputed, save_precomputed, cache_path


SCENARIOS_ALL = list(SCENARIO_YEARS.keys())


def main():
    parser = argparse.ArgumentParser(
        description='Compute climate-only deterministic arrays for MPC use.'
    )
    parser.add_argument('--scenario', choices=SCENARIOS_ALL + ['all'],
                        default='all',
                        help="Scenario name or 'all'. Default: 'all'.")
    parser.add_argument('--crop', default='rice',
                        help="Crop name. Default: 'rice'.")
    parser.add_argument('--force', action='store_true',
                        help='Recompute and overwrite cached files.')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-scenario summary.')
    args = parser.parse_args()

    scenarios_to_run = SCENARIOS_ALL if args.scenario == 'all' else [args.scenario]

    if not args.quiet:
        print(f"Crop: {args.crop}")
        print(f"Scenarios: {scenarios_to_run}")
        print()

    for scenario in scenarios_to_run:
        path = cache_path(scenario, args.crop)
        if path.exists() and not args.force:
            if not args.quiet:
                print(f"  [skip] {path.name} (already exists; use --force to recompute)")
            continue

        pre = compute_precomputed(scenario, args.crop)
        save_precomputed(pre)

        if not args.quiet:
            print(f"  [done] {path.name}")
            print(f"           final x2:    {pre.x2[-1]:.0f} GDD "
                  f"(maturity threshold {pre.x2[-1] / 1250 * 100:.0f}% of theta18)")
            print(f"           mean h2:     {pre.h2.mean():.3f}  (1.0 = no heat stress)")
            print(f"           mean h7:     {pre.h7.mean():.3f}  (1.0 = no cold stress)")
            print(f"           total Kc·ET: {pre.Kc_ET.sum():.0f} mm")


if __name__ == '__main__':
    main()
