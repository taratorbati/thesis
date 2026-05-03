# =============================================================================
# scripts/experiments/exp_weight_sensitivity.py
# Weight sensitivity analysis for the MPC cost function.
#
# Sweeps the four economically/agronomically grounded weights of the cost
# function (Section 4.1 of the controller-design report) across justified
# value ranges. Each sweep isolates ONE weight while holding the others at
# the nominal operating point:
#     α₁ = 1.0, α₂ = 0.01, α₃ = 0.1, α₄ = 0.5, α₅ = 0.005, α₆ = 0.0
# The α₂ baseline of 0.01 is kept here because it matches the 27 already-
# completed MPC runs at the nominal operating point. The α₂ sweep below
# explicitly tests the four water-pricing tiers that frame the thesis
# narrative on agricultural water economics.
#
# ── Sweep groups ──────────────────────────────────────────────────────────
#
# Group A — α₂ (water cost) on dry/100%/Hp=8  [4 runs]
#   Anchored to the four Iranian water-pricing tiers (toman/m³).
#   Conversion: α₂ ≈ (484 mm × 10 m³/(mm·ha) × price_per_m³) / rice_revenue,
#   where rice_revenue = yield_ref × price_rice = 4198 kg/ha × 500,000 toman/kg
#                     ≈ 2.099 × 10⁹ toman/ha.
#
#   Tier                    Price (toman/m³)   Water cost / revenue   α₂
#   ─────────────────────   ───────────────   ────────────────────   ──────
#   Subsidized agri (real)         175               0.040 %         0.0004
#   Domestic base                7,000               1.61  %         0.016
#   Domestic tier d (×2.8)      19,000               4.38  %         0.044
#   Industrial                  115,000              26.5  %         0.265
#
#   References:
#     - Iran Water Resources Management Co. tariff schedules (1402-1403 /
#       2023-2024). Domestic prices follow the step-tariff system with
#       coefficients (a, b, c, d, e) applied above 5 m³/month per family
#       member. Coefficient d = 2.8 corresponds to typical urban household
#       consumption in Tehran/Rasht.
#     - Mesgaran & Azadi (2018), "A national adaptation plan for water
#       scarcity in Iran," Stanford Iran 2040 Project, Working Paper 6.
#     - Nouri et al. (2023), "Water management dilemma in the agricultural
#       sector of Iran," Agricultural Water Management 278: 108162.
#
# Group B — α₃ (drought regularization) on dry/100%/Hp=8  [2 runs]
#   ±1 order of magnitude around the nominal α₃ = 0.1.
#   Justification: drought stress is internally accumulated through h3 in
#   the ABM; α₃ is the explicit MPC regularizer, so a ±1-decade sweep is
#   the standard sensitivity range for a regularization parameter
#   (Saltelli et al., 2004).
#
# Group C — α₄ (ponding penalty) on wet/100%/Hp=14  [2 runs]
#   1× and 4× and 10× the nominal α₄ = 0.5  →  {0.5 (existing), 2.0, 5.0}.
#   The Hp=14 wet/100% run already exists at α₄ = 0.5 (the 61-day
#   chronic-waterlogging anomaly), so only 2 additional runs are needed.
#   This sweep is run at Hp=14 specifically because the anomaly we are
#   trying to characterize manifests at Hp=14 — sweeping at Hp=8 would
#   not test the right thing. Fortunately wet/100%/Hp=14 solves in ~1 hour.
#   References:
#     - Setter et al. (1997), "Review of prospects for germplasm
#       improvement for waterlogging tolerance in wheat, barley and oats,"
#       Field Crops Research 51(1-2): 85-104. Establishes that the rice
#       waterlogging-tolerance threshold is shallow ponding for ≤2 days,
#       beyond which yield loss is rapid.
#
# Group D — α₆ (x1 > FC soft penalty) on dry/100%/Hp=8  [3 runs]
#   {0.1, 0.5, 2.0}. α₆ is a new term introduced in cost.py v2.3.
#   The dry scenario is chosen because the 21.6-waterlog-days anomaly
#   manifests there at Hp=14 (Issue 1). Hp=8 dry is chosen for the
#   sensitivity sweep because dry/Hp=8 also exhibits the symptom (1.9
#   waterlog days) at lower computational cost.
#
# Group E — α₆ validation at Hp=14  [1 run]
#   α₆ = 2.0 (the strongest level) on dry/100%/Hp=14, which was the worst
#   case for x1 > FC overshoot (21.6 waterlog days). This single
#   ~7.6-hour run definitively proves whether α₆ resolves Issue 1 at the
#   horizon length where it manifests.
#
# ── Total compute ─────────────────────────────────────────────────────────
#   Hp=8  runs:   4 (Group A) + 2 (Group B) + 3 (Group D) = 9 runs ≈ 9 h
#   Hp=14 runs:   2 (Group C) + 1 (Group E)               = 3 runs ≈ 9 h
#   Total: 12 runs / ~18 hours (split across multiple sessions if needed).
#
# ── Usage ─────────────────────────────────────────────────────────────────
#   python -m scripts.experiments.exp_weight_sensitivity              # all
#   python -m scripts.experiments.exp_weight_sensitivity --sweep a2
#   python -m scripts.experiments.exp_weight_sensitivity --sweep a4_wet
#   python -m scripts.experiments.exp_weight_sensitivity --sweep a6_validate
#   python -m scripts.experiments.exp_weight_sensitivity --force
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

# Nominal operating point (matches all 27 existing MPC runs)
NOMINAL = {
    'alpha1': 1.0,
    'alpha2': 0.01,
    'alpha3': 0.1,
    'alpha4': 0.5,
    'alpha5': 0.005,
    'alpha6': 0.0,
}

# ── Sweep configurations ──────────────────────────────────────────────────
# Each entry specifies:
#   key:            short identifier used in --sweep flag and filename
#   scenario:       'dry' | 'moderate' | 'wet'
#   budget_pct:     100 | 85 | 70
#   Hp:             8 | 14
#   weight_grid:    list of {'name', overrides...}
# ──────────────────────────────────────────────────────────────────────────

SWEEP_GROUPS = {

    # Group A — α₂ water-price sweep on dry/100%/Hp=8 (4 runs)
    'a2': {
        'scenario':   'dry',
        'budget_pct': 100,
        'Hp':         8,
        'weight_grid': [
            {'name': 'a2_agri_0p0004',     'alpha2': 0.0004},
            {'name': 'a2_domestic_0p016', 'alpha2': 0.016},
            {'name': 'a2_domesticD_0p044','alpha2': 0.044},
            {'name': 'a2_industrial_0p265','alpha2': 0.265},
        ],
    },

    # Group B — α₃ drought regularizer ±1 decade on dry/100%/Hp=8 (2 runs)
    'a3': {
        'scenario':   'dry',
        'budget_pct': 100,
        'Hp':         8,
        'weight_grid': [
            {'name': 'a3_0p03', 'alpha3': 0.03},
            {'name': 'a3_0p3',  'alpha3': 0.3},
        ],
    },

    # Group C — α₄ ponding penalty sweep on WET/100%/Hp=14 (2 runs)
    # The Hp=14 wet/100% run at α₄=0.5 already exists (the 61-day
    # waterlogging anomaly), so only 2 additional configs are needed.
    'a4_wet': {
        'scenario':   'wet',
        'budget_pct': 100,
        'Hp':         14,
        'weight_grid': [
            {'name': 'a4_2p0', 'alpha4': 2.0},
            {'name': 'a4_5p0', 'alpha4': 5.0},
        ],
    },

    # Group D — α₆ FC-overshoot penalty sweep on dry/100%/Hp=8 (3 runs)
    'a6': {
        'scenario':   'dry',
        'budget_pct': 100,
        'Hp':         8,
        'weight_grid': [
            {'name': 'a6_0p1', 'alpha6': 0.1},
            {'name': 'a6_0p5', 'alpha6': 0.5},
            {'name': 'a6_2p0', 'alpha6': 2.0},
        ],
    },

    # Group E — α₆ validation at Hp=14 on dry/100% (1 run, the strongest level)
    # This single run definitively proves Issue 1 (21.6-waterlog-day overshoot)
    # is resolved at the horizon where it manifests.
    'a6_validate': {
        'scenario':   'dry',
        'budget_pct': 100,
        'Hp':         14,
        'weight_grid': [
            {'name': 'a6_2p0_Hp14val', 'alpha6': 2.0},
        ],
    },
}


def run_sweep_group(group_key, group_spec, args, terrain, df_climate):
    """Run all configs within a single sweep group."""
    scenario   = group_spec['scenario']
    budget_pct = group_spec['budget_pct']
    Hp         = group_spec['Hp']

    crop    = get_crop('rice')
    climate = extract_scenario_by_name(df_climate, scenario, crop)
    climate['year'] = SCENARIO_YEARS[scenario]
    budget_total = 484.0 * (budget_pct / 100.0)

    if not args.quiet:
        print(f"\n{'='*72}")
        print(f"  SWEEP GROUP: {group_key.upper()}")
        print(f"  Scenario={scenario}  Budget={budget_pct}%  Hp={Hp}")
        print(f"  Configs: {len(group_spec['weight_grid'])}")
        print(f"{'='*72}")

    for config in group_spec['weight_grid']:
        name = config['name']

        # Build the weight dict by overriding NOMINAL with the active sweep parameter
        weights = dict(NOMINAL)
        for k, v in config.items():
            if k != 'name':
                weights[k] = v

        # Output filename pattern matches the existing convention used for
        # the 27 already-completed MPC runs, with a `_wsens_<name>` suffix.
        output_filename = (f"mpc_perfect_{scenario}_rice_{budget_pct}pct"
                           f"_Hp{Hp}_wsens_{name}.parquet")
        output_path = OUTPUT_DIR / output_filename

        if not args.quiet:
            override_str = ', '.join(
                f'{k}={v}' for k, v in config.items() if k != 'name'
            )
            print(f"\n  → Config '{name}' ({override_str})")
            print(f"    Output: {output_filename}")

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


def main():
    parser = argparse.ArgumentParser(
        description='Weight sensitivity analysis for MPC cost function.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--sweep',
        choices=list(SWEEP_GROUPS.keys()) + ['all'],
        default='all',
        help='Which sweep group to run. Default: all.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-run sweeps even if output files already exist.',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce log verbosity.',
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load shared resources once
    terrain    = load_terrain(str(DEM_PATH))
    df_climate = load_cleaned_data()

    if args.sweep == 'all':
        groups_to_run = list(SWEEP_GROUPS.keys())
    else:
        groups_to_run = [args.sweep]

    print(f"\nWeight Sensitivity Analysis (cost.py v2.3)")
    print(f"Nominal operating point: {NOMINAL}")
    print(f"Sweep groups to execute: {groups_to_run}")

    for group_key in groups_to_run:
        run_sweep_group(group_key, SWEEP_GROUPS[group_key],
                        args, terrain, df_climate)

    print(f"\n{'='*72}")
    print(f"  All requested sweeps completed.")
    print(f"{'='*72}\n")


if __name__ == '__main__':
    main()
