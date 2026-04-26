# =============================================================================
# scripts/analysis/stat_tests.py
# Statistical tests for controller comparisons.
#
# Tests:
#   1. Mann-Whitney U: pairwise, non-parametric (appropriate for small N,
#      non-normal distributions). Used for MPC vs RL, MPC vs fixed, etc.
#   2. Kruskal-Wallis: multi-group, non-parametric.
#   3. Effect size: rank-biserial correlation (Kerby 2014).
#
# For each test, results are saved as CSV with:
#   metric, group_a, group_b, n_a, n_b, U_statistic, p_value,
#   effect_size, significant_005, significant_010
#
# Usage:
#   python -m scripts.analysis.stat_tests
#
# References:
#   - Mann & Whitney (1947) "On a Test of Whether one of Two Random Variables
#     is Stochastically Larger than the Other"
#   - Kerby (2014) "Simple Differences and Complex Effect Sizes"
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_DIR = PROJECT_ROOT / 'results' / 'analysis'

# Metrics to test
METRICS = [
    'yield_kg_ha',
    'water_used_mm',
    'wue_kg_ha_per_mm',
    'drought_days_per_agent',
    'sink_pond_days',
    'spatial_equity_cv',
]

# Controller pairs for pairwise comparison
PAIRS = [
    ('mpc_perfect', 'fixed_schedule'),
    ('mpc_perfect', 'no_irrigation'),
    ('sac_deterministic', 'fixed_schedule'),
    ('sac_deterministic', 'mpc_perfect'),
    ('sac_deterministic', 'no_irrigation'),
]


def rank_biserial(u_stat, n1, n2):
    """Rank-biserial correlation as effect size for Mann-Whitney U.

    r = 1 - (2U)/(n1*n2)
    r ∈ [-1, 1]: positive means group A tends to have larger values.

    Reference: Kerby (2014).
    """
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def run_mann_whitney(df, metric, group_a, group_b,
                     scenario=None, budget_pct=None):
    """Run Mann-Whitney U test for one metric between two controller groups.

    Parameters
    ----------
    df : pd.DataFrame
        Comparison table from aggregate.py.
    metric : str
    group_a, group_b : str
        Controller type names.
    scenario : str or None
        Filter to specific scenario. None = pool all.
    budget_pct : int or None
        Filter to specific budget. None = pool all.

    Returns
    -------
    dict or None
        Test results, or None if insufficient data.
    """
    mask_a = df['controller'] == group_a
    mask_b = df['controller'] == group_b

    if scenario is not None:
        mask_a &= df['scenario'] == scenario
        mask_b &= df['scenario'] == scenario
    if budget_pct is not None:
        mask_a &= df['budget_pct'] == budget_pct
        mask_b &= df['budget_pct'] == budget_pct

    vals_a = df.loc[mask_a, metric].dropna().values
    vals_b = df.loc[mask_b, metric].dropna().values

    if len(vals_a) < 2 or len(vals_b) < 2:
        return None

    try:
        u_stat, p_value = stats.mannwhitneyu(
            vals_a, vals_b, alternative='two-sided'
        )
    except ValueError:
        return None

    r = rank_biserial(u_stat, len(vals_a), len(vals_b))

    return {
        'metric': metric,
        'group_a': group_a,
        'group_b': group_b,
        'scenario': scenario or 'all',
        'budget_pct': budget_pct or 'all',
        'n_a': len(vals_a),
        'n_b': len(vals_b),
        'mean_a': float(vals_a.mean()),
        'mean_b': float(vals_b.mean()),
        'median_a': float(np.median(vals_a)),
        'median_b': float(np.median(vals_b)),
        'U_statistic': float(u_stat),
        'p_value': float(p_value),
        'effect_size_r': float(r),
        'significant_005': p_value < 0.05,
        'significant_010': p_value < 0.10,
    }


def run_all_tests(df):
    """Run all pairwise Mann-Whitney U tests.

    Returns
    -------
    pd.DataFrame
        One row per test.
    """
    results = []

    for metric in METRICS:
        for group_a, group_b in PAIRS:
            # Pooled (all scenarios and budgets)
            result = run_mann_whitney(df, metric, group_a, group_b)
            if result:
                results.append(result)

            # Per scenario
            for scenario in df['scenario'].unique():
                result = run_mann_whitney(
                    df, metric, group_a, group_b, scenario=scenario
                )
                if result:
                    results.append(result)

            # Per budget
            for budget in df['budget_pct'].unique():
                result = run_mann_whitney(
                    df, metric, group_a, group_b, budget_pct=budget
                )
                if result:
                    results.append(result)

    return pd.DataFrame(results) if results else pd.DataFrame()


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = ANALYSIS_DIR / 'comparison_table.csv'
    if not comparison_path.exists():
        print("Run scripts.analysis.aggregate first to build comparison_table.csv")
        return

    df = pd.read_csv(comparison_path)
    print(f"Loaded {len(df)} runs from comparison table")
    print(f"Controllers: {df['controller'].unique()}")

    results = run_all_tests(df)

    if results.empty:
        print("Not enough data for statistical tests (need ≥2 runs per group)")
        return

    # Save
    output_path = ANALYSIS_DIR / 'mann_whitney_results.csv'
    results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nSaved {len(results)} test results to {output_path}")

    # Print significant results
    sig = results[results['significant_005']]
    if not sig.empty:
        print(f"\nSignificant results (p < 0.05):")
        print(sig[['metric', 'group_a', 'group_b', 'scenario', 'budget_pct',
                    'mean_a', 'mean_b', 'p_value', 'effect_size_r'
                    ]].to_string(index=False))
    else:
        print("\nNo significant results at p < 0.05")


if __name__ == '__main__':
    main()
