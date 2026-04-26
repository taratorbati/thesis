# =============================================================================
# scripts/analysis/aggregate.py
# Load all experiment results and produce a unified comparison table.
#
# Reads all .parquet + .json pairs from results/runs/, extracts final metrics,
# and produces:
#   1. results/analysis/comparison_table.csv — one row per run
#   2. results/analysis/comparison_table.parquet — same, parquet format
#   3. Printed summary table to stdout
#
# Usage:
#   python -m scripts.analysis.aggregate
# =============================================================================

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / 'results' / 'runs'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'analysis'


def load_all_runs(runs_dir=RUNS_DIR):
    """Load all completed runs and their metadata.

    Returns
    -------
    list of dict
        Each dict contains the parsed metadata + filename.
    """
    runs = []
    for json_path in sorted(runs_dir.glob('*.json')):
        # Skip partial checkpoints
        if '_partial' in json_path.stem:
            continue

        parquet_path = json_path.with_suffix('.parquet')
        if not parquet_path.exists():
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        meta['filename'] = parquet_path.name
        runs.append(meta)

    return runs


def parse_run_info(meta):
    """Extract structured fields from a run's metadata dict.

    Returns
    -------
    dict
        Flat dict suitable for a DataFrame row.
    """
    fm = meta.get('final_metrics', {})
    filename = meta.get('filename', '')

    # Parse controller type from filename
    controller = meta.get('controller', 'unknown')
    if 'no_irrigation' in filename:
        controller_type = 'no_irrigation'
    elif 'fixed_schedule' in filename:
        controller_type = 'fixed_schedule'
    elif 'mpc_perfect' in filename:
        controller_type = 'mpc_perfect'
    elif 'mpc_noisy' in filename:
        controller_type = 'mpc_noisy'
    elif 'sac_det' in filename:
        controller_type = 'sac_deterministic'
    elif 'sac_stoch' in filename:
        controller_type = 'sac_stochastic'
    else:
        controller_type = controller

    # Parse scenario from filename or metadata
    scenario = meta.get('scenario', 'unknown')

    # Parse budget percentage
    budget_total = meta.get('budget_total', 0)
    full_need = 484.0  # rice
    budget_pct = round(100 * budget_total / full_need) if full_need > 0 else 0

    # Parse horizon from filename
    horizon = 0
    if 'Hp8' in filename:
        horizon = 8
    elif 'Hp14' in filename:
        horizon = 14

    row = {
        'filename':                 filename,
        'controller':               controller_type,
        'scenario':                 scenario,
        'budget_pct':               budget_pct,
        'budget_total_mm':          budget_total,
        'horizon':                  horizon,
        'seed':                     meta.get('seed', 0),
        'yield_kg_ha':              fm.get('yield_kg_ha', 0),
        'water_used_mm':            fm.get('water_used_mm', 0),
        'wue_kg_ha_per_mm':         fm.get('wue_kg_ha_per_mm', 0),
        'budget_compliance':        fm.get('budget_compliance', 0),
        'drought_days_per_agent':   fm.get('drought_days_per_agent', 0),
        'waterlog_days_per_agent':  fm.get('waterlog_days_per_agent', 0),
        'sink_pond_days':           fm.get('sink_pond_days', 0),
        'sink_x5_max_mm':           fm.get('sink_x5_max_mm', 0),
        'spatial_equity_cv':        fm.get('spatial_equity_cv', 0),
        'wallclock_seconds':        meta.get('wallclock_seconds', 0),
        'solve_time_mean_ms':       meta.get('solve_time_mean_ms', 0),
        'solve_time_max_ms':        meta.get('solve_time_max_ms', 0),
        'terminal_biomass_g_m2':    fm.get('terminal_biomass_g_m2', 0),
    }
    return row


def build_comparison_table(runs_dir=RUNS_DIR):
    """Build the comparison DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    runs = load_all_runs(runs_dir)
    if not runs:
        print(f"No completed runs found in {runs_dir}")
        return pd.DataFrame()

    rows = [parse_run_info(m) for m in runs]
    df = pd.DataFrame(rows)

    # Sort for readability
    df = df.sort_values(
        ['controller', 'scenario', 'budget_pct', 'horizon', 'seed']
    ).reset_index(drop=True)

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_comparison_table()
    if df.empty:
        return

    # Save
    csv_path = OUTPUT_DIR / 'comparison_table.csv'
    parquet_path = OUTPUT_DIR / 'comparison_table.parquet'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    df.to_parquet(parquet_path, index=False)

    print(f"\nComparison table: {len(df)} runs")
    print(f"  Saved to: {csv_path}")
    print(f"  Saved to: {parquet_path}")

    # Print summary
    print(f"\n{'='*90}")
    summary_cols = ['controller', 'scenario', 'budget_pct', 'yield_kg_ha',
                    'water_used_mm', 'wue_kg_ha_per_mm', 'budget_compliance',
                    'wallclock_seconds']
    print(df[summary_cols].to_string(index=False))
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
