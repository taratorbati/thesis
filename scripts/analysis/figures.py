# =============================================================================
# scripts/analysis/figures.py
# Generate thesis figures from the comparison table and raw trajectories.
#
# Figures:
#   1. fig_yield_comparison.pdf     — Grouped bar: yield by controller × scenario
#   2. fig_wue_comparison.pdf       — Grouped bar: WUE by controller × budget
#   3. fig_yield_heatmap.pdf        — Heatmap: yield across scenario × budget
#   4. fig_trajectory_x1.pdf        — Time series: field-mean soil water
#   5. fig_trajectory_x4.pdf        — Time series: field-mean biomass
#   6. fig_spatial_yield.pdf        — Spatial map: terminal biomass per agent
#   7. fig_budget_usage.pdf         — Budget spending curves
#   8. fig_weight_sensitivity.pdf   — α2 sweep results
#   9. fig_solve_time.pdf           — MPC solve time distribution
#  10. fig_reward_curve.pdf         — RL training reward curve (if available)
#
# Usage:
#   python -m scripts.analysis.figures
#
# All figures use matplotlib with the 'seaborn-v0_8-paper' style for
# publication quality. Font sizes are thesis-appropriate (12pt base).
# =============================================================================

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.persistence import load_run, long_df_to_trajectory

ANALYSIS_DIR = PROJECT_ROOT / 'results' / 'analysis'
FIGURES_DIR = ANALYSIS_DIR / 'figures'
RUNS_DIR = PROJECT_ROOT / 'results' / 'runs'

# Plot style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette (colorblind-safe)
COLORS = {
    'no_irrigation':      '#999999',
    'fixed_schedule':     '#E69F00',
    'mpc_perfect':        '#0072B2',
    'mpc_noisy':          '#56B4E9',
    'sac_deterministic':  '#D55E00',
    'sac_stochastic':     '#CC79A7',
}

LABELS = {
    'no_irrigation':      'No Irrigation',
    'fixed_schedule':     'Fixed Schedule',
    'mpc_perfect':        'MPC (Perfect)',
    'mpc_noisy':          'MPC (Noisy)',
    'sac_deterministic':  'SAC (Det.)',
    'sac_stochastic':     'SAC (Stoch.)',
}


def load_comparison_table():
    """Load the aggregated comparison table."""
    path = ANALYSIS_DIR / 'comparison_table.csv'
    if not path.exists():
        raise FileNotFoundError(
            "Run scripts.analysis.aggregate first to build comparison_table.csv"
        )
    return pd.read_csv(path)


# ── Figure 1: Yield comparison bar chart ──────────────────────────────────────

def fig_yield_comparison(df):
    """Grouped bar chart: yield by controller for each scenario."""
    scenarios = ['dry', 'moderate', 'wet']
    # Filter to 100% budget only for clean comparison
    df_100 = df[df['budget_pct'] == 100]

    controllers = [c for c in COLORS if c in df_100['controller'].unique()]
    if not controllers:
        print("  Skipping fig_yield_comparison (no data)")
        return

    x = np.arange(len(scenarios))
    width = 0.8 / len(controllers)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ctrl in enumerate(controllers):
        vals = []
        for scen in scenarios:
            subset = df_100[(df_100['controller'] == ctrl) &
                           (df_100['scenario'] == scen)]
            vals.append(subset['yield_kg_ha'].mean() if len(subset) > 0 else 0)
        ax.bar(x + i * width, vals, width,
               label=LABELS.get(ctrl, ctrl),
               color=COLORS.get(ctrl, '#333333'))

    ax.set_xlabel('Climate Scenario')
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_title('Yield Comparison Across Controllers (100% Budget)')
    ax.set_xticks(x + width * (len(controllers) - 1) / 2)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(FIGURES_DIR / 'fig_yield_comparison.pdf')
    fig.savefig(FIGURES_DIR / 'fig_yield_comparison.png')
    plt.close(fig)
    print("  Saved fig_yield_comparison")


# ── Figure 2: WUE comparison ─────────────────────────────────────────────────

def fig_wue_comparison(df):
    """Grouped bar: WUE by controller × budget."""
    budgets = [100, 85, 70]
    # Pool across scenarios
    controllers = [c for c in COLORS if c in df['controller'].unique()]
    if not controllers:
        return

    x = np.arange(len(budgets))
    width = 0.8 / len(controllers)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ctrl in enumerate(controllers):
        vals = []
        for b in budgets:
            subset = df[(df['controller'] == ctrl) & (df['budget_pct'] == b)]
            vals.append(subset['wue_kg_ha_per_mm'].mean() if len(subset) > 0 else 0)
        ax.bar(x + i * width, vals, width,
               label=LABELS.get(ctrl, ctrl),
               color=COLORS.get(ctrl, '#333333'))

    ax.set_xlabel('Budget Level (%)')
    ax.set_ylabel('WUE (kg/ha per mm)')
    ax.set_title('Water Use Efficiency by Budget Level')
    ax.set_xticks(x + width * (len(controllers) - 1) / 2)
    ax.set_xticklabels([f'{b}%' for b in budgets])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(FIGURES_DIR / 'fig_wue_comparison.pdf')
    fig.savefig(FIGURES_DIR / 'fig_wue_comparison.png')
    plt.close(fig)
    print("  Saved fig_wue_comparison")


# ── Figure 3: Yield heatmap ──────────────────────────────────────────────────

def fig_yield_heatmap(df):
    """Heatmap of yield: rows=controller, cols=scenario×budget."""
    controllers = [c for c in COLORS if c in df['controller'].unique()]
    scenarios = ['dry', 'moderate', 'wet']
    budgets = [100, 85, 70]

    cols = [f'{s}/{b}%' for s in scenarios for b in budgets]
    data = np.zeros((len(controllers), len(cols)))

    for i, ctrl in enumerate(controllers):
        for j, (s, b) in enumerate([(s, b) for s in scenarios for b in budgets]):
            subset = df[(df['controller'] == ctrl) &
                       (df['scenario'] == s) & (df['budget_pct'] == b)]
            data[i, j] = subset['yield_kg_ha'].mean() if len(subset) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(data, cmap='YlGn', aspect='auto')

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(controllers)))
    ax.set_yticklabels([LABELS.get(c, c) for c in controllers])

    # Annotate cells
    for i in range(len(controllers)):
        for j in range(len(cols)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i, j]:.0f}', ha='center', va='center',
                       fontsize=9, color='black' if data[i, j] > 2500 else 'white')

    ax.set_title('Yield (kg/ha) by Controller × Scenario × Budget')
    fig.colorbar(im, ax=ax, label='Yield (kg/ha)')

    fig.savefig(FIGURES_DIR / 'fig_yield_heatmap.pdf')
    fig.savefig(FIGURES_DIR / 'fig_yield_heatmap.png')
    plt.close(fig)
    print("  Saved fig_yield_heatmap")


# ── Figure 9: MPC solve time distribution ────────────────────────────────────

def fig_solve_time(df):
    """Box plot of MPC solve times."""
    mpc_runs = df[df['controller'].str.contains('mpc')]
    if mpc_runs.empty:
        return

    # Load actual solve times from JSON sidecars
    all_times = {}
    for _, row in mpc_runs.iterrows():
        json_path = RUNS_DIR / row['filename'].replace('.parquet', '.json')
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            times = meta.get('solve_times', [])
            if times:
                label = f"{row['scenario']}/{row['budget_pct']}%"
                all_times[label] = [t / 1000 for t in times]  # ms → s

    if not all_times:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(all_times.keys())
    data = [all_times[l] for l in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS.get('mpc_perfect', '#0072B2'))
        patch.set_alpha(0.6)

    ax.set_ylabel('Solve Time (seconds)')
    ax.set_title('IPOPT Solve Time Distribution')
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(FIGURES_DIR / 'fig_solve_time.pdf')
    fig.savefig(FIGURES_DIR / 'fig_solve_time.png')
    plt.close(fig)
    print("  Saved fig_solve_time")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = load_comparison_table()
    except FileNotFoundError as e:
        print(str(e))
        return

    print(f"Generating figures from {len(df)} runs...\n")

    fig_yield_comparison(df)
    fig_wue_comparison(df)
    fig_yield_heatmap(df)
    fig_solve_time(df)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
