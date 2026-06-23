# =============================================================================
# plot_budget_scenario_option2.py
#
# OPTION 2 for the budget x scenario breakdown: heatmap grid.
# Rows = controller, columns = (scenario, budget) pairs, cell color/value =
# yield as % of stress-free ceiling. More compact than Option 1's small
# multiples, shows the full 3 controllers x 3 scenarios x 3 budgets pattern
# in one glance, but loses the bar-chart's intuitive magnitude comparison.
#
# DATA SOURCE: results/runs/final results/ via defense_data_loader.py
# OUTPUT: results/analysis/defense_slides/budget_scenario_option2.png
# USAGE: python plot_budget_scenario_option2.py
# Requires: numpy, pandas, pyarrow, matplotlib, defense_data_loader.py
# =============================================================================

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from defense_data_loader import build_full_grid, SCENARIOS, BUDGETS  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "defense_slides"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTROLLERS = ["MPC", "TD3 (Pool B)", "SAC (Pool B)"]
SCEN_LABELS = {"dry": "Dry", "moderate": "Mod.", "wet": "Wet"}


def main():
    df = build_full_grid("perfect")

    # Build matrix: rows = controllers, columns = (scenario, budget) in a
    # fixed order grouped by scenario.
    col_keys = [(s, b) for s in SCENARIOS for b in BUDGETS]
    col_labels = [f"{SCEN_LABELS[s]}\n{b}%" for s, b in col_keys]

    matrix = np.zeros((len(CONTROLLERS), len(col_keys)))
    for i, ctrl in enumerate(CONTROLLERS):
        for j, (scen, budget) in enumerate(col_keys):
            val = df[(df.controller == ctrl) & (df.scenario == scen) & (df.budget_pct == budget)]["pct_ceiling"]
            matrix[i, j] = val.values[0] if len(val) else np.nan

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12.5, "axes.titlesize": 16, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 13,
    })

    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=80, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(col_keys)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(CONTROLLERS)))
    ax.set_yticklabels(CONTROLLERS)

    # Light separators between the 3 scenario groups (every 3 columns).
    for g in (2.5, 5.5):
        ax.axvline(g, color="white", lw=2.5)

    for i in range(len(CONTROLLERS)):
        for j in range(len(col_keys)):
            val = matrix[i, j]
            text_color = "white" if val < 88 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=12, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Yield (% of stress-free ceiling)")

    ax.set_title("Yield vs. water budget, by climate scenario and controller")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "budget_scenario_option2.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
