# =============================================================================
# plot_budget_scenario_option1.py
#
# OPTION 1 for the budget x scenario breakdown: small multiples.
# One panel per climate scenario (dry / moderate / wet), grouped bar chart
# within each panel showing yield (% of ceiling) at each budget level
# (70/85/100%), for MPC, TD3 (Pool B), and SAC (Pool B).
#
# This shows how all controllers degrade together as the budget tightens,
# and whether that degradation pattern is uniform across climates (e.g. does
# the wet/OOD scenario behave differently under a tight budget than dry?).
#
# DATA SOURCE: results/runs/final results/ via defense_data_loader.py
# OUTPUT: results/analysis/defense_slides/budget_scenario_option1.png
# USAGE: python plot_budget_scenario_option1.py
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
COLORS = {"MPC": "#1e293b", "TD3 (Pool B)": "#16a34a", "SAC (Pool B)": "#d97706"}


def main():
    df = build_full_grid("perfect")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 18, "axes.labelsize": 14,
        "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14,
    })

    scen_titles = {"dry": "Dry (2022)", "moderate": "Wet (2018)", "wet": "Extreme wet (2024, OOD)"}
    x = np.arange(len(BUDGETS))
    width = 0.26

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3), sharey=True)

    for ax, scen in zip(axes, SCENARIOS):
        sub = df[df.scenario == scen]
        for i, ctrl in enumerate(CONTROLLERS):
            vals = [sub[(sub.controller == ctrl) & (sub.budget_pct == b)]["pct_ceiling"].values[0]
                    for b in BUDGETS]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=ctrl, color=COLORS[ctrl])
            ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}%" for b in BUDGETS])
        ax.set_xlabel("Water budget", fontsize=16)
        ax.set_title(scen_titles[scen])
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Yield (% of stress-free ceiling)", fontsize=18.5)
    axes[0].set_ylim(80, 101)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97), frameon=False)

    fig.suptitle("Yield vs. water budget, by climate scenario", fontsize=22, y=1.03)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "budget_scenario_option1.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
