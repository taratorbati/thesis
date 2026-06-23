# =============================================================================
# plot_r2_waterlog_efficiency.py
#
# SLIDE R2: "TD3 wins where it matters" -- MPC vs TD3 head-to-head on
# waterlog-days and water-use efficiency, the two metrics where TD3 beats
# the oracle MPC controller despite MPC having a near-yield-ceiling advantage
# elsewhere.
#
# Produces a two-panel grouped bar chart:
#   Left panel:  mean waterlog-days/agent, MPC vs TD3 (Pool B, 3-seed mean)
#   Right panel: water-use efficiency (kg yield per mm water applied),
#                MPC vs TD3
# Both panels show all 3 climate scenarios as grouped bars (budget averaged
# across the 3 budget levels, since R2's point is "the pattern holds across
# climates", not a budget-sensitivity story -- that's the separate R-budget
# slide).
#
# Only MPC and TD3 are shown (no SAC) -- this slide's whole point is the
# oracle-vs-deployable comparison; adding a third controller dilutes it.
#
# DATA SOURCE: results/runs/final results/ via defense_data_loader.py
# (must be in the same directory, or importable from PYTHONPATH).
#
# OUTPUT: results/analysis/defense_slides/r2_waterlog_efficiency.png
#
# USAGE: python plot_r2_waterlog_efficiency.py
# Requires: numpy, pandas, pyarrow, matplotlib, defense_data_loader.py
# =============================================================================

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from defense_data_loader import build_full_grid, SCENARIOS  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "defense_slides"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = build_full_grid("perfect")
    df["wue_kg_ha_per_mm"] = df["yield_kg_ha"] / df["water_used_mm"]

    # Average across the 3 budget levels for each (controller, scenario).
    summary = (
        df[df["controller"].isin(["MPC", "TD3 (Pool B)"])]
        .groupby(["controller", "scenario"])
        .agg(waterlog_days=("waterlog_days_per_agent", "mean"),
             wue=("wue_kg_ha_per_mm", "mean"))
        .reset_index()
    )
    print(summary.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
    })

    scen_labels = {"dry": "Dry\n(2022)", "moderate": "Moderate\n(2018)", "wet": "Wet\n(2024, OOD)"}
    x = np.arange(len(SCENARIOS))
    width = 0.32
    colors = {"MPC": "#1e293b", "TD3 (Pool B)": "#16a34a"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # --- Left panel: waterlog-days ---
    ax = axes[0]
    for i, ctrl in enumerate(["MPC", "TD3 (Pool B)"]):
        vals = [summary[(summary.controller == ctrl) & (summary.scenario == s)]["waterlog_days"].values[0]
                for s in SCENARIOS]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=ctrl, color=colors[ctrl])
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([scen_labels[s] for s in SCENARIOS])
    ax.set_ylabel("Waterlog-days / agent (lower = better)")
    ax.set_title("Waterlog stress")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    # --- Right panel: water-use efficiency ---
    ax = axes[1]
    for i, ctrl in enumerate(["MPC", "TD3 (Pool B)"]):
        vals = [summary[(summary.controller == ctrl) & (summary.scenario == s)]["wue"].values[0]
                for s in SCENARIOS]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=ctrl, color=colors[ctrl])
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([scen_labels[s] for s in SCENARIOS])
    ax.set_ylabel("Water-use efficiency (kg/ha per mm, higher = better)")
    ax.set_title("Water-use efficiency")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("TD3 vs. MPC: where the deployable controller wins", fontsize=16, y=1.02)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "r2_waterlog_efficiency.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
