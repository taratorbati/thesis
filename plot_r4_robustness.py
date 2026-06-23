# =============================================================================
# plot_r4_robustness.py
#
# SLIDE R4: forecast-noise robustness. Visualizes how much each controller's
# yield drops when moving from a perfect forecast to a realistic noisy
# forecast (multiplicative AR(1) noise, ~15% error at 1-day lead growing to
# ~42% at 8-day lead -- see ch3/ch5 noise model).
#
# Produces a slope ("dumbbell") chart: one horizontal line per controller,
# left point = perfect-forecast yield (% of ceiling), right point =
# noisy-forecast yield (% of ceiling). A flatter line = more robust to
# forecast error. This directly visualizes "robustness" rather than just
# stating a percentage-retained number.
#
# Controllers shown: MPC (Hp=8) and TD3 (Pool B, 3-seed mean) -- the two
# controllers for which a noisy-forecast evaluation exists in the
# ground-truth results folder. SAC is omitted because SAC's noisy results in
# the "final results" folder follow the same eval naming pattern but are not
# the comparison this slide is about (TD3 is the headline deployable
# controller).
#
# DATA SOURCE: results/runs/final results/ via defense_data_loader.py
# (must be in the same directory, or importable from PYTHONPATH).
#
# OUTPUT: results/analysis/defense_slides/r4_robustness.png
#
# USAGE: python plot_r4_robustness.py
# Requires: numpy, pandas, pyarrow, matplotlib, defense_data_loader.py
# =============================================================================

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from defense_data_loader import build_full_grid  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "defense_slides"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df_perfect = build_full_grid("perfect")
    df_noisy = build_full_grid("noisy")

    controllers = ["MPC", "TD3 (Pool B)"]
    rows = []
    for ctrl in controllers:
        perfect_pct = df_perfect[df_perfect.controller == ctrl]["pct_ceiling"].mean()
        noisy_pct = df_noisy[df_noisy.controller == ctrl]["pct_ceiling"].mean()
        rows.append((ctrl, perfect_pct, noisy_pct, noisy_pct - perfect_pct))
        print(f"{ctrl}: perfect={perfect_pct:.2f}%  noisy={noisy_pct:.2f}%  "
              f"drop={noisy_pct - perfect_pct:+.2f} pts")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
        "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 13,
    })

    colors = {"MPC": "#1e293b", "TD3 (Pool B)": "#16a34a"}
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    y_positions = {ctrl: i for i, ctrl in enumerate(reversed(controllers))}

    for ctrl, perfect_pct, noisy_pct, drop in rows:
        y = y_positions[ctrl]
        color = colors[ctrl]
        ax.plot([perfect_pct, noisy_pct], [y, y], color=color, lw=3, alpha=0.5, zorder=1)
        ax.scatter([perfect_pct], [y], s=220, color=color, marker="o",
                   edgecolor="white", linewidth=1.5, zorder=3,
                   label="Perfect forecast" if ctrl == controllers[0] else None)
        ax.scatter([noisy_pct], [y], s=220, color=color, marker="D",
                   edgecolor="white", linewidth=1.5, zorder=3,
                   label="Noisy forecast" if ctrl == controllers[0] else None)
        mid_x = (perfect_pct + noisy_pct) / 2
        label_y_offset = 0.18
        ax.annotate(f"{drop:+.2f} pts", (mid_x, y + label_y_offset),
                    ha="center", fontsize=11, color=color, fontweight="bold")

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel("Yield (% of stress-free ceiling)")
    ax.set_title("Forecast-noise robustness: perfect vs. noisy forecast")
    ax.set_ylim(-0.6, len(controllers) - 0.4)
    ax.grid(axis="x", alpha=0.25)

    # Single shared legend for marker shapes (perfect vs noisy), since color
    # already encodes controller.
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#64748b",
               markersize=12, label="Perfect forecast"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#64748b",
               markersize=11, label="Noisy forecast (AR(1), \u00b115\u201342%)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", framealpha=0.95)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "r4_robustness.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
