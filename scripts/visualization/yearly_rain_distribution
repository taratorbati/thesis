import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# 1. Standalone dynamic data map
climate_data = {
    2000: 18.64, 2001: 14.49, 2002: 27.08, 2003: 36.49, 2004: 46.10,
    2005: 22.85, 2006: 16.39, 2007: 32.66, 2008: 44.83, 2009: 58.10,
    2010: 46.97, 2011: 55.66, 2012: 79.79, 2013: 82.11, 2014: 30.80,
    2015: 52.92, 2016: 76.97, 2017: 29.55, 2018: 108.82, 2019: 44.29,
    2020: 42.11, 2021: 66.72, 2022: 39.70, 2023: 88.37, 2024: 176.81,
    2025: 46.52,
}

# Convert dictionary to DataFrame
yearly_totals = pd.DataFrame(list(climate_data.items()), columns=["YEAR", "RAINFALL"])

# Calculate exact KDE density values using the same default bandwidth estimator as seaborn (Scott's Rule)
# This allows us to calculate the exact height of the curve at any specific point
kde_evaluator = stats.gaussian_kde(yearly_totals["RAINFALL"], bw_method='scott')

# 2. Map the machine learning split configurations
test_years = [2022, 2024, 2018]
eval_b_only = [2004, 2013]
eval_a_only = [2016, 2023]
eval_shared = [2002]

def get_set_label(year):
    if year in test_years:
        return "Test Set (2018, 2022, 2024)"
    elif year in eval_shared:
        return "Shared Evaluation Year (2002)"
    elif year in eval_a_only:
        return "Evaluation Set A (2016, 2023)"
    elif year in eval_b_only:
        return "Evaluation Set B (2004, 2013)"
    else:
        return "Training Set"

yearly_totals["SET"] = yearly_totals["YEAR"].apply(get_set_label)

# 3. Setup the visual framework
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# 4. Plot histogram bars AND density line together
sns.histplot(
    data=yearly_totals,
    x="RAINFALL",
    stat="density",
    kde=True,
    bins=10,
    color="#b0bec5",
    alpha=0.4,
    edgecolor="white",
    linewidth=1,
    kde_kws={"cut": 0},  # Limits KDE calculation to historical minimum/maximum
    line_kws={"linewidth": 2.5, "color": "#78909c"},
    label="Historical Distribution Data"
)

# Color specifications for the benchmark milestones
colors = {
    "Test Set (2018, 2022, 2024)": "#d32f2f",      # Red
    "Evaluation Set A (2016, 2023)": "#1976d2",  # Blue
    "Evaluation Set B (2004, 2013)": "#388e3c",  # Green
    "Shared Evaluation Year (2002)": "#f57c00",       # Orange
}

# 5. Plot finite vertical lines ending with an 'o' marker on the curve
added_labels = set()

for _, row in yearly_totals.iterrows():
    set_type = row["SET"]

    if set_type != "Training Set":
        label_to_use = set_type if set_type not in added_labels else None
        added_labels.add(set_type)
        
        x_val = row["RAINFALL"]
        # Extract the precise y-intercept on the curve
        y_val = kde_evaluator.evaluate([x_val])[0]

        # Draw the finite vertical line from 0 up to the density curve point
        plt.vlines(
            x=x_val,
            ymin=0,
            ymax=y_val,
            colors=colors[set_type],
            linestyles="--",
            linewidth=2,
            label=label_to_use
        )
        
        # Place the dot ('o' marker) exactly at the top coordinate
        plt.plot(
            x_val, 
            y_val, 
            marker='o', 
            markersize=8, 
            color=colors[set_type],
            zorder=5
        )

        # Place year text label cleanly right above the 'o' dot marker
        plt.text(
            x=x_val,
            y=y_val + 0.0004,  # Tiny offset above the dot so it doesn't overlap
            s=str(int(row['YEAR'])),
            color=colors[set_type],
            weight="bold",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=9
        )

# 6. Graph polishing & coordinate boundaries
plt.title(
    "Rainfall Frequency & Bounded Density Distribution (DOY 141-233)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Total Seasonal Rainfall (mm)", fontsize=12)
plt.ylabel("Density Scale", fontsize=12)

# Ensure graph boundaries start cleanly at 0 and adapt gracefully to text height
plt.xlim(0, None)
plt.ylim(0, plt.gca().get_ylim()[1] * 1.05)

# Position legend strictly at top right corner
plt.legend(loc="upper right", frameon=True, shadow=True, facecolor="white")

# 7. Compile high-resolution figure asset
plt.tight_layout()
#plt.savefig("finite_stems_density_distribution.png", dpi=300)
#plt.close()
plt.show()
print("Graph saved with finite marker stems ending cleanly with 'o' dots on the density line.")