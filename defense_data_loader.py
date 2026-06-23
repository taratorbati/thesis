# =============================================================================
# defense_data_loader.py
#
# Shared, ground-truth data loading for the R2 / R4 / budget-scenario defense
# charts. All four chart scripts (plot_r2_*.py, plot_r4_*.py,
# plot_budget_scenario_option1.py, plot_budget_scenario_option2.py) import
# from this module so the loading/parsing logic only has to be written and
# verified once.
#
# Reads exclusively from results/runs/final results/ (ground truth, per the
# defense-prep decision: SAC Pool B = 200k checkpoint, TD3 = v2.21c 3-seed
# mean for both pools).
#
# Requires: numpy, pandas, pyarrow.
# =============================================================================

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
BASE = PROJECT_ROOT / "results" / "runs" / "final results"

SCENARIOS = ["dry", "moderate", "wet"]
BUDGETS = [70, 85, 100]

# Per-scenario stress-free ceiling yield (kg/ha), from compute_ceiling_yield.py,
# verified against the committed repo physics (drought/waterlog stress forced
# to 1.0, unlimited on-demand irrigation, everything else in the ABM unchanged).
CEILING_YIELD = {"dry": 4265.2, "moderate": 3913.6, "wet": 3833.4}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mpc_cell(scenario, budget_pct, hp="Hp8", mode="perfect", seed_tag=""):
    """Load a single MPC (scenario, budget) cell's final_metrics dict."""
    if mode == "perfect":
        pattern = str(BASE / "mpc" / f"mpc_perfect_{scenario}_rice_{budget_pct}pct_{hp}.json")
    else:
        pattern = str(BASE / "mpc" / f"mpc_noisy_{scenario}_rice_{budget_pct}pct_{hp}*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return _read_json(files[0])["final_metrics"]


def load_rl_cell(folder, scenario, budget_pct, forecast="perfect"):
    """Load a single RL (scenario, budget) cell's final_metrics dict.
    forecast: 'perfect' -> sac_perfect_det_*; 'noisy' -> sac_noisy_ns42_det_*
    (filename prefix is 'sac_' even for TD3 runs -- legacy artifact of the
    eval script, not a labelling error in the folder itself).
    """
    prefix = "sac_perfect_det" if forecast == "perfect" else "sac_noisy_ns42_det"
    pattern = str(BASE / folder / f"{prefix}_{scenario}_rice_{budget_pct}pct*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return _read_json(files[0])["final_metrics"]


def load_rl_cell_multiseed(seed_folders, scenario, budget_pct, forecast="perfect"):
    """Average a metric dict across multiple seed folders for one (scenario,
    budget) cell. Returns a dict of means across whatever keys are present in
    every seed's final_metrics."""
    metrics_list = []
    for folder in seed_folders:
        m = load_rl_cell(folder, scenario, budget_pct, forecast)
        if m is not None:
            metrics_list.append(m)
    if not metrics_list:
        return None
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys if isinstance(metrics_list[0][k], (int, float))}


# Canonical seed-folder lists for the two final RL controllers in the
# headline comparison (TD3 v2.21c, both pools; SAC v2.18, both checkpoints).
TD3_POOL_A_SEEDS = [
    "td3_v221c_pool_A_seed0_best_model",
    "td3_v221c_pool_A_seed1_best_mdoel",
    "td3_v221c_pool_A_seed2_best_mdoel",
]
TD3_POOL_B_SEEDS = [
    "td3_v221c_pool_B_seed0_best_model",
    "td3_v221c_pool_B_seed1_best_model",
    "td3_v221c_pool_B_seed2_best_model",
]
SAC_POOL_A_FOLDER = "sac_v218_pool_A_seed0_best_model"
SAC_POOL_B_FOLDER = "sac_v218_pool_B_seed0_200k_model"


def build_full_grid(forecast="perfect"):
    """Build a tidy long-format DataFrame: one row per
    (controller, scenario, budget_pct) with all final_metrics columns.
    controller in {'MPC', 'TD3 (Pool A)', 'TD3 (Pool B)', 'SAC (Pool A)', 'SAC (Pool B)'}.
    """
    rows = []
    for scenario in SCENARIOS:
        for budget_pct in BUDGETS:
            cells = {
                "MPC": load_mpc_cell(scenario, budget_pct, mode=forecast),
                "TD3 (Pool A)": load_rl_cell_multiseed(TD3_POOL_A_SEEDS, scenario, budget_pct, forecast),
                "TD3 (Pool B)": load_rl_cell_multiseed(TD3_POOL_B_SEEDS, scenario, budget_pct, forecast),
                "SAC (Pool A)": load_rl_cell(SAC_POOL_A_FOLDER, scenario, budget_pct, forecast),
                "SAC (Pool B)": load_rl_cell(SAC_POOL_B_FOLDER, scenario, budget_pct, forecast),
            }
            for controller, m in cells.items():
                if m is None:
                    continue
                row = {"controller": controller, "scenario": scenario, "budget_pct": budget_pct}
                row.update(m)
                row["ceiling_yield_kg_ha"] = CEILING_YIELD[scenario]
                row["pct_ceiling"] = 100.0 * row["yield_kg_ha"] / CEILING_YIELD[scenario]
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test: print a small slice to confirm the loader works end to end.
    df = build_full_grid("perfect")
    print(f"Loaded {len(df)} rows (expect 5 controllers x 3 scenarios x 3 budgets = 45)")
    print(df[["controller", "scenario", "budget_pct", "yield_kg_ha", "pct_ceiling",
              "drought_days_per_agent", "waterlog_days_per_agent", "water_used_mm"]]
          .to_string(index=False))
