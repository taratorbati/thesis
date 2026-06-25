# =============================================================================
# plot_ema_pareto_for_defense.py
# =============================================================================

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent

def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "results" / "runs" / "final results").is_dir():
            return candidate
    return start

PROJECT_ROOT = _find_project_root(_SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "results" / "runs" / "final results"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "ema_smoothing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HIGHLIGHT_ALPHA = 0.3

def load_action_matrix(parquet_path):
    df = pd.read_parquet(parquet_path, columns=["day", "agent", "u"])
    return df.pivot(index="day", columns="agent", values="u").to_numpy()

def mean_abs_delta_u(U):
    return float(np.mean(np.abs(np.diff(U, axis=0))))

def parse_run_name(parquet_path: Path):
    stem = parquet_path.stem
    pool_m = re.search(r"pool([AB])", stem)
    alpha_m = re.search(r"_a(\d+p\d+)_", stem)
    scen_m = re.search(r"_(dry|moderate|wet)_rice_", stem)
    budget_m = re.search(r"_(\d+)pct_", stem)
    seed_m = re.search(r"_seed(\d+)", stem)
    pool = pool_m.group(1) if pool_m else None
    alpha = float(alpha_m.group(1).replace("p", ".")) if alpha_m else None
    scenario = scen_m.group(1) if scen_m else None
    budget_pct = int(budget_m.group(1)) if budget_m else None
    seed = int(seed_m.group(1)) if seed_m else None
    return pool, alpha, scenario, budget_pct, seed

def build_mpc_reference(runs_dir, horizon=8):
    ref_yield = {}
    du_vals, yield_vals = [], []
    for jp in sorted((runs_dir / "mpc").glob(f"mpc_perfect_*_rice_*pct_Hp{horizon}.json")):
        with open(jp, "r", encoding="utf-8") as f:
            d = json.load(f)
        stem = jp.stem
        scen = re.search(r"mpc_perfect_(dry|moderate|wet)_rice_", stem).group(1)
        budget_pct = int(re.search(r"_(\d+)pct_", stem).group(1))
        y = d["final_metrics"]["yield_kg_ha"]
        ref_yield[(scen, budget_pct)] = y
        yield_vals.append(y)
        pq = jp.with_suffix(".parquet")
        if pq.exists():
            du_vals.append(mean_abs_delta_u(load_action_matrix(pq)))
    mpc_mean_du = float(np.mean(du_vals)) if du_vals else 0.97
    mpc_mean_yield = float(np.mean(yield_vals)) if yield_vals else None
    return ref_yield, mpc_mean_du, mpc_mean_yield

def collect_ema_runs(runs_dir):
    rows = []
    for sub in sorted(runs_dir.glob("td3_v221c_ema_pool*")):
        if not sub.is_dir():
            continue
        for pq in sorted(sub.glob("*.parquet")):
            pool, alpha, scenario, budget_pct, seed = parse_run_name(pq)
            if alpha is None or scenario is None or budget_pct is None:
                continue
            jp = pq.with_suffix(".json")
            if not jp.exists():
                continue
            with open(jp, "r", encoding="utf-8") as f:
                fm = json.load(f)["final_metrics"]
            rows.append({
                "pool": pool, "alpha": alpha, "scenario": scenario,
                "budget_pct": budget_pct, "seed": seed,
                "yield_kg_ha": fm["yield_kg_ha"],
                "mean_abs_du": mean_abs_delta_u(load_action_matrix(pq)),
                "drought_days": fm.get("drought_days_per_agent", np.nan),
                "waterlog_days": fm.get("waterlog_days_per_agent", np.nan),
                "water_used_mm": fm.get("water_used_mm", np.nan),
            })
    return pd.DataFrame(rows)

def aggregate(percell, mpc_ref):
    percell = percell.copy()
    percell["pct_mpc"] = percell.apply(
        lambda r: 100.0 * r["yield_kg_ha"] / mpc_ref[(r["scenario"], r["budget_pct"])], axis=1
    )
    agg = (
        percell.groupby(["pool", "alpha"])
        .agg(
            n_runs=("yield_kg_ha", "size"),
            yield_mean=("yield_kg_ha", "mean"),
            pct_mpc_mean=("pct_mpc", "mean"),
            mean_abs_du=("mean_abs_du", "mean"),
            drought_days=("drought_days", "mean"),
            waterlog_days=("waterlog_days", "mean"),
            water_used_mm=("water_used_mm", "mean"),
        )
        .reset_index()
        .sort_values(["pool", "alpha"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return agg

def make_defense_plot(agg, mpc_du, mpc_yield_pct, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    pool_styles = {
        "A": dict(color="#1f77b4", marker="o", ls="-", label="Pool A (years 2002/2016/2023)"),
        "B": dict(color="#d97706", marker="s", ls="-", label="Pool B (years 2002/2004/2013)"),
    }

    # =========================================================================
    # [EDIT HERE: MANUAL LABEL PLACEMENT]
    # Adjust these (X, Y) pairs to move the labels exactly where you want them.
    # Positive X moves text RIGHT, Negative X moves text LEFT.
    # Positive Y moves text UP, Negative Y moves text DOWN.
    # Format: Alpha_Value: (X_Offset, Y_Offset)
    # =========================================================================
    MANUAL_OFFSETS = {
        0.1: (10, 10),
        0.2: (-30, 10),
        0.3: (-10, 10),
        0.5: (-10, 12),
        0.7: (-10, 15),
        1.0: (-30, 12),
    }

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    
    # We only want to label the alphas once. We will label them when plotting Pool A.
    for pool in ["A", "B"]:
        sub = agg[agg["pool"] == pool].sort_values("mean_abs_du")
        style = pool_styles[pool]
        ax.plot(sub["mean_abs_du"], sub["pct_mpc_mean"],
                color=style["color"], ls=style["ls"], lw=2.4, alpha=0.85, zorder=2)
        ax.scatter(sub["mean_abs_du"], sub["pct_mpc_mean"],
                   color=style["color"], marker=style["marker"], s=140,
                   edgecolor="white", linewidth=1.2, zorder=4, label=style["label"])
        
        # Only add the annotations when iterating through the first pool (Pool A)
        if pool == "A":
            for _, row in sub.iterrows():
                alpha_val = row["alpha"]
                
                # Fetch the manual offset, or use a default if it's not in the dictionary
                offset = MANUAL_OFFSETS.get(alpha_val)
                
                ax.annotate(f"\u03b1={alpha_val:.1f}",
                            (row["mean_abs_du"], row["pct_mpc_mean"]),
                            textcoords="offset points", xytext=offset,
                            fontsize=13, color="black", fontweight="bold")

    hi = agg[agg["alpha"] == HIGHLIGHT_ALPHA]
    #if not hi.empty:
    #    ax.scatter(hi["mean_abs_du"], hi["pct_mpc_mean"], s=420, facecolor="none",
    #               edgecolor="#16a34a", linewidth=2.5, zorder=5)

    ax.scatter([mpc_du], [mpc_yield_pct], s=180, color="#1e293b", marker="*",
               zorder=6, label="MPC (Hp=8, reference)")
               
    ax.axhline(100.0, color="#94a3b8", ls=":", lw=1.3, zorder=1)
    ax.axvline(mpc_du, color="#94a3b8", ls="--", lw=1.3, zorder=1)

    ax.set_xlabel("Control effort  mean|\u0394u|  (mm/day)  \u2014  lower = smoother")
    ax.set_ylabel("Yield  (% of MPC Hp8 perfect)")
    ax.set_title("TD3: yield vs. control effort under post-hoc EMA smoothing")
    
    ax.set_ylim(98.7, 100.2)
    ax.grid(True, alpha=0.25)
    
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Saved chart to {out_png}")

def main():
    print(f"Reading from: {RUNS_DIR}")
    if not RUNS_DIR.is_dir():
        raise SystemExit(
            f"\n[error] Could not find the data folder:\n  {RUNS_DIR}\n"
        )

    mpc_ref, mpc_du, mpc_yield = build_mpc_reference(RUNS_DIR, horizon=8)
    if not mpc_ref or mpc_yield is None:
        raise SystemExit(
            f"\n[error] No MPC Hp8 perfect-forecast files found..."
        )
        
    percell = collect_ema_runs(RUNS_DIR)
    agg = aggregate(percell, mpc_ref)
    agg.to_csv(OUTPUT_DIR / "ema_pareto_defense.csv", index=False)
    
    mpc_yield_pct = 100.0
    make_defense_plot(agg, mpc_du, mpc_yield_pct, OUTPUT_DIR / "ema_pareto_defense.png")

if __name__ == "__main__":
    main()