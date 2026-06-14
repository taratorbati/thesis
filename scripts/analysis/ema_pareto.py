# =============================================================================
# scripts/analysis/ema_pareto.py
# Future-work item #5: build the yield-vs-smoothness Pareto frontier from the
# EMA smoothing sweep produced by scripts/experiments/exp_rl_ema_smoothing.py.
#
# For each smoothing weight alpha it aggregates, over the 9-cell grid x seeds:
#   * yield_kg_ha                 (mean +/- std across seeds, mean across cells)
#   * %MPC                         (per-cell yield / MPC-Hp8-perfect yield)
#   * mean|Delta u|                (the control-effort / pulsing metric)
#   * drought_days, waterlog_days, water_used
# and writes:
#   results/analysis/ema_smoothing/ema_pareto.csv         (one row per alpha)
#   results/analysis/ema_smoothing/ema_pareto_percell.csv (one row per cell run)
#   results/analysis/ema_smoothing/ema_pareto.png         (%MPC vs mean|Delta u|)
#   results/analysis/ema_smoothing/ema_pareto.md          (table + Pareto set)
#
# The MPC reference (mean|Delta u| ~= 0.97, 100% MPC yield) and the unsmoothed
# TD3 baseline (alpha = 1.0) are marked on the plot so the trade-off is legible.
#
# Usage:
#   python -m scripts.analysis.ema_pareto
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_lib.trajectory_metrics import (        # noqa: E402
    mean_abs_delta_u_from_parquet,
    read_final_metrics,
    build_mpc_reference,
    pct_of_mpc,
)

RUNS_DIR    = PROJECT_ROOT / 'results' / 'runs'
OUTPUT_DIR  = PROJECT_ROOT / 'results' / 'analysis' / 'ema_smoothing'

# MPC reference control-effort for the plot annotation (thesis comparison
# table, MPC Hp8 perfect).  Yield reference is computed per-cell from the
# committed MPC sidecars, not hard-coded.
MPC_MEAN_ABS_DU = 0.97


def _parse_run_name(parquet_path: Path):
    """Extract (alpha, scenario, budget_pct, seed) from an EMA run filename.

    Filename pattern:
      td3_v221c_ema_a<tag>_<fc>_det_<scenario>_rice_<budget>pct_seed<seed>.parquet
    where <tag> is e.g. '0p30' (alpha 0.30).
    """
    stem = parquet_path.stem
    parts = stem.split('_')
    # parts: ['td3','v221c','ema','a<tag>', <fc...>, 'det', <scenario>,
    #         'rice', '<budget>pct', 'seed<seed>']
    alpha = None
    scenario = None
    budget_pct = None
    seed = None
    for tok in parts:
        if tok.startswith('a') and 'p' in tok and tok[1:].replace('p', '').isdigit():
            alpha = float(tok[1:].replace('p', '.'))
        elif tok.endswith('pct'):
            try:
                budget_pct = int(tok.replace('pct', ''))
            except ValueError:
                pass
        elif tok.startswith('seed'):
            try:
                seed = int(tok.replace('seed', ''))
            except ValueError:
                pass
    for sc in ('dry', 'moderate', 'wet'):
        if f'_{sc}_' in stem:
            scenario = sc
            break
    return alpha, scenario, budget_pct, seed


def collect_runs(runs_dir=RUNS_DIR) -> pd.DataFrame:
    """Collect every EMA sweep run into a per-cell DataFrame with derived metrics."""
    mpc_ref = build_mpc_reference(runs_dir, horizon=8, crop='rice')
    if not mpc_ref:
        print("[warn] no MPC Hp8 perfect references found; %MPC will be NaN.")

    rows = []
    for sub in sorted(runs_dir.glob('td3_v221c_ema_a*')):
        if not sub.is_dir():
            continue
        for pq in sorted(sub.glob('*.parquet')):
            alpha, scenario, budget_pct, seed = _parse_run_name(pq)
            if alpha is None or scenario is None or budget_pct is None:
                continue
            jp = pq.with_suffix('.json')
            if not jp.exists():
                continue
            fm = read_final_metrics(jp)
            y = fm.get('yield_kg_ha', np.nan)
            rows.append({
                'alpha':            alpha,
                'scenario':         scenario,
                'budget_pct':       budget_pct,
                'seed':             seed,
                'yield_kg_ha':      y,
                'pct_mpc':          pct_of_mpc(y, scenario, budget_pct, mpc_ref),
                'mean_abs_du':      mean_abs_delta_u_from_parquet(pq),
                'water_used_mm':    fm.get('water_used_mm', np.nan),
                'drought_days':     fm.get('drought_days_per_agent', np.nan),
                'waterlog_days':    fm.get('waterlog_days_per_agent', np.nan),
                'spatial_equity_cv': fm.get('spatial_equity_cv', np.nan),
            })
    return pd.DataFrame(rows)


def aggregate_by_alpha(percell: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell runs to one row per alpha (mean over cells x seeds)."""
    if percell.empty:
        return percell
    g = percell.groupby('alpha')
    agg = g.agg(
        n_runs=('yield_kg_ha', 'size'),
        yield_mean=('yield_kg_ha', 'mean'),
        yield_std=('yield_kg_ha', 'std'),
        pct_mpc_mean=('pct_mpc', 'mean'),
        pct_mpc_std=('pct_mpc', 'std'),
        mean_abs_du=('mean_abs_du', 'mean'),
        mean_abs_du_std=('mean_abs_du', 'std'),
        drought_days=('drought_days', 'mean'),
        waterlog_days=('waterlog_days', 'mean'),
        water_used_mm=('water_used_mm', 'mean'),
    ).reset_index()
    return agg.sort_values('alpha', ascending=False).reset_index(drop=True)


def pareto_front(agg: pd.DataFrame) -> pd.DataFrame:
    """Flag the Pareto-optimal alphas in the (minimise |du|, maximise yield) plane.

    A point is dominated if another point has both lower-or-equal mean|du| AND
    higher-or-equal yield (with at least one strict).  Non-dominated points form
    the achievable trade-off frontier.
    """
    a = agg.copy().reset_index(drop=True)
    dominated = np.zeros(len(a), dtype=bool)
    du = a['mean_abs_du'].to_numpy()
    yld = a['yield_mean'].to_numpy()
    for i in range(len(a)):
        for j in range(len(a)):
            if i == j:
                continue
            if (du[j] <= du[i] and yld[j] >= yld[i]
                    and (du[j] < du[i] or yld[j] > yld[i])):
                dominated[i] = True
                break
    a['pareto_optimal'] = ~dominated
    return a


def make_plot(agg: pd.DataFrame, out_png: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    du = agg['mean_abs_du'].to_numpy()
    pct = agg['pct_mpc_mean'].to_numpy()
    alphas = agg['alpha'].to_numpy()

    # Trade-off curve (sorted by smoothing strength).
    order = np.argsort(du)
    ax.plot(du[order], pct[order], '-', color='0.6', zorder=1)
    sc = ax.scatter(du, pct, c=alphas, cmap='viridis', s=70, zorder=3,
                    edgecolor='k', linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('EMA smoothing weight alpha')

    for x, y, a in zip(du, pct, alphas):
        ax.annotate(f"a={a:.2f}", (x, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=8)

    # Baseline (alpha = 1.0) and MPC reference markers.
    base = agg[np.isclose(agg['alpha'], 1.0)]
    if not base.empty:
        ax.scatter(base['mean_abs_du'], base['pct_mpc_mean'], marker='s',
                   s=120, facecolor='none', edgecolor='crimson', linewidth=1.8,
                   zorder=4, label='TD3 baseline (alpha=1.0)')
    ax.axvline(MPC_MEAN_ABS_DU, color='steelblue', ls='--', lw=1.2,
               label=f'MPC control effort ({MPC_MEAN_ABS_DU})')
    ax.axhline(100.0, color='steelblue', ls=':', lw=1.0,
               label='MPC yield (100%)')

    ax.set_xlabel('Control effort  mean|Delta u|  (mm/day)  — lower = smoother')
    ax.set_ylabel('Yield  (% of MPC Hp8 perfect)')
    ax.set_title('TD3 v2.21c: yield vs control-effort under EMA smoothing')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_markdown(agg_pareto: pd.DataFrame, out_md: Path):
    lines = []
    lines.append("# EMA smoothing — yield vs control-effort Pareto sweep\n")
    lines.append("TD3 v2.21c checkpoints, re-evaluated under a causal EMA filter "
                 "on the policy action (no retraining). Aggregated over the "
                 "9-cell grid x seeds. `alpha=1.0` is the unsmoothed baseline.\n")
    lines.append(f"MPC Hp8 (perfect) reference: 100% yield at "
                 f"mean|Delta u| = {MPC_MEAN_ABS_DU} mm/day.\n")
    cols = ['alpha', 'n_runs', 'yield_mean', 'pct_mpc_mean', 'mean_abs_du',
            'drought_days', 'waterlog_days', 'water_used_mm', 'pareto_optimal']
    header = ('| alpha | n | yield kg/ha | %MPC | mean|du| | drought d | '
              'waterlog d | water mm | Pareto |')
    sep = '|' + '---|' * 9
    lines.append(header)
    lines.append(sep)
    for _, r in agg_pareto.iterrows():
        lines.append(
            f"| {r['alpha']:.2f} | {int(r['n_runs'])} | {r['yield_mean']:.0f} | "
            f"{r['pct_mpc_mean']:.1f} | {r['mean_abs_du']:.3f} | "
            f"{r['drought_days']:.1f} | {r['waterlog_days']:.1f} | "
            f"{r['water_used_mm']:.0f} | {'YES' if r['pareto_optimal'] else ''} |"
        )
    lines.append("")
    out_md.write_text('\n'.join(lines), encoding='utf-8')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    percell = collect_runs()
    if percell.empty:
        print("No EMA sweep runs found under results/runs/td3_v221c_ema_a*/.")
        print("Run: python -m scripts.experiments.exp_rl_ema_smoothing")
        return

    percell_path = OUTPUT_DIR / 'ema_pareto_percell.csv'
    percell.sort_values(['alpha', 'scenario', 'budget_pct', 'seed']).to_csv(
        percell_path, index=False, encoding='utf-8')

    agg = aggregate_by_alpha(percell)
    agg = pareto_front(agg)

    csv_path = OUTPUT_DIR / 'ema_pareto.csv'
    agg.to_csv(csv_path, index=False, encoding='utf-8')

    png_path = OUTPUT_DIR / 'ema_pareto.png'
    try:
        make_plot(agg, png_path)
    except Exception as e:
        print(f"[warn] plot failed ({e!r}); CSV/MD still written.")
        png_path = None

    md_path = OUTPUT_DIR / 'ema_pareto.md'
    write_markdown(agg, md_path)

    print(f"\nEMA Pareto sweep — {len(percell)} cell-runs, "
          f"{agg['alpha'].nunique()} alpha values")
    print(agg[['alpha', 'n_runs', 'yield_mean', 'pct_mpc_mean',
               'mean_abs_du', 'pareto_optimal']].to_string(index=False))
    print(f"\n  per-cell: {percell_path}")
    print(f"  per-alpha: {csv_path}")
    if png_path:
        print(f"  figure:   {png_path}")
    print(f"  summary:  {md_path}")


if __name__ == '__main__':
    main()
