# =============================================================================
# scripts/analysis/ema_pareto_a3.py
# Build a yield-vs-smoothness Pareto frontier PER alpha3 model from the sweep
# produced by scripts/experiments/exp_rl_ema_smoothing_a3.py, and recommend the
# best EMA smoothing alpha for each model.
#
# This mirrors scripts/analysis/ema_pareto.py exactly (same metrics, same MPC
# normalisation, same Pareto logic) but (a) reads the per-model output trees
# results/runs/<label>_ema_a*/ and (b) overlays every model on one figure and
# emits one recommendation per model.
#
# METRICS (aggregated over the 9-cell grid x seed, per alpha)
#   * yield_kg_ha, %MPC (per-cell yield / MPC-Hp8-perfect yield)
#   * mean|Delta u|   -- the control-effort / pulsing metric (lower = smoother)
#   * drought_days, waterlog_days, water_used
#
# BEST-ALPHA SELECTION RULE (explicit, matches the v2.21c reasoning)
#   Among the swept alphas, pick the SMOOTHEST (lowest mean|Delta u|) whose
#     - yield_mean >= baseline_yield(alpha=1.0) * (1 - yield_tol_pct/100), AND
#     - waterlog_days <= baseline_waterlog(alpha=1.0) + waterlog_tol_days
#   i.e. the knee of the frontier: maximum smoothing that costs no meaningful
#   yield and does not worsen waterlogging. Defaults: yield_tol=0.5%, wl_tol=1.0
#   day. If none qualifies (over-damped at every alpha), it falls back to the
#   Pareto-optimal alpha with the highest yield. The full frontier is always
#   written so you can choose a different operating point for the thesis.
#
# OUTPUTS  ->  results/analysis/ema_smoothing_a3/
#   ema_pareto_a3_percell.csv     one row per cell run (all models)
#   ema_pareto_a3.csv             one row per (model, alpha) with Pareto flag
#   ema_pareto_a3_recommend.csv   one row per model: chosen alpha + why
#   ema_pareto_a3.png             %MPC vs mean|du|, one frontier per model
#   ema_pareto_a3.md              tables + recommendations
#
# USAGE
#   python -m scripts.analysis.ema_pareto_a3 --labels a3_0p50 a3_1p15
#   # auto-discover every *_ema_a* tree that is not the v2.21c sweep:
#   python -m scripts.analysis.ema_pareto_a3
# =============================================================================

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.trajectory_metrics import (        # noqa: E402
    mean_abs_delta_u_from_parquet,
    read_final_metrics,
    build_mpc_reference,
    pct_of_mpc,
)

RUNS_DIR   = PROJECT_ROOT / 'results' / 'runs'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'analysis' / 'ema_smoothing_a3'

MPC_MEAN_ABS_DU = 0.97   # MPC Hp8 (perfect) control effort, for the plot annotation


def alpha_from_tag(tok: str):
    """'a0p30' -> 0.30 ; returns None if the token is not an alpha tag."""
    if tok.startswith('a') and 'p' in tok and tok[1:].replace('p', '').isdigit():
        return float(tok[1:].replace('p', '.'))
    return None


def parse_run_name(parquet_path: Path):
    """Extract (alpha, scenario, budget_pct, seed) from a sweep run filename.

    Filename: <label>_ema_a<tag>_<fc>_det_<scen>_rice_<b>pct_seed<s>.parquet
    The label may itself contain underscores; we key off the structural tokens.
    """
    stem = parquet_path.stem
    parts = stem.split('_')
    alpha = scenario = budget_pct = seed = None
    for tok in parts:
        a = alpha_from_tag(tok)
        if a is not None:
            alpha = a
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


def discover_labels(runs_dir: Path):
    """Find model labels from results/runs/<label>_ema_a*/ dirs (excluding v2.21c)."""
    labels = set()
    for d in runs_dir.glob('*_ema_a*'):
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith('td3_v221c_ema'):     # leave the v2.21c sweep to ema_pareto.py
            continue
        # strip the trailing _ema_a<tag>
        idx = name.rfind('_ema_a')
        if idx > 0:
            labels.add(name[:idx])
    return sorted(labels)


def collect_runs(labels, runs_dir=RUNS_DIR) -> pd.DataFrame:
    """Collect every sweep run for the given labels into a per-cell DataFrame."""
    mpc_ref = build_mpc_reference(runs_dir, horizon=8, crop='rice')
    if not mpc_ref:
        print("[warn] no MPC Hp8 perfect references found in results/runs/; "
              "%MPC will be NaN. (Generate them with scripts.experiments.exp_mpc.)")

    rows = []
    for label in labels:
        subdirs = sorted(runs_dir.glob(f"{label}_ema_a*"))
        if not subdirs:
            print(f"[warn] no run dirs for label '{label}' "
                  f"(expected {runs_dir}/{label}_ema_a*/).")
        for sub in subdirs:
            if not sub.is_dir():
                continue
            for pq in sorted(sub.glob('*.parquet')):
                alpha, scenario, budget_pct, seed = parse_run_name(pq)
                if alpha is None or scenario is None or budget_pct is None:
                    continue
                jp = pq.with_suffix('.json')
                if not jp.exists():
                    continue
                fm = read_final_metrics(jp)
                y = fm.get('yield_kg_ha', np.nan)
                rows.append({
                    'model':            label,
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
    if percell.empty:
        return percell
    g = percell.groupby(['model', 'alpha'])
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
    return agg.sort_values(['model', 'alpha'], ascending=[True, False]).reset_index(drop=True)


def pareto_front(agg: pd.DataFrame) -> pd.DataFrame:
    """Flag Pareto-optimal alphas per model in (min |du|, max yield)."""
    a = agg.copy().reset_index(drop=True)
    dominated = np.zeros(len(a), dtype=bool)
    du = a['mean_abs_du'].to_numpy()
    yld = a['yield_mean'].to_numpy()
    models = a['model'].to_numpy()
    for i in range(len(a)):
        for j in range(len(a)):
            if i == j or models[i] != models[j]:
                continue
            if (du[j] <= du[i] and yld[j] >= yld[i]
                    and (du[j] < du[i] or yld[j] > yld[i])):
                dominated[i] = True
                break
    a['pareto_optimal'] = ~dominated
    return a


def recommend(agg: pd.DataFrame, yield_tol_pct: float, waterlog_tol_days: float) -> pd.DataFrame:
    """Pick the best EMA alpha per model (see header for the rule)."""
    recs = []
    for model, sub in agg.groupby('model'):
        sub = sub.sort_values('alpha')              # alpha ascending = smoother first
        base = sub[np.isclose(sub['alpha'], 1.0)]
        if base.empty:
            base_yield = sub['yield_mean'].max()
            base_wl = sub['waterlog_days'].min()
            base_du = sub['mean_abs_du'].max()
        else:
            base_yield = float(base['yield_mean'].iloc[0])
            base_wl = float(base['waterlog_days'].iloc[0])
            base_du = float(base['mean_abs_du'].iloc[0])

        yield_floor = base_yield * (1.0 - yield_tol_pct / 100.0)
        wl_ceiling = base_wl + waterlog_tol_days

        ok = sub[(sub['yield_mean'] >= yield_floor)
                 & (sub['waterlog_days'] <= wl_ceiling)]
        if not ok.empty:
            chosen = ok.sort_values('mean_abs_du').iloc[0]   # smoothest that qualifies
            reason = (f"smoothest alpha with yield>={yield_floor:.0f} kg/ha "
                      f"(<= {yield_tol_pct:g}% below baseline) and "
                      f"waterlog<={wl_ceiling:.1f} d")
        else:
            par = sub[sub['pareto_optimal']] if 'pareto_optimal' in sub else sub
            chosen = par.sort_values('yield_mean', ascending=False).iloc[0]
            reason = "no alpha met the tolerance; fell back to highest-yield Pareto point"

        du_reduction = (1.0 - chosen['mean_abs_du'] / base_du) * 100.0 if base_du > 0 else np.nan
        recs.append({
            'model':              model,
            'best_alpha':         float(chosen['alpha']),
            'tau_days':           ema_tau(float(chosen['alpha'])),
            'yield_kg_ha':        float(chosen['yield_mean']),
            'pct_mpc':            float(chosen['pct_mpc_mean']),
            'mean_abs_du':        float(chosen['mean_abs_du']),
            'du_reduction_vs_raw_pct': du_reduction,
            'drought_days':       float(chosen['drought_days']),
            'waterlog_days':      float(chosen['waterlog_days']),
            'baseline_yield_kg_ha': base_yield,
            'baseline_mean_abs_du': base_du,
            'reason':             reason,
        })
    return pd.DataFrame(recs)


def ema_tau(alpha: float) -> float:
    """First-order time constant of the EMA, tau = -1/ln(1-alpha) days."""
    if alpha >= 1.0:
        return 0.0
    return float(-1.0 / np.log(1.0 - alpha))


def make_plot(agg: pd.DataFrame, recs: pd.DataFrame, out_png: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = sorted(agg['model'].unique())
    palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    rec_by_model = {r['model']: r for _, r in recs.iterrows()}

    for k, model in enumerate(models):
        sub = agg[agg['model'] == model].copy()
        color = palette[k % len(palette)]
        marker = markers[k % len(markers)]
        du = sub['mean_abs_du'].to_numpy()
        pct = sub['pct_mpc_mean'].to_numpy()
        alphas = sub['alpha'].to_numpy()
        order = np.argsort(du)
        ax.plot(du[order], pct[order], '-', color=color, alpha=0.5, zorder=1)
        ax.scatter(du, pct, c=color, marker=marker, s=70, zorder=3,
                   edgecolor='k', linewidth=0.5, label=f'{model}')
        for x, y, a in zip(du, pct, alphas):
            ax.annotate(f"{a:.1f}", (x, y), textcoords='offset points',
                        xytext=(5, 4), fontsize=7, color=color)
        # ring the recommended operating point
        r = rec_by_model.get(model)
        if r is not None:
            ax.scatter([r['mean_abs_du']], [r['pct_mpc']], s=240,
                       facecolors='none', edgecolors=color, linewidths=2.0, zorder=4)

    ax.axvline(MPC_MEAN_ABS_DU, color='steelblue', ls='--', lw=1.2,
               label=f'MPC control effort ({MPC_MEAN_ABS_DU})')
    ax.axhline(100.0, color='steelblue', ls=':', lw=1.0, label='MPC yield (100%)')

    ax.set_xlabel('Control effort  mean|Δu|  (mm/day)  — lower = smoother')
    ax.set_ylabel('Yield  (% of MPC Hp8 perfect)')
    ax.set_title('EMA smoothing: yield vs control-effort per alpha3 model\n'
                 '(ringed = recommended alpha)')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_markdown(agg: pd.DataFrame, recs: pd.DataFrame, out_md: Path):
    lines = ["# EMA smoothing per alpha3 model — yield vs control-effort\n"]
    lines.append("Post-hoc causal EMA filter on each model's action stream "
                 "(no retraining), aggregated over the 9-cell grid × seed. "
                 "`alpha=1.0` is the unsmoothed baseline.\n")
    lines.append(f"MPC Hp8 (perfect) reference: 100% yield at "
                 f"mean|Δu| = {MPC_MEAN_ABS_DU} mm/day.\n")

    lines.append("\n## Recommended smoothing per model\n")
    lines.append("| model | best α | τ (days) | yield kg/ha | %MPC | mean|Δu| | "
                 "Δu reduction | drought d | waterlog d | rule |")
    lines.append('|' + '---|' * 11)
    for _, r in recs.iterrows():
        lines.append(
            f"| {r['model']} | {r['best_alpha']:.2f} | {r['tau_days']:.1f} | "
            f"{r['yield_kg_ha']:.0f} | {r['pct_mpc']:.1f} | {r['mean_abs_du']:.3f} | "
            f"{r['du_reduction_vs_raw_pct']:.0f}% | {r['drought_days']:.1f} | "
            f"{r['waterlog_days']:.1f} | {r['reason']} |"
        )

    for model in sorted(agg['model'].unique()):
        sub = agg[agg['model'] == model]
        lines.append(f"\n## {model} — full frontier\n")
        lines.append("| alpha | n | yield kg/ha | %MPC | mean|Δu| | drought d | "
                     "waterlog d | water mm | Pareto |")
        lines.append('|' + '---|' * 9)
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['alpha']:.2f} | {int(r['n_runs'])} | {r['yield_mean']:.0f} | "
                f"{r['pct_mpc_mean']:.1f} | {r['mean_abs_du']:.3f} | "
                f"{r['drought_days']:.1f} | {r['waterlog_days']:.1f} | "
                f"{r['water_used_mm']:.0f} | {'YES' if r['pareto_optimal'] else ''} |"
            )
    lines.append("")
    out_md.write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(
        description="Per-model EMA Pareto frontier + best-alpha recommendation "
                    "for the alpha3 sweep winners.")
    p.add_argument('--labels', nargs='+', default=None,
                   help="Model labels used at sweep time (the <label> in "
                        "results/runs/<label>_ema_a*). Default: auto-discover.")
    p.add_argument('--yield-tol-pct', type=float, default=0.5,
                   help="Max yield drop vs unsmoothed baseline to still accept an "
                        "alpha, in %% (default 0.5).")
    p.add_argument('--waterlog-tol-days', type=float, default=1.0,
                   help="Max waterlog-day increase vs baseline to still accept an "
                        "alpha (default 1.0).")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = args.labels or discover_labels(RUNS_DIR)
    if not labels:
        print("No model labels given and none auto-discovered under "
              f"{RUNS_DIR}/<label>_ema_a*/. Run the sweep first:")
        print("  python -m scripts.experiments.exp_rl_ema_smoothing_a3 --model ...")
        return
    print(f"Models: {labels}")

    percell = collect_runs(labels)
    if percell.empty:
        print("No EMA sweep runs found for those labels.")
        return

    percell.sort_values(['model', 'alpha', 'scenario', 'budget_pct', 'seed']).to_csv(
        OUTPUT_DIR / 'ema_pareto_a3_percell.csv', index=False, encoding='utf-8')

    agg = pareto_front(aggregate_by_alpha(percell))
    agg.to_csv(OUTPUT_DIR / 'ema_pareto_a3.csv', index=False, encoding='utf-8')

    recs = recommend(agg, args.yield_tol_pct, args.waterlog_tol_days)
    recs.to_csv(OUTPUT_DIR / 'ema_pareto_a3_recommend.csv', index=False, encoding='utf-8')

    png_path = OUTPUT_DIR / 'ema_pareto_a3.png'
    try:
        make_plot(agg, recs, png_path)
    except Exception as e:
        print(f"[warn] plot failed ({e!r}); CSV/MD still written.")
        png_path = None

    write_markdown(agg, recs, OUTPUT_DIR / 'ema_pareto_a3.md')

    for model in sorted(agg['model'].unique()):
        sub = agg[agg['model'] == model]
        print(f"\n--- {model} "
              f"({len(percell[percell['model']==model])} cell-runs, "
              f"{sub['alpha'].nunique()} alphas) ---")
        print(sub[['alpha', 'n_runs', 'yield_mean', 'pct_mpc_mean',
                   'mean_abs_du', 'waterlog_days', 'pareto_optimal']].to_string(index=False))

    print("\n=== RECOMMENDED EMA alpha per model ===")
    print(recs[['model', 'best_alpha', 'tau_days', 'pct_mpc', 'mean_abs_du',
                'du_reduction_vs_raw_pct', 'waterlog_days']].to_string(index=False))

    print(f"\n  per-cell:    {OUTPUT_DIR / 'ema_pareto_a3_percell.csv'}")
    print(f"  per-alpha:   {OUTPUT_DIR / 'ema_pareto_a3.csv'}")
    print(f"  recommend:   {OUTPUT_DIR / 'ema_pareto_a3_recommend.csv'}")
    if png_path:
        print(f"  figure:      {png_path}")
    print(f"  summary:     {OUTPUT_DIR / 'ema_pareto_a3.md'}")


if __name__ == '__main__':
    main()
