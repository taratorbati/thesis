# =============================================================================
# scripts/experiments/exp_rl_ema_smoothing.py
# Future-work item #5: yield-vs-smoothness Pareto sweep for the TD3 policy.
#
# Re-evaluates the ALREADY-TRAINED v2.21c checkpoints (no retraining) under a
# range of EMA action-smoothing weights alpha, across the full 9-cell grid
# (3 scenarios x 3 budgets) and all available seeds.  alpha = 1.0 is the
# unsmoothed baseline and reproduces the headline v2.21c numbers.
#
# Outputs land under results/runs/td3_v221c_ema_a<tag>/ (one subdir per alpha),
# in the SAME parquet/JSON format as every other run, so the standard
# aggregation/plotting tooling reads them unchanged.  Runs are skipped if their
# output already exists, so a crashed sweep resumes exactly where it stopped
# (per-cell granularity); src.runner additionally checkpoints within each
# season every 10 days.
#
# Usage
# -----
#   # Full default sweep (alpha in {1.0,0.7,0.5,0.3,0.2,0.1}) x seeds {0,1,2}
#   # x 9 cells, perfect forecast:
#   python -m scripts.experiments.exp_rl_ema_smoothing
#
#   # A single alpha / single seed (quick check):
#   python -m scripts.experiments.exp_rl_ema_smoothing --alphas 0.3 --seeds 0
#
#   # Custom grid:
#   python -m scripts.experiments.exp_rl_ema_smoothing \
#       --alphas 1.0 0.5 0.25 0.1 --seeds 0 1 2 --scenario all --budget all
#
#   # Point at checkpoints explicitly (otherwise auto-discovered):
#   python -m scripts.experiments.exp_rl_ema_smoothing \
#       --ckpt 0=results/rl/td3_v221c_termyield_seed0_XXXX/best_model/best_model.zip
#
# After the sweep, build the Pareto frontier with:
#   python -m scripts.analysis.ema_pareto
# =============================================================================

import argparse
import sys
from pathlib import Path

# ── Windows / non-UTF8 console safety (same pattern as exp_rl.py) ─────────────
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.climate_data import SCENARIO_YEARS                       # noqa: E402
from src.model.soil_data import get_crop                                # noqa: E402
from src.model.terrain import load_terrain                          # noqa: E402

SCENARIOS_ALL = list(SCENARIO_YEARS.keys())                   # ['dry','moderate','wet']
BUDGET_LEVELS = {100: 1.00, 85: 0.85, 70: 0.70}
CROP_FULL_BUDGET_MM = {'rice': 484.0}

DEM_PATH        = PROJECT_ROOT / 'data/gilan_farm.tif'
RL_OUTPUT_DIR   = PROJECT_ROOT / 'results' / 'rl'
RUNS_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'runs'

# Default smoothing grid: 1.0 (= baseline anchor) down to heavy smoothing.
DEFAULT_ALPHAS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
DEFAULT_SEEDS  = [0, 1, 2]


def alpha_tag(alpha: float) -> str:
    """Filesystem-safe tag for an alpha value, e.g. 1.0 -> '1p00', 0.3 -> '0p30'."""
    return f"{alpha:.2f}".replace('.', 'p')


POOL_DEV_YEARS = {
    'A': (2002, 2016, 2023),
    'B': (2002, 2004, 2013),
}


def _read_manifest_dev_years(run_dir):
    """Read dev_years from a training run's manifest.json, or None."""
    import json as _json
    mp = run_dir / 'manifest.json'
    if not mp.exists():
        return None
    try:
        return tuple(_json.loads(mp.read_text(encoding='utf-8'))['dev_years'])
    except Exception:
        return None


def discover_v221c_checkpoints(pool: str = None) -> dict:
    """Locate the v2.21c best_model.zip for each available seed.

    Parameters
    ----------
    pool : str or None
        'A' — only checkpoints trained on DEV_YEARS = (2002, 2016, 2023).
        'B' — only checkpoints trained on DEV_YEARS = (2002, 2004, 2013).
        None — pick the latest per seed regardless of pool (legacy behaviour,
               can silently mix pools when both exist).

    Returns
    -------
    dict
        {seed:int -> Path-to-checkpoint}
    """
    target_dev = POOL_DEV_YEARS.get(pool.upper()) if pool else None

    found: dict = {}
    for run_dir in sorted(RL_OUTPUT_DIR.glob('td3_v221c_termyield_seed*')):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        try:
            seed = int(name.split('seed')[1].split('_')[0])
        except (IndexError, ValueError):
            continue
        # Pool filter: skip runs whose manifest dev_years don't match.
        if target_dev is not None:
            manifest_dev = _read_manifest_dev_years(run_dir)
            if manifest_dev is not None and tuple(manifest_dev) != target_dev:
                continue
        best = run_dir / 'best_model' / 'best_model.zip'
        if best.exists():
            found[seed] = best
            continue
        finals = sorted(run_dir.glob('*_final.zip'))
        if finals:
            found[seed] = finals[0]
    return found


def parse_ckpt_overrides(items) -> dict:
    """Parse --ckpt seed=path overrides into {seed:int -> Path}."""
    overrides: dict = {}
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f"--ckpt expects seed=path, got {item!r}")
        s, p = item.split('=', 1)
        overrides[int(s)] = Path(p)
    return overrides


def run_sweep(args):
    from src.rl.ema_smoothing import EMASmoothedRLController
    from src.sim.runner import run_season
    from src.model.climate_data import load_cleaned_data, extract_scenario_by_name

    crop         = get_crop('rice')
    terrain      = load_terrain(str(DEM_PATH))
    df_climate   = load_cleaned_data()
    full_need_mm = CROP_FULL_BUDGET_MM['rice']

    scenarios   = SCENARIOS_ALL if args.scenario == 'all' else [args.scenario]
    budget_pcts = list(BUDGET_LEVELS) if args.budget == 'all' else [int(args.budget)]
    alphas      = [float(a) for a in args.alphas]

    # Determine which pool(s) to sweep.
    pools_to_run = list(POOL_DEV_YEARS.keys()) if args.pool == 'both' else [args.pool.upper()]

    forecast_mode = args.forecast
    noise_seed    = args.noise_seed

    for pool in pools_to_run:
        pool_dev = POOL_DEV_YEARS[pool]
        pool_tag = f"pool{pool}"   # 'poolA' or 'poolB'

        # Resolve checkpoints for this pool.
        ckpts = discover_v221c_checkpoints(pool=pool)
        ckpts.update(parse_ckpt_overrides(args.ckpt))
        if not ckpts:
            print(f"\n[skip] No v2.21c checkpoints found for pool {pool} "
                  f"(DEV_YEARS={pool_dev}). Train them first or pass --ckpt.")
            continue

        seeds = [s for s in args.seeds if s in ckpts]
        missing = [s for s in args.seeds if s not in ckpts]
        if missing:
            print(f"[warn] pool {pool}: no checkpoint for seed(s) {missing}; "
                  f"skipping. Available: {sorted(ckpts)}")
        if not seeds:
            print(f"[skip] pool {pool}: none of the requested seeds have a checkpoint.")
            continue

        print(f"\n{'='*70}")
        print(f"EMA smoothing sweep — pool {pool} (DEV={pool_dev})")
        print(f"{'='*70}")
        print(f"  alphas:    {alphas}")
        print(f"  seeds:     {seeds}  (checkpoints: "
              f"{ {s: ckpts[s].parent.parent.name for s in seeds} })")
        print(f"  scenarios: {scenarios}")
        print(f"  budgets:   {budget_pcts}")
        print(f"  forecast:  {forecast_mode}"
              + (f"  (noise_seed={noise_seed})" if forecast_mode == 'noisy' else ""))
        n_total = len(alphas) * len(seeds) * len(scenarios) * len(budget_pcts)
        print(f"  total cells to evaluate: {n_total}\n")

        done = skipped = 0
        for alpha in alphas:
            tag = alpha_tag(alpha)
            runs_dir = RUNS_OUTPUT_DIR / f"td3_v221c_ema_{pool_tag}_a{tag}"
            runs_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                model_path = ckpts[seed]

                for scenario in scenarios:
                    climate = extract_scenario_by_name(df_climate, scenario, crop)
                    climate['year'] = SCENARIO_YEARS[scenario]

                    for budget_pct in budget_pcts:
                        budget_total = full_need_mm * BUDGET_LEVELS[budget_pct]

                        fc_tag = ('perfect' if forecast_mode == 'perfect'
                                  else f"noisy_ns{noise_seed}")
                        out_name = (f"td3_v221c_ema_{pool_tag}_a{tag}_{fc_tag}_det_"
                                    f"{scenario}_rice_{budget_pct}pct_seed{seed}.parquet")
                        out_path = runs_dir / out_name

                        if out_path.exists() and not args.force:
                            skipped += 1
                            if args.verbose:
                                print(f"  [skip] {pool} a={alpha:.2f} seed{seed} "
                                      f"{scenario}/{budget_pct}% (exists)")
                            continue

                        controller = EMASmoothedRLController(
                            model_path=str(model_path),
                            ema_alpha=alpha,
                            deterministic=True,
                            forecast_mode=forecast_mode,
                            noise_sigma=args.noise_sigma,
                            noise_rho=args.noise_rho,
                            noise_seed=noise_seed,
                            verbose=args.verbose,
                        )

                        print(f"  [run ] {pool} a={alpha:.2f} seed{seed} "
                              f"{scenario}/{budget_pct}%  ({model_path.parent.parent.name})")
                        run_season(
                            controller=controller,
                            terrain=terrain,
                            crop=crop,
                            climate=climate,
                            budget_total=budget_total,
                            output_path=out_path,
                            scenario_name=scenario,
                            seed=int(seed),
                            force=args.force,
                            verbose=args.verbose,
                        )
                        done += 1

        print(f"\n[ema sweep {pool}] {done} evaluated, {skipped} skipped. "
              f"Outputs: {RUNS_OUTPUT_DIR}/td3_v221c_ema_{pool_tag}_a*/")

    print("\nNext: python -m scripts.analysis.ema_pareto")


def main():
    p = argparse.ArgumentParser(
        description="EMA action-smoothing sweep over trained v2.21c TD3 checkpoints."
    )
    p.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS,
                   help=f'EMA smoothing weights in (0,1]. Default {DEFAULT_ALPHAS}.')
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                   help=f'Seeds to evaluate. Default {DEFAULT_SEEDS}.')
    p.add_argument('--pool', choices=['A', 'B', 'both'], default='both',
                   help="Which pool's checkpoints to use. "
                        "A = DEV {2002,2016,2023}, B = DEV {2002,2004,2013}, "
                        "both = sweep each pool separately (default).")
    p.add_argument('--scenario', choices=SCENARIOS_ALL + ['all'], default='all')
    p.add_argument('--budget', choices=[str(b) for b in BUDGET_LEVELS] + ['all'],
                   default='all')
    p.add_argument('--forecast', choices=['perfect', 'noisy'], default='perfect',
                   help="Forecast mode. Default 'perfect' (matches headline comparison).")
    p.add_argument('--noise-seed', type=int, default=42)
    p.add_argument('--noise-sigma', type=float, default=0.15)
    p.add_argument('--noise-rho', type=float, default=0.6)
    p.add_argument('--ckpt', nargs='+', default=None,
                   help='Explicit checkpoint(s) as seed=path (else auto-discovered).')
    p.add_argument('--force', action='store_true', help='Recompute existing cells.')
    p.add_argument('--quiet', dest='verbose', action='store_false', default=True,
                   help='Reduce per-day logging.')
    args = p.parse_args()
    run_sweep(args)


if __name__ == '__main__':
    main()
