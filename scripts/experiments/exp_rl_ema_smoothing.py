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

from climate_data import SCENARIO_YEARS                       # noqa: E402
from soil_data import get_crop                                # noqa: E402
from src.terrain import load_terrain                          # noqa: E402

SCENARIOS_ALL = list(SCENARIO_YEARS.keys())                   # ['dry','moderate','wet']
BUDGET_LEVELS = {100: 1.00, 85: 0.85, 70: 0.70}
CROP_FULL_BUDGET_MM = {'rice': 484.0}

DEM_PATH        = PROJECT_ROOT / 'gilan_farm.tif'
RL_OUTPUT_DIR   = PROJECT_ROOT / 'results' / 'rl'
RUNS_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'runs'

# Default smoothing grid: 1.0 (= baseline anchor) down to heavy smoothing.
DEFAULT_ALPHAS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
DEFAULT_SEEDS  = [0, 1, 2]


def alpha_tag(alpha: float) -> str:
    """Filesystem-safe tag for an alpha value, e.g. 1.0 -> '1p00', 0.3 -> '0p30'."""
    return f"{alpha:.2f}".replace('.', 'p')


def discover_v221c_checkpoints() -> dict:
    """Locate the v2.21c best_model.zip for each available seed.

    Globs results/rl/td3_v221c_termyield_seed<S>_*/best_model/best_model.zip
    (the layout written by train_v221c_td3).  Falls back to the *_final.zip if
    a best_model.zip is absent.

    Returns
    -------
    dict
        {seed:int -> Path-to-checkpoint}
    """
    found: dict = {}
    for run_dir in sorted(RL_OUTPUT_DIR.glob('td3_v221c_termyield_seed*')):
        name = run_dir.name
        try:
            seed = int(name.split('seed')[1].split('_')[0])
        except (IndexError, ValueError):
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
    from src.runner import run_season
    from climate_data import load_cleaned_data, extract_scenario_by_name

    crop         = get_crop('rice')
    terrain      = load_terrain(str(DEM_PATH))
    df_climate   = load_cleaned_data()
    full_need_mm = CROP_FULL_BUDGET_MM['rice']

    scenarios   = SCENARIOS_ALL if args.scenario == 'all' else [args.scenario]
    budget_pcts = list(BUDGET_LEVELS) if args.budget == 'all' else [int(args.budget)]
    alphas      = [float(a) for a in args.alphas]

    # Resolve checkpoints: explicit overrides win, otherwise auto-discover.
    ckpts = discover_v221c_checkpoints()
    ckpts.update(parse_ckpt_overrides(args.ckpt))
    if not ckpts:
        raise SystemExit(
            "No v2.21c checkpoints found under results/rl/td3_v221c_termyield_seed*/ "
            "and none given via --ckpt. Train them first (run_seeds_v221c) or pass "
            "--ckpt seed=path."
        )

    seeds = [s for s in args.seeds if s in ckpts]
    missing = [s for s in args.seeds if s not in ckpts]
    if missing:
        print(f"[warn] no checkpoint for seed(s) {missing}; skipping them. "
              f"Available seeds: {sorted(ckpts)}")
    if not seeds:
        raise SystemExit("None of the requested seeds have a checkpoint.")

    forecast_mode = args.forecast
    noise_seed    = args.noise_seed

    print("\nEMA smoothing sweep")
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
        runs_dir = RUNS_OUTPUT_DIR / f"td3_v221c_ema_a{tag}"
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
                    out_name = (f"td3_v221c_ema_a{tag}_{fc_tag}_det_"
                                f"{scenario}_rice_{budget_pct}pct_seed{seed}.parquet")
                    out_path = runs_dir / out_name

                    if out_path.exists() and not args.force:
                        skipped += 1
                        print(f"  [skip] a={alpha:.2f} seed{seed} "
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

                    print(f"  [run ] a={alpha:.2f} seed{seed} "
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

    print(f"\n[ema sweep] complete: {done} evaluated, {skipped} skipped "
          f"(already present). Outputs under {RUNS_OUTPUT_DIR}/td3_v221c_ema_a*/")
    print("Next: python -m scripts.analysis.ema_pareto")


def main():
    p = argparse.ArgumentParser(
        description="EMA action-smoothing sweep over trained v2.21c TD3 checkpoints."
    )
    p.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS,
                   help=f'EMA smoothing weights in (0,1]. Default {DEFAULT_ALPHAS}.')
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                   help=f'Seeds to evaluate. Default {DEFAULT_SEEDS}.')
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
