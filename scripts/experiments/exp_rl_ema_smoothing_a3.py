# =============================================================================
# scripts/experiments/exp_rl_ema_smoothing_a3.py
# EMA action-smoothing sweep for the alpha3 (drought-penalty r3) sweep winners.
#
# WHY A SEPARATE SCRIPT
# ---------------------
# scripts/experiments/exp_rl_ema_smoothing.py auto-discovers the v2.21c
# checkpoints (glob 'td3_v221c_termyield_seed*') and writes every alpha into a
# single 'td3_v221c_ema_pool*_a*' directory tree. That is perfect for ONE model
# but collides if you sweep TWO different models. This thin wrapper reuses the
# exact same proven core --
#   * the causal EMA filter            (src.rl.ema_smoothing.EMASmoothedRLController)
#   * the season runner + checkpointer (src.sim.runner.run_season)
#   * the identical 9-cell grid, budgets, forecast handling, parquet/JSON format
# -- but lets you point it at an arbitrary trained run directory and gives each
# model its OWN output tree (results/runs/<label>_ema_a<tag>/), so the two
# alpha3 winners never overwrite each other and the analysis can treat them as
# two separate Pareto frontiers.
#
# WHAT alpha3 IS (so the labels are meaningful)
# ---------------------------------------------
# alpha3 is the weight of the drought-stress reward term r3 in src/rl/gym_env.py:
#     r3 = -alpha3 * mean(max(ST - x1, 0)) / (ST - WP)
# i.e. how hard the policy is penalised for letting root-zone moisture x1 fall
# below the drought-stress threshold ST (124 mm; WP = 60 mm). You swept alpha3
# in [0.1, 1.5] and selected 0.5 and 1.15. EMA smoothing here is a *post-hoc*
# low-pass on each of those already-trained policies' action stream -- NO
# retraining -- exactly as was done for v2.21c.
#
# RESUMABILITY / "nothing needs rerunning"
# ----------------------------------------
# A cell is skipped if its .parquet already exists (per-cell resume), and
# run_season additionally checkpoints within each season (default every 10
# days) so a crash mid-season resumes from the last checkpoint. Re-running the
# command after any interruption only computes what is missing.
#
# USAGE
# -----
#   # Both winners, full default sweep (alpha in {1.0,0.7,0.5,0.3,0.2,0.1}),
#   # all 9 cells (3 scenarios x 3 budgets), perfect forecast:
#   python -m scripts.experiments.exp_rl_ema_smoothing_a3 \
#       --model a3_0p50=results/rl_model/rl_sweep_alpha3/td3_a3-0.5_seed0 \
#       --model a3_1p15=results/rl_model/rl_sweep_alpha3/td3_a3-1.15_seed0
#
#   # Quick smoke test: one model, one alpha, one cell
#   python -m scripts.experiments.exp_rl_ema_smoothing_a3 \
#       --model a3_0p50=results/rl_model/rl_sweep_alpha3/td3_a3-0.5_seed0 \
#       --alphas 0.3 --scenario dry --budget 100
#
#   # A path can also point straight at a .zip checkpoint instead of a run dir:
#   python -m scripts.experiments.exp_rl_ema_smoothing_a3 \
#       --model a3_1p15=results/rl_model/rl_sweep_alpha3/td3_a3-1.15_seed0/best_model/best_model.zip
#
# After the sweep, build the per-model frontiers with:
#   python -m scripts.analysis.ema_pareto_a3 --labels a3_0p50 a3_1p15
# =============================================================================

import argparse
import json
import re
import sys
from pathlib import Path

# Windows / non-UTF8 console safety (same pattern as exp_rl.py / the v2.21c sweep)
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
from src.model.terrain import load_terrain                             # noqa: E402

SCENARIOS_ALL = list(SCENARIO_YEARS.keys())                  # ['dry','moderate','wet']
BUDGET_LEVELS = {100: 1.00, 85: 0.85, 70: 0.70}
CROP_FULL_BUDGET_MM = {'rice': 484.0}                        # matches gym_env FULL_SEASON_NEED_MM

DEM_PATH        = PROJECT_ROOT / 'data' / 'gilan_farm.tif'
RUNS_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'runs'

# Same smoothing grid as the v2.21c sweep, for apples-to-apples comparability.
# alpha -> first-order time constant tau = -1/ln(1-alpha) days:
#   1.0->0.0 (no smoothing), 0.7->0.8, 0.5->1.4, 0.3->2.8, 0.2->4.5, 0.1->9.5
DEFAULT_ALPHAS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]


def alpha_tag(alpha: float) -> str:
    """Filesystem-safe tag for an alpha value, e.g. 1.0 -> '1p00', 0.3 -> '0p30'."""
    return f"{alpha:.2f}".replace('.', 'p')


def parse_model_args(items):
    """Parse repeated --model label=path into [(label, Path), ...]."""
    out = []
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f"--model expects label=path, got {item!r}")
        label, path = item.split('=', 1)
        label = label.strip()
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', label):
            raise SystemExit(f"--model label must be filesystem-safe, got {label!r}")
        out.append((label, Path(path)))
    return out


def resolve_checkpoint(path: Path) -> Path:
    """Resolve a run dir (or direct .zip) to the checkpoint .zip to load.

    Preference order inside a run directory: best_model/best_model.zip, then
    any *_final.zip. Raises if nothing usable is found.
    """
    if path.is_file() and path.suffix == '.zip':
        return path
    if path.is_dir():
        best = path / 'best_model' / 'best_model.zip'
        if best.exists():
            return best
        finals = sorted(path.glob('*_final.zip'))
        if finals:
            return finals[0]
        # last resort: a periodic checkpoint with the largest step count
        ckpts = sorted(path.glob('checkpoints/*_steps.zip'))
        if ckpts:
            def _steps(p):
                m = re.search(r'(\d+)_steps', p.name)
                return int(m.group(1)) if m else -1
            return max(ckpts, key=_steps)
    raise SystemExit(
        f"[error] No checkpoint found at {path}. Expected a run directory "
        f"containing best_model/best_model.zip (or *_final.zip), or a direct "
        f".zip path."
    )


def seed_from_name(name: str, default: int) -> int:
    """Extract the training seed from a run/zip name token 'seed<d>'."""
    m = re.search(r'seed(\d+)', name)
    return int(m.group(1)) if m else default


def read_manifest_alpha3(run_dir: Path):
    """Best-effort read of reward_alpha3 from a run's manifest.json (or None)."""
    mp = run_dir / 'manifest.json'
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding='utf-8')).get('reward_alpha3')
    except Exception:
        return None


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

    forecast_mode = args.forecast
    noise_seed    = args.noise_seed

    models = parse_model_args(args.model)
    if not models:
        raise SystemExit("Pass at least one --model label=path (run dir or .zip).")

    for label, model_path_arg in models:
        ckpt = resolve_checkpoint(model_path_arg)
        run_dir = ckpt.parent.parent if ckpt.name == 'best_model.zip' else ckpt.parent
        seed = args.seed if args.seed is not None else seed_from_name(
            model_path_arg.name + ' ' + run_dir.name, default=0)
        a3_val = read_manifest_alpha3(run_dir)

        print(f"\n{'='*72}")
        print(f"EMA smoothing sweep -- model '{label}'")
        print(f"{'='*72}")
        print(f"  checkpoint:   {ckpt}")
        print(f"  training seed:{seed}"
              + (f"   (manifest reward_alpha3 = {a3_val})" if a3_val is not None else ""))
        print(f"  alphas:       {alphas}")
        print(f"  scenarios:    {scenarios}")
        print(f"  budgets:      {budget_pcts}")
        print(f"  forecast:     {forecast_mode}"
              + (f"  (noise_seed={noise_seed})" if forecast_mode == 'noisy' else ""))
        n_total = len(alphas) * len(scenarios) * len(budget_pcts)
        print(f"  total cells to evaluate: {n_total}\n")

        done = skipped = 0
        for alpha in alphas:
            tag = alpha_tag(alpha)
            runs_dir = RUNS_OUTPUT_DIR / f"{label}_ema_a{tag}"
            runs_dir.mkdir(parents=True, exist_ok=True)

            for scenario in scenarios:
                climate = extract_scenario_by_name(df_climate, scenario, crop)
                climate['year'] = SCENARIO_YEARS[scenario]

                for budget_pct in budget_pcts:
                    budget_total = full_need_mm * BUDGET_LEVELS[budget_pct]

                    fc_tag = ('perfect' if forecast_mode == 'perfect'
                              else f"noisy_ns{noise_seed}")
                    out_name = (f"{label}_ema_a{tag}_{fc_tag}_det_"
                                f"{scenario}_rice_{budget_pct}pct_seed{seed}.parquet")
                    out_path = runs_dir / out_name

                    if out_path.exists() and not args.force:
                        skipped += 1
                        if args.verbose:
                            print(f"  [skip] {label} a={alpha:.2f} "
                                  f"{scenario}/{budget_pct}% (exists)")
                        continue

                    controller = EMASmoothedRLController(
                        model_path=str(ckpt),
                        ema_alpha=alpha,
                        deterministic=True,
                        forecast_mode=forecast_mode,
                        noise_sigma=args.noise_sigma,
                        noise_rho=args.noise_rho,
                        noise_seed=noise_seed,
                        verbose=args.verbose,
                    )

                    print(f"  [run ] {label} a={alpha:.2f} "
                          f"{scenario}/{budget_pct}%")
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

        print(f"\n[ema sweep {label}] {done} evaluated, {skipped} skipped. "
              f"Outputs: {RUNS_OUTPUT_DIR}/{label}_ema_a*/")

    labels = " ".join(lbl for lbl, _ in models)
    print(f"\nNext: python -m scripts.analysis.ema_pareto_a3 --labels {labels}")


def main():
    p = argparse.ArgumentParser(
        description="EMA action-smoothing sweep over arbitrary trained TD3 "
                    "checkpoints (built for the alpha3 sweep winners)."
    )
    p.add_argument('--model', nargs='+', required=True,
                   help="One or more label=path. path = a training run directory "
                        "(containing best_model/best_model.zip) or a direct .zip. "
                        "label becomes the output-dir prefix, e.g. "
                        "a3_0p50=results/rl_model/rl_sweep_alpha3/td3_a3-0.5_seed0")
    p.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS,
                   help=f'EMA smoothing weights in (0,1]. Default {DEFAULT_ALPHAS}.')
    p.add_argument('--seed', type=int, default=None,
                   help="Override the seed recorded in output filenames / passed to "
                        "run_season. Default: parsed from the model dir name "
                        "('seed<d>'), else 0.")
    p.add_argument('--scenario', choices=SCENARIOS_ALL + ['all'], default='all')
    p.add_argument('--budget', choices=[str(b) for b in BUDGET_LEVELS] + ['all'],
                   default='all')
    p.add_argument('--forecast', choices=['perfect', 'noisy'], default='perfect',
                   help="Forecast mode. Default 'perfect' (matches headline comparison).")
    p.add_argument('--noise-seed', type=int, default=42)
    p.add_argument('--noise-sigma', type=float, default=0.15)
    p.add_argument('--noise-rho', type=float, default=0.6)
    p.add_argument('--force', action='store_true', help='Recompute existing cells.')
    p.add_argument('--quiet', dest='verbose', action='store_false', default=True,
                   help='Reduce per-day logging.')
    args = p.parse_args()
    run_sweep(args)


if __name__ == '__main__':
    main()
