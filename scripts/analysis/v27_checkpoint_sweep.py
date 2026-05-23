# =============================================================================
# scripts/analysis/v27_checkpoint_sweep.py
#
# Evaluate every saved checkpoint from a v2.7 training run on the 2024 wet
# scenario AT 100% BUDGET, using the canonical RLController + run_season
# pipeline (the same code path that produced the published v2.7 best_model
# results of 3434 kg/ha on wet/100%).
#
# Why this script exists
# ----------------------
# A previous ad-hoc evaluation that drove IrrigationEnv directly produced
# yields ~1000 kg/ha lower than the canonical pipeline. That gap was a
# harness artifact, not a property of the checkpoints themselves. This
# script eliminates the variable by reusing the exact runner that produced
# the published numbers.
#
# What it does
# ------------
# For each checkpoint .zip in the supplied directory:
#   1. Loads via RLController (auto-detects v2.6/v2.7/v2.8 critic arch).
#   2. Runs the full 93-day 2024 wet season at 100% budget via run_season,
#      writing the per-step trajectory + final_metrics JSON to a per-checkpoint
#      output file.
#   3. Reads back the JSON and prints yield_kg_ha alongside MPC reference.
#
# Also evaluates best_model.zip from the same training run if it exists,
# so you can verify reproduction of the canonical 3434 kg/ha number.
#
# Usage
# -----
#   python -m scripts.analysis.v27_checkpoint_sweep
#
# or with custom paths:
#
#   python -m scripts.analysis.v27_checkpoint_sweep \
#       --ckpt-dir results/rl/sac_v27_seed0_20260518_011229/checkpoints \
#       --best-model results/rl/sac_v27_seed0_20260518_011229/best_model/best_model.zip \
#       --out-dir results/analysis/v27_sweep \
#       --scenario wet --budget 100
# =============================================================================

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ── path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── defaults (override on CLI) ───────────────────────────────────────────────
DEFAULT_CKPT_DIR = (
    PROJECT_ROOT / 'results' / 'rl'
    / 'sac_v27_seed0_20260518_011229' / 'checkpoints'
)
DEFAULT_BEST_MODEL = (
    PROJECT_ROOT / 'results' / 'rl'
    / 'sac_v27_seed0_20260518_011229' / 'best_model' / 'best_model.zip'
)
DEFAULT_OUT_DIR = PROJECT_ROOT / 'results' / 'analysis' / 'v27_sweep'
DEFAULT_DEM = PROJECT_ROOT / 'gilan_farm.tif'

# 100% budget for rice = 484 mm of seasonal need.
CROP_FULL_BUDGET_MM = {'rice': 484.0}
BUDGET_LEVELS = {100: 1.00, 85: 0.85, 70: 0.70}


# ── helpers ──────────────────────────────────────────────────────────────────
def _step_number_from_filename(fname: str) -> int:
    """Extract the step count from 'sac_<tag>_seed<S>_<N>_steps.zip'.

    Tolerant to different naming conventions: locates the integer immediately
    preceding the literal '_steps' suffix.
    """
    m = re.search(r'_(\d+)_steps', fname)
    if m is None:
        raise ValueError(f"Cannot find step number in filename: {fname}")
    return int(m.group(1))


def _list_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Return all .zip checkpoints in ckpt_dir, sorted by step number."""
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    files = [p for p in ckpt_dir.iterdir() if p.suffix == '.zip']
    if not files:
        raise FileNotFoundError(f"No .zip files in {ckpt_dir}")
    files.sort(key=lambda p: _step_number_from_filename(p.name))
    return files


def _eval_one_checkpoint(
    model_path: Path,
    label: str,
    output_path: Path,
    scenario: str,
    budget_pct: int,
    deterministic: bool,
    force: bool,
):
    """Run one full-season evaluation via RLController + run_season.

    Returns the parsed final_metrics dict on success, or None on skip/failure.
    """
    # Imports inside the function so a syntax error elsewhere doesn't crash
    # before argparse has a chance to print help.
    from src.rl.runner import RLController
    from src.runner import run_season
    from soil_data import get_crop
    from src.terrain import load_terrain
    from climate_data import (
        load_cleaned_data, extract_scenario_by_name, SCENARIO_YEARS,
    )

    crop = get_crop('rice')
    terrain = load_terrain(str(DEFAULT_DEM))
    df_clim = load_cleaned_data()
    climate = extract_scenario_by_name(df_clim, scenario, crop)
    climate['year'] = SCENARIO_YEARS[scenario]
    budget_total = CROP_FULL_BUDGET_MM['rice'] * BUDGET_LEVELS[budget_pct]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        # Re-use cached parquet (and its JSON sibling) — print stored yield.
        json_path = output_path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                return meta.get('final_metrics')
            except Exception:
                pass

    controller = RLController(
        model_path=str(model_path),
        deterministic=deterministic,
        forecast_mode='perfect',
        verbose=False,
    )
    print(f"[{label}]  running {scenario}/{budget_pct}%  ({model_path.name})")
    status = run_season(
        controller=controller,
        terrain=terrain,
        crop=crop,
        climate=climate,
        budget_total=budget_total,
        output_path=output_path,
        scenario_name=scenario,
        seed=0,
        force=force,
        verbose=False,
    )
    if status != 'completed':
        print(f"  ! run_season returned {status}")
        return None

    json_path = output_path.with_suffix('.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return meta.get('final_metrics')


def _load_mpc_reference(scenario: str, budget_pct: int) -> float | None:
    """Look up the published MPC Hp=3 perfect yield for this cell."""
    candidates = [
        PROJECT_ROOT / 'results' / 'runs'
        / f'mpc_perfect_{scenario}_rice_{budget_pct}pct_Hp3.json',
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    return float(json.load(f)['final_metrics']['yield_kg_ha'])
            except Exception:
                pass
    return None


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate every v2.7 checkpoint on a held-out scenario '
                    'using the canonical RLController + run_season pipeline.'
    )
    parser.add_argument('--ckpt-dir',   type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument('--best-model', type=Path, default=DEFAULT_BEST_MODEL,
                        help='Optional best_model.zip to also evaluate '
                             '(for canonical-number reproduction).')
    parser.add_argument('--out-dir',    type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--scenario',   choices=['dry', 'moderate', 'wet'],
                        default='wet')
    parser.add_argument('--budget',     type=int, choices=[100, 85, 70],
                        default=100)
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic policy at inference '
                             '(default: True; matches canonical eval).')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if output parquet/json exist.')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Checkpoint dir:    {args.ckpt_dir}")
    print(f"Best-model file:   {args.best_model} "
          f"({'found' if args.best_model.exists() else 'NOT FOUND — skipping'})")
    print(f"Output dir:        {args.out_dir}")
    print(f"Scenario / budget: {args.scenario} / {args.budget}%")
    print(f"Deterministic:     {args.deterministic}")
    print()

    # MPC reference yield for context.
    mpc_y = _load_mpc_reference(args.scenario, args.budget)
    if mpc_y is not None:
        print(f"MPC Hp=3 reference: {mpc_y:.1f} kg/ha "
              f"({args.scenario}/{args.budget}%)")
    else:
        print("MPC reference: not found (will skip relative-gap column)")
    print()

    rows: list[tuple[str, int, dict | None]] = []

    # 1. Best model first — confirms harness reproduces canonical yield.
    if args.best_model.exists():
        out_path = args.out_dir / (
            f'sweep_best_model_{args.scenario}_{args.budget}pct.parquet'
        )
        metrics = _eval_one_checkpoint(
            model_path=args.best_model,
            label='best_model',
            output_path=out_path,
            scenario=args.scenario,
            budget_pct=args.budget,
            deterministic=args.deterministic,
            force=args.force,
        )
        rows.append(('best_model', -1, metrics))

    # 2. Every numbered checkpoint, in step order.
    ckpts = _list_checkpoints(args.ckpt_dir)
    for ckpt in ckpts:
        step = _step_number_from_filename(ckpt.name)
        out_path = args.out_dir / (
            f'sweep_step{step:07d}_{args.scenario}_{args.budget}pct.parquet'
        )
        metrics = _eval_one_checkpoint(
            model_path=ckpt,
            label=f'step {step:>6}',
            output_path=out_path,
            scenario=args.scenario,
            budget_pct=args.budget,
            deterministic=args.deterministic,
            force=args.force,
        )
        rows.append((ckpt.name, step, metrics))

    # 3. Summary table.
    print()
    print('=' * 88)
    print(f'{"label":<42} {"step":>8} {"yield":>10} {"water":>9} {"vs_MPC":>10}')
    print('-' * 88)

    best_yield = float('-inf')
    best_label = None
    for label, step, m in rows:
        if m is None:
            print(f'{label:<42} {step if step >= 0 else "—":>8} '
                  f'{"FAILED":>10}')
            continue
        y = m['yield_kg_ha']
        w = m['water_used_mm']
        vs_mpc = (f'{(y/mpc_y - 1.0) * 100:>+9.2f}%'
                  if mpc_y else '       —')
        step_str = str(step) if step >= 0 else '—'
        print(f'{label:<42} {step_str:>8} {y:>10.1f} {w:>9.1f} {vs_mpc:>10}')
        if y > best_yield:
            best_yield = y
            best_label = label
    print('=' * 88)

    if best_label is not None:
        print(f'\nBest yield: {best_label}  →  {best_yield:.1f} kg/ha')
        if mpc_y is not None:
            print(f'Gap vs MPC: {(best_yield/mpc_y - 1.0) * 100:+.2f}%')

    print(f'\nAll trajectories written to: {args.out_dir}')


if __name__ == '__main__':
    main()
