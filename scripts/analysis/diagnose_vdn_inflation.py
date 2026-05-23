# =============================================================================
# scripts/analysis/diagnose_vdn_inflation.py
#
# Diagnose whether the VDN sum-aggregation produces Q-value bias amplification
# during v2.7 training, by inspecting per-agent local_q outputs of every saved
# checkpoint on REALISTIC observations from the 2024 wet scenario.
#
# Why this script exists
# ----------------------
# An earlier diagnostic used a zero observation as input, which gave zero
# per-agent variation (the shared critic produces identical outputs for
# identical inputs). The result was uninterpretable for the question of
# "does the network actually differentiate per agent?".
#
# This version uses observations sampled from a real 2024-wet rollout using
# each checkpoint's own policy, so the local_q values reflect what the critic
# would actually predict during inference. Per-agent variation in local_q is
# now meaningful and exposes whether (a) the shared network learns
# agent-specific values from the per-agent features, and (b) any systematic
# bias in local_q is amplified by the N=130 summation.
#
# Outputs for each checkpoint:
#   step          training step
#   |Q_total|     |sum_n local_q_n|  — this is what SAC's actor objective sees
#   |Q_mean|      |mean_n local_q_n| — what Q_total would be with mean-aggregation
#   local_mean    mean of local_q across N=130 agents
#   local_std     std of local_q across N=130 agents (0 ⇒ no per-agent diff)
#   local_max     max of |local_q| across N=130 agents
#   bias_ratio    |Q_total| / |Q_mean| (≈ N when local_q is uniform)
#
# Usage
# -----
#   python -m scripts.analysis.diagnose_vdn_inflation
#
# or with custom paths:
#
#   python -m scripts.analysis.diagnose_vdn_inflation \
#       --ckpt-dir results/rl/sac_v27_seed0_20260518_011229/checkpoints \
#       --scenario wet --budget 100 --n-samples 8
# =============================================================================

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch


# ── path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CKPT_DIR = (
    PROJECT_ROOT / 'results' / 'rl'
    / 'sac_v27_seed0_20260518_011229' / 'checkpoints'
)
DEFAULT_DEM = PROJECT_ROOT / 'gilan_farm.tif'

CROP_FULL_BUDGET_MM = {'rice': 484.0}
BUDGET_LEVELS = {100: 1.00, 85: 0.85, 70: 0.70}


# ── helpers ──────────────────────────────────────────────────────────────────
def _step_number_from_filename(fname: str) -> int:
    m = re.search(r'_(\d+)_steps', fname)
    if m is None:
        raise ValueError(f"Cannot find step number in filename: {fname}")
    return int(m.group(1))


def _list_checkpoints(ckpt_dir: Path) -> list[Path]:
    files = [p for p in ckpt_dir.iterdir() if p.suffix == '.zip']
    if not files:
        raise FileNotFoundError(f"No .zip files in {ckpt_dir}")
    files.sort(key=lambda p: _step_number_from_filename(p.name))
    return files


def _sample_observations(
    model_path: Path,
    scenario: str,
    budget_pct: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one closed-loop episode and return (obs_batch, action_batch).

    Samples `n_samples` (observation, action) pairs from across the season at
    evenly-spaced day indices, so the diagnostic sees the critic's behavior
    on the state distribution the policy actually visits.

    Returns
    -------
    obs_batch    : (n_samples, obs_dim) float32
    action_batch : (n_samples, N_agents) float32
    """
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

    # Build a controller that wraps the checkpoint, then drive it ourselves
    # so we can capture per-step (obs, action) without modifying run_season.
    controller = RLController(
        model_path=str(model_path),
        deterministic=True,
        forecast_mode='perfect',
        verbose=False,
    )

    # Mirror run_season's reset + step loop, but record obs/action.
    from abm import CropSoilABM   # abm.py lives at the repo root

    abm = CropSoilABM(
        gamma_flat=terrain['gamma_flat'],
        sends_to=terrain['sends_to'],
        Nr=terrain['Nr'],
        theta=crop,
        N=terrain['N'],
        runoff_mode='cascade',
        elevation=terrain['elevation_flat'],
    )
    abm.reset()
    controller.reset(
        terrain=terrain, crop=crop,
        season_days=crop['season_days'],
        budget_total=budget_total,
        scenario_name=scenario,
    )

    season_days = crop['season_days']
    budget_remaining = float(budget_total)
    obs_list:    list[np.ndarray] = []
    action_list: list[np.ndarray] = []

    for day in range(season_days):
        # _get_state() returns {'x1','x2','x3','x4','x5'}; build directly
        # so we don't depend on the leading-underscore name being stable.
        state = {
            'x1': abm.x1.copy(),
            'x2': abm.x2.copy(),
            'x3': abm.x3.copy(),
            'x4': abm.x4.copy(),
            'x5': abm.x5.copy(),
        }
        obs = controller._build_obs(day, state, budget_remaining)
        action_mm = controller.step(
            day=day, state=state,
            climate_today=None,                    # unused inside controller.step
            budget_remaining=budget_remaining,
        )
        # Record raw normalized action in [0, 1] (the form the critic was trained on).
        from src.rl.gym_env import UB_MM as _UB
        action_norm = (action_mm / _UB).astype(np.float32)
        obs_list.append(obs.astype(np.float32))
        action_list.append(action_norm)

        # advance ABM with the chosen action
        climate_today = {
            'rainfall':  float(climate['rainfall'][day]),
            'temp_mean': float(climate['temp_mean'][day]),
            'temp_max':  float(climate['temp_max'][day]),
            'radiation': float(climate['radiation'][day]),
            'ET':        float(climate['ET'][day]),
        }
        abm.step(action_mm, climate_today)
        budget_remaining = max(
            budget_remaining - float(np.mean(action_mm)), 0.0,
        )

    # Evenly-spaced samples across the season.
    indices = np.linspace(0, season_days - 1, n_samples).astype(int)
    obs_batch = np.stack([obs_list[i] for i in indices], axis=0)
    action_batch = np.stack([action_list[i] for i in indices], axis=0)
    return obs_batch, action_batch


def _compute_local_q(
    model,
    obs_batch_np: np.ndarray,
    action_batch_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the factorized critic forward and return per-agent local_q.

    Returns
    -------
    local_q_qf0 : (n_samples, N_agents) — first critic head's local_q
    local_q_qf1 : (n_samples, N_agents) — second critic head's local_q
    """
    from src.rl.networks import V27_N_AGENT_FEATURES, V27_PER_AGENT_CRITIC_INPUT_DIM

    obs = torch.as_tensor(obs_batch_np,    dtype=torch.float32)
    action = torch.as_tensor(action_batch_np, dtype=torch.float32)

    with torch.no_grad():
        features = model.critic.extract_features(
            obs, model.critic.features_extractor,
        )

    B = obs.shape[0]
    N = action.shape[1]
    F = V27_N_AGENT_FEATURES

    local_obs = features[:, : F * N].reshape(B, N, F)
    global_block = features[:, F * N:]
    global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)
    local_actions = action.reshape(B, N, 1)

    local_inputs = torch.cat(
        [local_obs, global_expanded, local_actions], dim=-1,
    )
    local_inputs_flat = local_inputs.reshape(
        B * N, V27_PER_AGENT_CRITIC_INPUT_DIM,
    )

    results = []
    for q_net in (model.critic.qf0, model.critic.qf1):
        # _FactorizedQNet inherits from nn.Sequential; forward via the
        # nn.Sequential method to bypass the per-batch reshape inside its
        # own forward(), since we already supply the flattened tensor.
        with torch.no_grad():
            local_q = torch.nn.Sequential.forward(q_net, local_inputs_flat)
        results.append(local_q.reshape(B, N).cpu().numpy())
    return results[0], results[1]


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='VDN bias-amplification diagnostic for v2.7 checkpoints.'
    )
    parser.add_argument('--ckpt-dir', type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument('--scenario', choices=['dry', 'moderate', 'wet'],
                        default='wet')
    parser.add_argument('--budget',   type=int, choices=[100, 85, 70],
                        default=100)
    parser.add_argument('--n-samples', type=int, default=8,
                        help='Number of (obs, action) pairs sampled across '
                             'the season for each checkpoint.')
    args = parser.parse_args()

    # Lazy import of _load_sac_model so argparse can print help even if
    # something in the SB3 stack is missing.
    from src.rl.runner import _load_sac_model

    ckpts = _list_checkpoints(args.ckpt_dir)
    print(f"Diagnosing {len(ckpts)} checkpoints from {args.ckpt_dir}")
    print(f"State distribution from rollouts on: "
          f"{args.scenario}/{args.budget}%   "
          f"(n_samples={args.n_samples} per checkpoint)\n")

    header = (
        f"{'step':>8} {'|Q_total|':>11} {'|Q_mean|':>11} "
        f"{'local_mean':>11} {'local_std':>11} {'local_max':>11} "
        f"{'bias_ratio':>11}"
    )
    print(header)
    print('-' * len(header))

    for ckpt in ckpts:
        step = _step_number_from_filename(ckpt.name)

        # Sample states using THIS checkpoint's policy, so each row reflects
        # the critic's behavior on the state distribution its own actor visits.
        obs_batch, action_batch = _sample_observations(
            model_path=ckpt,
            scenario=args.scenario,
            budget_pct=args.budget,
            n_samples=args.n_samples,
        )

        # Reload via the canonical loader (handles v2.6/2.7/2.8 arch detection).
        model, _arch_label, _layout = _load_sac_model(ckpt, device='cpu')

        local_q_qf0, _local_q_qf1 = _compute_local_q(
            model, obs_batch, action_batch,
        )
        # Use the first critic head; min(Q1, Q2) is what SAC bootstraps from
        # but for the bias diagnostic Q1 alone is sufficient and cleaner.

        # Aggregate across samples (mean over batch) and across agents.
        q_total_per_sample = local_q_qf0.sum(axis=1)             # (B,)
        q_mean_per_sample = local_q_qf0.mean(axis=1)            # (B,)
        local_mean_per_sample = local_q_qf0.mean(axis=1)         # (B,)
        local_std_per_sample = local_q_qf0.std(axis=1)          # (B,)
        local_max_per_sample = np.abs(local_q_qf0).max(axis=1)  # (B,)

        # Reported numbers: mean over the n_samples states in the rollout.
        Q_total_abs = float(np.mean(np.abs(q_total_per_sample)))
        Q_mean_abs = float(np.mean(np.abs(q_mean_per_sample)))
        local_mean = float(np.mean(local_mean_per_sample))
        local_std = float(np.mean(local_std_per_sample))
        local_max = float(np.mean(local_max_per_sample))
        bias_ratio = Q_total_abs / \
            Q_mean_abs if Q_mean_abs > 1e-9 else float('nan')

        print(
            f"{step:>8} {Q_total_abs:>11.2f} {Q_mean_abs:>11.4f} "
            f"{local_mean:>+11.4f} {local_std:>11.5f} {local_max:>11.4f} "
            f"{bias_ratio:>11.2f}"
        )

    print()
    print("Interpretation:")
    print("  local_std ≈ 0      → shared network gives identical output per agent")
    print("                       (per-agent features are not exploited at all)")
    print("  local_std > 0      → critic learns agent-specific values")
    print("  bias_ratio ≈ 130   → sum is N × mean (uniform local_q ⇒ N× amplification)")
    print("  bias_ratio < 130   → some agents cancel (uncorrelated noise)")


if __name__ == '__main__':
    main()
