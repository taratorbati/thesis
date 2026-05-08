# =============================================================================
# src/rl/train.py
# SAC training loop using Stable-Baselines3 with the CTDE policy.
#
# Design decisions (Chapter 4 of the thesis):
#   - Algorithm: SAC (entropy-regularized, continuous action)
#   - Policy: CTDESACPolicy — shared-parameter actor + centralized critic
#   - Reward: exact negation of the MPC path cost at alpha*
#     (constructed inside src/rl/gym_env.py)
#   - Multiple seeds for statistical robustness (3 minimum, 5 ideal)
#
# Kaggle deployment:
#   - 30 GPU-hours/week, 9-hour session limit
#   - Checkpoint every 10k steps
#   - save_replay_buffer=True is REQUIRED for safe session resumption.
#     SAC is off-policy; resuming with an empty replay buffer causes
#     catastrophic forgetting on the first batch update.
#
# Hyperparameters (from Haarnoja et al. 2018/2019):
#   - Learning rate 3e-4
#   - Batch size 256
#   - Gamma 0.99
#   - Tau 0.005
#   - Auto entropy tuning, target entropy = -dim(action)
# =============================================================================

import os
import json
import time
from pathlib import Path

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from src.rl.gym_env import IrrigationEnv
from src.rl.networks import CTDESACPolicy, make_sac_policy_kwargs


# ── Hyperparameters ──────────────────────────────────────────────────────────

DEFAULT_HP = {
    'learning_rate':    3e-4,       # Haarnoja et al. 2018
    'batch_size':       256,        # SAC default
    'buffer_size':      100_000,    # ~660 dims × 100k ≈ 0.5 GB on disk
    'gamma':            0.99,       # episodic, 93-step
    'tau':              0.005,      # soft target update
    'ent_coef':         'auto',     # automatic entropy tuning
    'learning_starts':  1000,       # random exploration steps
    'train_freq':       1,          # update every env step
    'gradient_steps':   1,          # one gradient step per env step
    'target_entropy':   'auto',     # -dim(action) = -130
}

# Architecture (per-agent shared actor + centralized critic)
ACTOR_HIDDEN = (128, 128)
CRITIC_HIDDEN = (256, 256)


def train_sac(
    scenario='dry',
    budget_pct=100,
    total_timesteps=500_000,
    seed=0,
    output_dir='results/rl',
    dem_path='gilan_farm.tif',
    checkpoint_freq=10_000,
    eval_freq=10_000,
    resume_path=None,
    hp_overrides=None,
    verbose=1,
):
    """Train a SAC agent on the irrigation environment.

    Parameters
    ----------
    scenario : str
        'dry' or 'wet'.
    budget_pct : int
        100, 85, or 70.
    total_timesteps : int
    seed : int
    output_dir : str
    dem_path : str
    checkpoint_freq : int
        Save model + replay buffer every N env steps.
    eval_freq : int
        Evaluate on a separate env every N env steps.
    resume_path : str or None
        Path to a .zip checkpoint to resume from. If a matching
        '_replay_buffer.pkl' exists alongside, it will be loaded too.
    hp_overrides : dict or None
        Override default hyperparameters.
    verbose : int
        0 = silent, 1 = info, 2 = debug.

    Returns
    -------
    model : SAC
    """
    output_dir = Path(output_dir)
    run_name = f"sac_{scenario}_{budget_pct}pct_seed{seed}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = run_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Merge hyperparameters
    hp = {**DEFAULT_HP, **(hp_overrides or {})}

    # ── Environments ──────────────────────────────────────────────────────

    train_env = Monitor(
        IrrigationEnv(
            scenario=scenario,
            budget_pct=budget_pct,
            dem_path=dem_path,
            seed=seed,
        ),
        filename=str(log_dir / 'train'),
    )

    eval_env = Monitor(
        IrrigationEnv(
            scenario=scenario,
            budget_pct=budget_pct,
            dem_path=dem_path,
            seed=seed + 1000,
        ),
        filename=str(log_dir / 'eval'),
    )

    # ── Policy kwargs ─────────────────────────────────────────────────────

    policy_kwargs = make_sac_policy_kwargs(
        N=train_env.unwrapped.N,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
    )

    # ── Create or resume model ────────────────────────────────────────────

    if resume_path is not None and Path(resume_path).exists():
        if verbose:
            print(f"Resuming from {resume_path}")
        model = SAC.load(
            resume_path,
            env=train_env,
            device='auto',
            custom_objects={'policy_class': CTDESACPolicy},
        )
        model.set_env(train_env)

        # Restore replay buffer if available alongside the checkpoint.
        # SB3 saves it as <checkpoint>_replay_buffer.pkl when
        # save_replay_buffer=True was used at save time.
        rb_path = Path(resume_path).with_suffix('').name + '_replay_buffer.pkl'
        rb_full = Path(resume_path).parent / rb_path
        if rb_full.exists():
            if verbose:
                print(f"Loading replay buffer from {rb_full}")
            model.load_replay_buffer(str(rb_full))
        else:
            print(f"WARNING: No replay buffer found at {rb_full}. "
                  f"Resuming with an empty buffer will cause catastrophic "
                  f"forgetting. Consider restarting training instead.")
    else:
        model = SAC(
            policy=CTDESACPolicy,                  # CTDE shared-actor architecture
            env=train_env,
            learning_rate=hp['learning_rate'],
            batch_size=hp['batch_size'],
            buffer_size=hp['buffer_size'],
            gamma=hp['gamma'],
            tau=hp['tau'],
            ent_coef=hp['ent_coef'],
            learning_starts=hp['learning_starts'],
            train_freq=hp['train_freq'],
            gradient_steps=hp['gradient_steps'],
            policy_kwargs=policy_kwargs,
            seed=seed,
            device='auto',
            verbose=verbose,
            tensorboard_log=str(log_dir),
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    # save_replay_buffer=True is REQUIRED for SAC session resumption.
    # The replay buffer is ~500 MB at buffer_size=100k and obs_dim=660,
    # which fits comfortably in Kaggle's 20 GB working directory.
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=run_name,
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / 'best_model'),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        verbose=verbose,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # ── Save config ───────────────────────────────────────────────────────

    config = {
        'scenario': scenario,
        'budget_pct': budget_pct,
        'seed': seed,
        'total_timesteps': total_timesteps,
        'policy_class': 'CTDESACPolicy',
        'hyperparameters': {k: str(v) if not isinstance(v, (int, float)) else v
                            for k, v in hp.items()},
        'actor_hidden': list(ACTOR_HIDDEN),
        'critic_hidden': list(CRITIC_HIDDEN),
        'obs_dim': train_env.observation_space.shape[0],
        'action_dim': train_env.action_space.shape[0],
        'N_agents': train_env.unwrapped.N,
        'checkpoint_freq': checkpoint_freq,
        'eval_freq': eval_freq,
        'save_replay_buffer': True,
        'resumed_from': resume_path,
    }
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # ── Train ─────────────────────────────────────────────────────────────

    t0 = time.time()
    if verbose:
        print(f"Training SAC: {run_name}")
        print(f"  Scenario: {scenario}/{budget_pct}%")
        print(f"  Timesteps: {total_timesteps:,}")
        print(f"  Seed: {seed}")
        print(f"  Device: {model.device}")
        print(f"  Policy: CTDESACPolicy (shared actor + centralized critic)")
        print(f"  Output: {run_dir}")
        print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=100,
        progress_bar=True,
    )

    train_time = time.time() - t0

    # ── Save final model ──────────────────────────────────────────────────

    final_path = run_dir / f'{run_name}_final'
    model.save(str(final_path))
    # Also save the replay buffer with the final model for completeness.
    model.save_replay_buffer(str(final_path) + '_replay_buffer.pkl')

    summary = {
        'run_name': run_name,
        'train_time_seconds': train_time,
        'total_timesteps': total_timesteps,
        'final_model_path': str(final_path.with_suffix('.zip')),
    }
    with open(run_dir / 'train_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nTraining complete in {train_time:.0f}s")
        print(f"Final model: {final_path}.zip")

    return model
