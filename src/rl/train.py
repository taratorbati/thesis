# =============================================================================
# src/rl/train.py
# SAC training loop using Stable-Baselines3.
#
# Design decisions (from ARCHITECTURE.md §5):
#   - Algorithm: SAC (entropy-regularized, continuous action)
#   - 5 seeds for statistical robustness
#   - Kaggle GPU: 30 GPU-hours/week, 9-hour session limit
#   - Checkpointing every 10k steps (crash recovery within 9h sessions)
#   - Evaluation every 10k steps on the training scenario
#
# Hyperparameters (grounded in literature):
#   - Learning rate: 3e-4 (Haarnoja et al. 2018, SAC default)
#   - Batch size: 256 (SAC default, good for continuous control)
#   - Buffer size: 100_000 (fits in Kaggle 16GB RAM for 660-dim obs)
#   - Gamma: 0.99 (standard for episodic tasks with 93 steps)
#   - Tau: 0.005 (soft target update, SAC default)
#   - Entropy: auto (automatic temperature tuning, Haarnoja et al. 2018)
#   - Training starts: 1000 (collect random experience before learning)
#
# References:
#   - Haarnoja et al. (2018) "Soft Actor-Critic Algorithms and Applications"
#   - SB3 docs: https://stable-baselines3.readthedocs.io/
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
from src.rl.networks import make_sac_policy_kwargs


# ── Hyperparameters ──────────────────────────────────────────────────────────

DEFAULT_HP = {
    'learning_rate':    3e-4,       # Haarnoja et al. 2018
    'batch_size':       256,        # SAC default
    'buffer_size':      100_000,    # ~660 dims × 100k ≈ 0.5 GB
    'gamma':            0.99,       # standard for 93-step episodes
    'tau':              0.005,      # soft target update
    'ent_coef':         'auto',     # automatic entropy tuning
    'learning_starts':  1000,       # random exploration steps
    'train_freq':       1,          # update every step
    'gradient_steps':   1,          # one gradient step per env step
    'target_entropy':   'auto',     # -dim(action) = -130
}

# Architecture
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
    budget_pct : int
    total_timesteps : int
    seed : int
    output_dir : str
    dem_path : str
    checkpoint_freq : int
        Save model every N steps.
    eval_freq : int
        Evaluate on a separate env every N steps.
    resume_path : str or None
        Path to a .zip checkpoint to resume from.
    hp_overrides : dict or None
        Override default hyperparameters.
    verbose : int
        0 = silent, 1 = info, 2 = debug.

    Returns
    -------
    model : SAC
        Trained model.
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

    # Training env
    train_env = Monitor(
        IrrigationEnv(
            scenario=scenario,
            budget_pct=budget_pct,
            dem_path=dem_path,
            seed=seed,
        ),
        filename=str(log_dir / 'train'),
    )

    # Evaluation env (same scenario, different seed for stochasticity)
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
        )
        # Update remaining timesteps if needed
        model.set_env(train_env)
    else:
        model = SAC(
            policy='MlpPolicy',
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

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=run_name,
        save_replay_buffer=False,  # too large for Kaggle disk
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / 'best_model'),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=3,    # 3 episodes per eval (deterministic)
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
        'hyperparameters': {k: str(v) if not isinstance(v, (int, float)) else v
                           for k, v in hp.items()},
        'actor_hidden': list(ACTOR_HIDDEN),
        'critic_hidden': list(CRITIC_HIDDEN),
        'obs_dim': train_env.observation_space.shape[0],
        'action_dim': train_env.action_space.shape[0],
        'checkpoint_freq': checkpoint_freq,
        'eval_freq': eval_freq,
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

    # Save training summary
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
