# =============================================================================
# src/rl/train.py
# SAC training loop using Stable-Baselines3 with the CTDE policy.
#
# Design (Chapter 4):
#   - One policy trained across all (year, budget) combinations.
#     At each episode reset the env samples a year from TRAINING_YEARS
#     (23 years, 2000-2025 minus eval years) and a budget from U(70%,100%).
#   - The 9 eval cells (2018/2022/2024 x 70/85/100%) are strictly held out
#     and never seen during training. Final evaluation runs post-training
#     via scripts/experiments/exp_rl.py --mode eval.
#   - 5 seeds for statistical robustness (conference-paper standard).
#
# Hyperparameters:
#   target_entropy = -65 (= -0.5 x dim(A), standard heuristic).
#   Pilot sweep over {-130, -65, -32}: -130 clearly worse; -65 vs -32
#   within noise at 100k steps (2 seeds each). -65 chosen as the
#   defensible default per Haarnoja et al. 2019.
#
#   buffer_size = 200k: at obs_dim=707, 200k transitions ~= 1.2 GB RAM.
#   gradient_steps = 1: measured on Kaggle T4 that gs=2 halves throughput
#   (37 vs 68 steps/sec) without guaranteed convergence benefit for plain
#   SAC (REDQ-style ensembles needed to safely push UTD > 1).
#
# Kaggle deployment (GPU T4):
#   Measured 68 steps/sec; 500k steps ~= 2 hrs/seed.
#   5 seeds in 5 parallel notebooks ~= 2 hrs wall-clock, ~10 GPU-hrs total.
#   Checkpoint every 50k steps (10 total per seed).
#   save_replay_buffer=True is REQUIRED for session resumption -- SAC is
#   off-policy; resuming with empty buffer causes catastrophic forgetting.
# =============================================================================

import re
import json
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from src.rl.gym_env import IrrigationEnv
from src.rl.networks import CTDESACPolicy, make_sac_policy_kwargs


# ── Hyperparameters ──────────────────────────────────────────────────────────

DEFAULT_HP = {
    'learning_rate':    3e-4,
    'batch_size':       256,
    'buffer_size':      200_000,    # 707-dim obs x 200k ~= 1.2 GB
    'gamma':            0.99,
    'tau':              0.005,
    'ent_coef':         'auto',
    'learning_starts':  1000,
    'train_freq':       1,
    'gradient_steps':   1,
    'target_entropy':   -65,        # -0.5 x dim(A); pilot confirmed
}

ACTOR_HIDDEN  = (128, 128)
CRITIC_HIDDEN = (256, 256)


def train_sac(
    total_timesteps=500_000,
    seed=0,
    output_dir='results/rl',
    dem_path='gilan_farm.tif',
    checkpoint_freq=50_000,
    eval_freq=25_000,
    resume_path=None,
    hp_overrides=None,
    verbose=1,
):
    """Train a single SAC policy across all (year, budget) combinations.

    Parameters
    ----------
    total_timesteps : int
        Total env steps. 500k ~= 2 hrs on Kaggle T4.
    seed : int
    output_dir : str
    dem_path : str
    checkpoint_freq : int
        Save model + replay buffer every N steps. Default 50k (10 total).
    eval_freq : int
        EvalCallback frequency. Default 25k.
    resume_path : str or None
        Path to a checkpoint .zip to resume from.
    hp_overrides : dict or None
    verbose : int

    Returns
    -------
    model : SAC
    """
    output_dir = Path(output_dir)
    run_name = f"sac_general_seed{seed}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = run_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    hp = {**DEFAULT_HP, **(hp_overrides or {})}

    # ── Environments ──────────────────────────────────────────────────────
    # Both train and eval envs use randomize=True (sampling from the
    # training distribution). The 9 held-out eval cells are NEVER used
    # here -- they are reserved for post-training final evaluation only.
    train_env = Monitor(
        IrrigationEnv(randomize=True, dem_path=dem_path, seed=seed),
        filename=str(log_dir / 'train'),
    )
    eval_env = Monitor(
        IrrigationEnv(randomize=True, dem_path=dem_path, seed=seed + 1000),
        filename=str(log_dir / 'eval'),
    )

    # ── Policy ────────────────────────────────────────────────────────────
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

        # Locate replay buffer. SB3 CheckpointCallback names it:
        #   {name_prefix}_replay_buffer_{n_steps}_steps.pkl
        # Parse name_prefix and n_steps from the model checkpoint filename.
        _m = re.match(r'^(.+)_(\d+)_steps$', Path(resume_path).stem)
        if _m is not None:
            rb_full = (Path(resume_path).parent /
                       f'{_m.group(1)}_replay_buffer_{_m.group(2)}_steps.pkl')
        else:
            # Fallback: final model (no _N_steps suffix in name)
            rb_full = Path(str(resume_path).replace('.zip', '_replay_buffer.pkl'))

        if rb_full.exists():
            if verbose:
                print(f"Loading replay buffer from {rb_full}")
            model.load_replay_buffer(str(rb_full))
        else:
            print(
                f"WARNING: No replay buffer at {rb_full}. "
                f"Resuming with empty buffer causes catastrophic forgetting. "
                f"Consider restarting from scratch."
            )
    else:
        model = SAC(
            policy=CTDESACPolicy,
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
            target_entropy=hp['target_entropy'],
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
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / 'best_model'),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=verbose,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # ── Save config ───────────────────────────────────────────────────────
    config = {
        'run_name': run_name,
        'seed': seed,
        'total_timesteps': total_timesteps,
        'policy_class': 'CTDESACPolicy',
        'hyperparameters': {
            k: str(v) if not isinstance(v, (int, float)) else v
            for k, v in hp.items()
        },
        'actor_hidden': list(ACTOR_HIDDEN),
        'critic_hidden': list(CRITIC_HIDDEN),
        'obs_dim': train_env.observation_space.shape[0],
        'action_dim': train_env.action_space.shape[0],
        'N_agents': train_env.unwrapped.N,
        'training': 'TRAINING_YEARS (23 years) x U(70-100%) budget, randomized per episode',
        'eval_years_held_out': [2018, 2022, 2024],
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
        print(f"  Timesteps:  {total_timesteps:,}")
        print(f"  Seed:       {seed}")
        print(f"  Device:     {model.device}")
        print(f"  Policy:     CTDESACPolicy (shared actor + centralized critic)")
        print(f"  Training:   23 years x U(70-100%) budget, randomized per episode")
        print(f"  Eval env:   randomized training-year env (held-out cells post-training only)")
        print(f"  Output:     {run_dir}")
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
