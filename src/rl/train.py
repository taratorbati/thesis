# =============================================================================
# src/rl/train.py
# SAC training loop using Stable-Baselines3 with the CTDE policy.
#
# Year split (v2.4):
#   TRAIN — 20 years (TRAINING_YEARS): sampled per episode from gym_env reset()
#   DEV   — 3 years  (DEV_YEARS = 2002, 2016, 2023): used by EvalCallback
#           during training for best_model selection and learning-curve
#           monitoring. Stratified by rainfall tercile so dev reward is a
#           meaningful generalization signal across the training distribution.
#   TEST  — 3 named scenarios (dry=2022, moderate=2018, wet=2024): touched
#           only post-training for the headline thesis comparison.
#
# Hyperparameters:
#   target_entropy = -65 (= -0.5 × dim(A), standard heuristic).
#   Pilot sweep over {-130, -65, -32}, 2 seeds each at 100k steps:
#   -130 clearly worse; -65 vs -32 within noise. -65 chosen per Haarnoja
#   et al. 2019 and cited as standard heuristic in the thesis.
#
#   buffer_size = 500k (v2.4.1, was 200k). At 2M total steps the buffer
#   cycles ~4× which is reasonable churn for off-policy SAC. Memory cost:
#   ~3.1 GB RAM and ~3.1 GB disk (single saved buffer file).
#
#   gradient_steps = 1: measured on Kaggle T4 that gs=2 halves throughput
#   (37 vs 68 steps/sec) without guaranteed convergence benefit for plain
#   SAC without REDQ-style critic ensembles.
#
# v2.4 changes (post-500k pilot diagnosis):
#   - LAMBDA_BUDGET reduced 5.0 → 0.1 in gym_env.py.
#   - Per-year precomputed quantities in gym_env.py (no Markov leak).
#   - Eval env samples from DEV_YEARS, n_eval_episodes=9.
#
# v2.4.1 changes (disk-quota fix):
#   - Custom RotatingReplayBufferCheckpoint replaces SB3's CheckpointCallback.
#     Keeps only the LATEST replay buffer on disk (overwrites previous before
#     saving new). Reason: at buffer_size=500k, obs_dim=707, each saved buffer
#     is ~3.1 GB. SB3's CheckpointCallback saves all snapshots, which over
#     40 checkpoints (2M / 50k) would consume ~124 GB — Kaggle's
#     /kaggle/working hard limit is 20 GB. The v2.4 run crashed at
#     checkpoint #15 with OSError [Errno 28] No space left on device.
#     With rotation, total disk footprint stays at ~3.1 GB regardless of
#     run length, while still allowing resume-from-latest. Model weights
#     (~5 MB each) are still saved at every checkpoint for the learning-
#     curve story.
#
# Kaggle GPU T4: measured 68 steps/sec; 2M steps ≈ 8 hrs/seed.
# =============================================================================

import os
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
from climate_data import DEV_YEARS


DEFAULT_HP = {
    'learning_rate':    3e-4,
    'batch_size':       256,
    'buffer_size':      500_000,   # v2.4.1: raised from 200k for 2M-step runs
    'gamma':            0.99,
    'tau':              0.005,
    'ent_coef':         'auto',
    'learning_starts':  1000,
    'train_freq':       1,
    'gradient_steps':   1,
    'target_entropy':   -65,
}

ACTOR_HIDDEN  = (128, 128)
CRITIC_HIDDEN = (256, 256)


# ── Custom checkpoint callback ────────────────────────────────────────────────
#
# SB3's CheckpointCallback saves a replay buffer at every save_freq step
# and never deletes the old ones. With buffer_size=500k and obs_dim=707,
# each saved buffer is ~3.1 GB. Over a 2M-step run at save_freq=50k
# (40 checkpoints), this would consume ~124 GB — Kaggle's 20 GB quota
# rejects this with [Errno 28] No space left on device.
#
# This subclass overrides the replay-buffer save path to always use a
# single fixed filename, so each new save overwrites the previous. Model
# weights are still saved at every checkpoint (small, ~5 MB each).
#
# Resume semantics unchanged: the latest replay buffer is always available
# at <save_path>/replay_buffer_latest.pkl alongside the numbered model
# checkpoints.

class RotatingReplayBufferCheckpoint(CheckpointCallback):
    """CheckpointCallback that keeps only the latest replay buffer on disk.

    Models are still saved at every save_freq step (numbered, for the
    learning curve). The replay buffer is overwritten at a single fixed
    path so disk usage stays bounded regardless of run length.
    """

    LATEST_BUFFER_NAME = 'replay_buffer_latest.pkl'

    def _checkpoint_path(self, checkpoint_type='', extension=''):
        """Return numbered path for models, fixed path for replay_buffer_."""
        # SB3 calls this with checkpoint_type='replay_buffer_' for the buffer
        # and '' (empty) for the model. We only intercept the buffer call.
        if checkpoint_type == 'replay_buffer_':
            return os.path.join(self.save_path, self.LATEST_BUFFER_NAME)
        # Fall through to default behavior for models, vecnormalize, etc.
        return super()._checkpoint_path(checkpoint_type, extension=extension)


def train_sac(
    total_timesteps=2_000_000,
    seed=0,
    output_dir='results/rl',
    dem_path='gilan_farm.tif',
    checkpoint_freq=50_000,
    eval_freq=25_000,
    n_eval_episodes=9,
    resume_path=None,
    hp_overrides=None,
    verbose=1,
):
    """Train a general SAC policy across all (training-year, budget) combinations.

    Parameters
    ----------
    total_timesteps : int
        Total env steps. Default 2M (~8 hrs on Kaggle T4).
    seed : int
    output_dir : str
    dem_path : str
    checkpoint_freq : int
        Default 50k. Model weights saved at every interval; replay buffer
        rotated (one file overwritten in place — see
        RotatingReplayBufferCheckpoint above).
    eval_freq : int
        EvalCallback frequency. Default 25k.
    n_eval_episodes : int
        Default 9 = 3 × |DEV_YEARS|.
    resume_path : str or None
        Path to checkpoint .zip to resume from. Replay buffer will be
        loaded from <same_dir>/replay_buffer_latest.pkl. If that file
        doesn't exist (e.g. resuming from an old run), tries the
        legacy SB3 naming.
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
    train_env = Monitor(
        IrrigationEnv(dem_path=dem_path, seed=seed),
        filename=str(log_dir / 'train'),
    )

    eval_env = Monitor(
        IrrigationEnv(
            dem_path=dem_path,
            seed=seed + 1000,
            year_pool=DEV_YEARS,
        ),
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

        # Try the v2.4.1 rotating-buffer name first.
        resume_dir = Path(resume_path).parent
        rotating_rb = resume_dir / RotatingReplayBufferCheckpoint.LATEST_BUFFER_NAME
        if rotating_rb.exists():
            rb_full = rotating_rb
        else:
            # Fall back to legacy SB3 naming for compatibility with old runs.
            _m = re.match(r'^(.+)_(\d+)_steps$', Path(resume_path).stem)
            if _m is not None:
                rb_full = (resume_dir /
                           f'{_m.group(1)}_replay_buffer_{_m.group(2)}_steps.pkl')
            else:
                rb_full = Path(str(resume_path).replace('.zip', '_replay_buffer.pkl'))

        if rb_full.exists():
            if verbose:
                print(f"Loading replay buffer from {rb_full}")
            model.load_replay_buffer(str(rb_full))
        else:
            print(
                f"WARNING: No replay buffer found "
                f"(tried {rotating_rb} and legacy paths). "
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
    # Rotating checkpoint: numbered model files + single overwritten buffer.
    checkpoint_callback = RotatingReplayBufferCheckpoint(
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
        n_eval_episodes=n_eval_episodes,
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
        'training_pool_size': 20,
        'dev_years': list(DEV_YEARS),
        'eval_test_years_held_out': [2018, 2022, 2024],
        'n_eval_episodes': n_eval_episodes,
        'checkpoint_freq': checkpoint_freq,
        'eval_freq': eval_freq,
        'resumed_from': resume_path,
        'lambda_budget_note': 'v2.4: reduced 5.0 -> 0.1',
        'precomputed_note': 'v2.4: per-year on-the-fly (no Markov leak)',
        'buffer_checkpoint_note': 'v2.4.1: rotating buffer (single file, '
                                  'overwritten each checkpoint) to fit '
                                  'Kaggle 20 GB disk quota',
    }
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    if verbose:
        print(f"Training SAC: {run_name}")
        print(f"  Timesteps:        {total_timesteps:,}")
        print(f"  Seed:             {seed}")
        print(f"  Device:           {model.device}")
        print(f"  Policy:           CTDESACPolicy (shared actor + centralized critic)")
        print(f"  Training years:   20 (TRAINING_YEARS) x U(70-100%) budget")
        print(f"  Dev years (eval): {DEV_YEARS}")
        print(f"  n_eval_episodes:  {n_eval_episodes}")
        print(f"  Buffer size:      {hp['buffer_size']:,}")
        print(f"  Output:           {run_dir}")
        print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=100,
        progress_bar=True,
    )

    train_time = time.time() - t0

    # ── Save final model ──────────────────────────────────────────────────
    # The replay buffer is also saved here. To save disk, we overwrite the
    # same rotating file used by the checkpoint callback instead of writing
    # a separate <final_model>_replay_buffer.pkl.
    final_path = run_dir / f'{run_name}_final'
    model.save(str(final_path))
    final_rb_path = checkpoint_dir / RotatingReplayBufferCheckpoint.LATEST_BUFFER_NAME
    model.save_replay_buffer(str(final_rb_path))

    summary = {
        'run_name': run_name,
        'train_time_seconds': train_time,
        'total_timesteps': total_timesteps,
        'final_model_path': str(final_path.with_suffix('.zip')),
        'final_replay_buffer_path': str(final_rb_path),
    }
    with open(run_dir / 'train_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nTraining complete in {train_time:.0f}s")
        print(f"Final model:  {final_path}.zip")
        print(f"Final buffer: {final_rb_path}")

    return model
