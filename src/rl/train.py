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
#   target_entropy = -13 (= -0.1 × dim(A)).
#
#   v2.4.2 change (post-20k pilot diagnosis): reduced from -65 (= -0.5 × dim)
#   to -13 (= -0.1 × dim). The v2.4.1 pilot showed catastrophic entropy
#   collapse: ent_coef plummeted from 0.172 to 0.0037 in 14k steps, killing
#   exploration. Root cause: with the (correctly) reduced LAMBDA_BUDGET=0.1
#   in v2.4, agronomic reward magnitudes are on the order 0.01-1, so the
#   entropy-penalty term (ent_coef × entropy) dominates the SAC objective
#   when ent_coef is anywhere near typical values. The dual optimizer for
#   ent_coef then has a strong gradient to crush it toward zero, collapsing
#   policy stochasticity. A more conservative target_entropy (-0.1 × dim
#   instead of -0.5 × dim) keeps the equilibrium ent_coef higher and
#   preserves exploration. If this still collapses (kill the run if
#   ent_coef < 0.01 at step 20k), fall back to fixed ent_coef = 0.1.
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
#   - RotatingReplayBufferCheckpoint: keeps only latest buffer on disk so
#     a 2M-step run fits Kaggle's 20 GB /kaggle/working quota.
#
# v2.4.2 changes (entropy + observability):
#   - target_entropy = -13 (see entropy-collapse diagnosis above).
#   - Optional WandB integration via wandb_project parameter. If wandb is
#     installed AND WANDB_API_KEY is set (as env var, Kaggle Secret, or
#     Colab userdata), training metrics and model checkpoints stream to
#     the cloud in real time. If wandb is missing or unconfigured, training
#     proceeds with local-only logging and prints a warning. The smoke test
#     does not require wandb.
#
# Kaggle GPU T4: measured 68 steps/sec; 2M steps ≈ 8 hrs/seed.
# =============================================================================

import os
import re
import json
import time
import warnings
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
    'target_entropy':   -13,       # v2.4.2: was -65; see header docstring
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


# ── WandB integration (optional) ──────────────────────────────────────────────
#
# WandB is loaded lazily and only if requested. If wandb is not installed
# or the API key is not available, training falls back to local-only
# logging and prints a warning. This keeps the smoke test and offline runs
# working without a wandb account.
#
# API key resolution order:
#   1. WANDB_API_KEY environment variable (set by user or by host setup)
#   2. Kaggle Secret 'WANDB_API_KEY' (if running on Kaggle)
#   3. Colab userdata 'WANDB_API_KEY' (if running on Colab)
#   4. None found → return without initializing wandb

def _resolve_wandb_api_key():
    """Try to find WANDB_API_KEY in env, Kaggle secrets, or Colab userdata."""
    if os.environ.get('WANDB_API_KEY'):
        return 'env', os.environ['WANDB_API_KEY']

    # Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        try:
            key = UserSecretsClient().get_secret('WANDB_API_KEY')
            if key:
                return 'kaggle_secrets', key
        except Exception:
            pass
    except ImportError:
        pass

    # Colab userdata
    try:
        from google.colab import userdata
        try:
            key = userdata.get('WANDB_API_KEY')
            if key:
                return 'colab_userdata', key
        except Exception:
            pass
    except ImportError:
        pass

    return None, None


def _init_wandb(wandb_project, wandb_entity, run_name, config, total_timesteps,
                log_dir, verbose=1):
    """Initialize wandb run if possible. Returns (wandb_run, wandb_callback)
    or (None, None) if wandb is unavailable/unconfigured."""

    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
    except ImportError:
        if verbose:
            print("⚠  wandb not installed (pip install wandb). "
                  "Training will run with local-only logging.")
        return None, None

    source, key = _resolve_wandb_api_key()
    if key is None:
        if verbose:
            print("⚠  WANDB_API_KEY not found in env, Kaggle Secrets, or "
                  "Colab userdata. Training will run with local-only logging.")
            print("   To enable cloud sync: set WANDB_API_KEY in your "
                  "platform's secret manager (Kaggle Add-ons > Secrets, "
                  "or Colab sidebar key icon).")
        return None, None

    # wandb.login() reads from the env var; set it from whichever source we found.
    os.environ['WANDB_API_KEY'] = key
    if verbose:
        print(f"✓  WandB API key loaded from {source}.")

    try:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
            dir=str(log_dir),
            reinit=True,
        )
    except Exception as e:
        warnings.warn(f"wandb.init() failed ({type(e).__name__}: {e}). "
                      "Continuing with local-only logging.")
        return None, None

    callback = WandbCallback(
        gradient_save_freq=0,          # disable; expensive and not useful here
        model_save_path=str(log_dir / 'wandb_models'),
        model_save_freq=50_000,        # match the local checkpoint cadence
        verbose=2 if verbose else 0,
    )

    if verbose:
        print(f"✓  WandB run initialized: {run.url}")

    return run, callback


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
    wandb_project=None,
    wandb_entity=None,
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
        doesn't exist (e.g. resuming from an old run), tries the legacy
        SB3 naming.
    hp_overrides : dict or None
    wandb_project : str or None
        WandB project name. If None, WandB is disabled. If set (e.g.
        'sac-irrigation-thesis'), and WANDB_API_KEY can be resolved from
        env / Kaggle Secrets / Colab userdata, metrics and model
        checkpoints stream to the WandB dashboard in real time.
    wandb_entity : str or None
        WandB entity (username or team). Default None = personal account.
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

    # ── Build config (used by both wandb and local config.json) ──────────
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
        'target_entropy_note': 'v2.4.2: -13 (= -0.1 x dim); was -65 in v2.4',
    }

    # ── WandB init (optional) ─────────────────────────────────────────────
    wandb_run = None
    wandb_callback = None
    if wandb_project is not None:
        wandb_run, wandb_callback = _init_wandb(
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            run_name=run_name,
            config=config,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            verbose=verbose,
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

    callback_list = [checkpoint_callback, eval_callback]
    if wandb_callback is not None:
        callback_list.append(wandb_callback)
    callbacks = CallbackList(callback_list)

    # ── Save config ───────────────────────────────────────────────────────
    config['wandb_active'] = wandb_run is not None
    if wandb_run is not None:
        config['wandb_url'] = wandb_run.url
        config['wandb_project'] = wandb_project
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
        print(f"  target_entropy:   {hp['target_entropy']}")
        print(f"  WandB:            {'active (' + wandb_run.url + ')' if wandb_run else 'disabled'}")
        print(f"  Output:           {run_dir}")
        print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True,
        )
    finally:
        # Ensure wandb run is properly closed even if training errors out.
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as e:
                warnings.warn(f"wandb.finish() failed: {e}")

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
        'wandb_url': wandb_run.url if wandb_run else None,
    }
    with open(run_dir / 'train_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nTraining complete in {train_time:.0f}s")
        print(f"Final model:  {final_path}.zip")
        print(f"Final buffer: {final_rb_path}")
        if wandb_run:
            print(f"WandB run:    {wandb_run.url}")

    return model
