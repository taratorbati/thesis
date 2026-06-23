# src/rl/train_sac.py
# -----------------------------------------------------------------------------
# SAC trainer for the irrigation controller (the chosen "v2.18-P3b" configuration).
#
# Architecture: SacVdnPolicy — parameter-shared LeakyReLU actor + twin VDN
# LayerNorm critic, asymmetric actor LR. Training tricks that produced the
# standing SAC baseline (~99% of MPC yield):
#   * ent_coef = 0.002  — a weak entropy pin, so the policy can push the action
#     toward 0 mm in wet states instead of hovering at the range midpoint.
#   * Two-phase exploration noise: anneal 0.30 -> 0 over the first 60k steps,
#     then re-inject a triangular pulse (peak 0.15) over [150k, 180k] once the
#     critic is well trained, repopulating low-water transitions in wet states.
#
# Best-model selection uses the held-out DEV_YEARS via a deterministic schedule
# (FixedScheduleEvalCallback) — the same protocol as train_td3.py, so the two
# controllers are directly comparable.
#
# New runs use the clean observation (today's weather fed once); existing
# checkpoints trained on the legacy layout still load via runner auto-detection.
# Runs nothing on import; launch from the CLI (see __main__).
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from climate_data import DEV_YEARS, TRAINING_YEARS
from src.rl.gym_env import IrrigationEnv
from src.rl.networks import SacVdnPolicy, make_sac_policy_kwargs
from src.rl.common import (
    make_lr_schedule, init_wandb, RotatingReplayBufferCheckpoint,
    GradClipCallback, AsymmetricLRSAC,
)
from src.rl.callbacks_train import (
    BiasRatioCallback, ActionStatsCallback, OptimizerLRCallback,
)
from src.rl.callbacks_exploration import (
    LateNoiseReinjectionCallback, LowActionCoverageCallback,
    CollapseGuardCallback, NonFiniteGuardCallback,
)
from src.rl.callbacks_eval import FixedScheduleEvalCallback

# ── hyperparameters ───────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 250_000
BUFFER_SIZE     = 250_000
BATCH_SIZE      = 256

GAMMA           = 0.99
TAU             = 0.005
LR_START        = 3e-4
LR_END          = 5e-5
ACTOR_LR_MULT   = 5.0
ENT_COEF        = 0.002

MAX_GRAD_NORM   = 1.0
LEARNING_STARTS = 1_000
GRADIENT_STEPS  = 1
TRAIN_FREQ      = 1

EVAL_FREQ       = 25_000
CHECKPOINT_FREQ = 25_000

ACTOR_HIDDEN  = [128, 128]
CRITIC_HIDDEN = [256, 256]

BIAS_RATIO_FREQ       = 25_000
BIAS_RATIO_N_EPISODES = 3
ACTION_STATS_FREQ     = 1_000
LR_LOG_FREQ           = 1_000

# Two-phase exploration noise.
EXPLORE_SIGMA_START    = 0.30
EXPLORE_SIGMA_FLOOR    = 0.0
EXPLORE_SIGMA_REINJECT = 0.15
EXPLORE_DECAY_STEPS    = 60_000
REINJECT_START         = 150_000
REINJECT_END           = 180_000
EXPLORE_LOG_FREQ       = 1_000
COVERAGE_LOG_FREQ      = 1_000
WET_RAIN_THRESHOLD_MM  = 120.0

# Collapse guard — diagnostic only (does not abort), matching the TD3 runs.
GUARD_CHECK_FREQ    = 2_000
GUARD_WINDOW        = 10
GUARD_COLLAPSE_FRAC = 0.60
GUARD_WARMUP_STEPS  = 10_000

N_AGENTS = 130

# Deterministic held-out eval schedule (3 DEV_YEARS x 3 budgets = 9 episodes).
EVAL_BUDGET_FRACS  = [0.70, 0.85, 1.00]
EVAL_SCHEDULE      = [(yr, bf) for yr in DEV_YEARS for bf in EVAL_BUDGET_FRACS]
N_EVAL_EPISODES    = len(EVAL_SCHEDULE)
BIAS_EVAL_SCHEDULE = [(yr, 1.00) for yr in DEV_YEARS]


def train_sac(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    ent_coef: float = ENT_COEF,
    actor_lr_mult: float = ACTOR_LR_MULT,
) -> AsymmetricLRSAC:
    """Train the SAC irrigation controller with deterministic dev-set evaluation."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_seed{seed}_{ts}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "algorithm": "SAC + asymmetric actor LR + two-phase action noise",
        "policy_class": "SacVdnPolicy (shared LeakyReLU actor + VDN LayerNorm critic)",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "ent_coef": ent_coef,
        "gamma": GAMMA, "tau": TAU, "actor_lr_mult": actor_lr_mult,
        "learning_starts": LEARNING_STARTS,
        "explore": {"start": EXPLORE_SIGMA_START, "floor": EXPLORE_SIGMA_FLOOR,
                    "reinject": EXPLORE_SIGMA_REINJECT, "decay_steps": EXPLORE_DECAY_STEPS,
                    "reinject_window": [REINJECT_START, REINJECT_END]},
        "dev_years": list(DEV_YEARS),
        "training_years": list(TRAINING_YEARS),
        "eval_schedule": EVAL_SCHEDULE,
        "eval_method": "deterministic dev-set (FixedScheduleEvalCallback)",
    }
    (save_dir / "manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    wandb_active = init_wandb(wandb_project, run_name, config) if wandb_project else False

    def _make_train_env():
        return IrrigationEnv(randomize=True)

    def _make_eval_env(schedule):
        return IrrigationEnv(randomize=False, eval_schedule=schedule)

    train_env     = DummyVecEnv([_make_train_env])
    eval_env      = DummyVecEnv([lambda: _make_eval_env(EVAL_SCHEDULE)])
    bias_eval_env = DummyVecEnv([lambda: _make_eval_env(BIAS_EVAL_SCHEDULE)])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)
    bias_eval_env.seed(seed + 2000)

    policy_kwargs = make_sac_policy_kwargs(
        N=N_AGENTS, actor_hidden=ACTOR_HIDDEN, critic_hidden=CRITIC_HIDDEN)
    lr_schedule = make_lr_schedule(LR_START, LR_END)
    action_noise = NormalActionNoise(
        mean=np.zeros(N_AGENTS, dtype=np.float64),
        sigma=EXPLORE_SIGMA_START * np.ones(N_AGENTS, dtype=np.float64))

    AsymmetricLRSAC.actor_lr_mult = float(actor_lr_mult)
    model = AsymmetricLRSAC(
        policy=SacVdnPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        ent_coef=ent_coef,
        action_noise=action_noise,
        learning_starts=LEARNING_STARTS,
        gradient_steps=GRADIENT_STEPS,
        train_freq=TRAIN_FREQ,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    callbacks = [
        FixedScheduleEvalCallback(
            eval_env, best_model_save_path=str(save_dir / "best_model"),
            log_path=str(save_dir / "eval_logs"), eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES, deterministic=True, render=False),
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ, save_path=str(save_dir / "checkpoints"),
            name_prefix=run_name, save_replay_buffer=False, verbose=1),
        RotatingReplayBufferCheckpoint(save_freq=CHECKPOINT_FREQ, save_path=save_dir, verbose=1),
        GradClipCallback(max_grad_norm=MAX_GRAD_NORM),
        BiasRatioCallback(eval_env=bias_eval_env, eval_freq=BIAS_RATIO_FREQ,
                          n_eval_episodes=BIAS_RATIO_N_EPISODES, save_path=str(save_dir), verbose=1),
        ActionStatsCallback(log_freq=ACTION_STATS_FREQ),
        OptimizerLRCallback(log_freq=LR_LOG_FREQ),
        LateNoiseReinjectionCallback(
            sigma_start=EXPLORE_SIGMA_START, sigma_floor=EXPLORE_SIGMA_FLOOR,
            sigma_reinject=EXPLORE_SIGMA_REINJECT, decay_steps=EXPLORE_DECAY_STEPS,
            reinject_start=REINJECT_START, reinject_end=REINJECT_END,
            log_freq=EXPLORE_LOG_FREQ, csv_path=str(save_dir / "exploration_sigma_log.csv"), verbose=1),
        LowActionCoverageCallback(
            low_thresh=1.0 / 12.0, log_freq=COVERAGE_LOG_FREQ,
            csv_path=str(save_dir / "low_action_coverage_log.csv"),
            wet_rain_threshold_mm=WET_RAIN_THRESHOLD_MM, verbose=0),
        CollapseGuardCallback(
            collapse_frac=GUARD_COLLAPSE_FRAC, warmup_steps=GUARD_WARMUP_STEPS,
            check_freq=GUARD_CHECK_FREQ, window=GUARD_WINDOW, abort_on_collapse=False,
            csv_path=str(save_dir / "collapse_guard_log.csv"), verbose=1),
        NonFiniteGuardCallback(
            stop_on_nonfinite=True, csv_path=str(save_dir / "nonfinite_guard_log.csv"), verbose=1),
    ]
    if wandb_active:
        try:
            from wandb.integration.sb3 import WandbCallback
            callbacks.append(WandbCallback(
                model_save_path=str(save_dir / "wandb_models"),
                model_save_freq=CHECKPOINT_FREQ, verbose=0))
        except Exception as e:
            print(f"[WandB] WandbCallback unavailable ({e}); continuing without it.")

    print(f"\n{'='*72}\n  SAC training - seed {seed}\n"
          f"  ent_coef={ent_coef}  actor_lr_mult={actor_lr_mult}  gamma={GAMMA}\n"
          f"  noise: decay {EXPLORE_SIGMA_START}->{EXPLORE_SIGMA_FLOOR} over {EXPLORE_DECAY_STEPS:,}; "
          f"pulse {EXPLORE_SIGMA_REINJECT} over [{REINJECT_START:,},{REINJECT_END:,}]\n"
          f"  dev years {list(DEV_YEARS)}  | total steps {total_timesteps:,}\n"
          f"  output: {save_dir}\n{'='*72}\n")

    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks),
                    reset_num_timesteps=True, progress_bar=True)
    finally:
        if wandb_active:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    final_path = save_dir / f"{run_name}_final"
    model.save(str(final_path))
    print(f"\n[train] Final model saved to {final_path}.zip")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the SAC irrigation controller.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/rl")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--ent-coef", type=float, default=ENT_COEF)
    parser.add_argument("--actor-lr-mult", type=float, default=ACTOR_LR_MULT)
    args = parser.parse_args()
    train_sac(seed=args.seed, output_dir=args.output_dir, wandb_project=args.wandb_project,
              total_timesteps=args.total_timesteps, ent_coef=args.ent_coef,
              actor_lr_mult=args.actor_lr_mult)
