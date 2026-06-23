# src/rl/train_td3.py
# -----------------------------------------------------------------------------
# TD3 trainer for the irrigation controller (the chosen "v2.21c" configuration).
#
# Architecture: Td3VdnPolicy — deterministic parameter-shared actor + the same
# twin VDN LayerNorm critic as SAC. Three ingredients give the stable line that
# reaches ~99.7% of MPC yield while beating it on wet-year waterlogging:
#
#   1. EXACT n-step returns (n=5) via the model-gamma trick. The buffer
#      accumulates R_n with gamma_base=0.99; the model's gamma is set to
#      gamma_base**n, so SB3's stock TD3 target computes
#          R_n + (1-done) * gamma_base^n * Q(s_{t+n})
#      exactly. This bounds the bootstrap horizon (the source of the earlier
#      q_pred divergence) while the critic still learns the gamma_base return.
#   2. Control-rate smoothing reward r5 (alpha5=0.005), mirroring MPC's term 5.
#   3. An additive terminal-yield bonus (alpha_T=1.0 * x4_final/X4_REF) paid once
#      at episode end, lifting the endpoint weight toward MPC's terminal-biomass
#      cost without removing the dense per-step increment.
#
# Best-model selection uses the held-out DEV_YEARS via a deterministic schedule,
# identical to train_sac.py. New runs use the clean observation; legacy
# checkpoints still load via runner auto-detection.
# Runs nothing on import; launch from the CLI (see __main__).
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from climate_data import DEV_YEARS, TRAINING_YEARS
from src.rl.gym_env import IrrigationEnv
from src.rl.networks_td3 import Td3VdnPolicy, make_td3_policy_kwargs
from src.rl.nstep_buffer import NStepReplayBuffer
from src.rl.common import (
    make_lr_schedule, init_wandb, tensorboard_dir, RotatingReplayBufferCheckpoint,
    GradClipCallback, WarmupAsymmetricLRTD3,
)
from src.rl.callbacks_train import (
    BiasRatioCallback, ActionStatsCallback, OptimizerLRCallback,
)
from src.rl.callbacks_exploration import (
    ExplorationNoiseDecayCallback, LowActionCoverageCallback,
    CollapseGuardCallback, NonFiniteGuardCallback,
)
from src.rl.callbacks_eval import FixedScheduleEvalCallback

# ── hyperparameters ───────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 250_000
BUFFER_SIZE     = 250_000
BATCH_SIZE      = 256

GAMMA_BASE      = 0.99
TAU             = 0.005
LR_START        = 3e-4
LR_END          = 5e-5

MAX_GRAD_NORM   = 1.0
LEARNING_STARTS = 50_000
GRADIENT_STEPS  = 1
TRAIN_FREQ      = 1

# n-step + TD3 damping.
N_STEPS             = 5
POLICY_DELAY        = 2
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP   = 0.5
ACTOR_LR_MULT       = 1.0
ACTOR_WARMUP_UPDATES = 0

# Reward shaping (MPC-aligned).
REWARD_DU_ALPHA       = 0.005   # control-rate smoothing r5 (MPC term 5)
REWARD_TERMINAL_YIELD = 1.0     # additive terminal-yield bonus

# Exploration noise (decay to a sustained floor).
EXPLORE_SIGMA_START = 0.40
EXPLORE_SIGMA_END   = 0.15
EXPLORE_DECAY_STEPS = 150_000
EXPLORE_LOG_FREQ    = 1_000
COVERAGE_LOG_FREQ   = 1_000

# Collapse guard (aborts a run that has clearly starved exploration).
GUARD_COLLAPSE_FRAC = 0.60
GUARD_WARMUP_STEPS  = 30_000
GUARD_CHECK_FREQ    = 2_000
GUARD_WINDOW        = 8
GUARD_ABORT         = True

EVAL_FREQ        = 25_000
CHECKPOINT_FREQ  = 25_000
ACTOR_HIDDEN     = [128, 128]
CRITIC_HIDDEN    = [256, 256]
BIAS_RATIO_FREQ  = 25_000
ACTION_STATS_FREQ = 1_000
LR_LOG_FREQ       = 1_000

N_AGENTS = 130

EVAL_BUDGET_FRACS  = (0.70, 0.85, 1.00)
EVAL_SCHEDULE      = [(yr, bf) for yr in DEV_YEARS for bf in EVAL_BUDGET_FRACS]
N_EVAL_EPISODES    = len(EVAL_SCHEDULE)
BIAS_EVAL_SCHEDULE = [(yr, 1.00) for yr in DEV_YEARS]
BIAS_RATIO_N_EPISODES = len(BIAS_EVAL_SCHEDULE)


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(Path(__file__).resolve().parent),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def train_td3(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
) -> WarmupAsymmetricLRTD3:
    """Train the TD3 irrigation controller (exact n-step + terminal-yield bonus)."""
    model_gamma = GAMMA_BASE ** N_STEPS   # exact n-step bootstrap discount

    run_name = f"td3_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "algorithm": "WarmupAsymmetricLRTD3 (SB3 TD3) + exact n-step VDN",
        "policy_class": "Td3VdnPolicy (deterministic shared actor + VDN LayerNorm critic)",
        "git_sha": _git_sha(),
        "seed": seed,
        "total_timesteps": total_timesteps,
        "n_steps": N_STEPS, "gamma_base": GAMMA_BASE, "model_gamma": model_gamma,
        "policy_delay": POLICY_DELAY, "target_policy_noise": TARGET_POLICY_NOISE,
        "target_noise_clip": TARGET_NOISE_CLIP,
        "learning_starts": LEARNING_STARTS,
        "reward_du_alpha": REWARD_DU_ALPHA, "reward_terminal_yield": REWARD_TERMINAL_YIELD,
        "tau": TAU, "buffer_size": BUFFER_SIZE, "batch_size": BATCH_SIZE,
        "lr_start": LR_START, "lr_end": LR_END,
        "explore": {"start": EXPLORE_SIGMA_START, "end": EXPLORE_SIGMA_END,
                    "decay_steps": EXPLORE_DECAY_STEPS},
        "dev_years": list(DEV_YEARS), "training_years": list(TRAINING_YEARS),
        "eval_schedule": EVAL_SCHEDULE,
        "eval_method": "deterministic dev-set (FixedScheduleEvalCallback)",
    }
    (save_dir / "manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    wandb_active = init_wandb(wandb_project, run_name, config) if wandb_project else False

    def _make_env(randomize, schedule=None):
        return IrrigationEnv(
            randomize=randomize, eval_schedule=schedule,
            reward_du_alpha=REWARD_DU_ALPHA, reward_terminal_yield=REWARD_TERMINAL_YIELD)

    train_env     = DummyVecEnv([lambda: _make_env(True)])
    eval_env      = DummyVecEnv([lambda: _make_env(False, EVAL_SCHEDULE)])
    bias_eval_env = DummyVecEnv([lambda: _make_env(False, BIAS_EVAL_SCHEDULE)])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)
    bias_eval_env.seed(seed + 2000)

    policy_kwargs = make_td3_policy_kwargs(
        N=N_AGENTS, actor_hidden=ACTOR_HIDDEN, critic_hidden=CRITIC_HIDDEN)
    lr_schedule = make_lr_schedule(LR_START, LR_END)
    action_noise = NormalActionNoise(
        mean=np.zeros(N_AGENTS, dtype=np.float64),
        sigma=EXPLORE_SIGMA_START * np.ones(N_AGENTS, dtype=np.float64))

    WarmupAsymmetricLRTD3.actor_lr_mult = ACTOR_LR_MULT
    WarmupAsymmetricLRTD3.actor_warmup_updates = ACTOR_WARMUP_UPDATES
    model = WarmupAsymmetricLRTD3(
        policy=Td3VdnPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=model_gamma,
        tau=TAU,
        action_noise=action_noise,
        policy_delay=POLICY_DELAY,
        target_policy_noise=TARGET_POLICY_NOISE,
        target_noise_clip=TARGET_NOISE_CLIP,
        learning_starts=LEARNING_STARTS,
        gradient_steps=GRADIENT_STEPS,
        train_freq=TRAIN_FREQ,
        replay_buffer_class=NStepReplayBuffer,
        replay_buffer_kwargs=dict(n_steps=N_STEPS, gamma=GAMMA_BASE),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_dir(save_dir / "tensorboard"),
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
        ExplorationNoiseDecayCallback(
            sigma_start=EXPLORE_SIGMA_START, sigma_end=EXPLORE_SIGMA_END,
            decay_steps=EXPLORE_DECAY_STEPS, log_freq=EXPLORE_LOG_FREQ,
            csv_path=str(save_dir / "exploration_sigma_log.csv"), verbose=1),
        LowActionCoverageCallback(
            log_freq=COVERAGE_LOG_FREQ,
            csv_path=str(save_dir / "low_action_coverage_log.csv"), verbose=0),
        CollapseGuardCallback(
            collapse_frac=GUARD_COLLAPSE_FRAC, warmup_steps=GUARD_WARMUP_STEPS,
            check_freq=GUARD_CHECK_FREQ, window=GUARD_WINDOW, abort_on_collapse=GUARD_ABORT,
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

    print(f"\n{'='*72}\n  TD3 training - seed {seed}\n"
          f"  n-step n={N_STEPS}  gamma_base={GAMMA_BASE}  model_gamma={model_gamma:.6f}\n"
          f"  policy_delay={POLICY_DELAY}  target_noise={TARGET_POLICY_NOISE}/{TARGET_NOISE_CLIP}\n"
          f"  learning_starts={LEARNING_STARTS:,}  r5(du)={REWARD_DU_ALPHA}  "
          f"terminal_yield={REWARD_TERMINAL_YIELD}\n"
          f"  noise {EXPLORE_SIGMA_START}->{EXPLORE_SIGMA_END} over {EXPLORE_DECAY_STEPS:,}\n"
          f"  dev years {list(DEV_YEARS)}  git={config['git_sha']}  total steps {total_timesteps:,}\n"
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
    try:
        config["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (save_dir / "manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(f"\n[train] Final model saved to {final_path}.zip")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the TD3 irrigation controller.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/rl")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()
    train_td3(seed=args.seed, output_dir=args.output_dir,
              wandb_project=args.wandb_project, total_timesteps=args.total_timesteps)
