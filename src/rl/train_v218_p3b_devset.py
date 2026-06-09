# src/rl/train_v218_p3b_devset.py  v2.18-P3b-devset
# -----------------------------------------------------------------------------
# IDENTICAL to train_v218_p3b.py EXCEPT:
#   1. Eval env uses DETERMINISTIC dev-set scheduling over DEV_YEARS
#      {2002,2004,2013} x {0.70, 0.85, 1.00} = 9 episodes (same protocol as
#      the TD3 v2.19b-v2.21c line), so best-model selection is comparable.
#   2. Bias-ratio eval also uses deterministic DEV_YEARS @ full budget.
#   3. Adds CollapseGuardCallback + NonFiniteGuardCallback (diagnostic only;
#      they do NOT change the training trajectory).
#   4. Writes manifest.json at run start for reproducibility tracking.
#
# The training env, architecture, reward, hyperparameters, exploration schedule,
# and late noise reinjection are BYTE-IDENTICAL to v2.18-P3b.
#
# WHY THIS EXISTS: v2.18-P3b used randomized eval (drawing random training
# years for best-model selection), while all TD3 runs used deterministic
# dev-set eval.  This re-run makes the comparison fair: both controllers
# are trained on the same 20-year pool (TRAINING_YEARS derived from the
# same DEV_YEARS={2002,2004,2013}) and best-model is selected on the same
# held-out dev schedule.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.gym_env import IrrigationEnv, RAIN_REF_V216
from src.rl.networks import V216CTDESACPolicy, make_sac_policy_kwargs
from src.rl.callbacks_v210 import (
    BiasRatioCallback,
    ActionStatsCallback,
    OptimizerLRCallback,
)
from src.rl.callbacks_exploration import (
    LateNoiseReinjectionCallback,
    LowActionCoverageCallback,
    CollapseGuardCallback,
    NonFiniteGuardCallback,
)
from src.rl.callbacks_eval import FixedScheduleEvalCallback
from src.rl.train import (
    RotatingReplayBufferCheckpoint,
    GradClipCallback,
    _make_lr_schedule,
    _init_wandb,
)
from src.rl.train_v212 import AsymmetricLRSAC

from climate_data import DEV_YEARS, TRAINING_YEARS


# -----------------------------------------------------------------------------
# Hyperparameters -- IDENTICAL to v2.18-P3b (no changes).
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256

GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5
ACTOR_LR_MULT    = 5.0
ENT_COEF         = 0.002

MAX_GRAD_NORM    = 1.0
LEARNING_STARTS  = 1_000
GRADIENT_STEPS   = 1
TRAIN_FREQ       = 1

EVAL_FREQ        = 25_000
CHECKPOINT_FREQ  = 25_000

ACTOR_HIDDEN  = [128, 128]
CRITIC_HIDDEN = [256, 256]

BIAS_RATIO_FREQ          = 25_000
BIAS_RATIO_N_EPISODES    = 3
ACTION_STATS_FREQ        = 1_000
LR_LOG_FREQ              = 1_000

EXPLORE_SIGMA_START    = 0.30
EXPLORE_SIGMA_FLOOR    = 0.0
EXPLORE_SIGMA_REINJECT = 0.15
EXPLORE_DECAY_STEPS    = 60_000
REINJECT_START         = 150_000
REINJECT_END           = 180_000
EXPLORE_LOG_FREQ       = 1_000
COVERAGE_LOG_FREQ      = 1_000
WET_RAIN_THRESHOLD_MM  = 120.0

# Collapse guard (diagnostic only — does NOT abort; consistent with TD3 runs)
GUARD_CHECK_FREQ      = 2_000
GUARD_WINDOW          = 10
GUARD_COLLAPSE_FRAC   = 0.60
GUARD_WARMUP_STEPS    = 10_000
GUARD_ABORT           = False       # telemetry only

N_AGENTS = 130

REWARD_OVERSHOOT_MODE = 'linear'
RAIN_NORMALISER       = RAIN_REF_V216   # 30.0

# *** NEW: deterministic eval schedule over DEV_YEARS ***
EVAL_BUDGET_FRACS  = [0.70, 0.85, 1.00]
EVAL_SCHEDULE      = [(yr, bf) for yr in DEV_YEARS for bf in EVAL_BUDGET_FRACS]
N_EVAL_EPISODES    = len(EVAL_SCHEDULE)  # 9
BIAS_EVAL_SCHEDULE = [(yr, 1.00) for yr in DEV_YEARS]


def train_sac_v218_p3b_devset(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    gamma: float = GAMMA,
    actor_lr_mult: float = ACTOR_LR_MULT,
    ent_coef: float = ENT_COEF,
    reward_overshoot_mode: str = REWARD_OVERSHOOT_MODE,
    rain_normaliser: float = RAIN_NORMALISER,
    explore_sigma_start: float = EXPLORE_SIGMA_START,
    explore_sigma_floor: float = EXPLORE_SIGMA_FLOOR,
    explore_sigma_reinject: float = EXPLORE_SIGMA_REINJECT,
    explore_decay_steps: int = EXPLORE_DECAY_STEPS,
    reinject_start: int = REINJECT_START,
    reinject_end: int = REINJECT_END,
) -> SAC:
    """Train SAC v2.18-P3b with deterministic dev-set evaluation.

    Identical to train_sac_v218_p3b() except the eval env walks the fixed
    DEV_YEARS x {0.70, 0.85, 1.00} schedule used by the TD3 line, so
    best-model selection is comparable across algorithms.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_v218_p3b_devset_seed{seed}_{ts}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.18.0-P3b-devset",
        "experiment": "v218_p3b_deterministic_devset_eval",
        "seed": seed,
        "algorithm": "SAC (stable_baselines3) + asymmetric actor LR + action noise",
        "policy_class": "V216CTDESACPolicy (marker=2.16)",
        "total_timesteps": total_timesteps,
        "ent_coef": ent_coef,
        "gamma": gamma,
        "tau": TAU,
        "actor_lr_mult": actor_lr_mult,
        "learning_starts": LEARNING_STARTS,
        "explore_sigma_start": explore_sigma_start,
        "explore_sigma_floor": explore_sigma_floor,
        "explore_sigma_reinject": explore_sigma_reinject,
        "explore_decay_steps": explore_decay_steps,
        "reinject_start": reinject_start,
        "reinject_end": reinject_end,
        "rain_normaliser": rain_normaliser,
        "reward_overshoot_mode": reward_overshoot_mode,
        "dev_years": list(DEV_YEARS),
        "training_years": list(TRAINING_YEARS),
        "eval_schedule": EVAL_SCHEDULE,
        "eval_method": "deterministic_devset (FixedScheduleEvalCallback)",
        "change_from_v218": (
            "ONLY the eval method changed: randomized training-year eval "
            "replaced with deterministic DEV_YEARS x budgets scheduling. "
            "Training env, architecture, reward, and all hyperparameters "
            "are byte-identical to v2.18-P3b."
        ),
    }

    # Write manifest BEFORE training
    manifest_path = save_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[manifest] Written to {manifest_path}")

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments
    # -------------------------------------------------------------------------
    def _make_env():
        """Training env: randomized years + budgets (IDENTICAL to v2.18-P3b)."""
        return IrrigationEnv(
            randomize=True,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,
            rain_normaliser=rain_normaliser,
        )

    def _make_eval_env():
        """*** CHANGED: deterministic dev-set eval (was randomized). ***"""
        return IrrigationEnv(
            randomize=False,
            eval_schedule=EVAL_SCHEDULE,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,
            rain_normaliser=rain_normaliser,
        )

    def _make_bias_eval_env():
        """*** CHANGED: deterministic bias eval (was randomized). ***"""
        return IrrigationEnv(
            randomize=False,
            eval_schedule=BIAS_EVAL_SCHEDULE,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,
            rain_normaliser=rain_normaliser,
        )

    train_env     = DummyVecEnv([_make_env])
    eval_env      = DummyVecEnv([_make_eval_env])
    bias_eval_env = DummyVecEnv([_make_bias_eval_env])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)        # harmless: eval env makes no RNG draw
    bias_eval_env.seed(seed + 2000)

    policy_kwargs = make_sac_policy_kwargs(
        N=N_AGENTS,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
    )

    lr_schedule = _make_lr_schedule(LR_START, LR_END)

    action_noise = NormalActionNoise(
        mean=np.zeros(N_AGENTS, dtype=np.float64),
        sigma=explore_sigma_start * np.ones(N_AGENTS, dtype=np.float64),
    )

    AsymmetricLRSAC.actor_lr_mult = float(actor_lr_mult)
    model = AsymmetricLRSAC(
        policy=V216CTDESACPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=gamma,
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

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    # *** CHANGED: FixedScheduleEvalCallback (was plain EvalCallback) ***
    eval_callback = FixedScheduleEvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(save_dir / "checkpoints"),
        name_prefix=run_name,
        save_replay_buffer=False,
        verbose=1,
    )
    rotating_buffer_callback = RotatingReplayBufferCheckpoint(
        save_freq=CHECKPOINT_FREQ,
        save_path=save_dir,
        verbose=1,
    )
    grad_clip_callback = GradClipCallback(max_grad_norm=MAX_GRAD_NORM)

    bias_ratio_cb = BiasRatioCallback(
        eval_env=bias_eval_env,
        eval_freq=BIAS_RATIO_FREQ,
        n_eval_episodes=BIAS_RATIO_N_EPISODES,
        save_path=str(save_dir),
        verbose=1,
    )
    action_stats_cb = ActionStatsCallback(log_freq=ACTION_STATS_FREQ)
    optimizer_lr_cb = OptimizerLRCallback(log_freq=LR_LOG_FREQ)

    reinjection_cb = LateNoiseReinjectionCallback(
        sigma_start=explore_sigma_start,
        sigma_floor=explore_sigma_floor,
        sigma_reinject=explore_sigma_reinject,
        decay_steps=explore_decay_steps,
        reinject_start=reinject_start,
        reinject_end=reinject_end,
        log_freq=EXPLORE_LOG_FREQ,
        csv_path=str(save_dir / "exploration_sigma_log.csv"),
        verbose=1,
    )
    coverage_cb = LowActionCoverageCallback(
        low_thresh=1.0 / 12.0,
        log_freq=COVERAGE_LOG_FREQ,
        csv_path=str(save_dir / "low_action_coverage_log.csv"),
        wet_rain_threshold_mm=WET_RAIN_THRESHOLD_MM,
        verbose=0,
    )

    # *** NEW: diagnostic guards (telemetry only, no abort) ***
    collapse_guard_cb = CollapseGuardCallback(
        collapse_frac=GUARD_COLLAPSE_FRAC,
        warmup_steps=GUARD_WARMUP_STEPS,
        check_freq=GUARD_CHECK_FREQ,
        window=GUARD_WINDOW,
        abort_on_collapse=GUARD_ABORT,
        csv_path=str(save_dir / "collapse_guard_log.csv"),
        verbose=1,
    )
    nonfinite_guard_cb = NonFiniteGuardCallback(
        stop_on_nonfinite=True,
        csv_path=str(save_dir / "nonfinite_guard_log.csv"),
        verbose=1,
    )

    cb_list = [
        eval_callback,
        checkpoint_callback,
        rotating_buffer_callback,
        grad_clip_callback,
        bias_ratio_cb,
        action_stats_cb,
        optimizer_lr_cb,
        reinjection_cb,
        coverage_cb,
        collapse_guard_cb,
        nonfinite_guard_cb,
    ]

    if wandb_active:
        try:
            from wandb.integration.sb3 import WandbCallback
            cb_list.append(WandbCallback(
                model_save_path=str(save_dir / "wandb_models"),
                model_save_freq=CHECKPOINT_FREQ, verbose=0,
            ))
        except Exception as e:
            print(f"[WandB] WandbCallback unavailable ({e}); continuing without it.")

    callbacks = CallbackList(cb_list)

    print(f"\n{'='*72}")
    print(f"  SAC v2.18-P3b-devset (deterministic dev-set eval) — seed {seed}")
    print(f"  Architecture: v2.16/v2.17 (V211 LN critic + LeakyReLU actor)")
    print(f"  ent_coef:     {ent_coef}")
    print(f"  noise:        decay {explore_sigma_start:.2f}->{explore_sigma_floor:.2f} "
          f"over {explore_decay_steps:,}; pulse peak {explore_sigma_reinject:.2f} "
          f"over [{reinject_start:,}, {reinject_end:,}]")
    print(f"  rain_norm:    {rain_normaliser:.1f}  | r6: {reward_overshoot_mode}")
    print(f"  GAMMA:        {gamma}  | TAU: {TAU}  | actor LR x{actor_lr_mult}")
    print(f"  DEV_YEARS:    {list(DEV_YEARS)}")
    print(f"  EVAL:         deterministic devset {N_EVAL_EPISODES} episodes "
          f"(FixedScheduleEvalCallback)")
    print(f"  TRAINING:     {len(TRAINING_YEARS)} years: {list(TRAINING_YEARS)}")
    print(f"  Total steps:  {total_timesteps:,}")
    print(f"  Output:       {save_dir}")
    print(f"{'='*72}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
            progress_bar=True,
        )
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
    parser = argparse.ArgumentParser(
        description=(
            "Train SAC v2.18-P3b with deterministic dev-set eval on "
            "DEV_YEARS={2002,2004,2013}. Identical to v2.18-P3b except "
            "the eval scheduling, for fair comparison with TD3 v2.21c."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA)
    parser.add_argument("--actor-lr-mult",   type=float, default=ACTOR_LR_MULT)
    parser.add_argument("--ent-coef",        type=float, default=ENT_COEF)
    parser.add_argument("--reward-overshoot-mode", type=str,
                        default=REWARD_OVERSHOOT_MODE,
                        choices=['quadratic', 'linear', 'sqrt'])
    parser.add_argument("--rain-normaliser", type=float, default=RAIN_NORMALISER)
    parser.add_argument("--explore-sigma-start",    type=float, default=EXPLORE_SIGMA_START)
    parser.add_argument("--explore-sigma-floor",    type=float, default=EXPLORE_SIGMA_FLOOR)
    parser.add_argument("--explore-sigma-reinject", type=float, default=EXPLORE_SIGMA_REINJECT)
    parser.add_argument("--explore-decay-steps",    type=int,   default=EXPLORE_DECAY_STEPS)
    parser.add_argument("--reinject-start",         type=int,   default=REINJECT_START)
    parser.add_argument("--reinject-end",           type=int,   default=REINJECT_END)
    args = parser.parse_args()

    train_sac_v218_p3b_devset(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        actor_lr_mult=args.actor_lr_mult,
        ent_coef=args.ent_coef,
        reward_overshoot_mode=args.reward_overshoot_mode,
        rain_normaliser=args.rain_normaliser,
        explore_sigma_start=args.explore_sigma_start,
        explore_sigma_floor=args.explore_sigma_floor,
        explore_sigma_reinject=args.explore_sigma_reinject,
        explore_decay_steps=args.explore_decay_steps,
        reinject_start=args.reinject_start,
        reinject_end=args.reinject_end,
    )
