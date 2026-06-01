# src/rl/train_v218_p3b.py  v2.18-P3b  (v2.16 architecture + alpha=0.002 + late noise reinjection)
# -----------------------------------------------------------------------------
# WHY v2.18-P3b EXISTS  (push the Path-3 lever further)
# ----------------------------------------------------
# v2.17-P3 (alpha=0.005 + decaying noise 0.30->0 over 60k) succeeded: it took
# wet-year over-irrigation from the v2.16-fixed baseline toward MPC, with the
# cleanest stability of any recent run.  Measured (wet-year, mean of 3 budgets,
# perfect forecast):
#       metric            v2.16-fixed   v2.17-P3   auto-alpha(crashed)   MPC
#       x1 median (mm)        152          144            132            130
#       waterlog days          76           54             35             18
#       water used (mm)       397          362            324            309
#       wet yield (kg/ha)    3391         3604           3633           3752
#   Stability: q_pred_mean stayed bounded (15-41, never negative); final
#   q_inflation +6.4% (v2.16-fixed ended at -146%).
#
# v2.17-P3 got ~80% of the way to the auto-alpha operating point but left a
# residual: wet x1 still ~4 mm above FC and ~58% of cell-days over FC.  Two
# compounding causes for the residual:
#   (a) ENTROPY PIN: alpha=0.005 still exerts the tanh-center spring that pulls
#       mu toward 6 mm/day (the action-space midpoint).  The auto-alpha run
#       reached x1=132 only because alpha collapsed to ~0.
#   (b) COVERAGE DECAY: exploration noise hit 0 at 60k, then the policy
#       exploited for 190k steps and stopped sampling the EVEN-LOWER actions
#       (0-2 mm in wet states); the critic's data there went stale.
#
# THE v2.18-P3b CHANGES  (two training-time only; architecture untouched):
#   1. ent_coef: 0.005 -> 0.002   (weaken the entropy pin further; attacks (a)).
#      NOTE confound: changing alpha AND adding a late pulse together means a
#      positive result is not cleanly attributable to one.  Accepted per the
#      user's choice to run both; the coverage telemetry (now fixed, see below)
#      plus the sigma/phase log let a follow-up disambiguate cheaply.
#   2. LATE NOISE REINJECTION (attacks (b)): exploration noise anneals 0.30->0
#      over the first 60k (as v2.17), then a triangular pulse of height 0.15 is
#      re-injected over [150k, 180k], peaking at 165k, then back to 0.  By 150k
#      the critic is well-trained where the policy visits, so the pulse's low-
#      water transitions get accurate values immediately and can pull mu down
#      the last bit -- without early-training instability.
#
# STABILITY WATCH: at alpha=0.002 the entropy regularisation is weaker than the
# v2.17 run that stayed clean.  The bias-ratio log is the guard: if q_pred_mean
# goes negative or |q_inflation_pct| exceeds ~80% late in training, alpha=0.002
# removed too much smoothing -> that is the signal to migrate to TD3 (which
# replaces entropy's smoothing with target-policy smoothing) rather than push
# alpha lower.
#
# TELEMETRY FIX (the v2.17 gap): LowActionCoverageCallback now detects wet
# episodes PHYSICALLY by seasonal rainfall (>=120 mm) instead of a label that
# almost never exists during training -- so frac_low_action_wet will actually
# be populated this run, making the coverage hypothesis directly checkable.
#
# Architecturally BYTE-IDENTICAL to v2.16/v2.17 (V216CTDESACPolicy,
# obs_norm_marker=2.16, RAIN_REF=30, linear r6).  runner.py evaluates these
# checkpoints with ZERO eval-side changes.
#
# ACCEPTANCE CRITERIA (evaluate on the 9-cell grid via runner.py; decide on
# x1/waterlog, NOT corr or yield):
#   PRIMARY:   wet x1 median < 138 mm (v2.17: 144; auto-alpha: 132; MPC: 130)
#              wet waterlog  < 48     (v2.17: 54; auto-alpha: 35; MPC: 18)
#   SECONDARY: wet water < 350 mm (v2.17: 362); wet/100 u_mean clearly < dry/100.
#   COVERAGE:  p3/frac_low_action_wet > 0 in BOTH the initial-decay phase AND
#              the reinjection pulse (proves the late pulse repopulated low water
#              in wet states).
#   STABILITY: critic_loss < 100 throughout; q_pred_mean never negative;
#              |q_inflation_pct| < 80%.  Breach => go TD3, do not lower alpha.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
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
)
from src.rl.train import (
    RotatingReplayBufferCheckpoint,
    GradClipCallback,
    _make_lr_schedule,
    _init_wandb,
)
from src.rl.train_v212 import AsymmetricLRSAC


# -----------------------------------------------------------------------------
# Hyperparameters -- identical to v2.16/v2.17 EXCEPT ent_coef and noise schedule.
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256

GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5
ACTOR_LR_MULT    = 5.0
ENT_COEF         = 0.002         # *** v2.18-P3b CHANGE: 0.005 -> 0.002 ***

MAX_GRAD_NORM    = 1.0
LEARNING_STARTS  = 1_000
GRADIENT_STEPS   = 1
TRAIN_FREQ       = 1

EVAL_FREQ        = 25_000
N_EVAL_EPISODES  = 9
CHECKPOINT_FREQ  = 25_000

ACTOR_HIDDEN  = [128, 128]
CRITIC_HIDDEN = [256, 256]

BIAS_RATIO_FREQ          = 25_000
BIAS_RATIO_N_EPISODES    = 3
ACTION_STATS_FREQ        = 1_000
LR_LOG_FREQ              = 1_000

# v2.18-P3b exploration schedule: initial decay + late triangular pulse.
EXPLORE_SIGMA_START    = 0.30
EXPLORE_SIGMA_FLOOR    = 0.0
EXPLORE_SIGMA_REINJECT = 0.15      # peak of the late pulse (half the start)
EXPLORE_DECAY_STEPS    = 60_000
REINJECT_START         = 150_000
REINJECT_END           = 180_000
EXPLORE_LOG_FREQ       = 1_000
COVERAGE_LOG_FREQ      = 1_000
WET_RAIN_THRESHOLD_MM  = 120.0     # season rain >= this -> "wet" episode

N_AGENTS = 130

# Env config carried from v2.16/v2.17 (unchanged).
REWARD_OVERSHOOT_MODE = 'linear'
RAIN_NORMALISER       = RAIN_REF_V216   # 30.0


def train_sac_v218_p3b(
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
    """Train SAC v2.18-P3b: v2.16 architecture + alpha=0.002 + late noise reinjection.

    Architecturally byte-identical to v2.16/v2.17 (V216CTDESACPolicy,
    marker=2.16, RAIN_REF=30, linear r6, LayerNorm VDN critic, asymmetric actor
    LR).  Training-time changes only: ent_coef lowered to 0.002, and a two-phase
    exploration-noise schedule (initial 0.30->0 anneal over 60k, then a
    triangular re-injection pulse peaking at 0.15 over [150k, 180k]).
    """
    run_name = f"sac_v218_p3b_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.18.0-P3b",
        "experiment": "v218_p3b_v216_arch_PLUS_alpha002_PLUS_late_reinjection",
        "seed": seed,
        "algorithm": "SAC (stable_baselines3) + asymmetric actor LR + action noise",
        "policy_class": "V216CTDESACPolicy (architecturally identical to v2.16/v2.17, marker=2.16)",
        "total_timesteps": total_timesteps,
        "ent_coef": ent_coef,
        "explore_sigma_start": explore_sigma_start,
        "explore_sigma_floor": explore_sigma_floor,
        "explore_sigma_reinject": explore_sigma_reinject,
        "explore_decay_steps": explore_decay_steps,
        "reinject_start": reinject_start,
        "reinject_end": reinject_end,
        "rain_normaliser": rain_normaliser,
        "reward_overshoot_mode": reward_overshoot_mode,
        "gamma": gamma,
        "tau": TAU,
        "actor_lr_mult": actor_lr_mult,
        "hypothesis": (
            "Residual wet-year over-irrigation after v2.17-P3 is (a) the "
            "alpha=0.005 entropy pin and (b) stale critic data on sub-2mm "
            "actions after noise decayed at 60k.  Lowering alpha to 0.002 and "
            "re-injecting a noise pulse at 150-180k (when the critic is well "
            "trained) should pull wet x1 from 144 toward the MPC/auto-alpha "
            "operating point (~130) while preserving stability."
        ),
    }

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments -- identical to v2.16/v2.17.
    # -------------------------------------------------------------------------
    def _make_env():
        return IrrigationEnv(
            randomize=True,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,
            rain_normaliser=rain_normaliser,
        )

    train_env     = DummyVecEnv([_make_env])
    eval_env      = DummyVecEnv([_make_env])
    bias_eval_env = DummyVecEnv([_make_env])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)
    bias_eval_env.seed(seed + 2000)

    policy_kwargs = make_sac_policy_kwargs(
        N=N_AGENTS,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
    )

    lr_schedule = _make_lr_schedule(LR_START, LR_END)

    # Exploration noise; sigma is overwritten every step by the reinjection
    # callback. The value here is just the t=0 magnitude.
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
        ent_coef=ent_coef,                     # *** 0.002 ***
        action_noise=action_noise,             # *** injected exploration ***
        learning_starts=LEARNING_STARTS,
        gradient_steps=GRADIENT_STEPS,
        train_freq=TRAIN_FREQ,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # -------------------------------------------------------------------------
    # Callbacks: full v2.16/v2.17 set + the late-reinjection scheduler + the
    # (now fixed) coverage telemetry.
    # -------------------------------------------------------------------------
    eval_callback = EvalCallback(
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
        low_thresh=1.0 / 12.0,                 # < 1 mm/day at UB_MM=12
        log_freq=COVERAGE_LOG_FREQ,
        csv_path=str(save_dir / "low_action_coverage_log.csv"),
        wet_rain_threshold_mm=WET_RAIN_THRESHOLD_MM,   # *** fixed wet detection ***
        verbose=0,
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
    ]

    if wandb_active:
        try:
            from wandb.integration.sb3 import WandbCallback
            wandb_cb = WandbCallback(
                model_save_path=str(save_dir / "wandb_models"),
                model_save_freq=CHECKPOINT_FREQ,
                verbose=0,
            )
            cb_list.append(wandb_cb)
        except Exception as e:
            print(f"[WandB] WandbCallback unavailable ({e}); continuing without it.")

    callbacks = CallbackList(cb_list)

    print(f"\n{'='*72}")
    print(f"  SAC training - v2.18-P3b (alpha=0.002 + late reinjection) - seed {seed}")
    print(f"  Architecture: v2.16/v2.17 (V211 LN critic + LeakyReLU actor, marker=2.16)")
    print(f"  ent_coef:     {ent_coef}   *** v2.18-P3b: lowered from 0.005 ***")
    print(f"  noise:        decay {explore_sigma_start:.2f}->{explore_sigma_floor:.2f} "
          f"over {explore_decay_steps:,}; pulse peak {explore_sigma_reinject:.2f} "
          f"over [{reinject_start:,}, {reinject_end:,}]  *** v2.18-P3b ***")
    print(f"  rain_norm:    {rain_normaliser:.1f}  | r6: {reward_overshoot_mode}")
    print(f"  GAMMA:        {gamma}  | TAU: {TAU}  | actor LR x{actor_lr_mult}")
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
            "Train SAC v2.18-P3b: v2.16 architecture with ent_coef 0.005->0.002 "
            "and a late noise-reinjection pulse, to push residual wet-year "
            "over-irrigation toward the MPC operating point. Architecturally "
            "identical to v2.16/v2.17; runner.py evaluates unchanged."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA)
    parser.add_argument("--actor-lr-mult",   type=float, default=ACTOR_LR_MULT)
    parser.add_argument("--ent-coef",        type=float, default=ENT_COEF,
                        help="Entropy coefficient (default 0.002 = v2.18-P3b).")
    parser.add_argument("--reward-overshoot-mode", type=str,
                        default=REWARD_OVERSHOOT_MODE,
                        choices=['quadratic', 'linear', 'sqrt'],
                        help="r6 shape (default 'linear' carried from v2.15-v2.17).")
    parser.add_argument("--rain-normaliser", type=float, default=RAIN_NORMALISER,
                        help="Rainfall denominator (default 30.0 = v2.16/v2.17/v2.18).")
    parser.add_argument("--explore-sigma-start",    type=float, default=EXPLORE_SIGMA_START)
    parser.add_argument("--explore-sigma-floor",    type=float, default=EXPLORE_SIGMA_FLOOR)
    parser.add_argument("--explore-sigma-reinject", type=float, default=EXPLORE_SIGMA_REINJECT,
                        help="Peak std of the late re-injection pulse (default 0.15).")
    parser.add_argument("--explore-decay-steps",    type=int,   default=EXPLORE_DECAY_STEPS)
    parser.add_argument("--reinject-start",         type=int,   default=REINJECT_START)
    parser.add_argument("--reinject-end",           type=int,   default=REINJECT_END)
    args = parser.parse_args()

    train_sac_v218_p3b(
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
