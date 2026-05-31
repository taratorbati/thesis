# src/rl/train_v217_p3.py  v2.17-P3  (v2.16 architecture + alpha=0.005 + decaying exploration noise)
# -----------------------------------------------------------------------------
# WHY v2.17-P3 EXISTS  (the "exploration injection" diagnostic, Path 3)
# --------------------------------------------------------------------
# Diagnosis carried in from the v2.14-v2.16 post-mortem:
#   * The wet-year defect is chronic over-irrigation: x1 median ~152 mm with
#     ~80% of cell-days above field capacity (FC=140), vs MPC's ~130 mm / ~20%.
#     This drives the wet-year yield gap (~3.4 t/ha vs MPC ~3.75) AND the
#     waterlog gap (~76 days/agent vs MPC ~20).
#   * The reward is NOT the cause: r6 (waterlog) already dominates the wet-year
#     return ~10x over r1+r2+r3 (season-sum r6 ~= -13.7 vs r1 ~= +1.27), and r6
#     is linear (carried from v2.15), matching the ABM's linear h6 stress.
#   * The actor's mean action barely moves between dry and wet (dry/100 u_mean
#     5.20 mm vs wet/100 u_mean 4.83 mm -- only 0.37 mm less despite rain),
#     i.e. the policy is near open-loop in water and the budget clip, not the
#     policy, sets total water use.
#   * Behavioural check: the actor NEVER voluntarily chooses u < 0.5 mm while
#     budget remains (frac = 0.00 across all 9 scenarios). Every near-zero
#     action coincides with budget exhaustion, not a learned rain response.
#
# HYPOTHESIS (buffer-coverage starvation):
#   The policy never SAMPLES low-water actions in unconstrained wet states, so
#   its critic never sees that low water is good there, so it never learns to
#   reduce water. Injecting decaying exploration noise that forces low-water-
#   in-wet transitions into the replay buffer should let the critic learn their
#   value and pull wet-year x1 down -- with NO change to reward, architecture,
#   or rain scaling.
#
# THE v2.17-P3 CHANGES  (two, both training-time only; architecture untouched):
#   1. ent_coef: 0.01 -> 0.005   (partial release of the entropy mu-centering
#      "action floor"; user-selected midpoint between v2.16's 0.01 and the
#      auto-alpha run's ~0 that accidentally produced the best wet behaviour).
#   2. Symmetric NormalActionNoise N(0, sigma(t)) added at collection time,
#      sigma linearly 0.30 -> 0.0 over the first 60,000 steps (24% of 250k),
#      then 0. Symmetric (not downward-biased) so a drop in x1 reflects the
#      critic LEARNING to prefer low water, not a biased data distribution.
#
# IMPORTANT: architecturally BYTE-IDENTICAL to v2.16. The policy class
# (V216CTDESACPolicy) and its actor.obs_norm_marker (2.16) are unchanged, so
# the existing runner.py evaluates v2.17-P3 checkpoints correctly with ZERO
# eval-side changes (marker 2.16 -> V216CTDESACPolicy + RAIN_REF=30 at eval).
#
# ACCEPTANCE CRITERIA (evaluate on the standard 9-cell grid via runner.py)
# -----------------------------------------------------------------------
# PRIMARY (confirms/falsifies coverage hypothesis):
#   - wet-year (mean of 3 budgets) x1 median  < 140 mm   (v2.16-fixed: 152; auto-alpha: 132; MPC: 130)
#   - wet-year waterlog_days_per_agent        < 55       (v2.16-fixed: 76;  auto-alpha: 35;  MPC: 18-25)
# SECONDARY:
#   - wet-year water_used_mm                  < 360 mm   (v2.16-fixed: 397)
#   - wet/100 u_mean meaningfully < dry/100 u_mean (currently 4.83 vs 5.20)
# COVERAGE CHECK (makes a NULL result interpretable -- new telemetry):
#   - p3/frac_low_action_wet > 0 during the exploration phase (proves the
#     buffer actually received low-water-in-wet transitions).
# STABILITY GUARD:
#   - train/critic_loss < 100 throughout; |q_inflation_pct| < 50%;
#     v210/action_std_spatial does not collapse < 0.1.
# NOT success metrics (log, do not decide on): corr(u, rain_fwd7) and yield
#   (single-seed yield is noise-dominated; corr is the wrong target).
#
# DECISION RULE:
#   wet x1 drops  -> coverage was the bottleneck; Path 3 is the fix.
#   wet x1 flat BUT frac_low_action_wet was > 0 -> coverage achieved but policy
#                    did not move: entropy mu-pin / r1 local optimum binds ->
#                    proceed to Path 1 (TD3, train_v218_td3.py).
#   training destabilises -> alpha=0.005 + noise removed too much smoothing ->
#                    argues for TD3 target-policy smoothing (Path 1).
#
# CONFOUND (honest): alpha and noise change together, so a positive result does
# not attribute the gain to one or the other. Accepted: both push toward the
# same goal (reach low water) and the aim here is effectiveness, not a clean
# single-variable ablation. The coverage telemetry enables a cheap alpha-only
# vs noise-only follow-up if this works.
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
    ExplorationNoiseDecayCallback,
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
# Hyperparameters -- identical to v2.16 EXCEPT ent_coef and the action noise.
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256

GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5
ACTOR_LR_MULT    = 5.0
ENT_COEF         = 0.005         # *** v2.17-P3 CHANGE: 0.01 -> 0.005 ***

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

# v2.17-P3 exploration-noise schedule (*** THE OTHER v2.17-P3 CHANGE ***)
EXPLORE_SIGMA_START  = 0.30      # normalised action units ([0,1]); ~1.8 mm std
EXPLORE_SIGMA_END    = 0.0       # exploration fully annealed off
EXPLORE_DECAY_STEPS  = 60_000    # 24% of 250k
EXPLORE_LOG_FREQ     = 1_000
COVERAGE_LOG_FREQ    = 1_000

N_AGENTS = 130

# Env config carried from v2.16 (unchanged).
REWARD_OVERSHOOT_MODE = 'linear'
RAIN_NORMALISER       = RAIN_REF_V216   # 30.0


def train_sac_v217_p3(
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
    explore_sigma_end: float = EXPLORE_SIGMA_END,
    explore_decay_steps: int = EXPLORE_DECAY_STEPS,
) -> SAC:
    """Train SAC v2.17-P3: v2.16 architecture + alpha=0.005 + decaying noise.

    Architecturally byte-identical to v2.16 (V216CTDESACPolicy, marker=2.16,
    RAIN_REF=30, linear r6, LayerNorm VDN critic, asymmetric actor LR). The only
    differences are training-time: ent_coef lowered to 0.005, and symmetric
    Gaussian exploration noise injected at collection time and annealed to 0
    over the first `explore_decay_steps` steps.
    """
    run_name = f"sac_v217_p3_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.17.0-P3",
        "experiment": "v217_p3_v216_arch_PLUS_alpha005_PLUS_decaying_noise",
        "seed": seed,
        "algorithm": "SAC (stable_baselines3) + asymmetric actor LR + action noise",
        "policy_class": "V216CTDESACPolicy (architecturally identical to v2.16, marker=2.16)",
        "total_timesteps": total_timesteps,
        "ent_coef": ent_coef,
        "explore_sigma_start": explore_sigma_start,
        "explore_sigma_end": explore_sigma_end,
        "explore_decay_steps": explore_decay_steps,
        "rain_normaliser": rain_normaliser,
        "reward_overshoot_mode": reward_overshoot_mode,
        "gamma": gamma,
        "tau": TAU,
        "actor_lr_mult": actor_lr_mult,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "hypothesis": (
            "Wet-year over-irrigation is buffer-coverage starvation: the policy "
            "never samples low-water actions in unconstrained wet states, so the "
            "critic never learns their value. Injecting decaying symmetric "
            "exploration noise (and partially releasing the entropy mu-pin via "
            "alpha 0.01->0.005) should populate the buffer with low-water-in-wet "
            "transitions and pull wet-year x1 down toward FC."
        ),
    }

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments -- identical to v2.16.
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

    # Symmetric Gaussian exploration noise across the 130 agent actions. Its
    # sigma is overwritten every step by ExplorationNoiseDecayCallback; the
    # value here is just the t=0 magnitude.
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
        ent_coef=ent_coef,                     # *** 0.005 ***
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
    # Callbacks -- the full v2.16 set (crash recovery, eval, bias-ratio,
    # action stats, LR readback) PLUS the two Path-3 exploration callbacks.
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

    # Path-3 callbacks.
    noise_decay_cb = ExplorationNoiseDecayCallback(
        sigma_start=explore_sigma_start,
        sigma_end=explore_sigma_end,
        decay_steps=explore_decay_steps,
        log_freq=EXPLORE_LOG_FREQ,
        csv_path=str(save_dir / "exploration_sigma_log.csv"),
        verbose=1,
    )
    coverage_cb = LowActionCoverageCallback(
        low_thresh=1.0 / 12.0,     # < 1 mm/day at UB_MM=12
        log_freq=COVERAGE_LOG_FREQ,
        csv_path=str(save_dir / "low_action_coverage_log.csv"),
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
        noise_decay_cb,
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
    print(f"  SAC training - v2.17-P3 (exploration injection) - seed {seed}")
    print(f"  Architecture: v2.16 (V211 LN critic + LeakyReLU actor + recenter, marker=2.16)")
    print(f"  Reward r6:    {reward_overshoot_mode}  (carried from v2.15/v2.16)")
    print(f"  rain_normaliser: {rain_normaliser:.1f} mm/day  (v2.16 = 30.0)")
    print(f"  ent_coef:     {ent_coef}   *** v2.17-P3: lowered from 0.01 ***")
    print(f"  action noise: N(0, sigma)  sigma {explore_sigma_start:.2f} -> "
          f"{explore_sigma_end:.2f} over {explore_decay_steps:,} steps  *** v2.17-P3 ***")
    print(f"  GAMMA:        {gamma}")
    print(f"  Critic LR:    {LR_START:.0e} -> {LR_END:.0e} (linear)")
    print(f"  Actor  LR:    {actor_lr_mult}x critic LR (asymmetric)")
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
            "Train SAC v2.17-P3: v2.16 architecture (LayerNorm VDN critic, "
            "LeakyReLU actor, RAIN_REF=30, linear r6, asymmetric LR) with two "
            "training-time changes - ent_coef 0.01->0.005 and decaying symmetric "
            "exploration noise - to test the buffer-coverage-starvation "
            "hypothesis for wet-year over-irrigation. Architecturally identical "
            "to v2.16; runner.py evaluates the checkpoints unchanged."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA)
    parser.add_argument("--actor-lr-mult",   type=float, default=ACTOR_LR_MULT)
    parser.add_argument("--ent-coef",        type=float, default=ENT_COEF,
                        help="Entropy coefficient (default 0.005 = v2.17-P3).")
    parser.add_argument("--reward-overshoot-mode", type=str,
                        default=REWARD_OVERSHOOT_MODE,
                        choices=['quadratic', 'linear', 'sqrt'],
                        help="r6 shape (default 'linear' carried from v2.15/v2.16).")
    parser.add_argument("--rain-normaliser", type=float, default=RAIN_NORMALISER,
                        help="Rainfall denominator (default 30.0 = v2.16/v2.17).")
    parser.add_argument("--explore-sigma-start", type=float, default=EXPLORE_SIGMA_START,
                        help="Initial exploration noise std in [0,1] action units (default 0.30).")
    parser.add_argument("--explore-sigma-end",   type=float, default=EXPLORE_SIGMA_END,
                        help="Final exploration noise std (default 0.0).")
    parser.add_argument("--explore-decay-steps",  type=int,   default=EXPLORE_DECAY_STEPS,
                        help="Env steps over which noise anneals (default 60000).")
    args = parser.parse_args()

    train_sac_v217_p3(
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
        explore_sigma_end=args.explore_sigma_end,
        explore_decay_steps=args.explore_decay_steps,
    )
