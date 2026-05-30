# src/rl/train_v216.py  v2.16.0 (v2.15 architecture + RAIN_REF=30 + capped auto-alpha)
# -----------------------------------------------------------------------------
# WHY v2.16 EXISTS
# ----------------
# v2.15 (v2.14 architecture + linear r6) trained stably and reached
# u_min=1.32 mm at 250k (proving the actor CAN produce low actions) but did
# not improve yield over v2.14 - in particular, wet-year yield dropped
# 31-64 kg/ha across all three budgets, with wet/100 water USE rising from
# 402 mm (v2.14) to 428 mm (v2.15).  Direct gradient analysis on the v2.15
# 250k actor (BS=2000, realistic-distribution inputs) revealed:
#
#   feature                  |dmu/dx_i|   input range   effective sensitivity
#   ----------------------------------------------------------------------
#   rain forecast (8d mean)    0.43         0.36              0.16
#   ETc  forecast (8d mean)    1.24         1.44              1.78
#   rad  forecast (8d mean)    1.40         1.56              2.19
#   today's rain               0.67         0.36              0.24
#   today's ETc                0.71         1.44              1.02
#   x1 (soil moisture)         2.03         3.00              6.08
#
# The actor IS responsive to ET and radiation forecasts (effective sensitivity
# > 1.7) but rain has effective sensitivity 0.16 because the input dynamic
# range is compressed: rainfall is divided by RAIN_REF=70.0 mm/day but the
# empirical growing-season distribution has median 0.07 mm and p99 12.7 mm.
# So rain/70 has median 0.001 and p99 0.18, and after the 2x-1 re-centering
# the rain input occupies only the bottom 0.36 of the [-1, +1] interval.
#
# This is the rain-blindness diagnostic.  It explains corr(u, rain_fwd7)
# approx +0.03 in v2.15 (vs MPC's -0.42): the rain channel cannot move
# enough to drive the actor's output even though the gradient through it
# is moderate.
#
# TWO PAIRED CHANGES IN v2.16
# ---------------------------
# 1. RAIN_REF: 70.0 -> 30.0.  Rationale documented in gym_env.py.
#    - Effective sensitivity to rain rises from 0.16 to ~0.36-0.40.
#    - Median rain still maps to ~-1 (near floor, correct: no rain = no signal).
#    - 5 mm rain: -1.0 -> -0.67 (was -0.98)  --  small rain becomes legible.
#    - 15 mm rain: -1.0 -> 0.0   (was -0.57)
#    - 30 mm rain: -1.0 -> +1.0  (was -0.14)  --  heavy rain saturates.
#    - 2024 wet-year max (36 mm) clips at +1.0 with 1.08% of days clipped.
#    - Training-year clip rate: 0.09%.
#    - Compared to RAIN_REF=15 (clips 2.15% of 2024 days including 14, 19, 35 mm
#      events), RAIN_REF=30 preserves the moderate-heavy rain events the
#      wet-year over-irrigation pathology occurs around.
#    - All other normalisers (ETC_REF=8, RAD_REF=35) unchanged.
#
# 2. ent_coef: 0.01 fixed -> auto-tuned, capped at 0.1, target_entropy=-65.
#    - SB3's default target_entropy for SAC is -dim(action) = -130 for our
#      action space.  This default was derived for monolithic action spaces
#      (one independent head per action dimension).  Our actor is VDN-factorised:
#      130 actions are produced by a SHARED per-agent network.  Per-cell
#      entropy is the actual control variable.  Target -130 means -1.0
#      per cell (very low entropy per cell, encouraging near-deterministic
#      per-cell actions).  Target -65 means -0.5 per cell (moderately
#      stochastic per cell), which better encourages buffer diversity for
#      the shared actor structure.
#    - target_entropy=-65 anchors to the original v2.4-v2.6 target_entropy
#      range (-13 to -65) which had been tuned for this VDN-factorised actor
#      before SB3's auto-α default was adopted.
#    - alpha cap at 0.1: v2.7's cascade happened at fixed alpha=0.05, so 0.1
#      leaves some safety margin above without being unbounded.  SB3 defaults
#      to alpha=1.0 unbounded.
#    - log_ent_coef initialised at log(0.05) approx -3.0 (v2.14/v2.15 known-
#      stable level), free to descend to arbitrarily small or rise to cap.
#    - After each ent_coef gradient step, log_ent_coef is clipped to
#      [-inf, log(0.1)].
#
# Architecturally byte-identical to v2.15.  All other hyperparameters
# unchanged: V211 LayerNorm critic, LeakyReLU actor, actor input re-center,
# normalised globals, asymmetric actor LR (5x), gamma=0.99, tau=0.005,
# linear r6, batch 256, buffer 250k, 250k steps, MAX_GRAD_NORM=1.0.
#
# ACCEPTANCE CRITERIA
# -------------------
# Primary  ("did the actor become rain-responsive?"):
#   - corr(u, rain_fwd7) in wet/100 becomes more negative than -0.10
#     (v2.15: +0.03; MPC: -0.42).  Even -0.10 would be a 100% improvement.
#   - corr(u, rain_today) more negative than -0.30  (v2.15: -0.28).
# Secondary  ("did wet-year over-irrigation drop?"):
#   - wet/100 water_used_mm < 400 mm  (v2.15: 428; v2.14: 402; MPC: 310).
#   - wet/100 waterlog_days_per_agent < 60  (v2.14: 80; MPC: 19).
# Stability  ("alpha trajectory and Q stayed sane"):
#   - ent_coef stays in [1e-6, 0.1] (cap not violated; entropy doesn't collapse).
#   - critic_loss < 100 throughout; |q_inflation_pct| < 50%.
# Stretch  ("policy quality improved"):
#   - 9-cell mean yield >= 3720  (v2.14: 3710; v2.15: 3680).
#   - wet-year mean yield > 3450  (v2.15: 3392; v2.14: 3444; MPC: 3752).
#
# RISKS (honest)
# --------------
#   - The actor was already responsive to ET and rad forecasts.  If wet-year
#     over-irrigation has causes BEYOND rain-blindness (e.g. the actor's
#     biomass-incentive in r1 making FC the local reward optimum), rain
#     rescaling alone won't close the gap.  We'd see corr(u, rain_fwd7)
#     improve but yield and water use stay similar.
#   - Auto-alpha could descend close to 0, collapsing exploration.  Floor at
#     log(1e-6) approx -13.8 by construction (SB3 does not impose a floor;
#     the optimizer's gradient typically keeps it well above this) but worth
#     monitoring.
#   - Auto-alpha could pin at the 0.1 cap if entropy keeps trying to rise.
#     This would happen only if the actor is producing very low entropy and
#     auto-alpha is pushing back hard - the symptom of action collapse.  The
#     cap then prevents the cascade.  Cap at 0.1 is consistent with v2.13's
#     0.05 plus 2x safety margin.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.gym_env import IrrigationEnv, RAIN_REF_V216
from src.rl.networks import V216CTDESACPolicy, make_sac_policy_kwargs
from src.rl.callbacks_v210 import (
    BiasRatioCallback,
    ActionStatsCallback,
    OptimizerLRCallback,
)
from src.rl.train import (
    RotatingReplayBufferCheckpoint,
    GradClipCallback,
    _make_lr_schedule,
    _init_wandb,
)
from src.rl.train_v212 import AsymmetricLRSAC


# -----------------------------------------------------------------------------
# Hyperparameters - identical to v2.15 EXCEPT ent_coef, target_entropy, and the
# env's rain_normaliser.
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256

GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5
ACTOR_LR_MULT    = 5.0           # same asymmetric LR as v2.12-v2.15

# v2.16 ent_coef config:
#   SB3 string 'auto_X.XX' = auto-tune log_ent_coef, initialised at log(X.XX).
#   Initial value chosen at 0.05 (v2.14/v2.15 known-stable; also equal to the
#   cap so the cap engages immediately on the first upward step).
ENT_COEF_INIT    = "auto_0.05"
ENT_COEF_CAP     = 0.1           # log(0.1) approx -2.303 is the upper bound
TARGET_ENTROPY   = -65.0         # vs SB3 default -130 (-dim(action_space))

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

# v2.16 env config:
REWARD_OVERSHOOT_MODE = 'linear'        # carried forward from v2.15 (unchanged)
RAIN_NORMALISER       = RAIN_REF_V216   # 30.0 — *** THE v2.16 ENV CHANGE ***


# -----------------------------------------------------------------------------
# AsymmetricLRSAC + capped auto-alpha
# -----------------------------------------------------------------------------
# SB3's SAC implements auto-alpha by storing log_ent_coef as a learnable
# scalar parameter (self.log_ent_coef).  Each train() step:
#   1. computes ent_coef = exp(log_ent_coef.detach()) and uses it in the
#      critic target and actor loss,
#   2. computes ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
#   3. backprops and steps self.ent_coef_optimizer.
# To cap alpha at 0.1 we clip log_ent_coef.data <= log(0.1) AFTER each ent_coef
# optimizer step.  This is the minimum-invasiveness implementation; it does
# not modify the gradient computation, only the post-update parameter value.
# -----------------------------------------------------------------------------
class CappedAutoAlphaAsymmetricLRSAC(AsymmetricLRSAC):
    """AsymmetricLRSAC variant that caps log_ent_coef at log(ent_coef_cap)
    after every gradient step.  Class attribute ent_coef_cap controls the
    cap (set externally before instantiation).

    The cap is enforced AFTER the auto-tune optimizer step, so the gradient
    is computed against the unrestricted objective and only the resulting
    parameter is clipped.  This matches the standard "projected gradient"
    pattern.
    """

    ent_coef_cap: float = 0.1   # default; set externally before .learn()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Run the standard SAC train (which steps log_ent_coef if
        # ent_coef_optimizer is present).
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        # Clip log_ent_coef to enforce the cap.  Only applies when SB3 is
        # running in auto-tune mode (ent_coef_optimizer is not None).
        if (self.ent_coef_optimizer is not None
                and getattr(self, "log_ent_coef", None) is not None):
            cap = math.log(float(self.ent_coef_cap))
            with torch.no_grad():
                self.log_ent_coef.data.clamp_(max=cap)


def train_sac_v216(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    gamma: float = GAMMA,
    actor_lr_mult: float = ACTOR_LR_MULT,
    ent_coef_init: str = ENT_COEF_INIT,
    ent_coef_cap: float = ENT_COEF_CAP,
    target_entropy: float = TARGET_ENTROPY,
    reward_overshoot_mode: str = REWARD_OVERSHOOT_MODE,
    rain_normaliser: float = RAIN_NORMALISER,
) -> SAC:
    """Train a SAC v2.16 agent.

    v2.16 = v2.15 architecture (LayerNorm critic, LeakyReLU actor, normalised
    globals, actor input re-center, asymmetric LR, linear r6) with two
    paired training-time changes:
      - rain normaliser tightened from 70 to 30 to give the rain forecast
        channel a usable input dynamic range (3x effective sensitivity);
      - SAC ent_coef switches from fixed 0.01 to auto-tuned with cap 0.1 and
        target_entropy=-65 (matches the VDN-factorised actor structure
        better than SB3's default -130).
    """
    run_name = f"sac_v216_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.16.0",
        "experiment": "v216_v215_arch_PLUS_rain_ref_30_PLUS_capped_autoalpha",
        "seed": seed,
        "algorithm": "SAC (stable_baselines3) + asymmetric LR + capped auto-alpha",
        "policy_class": "V216CTDESACPolicy (V215 actor/critic, marker=2.16)",
        "total_timesteps": total_timesteps,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": gamma,
        "tau": TAU,
        "lr_start": LR_START,
        "lr_end": LR_END,
        "actor_lr_mult": actor_lr_mult,
        "ent_coef_init": ent_coef_init,            # *** NEW: auto-tune ***
        "ent_coef_cap": ent_coef_cap,              # *** NEW: alpha cap ***
        "target_entropy": target_entropy,          # *** NEW: -65 vs SB3 -130 ***
        "reward_overshoot_mode": reward_overshoot_mode,
        "rain_normaliser": rain_normaliser,        # *** NEW: 30 vs prior 70 ***
        "alpha6_lin": 1.5,
        "max_grad_norm": MAX_GRAD_NORM,
        "actor_hidden": ACTOR_HIDDEN,
        "critic_hidden": CRITIC_HIDDEN,
        "critic_layernorm": True,
        "actor_activation": "LeakyReLU(0.01)",
        "actor_input_recenter": "x' = 2*x - 1 (actor-only)",
        "normalize_globals": True,
        "learning_starts": LEARNING_STARTS,
        "gradient_steps": GRADIENT_STEPS,
        "train_freq": TRAIN_FREQ,
        "eval_freq": EVAL_FREQ,
        "n_eval_episodes": N_EVAL_EPISODES,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "obs_dim": 1097,
        "n_agent_features": 8,
        "changes_vs_v215": [
            "rain_normaliser: 70.0 -> 30.0.  RAIN_REF was diagnostically too "
            "large: the growing-season rainfall distribution has median 0.07 "
            "mm and p99 12.7 mm, so rain/70 occupied only the bottom 0.36 of "
            "the recentered [-1, +1] input range, vs ETc/8 and rad/35 which "
            "spanned 1.4-1.6.  Direct gradient analysis on the v2.15 250k "
            "actor showed effective sensitivity to rain was 0.16, vs 1.78 for "
            "ETc and 2.19 for radiation - the actor was rain-blind not because "
            "the gradient through rain was small but because the input never "
            "moved.  RAIN_REF=30 triples the effective sensitivity.",
            "ent_coef: 0.01 fixed -> auto-tuned, capped at 0.1, target_entropy "
            "= -65.  SB3's default target_entropy (-dim(action) = -130) was "
            "derived for monolithic action spaces and is too aggressive for "
            "our VDN-factorised actor structure (shared per-agent network "
            "producing 130 actions).  Target -65 corresponds to -0.5 per cell, "
            "matching the original v2.4-v2.6 target_entropy range that had "
            "been hand-tuned for this architecture.  Initial log_ent_coef at "
            "log(0.05) approx -3.0 (v2.14/v2.15 known-stable value).  Cap at "
            "log(0.1) approx -2.303 enforced by post-step parameter clipping "
            "in CappedAutoAlphaAsymmetricLRSAC.train()."
        ],
    }

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments - v2.15 settings PLUS rain_normaliser=30.
    # -------------------------------------------------------------------------
    def _make_env():
        return IrrigationEnv(
            randomize=True,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,
            rain_normaliser=rain_normaliser,            # *** v2.16 ***
        )

    train_env     = DummyVecEnv([_make_env])
    eval_env      = DummyVecEnv([_make_env])
    bias_eval_env = DummyVecEnv([_make_env])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)
    bias_eval_env.seed(seed + 2000)

    policy_kwargs = make_sac_policy_kwargs(
        N=130,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
    )

    lr_schedule = _make_lr_schedule(LR_START, LR_END)

    # Configure the asymmetric-LR SAC subclass:
    AsymmetricLRSAC.actor_lr_mult = float(actor_lr_mult)
    CappedAutoAlphaAsymmetricLRSAC.actor_lr_mult = float(actor_lr_mult)
    CappedAutoAlphaAsymmetricLRSAC.ent_coef_cap  = float(ent_coef_cap)

    model = CappedAutoAlphaAsymmetricLRSAC(
        policy=V216CTDESACPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=gamma,
        tau=TAU,
        ent_coef=ent_coef_init,        # *** auto-tune, initial 0.05 ***
        target_entropy=target_entropy, # *** -65 (vs SB3 default -130) ***
        learning_starts=LEARNING_STARTS,
        gradient_steps=GRADIENT_STEPS,
        train_freq=TRAIN_FREQ,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # Verify the cap is enforced from the start by clipping the initial value.
    # log(0.05) approx -3.0 is below log(0.1) approx -2.303, so this is a no-op,
    # but assert to fail loudly if SB3 changes the auto-alpha initialisation.
    if model.ent_coef_optimizer is not None and model.log_ent_coef is not None:
        cap_log = math.log(float(ent_coef_cap))
        initial_log_alpha = float(model.log_ent_coef.detach().item())
        assert initial_log_alpha <= cap_log + 1e-6, (
            f"Initial log_ent_coef {initial_log_alpha:.4f} exceeds cap "
            f"log({ent_coef_cap}) = {cap_log:.4f}.  SB3's auto-alpha "
            f"initialisation has changed; review CappedAutoAlphaAsymmetricLRSAC."
        )

    # -------------------------------------------------------------------------
    # Callbacks (identical set to v2.14/v2.15).
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

    cb_list = [
        eval_callback,
        checkpoint_callback,
        rotating_buffer_callback,
        grad_clip_callback,
        bias_ratio_cb,
        action_stats_cb,
        optimizer_lr_cb,
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
    print(f"  SAC training - v2.16 (RAIN_REF=30 + capped auto-alpha) - seed {seed}")
    print(f"  Architecture: v2.15 (V211 LN critic + LeakyReLU actor + recenter)")
    print(f"  Reward r6:    {reward_overshoot_mode}  (carried from v2.15)")
    print(f"  rain_normaliser: {rain_normaliser:.1f} mm/day  (was 70.0 in v2.15)")
    print(f"  ent_coef:     auto-tune, init {ent_coef_init}, cap {ent_coef_cap}")
    print(f"  target_entropy: {target_entropy}  (SB3 default would be -130)")
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
            "Train SAC v2.16: v2.15 architecture (LayerNorm critic, LeakyReLU "
            "actor, normalised globals, actor input re-center, asymmetric LR, "
            "linear r6) with two paired training-time changes: (1) rainfall "
            "normaliser tightened from 70 to 30, and (2) ent_coef switches "
            "from fixed 0.01 to auto-tuned with cap 0.1 and target_entropy=-65."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA)
    parser.add_argument("--actor-lr-mult",   type=float, default=ACTOR_LR_MULT)
    parser.add_argument("--ent-coef-init",   type=str,   default=ENT_COEF_INIT,
                        help="SB3 ent_coef string; 'auto_X.XX' enables tuning at init X.XX.")
    parser.add_argument("--ent-coef-cap",    type=float, default=ENT_COEF_CAP,
                        help="Upper bound on alpha (default 0.1 = v2.16 anchor).")
    parser.add_argument("--target-entropy",  type=float, default=TARGET_ENTROPY,
                        help="Target entropy (default -65 vs SB3 default -130).")
    parser.add_argument("--reward-overshoot-mode", type=str,
                        default=REWARD_OVERSHOOT_MODE,
                        choices=['quadratic', 'linear', 'sqrt'],
                        help="r6 shape (default 'linear' carried from v2.15).")
    parser.add_argument("--rain-normaliser", type=float, default=RAIN_NORMALISER,
                        help="Rainfall denominator (default 30.0 = v2.16 anchor).")
    args = parser.parse_args()

    train_sac_v216(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        actor_lr_mult=args.actor_lr_mult,
        ent_coef_init=args.ent_coef_init,
        ent_coef_cap=args.ent_coef_cap,
        target_entropy=args.target_entropy,
        reward_overshoot_mode=args.reward_overshoot_mode,
        rain_normaliser=args.rain_normaliser,
    )
