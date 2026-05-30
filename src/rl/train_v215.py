# src/rl/train_v215.py  v2.15.0 (v2.14 architecture trained with LINEAR r6)
# -----------------------------------------------------------------------------
# WHY v2.15 EXISTS
# ----------------
# v2.14 (LayerNorm critic + LeakyReLU actor + actor-only input re-center +
# normalised globals + asymmetric LR + α=0.01) achieved mean yield 3710 kg/ha
# across the 9-cell perfect-forecast grid - matching v2.7's transient
# cascade-peak (3716) with stable training.  But a water-use audit on the same
# grid revealed v2.14 over-irrigates relative to MPC by +37.5 mm and waterlogs
# the field +44 days more per agent, with the gap concentrated in wet years.
#
# Behavioural diagnostic (v2.14 best_model on perfect-forecast wet/100):
#   metric                v2.14    MPC      gap
#   daily u 10th-pctile    2.50    0.16     v2.14 cannot push action low
#   daily u 1st-pctile     2.65    0.03     v2.14 has a hard ~2.5 mm floor
#   corr(u, x1_soil)      -0.17   -0.24     soil sensitivity present (correct sign)
#   corr(u, rain_today)   -0.28   -0.39     same-day rain sensitivity present
#   corr(u, rain_fwd7)    +0.03   -0.42     FORECAST blindness (key defect)
#   waterlog days/agent   80.2    19.0      4x more saturated days
#   WUE (kg/ha/mm)         8.62   12.12     12% efficiency gap
#
# The actor IS state-responsive (correct-sign corr with x1, x5, ET, today's
# rain), but it cannot push actions toward zero on days when rain is coming.
# Two compounding mechanisms identified:
#
#   1. Quadratic r6 has weak marginal penalty at moderate overshoot.
#      d(r6_quad)/d(overshoot) = -2*ALPHA6*overshoot/FC^2.
#      At overshoot=12 mm (the typical v2.14 wet-year operating point):
#        |gradient| = 2*8*12/19600 ~ 0.0098
#      At overshoot=2 mm (dry-year transients):
#        |gradient| = 2*8*2/19600 ~ 0.0016 - essentially zero.
#      The critic's dQ/du signal at the operating point is small, so the
#      actor settles at a moderate-negative μ instead of going further.
#
#   2. The ABM's waterlog yield-loss term h6 is LINEAR in (x1-FC)/FC:
#        h6 = clip(1 - (x1-FC)/FC, 0, 1)
#      but the reward r6 is QUADRATIC.  The critic is learning a proxy that
#      doesn't match the actual physical yield loss.  MPC optimises the
#      physical objective directly; SAC optimises the proxy.
#
# THE v2.15 CHANGE (one variable):
#   r6 = -ALPHA6_LIN * mean(overshoot) / FC,   with ALPHA6_LIN = 1.5
#  (was: r6 = -ALPHA6 * mean(overshoot^2) / FC^2,  ALPHA6 = 8.0)
#
# Everything else identical to v2.14: V211 LayerNorm critic, LeakyReLU actor,
# actor input re-centre, normalised globals, asymmetric actor LR (5x),
# γ=0.99, τ=0.005, α=0.01, batch 256, buffer 250k, 250k steps, single-step
# ReplayBuffer, MAX_GRAD_NORM=1.0.  The policy class V215CTDESACPolicy differs
# from V214 only in the marker buffer value (2.15 vs 2.14).
#
# CALIBRATION OF ALPHA6_LIN
# -------------------------
# Computed over 27 stored rollouts (v2.14 + MPC, 9 perfect-forecast scenarios):
#   - Quadratic r6 season-sum (current value): 6.79 +- 7.86 (full distribution)
#                                               16.99 +- 4.24 (wet-year only)
#   - Linear r6 season-sum at ALPHA6_LIN=1.0:   4.63 +- 4.22
#   - ALPHA6_LIN to match full-dist season-sum: 1.4657
#   - ALPHA6_LIN to match wet-only season-sum:  1.6880
#   - ALPHA6_LIN to match dQ/du at RMS overshoot 13.4mm: 1.5285
# ALPHA6_LIN = 1.5 chosen as bracketed by all three calibrations.  Preserves
# r6's role as the dominant reward term in wet years while uniformising the
# gradient signal across the overshoot range.
#
# WHY LINEAR (not sqrt, not cubic, not gentler scaling)
# -----------------------------------------------------
# Linear is the unique shape that:
#   - Aligns r6 with the ABM's linear h6 stress (correct physical proxy).
#   - Produces uniform marginal penalty across overshoot magnitudes
#     (sharpens gradient at small overshoots, softens at large ones).
#   - Is the simplest single-parameter modification to test the
#     "quadratic was the bottleneck" hypothesis.
# Sub-quadratic (sqrt) is provided in the env for ablation but not used by
# default - the sqrt shape would still attenuate at moderate overshoot.
#
# ACCEPTANCE CRITERIA
# -------------------
# Primary  ("did the action floor drop?"):
#   - wet/100 daily u 10th-pctile < 1.5 mm (v2.14: 2.50; MPC: 0.16)
#   - wet/100 daily u 1st-pctile  < 0.8 mm (v2.14: 2.65; MPC: 0.03)
# Secondary  ("did forecast response emerge?"):
#   - wet/100 corr(u, rain_fwd7) < -0.20  (v2.14: +0.03; MPC: -0.42)
# Stability  ("training stayed clean"):
#   - |q_inflation_pct| < 50% throughout; critic_loss < 100
# Stretch  ("policy quality improved"):
#   - 9-cell mean yield >= 3720 (v2.14: 3710)
#   - wet-year mean yield > 3500 (v2.14: 3444; MPC: 3754)
#   - dry-year mean yield NOT below 3950 (v2.14: 3975; MPC: 3993)
#
# RISKS (honest)
# --------------
#   - Linear penalty applies even at small overshoots, so dry-year states
#     where the actor briefly exceeds FC for legitimate reasons (biomass
#     incentive) get penalised.  This could DROP dry-year yield by a few
#     percent.  Watch dry/100 in particular.  If dry yield drops more than
#     wet yield rises, the linear shape is wrong and sqrt is the fallback.
#   - The reward-shape change does not address replay-buffer concentration
#     or twin-critic OOD pessimism.  If linear r6 helps but plateaus before
#     closing the gap, v2.16 will be exploration-σ boost or min->mean Q.
#   - The seed-0 training has the same single-sample uncertainty as v2.14.
#     Multi-seed validation (seeds 1, 2) remains a separate work item.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.gym_env import IrrigationEnv
from src.rl.networks import V215CTDESACPolicy, make_sac_policy_kwargs
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
# Hyperparameters - identical to v2.14 EXCEPT the env's reward_overshoot_mode.
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256

GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5
ACTOR_LR_MULT    = 5.0           # same asymmetric LR as v2.12/v2.13/v2.14
ENT_COEF         = 0.01          # carried forward from v2.14 (unchanged)

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

# v2.15 single-variable change:
REWARD_OVERSHOOT_MODE = 'linear'   # was implicitly 'quadratic' in v2.7-v2.14


def train_sac_v215(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    gamma: float = GAMMA,
    actor_lr_mult: float = ACTOR_LR_MULT,
    ent_coef: float = ENT_COEF,
    reward_overshoot_mode: str = REWARD_OVERSHOOT_MODE,
) -> SAC:
    """Train a SAC v2.15 agent.

    v2.15 = v2.14 architecture (LayerNorm critic, LeakyReLU actor, normalised
    globals, actor-only input re-center, asymmetric LR, α=0.01) trained on a
    LINEAR r6 reward shape instead of quadratic.  Single-variable test of the
    reward-shape hypothesis: aligning the overshoot penalty with the ABM's
    linear waterlog stress term should give the critic uniform gradient
    signal across the overshoot range and pull the actor's μ further negative
    in wet states.
    """
    run_name = f"sac_v215_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.15.0",
        "experiment": "v215_v214_arch_PLUS_linear_r6",
        "seed": seed,
        "algorithm": "SAC (stable_baselines3) + asymmetric actor LR",
        "policy_class": "V215CTDESACPolicy (V214 actor/critic, marker=2.15)",
        "total_timesteps": total_timesteps,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": gamma,
        "tau": TAU,
        "lr_start": LR_START,
        "lr_end": LR_END,
        "actor_lr_mult": actor_lr_mult,
        "ent_coef": ent_coef,
        "reward_overshoot_mode": reward_overshoot_mode,   # *** THE NEW VARIABLE ***
        "alpha6_lin": 1.5,                                 # calibrated constant
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
        "changes_vs_v214": [
            "r6 reward shape: QUADRATIC -> LINEAR.  Was "
            "r6 = -ALPHA6 * mean(overshoot^2) / FC^2 with ALPHA6=8.0; "
            "now r6 = -ALPHA6_LIN * mean(overshoot) / FC with "
            "ALPHA6_LIN=1.5.  ALPHA6_LIN calibrated from 27 v2.14+MPC "
            "rollouts to (a) preserve season-sum r6 magnitude across the "
            "full distribution (6.79 -> matched at ALPHA6_LIN~1.47), and "
            "(b) match the |dQ/du| gradient at the RMS overshoot of "
            "13.4mm (matched at ALPHA6_LIN~1.53).  Chosen ALPHA6_LIN=1.5 "
            "is bracketed by both targets.  Linear shape aligns r6 with "
            "the ABM's linear waterlog stress h6, and provides uniform "
            "gradient signal across the overshoot range (vs quadratic's "
            "0 gradient at zero overshoot)."
        ],
    }

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments - same as v2.14 (8 feat/agent, normalised globals) PLUS
    # reward_overshoot_mode='linear'.
    # -------------------------------------------------------------------------
    def _make_env():
        return IrrigationEnv(
            randomize=True,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,   # *** v2.15 ***
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

    AsymmetricLRSAC.actor_lr_mult = float(actor_lr_mult)
    model = AsymmetricLRSAC(
        policy=V215CTDESACPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=gamma,
        tau=TAU,
        ent_coef=ent_coef,
        learning_starts=LEARNING_STARTS,
        gradient_steps=GRADIENT_STEPS,
        train_freq=TRAIN_FREQ,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # -------------------------------------------------------------------------
    # Callbacks (identical set to v2.14).
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
    print(f"  SAC training - v2.15 (linear r6 reward shape) - seed {seed}")
    print(f"  Architecture: v2.14 (V211 LN critic + LeakyReLU actor + recenter)")
    print(f"  Entropy:      ent_coef={ent_coef} FIXED  (v2.14 = same)")
    print(f"  Reward r6:    {reward_overshoot_mode}  (v2.14 = quadratic)")
    print(f"  ALPHA6_LIN:   1.5  (calibrated from v2.14 rollouts)")
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
            "Train SAC v2.15: v2.14 architecture (LayerNorm critic, LeakyReLU "
            "actor, normalised globals, actor input re-center, asymmetric LR, "
            "alpha=0.01) with the r6 reward changed from quadratic to linear "
            "in overshoot.  Single-variable test of the reward-shape hypothesis."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA)
    parser.add_argument("--actor-lr-mult",   type=float, default=ACTOR_LR_MULT)
    parser.add_argument("--ent-coef",        type=float, default=ENT_COEF,
                        help="Entropy coefficient (default 0.01 = inherited from v2.14).")
    parser.add_argument("--reward-overshoot-mode", type=str,
                        default=REWARD_OVERSHOOT_MODE,
                        choices=['quadratic', 'linear', 'sqrt'],
                        help="r6 shape (default 'linear' = v2.15 anchor).")
    args = parser.parse_args()

    train_sac_v215(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        actor_lr_mult=args.actor_lr_mult,
        ent_coef=args.ent_coef,
        reward_overshoot_mode=args.reward_overshoot_mode,
    )
