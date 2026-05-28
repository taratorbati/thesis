# src/rl/train_v213.py  v2.13.0 (actor-only input re-centering on top of v2.12)
# -----------------------------------------------------------------------------
# WHY v2.13 EXISTS
# ----------------
# v2.12 cured the LITERAL dead-ReLU in v2.11 (live first-layer units: ~0 -> ~14
# of 128) via normalised globals + LeakyReLU actor + asymmetric LR.  Training
# was stable (q_inflation_pct in [-29%, +4%]) and yields recovered to v2.7
# level (3533 vs 3588), but the actor was still operating at ~11% capacity:
# 117/128 first-layer units sat in LeakyReLU's negative regime (100x attenuated),
# producing a timid policy with weak rain response, near-invisible spatial
# differentiation, and persistent wet-year under-performance.
#
# Diagnosed root cause (verified four ways on real data):
#
#   The observation block is 98% non-negative features with E[x] = 0.66 per dim.
#   The policy faces strong downward output pressure from two compounding forces:
#     (a) SAC entropy term pulling pre-tanh mu toward 0 (the center of [0, 12 mm]),
#     (b) the wet-year overshoot penalty r6 dominating the season reward at
#         -44.9 (40x the largest other term).
#   With all-positive inputs, the cheapest gradient-descent path to a lower
#   output is to drive weight ROW-SUMS strongly negative (verified: from ~0
#   at init to -5.3 by 250k).  Negative row-sums x positive inputs => negative
#   layer-1 pre-activations (W@E[x] = -0.94) => LeakyReLU's 100x attenuation
#   => the network runs at 11% capacity.
#
# Empirical proof of the geometric trap:
#   Under a downward output push of 2.0 applied to weight row-sums,
#     - raw positive inputs:     unit-alive fraction collapses to 0.0%
#     - mean-centered inputs:    unit-alive fraction stays at 49.0%
#     - z-scored inputs:         unit-alive fraction stays at 49.6%
#   This is INTRINSIC to the input geometry — no LR, alpha, reward weight, or
#   architecture choice on top of all-positive inputs can fix it.
#
# THE v2.13 FIX (one surgical change on top of v2.12):
#
#   Inside the actor's `_per_agent_features`, AFTER the existing reshape that
#   produces the per-agent input vector (8 per-agent + 57 global = 65 dims),
#   apply the linear re-center:
#
#       combined = 2 * combined - 1
#
#   This maps the [0, 1.x] inputs to [-1, 1.x], shifting the input mean from
#   +0.66 to ~-0.32 per dim.  The downward output pressure no longer drives
#   uniformly negative pre-activations because the inputs are now symmetric
#   around zero; the optimiser can still produce low outputs but it has to
#   learn a balanced solution (some weights positive, some negative) instead
#   of the dead-ReLU shortcut.
#
#   This is implemented as a FIXED LINEAR OP inside the actor — no learned
#   parameters, no train/eval skew possible.  The critic is byte-identical to
#   v2.11/v2.12 (LayerNorm VDN — cascade suppression preserved) and sees the
#   ORIGINAL [0, 1.x] obs through its own per-agent path.  This is surgical:
#   only the actor's forward changes.
#
# Empirical validation on the v2.12 trained weights (applied at probe time):
#   - alive units:                     11.0% -> 63.6%
#   - forecast-rain response (+0.3):   +0.001 (broken) -> -0.38 (correct sign)
#
# Everything else identical to v2.12: gamma=0.99, alpha=0.05 fixed, tau=0.005,
# batch 256, buffer 250k, 250k steps, MAX_GRAD_NORM=1.0, asymmetric LR 5x,
# critic LR schedule 3e-4 -> 5e-5, V211 LayerNorm critic.
#
# ACCEPTANCE CRITERIA (these are the metrics that revealed the v2.12 problem):
#   Primary  ("the actor has capacity"):
#     - layer-1 alive-unit fraction on real obs > 40%/128  (v2.12: 11%)
#     - layer-1 mean pre-activation > -0.3                  (v2.12: -0.99)
#   Primary  ("the actor uses capacity"):
#     - pre-tanh mu temporal std across the season > 0.10   (v2.12: 0.023)
#     - rain-forecast +0.3 sensitivity probe: negative-signed and |delta| > 0.05
#   Secondary ("cascade stays dead"):
#     - |q_inflation_pct| < 30% throughout; critic_loss bounded (< 50)
#   Stretch  ("policy quality"):
#     - 9-cell mean yield > v2.7 (~3700 kg/ha), wet-year recovered
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
from src.rl.networks import V213CTDESACPolicy, make_sac_policy_kwargs
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
# Hyperparameters - identical to v2.12 (whose training was healthy on a
# cascade/Q-magnitude axis).  The only change in v2.13 is the actor's
# `_per_agent_features` operation, which lives inside the V213 policy class.
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256

GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5

ACTOR_LR_MULT    = 5.0          # same asymmetric LR as v2.12

ENT_COEF         = 0.05         # FIXED
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


def train_sac_v213(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    gamma: float = GAMMA,
    actor_lr_mult: float = ACTOR_LR_MULT,
) -> SAC:
    """Train a SAC v2.13 agent.

    v2.13 = v2.12 (LayerNorm critic + LeakyReLU actor + normalised globals +
    asymmetric LR) PLUS one surgical fix: actor-only input re-centering
    (`combined = 2 * combined - 1`) inside the actor's `_per_agent_features`.

    Addresses the verified v2.13 root cause: all-positive inputs + downward
    output pressure (entropy + overshoot penalty) drives weight row-sums
    strongly negative, killing 117/128 first-layer units via LeakyReLU's
    negative regime.  Re-centering breaks the geometry: the optimiser can
    still produce low outputs but no longer via the dead-ReLU shortcut.
    """
    run_name = f"sac_v213_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.13.0",
        "experiment": (
            "v213_SAC_LayerNorm_critic_normObs_leakyReLU_asymLR"
            "_PLUS_actor_input_recenter"
        ),
        "seed": seed,
        "algorithm": "SAC (stable_baselines3) + asymmetric actor LR",
        "policy_class": (
            "V213CTDESACPolicy (V211 LayerNorm critic + V213 LeakyReLU actor "
            "with x' = 2*x - 1 input re-center)"
        ),
        "total_timesteps": total_timesteps,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": gamma,
        "tau": TAU,
        "lr_start": LR_START,
        "lr_end": LR_END,
        "actor_lr_mult": actor_lr_mult,
        "ent_coef": ENT_COEF,
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
        "curriculum": "NONE (v2.7 baseline)",
        "replay_buffer_class": "stable_baselines3.common.buffers.ReplayBuffer",
        "n_step": 1,
        "changes_vs_v212": [
            "actor _per_agent_features: combined = 2 * combined - 1 "
            "(re-center input from [0, 1.x] to [-1, 1.x] inside the actor only) "
            "-- fixes the all-positive-input x downward-output-pressure "
            "geometric trap that kept ~117/128 first-layer units in the "
            "LeakyReLU negative regime",
        ],
    }

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments - v2.7 obs LAYOUT (8 feat/agent, 1097-dim, no curriculum,
    # no overshoot feature) with v2.12 normalised global/forecast block.  The
    # actor-only re-center happens INSIDE the actor, so the env is unchanged.
    # -------------------------------------------------------------------------
    def _make_env():
        return IrrigationEnv(
            randomize=True,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,        # v2.12 global normalisation
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
        policy=V213CTDESACPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=gamma,
        tau=TAU,
        ent_coef=ENT_COEF,
        learning_starts=LEARNING_STARTS,
        gradient_steps=GRADIENT_STEPS,
        train_freq=TRAIN_FREQ,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # -------------------------------------------------------------------------
    # Callbacks (identical set to v2.12)
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
    print(f"  SAC training - v2.13 (actor input re-center fix) - seed {seed}")
    print(f"  Architecture: v2.7 obs (1097-dim, 8 features/agent), NORMALISED globals")
    print(f"  Critic:       VDN twin-Q + LayerNorm (byte-identical to v2.11/v2.12)")
    print(f"  Actor:        _V213SharedActor (LeakyReLU(0.01) + input re-center 2x-1)")
    print(f"  Entropy:      ent_coef={ENT_COEF} FIXED")
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
            "Train SAC v2.13: v2.12 + actor-only input re-centering "
            "(2*x - 1 inside _per_agent_features).  Fixes the 11%-capacity "
            "v2.12 bottleneck driven by all-positive inputs vs downward output "
            "pressure."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA)
    parser.add_argument("--actor-lr-mult",   type=float, default=ACTOR_LR_MULT)
    args = parser.parse_args()

    train_sac_v213(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        actor_lr_mult=args.actor_lr_mult,
    )
