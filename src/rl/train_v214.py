# src/rl/train_v214.py  v2.14.0 (v2.13 architecture trained with α=0.01)
# -----------------------------------------------------------------------------
# WHY v2.14 EXISTS
# ----------------
# v2.13 (the latest stable architecture: LayerNorm critic + LeakyReLU actor +
# normalised globals + actor-only input re-center + asymmetric LR) trained
# cleanly and produced healthy training dynamics, but its actor's pre-tanh mu
# range stayed at ~[-0.20, +0.08] (std 0.04), giving deterministic actions in
# a narrow [5.0, 6.5] mm/day window.  Mean yield 3576 kg/ha — comparable to
# v2.7 PRE-cascade (3396-3524) but well below v2.7 PEAK (3716, captured during
# cascade-induced gradient amplification at step 200k).
#
# Empirical anchor (the data behind α=0.01):
#
#   The v2.7 sweep data settles the diagnosis.  Across v2.7's 10 saved
#   checkpoints (50k..500k) we measured per-agent Q, per-agent entropy term
#   α·H, mu_std, and sweep yields:
#
#       step    critic_loss   Q/α·H ratio   mu_std    sweep_yield
#       50000          1.06          73.7   0.034            3524
#       150000        11.8           75.7   0.032            3477
#       200000      23,554           290    0.312            3716  <- peak
#       250000   ~1.6e8            (broken) 2.55             3243
#
#   The actor only became responsive (mu_std > 0.1) when the cascade pushed
#   per-agent |Q| above ~10, raising the Q/(α·H) signal-to-noise ratio above
#   ~250.  Pre-cascade v2.7 and v2.12/v2.13 all share the same SNR ~75-85 and
#   the same timid behaviour.  The architecture cannot raise Q without
#   reintroducing the cascade (which destroys the policy after the peak), so
#   the only stable path to v2.7-200k SNR is to lower α.
#
#   Target SNR = 290 (v2.7-200k).  v2.13 |Q| ≈ 3 per agent (LayerNorm-bounded).
#   At α=0.05 SNR ≈ 80.  To raise SNR to 290 with Q held constant, lower α by
#   the ratio 290/80 ≈ 3.6.  α = 0.05 / 3.6 ≈ 0.014.  Round to α = 0.01 for a
#   slightly stronger lever (cushion in case the v2.7-200k peak was fragile).
#
# THE v2.14 CHANGE (one variable):
#   ENT_COEF = 0.01  (vs 0.05 in v2.13)
#
#   Everything else identical to v2.13: V211 LayerNorm critic, LeakyReLU actor,
#   actor input re-centre, normalised globals, asymmetric actor LR (5×),
#   γ=0.99, τ=0.005, batch 256, buffer 250k, 250k steps, single-step
#   ReplayBuffer, MAX_GRAD_NORM=1.0.  The policy class V214CTDESACPolicy
#   differs from V213 only in the marker buffer value (2.14 vs 2.13), so the
#   eval runner can identify α-tuned checkpoints.
#
# RISKS (honest):
#   - α=0.01 is below SB3's auto-α "natural" range and below most published
#     SAC settings.  If exploration collapses too quickly the actor may
#     prematurely commit to a sub-optimal mode.  Mitigated by: the actor
#     receives plenty of exploration noise from the SquashedDiagGaussian's
#     learned log_std, which auto-α had been adjusting around log_std ≈ -0.14
#     across all prior runs.  Fixed log_std stays sufficient with α=0.01.
#   - If Q is the wrong SHAPE (flat in action dimension regardless of
#     magnitude), lowering α won't help.  This is testable: if mu_std rises
#     but yields don't improve, α was the lever but Q is uninformative — and
#     we move on to critic-shape fixes (GNN extractor, attention mixing).
#
# ACCEPTANCE CRITERIA:
#   Primary  ("actor became responsive"):
#     - pre-tanh mu temporal std across the season > 0.10 (target ≥ 0.20)
#     - mm/day range spans > 2.5 mm (target [3.0, 8.0]-ish)
#   Secondary  ("training stayed stable"):
#     - |q_inflation_pct| < 30% throughout; critic_loss < 100
#     - log_std doesn't drift to -inf (actor preserves some stochasticity)
#   Stretch   ("policy quality"):
#     - 9-cell mean yield > 3700 kg/ha
#     - wet/100% yield ≥ 3400 (vs v2.13's 3017)
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
from src.rl.networks import V214CTDESACPolicy, make_sac_policy_kwargs
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
# Hyperparameters - identical to v2.13 EXCEPT ENT_COEF.
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256

GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5
ACTOR_LR_MULT    = 5.0           # same asymmetric LR as v2.12/v2.13

ENT_COEF         = 0.01          # *** v2.14 CHANGE: was 0.05 ***
                                 # Derived from v2.7-200k Q/(αH) SNR anchor.

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


def train_sac_v214(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    gamma: float = GAMMA,
    actor_lr_mult: float = ACTOR_LR_MULT,
    ent_coef: float = ENT_COEF,
) -> SAC:
    """Train a SAC v2.14 agent.

    v2.14 = v2.13 architecture (LayerNorm critic, LeakyReLU actor, normalised
    globals, actor-only input re-center, asymmetric LR) trained with α=0.01
    instead of 0.05.  Single-variable test of the entropy/reward SNR hypothesis
    anchored to the v2.7-200k peak-actor snapshot.
    """
    run_name = f"sac_v214_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.14.0",
        "experiment": "v214_v213_arch_PLUS_alpha_0.01",
        "seed": seed,
        "algorithm": "SAC (stable_baselines3) + asymmetric actor LR",
        "policy_class": "V214CTDESACPolicy (V213 actor/critic, marker=2.14)",
        "total_timesteps": total_timesteps,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": gamma,
        "tau": TAU,
        "lr_start": LR_START,
        "lr_end": LR_END,
        "actor_lr_mult": actor_lr_mult,
        "ent_coef": ent_coef,            # *** THE ONE NEW VARIABLE ***
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
        "changes_vs_v213": [
            "ENT_COEF: 0.05 -> 0.01 (single-variable test). "
            "Anchored to v2.7-200k SNR ratio Q/(α·H) ≈ 290 derived from sweep "
            "data; v2.13's stable Q ≈ 3 per agent and the entropy contribution "
            "α·H ≈ 0.039 per agent yields SNR ≈ 80 at α=0.05.  At α=0.01 the "
            "SNR climbs back into the v2.7-200k regime, allowing the actor "
            "gradient to overcome the entropy restoring force."
        ],
    }

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments - same as v2.13 (8 feat/agent, normalised globals).
    # -------------------------------------------------------------------------
    def _make_env():
        return IrrigationEnv(
            randomize=True,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
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
        policy=V214CTDESACPolicy,
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
    # Callbacks (identical set to v2.13).
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
    print(f"  SAC training - v2.14 (α=0.01 SNR recovery) - seed {seed}")
    print(f"  Architecture: v2.13 (V211 LN critic + LeakyReLU actor + recenter)")
    print(f"  Entropy:      ent_coef={ent_coef} FIXED  (v2.13 was 0.05)")
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
            "Train SAC v2.14: v2.13 architecture (LayerNorm critic, LeakyReLU "
            "actor, normalised globals, actor input re-center, asymmetric LR) "
            "with ENT_COEF lowered to 0.01.  Single-variable test of the "
            "entropy/reward SNR hypothesis anchored to v2.7-200k."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA)
    parser.add_argument("--actor-lr-mult",   type=float, default=ACTOR_LR_MULT)
    parser.add_argument("--ent-coef",        type=float, default=ENT_COEF,
                        help="Entropy coefficient (default 0.01 = v2.14 anchor).")
    args = parser.parse_args()

    train_sac_v214(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        actor_lr_mult=args.actor_lr_mult,
        ent_coef=args.ent_coef,
    )
