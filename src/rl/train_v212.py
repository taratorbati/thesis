# src/rl/train_v212.py  v2.12.0 (dead-ReLU fix: normalised obs + LeakyReLU actor
#                                 + asymmetric LR, on top of the v2.11 LayerNorm critic)
# -----------------------------------------------------------------------------
# WHY v2.12 EXISTS
# ----------------
# v2.11 (LayerNorm critic, gamma=0.99, alpha=0.05) successfully suppressed the
# v2.7 deadly-triad cascade (critic_loss stayed in [0.02, 0.36] across 250k
# steps) but produced a degenerate, completely uniform 5.97 mm/day policy.
#
# Direct instrumentation of the trained v2.11 checkpoints (loading policy.pth
# and running real observations through the actor) established the proximate
# cause beyond reasonable doubt:
#
#   THE v2.11 ACTOR IS DEAD-ReLU COLLAPSED AT ITS FIRST HIDDEN LAYER.
#   Across a full 93-day season of real observations, ~0.1 of 128 first-layer
#   units fire (vs ~21/128 for v2.7).  Every ReLU output is ~0, so the actor's
#   mu head sees only its bias and emits a constant pre-tanh value, which maps
#   through tanh -> [0,1] -> *UB_MM to a flat ~5.9 mm/day regardless of state.
#
# ROOT CAUSE of the dead ReLU (verified by per-feature decomposition):
#   The global scalar block and the 48-dim forecast block were fed to the actor
#   with RAW physical magnitudes -- rainfall up to ~11 (in-season) / 64 (record),
#   Kc_ET up to ~7, radiation up to ~32.  The per-agent dynamic block was already
#   normalised to [0,1.5], but the raw global block dominated the first-layer
#   pre-activations: the radiation sub-block alone contributed about -3.2 (and the
#   whole global block about -4.6) to the mean pre-activation.  With the actor's
#   post-training weight spread narrowed (small reward gradient under the LayerNorm-
#   bounded critic), this slid the ENTIRE pre-activation distribution below zero
#   and the ReLUs died.  v2.7 survived the same raw inputs only because its
#   un-bounded (pre-cascade) critic produced large gradients that grew the actor's
#   weights (max_abs ~2.0 vs ~0.3 in v2.11), keeping ~20 units alive.
#
# A SECOND, INDEPENDENT MECHANISM (verified via the E4 gamma=0.98 checkpoint):
#   E4 has ~16/128 live ReLUs yet its pre-tanh output still barely modulates
#   (temporal std 0.017 vs v2.7's 0.285).  So even with live units, a stable
#   critic gives the actor a weak learning signal.  This is a gradient-strength
#   problem, addressed here with an asymmetric (higher) actor learning rate --
#   NOT by lowering gamma (that re-introduces 50-day myopia on the 93-day task)
#   and NOT by lowering alpha in the same run (risks the v2.5 entropy-collapse).
#
# THE v2.12 FIX (three coordinated changes; critic byte-identical to v2.11):
#   1. Observation normalisation (gym_env.py + runner.py): rainfall/RAIN_REF,
#      Kc_ET/ETC_REF, radiation/RAD_REF in BOTH the scalar and forecast blocks.
#      This is the root-cause fix.  normalize_globals=True in the env.
#   2. Actor activation LeakyReLU(0.01) (networks._V212SharedActor): insurance so
#      a unit that drifts negative still passes gradient and can recover.
#   3. Asymmetric learning rate: actor LR = ACTOR_LR_MULT x critic LR, applied by
#      overriding SAC._update_learning_rate.  Counters the LayerNorm-bounded
#      critic gradient that left E4's actor under-driven.
#
#   Everything else is identical to v2.11: gamma=0.99, alpha=0.05 fixed,
#   tau=0.005, batch 256, buffer 250k, 250k steps, standard 1-step ReplayBuffer,
#   MAX_GRAD_NORM=1.0, LayerNorm VDN critic.
#
# ACCEPTANCE CRITERIA (primary -- "the actor woke up"):
#   - actor first-layer live-unit count > 30/128 on real obs by step 100k
#   - action_std_spatial in [0.20, 0.40] and NOT collapsing to ~0
#   - pre-tanh mu temporal std across the season >> 0.05 (state-responsive)
#   - mu.weight std grows past ~0.10 (vs v2.11's shrink to ~0.027)
# Secondary ("cascade stays dead"):
#   - |q_inflation_pct| < 30% throughout; critic_loss bounded (< 50)
# Stretch ("policy quality"):
#   - 9-cell mean yield >= v2.7 best_model (>= ~3700 kg/ha), wet/100% recovered
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn

from src.rl.gym_env import IrrigationEnv
from src.rl.networks import V212CTDESACPolicy, make_sac_policy_kwargs
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


# -----------------------------------------------------------------------------
# Hyperparameters -- identical to v2.11 EXCEPT the asymmetric actor LR.
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256

GAMMA            = 0.99        # keep the 100-step credit horizon (E4 proved
                               # gamma<0.99 breaks the 93-day task)
TAU              = 0.005
LR_START         = 3e-4        # critic LR start (v2.7/v2.11 baseline)
LR_END           = 5e-5        # critic LR end (linear decay)

# Asymmetric actor learning rate.  The actor optimiser LR = ACTOR_LR_MULT x the
# scheduled (critic) LR at every train() call.  Counters the LayerNorm-bounded
# critic gradient that under-drives the actor.  5x is a conservative starting
# point well inside stable SAC practice; raise toward 8-10x only if the actor
# still under-modulates and training stays stable.
ACTOR_LR_MULT    = 5.0

ENT_COEF         = 0.05        # FIXED (auto-tuning is the v2.5 collapse mode)
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


# -----------------------------------------------------------------------------
# SAC subclass: asymmetric actor/critic learning rate.
#
# SB3's SAC.train() calls self._update_learning_rate([actor.opt, critic.opt,
# (ent_coef.opt)]) once per train() using the single lr schedule.  We let that
# run normally (so the critic + ent_coef follow the schedule and SB3's logging
# stays correct), then multiply ONLY the actor optimiser's LR by ACTOR_LR_MULT.
# -----------------------------------------------------------------------------
class AsymmetricLRSAC(SAC):
    actor_lr_mult: float = ACTOR_LR_MULT

    def _update_learning_rate(self, optimizers) -> None:  # type: ignore[override]
        super()._update_learning_rate(optimizers)
        # After the base class set every optimiser to the scheduled LR, bump the
        # actor's LR.  self.actor.optimizer is the actor's Adam instance.
        for pg in self.actor.optimizer.param_groups:
            pg["lr"] = pg["lr"] * self.actor_lr_mult


def train_sac_v212(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    gamma: float = GAMMA,
    actor_lr_mult: float = ACTOR_LR_MULT,
) -> SAC:
    """Train a SAC v2.12 agent.

    v2.12 = v2.11 LayerNorm VDN critic + normalised global/forecast observations
    + LeakyReLU actor + asymmetric (higher) actor learning rate.  Fixes the
    v2.11 dead-ReLU collapse so the actor becomes state-responsive while the
    cascade stays suppressed.
    """
    run_name = f"sac_v212_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.12.0",
        "experiment": "v212_SAC_LayerNorm_critic_normObs_leakyReLU_asymLR",
        "seed": seed,
        "algorithm": "SAC (stable_baselines3) + asymmetric actor LR",
        "policy_class": "V212CTDESACPolicy (VDN twin-Q + LayerNorm critic, LeakyReLU actor)",
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
        "changes_vs_v211": [
            "obs: normalise global+forecast block (rainfall/70, Kc_ET/8, radiation/35)"
            " -- root-cause fix for the dead-ReLU collapse",
            "actor: LeakyReLU(0.01) instead of ReLU -- dead-unit insurance",
            f"actor LR = {actor_lr_mult}x critic LR -- counters LayerNorm-bounded"
            " critic gradient",
        ],
    }

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments -- v2.7 obs LAYOUT (8 feat/agent, 1097-dim, no curriculum,
    # no overshoot feature) but with v2.12 normalised global/forecast block.
    # -------------------------------------------------------------------------
    def _make_env():
        return IrrigationEnv(
            randomize=True,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,        # <-- v2.12 root-cause fix
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

    # -------------------------------------------------------------------------
    # SAC model (asymmetric-LR subclass).  ent_coef fixed; V212 policy.
    # -------------------------------------------------------------------------
    AsymmetricLRSAC.actor_lr_mult = float(actor_lr_mult)
    model = AsymmetricLRSAC(
        policy=V212CTDESACPolicy,
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
    # Callbacks (identical set to v2.11)
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
    print(f"  SAC training - v2.12 (dead-ReLU fix) - seed {seed}")
    print(f"  Architecture: v2.7 obs (1097-dim, 8 features/agent), NORMALISED globals")
    print(f"  Critic:       VDN twin-Q + LayerNorm (byte-identical to v2.11)")
    print(f"  Actor:        _V212SharedActor (LeakyReLU(0.01))")
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
            "Train SAC v2.12: v2.11 LayerNorm critic + normalised global/forecast "
            "observations + LeakyReLU actor + asymmetric actor LR.  Fixes the "
            "v2.11 dead-ReLU collapse."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA)
    parser.add_argument("--actor-lr-mult",   type=float, default=ACTOR_LR_MULT,
                        help="Actor LR as a multiple of the (scheduled) critic LR. "
                             "Default 5.0.")
    args = parser.parse_args()

    train_sac_v212(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        actor_lr_mult=args.actor_lr_mult,
    )
