# src/rl/train_v211.py  v2.11.0 (LayerNorm critic - cascade-prevention experiment)
# -----------------------------------------------------------------------------
# v2.11 experiment: v2.7 architecture + LayerNorm in the critic's hidden layers.
#
# This run tests the hypothesis that the v2.7 deadly-triad cascade
# (Q_pred sign-flip and critic_loss geometric explosion around step 155k-170k
# documented in the Phase 1 study) is driven by neural-network optimization
# dynamics (NTK-driven Self-Excite Eigenvalue growth) and can be suppressed
# by inserting LayerNorm into the critic's hidden MLP layers WITHOUT changing
# any other component of the v2.7 baseline.
#
# Why this experiment exists - context vs the v2.10 study
# --------------------------------------------------------
# Phase 1 (v2.10 E2/E3/E4) ruled out:
#   E2: TQC k=5 quantile truncation - structurally inert under VDN-sum
#       because summing per-agent quantile distributions narrows the spread
#       (central-limit-theorem effect), so the top quantiles to drop are
#       arbitrarily close to the median.
#   E3: TQC + custom n-step buffer - the buffer dropped the soft-Bellman
#       entropy bonus on intermediate steps (~8 units/transition systematic
#       under-bias), driving a negative Q-cascade much earlier than v2.7.
#   E4: SAC + gamma=0.98 - prevented the cascade (q_inflation_pct in
#       [-16%, -5%] through 250k steps) but produced a 6-mm/day flat policy
#       (mean -256 kg/ha vs v2.7; -16% wet-year regression).  The effective
#       horizon 1/(1-gamma)=50 is shorter than the 93-day task horizon, so
#       credit assignment across the season is broken.
#
# Phase 1.5 (this run) tests a fundamentally different cascade hypothesis:
#   Yue et al. NeurIPS 2023 ("Understanding, Predicting and Better Resolving
#   Q-Value Divergence in Offline-RL", arXiv:2310.04411) prove via Neural
#   Tangent Kernel analysis that Q-divergence in deep RL can arise from
#   neural-network optimization dynamics (the Self-Excite Eigenvalue Measure,
#   SEEM, crossing a critical threshold).  When SEEM > 0, the network's own
#   gradient updates amplify Q errors polynomially in training step.
#   They show empirically that inserting LayerNorm after each hidden Linear
#   layer in the critic reliably keeps SEEM below the divergence threshold
#   with no detrimental bias on the learned policy.
#
#   Nauman et al. RLC 2024 ("Dissecting Deep RL with High Update Ratios:
#   Combatting Value Overestimation and Divergence", arXiv:2403.05996)
#   confirm the same effect in *online* RL on the dm_control suite using
#   a related unit-ball normalization.
#
# Reading of the v2.7 cascade through this lens
# ----------------------------------------------
# v2.7 seed 0 critic_loss trajectory (from output_v27_seed0.log):
#   step 100k: 1.07     step 200k: 23,600
#   step 150k: 11.8     step 250k: 157,000
#   step 160k: 98.5     step 300k: 12,300,000,000
#   step 170k: 114
# This is roughly factor-of-10 growth per 10k steps - clean exponential,
# characteristic of NTK-self-excitation rather than purely a Bellman fixed-
# point bifurcation (which would saturate or oscillate, not grow geometrically
# across 12 orders of magnitude).  v2.7 seed 1 cascades identically at the
# same step.  This signature is what LayerNorm is designed to suppress.
#
# Single change vs v2.7 (locked - no other hyperparameters touched)
# ------------------------------------------------------------------
#   v2.7 critic: Linear(66, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 1)
#   v2.11 critic: Linear(66, 256) -> LayerNorm(256) -> ReLU
#                 -> Linear(256, 256) -> LayerNorm(256) -> ReLU -> Linear(256, 1)
#
# The actor and observation layout are byte-identical to v2.7
# (V211CTDESACPolicy.make_actor returns the same _V27SharedActor instance).
# Reward, environment, gamma=0.99, ent_coef=0.05 fixed, all hyperparameters
# - identical to v2.7.
#
# Acceptance criteria
# -------------------
# Primary (cascade suppressed):
#   - |q_inflation_pct| < 30% throughout 250k steps (v2.7 hits +200% at step 200k)
#   - critic_loss never exceeds 50 in a rolling-1k-step mean past step 100k
#   - actor/std/spatial stays in [0.20, 0.40] throughout (v2.7 drops to <0.10
#     post-cascade)
#
# Secondary (policy quality preserved or improved):
#   - 9-cell perfect-forecast yields within +-3% of v2.7 best_model (seed 0)
#   - Specifically: dry/100% yield >= 4040 (97% of v2.7's 4163)
#                   wet/100% yield >= 3330 (97% of v2.7's 3434)
#
# Stretch (improvement over v2.7):
#   - Late-training (post-step-200k) yields exceed step-200k baseline by
#     >50 kg/ha mean across the 9 cells, indicating the architecture can
#     benefit from training beyond the v2.7 cascade window
#
# Early-kill rule
# ---------------
# If at any checkpoint past step 150k BOTH:
#   q_inflation_pct > 100% AND actor/std/spatial < 0.15
# then LayerNorm has not suppressed the cascade.  Stop the run, save the
# pre-cascade checkpoint as best_model, and the result is "LayerNorm
# insufficient on this problem - cascade hypothesis is bifurcation-driven,
# not NTK-driven."  Next attempt would be combining LayerNorm with a
# secondary cascade brake (gamma reduction or entropy-normalized actor loss).
#
# References
# ----------
# Yue, Kang, Shi, Ma, Liu, Zhao 2023: "Understanding, Predicting and Better
#     Resolving Q-Value Divergence in Offline-RL", NeurIPS 2023.
#     https://arxiv.org/abs/2310.04411
# Nauman, Bortkiewicz, Milos, Trzcinski, Ostaszewski, Cygan 2024: "Dissecting
#     Deep RL with High Update Ratios", RLC 2024.
#     https://arxiv.org/abs/2403.05996
# Ba, Kiros, Hinton 2016: "Layer Normalization", arXiv:1607.06450.
#     Original LayerNorm paper.
# Haarnoja et al. 2018: "Soft Actor-Critic", ICML 2018.  Baseline algorithm.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.gym_env import IrrigationEnv
from src.rl.networks import V211CTDESACPolicy, make_sac_policy_kwargs
from src.rl.callbacks_v210 import (
    BiasRatioCallback,
    ActionStatsCallback,
    OptimizerLRCallback,
)
# Re-use the v2.7 infrastructure helpers and rotating-buffer callback.
from src.rl.train import (
    RotatingReplayBufferCheckpoint,
    GradClipCallback,
    _make_lr_schedule,
    _init_wandb,
)


# -----------------------------------------------------------------------------
# Hyperparameters - identical to v2.7 baseline.
# The single architectural change is V211CTDESACPolicy (LayerNorm critic).
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000     # v2.7 baseline
BUFFER_SIZE      = 250_000     # v2.7 baseline
BATCH_SIZE       = 256         # v2.7 baseline

GAMMA            = 0.99        # v2.7 baseline - UNCHANGED.
                               # v2.11 keeps the 100-step credit-assignment
                               # range that gamma=0.99 affords.  E4 proved
                               # that gamma=0.98 stops the cascade but kills
                               # the policy via myopia.

TAU              = 0.005       # v2.7 baseline; Haarnoja et al. 2018 default
LR_START         = 3e-4        # v2.7 baseline
LR_END           = 5e-5        # v2.7 baseline; linear decay

ENT_COEF         = 0.05        # FIXED.  Auto-tuning is the v2.5 failure
                               # mode (entropy collapse, see HANDOFF_v4
                               # Section 4.3).
MAX_GRAD_NORM    = 1.0         # v2.7 baseline
LEARNING_STARTS  = 1_000       # v2.7 baseline
GRADIENT_STEPS   = 1           # v2.7 baseline
TRAIN_FREQ       = 1           # one gradient step per env step

EVAL_FREQ        = 25_000      # v2.7 baseline
N_EVAL_EPISODES  = 9           # v2.7 baseline
CHECKPOINT_FREQ  = 25_000      # match v2.10 cadence (was 50_000 in v2.7;
                               # finer cadence helps locate cascade onset
                               # if it still happens, and is cheap)

ACTOR_HIDDEN  = [128, 128]     # v2.7 baseline
CRITIC_HIDDEN = [256, 256]     # v2.7 baseline - LayerNorm is inserted between

# Diagnostics frequencies
BIAS_RATIO_FREQ          = 25_000
BIAS_RATIO_N_EPISODES    = 3
ACTION_STATS_FREQ        = 1_000
LR_LOG_FREQ              = 1_000


def train_sac_v211(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    gamma: float = GAMMA,
) -> SAC:
    """Train a SAC v2.11 agent (v2.7 architecture + LayerNorm critic).

    Parameters
    ----------
    seed : int
        Random seed.  Use seed 0 first for paired comparison against v2.7
        seed 0 published numbers; only expand to seeds 1, 2 after seed-0
        results meet the acceptance criterion.
    output_dir : str
        Root directory for checkpoints and best-model artefacts.  Results go
        to <output_dir>/sac_v211_seed{seed}/.
    wandb_project : str or None
        WandB project name.  None disables WandB logging entirely.
    total_timesteps : int
        Default 250_000 (matches v2.7 / v2.10).
    gamma : float
        Discount factor.  Defaults to GAMMA constant above (0.99 - v2.7 value).
        Override only for follow-up experiments (e.g. combining LayerNorm
        with gamma=0.985).  The v2.11 experiment proper is gamma=0.99.

    Returns
    -------
    SAC
        The trained model.  Also saved to disk as
        <output_dir>/sac_v211_seed{seed}/sac_v211_seed{seed}_final.zip.

    Notes
    -----
    The returned model is loadable via the existing v2.7 SAC eval script:
        python -m scripts.experiments.exp_rl --mode eval \\
            --model <best_model.zip> --scenario all --budget all --forecast perfect

    The runner's architecture auto-detection (runner.py:_detect_critic_arch)
    will recognise this checkpoint as a v2.11-format LayerNorm VDN SAC (by
    the presence of a 1-D 'critic.qf0.1.weight' LayerNorm gamma key) and
    load it via the V211CTDESACPolicy path.  The observation layout is
    identical to v2.7 (1097-dim, 8 features/agent), so the runner's
    observation builder needs no v2.11-specific code.
    """
    run_name = f"sac_v211_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Config dict (logged to WandB for provenance).
    # -------------------------------------------------------------------------
    config = {
        "version": "2.11.0",
        "experiment": "v211_SAC_LayerNorm_critic",
        "seed": seed,
        "algorithm": "SAC (stable_baselines3)",
        "policy_class": "V211CTDESACPolicy (VDN twin-Q + LayerNorm critic)",
        "total_timesteps": total_timesteps,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": gamma,
        "tau": TAU,
        "lr_start": LR_START,
        "lr_end": LR_END,
        "ent_coef": ENT_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "actor_hidden": ACTOR_HIDDEN,
        "critic_hidden": CRITIC_HIDDEN,
        "critic_layernorm": True,
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
        "changes_vs_v27": [
            "critic: LayerNorm after each hidden Linear (Yue NeurIPS 2023 "
            "placement) - the single experimental variable",
            f"checkpoint_freq {CHECKPOINT_FREQ} (was 50_000 in v2.7)",
        ],
    }

    # -------------------------------------------------------------------------
    # WandB
    # -------------------------------------------------------------------------
    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments - v2.7 layout exactly (no curriculum, no overshoot feat).
    # -------------------------------------------------------------------------
    train_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,
        use_overshoot_feature=False,
    )])
    eval_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,
        use_overshoot_feature=False,
    )])
    bias_eval_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,
        use_overshoot_feature=False,
    )])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)
    bias_eval_env.seed(seed + 2000)

    # -------------------------------------------------------------------------
    # Policy kwargs - shared with v2.7 (architecture differs at make_critic()).
    # -------------------------------------------------------------------------
    policy_kwargs = make_sac_policy_kwargs(
        N=130,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
    )

    lr_schedule = _make_lr_schedule(LR_START, LR_END)

    # -------------------------------------------------------------------------
    # SAC model.
    #
    # CRITICAL: ent_coef=ENT_COEF (0.05 fixed) - NOT 'auto'.  SAC's default
    # is 'auto' which is the v2.5 failure mode (entropy collapse).
    #
    # policy=V211CTDESACPolicy: the only architectural change vs v2.7.
    # This policy class uses _V27SharedActor (8-feature actor, identical to
    # v2.7) and _V211FactorizedContinuousCritic (LayerNorm critic).
    #
    # No replay_buffer_class override: this uses SB3's standard 1-step
    # ReplayBuffer.  The custom NStepReplayBuffer (src/rl/nstep_buffer.py)
    # was the source of the E3 cascade; it is intentionally NOT used here.
    # -------------------------------------------------------------------------
    model = SAC(
        policy=V211CTDESACPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=gamma,
        tau=TAU,
        ent_coef=ENT_COEF,                                # FIXED (not 'auto')
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

    # v2.10 diagnostics - the BiasRatioCallback is the primary cascade detector.
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

    # -------------------------------------------------------------------------
    # Banner
    # -------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  SAC training - v2.11 (LayerNorm critic) - seed {seed}")
    print(f"  Architecture: v2.7 obs (1097-dim, 8 features/agent)")
    print(f"  Algorithm:    SAC (stable_baselines3) + standard 1-step ReplayBuffer")
    print(f"  Critic:       VDN twin-Q with LayerNorm after each hidden Linear")
    print(f"  Actor:        _V27SharedActor (unchanged from v2.7)")
    print(f"  Entropy:      ent_coef={ENT_COEF} FIXED (auto-tuning disabled)")
    print(f"  GAMMA:        {gamma}  (v2.7 baseline; unchanged)")
    print(f"  LR schedule:  {LR_START:.0e} -> {LR_END:.0e} (linear)")
    print(f"  Total steps:  {total_timesteps:,}")
    print(f"  Checkpoint:   every {CHECKPOINT_FREQ:,} steps")
    print(f"  Output:       {save_dir}")
    print(f"{'='*72}\n")

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
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
            "Train SAC v2.11 (v2.7 architecture with LayerNorm in the critic - "
            "Yue et al. NeurIPS 2023 cascade-prevention placement)."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA,
                        help=(
                            "Discount factor.  Default 0.99 (v2.7 baseline). "
                            "v2.11 keeps gamma=0.99; override only for "
                            "follow-up experiments combining LayerNorm with "
                            "a different gamma."
                        ))
    args = parser.parse_args()

    train_sac_v211(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
    )
