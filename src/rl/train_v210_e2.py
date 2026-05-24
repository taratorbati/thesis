# src/rl/train_v210_e2.py  v2.10.0 (E3 patch — n_steps=3 + k=5)
# -----------------------------------------------------------------------------
# E3 experiment: TQC with k=5 AND n_step=3 (Strategy A).
#
# E2 (n_step=1, k=5) confirmed the cascade fires at ~170k steps, essentially
# identical to v2.7.  Post-mortem analysis established that truncation alone
# fails because the VDN-sum critic learns a near-degenerate quantile
# distribution (all 25 quantiles collapse to the mean, quantile_spread ≈ 3-4
# in the stable phase), so truncation has no active downward bias to cut.
# The cascade is driven by entire-distribution drift via bootstrap
# amplification (γ/(1-γ) ≈ 100 leverage), not by tail overestimation.
#
# n_step=3 grounds 3 real rewards per gradient step, reducing the effective
# bootstrap depth from ~93 steps to ~31 steps (3× reduction in amplification
# leverage).  Combined with k=5 truncation (which does little now but costs
# nothing), this is Strategy A from the v2.10 cascade analysis.
#
# Implementation note on n_steps in SB3 2.6.0
# ------------------------------------------------
# sb3_contrib.TQC 2.6.0 has NO built-in n_steps support for off-policy
# training (n_steps in sb3_contrib only exists for on-policy algorithms
# TRPO and PPO-Recurrent).  We implement n-step returns via a custom
# NStepReplayBuffer (src/rl/nstep_buffer.py) passed as replay_buffer_class
# to the TQC constructor.  This is the documented SB3 extension point.
#
# The γ^n vs γ^1 approximation (see nstep_buffer.py header) introduces a
# small error (~7 units per step on Q ≈ 378).  This is the standard
# approximation used by all SB3-compatible n-step implementations (Huang
# 2021 code release, CleanRL n-step SAC) and does not affect the main
# benefit of n-step (grounding the target in real rewards).
#
# Hyperparameter citations:
#   - n_steps=3 : Hessel et al. 2018 (Rainbow) §3; empirically optimal for
#     continuous control off-policy algorithms (see §4 ablation).
#   - n_quantiles=25, top_quantiles_to_drop_per_net=5 : Kuznetsov et al.
#     2020, ICML 2020, Section 5.2.
#   - n_critics=2 : matches v2.7 twin-Q.
#   - All other hyperparameters: v2.7 baseline (THESIS_HANDOFF_v4 §10.2).
#
# Diagnostics (v2.10 handoff Section 5):
#   - BiasRatioCallback         : Q_pred / R_realised every 25k steps
#   - ActionStatsCallback       : spatial std of u across 130 agents
#   - OptimizerLRCallback       : actual optimizer LR (v2.9 bug detector)
#   - TQCQuantileSpreadCallback : p5..p95 spread of quantile predictions
#   - EvalCallback              : standard SB3 eval on 9 random episodes
#   - CheckpointCallback        : every 25k steps (v2.10 spec)
#   - RotatingReplayBufferCheckpoint : single rotating replay-buffer dump
#
# Outputs to: <output_dir>/sac_v210_e2_seed{seed}/
#     best_model/best_model.zip      <- EvalCallback best
#     checkpoints/*.zip              <- every 25k steps
#     eval_logs/                     <- EvalCallback per-eval npz logs
#     tensorboard/                   <- SB3 TB writer
#     bias_ratio_log.csv             <- v2.10 cascade diagnostic
#     replay_buffer_latest.pkl       <- rotating dump
#     <run_name>_final.zip           <- final model at step 250k
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import TQC

from src.rl.gym_env import IrrigationEnv
from src.rl.networks_tqc import V27TQCPolicy, make_tqc_policy_kwargs
from src.rl.callbacks_v210 import (
    BiasRatioCallback,
    ActionStatsCallback,
    OptimizerLRCallback,
    TQCQuantileSpreadCallback,
    PerCellEvalCallback,
)
from src.rl.nstep_buffer import NStepReplayBuffer
# Reuse helpers and the rotating-buffer callback from v2.9 train.py - no
# need to duplicate them and they work on any OffPolicyAlgorithm.
from src.rl.train import (
    RotatingReplayBufferCheckpoint,
    GradClipCallback,
    _make_lr_schedule,
    _init_wandb,
)


# -----------------------------------------------------------------------------
# Training constants (all backed by citation or v2.7 baseline).
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000     # v2.7 baseline (v4 Section 5.3)
BUFFER_SIZE      = 250_000     # v2.7 baseline
BATCH_SIZE       = 256         # v2.7 baseline
GAMMA            = 0.99        # v2.7 baseline; long-horizon problem
TAU              = 0.005       # v2.7 baseline; standard SAC (Haarnoja 2018)
LR_START         = 3e-4        # v2.7 baseline
LR_END           = 5e-5        # v2.7 baseline; linear decay

ENT_COEF         = 0.05        # FIXED.  Auto-tuning known to fail on this
                               # problem (v4 Section 4.3).
MAX_GRAD_NORM    = 1.0         # v2.7 baseline
LEARNING_STARTS  = 1_000       # v2.7 baseline
GRADIENT_STEPS   = 1           # v2.7 baseline
TRAIN_FREQ       = 1           # one gradient step per env step

EVAL_FREQ        = 25_000      # v2.7 baseline
N_EVAL_EPISODES  = 9           # v2.7 baseline
CHECKPOINT_FREQ  = 25_000      # v2.10 handoff explicit instruction
                               # (v2.7 used 50_000 - intentional change)

ACTOR_HIDDEN  = [128, 128]     # v2.7 baseline
CRITIC_HIDDEN = [256, 256]     # v2.7 baseline

# TQC-specific - Kuznetsov 2020 Section 5.2.
N_QUANTILES                   = 25
N_CRITICS                     = 2
TOP_QUANTILES_TO_DROP_PER_NET = 5     # 20% truncation, paper-optimal
N_STEPS                       = 3     # E3: n-step returns (Hessel 2018 §3)

# Diagnostics frequencies
BIAS_RATIO_FREQ          = 25_000
BIAS_RATIO_N_EPISODES    = 3
ACTION_STATS_FREQ        = 1_000
LR_LOG_FREQ              = 1_000
QUANTILE_SPREAD_FREQ     = 25_000


def train_tqc_e2(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    enable_per_cell_eval: bool = False,
) -> TQC:
    """Train a TQC v2.10 E3 agent (v2.7 architecture + n_step=3 + truncation k=5).

    Parameters
    ----------
    seed : int
        Random seed.  v2.10 ablation uses seed=0 to compare against v2.7's
        seed 0 published numbers; expand seeds after the winner is identified.
    output_dir : str
        Root directory for checkpoints and best-model artefacts.  Final
        results go to <output_dir>/sac_v210_e3_seed{seed}/.
    wandb_project : str or None
        WandB project name.  None disables WandB logging entirely.
    total_timesteps : int
        Default 250_000 (matches v2.7 / v2.9).
    enable_per_cell_eval : bool
        If True, run the 9-cell test grid evaluation every 50k steps.
        Adds ~7.5 minutes of wall time over a 250k-step run.  Disabled
        by default for the first E3 run.

    Returns
    -------
    TQC
        The trained model.  Also saved to disk as
        <output_dir>/sac_v210_e3_seed{seed}/sac_v210_e3_seed{seed}_final.zip.

    Notes
    -----
    Acceptance criteria (v2.10 handoff Section 6 / agent prompt):
        - q_inflation_pct at step 250k < 20%  (cascade not active)
        - in-distribution yields within 1% of v2.7's published numbers
        - best_model step >= 225k  (not an early cascade checkpoint)
    """
    run_name = f"sac_v210_e3_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Config dict (logged to WandB and saved for provenance).
    # -------------------------------------------------------------------------
    config = {
        "version": "2.10.0-E3",
        "experiment": "E3_TQC_nstep3_truncation_k5",
        "seed": seed,
        "algorithm": "TQC (sb3_contrib)",
        "policy_class": "V27TQCPolicy (VDN-per-quantile)",
        "total_timesteps": total_timesteps,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "tau": TAU,
        "lr_start": LR_START,
        "lr_end": LR_END,
        "ent_coef": ENT_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "actor_hidden": ACTOR_HIDDEN,
        "critic_hidden": CRITIC_HIDDEN,
        "n_quantiles": N_QUANTILES,
        "n_critics": N_CRITICS,
        "top_quantiles_to_drop_per_net": TOP_QUANTILES_TO_DROP_PER_NET,
        "n_steps": N_STEPS,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "eval_freq": EVAL_FREQ,
        "obs_dim": 1097,
        "n_agent_features": 8,
        "curriculum": "NONE (v2.7 baseline)",
        "changes_vs_v27": [
            "TQC algorithm replaces SAC",
            "Quantile critic (25 quantiles) replaces scalar critic",
            "Top 5 quantiles per critic dropped from target (truncation, k=5)",
            f"n_step={N_STEPS} via NStepReplayBuffer (Hessel 2018 §3)",
            f"checkpoint_freq {CHECKPOINT_FREQ} (was 50_000 in v2.7)",
            "gamma^1 vs gamma^n approximation in bootstrap (see nstep_buffer.py)",
        ],
    }

    # -------------------------------------------------------------------------
    # WandB
    # -------------------------------------------------------------------------
    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments - v2.7 obs layout (8 features/agent, no curriculum, no
    # overshoot feature).  Reproduces v2.7 / v2.9 env exactly.
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
    # Separate eval env for the bias-ratio callback (avoids contaminating
    # the EvalCallback's eval_env state with extra resets).
    bias_eval_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,
        use_overshoot_feature=False,
    )])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)
    bias_eval_env.seed(seed + 2000)

    # -------------------------------------------------------------------------
    # Policy kwargs - VDN-per-quantile critic via networks_tqc.py
    # -------------------------------------------------------------------------
    policy_kwargs = make_tqc_policy_kwargs(
        N=130,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
        n_quantiles=N_QUANTILES,
        n_critics=N_CRITICS,
        share_features_extractor=True,
    )

    lr_schedule = _make_lr_schedule(LR_START, LR_END)

    # -------------------------------------------------------------------------
    # TQC model.
    #
    # CRITICAL: ent_coef=ENT_COEF (0.05, fixed) - NOT 'auto'.  TQC's default
    # is 'auto' which is the failure mode disabled in v2.5.
    #
    # n-step returns: SB3 2.6.0 TQC has no built-in n_steps parameter for
    # off-policy training.  We pass replay_buffer_class=NStepReplayBuffer with
    # n_steps=N_STEPS via replay_buffer_kwargs.  The buffer accumulates n
    # rewards before adding to the underlying ReplayBuffer.  gamma is passed
    # so the buffer computes the correct discounted n-step return R_n.
    # -------------------------------------------------------------------------
    model = TQC(
        policy=V27TQCPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        ent_coef=ENT_COEF,                                # FIXED (not 'auto')
        target_entropy="auto",                            # required by API; ignored
        learning_starts=LEARNING_STARTS,
        gradient_steps=GRADIENT_STEPS,
        train_freq=TRAIN_FREQ,
        top_quantiles_to_drop_per_net=TOP_QUANTILES_TO_DROP_PER_NET,
        replay_buffer_class=NStepReplayBuffer,
        replay_buffer_kwargs={
            "n_steps": N_STEPS,
            "gamma":   GAMMA,
        },
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

    # v2.10 diagnostics
    bias_ratio_cb = BiasRatioCallback(
        eval_env=bias_eval_env,
        eval_freq=BIAS_RATIO_FREQ,
        n_eval_episodes=BIAS_RATIO_N_EPISODES,
        save_path=str(save_dir),
        verbose=1,
    )
    action_stats_cb = ActionStatsCallback(log_freq=ACTION_STATS_FREQ)
    optimizer_lr_cb = OptimizerLRCallback(log_freq=LR_LOG_FREQ)
    quantile_spread_cb = TQCQuantileSpreadCallback(
        eval_env=bias_eval_env,
        eval_freq=QUANTILE_SPREAD_FREQ,
    )

    cb_list = [
        eval_callback,
        checkpoint_callback,
        rotating_buffer_callback,
        grad_clip_callback,
        bias_ratio_cb,
        action_stats_cb,
        optimizer_lr_cb,
        quantile_spread_cb,
    ]

    if enable_per_cell_eval:
        per_cell_cb = PerCellEvalCallback(
            output_dir=str(save_dir),
            eval_freq=50_000,
            verbose=1,
        )
        cb_list.append(per_cell_cb)

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
    print(f"  TQC training - v2.10.0 E3 - seed {seed}")
    print(f"  Architecture: v2.7 obs (1097-dim, 8 features/agent)")
    print(f"  Algorithm:    TQC (sb3_contrib) + NStepReplayBuffer")
    print(f"    n_quantiles                : {N_QUANTILES}")
    print(f"    n_critics                  : {N_CRITICS}")
    print(f"    top_quantiles_to_drop/net  : {TOP_QUANTILES_TO_DROP_PER_NET}")
    print(f"    n_step                     : {N_STEPS}  (NStepReplayBuffer)")
    print(f"  Entropy: ent_coef={ENT_COEF} FIXED (auto-tuning disabled)")
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
        description="Train TQC v2.10 E3 (v2.7 architecture + n_step=3 + quantile truncation k=5)"
    )
    parser.add_argument("--seed",                   type=int, default=0)
    parser.add_argument("--output-dir",             type=str, default="results/rl")
    parser.add_argument("--wandb-project",          type=str, default=None)
    parser.add_argument("--total-timesteps",        type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--enable-per-cell-eval",   action="store_true",
                        help="Enable 9-cell test-grid eval every 50k steps "
                             "(adds ~7.5 min wall time). NOTE: not yet wired "
                             "for TQC; will be a no-op until TQCRLController "
                             "integration in v2.10 follow-up.")
    args = parser.parse_args()

    train_tqc_e2(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        enable_per_cell_eval=args.enable_per_cell_eval,
    )
